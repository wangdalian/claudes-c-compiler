//! Linear scan register allocator.
//!
//! Assigns physical registers to IR values based on their live intervals.
//! Values with the longest live ranges and most uses get priority for register
//! assignment. Values that don't fit in available registers remain on the stack.
//!
//! This allocator is designed for the RISC-V backend initially, using callee-saved
//! registers (s1-s11) to avoid save/restore around every call. Other backends can
//! adopt it later.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::types::IrType;
use crate::ir::ir::*;
use super::liveness::{LiveInterval, compute_live_intervals};

/// A physical register assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysReg(pub u8);

/// Result of register allocation for a function.
pub struct RegAllocResult {
    /// Map from value ID -> assigned physical register.
    pub assignments: FxHashMap<u32, PhysReg>,
    /// Set of physical registers actually used (for prologue/epilogue save/restore).
    pub used_regs: Vec<PhysReg>,
}

/// Configuration for the register allocator.
pub struct RegAllocConfig {
    /// Available registers for allocation (e.g., s1-s11 for RISC-V).
    pub available_regs: Vec<PhysReg>,
}

/// Run the linear scan register allocator on a function.
///
/// Strategy: We assign callee-saved registers to values with the longest
/// live intervals. This is a simplified linear scan that doesn't split
/// intervals — values either get a register for their entire lifetime or
/// remain on the stack.
///
/// We avoid allocating registers to:
/// - Alloca values (they represent stack addresses)
/// - i128 values (they need register pairs, handled specially)
/// - Values defined by Call/CallIndirect (result is in a0, needs immediate spill)
/// - Values used only once right after definition (no benefit from register)
pub fn allocate_registers(
    func: &IrFunction,
    config: &RegAllocConfig,
) -> RegAllocResult {
    if config.available_regs.is_empty() {
        return RegAllocResult {
            assignments: FxHashMap::default(),
            used_regs: Vec::new(),
        };
    }

    // Disable register allocation for functions with inline assembly or atomics.
    // Inline asm can clobber callee-saved registers without the allocator knowing.
    // Atomic operations use complex multi-instruction sequences that may conflict.
    let has_unsafe = func.blocks.iter().any(|block| {
        block.instructions.iter().any(|inst| matches!(inst,
            Instruction::InlineAsm { .. } |
            Instruction::AtomicRmw { .. } |
            Instruction::AtomicCmpxchg { .. } |
            Instruction::AtomicLoad { .. } |
            Instruction::AtomicStore { .. }
        ))
    });
    if has_unsafe {
        return RegAllocResult {
            assignments: FxHashMap::default(),
            used_regs: Vec::new(),
        };
    }

    // TODO: Implement proper loop-aware liveness analysis (backward dataflow with
    // live-in/live-out per block) to extend intervals across back-edges. Currently
    // uses simple linear numbering which doesn't correctly handle loops.
    //
    // Disable register allocation for functions with loops (back-edges in CFG).
    // A value defined at point P used at point Q < P (via back-edge) won't have
    // its interval extended properly, causing the register to be reused while still live.
    if has_back_edges(func) {
        return RegAllocResult {
            assignments: FxHashMap::default(),
            used_regs: Vec::new(),
        };
    }

    let liveness = compute_live_intervals(func);

    // Count uses per value for prioritization.
    let mut use_count: FxHashMap<u32, u32> = FxHashMap::default();

    // Use a whitelist approach: only allocate registers for values produced
    // by simple, well-understood instructions that go through operand_to_t0/store_t0_to.
    let mut eligible: FxHashSet<u32> = FxHashSet::default();

    for block in &func.blocks {
        for inst in &block.instructions {
            // Only BinOp, UnaryOp, Cmp, Cast, Load, and GEP results are eligible.
            // These all store their result via emit_store_result -> store_t0_to.
            // Exclude float and i128 types since they use different register paths.
            match inst {
                Instruction::BinOp { dest, ty, .. }
                | Instruction::UnaryOp { dest, ty, .. } => {
                    if !ty.is_float() && !ty.is_long_double()
                        && !matches!(ty, IrType::I128 | IrType::U128) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::Cmp { dest, .. } => {
                    eligible.insert(dest.0);
                }
                Instruction::Cast { dest, to_ty, from_ty, .. } => {
                    if !to_ty.is_float() && !to_ty.is_long_double()
                        && !from_ty.is_float() && !from_ty.is_long_double()
                        && !matches!(to_ty, IrType::I128 | IrType::U128)
                        && !matches!(from_ty, IrType::I128 | IrType::U128) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::Load { dest, ty, .. } => {
                    if !ty.is_float() && !ty.is_long_double()
                        && !matches!(ty, IrType::I128 | IrType::U128) {
                        eligible.insert(dest.0);
                    }
                }
                Instruction::GetElementPtr { dest, .. } => {
                    eligible.insert(dest.0);
                }
                _ => {}
            }

            // Count uses of operands.
            count_operand_uses(inst, &mut use_count);
        }
        // Count uses in terminator.
        count_terminator_uses(&block.terminator, &mut use_count);
    }

    // All non-eligible values are excluded.
    // (We'll filter by the eligible set below instead of using excluded.)

    // Filter intervals to only eligible values.
    let mut candidates: Vec<&LiveInterval> = liveness.intervals.iter()
        .filter(|iv| eligible.contains(&iv.value_id))
        .filter(|iv| iv.end > iv.start) // Must span at least 2 points
        .collect();

    // Prioritize: longer intervals with more uses get registers first.
    // Score = interval_length * use_count (heuristic).
    candidates.sort_by(|a, b| {
        let score_a = (a.end - a.start) as u64 * use_count.get(&a.value_id).copied().unwrap_or(1) as u64;
        let score_b = (b.end - b.start) as u64 * use_count.get(&b.value_id).copied().unwrap_or(1) as u64;
        score_b.cmp(&score_a) // Descending: highest score first
    });

    // Linear scan: greedily assign registers.
    // Each register tracks when it becomes free (the end point of the value using it).
    let num_regs = config.available_regs.len();
    let mut reg_free_until: Vec<u32> = vec![0; num_regs];
    let mut assignments: FxHashMap<u32, PhysReg> = FxHashMap::default();
    let mut used_regs_set: FxHashSet<u8> = FxHashSet::default();

    for interval in &candidates {
        // Find a register that is free at this interval's start.
        let mut best_reg: Option<usize> = None;
        let mut best_free_time: u32 = u32::MAX;

        for (i, &free_until) in reg_free_until.iter().enumerate() {
            if free_until <= interval.start {
                // This register is free — pick the one that's been free longest
                // (earliest free_until) to leave recently-freed regs available.
                if best_reg.is_none() || free_until < best_free_time {
                    best_reg = Some(i);
                    best_free_time = free_until;
                }
            }
        }

        if let Some(reg_idx) = best_reg {
            // Assign this register.
            reg_free_until[reg_idx] = interval.end + 1;
            assignments.insert(interval.value_id, config.available_regs[reg_idx]);
            used_regs_set.insert(config.available_regs[reg_idx].0);
        }
        // If no register available, the value stays on the stack (no spill insertion
        // needed since the stack slot already exists).
    }

    // Build sorted list of used registers.
    let mut used_regs: Vec<PhysReg> = used_regs_set.iter().map(|&r| PhysReg(r)).collect();
    used_regs.sort_by_key(|r| r.0);

    RegAllocResult {
        assignments,
        used_regs,
    }
}

/// Count operand uses in an instruction.
fn count_operand_uses(inst: &Instruction, use_count: &mut FxHashMap<u32, u32>) {
    let mut count_op = |op: &Operand| {
        if let Operand::Value(v) = op {
            *use_count.entry(v.0).or_insert(0) += 1;
        }
    };

    match inst {
        Instruction::Alloca { .. } => {}
        Instruction::DynAlloca { size, .. } => count_op(size),
        Instruction::Store { val, .. } => count_op(val),
        Instruction::Load { .. } => {}
        Instruction::BinOp { lhs, rhs, .. } => { count_op(lhs); count_op(rhs); }
        Instruction::UnaryOp { src, .. } => count_op(src),
        Instruction::Cmp { lhs, rhs, .. } => { count_op(lhs); count_op(rhs); }
        Instruction::Call { args, .. } => { for a in args { count_op(a); } }
        Instruction::CallIndirect { func_ptr, args, .. } => { count_op(func_ptr); for a in args { count_op(a); } }
        Instruction::GetElementPtr { offset, .. } => count_op(offset),
        Instruction::Cast { src, .. } => count_op(src),
        Instruction::Copy { src, .. } => count_op(src),
        Instruction::GlobalAddr { .. } => {}
        Instruction::Memcpy { .. } => {}
        Instruction::VaArg { .. } => {}
        Instruction::VaStart { .. } => {}
        Instruction::VaEnd { .. } => {}
        Instruction::VaCopy { .. } => {}
        Instruction::AtomicRmw { ptr, val, .. } => { count_op(ptr); count_op(val); }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => {
            count_op(ptr); count_op(expected); count_op(desired);
        }
        Instruction::AtomicLoad { ptr, .. } => count_op(ptr),
        Instruction::AtomicStore { ptr, val, .. } => { count_op(ptr); count_op(val); }
        Instruction::Fence { .. } => {}
        Instruction::Phi { incoming, .. } => { for (op, _) in incoming { count_op(op); } }
        Instruction::LabelAddr { .. } => {}
        Instruction::GetReturnF64Second { .. } => {}
        Instruction::SetReturnF64Second { src } => count_op(src),
        Instruction::GetReturnF32Second { .. } => {}
        Instruction::SetReturnF32Second { src } => count_op(src),
        Instruction::InlineAsm { inputs, .. } => {
            for (_, op, _) in inputs { count_op(op); }
        }
        Instruction::Intrinsic { args, .. } => { for a in args { count_op(a); } }
    }
}

/// Count operand uses in a terminator.
fn count_terminator_uses(term: &Terminator, use_count: &mut FxHashMap<u32, u32>) {
    match term {
        Terminator::Return(Some(op)) => {
            if let Operand::Value(v) = op {
                *use_count.entry(v.0).or_insert(0) += 1;
            }
        }
        Terminator::CondBranch { cond, .. } => {
            if let Operand::Value(v) = cond {
                *use_count.entry(v.0).or_insert(0) += 1;
            }
        }
        Terminator::IndirectBranch { target, .. } => {
            if let Operand::Value(v) = target {
                *use_count.entry(v.0).or_insert(0) += 1;
            }
        }
        _ => {}
    }
}

/// Detect whether a function's CFG has back-edges (i.e., loops).
/// A back-edge is a branch from a block to an earlier-or-same block in the
/// block ordering. This is a conservative check: it may flag irreducible
/// control flow as loops, which is fine (we just skip register allocation).
fn has_back_edges(func: &IrFunction) -> bool {
    // Build a map from BlockId -> position in block list.
    let block_index: FxHashMap<u32, usize> = func.blocks.iter().enumerate()
        .map(|(i, b)| (b.label.0, i))
        .collect();

    for (idx, block) in func.blocks.iter().enumerate() {
        let targets = match &block.terminator {
            Terminator::Branch(target) => vec![target.0],
            Terminator::CondBranch { true_label, false_label, .. } => {
                vec![true_label.0, false_label.0]
            }
            Terminator::IndirectBranch { possible_targets, .. } => {
                possible_targets.iter().map(|t| t.0).collect()
            }
            _ => vec![],
        };

        for target_id in targets {
            if let Some(&target_idx) = block_index.get(&target_id) {
                if target_idx <= idx {
                    return true; // Back-edge found: branch to same or earlier block
                }
            }
        }
    }

    false
}
