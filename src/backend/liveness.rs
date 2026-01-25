//! Liveness analysis for IR values.
//!
//! Computes live intervals for each IR value in an IrFunction. A live interval
//! represents the range [def_point, last_use_point] where a value is live and
//! needs to be preserved (either in a register or a stack slot).
//!
//! The analysis uses a linear numbering scheme: each instruction and terminator
//! gets a sequential program point. The result maps each Value to its live interval.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::ir::*;

/// A live interval for an IR value: [start, end] in program point numbering.
/// start = the point where the value is defined
/// end = the last point where the value is used
#[derive(Debug, Clone, Copy)]
pub struct LiveInterval {
    pub start: u32,
    pub end: u32,
    pub value_id: u32,
}

/// Result of liveness analysis: maps value IDs to their live intervals.
pub struct LivenessResult {
    pub intervals: Vec<LiveInterval>,
    /// Total number of program points (for debugging/sizing).
    pub num_points: u32,
}

/// Compute live intervals for all non-alloca values in a function.
///
/// We assign sequential program points to each instruction and terminator.
/// Then for each value, record where it is defined (instruction point) and
/// the latest point at which it is used. Values used across basic block
/// boundaries get extended to cover the full range.
///
/// Alloca values are excluded because they represent stack addresses, not
/// register-allocatable temporaries.
pub fn compute_live_intervals(func: &IrFunction) -> LivenessResult {
    // Phase 1: Assign program points and record defs/uses.
    // We number sequentially: block0_inst0=0, block0_inst1=1, ..., block0_term=N,
    // block1_inst0=N+1, etc.
    let mut point: u32 = 0;
    let mut def_points: FxHashMap<u32, u32> = FxHashMap::default();
    let mut last_use_points: FxHashMap<u32, u32> = FxHashMap::default();
    let mut alloca_set: FxHashSet<u32> = FxHashSet::default();

    // Collect alloca values (not register-allocatable).
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                alloca_set.insert(dest.0);
            }
        }
    }

    // Phase 2: Walk all instructions recording defs and uses.
    for block in &func.blocks {
        for inst in &block.instructions {
            // Record uses of operands at this point.
            record_instruction_uses(inst, point, &alloca_set, &mut last_use_points);

            // Record def at this point.
            if let Some(dest) = inst.dest() {
                if !alloca_set.contains(&dest.0) {
                    def_points.entry(dest.0).or_insert(point);
                }
            }

            point += 1;
        }

        // Record uses in terminator.
        record_terminator_uses(&block.terminator, point, &alloca_set, &mut last_use_points);
        point += 1;
    }

    let num_points = point;

    // Cross-block liveness: since we use sequential program point numbering
    // across all blocks, a value defined in block A and used in a later block B
    // will naturally have its interval span [def_point_in_A, use_point_in_B].
    // This correctly covers all intermediate points for forward-only control flow.
    // Functions with loops (back-edges) are excluded by the register allocator.

    // Phase 3: Build intervals.
    let mut intervals: Vec<LiveInterval> = Vec::new();
    for (&value_id, &start) in &def_points {
        let end = last_use_points.get(&value_id).copied().unwrap_or(start);
        // Only include values that are actually used (end >= start).
        intervals.push(LiveInterval {
            start,
            end: end.max(start),
            value_id,
        });
    }

    // Sort by start point (required by linear scan).
    intervals.sort_by_key(|iv| iv.start);

    LivenessResult {
        intervals,
        num_points,
    }
}

/// Record uses of operands in an instruction.
fn record_instruction_uses(
    inst: &Instruction,
    point: u32,
    alloca_set: &FxHashSet<u32>,
    last_use: &mut FxHashMap<u32, u32>,
) {
    for_each_operand_in_instruction(inst, |op| {
        if let Operand::Value(v) = op {
            if !alloca_set.contains(&v.0) {
                let entry = last_use.entry(v.0).or_insert(point);
                if point > *entry {
                    *entry = point;
                }
            }
        }
    });

    // Also record value-type uses that aren't in Operand wrappers.
    // Store, Load, GEP, Memcpy, VaArg etc. use Value directly for pointers.
    for_each_value_use_in_instruction(inst, |v| {
        if !alloca_set.contains(&v.0) {
            let entry = last_use.entry(v.0).or_insert(point);
            if point > *entry {
                *entry = point;
            }
        }
    });
}

/// Record uses in a terminator.
fn record_terminator_uses(
    term: &Terminator,
    point: u32,
    alloca_set: &FxHashSet<u32>,
    last_use: &mut FxHashMap<u32, u32>,
) {
    match term {
        Terminator::Return(Some(op)) => {
            if let Operand::Value(v) = op {
                if !alloca_set.contains(&v.0) {
                    let entry = last_use.entry(v.0).or_insert(point);
                    if point > *entry {
                        *entry = point;
                    }
                }
            }
        }
        Terminator::CondBranch { cond, .. } => {
            if let Operand::Value(v) = cond {
                if !alloca_set.contains(&v.0) {
                    let entry = last_use.entry(v.0).or_insert(point);
                    if point > *entry {
                        *entry = point;
                    }
                }
            }
        }
        Terminator::IndirectBranch { target, .. } => {
            if let Operand::Value(v) = target {
                if !alloca_set.contains(&v.0) {
                    let entry = last_use.entry(v.0).or_insert(point);
                    if point > *entry {
                        *entry = point;
                    }
                }
            }
        }
        _ => {}
    }
}

/// Iterate over all Operand references in an instruction.
fn for_each_operand_in_instruction(inst: &Instruction, mut f: impl FnMut(&Operand)) {
    match inst {
        Instruction::Alloca { .. } => {}
        Instruction::DynAlloca { size, .. } => f(size),
        Instruction::Store { val, .. } => f(val),
        Instruction::Load { .. } => {}
        Instruction::BinOp { lhs, rhs, .. } => { f(lhs); f(rhs); }
        Instruction::UnaryOp { src, .. } => f(src),
        Instruction::Cmp { lhs, rhs, .. } => { f(lhs); f(rhs); }
        Instruction::Call { args, .. } => { for a in args { f(a); } }
        Instruction::CallIndirect { func_ptr, args, .. } => { f(func_ptr); for a in args { f(a); } }
        Instruction::GetElementPtr { offset, .. } => f(offset),
        Instruction::Cast { src, .. } => f(src),
        Instruction::Copy { src, .. } => f(src),
        Instruction::GlobalAddr { .. } => {}
        Instruction::Memcpy { .. } => {}
        Instruction::VaArg { .. } => {}
        Instruction::VaStart { .. } => {}
        Instruction::VaEnd { .. } => {}
        Instruction::VaCopy { .. } => {}
        Instruction::AtomicRmw { ptr, val, .. } => { f(ptr); f(val); }
        Instruction::AtomicCmpxchg { ptr, expected, desired, .. } => { f(ptr); f(expected); f(desired); }
        Instruction::AtomicLoad { ptr, .. } => f(ptr),
        Instruction::AtomicStore { ptr, val, .. } => { f(ptr); f(val); }
        Instruction::Fence { .. } => {}
        Instruction::Phi { incoming, .. } => { for (op, _) in incoming { f(op); } }
        Instruction::LabelAddr { .. } => {}
        Instruction::GetReturnF64Second { .. } => {}
        Instruction::SetReturnF64Second { src } => f(src),
        Instruction::GetReturnF32Second { .. } => {}
        Instruction::SetReturnF32Second { src } => f(src),
        Instruction::InlineAsm { inputs, .. } => {
            for (_, op, _) in inputs { f(op); }
        }
        Instruction::Intrinsic { args, .. } => { for a in args { f(a); } }
    }
}

/// Iterate over Value references (non-Operand) used in an instruction.
/// These are pointer/base values that are used directly, not as Operand.
fn for_each_value_use_in_instruction(inst: &Instruction, mut f: impl FnMut(&Value)) {
    match inst {
        Instruction::Store { ptr, .. } => f(ptr),
        Instruction::Load { ptr, .. } => f(ptr),
        Instruction::GetElementPtr { base, .. } => f(base),
        Instruction::Memcpy { dest, src, .. } => { f(dest); f(src); }
        Instruction::VaArg { va_list_ptr, .. } => f(va_list_ptr),
        Instruction::VaStart { va_list_ptr } => f(va_list_ptr),
        Instruction::VaEnd { va_list_ptr } => f(va_list_ptr),
        Instruction::VaCopy { dest_ptr, src_ptr } => { f(dest_ptr); f(src_ptr); }
        Instruction::InlineAsm { outputs, .. } => {
            for (_, v, _) in outputs { f(v); }
        }
        Instruction::Intrinsic { dest_ptr, .. } => {
            if let Some(dp) = dest_ptr { f(dp); }
        }
        _ => {}
    }
}
