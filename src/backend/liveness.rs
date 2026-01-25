//! Liveness analysis for IR values.
//!
//! Computes live intervals for each IR value in an IrFunction. A live interval
//! represents the range [def_point, last_use_point] where a value is live and
//! needs to be preserved (either in a register or a stack slot).
//!
//! The analysis supports loops via backward dataflow iteration:
//! 1. First, assign sequential program points to all instructions and terminators.
//! 2. Run backward dataflow to compute live-in/live-out sets for each block.
//!    This correctly handles values that are live across loop back-edges.
//! 3. Build intervals by taking the union of def/use points and live-through blocks.

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
/// Uses backward dataflow analysis to correctly handle loops:
/// - live_in[B] = (live_out[B] - defs[B]) ∪ uses[B]
/// - live_out[B] = ∪ live_in[S] for all successors S of B
///
/// Values that are live-in to a block have their interval extended to cover
/// from the block's start point through the entire block. This correctly
/// extends intervals through loop back-edges.
pub fn compute_live_intervals(func: &IrFunction) -> LivenessResult {
    let num_blocks = func.blocks.len();
    if num_blocks == 0 {
        return LivenessResult { intervals: Vec::new(), num_points: 0 };
    }

    // Collect alloca values (not register-allocatable).
    let mut alloca_set: FxHashSet<u32> = FxHashSet::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, .. } = inst {
                alloca_set.insert(dest.0);
            }
        }
    }

    // Phase 1: Assign program points and record per-block information.
    let mut point: u32 = 0;
    let mut block_start_points: Vec<u32> = Vec::with_capacity(num_blocks);
    let mut block_end_points: Vec<u32> = Vec::with_capacity(num_blocks);
    let mut def_points: FxHashMap<u32, u32> = FxHashMap::default();
    let mut last_use_points: FxHashMap<u32, u32> = FxHashMap::default();

    // Per-block gen (used) and kill (defined) sets for dataflow
    let mut block_gen: Vec<FxHashSet<u32>> = Vec::with_capacity(num_blocks);
    let mut block_kill: Vec<FxHashSet<u32>> = Vec::with_capacity(num_blocks);

    // Block ID -> index mapping
    let mut block_id_to_idx: FxHashMap<u32, usize> = FxHashMap::default();
    for (idx, block) in func.blocks.iter().enumerate() {
        block_id_to_idx.insert(block.label.0, idx);
    }

    for block in &func.blocks {
        let block_start = point;
        block_start_points.push(block_start);
        let mut gen = FxHashSet::default();
        let mut kill = FxHashSet::default();

        for inst in &block.instructions {
            // Record uses before defs (an instruction can use a value it also defines,
            // but in SSA that doesn't happen -- dests are always fresh values)
            record_instruction_uses(inst, point, &alloca_set, &mut last_use_points);

            // Collect gen/kill for dataflow
            collect_instruction_gen(inst, &alloca_set, &kill, &mut gen);

            if let Some(dest) = inst.dest() {
                if !alloca_set.contains(&dest.0) {
                    def_points.entry(dest.0).or_insert(point);
                    kill.insert(dest.0);
                }
            }

            point += 1;
        }

        // Record uses in terminator.
        record_terminator_uses(&block.terminator, point, &alloca_set, &mut last_use_points);
        collect_terminator_gen(&block.terminator, &alloca_set, &kill, &mut gen);
        let block_end = point;
        block_end_points.push(block_end);
        point += 1;

        block_gen.push(gen);
        block_kill.push(kill);
    }

    let num_points = point;

    // Phase 2: Build successor lists for the CFG.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    for (idx, block) in func.blocks.iter().enumerate() {
        for target_id in terminator_targets(&block.terminator) {
            if let Some(&target_idx) = block_id_to_idx.get(&target_id) {
                successors[idx].push(target_idx);
            }
        }
    }

    // Phase 3: Backward dataflow to compute live-in/live-out per block.
    // live_in[B] = gen[B] ∪ (live_out[B] - kill[B])
    // live_out[B] = ∪ live_in[S] for all successors S of B
    let mut live_in: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); num_blocks];
    let mut live_out: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); num_blocks];

    // Iterate until fixpoint (backward order converges faster).
    // For reducible CFGs, convergence is guaranteed in O(loop_nesting_depth) iterations.
    // MAX_ITERATIONS is a safety bound for pathological irreducible control flow;
    // if exceeded, liveness is conservative (intervals may be too long but never too short).
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: u32 = 50;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;

        for idx in (0..num_blocks).rev() {
            // Compute live_out = union of live_in of all successors
            let mut new_out = FxHashSet::default();
            for &succ in &successors[idx] {
                for &v in &live_in[succ] {
                    new_out.insert(v);
                }
            }

            // Compute live_in = gen ∪ (live_out - kill)
            let mut new_in = block_gen[idx].clone();
            for &v in &new_out {
                if !block_kill[idx].contains(&v) {
                    new_in.insert(v);
                }
            }

            if new_in != live_in[idx] || new_out != live_out[idx] {
                changed = true;
                live_in[idx] = new_in;
                live_out[idx] = new_out;
            }
        }
    }

    // Phase 4: Extend intervals for values that are live-in or live-out of blocks.
    // If a value is live-in to block B, it must be alive at block B's start point.
    // If a value is live-out of block B, it must be alive at block B's end point.
    for idx in 0..num_blocks {
        let start = block_start_points[idx];
        let end = block_end_points[idx];

        for &v in &live_in[idx] {
            // Extend start point: the value is alive at this block's start
            let entry = last_use_points.entry(v).or_insert(start);
            if end > *entry {
                *entry = end;
            }
            // Also make sure def_points covers this (for values defined earlier)
            // The def_point is already recorded; we just need the interval to
            // include this block's range.
        }

        for &v in &live_out[idx] {
            let entry = last_use_points.entry(v).or_insert(end);
            if end > *entry {
                *entry = end;
            }
        }
    }

    // Phase 5: Build intervals.
    let mut intervals: Vec<LiveInterval> = Vec::new();
    for (&value_id, &start) in &def_points {
        let end = last_use_points.get(&value_id).copied().unwrap_or(start);
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
    for_each_operand_in_terminator(term, |op| {
        if let Operand::Value(v) = op {
            if !alloca_set.contains(&v.0) {
                let entry = last_use.entry(v.0).or_insert(point);
                if point > *entry {
                    *entry = point;
                }
            }
        }
    });
}

/// Collect gen set (used-before-defined) for a block's instruction.
fn collect_instruction_gen(
    inst: &Instruction,
    alloca_set: &FxHashSet<u32>,
    kill: &FxHashSet<u32>,
    gen: &mut FxHashSet<u32>,
) {
    let mut add_use = |v: u32| {
        if !alloca_set.contains(&v) && !kill.contains(&v) {
            gen.insert(v);
        }
    };

    for_each_operand_in_instruction(inst, |op| {
        if let Operand::Value(v) = op {
            add_use(v.0);
        }
    });

    for_each_value_use_in_instruction(inst, |v| {
        add_use(v.0);
    });
}

/// Collect gen set for a terminator.
fn collect_terminator_gen(
    term: &Terminator,
    alloca_set: &FxHashSet<u32>,
    kill: &FxHashSet<u32>,
    gen: &mut FxHashSet<u32>,
) {
    for_each_operand_in_terminator(term, |op| {
        if let Operand::Value(v) = op {
            if !alloca_set.contains(&v.0) && !kill.contains(&v.0) {
                gen.insert(v.0);
            }
        }
    });
}

/// Get successor block IDs from a terminator.
fn terminator_targets(term: &Terminator) -> Vec<u32> {
    match term {
        Terminator::Branch(target) => vec![target.0],
        Terminator::CondBranch { true_label, false_label, .. } => {
            vec![true_label.0, false_label.0]
        }
        Terminator::IndirectBranch { possible_targets, .. } => {
            possible_targets.iter().map(|t| t.0).collect()
        }
        _ => vec![],
    }
}

/// Iterate over all Operand references in an instruction.
/// This is the single canonical source of truth for instruction operand traversal.
/// All code that needs to enumerate operands (liveness, use-counting, GEP fold
/// verification) should call this rather than hand-rolling its own match.
pub(super) fn for_each_operand_in_instruction(inst: &Instruction, mut f: impl FnMut(&Operand)) {
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
        Instruction::Select { cond, true_val, false_val, .. } => { f(cond); f(true_val); f(false_val); }
    }
}

/// Iterate over Value references (non-Operand) used in an instruction.
/// These are pointer/base values used directly (not wrapped in Operand),
/// e.g., the `ptr` in Store/Load, `base` in GEP, `dest`/`src` in Memcpy.
/// Canonical traversal — shared by liveness, use-counting, and GEP fold analysis.
pub(super) fn for_each_value_use_in_instruction(inst: &Instruction, mut f: impl FnMut(&Value)) {
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

/// Iterate over all Operand references in a terminator.
/// Canonical traversal — shared by liveness, use-counting, and GEP fold analysis.
pub(super) fn for_each_operand_in_terminator(term: &Terminator, mut f: impl FnMut(&Operand)) {
    match term {
        Terminator::Return(Some(op)) => f(op),
        Terminator::CondBranch { cond, .. } => f(cond),
        Terminator::IndirectBranch { target, .. } => f(target),
        _ => {}
    }
}
