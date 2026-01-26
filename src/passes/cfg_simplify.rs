//! CFG Simplification pass.
//!
//! Simplifies the control flow graph by:
//! 1. Folding `CondBranch` with a known-constant condition to `Branch`
//! 2. Converting `CondBranch` where both targets are the same to `Branch`
//! 3. Threading jump chains: if block A branches to empty block B which just
//!    branches to C, redirect A to branch directly to C (only when safe)
//! 4. Removing dead (unreachable) blocks that have no predecessors
//! 5. Simplifying trivial phi nodes (single-entry or all-same-value) to Copy
//!
//! This pass runs to a fixpoint, since one simplification can enable others.
//! Phi nodes in successor blocks are updated when edges are redirected.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::ir::*;

/// Maximum depth for resolving transitive jump chains (A→B→C→...),
/// to prevent pathological cases.
const MAX_CHAIN_DEPTH: u32 = 32;

/// Run CFG simplification on the entire module.
/// Returns the number of simplifications made.
pub fn run(module: &mut IrModule) -> usize {
    module.for_each_function(simplify_cfg)
}

/// Simplify the CFG of a single function.
/// Iterates until no more simplifications are possible (fixpoint).
fn simplify_cfg(func: &mut IrFunction) -> usize {
    if func.blocks.is_empty() {
        return 0;
    }

    let mut total = 0;
    loop {
        let mut changed = 0;
        changed += fold_constant_cond_branches(func);
        changed += simplify_redundant_cond_branches(func);
        changed += thread_jump_chains(func);
        changed += remove_dead_blocks(func);
        changed += simplify_trivial_phis(func);
        if changed == 0 {
            break;
        }
        total += changed;
    }
    total
}

/// Convert `CondBranch { cond, true_label: X, false_label: X }` to `Branch(X)`.
/// The condition is dead and will be cleaned up by DCE.
fn simplify_redundant_cond_branches(func: &mut IrFunction) -> usize {
    let mut count = 0;
    for block in &mut func.blocks {
        if let Terminator::CondBranch { true_label, false_label, .. } = &block.terminator {
            if true_label == false_label {
                let target = *true_label;
                block.terminator = Terminator::Branch(target);
                count += 1;
            }
        }
    }
    count
}

/// Fold `CondBranch` with a known-constant condition into an unconditional `Branch`.
///
/// After constant folding + copy propagation, a CondBranch may have a constant
/// condition (e.g., `CondBranch { cond: Const(1), true_label, false_label }`).
/// This arises in switch(sizeof(T)) patterns where the dispatch comparisons
/// fold to constants. Converting these to unconditional branches enables dead
/// block removal to eliminate the unreachable switch cases.
///
/// When folding, we must clean up phi nodes in the not-taken target block:
/// the phi entries referencing the current block must be removed since the edge
/// no longer exists. Without this cleanup, stale phi entries can cause
/// miscompilation when the not-taken block is still reachable from other paths.
fn fold_constant_cond_branches(func: &mut IrFunction) -> usize {
    // First pass: collect the folding decisions.
    // Each entry: (block_index, taken_target, not_taken_target, block_label)
    let mut folds: Vec<(usize, BlockId, BlockId, BlockId)> = Vec::new();

    for (idx, block) in func.blocks.iter().enumerate() {
        if let Terminator::CondBranch { cond, true_label, false_label } = &block.terminator {
            let const_val = match cond {
                Operand::Const(c) => Some(is_const_nonzero(c)),
                _ => None,
            };
            if let Some(is_true) = const_val {
                let taken = if is_true { *true_label } else { *false_label };
                let not_taken = if is_true { *false_label } else { *true_label };
                folds.push((idx, taken, not_taken, block.label));
            }
        }
    }

    if folds.is_empty() {
        return 0;
    }

    let count = folds.len();

    // Apply the folds: change terminators to unconditional branches
    for &(idx, taken, _, _) in &folds {
        func.blocks[idx].terminator = Terminator::Branch(taken);
    }

    // Clean up phi nodes in not-taken target blocks.
    // Remove phi entries that reference the folding block, since that edge
    // no longer exists. Only remove when the not-taken target differs from
    // the taken target (if they're the same, the edge still exists).
    for &(_, taken, not_taken, block_label) in &folds {
        if taken == not_taken {
            // Both branches go to the same block - edge is preserved, no cleanup needed
            continue;
        }
        // Find the not-taken block and remove phi entries from block_label
        for block in &mut func.blocks {
            if block.label == not_taken {
                for inst in &mut block.instructions {
                    if let Instruction::Phi { incoming, .. } = inst {
                        incoming.retain(|(_, label)| *label != block_label);
                    }
                }
                break;
            }
        }
    }

    count
}

/// Check if a constant value is nonzero (truthy).
fn is_const_nonzero(c: &IrConst) -> bool {
    match c {
        IrConst::I8(v) => *v != 0,
        IrConst::I16(v) => *v != 0,
        IrConst::I32(v) => *v != 0,
        IrConst::I64(v) => *v != 0,
        IrConst::I128(v) => *v != 0,
        IrConst::F32(v) => *v != 0.0,
        IrConst::F64(v) => *v != 0.0,
        IrConst::LongDouble(v) => *v != 0.0,
        IrConst::Zero => false,
    }
}

/// Check if threading a CondBranch's two edges to the same final target would
/// create a phi conflict. This happens when the final target block has a phi
/// node that carries different values from the two paths.
///
/// Example: Block A CondBranch(true: B, false: C), B is empty forwarding to C.
/// C has Phi with (val_from_A, A) and (val_from_B, B) where the values differ.
/// Threading B out would merge both edges to C, and the subsequent redundant-
/// branch simplification converts CondBranch(true:C, false:C) to Branch(C).
/// Dead block removal then deletes B, removing (val_from_B, B) from the phi,
/// losing the true-path value.
///
/// For multi-hop chains (B -> C -> D), the phi in D references C (the last hop),
/// not B (the start). We must use the last hop for phi lookups.
///
/// Parameters:
/// - `block_label`: the predecessor block (A) with the CondBranch
/// - `true_label`: the true target (B, which may forward to the final target)
/// - `false_label`: the false target (C, which may forward or be the final target)
/// - `target`: the final target block both paths would reach
/// - `resolved`: the forwarding resolution map (block -> (final_target, phi_lookup_block))
fn would_create_phi_conflict(
    func: &IrFunction,
    block_label: BlockId,
    true_label: BlockId,
    false_label: BlockId,
    target: BlockId,
    resolved: &FxHashMap<BlockId, (BlockId, BlockId)>,
) -> bool {
    // Find the target block
    let target_block = match func.blocks.iter().find(|b| b.label == target) {
        Some(b) => b,
        None => return false,
    };

    // Determine the phi-lookup label for each path.
    // For a path through an intermediate chain, the phi references the immediate
    // predecessor of target (the phi_lookup_block from resolved), not the first
    // intermediate. For a direct path (not resolved), the phi references block_label.
    let true_phi_label = if let Some(&(_, phi_block)) = resolved.get(&true_label) {
        phi_block
    } else {
        block_label
    };
    let false_phi_label = if let Some(&(_, phi_block)) = resolved.get(&false_label) {
        phi_block
    } else {
        block_label
    };

    // If both paths use the same phi label, there's no way they carry different values
    if true_phi_label == false_phi_label {
        return false;
    }

    // Check each phi in the target block
    for inst in &target_block.instructions {
        if let Instruction::Phi { incoming, .. } = inst {
            let mut true_value = None;
            let mut false_value = None;

            for (val, label) in incoming {
                if *label == true_phi_label {
                    true_value = Some(val);
                }
                if *label == false_phi_label {
                    false_value = Some(val);
                }
            }

            // If both paths have entries and they differ, this is a conflict
            if let (Some(tv), Some(fv)) = (true_value, false_value) {
                if !operands_equal(tv, fv) {
                    return true;
                }
            }
        }
    }

    false
}

/// Compare two operands for structural equality.
fn operands_equal(a: &Operand, b: &Operand) -> bool {
    match (a, b) {
        (Operand::Value(v1), Operand::Value(v2)) => v1.0 == v2.0,
        (Operand::Const(c1), Operand::Const(c2)) => consts_equal_for_phi(c1, c2),
        _ => false,
    }
}

/// Compare two IR constants for phi simplification.
/// Integer constants of different widths but same numeric value are considered equal
/// (e.g., I32(0) == I64(0)). This is important because different IR paths may produce
/// the same semantic value at different IR type widths (e.g., a Cmp returns I32 while
/// the short-circuit && default uses I64).
fn consts_equal_for_phi(a: &IrConst, b: &IrConst) -> bool {
    // Fast path: same variant
    if a.to_hash_key() == b.to_hash_key() {
        return true;
    }
    // Cross-width integer comparison: extract as i64 and compare
    match (a.to_i64(), b.to_i64()) {
        (Some(va), Some(vb)) => va == vb,
        _ => false,
    }
}

/// Thread jump chains: if a block branches to an empty forwarding block
/// (no instructions, terminates with unconditional Branch), redirect to
/// skip the intermediate block.
///
/// We only thread through a block if:
/// - The intermediate block has NO instructions (including no phi nodes)
/// - The intermediate block terminates with an unconditional Branch
///
/// After threading, we update phi nodes in the target block to replace
/// references to the intermediate block with references to the redirected
/// predecessor.
///
/// Special care: when threading would cause a CondBranch's true and false
/// targets to become the same block (both edges merge), AND the target has
/// phi nodes that carry different values from the two paths, we must NOT
/// thread. Otherwise the merge block's phi loses the ability to distinguish
/// the two control flow paths, causing miscompilation.
fn thread_jump_chains(func: &mut IrFunction) -> usize {
    // Build a map of block_id -> forwarding target for empty blocks.
    // An "empty forwarding block" has no instructions (including no phis)
    // and terminates with Branch(target).
    let forwarding: FxHashMap<BlockId, BlockId> = func.blocks.iter()
        .filter(|block| {
            block.instructions.is_empty()
                && matches!(&block.terminator, Terminator::Branch(_))
        })
        .map(|block| {
            if let Terminator::Branch(target) = &block.terminator {
                (block.label, *target)
            } else {
                unreachable!()
            }
        })
        .collect();

    if forwarding.is_empty() {
        return 0;
    }

    // Resolve transitive chains with cycle detection.
    // If A -> B -> C where both B and C are forwarding blocks, resolve to A -> final.
    // We also track the immediate predecessor of the final target for phi updates:
    // in chain B -> C -> D, the final target is D and the immediate predecessor is C.
    // Phi nodes in D reference C (not B), so we need C to look up phi values.
    let resolved: FxHashMap<BlockId, (BlockId, BlockId)> = {
        let mut resolved = FxHashMap::default();
        for &start in forwarding.keys() {
            let mut prev = start;
            let mut current = start;
            let mut depth = 0;
            while let Some(&next) = forwarding.get(&current) {
                if next == start || depth > MAX_CHAIN_DEPTH {
                    break; // cycle or too deep
                }
                prev = current;
                current = next;
                depth += 1;
            }
            if current != start {
                // resolved maps: start -> (final_target, immediate_predecessor_of_final)
                resolved.insert(start, (current, prev));
            }
        }
        resolved
    };

    if resolved.is_empty() {
        return 0;
    }

    // Collect the redirections we need to make.
    // Each edge change: (old_intermediate, new_target, phi_lookup_block)
    // phi_lookup_block is the immediate predecessor of new_target in the chain,
    // which is the block whose label appears in new_target's phi nodes.
    let mut redirections: Vec<(usize, Vec<(BlockId, BlockId, BlockId)>)> = Vec::new();

    for block_idx in 0..func.blocks.len() {
        let mut edge_changes = Vec::new();
        let block_label = func.blocks[block_idx].label;

        match &func.blocks[block_idx].terminator {
            Terminator::Branch(target) => {
                if let Some(&(resolved_target, phi_block)) = resolved.get(target) {
                    edge_changes.push((*target, resolved_target, phi_block));
                }
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                let true_resolved = resolved.get(true_label).copied();
                let false_resolved = resolved.get(false_label).copied();

                // Determine the final targets after potential threading
                let true_final = true_resolved.map(|(t, _)| t).unwrap_or(*true_label);
                let false_final = false_resolved.map(|(t, _)| t).unwrap_or(*false_label);

                if true_final == false_final && true_label != false_label {
                    // Threading would make both branches go to the same block.
                    // This is dangerous when the target has phi nodes: the phi
                    // entries from the two intermediates carry path-specific
                    // values, and merging both edges into one would lose this
                    // distinction. Check for phi conflict.
                    if would_create_phi_conflict(func, block_label, *true_label,
                                                  *false_label, true_final, &resolved) {
                        // Phi conflict: don't thread either edge
                    } else {
                        // No phi conflict - safe to thread both edges
                        if let Some((rt, rt_phi)) = true_resolved {
                            edge_changes.push((*true_label, rt, rt_phi));
                        }
                        if let Some((rf, rf_phi)) = false_resolved {
                            if !edge_changes.iter().any(|(old, new, _)| *old == *false_label && *new == rf) {
                                edge_changes.push((*false_label, rf, rf_phi));
                            }
                        }
                    }
                } else if true_final != false_final {
                    // Different final targets - safe to thread
                    if let Some((rt, rt_phi)) = true_resolved {
                        edge_changes.push((*true_label, rt, rt_phi));
                    }
                    if let Some((rf, rf_phi)) = false_resolved {
                        if !edge_changes.iter().any(|(old, new, _)| *old == *false_label && *new == rf) {
                            edge_changes.push((*false_label, rf, rf_phi));
                        }
                    }
                }
            }
            // TODO: IndirectBranch targets could also be threaded through
            // empty blocks, but computed goto is rare enough to skip for now.
            _ => {}
        }

        if !edge_changes.is_empty() {
            redirections.push((block_idx, edge_changes));
        }
    }

    if redirections.is_empty() {
        return 0;
    }

    // Apply the redirections.
    let mut count = 0;
    for (block_idx, edge_changes) in &redirections {
        let block_label = func.blocks[*block_idx].label;

        // Update the terminator
        match &mut func.blocks[*block_idx].terminator {
            Terminator::Branch(target) => {
                for (old, new, _) in edge_changes {
                    if target == old {
                        *target = *new;
                    }
                }
            }
            Terminator::CondBranch { true_label, false_label, .. } => {
                for (old, new, _) in edge_changes {
                    if true_label == old {
                        *true_label = *new;
                    }
                    if false_label == old {
                        *false_label = *new;
                    }
                }
            }
            _ => {}
        }
        count += 1;

        // Update phi nodes in the new target blocks.
        // For each edge change, we use phi_lookup_block (the immediate predecessor
        // of new_target in the forwarding chain) to find the correct phi value.
        // In a chain A -> B -> C -> D, when redirecting A from B to D:
        //   - old_intermediate = B (the original target)
        //   - new_target = D (the final resolved target)
        //   - phi_lookup_block = C (D's phi entries reference C, not B)
        for (_old_intermediate, new_target, phi_lookup_block) in edge_changes {
            // Find the new_target block and update its phi nodes
            for block in &mut func.blocks {
                if block.label == *new_target {
                    for inst in &mut block.instructions {
                        if let Instruction::Phi { incoming, .. } = inst {
                            // Look up the phi value using the immediate predecessor
                            // in the chain (phi_lookup_block), which is the block
                            // that phi entries in new_target actually reference.
                            let value_from_chain = incoming.iter()
                                .find(|(_, label)| *label == *phi_lookup_block)
                                .map(|(val, _)| *val);
                            if let Some(val) = value_from_chain {
                                // Only add if block_label doesn't already have an entry
                                let already_has_entry = incoming.iter()
                                    .any(|(_, label)| *label == block_label);
                                if !already_has_entry {
                                    incoming.push((val, block_label));
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    count
}

/// Remove blocks that have no predecessors (except the entry block, blocks[0]).
/// Returns the number of blocks removed.
fn remove_dead_blocks(func: &mut IrFunction) -> usize {
    if func.blocks.len() <= 1 {
        return 0;
    }

    // Compute the set of blocks reachable from the entry block
    let entry = func.blocks[0].label;
    let mut reachable = FxHashSet::default();
    reachable.insert(entry);

    // Build a map from block ID to index for quick lookup
    let block_map: FxHashMap<BlockId, usize> = func.blocks.iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

    // BFS from entry block
    let mut worklist = vec![entry];
    while let Some(block_id) = worklist.pop() {
        if let Some(&idx) = block_map.get(&block_id) {
            // Successor blocks from terminator
            let targets = get_terminator_targets(&func.blocks[idx].terminator);
            for target in targets {
                if reachable.insert(target) {
                    worklist.push(target);
                }
            }
            // LabelAddr and InlineAsm goto labels (computed goto targets)
            for inst in &func.blocks[idx].instructions {
                if let Instruction::LabelAddr { label, .. } = inst {
                    if reachable.insert(*label) {
                        worklist.push(*label);
                    }
                }
                if let Instruction::InlineAsm { goto_labels, .. } = inst {
                    for (_, label) in goto_labels {
                        if reachable.insert(*label) {
                            worklist.push(*label);
                        }
                    }
                }
            }
        }
    }

    // Collect dead blocks
    let dead_blocks: FxHashSet<BlockId> = func.blocks.iter()
        .map(|b| b.label)
        .filter(|label| !reachable.contains(label))
        .collect();

    if dead_blocks.is_empty() {
        return 0;
    }

    // Clean up phi nodes in reachable blocks that reference dead blocks
    for block in &mut func.blocks {
        if !reachable.contains(&block.label) {
            continue;
        }
        for inst in &mut block.instructions {
            if let Instruction::Phi { incoming, .. } = inst {
                incoming.retain(|(_, label)| !dead_blocks.contains(label));
            }
        }
    }

    let original_len = func.blocks.len();
    func.blocks.retain(|b| reachable.contains(&b.label));
    original_len - func.blocks.len()
}

/// Simplify trivial phi nodes: phi nodes with exactly one incoming edge,
/// or where all incoming values are identical, are replaced with Copy
/// instructions. This enables copy propagation to propagate the value
/// to all uses, and subsequent constant branch folding can then eliminate
/// dead branches.
///
/// This is critical for patterns like `if (1 || expr)` where the `||`
/// short-circuit generates a phi that merges two paths, but after constant
/// branch folding removes the dead path, the phi has only one incoming
/// edge remaining. Without this simplification, the phi result stays as
/// a non-constant Value, preventing the outer `if` from being folded.
fn simplify_trivial_phis(func: &mut IrFunction) -> usize {
    let mut count = 0;

    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if let Instruction::Phi { dest, incoming, .. } = inst {
                let replacement = if incoming.len() == 1 {
                    // Single incoming edge: replace with Copy
                    Some(incoming[0].0)
                } else if incoming.len() > 1 {
                    // Check if all incoming values are identical
                    let first = &incoming[0].0;
                    if incoming.iter().all(|(val, _)| operands_equal(val, first)) {
                        Some(*first)
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(src) = replacement {
                    *inst = Instruction::Copy { dest: *dest, src };
                    count += 1;
                }
            }
        }
    }

    count
}

/// Get the branch targets of a terminator.
fn get_terminator_targets(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Branch(target) => vec![*target],
        Terminator::CondBranch { true_label, false_label, .. } => {
            vec![*true_label, *false_label]
        }
        Terminator::IndirectBranch { possible_targets, .. } => possible_targets.clone(),
        Terminator::Return(_) | Terminator::Unreachable => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IrType;

    #[test]
    fn test_redundant_cond_branch() {
        let mut func = IrFunction::new("test".to_string(), IrType::Void, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(1)) },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(1),
            },
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Return(None),
        });

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(1))));
    }

    #[test]
    fn test_jump_chain_threading() {
        let mut func = IrFunction::new("test".to_string(), IrType::Void, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(1)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(2)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Return(None),
        });

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(2))));
    }

    #[test]
    fn test_dead_block_elimination() {
        let mut func = IrFunction::new("test".to_string(), IrType::Void, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![],
            terminator: Terminator::Return(None),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(42)) },
            ],
            terminator: Terminator::Return(None),
        });

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].label, BlockId(0));
    }

    #[test]
    fn test_combined_simplifications() {
        let mut func = IrFunction::new("test".to_string(), IrType::Void, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(1)) },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(1),
            },
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(2)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Return(None),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(None),
        });

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        assert!(func.blocks.len() <= 3);
        match &func.blocks[0].terminator {
            Terminator::Branch(target) => assert_eq!(*target, BlockId(2)),
            _ => panic!("Expected Branch terminator"),
        }
    }

    #[test]
    fn test_phi_update_on_thread() {
        // Block 0 -> Block 1 (empty) -> Block 2 (has phi referencing Block 1)
        let mut func = IrFunction::new("test".to_string(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(42)) },
            ],
            terminator: Terminator::Branch(BlockId(1)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(2)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(1),
                    ty: IrType::I32,
                    incoming: vec![(Operand::Value(Value(0)), BlockId(1))],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(1)))),
        });

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(2))));

        // The phi in Block 2 should have an entry from Block 0
        let last_block = func.blocks.last().unwrap();
        if let Instruction::Phi { incoming, .. } = &last_block.instructions[0] {
            assert!(incoming.iter().any(|(_, label)| *label == BlockId(0)));
        } else {
            panic!("Expected Phi instruction");
        }
    }

    #[test]
    fn test_no_thread_through_block_with_instructions() {
        // Block 1 has an instruction, so it should NOT be threaded
        let mut func = IrFunction::new("test".to_string(), IrType::I32, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(1)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(42)) },
            ],
            terminator: Terminator::Branch(BlockId(2)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Value(Value(0)))),
        });

        let count = simplify_cfg(&mut func);
        // No jump threading should occur (block 1 has instructions)
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(1))));
        // But the function still has all 3 blocks
        assert_eq!(func.blocks.len(), 3);
    }

    #[test]
    fn test_cond_branch_threading() {
        // Block 0 cond-branches to Block 1 (empty fwd) and Block 2 (empty fwd),
        // both forward to Block 3
        let mut func = IrFunction::new("test".to_string(), IrType::Void, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Copy { dest: Value(0), src: Operand::Const(IrConst::I32(1)) },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(None),
        });

        let count = simplify_cfg(&mut func);
        assert!(count > 0);
        // After threading, Block 0 should go directly to Block 3 for both targets
        // Then redundant cond branch converts to Branch(3)
        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(BlockId(3))));
    }

    #[test]
    fn test_no_thread_when_phi_conflict() {
        // This tests the bug where threading both branches of a conditional
        // through empty forwarding blocks to the same target would lose phi
        // node distinctions.
        //
        // Block 0: CondBranch(cond, true:.L1, false:.L3)
        // Block 1 (.L1): empty, Branch(.L3)
        // Block 3 (.L3): Phi(dest=%8, [(Value(2), .L0), (Const(1), .L1)])
        //
        // The phi distinguishes between the true path (constant 1, via .L1)
        // and the false path (value 2, direct from .L0).
        //
        // Bug: threading .L1 to .L3 would make .L0 go directly to .L3 for
        // both branches, then simplify to unconditional Branch(.L3), losing
        // the Const(1) phi incoming.
        let mut func = IrFunction::new("test".to_string(), IrType::I64, vec![], false);
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![
                Instruction::Cmp {
                    dest: Value(5),
                    op: IrCmpOp::Eq,
                    lhs: Operand::Value(Value(2)),
                    rhs: Operand::Const(IrConst::I64(0)),
                    ty: IrType::I64,
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(5)),
                true_label: BlockId(1),
                false_label: BlockId(3),
            },
        });
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![],
            terminator: Terminator::Branch(BlockId(3)),
        });
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(8),
                    ty: IrType::I64,
                    incoming: vec![
                        (Operand::Value(Value(2)), BlockId(0)),
                        (Operand::Const(IrConst::I64(1)), BlockId(1)),
                    ],
                },
            ],
            terminator: Terminator::Return(Some(Operand::Value(Value(8)))),
        });

        simplify_cfg(&mut func);

        // The CondBranch must NOT be simplified to an unconditional Branch.
        // The phi must still have two distinct incoming values.
        match &func.blocks[0].terminator {
            Terminator::CondBranch { true_label, false_label, .. } => {
                // The true branch should still go through .L1 (not threaded)
                // to preserve the phi distinction
                assert!(
                    *true_label == BlockId(1) || *false_label != *true_label,
                    "Should not thread both branches to same target when phi has different values"
                );
            }
            Terminator::Branch(_) => {
                panic!("CondBranch was incorrectly simplified to unconditional Branch, losing phi distinction");
            }
            _ => panic!("Unexpected terminator"),
        }

        // Verify the phi still has distinct entries
        let merge_block = func.blocks.iter().find(|b| b.label == BlockId(3)).unwrap();
        if let Instruction::Phi { incoming, .. } = &merge_block.instructions[0] {
            assert!(incoming.len() >= 2, "Phi must retain at least 2 incoming edges");
            // Find the constant 1 entry - it must still be present
            let has_const_1 = incoming.iter().any(|(val, _)| {
                matches!(val, Operand::Const(IrConst::I64(1)))
            });
            assert!(has_const_1, "Phi must still have the Const(1) incoming value");
        } else {
            panic!("Expected Phi instruction in merge block");
        }
    }

    #[test]
    fn test_trivial_phi_simplification() {
        // Simulates the pattern from __builtin_constant_p(v) && expr:
        //
        // Block 0: CondBranch { cond: Const(0), true: Block 1, false: Block 2 }
        // Block 1: (RHS of &&)
        //   %1 = Cmp ...
        //   Branch(Block 2)
        // Block 2: (merge)
        //   %2 = Phi [(Const(0), Block 0), (%1, Block 1)]
        //   CondBranch { cond: %2, true: Block 3, false: Block 4 }
        //
        // After fold_constant_cond_branches folds Block 0's CondBranch (cond=0)
        // to Branch(Block 2) and removes Block 1's phi edge, then
        // remove_dead_blocks eliminates Block 1, the phi becomes:
        //   %2 = Phi [(Const(0), Block 0)]
        //
        // simplify_trivial_phis converts this to:
        //   %2 = Copy Const(0)
        //
        // Note: cfg_simplify alone cannot fold the outer CondBranch in Block 2
        // because fold_constant_cond_branches only matches Operand::Const, not
        // Values defined by Copy. The full pipeline (cfg_simplify -> copy_prop ->
        // cfg_simplify) is needed to complete the elimination. This test verifies
        // only what cfg_simplify accomplishes: the phi-to-Copy conversion and
        // dead block removal.
        let mut func = IrFunction::new("test".to_string(), IrType::I32, vec![], false);

        // Block 0: constant false condition
        func.blocks.push(BasicBlock {
            label: BlockId(0),
            instructions: vec![],
            terminator: Terminator::CondBranch {
                cond: Operand::Const(IrConst::I64(0)),
                true_label: BlockId(1),
                false_label: BlockId(2),
            },
        });

        // Block 1: RHS of && (will become dead)
        func.blocks.push(BasicBlock {
            label: BlockId(1),
            instructions: vec![
                Instruction::Cmp {
                    dest: Value(1),
                    op: IrCmpOp::Ne,
                    lhs: Operand::Value(Value(0)),
                    rhs: Operand::Const(IrConst::I64(0)),
                    ty: IrType::I64,
                },
            ],
            terminator: Terminator::Branch(BlockId(2)),
        });

        // Block 2: merge with phi
        func.blocks.push(BasicBlock {
            label: BlockId(2),
            instructions: vec![
                Instruction::Phi {
                    dest: Value(2),
                    ty: IrType::I64,
                    incoming: vec![
                        (Operand::Const(IrConst::I64(0)), BlockId(0)),
                        (Operand::Value(Value(1)), BlockId(1)),
                    ],
                },
            ],
            terminator: Terminator::CondBranch {
                cond: Operand::Value(Value(2)),
                true_label: BlockId(3),
                false_label: BlockId(4),
            },
        });

        // Block 3: dead branch (e.g., __field_overflow call)
        func.blocks.push(BasicBlock {
            label: BlockId(3),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(1)))),
        });

        // Block 4: continuation
        func.blocks.push(BasicBlock {
            label: BlockId(4),
            instructions: vec![],
            terminator: Terminator::Return(Some(Operand::Const(IrConst::I32(0)))),
        });

        let _count = simplify_cfg(&mut func);

        // Verify Block 1 (the RHS of &&) is eliminated as unreachable
        let has_block_1 = func.blocks.iter().any(|b| b.label == BlockId(1));
        assert!(!has_block_1, "Dead block 1 (RHS of &&) should be eliminated");

        // Verify the phi in Block 2 was converted to a Copy
        let block_2 = func.blocks.iter().find(|b| b.label == BlockId(2)).unwrap();
        match &block_2.instructions[0] {
            Instruction::Copy { dest, src } => {
                assert_eq!(dest.0, 2);
                assert!(matches!(src, Operand::Const(IrConst::I64(0))),
                    "Copy source should be Const(0)");
            }
            other => panic!("Expected Copy instruction, got {:?}", other),
        }
    }
}
