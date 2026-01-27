//! Phi elimination: lower SSA phi nodes to copies in predecessor blocks.
//!
//! This pass runs after all SSA optimizations and before backend codegen.
//! It converts each Phi instruction into Copy instructions placed at the end
//! of each predecessor block (before the terminator).
//!
//! Smart temporary allocation:
//! When a block has multiple phis, we analyze the copy graph per-predecessor
//! to determine which copies actually need temporaries (only those involved
//! in cycles, e.g., swap patterns) and which can be direct copies. This
//! dramatically reduces the number of temporaries and copy instructions,
//! especially for large switch statements where most phis pass through
//! values unchanged.
//!
//! For non-conflicting phis (the common case), we emit direct copies:
//!   pred_block:
//!     %phi1_dest = copy src1
//!     %phi2_dest = copy src2
//!     <terminator>
//!
//! For conflicting phis (cycles), we use shared temporaries and a two-phase
//! copy sequence to avoid the lost-copy problem:
//!   pred_block:
//!     %tmp1 = copy src1  // save source before it's overwritten
//!     <terminator>
//!   target_block:
//!     %phi1_dest = copy %tmp1  // restore from temporary
//!     ... rest of block ...
//!
//! Critical edge splitting:
//! When a predecessor block has multiple successors (e.g., a CondBranch) and
//! the target block has phis, placing copies at the end of the predecessor
//! would execute them on ALL outgoing paths, not just the edge to the phi's
//! block. This corrupts values used on other paths. To fix this, we split
//! the critical edge by inserting a new trampoline block that contains only
//! the phi copies and branches unconditionally to the target.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::ir::*;

/// Eliminate all phi nodes in the module by lowering them to copies.
pub fn eliminate_phis(module: &mut IrModule) {
    // Compute the global max block ID across ALL functions to avoid label collisions
    // when creating trampoline blocks. Labels are module-wide (.L0, .L1, ...).
    let mut next_block_id = 0u32;
    for func in &module.functions {
        for block in &func.blocks {
            if block.label.0 >= next_block_id {
                next_block_id = block.label.0 + 1;
            }
        }
    }

    for func in &mut module.functions {
        if func.is_declaration || func.blocks.is_empty() {
            continue;
        }
        eliminate_phis_in_function(func, &mut next_block_id);
    }
}

/// Returns the number of distinct successor block IDs for a terminator.
fn successor_count(term: &Terminator) -> usize {
    match term {
        Terminator::Return(_) | Terminator::Unreachable => 0,
        Terminator::Branch(_) => 1,
        Terminator::CondBranch { true_label, false_label, .. } => {
            if true_label == false_label { 1 } else { 2 }
        }
        Terminator::IndirectBranch { possible_targets, .. } => possible_targets.len(),
        Terminator::Switch { cases, default, .. } => {
            let mut count = 1; // default
            let mut seen = vec![*default];
            for &(_, label) in cases {
                if !seen.contains(&label) {
                    seen.push(label);
                    count += 1;
                }
            }
            count
        }
    }
}

/// Replace one occurrence of `old_target` with `new_target` in a terminator.
fn retarget_terminator_once(term: &mut Terminator, old_target: BlockId, new_target: BlockId) {
    match term {
        Terminator::Branch(t) => {
            if *t == old_target {
                *t = new_target;
            }
        }
        Terminator::CondBranch { true_label, false_label, .. } => {
            // Only retarget one edge to avoid changing both sides of a diamond
            if *true_label == old_target {
                *true_label = new_target;
            } else if *false_label == old_target {
                *false_label = new_target;
            }
        }
        Terminator::IndirectBranch { possible_targets, .. } => {
            for t in possible_targets.iter_mut() {
                if *t == old_target {
                    *t = new_target;
                    break;
                }
            }
        }
        Terminator::Switch { cases, default, .. } => {
            if *default == old_target {
                *default = new_target;
            } else {
                for (_, t) in cases.iter_mut() {
                    if *t == old_target {
                        *t = new_target;
                        break;
                    }
                }
            }
        }
        _ => {}
    }
}

struct TrampolineBlock {
    label: BlockId,
    copies: Vec<Instruction>,
    branch_target: BlockId,
    pred_idx: usize,
    old_target: BlockId,
}

/// Get or create a trampoline block for a (pred, target) critical edge.
fn get_or_create_trampoline(
    trampoline_map: &mut FxHashMap<(usize, BlockId), usize>,
    trampolines: &mut Vec<TrampolineBlock>,
    pred_idx: usize,
    target_block_id: BlockId,
    next_block_id: &mut u32,
) -> usize {
    *trampoline_map
        .entry((pred_idx, target_block_id))
        .or_insert_with(|| {
            let idx = trampolines.len();
            let label = BlockId(*next_block_id);
            *next_block_id += 1;
            trampolines.push(TrampolineBlock {
                label,
                copies: Vec::new(),
                branch_target: target_block_id,
                pred_idx,
                old_target: target_block_id,
            });
            idx
        })
}

/// Determine which phi copies on a given predecessor edge need temporaries.
///
/// A copy `dest_i = src_i` needs a temporary if `src_i` is the destination
/// of another phi copy (i.e., another phi writes to the value we're reading).
/// This detects interference in the copy graph (e.g., swap patterns: x=y, y=x).
///
/// Returns a set of phi indices that need temporaries.
fn find_conflicting_phis(copies: &[(u32, Option<u32>)]) -> FxHashSet<usize> {
    // Build set of all phi destinations for this edge.
    let dest_set: FxHashSet<u32> = copies.iter().map(|(d, _)| *d).collect();

    // A phi copy needs a temporary if its source is overwritten by another phi
    // destination on this same edge. Specifically, phi i needs a temp if
    // src_i is in dest_set AND src_i != dest_i (self-copies don't conflict).
    let mut needs_temp: FxHashSet<usize> = FxHashSet::default();

    for (i, &(dest_i, src_i_opt)) in copies.iter().enumerate() {
        if let Some(src_i) = src_i_opt {
            if src_i != dest_i && dest_set.contains(&src_i) {
                needs_temp.insert(i);
            }
        }
    }

    // Conservative safety net: also mark phis whose destination is read by a
    // conflicting phi. This ensures the two-phase ordering (temp saves in
    // Pass 1, direct copies in Pass 2) remains correct even for chain
    // patterns like `a = b, b = c, c = a` where multiple values are involved
    // in cycles. This may over-approximate slightly (e.g., marking a phi
    // whose destination is read but already saved by another temp), but
    // correctness is paramount.
    if !needs_temp.is_empty() {
        let conflicting_sources: FxHashSet<u32> = needs_temp.iter()
            .filter_map(|&i| copies[i].1)
            .collect();
        for (j, &(dest_j, _)) in copies.iter().enumerate() {
            if conflicting_sources.contains(&dest_j) {
                needs_temp.insert(j);
            }
        }
    }

    needs_temp
}

fn eliminate_phis_in_function(func: &mut IrFunction, next_block_id: &mut u32) {
    // Use cached next_value_id if available, otherwise scan
    let mut next_value = if func.next_value_id > 0 {
        func.next_value_id
    } else {
        func.max_value_id() + 1
    };

    // Build label -> block index map
    let label_to_idx: FxHashMap<BlockId, usize> = func.blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label, i))
        .collect();

    // Determine which blocks have multiple successors (for critical edge detection)
    let multi_succ: Vec<bool> = func.blocks.iter()
        .map(|b| successor_count(&b.terminator) > 1)
        .collect();

    // Identify blocks that end with an IndirectBranch (computed goto).
    // These CANNOT use trampoline blocks because the runtime jump target is a
    // computed address that bypasses any CFG retargeting of possible_targets.
    // Phi copies for IndirectBranch predecessors must go before the terminator
    // in the predecessor block itself.
    let is_indirect_branch: Vec<bool> = func.blocks.iter()
        .map(|b| matches!(&b.terminator, Terminator::IndirectBranch { .. }))
        .collect();

    // Collect phi information from all blocks.
    struct PhiInfo {
        dest: Value,
        incoming: Vec<(Operand, BlockId)>,
    }

    let mut block_phis: Vec<Vec<PhiInfo>> = Vec::new();
    for block in &func.blocks {
        let mut phis = Vec::new();
        for inst in &block.instructions {
            if let Instruction::Phi { dest, incoming, .. } = inst {
                phis.push(PhiInfo {
                    dest: *dest,
                    incoming: incoming.clone(),
                });
            }
        }
        block_phis.push(phis);
    }

    // For each block with phis, determine where to place copies.
    // If the predecessor has multiple successors (critical edge), we split the
    // edge by creating a trampoline block for the copies.

    // pred_copies: pred_block_idx -> Vec<Copy instructions>
    // For predecessors with a single successor (safe to append copies directly).
    let mut pred_copies: FxHashMap<usize, Vec<Instruction>> = FxHashMap::default();

    // target_copies: copies to prepend at start of target blocks
    let mut target_copies: Vec<Vec<Instruction>> = vec![Vec::new(); func.blocks.len()];

    // Trampoline blocks for critical edge splitting.
    let mut trampolines: Vec<TrampolineBlock> = Vec::new();

    // Track which (pred_idx, target_block_id) pairs already have trampolines
    let mut trampoline_map: FxHashMap<(usize, BlockId), usize> = FxHashMap::default();

    for (block_idx, phis) in block_phis.iter().enumerate() {
        if phis.is_empty() {
            continue;
        }

        let target_block_id = func.blocks[block_idx].label;

        if phis.len() == 1 {
            // Single phi: copy directly (no temporaries needed)
            let phi = &phis[0];
            for (src, pred_label) in &phi.incoming {
                if let Some(&pred_idx) = label_to_idx.get(pred_label) {
                    // Skip self-copies: when src is the same value as dest
                    if let Operand::Value(v) = src {
                        if v.0 == phi.dest.0 {
                            continue;
                        }
                    }

                    let copy_inst = Instruction::Copy {
                        dest: phi.dest,
                        src: src.clone(),
                    };

                    if multi_succ[pred_idx] && !is_indirect_branch[pred_idx] {
                        let tramp_idx = get_or_create_trampoline(
                            &mut trampoline_map, &mut trampolines,
                            pred_idx, target_block_id, next_block_id,
                        );
                        trampolines[tramp_idx].copies.push(copy_inst);
                    } else {
                        pred_copies.entry(pred_idx).or_default().push(copy_inst);
                    }
                }
            }
        } else {
            // Multiple phis: use smart temporary allocation.
            //
            // Strategy: For each phi, determine if ANY predecessor edge has a
            // conflict for that phi. If so, the phi uses a shared temporary
            // (same temporary across all predecessors). If not, all predecessors
            // use direct copies for that phi.
            //
            // This is correct because:
            // - The temporary is shared across all predecessors, so the target
            //   block has exactly ONE copy (temp -> dest) that works regardless
            //   of which predecessor was taken.
            // - Non-conflicting phis use direct copies in each predecessor,
            //   writing directly to the phi destination.

            // Collect all unique predecessor labels. Use a HashSet for O(1) dedup,
            // which matters for large switch statements with hundreds of cases.
            let mut pred_label_set: FxHashSet<BlockId> = FxHashSet::default();
            let mut pred_labels: Vec<BlockId> = Vec::new();
            for phi in phis {
                for (_, pred_label) in &phi.incoming {
                    if pred_label_set.insert(*pred_label) {
                        pred_labels.push(*pred_label);
                    }
                }
            }

            // Precompute per-phi source lookup tables: for each phi, build a
            // map from predecessor BlockId to the source Operand. This avoids
            // O(incoming) linear search per phi per predecessor in the loops below.
            let phi_src_maps: Vec<FxHashMap<BlockId, &Operand>> = phis.iter()
                .map(|phi| {
                    phi.incoming.iter()
                        .map(|(src, pred_label)| (*pred_label, src))
                        .collect()
                })
                .collect();

            // Analyze each predecessor edge to find which phis have conflicts.
            // A phi is "globally conflicting" if it has a conflict on ANY edge.
            let mut globally_needs_temp: FxHashSet<usize> = FxHashSet::default();

            for pred_label in &pred_labels {
                if label_to_idx.get(pred_label).is_none() {
                    continue;
                }

                // Build the list of (dest, src_value_id) for this predecessor.
                let copies_info: Vec<(u32, Option<u32>)> = phis.iter().enumerate().map(|(i, phi)| {
                    let src_val_id = phi_src_maps[i].get(pred_label).and_then(|s| {
                        if let Operand::Value(v) = *s { Some(v.0) } else { None }
                    });
                    (phi.dest.0, src_val_id)
                }).collect();

                let edge_conflicts = find_conflicting_phis(&copies_info);
                for &i in &edge_conflicts {
                    globally_needs_temp.insert(i);
                }
            }

            // Allocate ONE shared temporary per globally-conflicting phi.
            let mut phi_temps: Vec<Option<Value>> = vec![None; phis.len()];
            for &i in &globally_needs_temp {
                phi_temps[i] = Some(Value(next_value));
                next_value += 1;
            }

            // Emit target block copies for conflicting phis (once, shared).
            for (i, phi) in phis.iter().enumerate() {
                if let Some(tmp) = phi_temps[i] {
                    target_copies[block_idx].push(Instruction::Copy {
                        dest: phi.dest,
                        src: Operand::Value(tmp),
                    });
                }
            }

            // Now emit copies for each predecessor edge.
            for pred_label in &pred_labels {
                let pred_idx = match label_to_idx.get(pred_label) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let mut edge_copies: Vec<Instruction> = Vec::new();

                // Pass 1: Emit temporary saves for conflicting phis (must come first
                // to capture source values before any direct copies overwrite them).
                // Note: we cannot skip self-copies here because the target block
                // unconditionally copies tmp -> dest for all conflicting phis.
                for (i, _phi) in phis.iter().enumerate() {
                    if let Some(tmp) = phi_temps[i] {
                        if let Some(src) = phi_src_maps[i].get(pred_label) {
                            edge_copies.push(Instruction::Copy {
                                dest: tmp,
                                src: (*src).clone(),
                            });
                        }
                    }
                }

                // Pass 2: Emit direct copies for non-conflicting phis.
                for (i, phi) in phis.iter().enumerate() {
                    if phi_temps[i].is_none() {
                        if let Some(src) = phi_src_maps[i].get(pred_label) {
                            // Skip self-copies
                            if let Operand::Value(v) = *src {
                                if v.0 == phi.dest.0 {
                                    continue;
                                }
                            }
                            edge_copies.push(Instruction::Copy {
                                dest: phi.dest,
                                src: (*src).clone(),
                            });
                        }
                    }
                }

                // Place copies in the appropriate location.
                if multi_succ[pred_idx] && !is_indirect_branch[pred_idx] {
                    let tramp_idx = get_or_create_trampoline(
                        &mut trampoline_map, &mut trampolines,
                        pred_idx, target_block_id, next_block_id,
                    );
                    trampolines[tramp_idx].copies.extend(edge_copies);
                } else {
                    pred_copies.entry(pred_idx).or_default().extend(edge_copies);
                }
            }
        }
    }

    // Apply transformations:
    // 1. Remove phi instructions from all blocks
    // 2. Prepend target copies
    // 3. Insert predecessor copies (for single-successor predecessors only)
    for (block_idx, block) in func.blocks.iter_mut().enumerate() {
        // Remove phi instructions (and their spans)
        if !block.source_spans.is_empty() {
            let mut span_idx = 0;
            block.source_spans.retain(|_| {
                let keep = !matches!(block.instructions.get(span_idx), Some(Instruction::Phi { .. }));
                span_idx += 1;
                keep
            });
        }
        block.instructions.retain(|inst| !matches!(inst, Instruction::Phi { .. }));

        // Prepend target copies (these go at the start, replacing the phis)
        if !target_copies[block_idx].is_empty() {
            let num_copies = target_copies[block_idx].len();
            let mut new_insts = target_copies[block_idx].clone();
            new_insts.append(&mut block.instructions);
            block.instructions = new_insts;
            if !block.source_spans.is_empty() {
                let mut new_spans = vec![crate::common::source::Span::dummy(); num_copies];
                new_spans.append(&mut block.source_spans);
                block.source_spans = new_spans;
            }
        }

        // Insert predecessor copies before terminator (only for single-successor blocks)
        if let Some(copies) = pred_copies.remove(&block_idx) {
            let num_copies = copies.len();
            block.instructions.extend(copies);
            if !block.source_spans.is_empty() {
                block.source_spans.extend(std::iter::repeat(crate::common::source::Span::dummy()).take(num_copies));
            }
        }
    }

    // 4. Retarget predecessors that need trampolines
    for trampoline in &trampolines {
        retarget_terminator_once(
            &mut func.blocks[trampoline.pred_idx].terminator,
            trampoline.old_target,
            trampoline.label,
        );
    }

    // 5. Append trampoline blocks to the function
    for trampoline in trampolines {
        let num_copies = trampoline.copies.len();
        func.blocks.push(BasicBlock {
            label: trampoline.label,
            instructions: trampoline.copies,
            source_spans: vec![crate::common::source::Span::dummy(); num_copies],
            terminator: Terminator::Branch(trampoline.branch_target),
        });
    }

    // Update cached next_value_id for downstream passes
    func.next_value_id = next_value;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_conflicts_independent_phis() {
        // a = 1, b = 2 — no overlap between dests and sources
        let copies = vec![(10, Some(1)), (20, Some(2))];
        let result = find_conflicting_phis(&copies);
        assert!(result.is_empty(), "Independent phis should have no conflicts");
    }

    #[test]
    fn test_swap_pattern() {
        // a = b, b = a — classic swap cycle
        let copies = vec![(10, Some(20)), (20, Some(10))];
        let result = find_conflicting_phis(&copies);
        assert!(result.contains(&0), "First phi in swap should be conflicting");
        assert!(result.contains(&1), "Second phi in swap should be conflicting");
    }

    #[test]
    fn test_three_way_cycle() {
        // a = b, b = c, c = a — three-way rotation
        let copies = vec![(10, Some(20)), (20, Some(30)), (30, Some(10))];
        let result = find_conflicting_phis(&copies);
        assert_eq!(result.len(), 3, "All three phis in a 3-way cycle should be conflicting");
    }

    #[test]
    fn test_mixed_conflicting_and_non_conflicting() {
        // a = b, b = a (conflict), c = 99 (independent)
        let copies = vec![(10, Some(20)), (20, Some(10)), (30, Some(99))];
        let result = find_conflicting_phis(&copies);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(!result.contains(&2), "Independent phi should not be marked conflicting");
    }

    #[test]
    fn test_self_copy_not_conflicting() {
        // a = a — self-copy, not a conflict
        let copies = vec![(10, Some(10)), (20, Some(30))];
        let result = find_conflicting_phis(&copies);
        assert!(result.is_empty(), "Self-copy should not be conflicting");
    }

    #[test]
    fn test_constant_source_not_conflicting() {
        // a = <const>, b = <const> — None sources (constants)
        let copies = vec![(10, None), (20, None)];
        let result = find_conflicting_phis(&copies);
        assert!(result.is_empty(), "Constant sources should have no conflicts");
    }

    #[test]
    fn test_chain_pattern_conservative() {
        // a = b, b = c — b is dest of phi 1 and source of phi 0
        // phi 0 reads b (which is the dest of phi 1), so phi 0 is directly conflicting.
        // The safety net then marks phi 1 because its dest (b=20) is a source
        // of a conflicting phi.
        let copies = vec![(10, Some(20)), (20, Some(30))];
        let result = find_conflicting_phis(&copies);
        assert!(result.contains(&0), "Chain phi reading overwritten dest should be conflicting");
        assert!(result.contains(&1), "Chain phi whose dest is read by conflicting phi should also be marked");
    }
}
