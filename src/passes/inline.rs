//! Function inlining pass.
//!
//! Inlines small static/static-inline functions at their call sites.
//! This is critical for eliminating dead branches guarded by constant-returning
//! inline functions (e.g., kernel's `IS_ENABLED()` patterns), which would
//! otherwise reference undefined external symbols and cause link errors.
//!
//! After inlining, subsequent passes (constant fold, DCE, CFG simplify) clean up
//! the inlined code and eliminate dead branches.

use crate::ir::ir::{
    BasicBlock, BlockId, IrFunction, IrModule, Instruction, Operand, Terminator, Value,
};
use crate::common::types::{IrType, AddressSpace};
use std::collections::HashMap;

/// Maximum number of IR instructions (across all blocks) in a callee for it
/// to be eligible for inlining. Conservative limit to avoid miscompiling
/// complex functions while still handling simple constant-returning helpers
/// like IS_ENABLED() wrappers and small accessor functions.
const MAX_INLINE_INSTRUCTIONS: usize = 20;

/// Maximum number of basic blocks in a callee for inlining eligibility.
/// Keeps inlining conservative to avoid code bloat while still handling
/// the kernel's IS_ENABLED() pattern, accessor functions, and small helpers
/// with simple control flow (e.g., single if-else).
const MAX_INLINE_BLOCKS: usize = 1;

/// Maximum total inlining budget per caller function (total inlined instructions).
/// Prevents exponential blowup from recursive inlining chains.
const MAX_INLINE_BUDGET_PER_CALLER: usize = 800;

/// Run the inlining pass on the module.
/// Returns the number of call sites inlined.
pub fn run(module: &mut IrModule) -> usize {
    let mut total_inlined = 0;
    let debug_inline = std::env::var("CCC_INLINE_DEBUG").is_ok();
    let skip_list: Vec<String> = std::env::var("CCC_INLINE_SKIP")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    // Build a snapshot of eligible callees (we can't borrow module mutably while reading callees).
    // We clone the callee function bodies since we need them while mutating callers.
    let callee_map = build_callee_map(module);

    if callee_map.is_empty() {
        return 0;
    }

    if debug_inline {
        eprintln!("[INLINE] Callee map has {} eligible functions:", callee_map.len());
        for (name, data) in &callee_map {
            let ic: usize = data.blocks.iter().map(|b| b.instructions.len()).sum();
            eprintln!("[INLINE]   '{}': {} blocks, {} instructions, {} params",
                name, data.blocks.len(), ic, data.num_params);
        }
    }

    // Compute the module-global max block ID. Block labels (.L{id}) are global in the
    // assembly output, so inlined blocks must use IDs that don't collide with ANY
    // function's block IDs, not just the caller's.
    let mut global_max_block_id: u32 = 0;
    for func in &module.functions {
        for block in &func.blocks {
            if block.label.0 > global_max_block_id {
                global_max_block_id = block.label.0;
            }
        }
    }

    // Process each function as a potential caller
    for func_idx in 0..module.functions.len() {
        if module.functions[func_idx].is_declaration {
            continue;
        }

        let mut budget_remaining = MAX_INLINE_BUDGET_PER_CALLER;
        let mut changed = true;

        // Iterate to handle chains of inlined calls (A calls B calls C, all small inline).
        // Limit iterations to prevent infinite loops from recursive inline functions.
        let max_rounds = 200;
        for _round in 0..max_rounds {
            if !changed {
                break;
            }
            changed = false;

            // Find call sites to inline in the current function
            let call_sites = find_inline_call_sites(&module.functions[func_idx], &callee_map, &skip_list);
            if call_sites.is_empty() {
                break;
            }

            // Inline one call site per round. After inlining, block indices shift,
            // so we must re-scan to get correct indices for subsequent call sites.
            let site = &call_sites[0];
            let callee_data = &callee_map[&site.callee_name];
            let callee_inst_count: usize = callee_data.blocks.iter()
                .map(|b| b.instructions.len())
                .sum();

            if callee_inst_count > budget_remaining {
                break;
            }

            let success = inline_call_site(
                &mut module.functions[func_idx],
                site,
                callee_data,
                &mut global_max_block_id,
            );

            if success {
                if debug_inline {
                    eprintln!("[INLINE] Inlined '{}' into '{}'", site.callee_name, module.functions[func_idx].name);
                }
                if std::env::var("CCC_INLINE_VALIDATE").is_ok() {
                    validate_function_values(&module.functions[func_idx], &site.callee_name);
                }
                if std::env::var("CCC_INLINE_DUMP_IR").is_ok() {
                    dump_function_ir(&module.functions[func_idx],
                        &format!("after inlining '{}' into '{}'", site.callee_name, module.functions[func_idx].name));
                }
                budget_remaining = budget_remaining.saturating_sub(callee_inst_count);
                total_inlined += 1;
                changed = true;
                module.functions[func_idx].has_inlined_calls = true;
            } else {
                break;
            }
        }
    }

    total_inlined
}

/// Debug validation: check that every Value used as an operand is defined by some instruction.
fn validate_function_values(func: &IrFunction, last_inlined_callee: &str) {
    use std::collections::HashSet;

    // Collect all defined values
    let mut defined: HashSet<u32> = HashSet::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(v) = inst.dest() {
                defined.insert(v.0);
            }
        }
    }

    // Check all used values
    let mut errors = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            for v in instruction_used_values(inst) {
                if !defined.contains(&v) {
                    errors.push(format!(
                        "  block[{}] (label .L{}) inst[{}]: uses undefined Value({}), inst={:?}",
                        block_idx, block.label.0, inst_idx, v, short_inst_name(inst)
                    ));
                }
            }
        }
        for v in terminator_used_values(&block.terminator) {
            if !defined.contains(&v) {
                errors.push(format!(
                    "  block[{}] (label .L{}) terminator: uses undefined Value({})",
                    block_idx, block.label.0, v
                ));
            }
        }
    }

    if !errors.is_empty() {
        eprintln!("[INLINE_VALIDATE] ERRORS in '{}' after inlining '{}': {} undefined value uses",
            func.name, last_inlined_callee, errors.len());
        for e in errors.iter().take(20) {
            eprintln!("{}", e);
        }
        if errors.len() > 20 {
            eprintln!("  ... and {} more", errors.len() - 20);
        }
    }
}

fn short_inst_name(inst: &Instruction) -> &'static str {
    match inst {
        Instruction::Alloca { .. } => "Alloca",
        Instruction::Store { .. } => "Store",
        Instruction::Load { .. } => "Load",
        Instruction::BinOp { .. } => "BinOp",
        Instruction::UnaryOp { .. } => "UnaryOp",
        Instruction::Cmp { .. } => "Cmp",
        Instruction::Call { .. } => "Call",
        Instruction::CallIndirect { .. } => "CallIndirect",
        Instruction::GetElementPtr { .. } => "GEP",
        Instruction::Cast { .. } => "Cast",
        Instruction::Copy { .. } => "Copy",
        Instruction::GlobalAddr { .. } => "GlobalAddr",
        Instruction::Memcpy { .. } => "Memcpy",
        Instruction::Phi { .. } => "Phi",
        Instruction::Select { .. } => "Select",
        _ => "Other",
    }
}

/// Collect all Value IDs used (as operands, not defined) by an instruction.
fn instruction_used_values(inst: &Instruction) -> Vec<u32> {
    let mut used = Vec::new();
    match inst {
        Instruction::Store { val, ptr, .. } => {
            if let Operand::Value(v) = val { used.push(v.0); }
            used.push(ptr.0);
        }
        Instruction::Load { ptr, .. } => {
            used.push(ptr.0);
        }
        Instruction::BinOp { lhs, rhs, .. } => {
            if let Operand::Value(v) = lhs { used.push(v.0); }
            if let Operand::Value(v) = rhs { used.push(v.0); }
        }
        Instruction::UnaryOp { src, .. } => {
            if let Operand::Value(v) = src { used.push(v.0); }
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            if let Operand::Value(v) = lhs { used.push(v.0); }
            if let Operand::Value(v) = rhs { used.push(v.0); }
        }
        Instruction::Call { args, .. } => {
            for a in args {
                if let Operand::Value(v) = a { used.push(v.0); }
            }
        }
        Instruction::CallIndirect { func_ptr, args, .. } => {
            if let Operand::Value(v) = func_ptr { used.push(v.0); }
            for a in args {
                if let Operand::Value(v) = a { used.push(v.0); }
            }
        }
        Instruction::GetElementPtr { base, offset, .. } => {
            used.push(base.0);
            if let Operand::Value(v) = offset { used.push(v.0); }
        }
        Instruction::Cast { src, .. } => {
            if let Operand::Value(v) = src { used.push(v.0); }
        }
        Instruction::Copy { src, .. } => {
            if let Operand::Value(v) = src { used.push(v.0); }
        }
        Instruction::Memcpy { dest, src, .. } => {
            used.push(dest.0);
            used.push(src.0);
        }
        Instruction::Phi { incoming, .. } => {
            for (op, _) in incoming {
                if let Operand::Value(v) = op { used.push(v.0); }
            }
        }
        Instruction::Select { cond, true_val, false_val, .. } => {
            if let Operand::Value(v) = cond { used.push(v.0); }
            if let Operand::Value(v) = true_val { used.push(v.0); }
            if let Operand::Value(v) = false_val { used.push(v.0); }
        }
        Instruction::Intrinsic { dest_ptr, args, .. } => {
            if let Some(v) = dest_ptr { used.push(v.0); }
            for a in args {
                if let Operand::Value(v) = a { used.push(v.0); }
            }
        }
        _ => {} // Alloca, GlobalAddr, Fence, etc. don't use values as operands
    }
    used
}

/// Collect all Value IDs used by a terminator.
fn terminator_used_values(term: &Terminator) -> Vec<u32> {
    let mut used = Vec::new();
    match term {
        Terminator::Return(Some(op)) => {
            if let Operand::Value(v) = op { used.push(v.0); }
        }
        Terminator::CondBranch { cond, .. } => {
            if let Operand::Value(v) = cond { used.push(v.0); }
        }
        Terminator::IndirectBranch { target, .. } => {
            if let Operand::Value(v) = target { used.push(v.0); }
        }
        _ => {}
    }
    used
}

/// Information about a callee function eligible for inlining.
struct CalleeData {
    blocks: Vec<BasicBlock>,
    params: Vec<IrType>,  // types of parameters
    /// For each param, Some(size) if it's a struct-by-value parameter, None otherwise.
    param_struct_sizes: Vec<Option<usize>>,
    return_type: IrType,
    num_params: usize,
    next_value_id: u32,
    /// Maximum BlockId used in the callee
    max_block_id: u32,
}

/// A call site that is eligible for inlining.
struct InlineCallSite {
    /// Index of the block containing the call
    block_idx: usize,
    /// Index of the instruction within the block
    inst_idx: usize,
    /// Name of the callee function
    callee_name: String,
    /// The destination value of the call (None for void)
    dest: Option<Value>,
    /// Arguments passed to the call
    args: Vec<Operand>,
}

/// Build a map of function name -> callee data for functions eligible for inlining.
fn build_callee_map(module: &IrModule) -> HashMap<String, CalleeData> {
    let mut map = HashMap::new();

    let debug_callee = std::env::var("CCC_INLINE_DEBUG").is_ok();
    for func in &module.functions {
        if func.is_declaration {
            continue;
        }
        // Only inline static inline functions (internal linkage + inline hint).
        // Non-static functions have external linkage and can't be removed after inlining.
        // Non-inline static functions are less likely to benefit from inlining and
        // may cause correctness issues if they have complex semantics.
        if !func.is_static || !func.is_inline {
            if debug_callee && func.name.contains("write16") {
                eprintln!("[INLINE_DEBUG] {} skipped: is_static={}, is_inline={}, is_declaration={}",
                    func.name, func.is_static, func.is_inline, func.is_declaration);
            }
            continue;
        }
        if debug_callee && func.name.contains("write16") {
            let ic: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
            eprintln!("[INLINE_DEBUG] {} candidate: blocks={}, inst_count={}, is_variadic={}, params={}",
                func.name, func.blocks.len(), ic, func.is_variadic, func.params.len());
        }
        // Don't inline variadic functions (complex ABI)
        if func.is_variadic {
            continue;
        }

        // Check size limits
        let inst_count: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
        if inst_count > MAX_INLINE_INSTRUCTIONS {
            continue;
        }
        if func.blocks.len() > MAX_INLINE_BLOCKS {
            continue;
        }

        // Skip functions containing constructs that are hard to inline correctly:
        // - Inline assembly (may have constraints tied to function-level semantics)
        // - VaStart/VaEnd/VaArg/VaCopy (variadic machinery)
        // - DynAlloca (dynamic stack allocation)
        // - Indirect branches (computed goto)
        // - StackSave/StackRestore (VLA-related)
        let mut has_problematic = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::InlineAsm { .. }
                    | Instruction::VaStart { .. }
                    | Instruction::VaEnd { .. }
                    | Instruction::VaArg { .. }
                    | Instruction::VaCopy { .. }
                    | Instruction::DynAlloca { .. }
                    | Instruction::StackSave { .. }
                    | Instruction::StackRestore { .. } => {
                        has_problematic = true;
                        break;
                    }
                    _ => {}
                }
            }
            if has_problematic {
                break;
            }
            if matches!(block.terminator, Terminator::IndirectBranch { .. }) {
                has_problematic = true;
                break;
            }
        }
        if has_problematic {
            continue;
        }

        // Clone the function's blocks for use during inlining
        let max_block_id = func.blocks.iter()
            .map(|b| b.label.0)
            .max()
            .unwrap_or(0);

        let param_types: Vec<IrType> = func.params.iter().map(|p| p.ty).collect();
        let param_struct_sizes: Vec<Option<usize>> = func.params.iter().map(|p| p.struct_size).collect();

        map.insert(func.name.clone(), CalleeData {
            blocks: func.blocks.clone(),
            params: param_types,
            param_struct_sizes,
            return_type: func.return_type,
            num_params: func.params.len(),
            next_value_id: func.next_value_id,
            max_block_id,
        });
    }

    map
}

/// Find call sites in a function that are eligible for inlining.
fn find_inline_call_sites(
    func: &IrFunction,
    callee_map: &HashMap<String, CalleeData>,
    skip_list: &[String],
) -> Vec<InlineCallSite> {
    let mut sites = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Call { dest, func: callee_name, args, .. } = inst {
                if callee_map.contains_key(callee_name) {
                    // Don't inline recursive calls
                    if callee_name != &func.name {
                        // Skip functions listed in CCC_INLINE_SKIP
                        if skip_list.iter().any(|s| s == callee_name) {
                            continue;
                        }
                        sites.push(InlineCallSite {
                            block_idx,
                            inst_idx,
                            callee_name: callee_name.clone(),
                            dest: *dest,
                            args: args.clone(),
                        });
                    }
                }
            }
        }
    }

    sites
}

/// Inline a single call site. Returns true if successful.
/// `global_max_block_id` is the module-global max block ID, updated on success.
fn inline_call_site(
    caller: &mut IrFunction,
    site: &InlineCallSite,
    callee: &CalleeData,
    global_max_block_id: &mut u32,
) -> bool {
    if callee.blocks.is_empty() {
        return false;
    }

    // Compute ID offsets for remapping callee values and blocks into caller's namespace
    let caller_next_value = if caller.next_value_id > 0 {
        caller.next_value_id
    } else {
        caller.max_value_id() + 1
    };

    // Use the global max block ID to avoid collisions with ANY function's blocks
    let value_offset = caller_next_value;
    let block_offset = *global_max_block_id + 1;

    let debug_inline_detail = std::env::var("CCC_INLINE_DEBUG_DETAIL").is_ok();
    if debug_inline_detail {
        eprintln!("[INLINE_DETAIL] Inlining '{}' into '{}': value_offset={}, block_offset={}, callee.next_value_id={}, caller.next_value_id={}",
            site.callee_name, caller.name, value_offset, block_offset, callee.next_value_id, caller.next_value_id);
        eprintln!("[INLINE_DETAIL]   site.block_idx={}, site.inst_idx={}", site.block_idx, site.inst_idx);
        for (i, arg) in site.args.iter().enumerate() {
            eprintln!("[INLINE_DETAIL]   arg[{}] = {:?}", i, arg);
        }
    }

    // Clone and remap the callee's blocks
    let mut inlined_blocks: Vec<BasicBlock> = Vec::with_capacity(callee.blocks.len());

    for callee_block in &callee.blocks {
        let mut new_block = BasicBlock {
            label: BlockId(callee_block.label.0 + block_offset),
            instructions: Vec::with_capacity(callee_block.instructions.len()),
            terminator: remap_terminator(&callee_block.terminator, value_offset, block_offset),
        };

        for inst in &callee_block.instructions {
            new_block.instructions.push(remap_instruction(inst, value_offset, block_offset));
        }

        inlined_blocks.push(new_block);
    }

    // Create a merge block that the callee's return statements will branch to
    let merge_block_id = BlockId(block_offset + callee.max_block_id + 1);

    // Collect return values from all return blocks to build a Phi node.
    // Each Return(Some(val)) becomes a branch to the merge block, and
    // the return value feeds into a Phi in the merge block.
    let mut phi_incoming: Vec<(Operand, BlockId)> = Vec::new();

    // Replace Return terminators in inlined blocks
    for block in &mut inlined_blocks {
        match &block.terminator {
            Terminator::Return(ret_val) => {
                if let (Some(_call_dest), Some(ret_operand)) = (site.dest, ret_val) {
                    phi_incoming.push((*ret_operand, block.label));
                }
                block.terminator = Terminator::Branch(merge_block_id);
            }
            _ => {}
        }
    }

    // Now we need to wire up the arguments. The callee's first N allocas are parameter allocas.
    // We need to store the caller's arguments into those allocas.
    // The param allocas are the first N Alloca instructions in the callee's entry block.
    let entry_block = &mut inlined_blocks[0];
    let mut param_alloca_info: Vec<(Value, IrType, usize)> = Vec::new(); // (dest, ty, size)
    for inst in &entry_block.instructions {
        if let Instruction::Alloca { dest, ty, size, .. } = inst {
            param_alloca_info.push((*dest, *ty, *size));
            if param_alloca_info.len() >= callee.num_params {
                break;
            }
        }
    }

    // Insert stores/memcpys of arguments into param allocas at the beginning of the
    // entry block (after the allocas themselves)
    let mut insert_pos = 0;
    // Find position after all allocas in the entry block
    for (i, inst) in entry_block.instructions.iter().enumerate() {
        if matches!(inst, Instruction::Alloca { .. }) {
            insert_pos = i + 1;
        } else {
            break;
        }
    }

    // Insert stores in reverse order so indices stay valid
    let num_args_to_store = std::cmp::min(site.args.len(), param_alloca_info.len());
    for i in (0..num_args_to_store).rev() {
        let param_struct_size = callee.param_struct_sizes.get(i).copied().flatten();
        if let Some(struct_size) = param_struct_size {
            // Struct-by-value parameter: the caller passes a pointer to the struct data.
            // We must copy the struct data from that pointer into the callee's param alloca.
            if let Operand::Value(src_ptr) = site.args[i] {
                entry_block.instructions.insert(insert_pos, Instruction::Memcpy {
                    dest: param_alloca_info[i].0,
                    src: src_ptr,
                    size: struct_size,
                });
            } else {
                // Struct arg should always be a Value (pointer), not a Const.
                // If somehow it's a Const, bail out of inlining.
                return false;
            }
        } else {
            // Scalar parameter: store the value directly into the param alloca.
            let store_ty = param_alloca_info[i].1;
            entry_block.instructions.insert(insert_pos, Instruction::Store {
                val: site.args[i],
                ptr: param_alloca_info[i].0,
                ty: store_ty,
                seg_override: AddressSpace::Default,
            });
        }
    }

    // Now split the caller's block at the call site:
    // Block before call -> instructions before the call + branch to callee entry
    // Block after call (merge block) -> instructions after the call + original terminator

    let call_block_idx = site.block_idx;
    let call_inst_idx = site.inst_idx;

    // Save instructions after the call and the terminator
    let after_call_instructions: Vec<Instruction> = caller.blocks[call_block_idx]
        .instructions
        .split_off(call_inst_idx + 1);
    let original_terminator = std::mem::replace(
        &mut caller.blocks[call_block_idx].terminator,
        Terminator::Branch(inlined_blocks[0].label),
    );

    // Remove the call instruction itself
    caller.blocks[call_block_idx].instructions.pop();

    // Create the merge block with the remaining instructions and original terminator.
    // If the callee had a non-void return, insert a Phi (or Copy for single-predecessor)
    // at the start of the merge block to define the call's result value.
    let mut merge_instructions = Vec::new();
    if let Some(call_dest) = site.dest {
        if phi_incoming.len() == 1 {
            // Single return path: just copy the value directly (no phi needed)
            merge_instructions.push(Instruction::Copy {
                dest: call_dest,
                src: phi_incoming[0].0,
            });
        } else if phi_incoming.len() > 1 {
            // Multiple return paths: need a Phi node
            merge_instructions.push(Instruction::Phi {
                dest: call_dest,
                ty: callee.return_type,
                incoming: phi_incoming,
            });
        }
        // If phi_incoming is empty, the callee never returns a value (e.g., all paths
        // are noreturn/unreachable). The call_dest will be undefined, which is fine
        // since it won't be used.
    }
    merge_instructions.extend(after_call_instructions);

    let merge_block = BasicBlock {
        label: merge_block_id,
        instructions: merge_instructions,
        terminator: original_terminator,
    };

    // Insert the inlined blocks and merge block after the call block
    let insert_position = call_block_idx + 1;
    // Insert merge block first, then inlined blocks before it
    caller.blocks.insert(insert_position, merge_block);
    for (i, block) in inlined_blocks.into_iter().enumerate() {
        caller.blocks.insert(insert_position + i, block);
    }

    // Update Phi nodes: the original caller block was split at the call site.
    // The merge block inherited the original block's terminator (and thus its
    // successors). Any Phi node in a successor block that references the original
    // split block as an incoming predecessor must now reference the merge block
    // instead, since control flow from the split block now goes through the
    // inlined code and arrives at the successor via the merge block.
    let split_block_label = caller.blocks[call_block_idx].label;
    for block in &mut caller.blocks {
        for inst in &mut block.instructions {
            if let Instruction::Phi { incoming, .. } = inst {
                for (_operand, block_id) in incoming.iter_mut() {
                    if *block_id == split_block_label {
                        *block_id = merge_block_id;
                    }
                }
            }
        }
    }

    // Update caller's next_value_id to account for the new values
    let new_next_value_id = value_offset + callee.next_value_id;
    if caller.next_value_id > 0 || new_next_value_id > caller.next_value_id {
        caller.next_value_id = std::cmp::max(
            new_next_value_id,
            caller.next_value_id,
        );
    }
    if debug_inline_detail {
        eprintln!("[INLINE_DETAIL]   after inline: caller.next_value_id={}", caller.next_value_id);
    }

    // Update the global max block ID so subsequent inlines use fresh IDs.
    // The merge block has the highest ID we assigned.
    *global_max_block_id = merge_block_id.0;

    true
}

/// Remap a Value by adding an offset.
fn remap_value(v: Value, offset: u32) -> Value {
    Value(v.0 + offset)
}

/// Remap a BlockId by adding an offset.
fn remap_block(b: BlockId, offset: u32) -> BlockId {
    BlockId(b.0 + offset)
}

/// Remap an Operand (only Value operands need remapping; constants stay the same).
fn remap_operand(op: &Operand, value_offset: u32) -> Operand {
    match op {
        Operand::Value(v) => Operand::Value(remap_value(*v, value_offset)),
        Operand::Const(c) => Operand::Const(*c),
    }
}

/// Remap all values and block references in an instruction.
fn remap_instruction(inst: &Instruction, vo: u32, bo: u32) -> Instruction {
    match inst {
        Instruction::Alloca { dest, ty, size, align, volatile } => Instruction::Alloca {
            dest: remap_value(*dest, vo),
            ty: *ty,
            size: *size,
            align: *align,
            volatile: *volatile,
        },
        Instruction::DynAlloca { dest, size, align } => Instruction::DynAlloca {
            dest: remap_value(*dest, vo),
            size: remap_operand(size, vo),
            align: *align,
        },
        Instruction::Store { val, ptr, ty, seg_override } => Instruction::Store {
            val: remap_operand(val, vo),
            ptr: remap_value(*ptr, vo),
            ty: *ty,
            seg_override: *seg_override,
        },
        Instruction::Load { dest, ptr, ty, seg_override } => Instruction::Load {
            dest: remap_value(*dest, vo),
            ptr: remap_value(*ptr, vo),
            ty: *ty,
            seg_override: *seg_override,
        },
        Instruction::BinOp { dest, op, lhs, rhs, ty } => Instruction::BinOp {
            dest: remap_value(*dest, vo),
            op: *op,
            lhs: remap_operand(lhs, vo),
            rhs: remap_operand(rhs, vo),
            ty: *ty,
        },
        Instruction::UnaryOp { dest, op, src, ty } => Instruction::UnaryOp {
            dest: remap_value(*dest, vo),
            op: *op,
            src: remap_operand(src, vo),
            ty: *ty,
        },
        Instruction::Cmp { dest, op, lhs, rhs, ty } => Instruction::Cmp {
            dest: remap_value(*dest, vo),
            op: *op,
            lhs: remap_operand(lhs, vo),
            rhs: remap_operand(rhs, vo),
            ty: *ty,
        },
        Instruction::Call { dest, func, args, arg_types, return_type, is_variadic, num_fixed_args, struct_arg_sizes } => Instruction::Call {
            dest: dest.map(|v| remap_value(v, vo)),
            func: func.clone(),
            args: args.iter().map(|a| remap_operand(a, vo)).collect(),
            arg_types: arg_types.clone(),
            return_type: *return_type,
            is_variadic: *is_variadic,
            num_fixed_args: *num_fixed_args,
            struct_arg_sizes: struct_arg_sizes.clone(),
        },
        Instruction::CallIndirect { dest, func_ptr, args, arg_types, return_type, is_variadic, num_fixed_args, struct_arg_sizes } => Instruction::CallIndirect {
            dest: dest.map(|v| remap_value(v, vo)),
            func_ptr: remap_operand(func_ptr, vo),
            args: args.iter().map(|a| remap_operand(a, vo)).collect(),
            arg_types: arg_types.clone(),
            return_type: *return_type,
            is_variadic: *is_variadic,
            num_fixed_args: *num_fixed_args,
            struct_arg_sizes: struct_arg_sizes.clone(),
        },
        Instruction::GetElementPtr { dest, base, offset, ty } => Instruction::GetElementPtr {
            dest: remap_value(*dest, vo),
            base: remap_value(*base, vo),
            offset: remap_operand(offset, vo),
            ty: *ty,
        },
        Instruction::Cast { dest, src, from_ty, to_ty } => Instruction::Cast {
            dest: remap_value(*dest, vo),
            src: remap_operand(src, vo),
            from_ty: *from_ty,
            to_ty: *to_ty,
        },
        Instruction::Copy { dest, src } => Instruction::Copy {
            dest: remap_value(*dest, vo),
            src: remap_operand(src, vo),
        },
        Instruction::GlobalAddr { dest, name } => Instruction::GlobalAddr {
            dest: remap_value(*dest, vo),
            name: name.clone(),
        },
        Instruction::Memcpy { dest, src, size } => Instruction::Memcpy {
            dest: remap_value(*dest, vo),
            src: remap_value(*src, vo),
            size: *size,
        },
        Instruction::VaArg { dest, va_list_ptr, result_ty } => Instruction::VaArg {
            dest: remap_value(*dest, vo),
            va_list_ptr: remap_value(*va_list_ptr, vo),
            result_ty: *result_ty,
        },
        Instruction::VaStart { va_list_ptr } => Instruction::VaStart {
            va_list_ptr: remap_value(*va_list_ptr, vo),
        },
        Instruction::VaEnd { va_list_ptr } => Instruction::VaEnd {
            va_list_ptr: remap_value(*va_list_ptr, vo),
        },
        Instruction::VaCopy { dest_ptr, src_ptr } => Instruction::VaCopy {
            dest_ptr: remap_value(*dest_ptr, vo),
            src_ptr: remap_value(*src_ptr, vo),
        },
        Instruction::AtomicRmw { dest, op, ptr, val, ty, ordering } => Instruction::AtomicRmw {
            dest: remap_value(*dest, vo),
            op: *op,
            ptr: remap_operand(ptr, vo),
            val: remap_operand(val, vo),
            ty: *ty,
            ordering: *ordering,
        },
        Instruction::AtomicCmpxchg { dest, ptr, expected, desired, ty, success_ordering, failure_ordering, returns_bool } => Instruction::AtomicCmpxchg {
            dest: remap_value(*dest, vo),
            ptr: remap_operand(ptr, vo),
            expected: remap_operand(expected, vo),
            desired: remap_operand(desired, vo),
            ty: *ty,
            success_ordering: *success_ordering,
            failure_ordering: *failure_ordering,
            returns_bool: *returns_bool,
        },
        Instruction::AtomicLoad { dest, ptr, ty, ordering } => Instruction::AtomicLoad {
            dest: remap_value(*dest, vo),
            ptr: remap_operand(ptr, vo),
            ty: *ty,
            ordering: *ordering,
        },
        Instruction::AtomicStore { ptr, val, ty, ordering } => Instruction::AtomicStore {
            ptr: remap_operand(ptr, vo),
            val: remap_operand(val, vo),
            ty: *ty,
            ordering: *ordering,
        },
        Instruction::Fence { ordering } => Instruction::Fence {
            ordering: *ordering,
        },
        Instruction::Phi { dest, ty, incoming } => Instruction::Phi {
            dest: remap_value(*dest, vo),
            ty: *ty,
            incoming: incoming.iter().map(|(op, bid)| {
                (remap_operand(op, vo), remap_block(*bid, bo))
            }).collect(),
        },
        Instruction::LabelAddr { dest, label } => Instruction::LabelAddr {
            dest: remap_value(*dest, vo),
            label: remap_block(*label, bo),
        },
        Instruction::GetReturnF64Second { dest } => Instruction::GetReturnF64Second {
            dest: remap_value(*dest, vo),
        },
        Instruction::SetReturnF64Second { src } => Instruction::SetReturnF64Second {
            src: remap_operand(src, vo),
        },
        Instruction::GetReturnF32Second { dest } => Instruction::GetReturnF32Second {
            dest: remap_value(*dest, vo),
        },
        Instruction::SetReturnF32Second { src } => Instruction::SetReturnF32Second {
            src: remap_operand(src, vo),
        },
        Instruction::InlineAsm { template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols, seg_overrides } => Instruction::InlineAsm {
            template: template.clone(),
            outputs: outputs.iter().map(|(c, v, n)| (c.clone(), remap_value(*v, vo), n.clone())).collect(),
            inputs: inputs.iter().map(|(c, op, n)| (c.clone(), remap_operand(op, vo), n.clone())).collect(),
            clobbers: clobbers.clone(),
            operand_types: operand_types.clone(),
            goto_labels: goto_labels.iter().map(|(name, bid)| (name.clone(), remap_block(*bid, bo))).collect(),
            input_symbols: input_symbols.clone(),
            seg_overrides: seg_overrides.clone(),
        },
        Instruction::Intrinsic { dest, op, dest_ptr, args } => Instruction::Intrinsic {
            dest: dest.map(|v| remap_value(v, vo)),
            op: op.clone(),
            dest_ptr: dest_ptr.map(|v| remap_value(v, vo)),
            args: args.iter().map(|a| remap_operand(a, vo)).collect(),
        },
        Instruction::Select { dest, cond, true_val, false_val, ty } => Instruction::Select {
            dest: remap_value(*dest, vo),
            cond: remap_operand(cond, vo),
            true_val: remap_operand(true_val, vo),
            false_val: remap_operand(false_val, vo),
            ty: *ty,
        },
        Instruction::StackSave { dest } => Instruction::StackSave {
            dest: remap_value(*dest, vo),
        },
        Instruction::StackRestore { ptr } => Instruction::StackRestore {
            ptr: remap_value(*ptr, vo),
        },
    }
}

/// Remap block references in a terminator.
fn remap_terminator(term: &Terminator, vo: u32, bo: u32) -> Terminator {
    match term {
        Terminator::Return(op) => Terminator::Return(op.map(|o| remap_operand(&o, vo))),
        Terminator::Branch(bid) => Terminator::Branch(remap_block(*bid, bo)),
        Terminator::CondBranch { cond, true_label, false_label } => Terminator::CondBranch {
            cond: remap_operand(cond, vo),
            true_label: remap_block(*true_label, bo),
            false_label: remap_block(*false_label, bo),
        },
        Terminator::IndirectBranch { target, possible_targets } => Terminator::IndirectBranch {
            target: remap_operand(target, vo),
            possible_targets: possible_targets.iter().map(|b| remap_block(*b, bo)).collect(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

/// Debug: dump function IR in a readable text format.
fn dump_function_ir(func: &IrFunction, context: &str) {
    eprintln!("=== IR DUMP {} ===", context);
    eprintln!("function {} (next_value_id={})", func.name, func.next_value_id);
    for (bi, block) in func.blocks.iter().enumerate() {
        eprintln!("  block[{}] .L{}:", bi, block.label.0);
        for (ii, inst) in block.instructions.iter().enumerate() {
            eprintln!("    [{}] {}", ii, format_instruction(inst));
        }
        eprintln!("    terminator: {}", format_terminator(&block.terminator));
    }
    eprintln!("=== END IR DUMP ===");
}

fn format_operand(op: &Operand) -> String {
    match op {
        Operand::Value(v) => format!("v{}", v.0),
        Operand::Const(c) => format!("{:?}", c),
    }
}

fn format_instruction(inst: &Instruction) -> String {
    match inst {
        Instruction::Alloca { dest, ty, size, align, .. } => {
            format!("v{} = alloca {:?} size={} align={}", dest.0, ty, size, align)
        }
        Instruction::Store { val, ptr, ty, .. } => {
            format!("store {:?} {} -> v{}", ty, format_operand(val), ptr.0)
        }
        Instruction::Load { dest, ptr, ty, .. } => {
            format!("v{} = load {:?} v{}", dest.0, ty, ptr.0)
        }
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            format!("v{} = {:?} {:?} {}, {}", dest.0, op, ty, format_operand(lhs), format_operand(rhs))
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            format!("v{} = {:?} {:?} {}", dest.0, op, ty, format_operand(src))
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            format!("v{} = cmp {:?} {:?} {}, {}", dest.0, op, ty, format_operand(lhs), format_operand(rhs))
        }
        Instruction::Call { dest, func, args, .. } => {
            let args_str: Vec<String> = args.iter().map(|a| format_operand(a)).collect();
            if let Some(d) = dest {
                format!("v{} = call {}({})", d.0, func, args_str.join(", "))
            } else {
                format!("call {}({})", func, args_str.join(", "))
            }
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            format!("v{} = cast {:?}->{:?} {}", dest.0, from_ty, to_ty, format_operand(src))
        }
        Instruction::Copy { dest, src } => {
            format!("v{} = copy {}", dest.0, format_operand(src))
        }
        Instruction::GetElementPtr { dest, base, offset, ty } => {
            format!("v{} = gep {:?} v{}, {}", dest.0, ty, base.0, format_operand(offset))
        }
        Instruction::Phi { dest, ty, incoming } => {
            let inc_str: Vec<String> = incoming.iter()
                .map(|(op, bid)| format!("[{}, .L{}]", format_operand(op), bid.0))
                .collect();
            format!("v{} = phi {:?} {}", dest.0, ty, inc_str.join(", "))
        }
        Instruction::GlobalAddr { dest, name } => {
            format!("v{} = globaladdr @{}", dest.0, name)
        }
        Instruction::Memcpy { dest, src, size } => {
            format!("memcpy v{}, v{}, {}", dest.0, src.0, size)
        }
        Instruction::Select { dest, cond, true_val, false_val, ty } => {
            format!("v{} = select {:?} {}, {}, {}", dest.0, ty, format_operand(cond), format_operand(true_val), format_operand(false_val))
        }
        _ => format!("{:?}", inst),
    }
}

fn format_terminator(term: &Terminator) -> String {
    match term {
        Terminator::Return(Some(op)) => format!("ret {}", format_operand(op)),
        Terminator::Return(None) => "ret void".to_string(),
        Terminator::Branch(bid) => format!("br .L{}", bid.0),
        Terminator::CondBranch { cond, true_label, false_label } => {
            format!("condbr {}, .L{}, .L{}", format_operand(cond), true_label.0, false_label.0)
        }
        Terminator::IndirectBranch { target, .. } => {
            format!("indirectbr {}", format_operand(target))
        }
        Terminator::Unreachable => "unreachable".to_string(),
    }
}
