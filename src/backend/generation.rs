//! Module, function, and instruction generation dispatch.
//!
//! This module contains the top-level entry points that drive code generation:
//! - `generate_module`: emits data sections and iterates over functions
//! - `generate_function`: emits prologue, basic blocks, and epilogue
//! - `generate_instruction`: dispatches each IR instruction to arch trait methods
//! - `generate_terminator`: dispatches terminators to arch trait methods
//!
//! These functions are arch-independent — they use the `ArchCodegen` trait to call
//! into the backend-specific implementations.

use crate::ir::ir::*;
use crate::common::types::IrType;
use super::common;
use super::traits::ArchCodegen;
use super::state::StackSlot;

/// Generate assembly for a module using the given architecture's codegen.
pub fn generate_module(cg: &mut dyn ArchCodegen, module: &IrModule) -> String {
    // Build the set of locally-defined symbols for PIC mode.
    {
        let state = cg.state();
        for func in &module.functions {
            if func.is_static {
                state.local_symbols.insert(func.name.clone());
            }
        }
        for global in &module.globals {
            if global.is_static {
                state.local_symbols.insert(global.name.clone());
            }
        }
        for (label, _) in &module.string_literals {
            state.local_symbols.insert(label.clone());
        }
        for (label, _) in &module.wide_string_literals {
            state.local_symbols.insert(label.clone());
        }
    }

    // Emit data sections
    let ptr_dir = cg.ptr_directive();
    common::emit_data_sections(&mut cg.state().out, module, ptr_dir);

    // Text section
    cg.state().emit(".section .text");
    for func in &module.functions {
        if !func.is_declaration {
            generate_function(cg, func);
        }
    }

    // Emit .init_array for constructor functions
    for ctor in &module.constructors {
        cg.state().emit("");
        cg.state().emit(".section .init_array,\"aw\",@init_array");
        cg.state().emit(".align 8");
        cg.state().emit(&format!("{} {}", ptr_dir.as_str(), ctor));
    }

    // Emit .fini_array for destructor functions
    for dtor in &module.destructors {
        cg.state().emit("");
        cg.state().emit(".section .fini_array,\"aw\",@fini_array");
        cg.state().emit(".align 8");
        cg.state().emit(&format!("{} {}", ptr_dir.as_str(), dtor));
    }

    // Emit .note.GNU-stack section to indicate non-executable stack
    cg.state().emit("");
    cg.state().emit(".section .note.GNU-stack,\"\",@progbits");

    std::mem::take(&mut cg.state().out.buf)
}

/// Generate code for a single function.
fn generate_function(cg: &mut dyn ArchCodegen, func: &IrFunction) {
    cg.state().reset_for_function();

    let type_dir = cg.function_type_directive();
    if !func.is_static {
        cg.state().emit(&format!(".globl {}", func.name));
    }
    cg.state().emit(&format!(".type {}, {}", func.name, type_dir));
    cg.state().emit(&format!("{}:", func.name));

    // Pre-scan for DynAlloca: if present, the epilogue must restore SP from
    // the frame pointer instead of adding back the compile-time frame size.
    let has_dyn_alloca = func.blocks.iter().any(|block| {
        block.instructions.iter().any(|inst| matches!(inst, Instruction::DynAlloca { .. }))
    });
    cg.state().has_dyn_alloca = has_dyn_alloca;

    // Calculate stack space and emit prologue
    let raw_space = cg.calculate_stack_space(func);
    let frame_size = cg.aligned_frame_size(raw_space);
    cg.emit_prologue(func, frame_size);

    // Store parameters
    cg.emit_store_params(func);

    // Generate basic blocks
    for block in &func.blocks {
        if block.label != "entry" {
            cg.state().emit(&format!("{}:", block.label));
        }
        for inst in &block.instructions {
            generate_instruction(cg, inst);
        }
        generate_terminator(cg, &block.terminator, frame_size);
    }

    cg.state().emit(&format!(".size {}, .-{}", func.name, func.name));
    cg.state().emit("");
}

/// Dispatch a single IR instruction to the appropriate arch method.
fn generate_instruction(cg: &mut dyn ArchCodegen, inst: &Instruction) {
    match inst {
        Instruction::Alloca { .. } => {
            // Space already allocated in prologue
        }
        Instruction::DynAlloca { dest, size, align } => {
            cg.emit_dyn_alloca(dest, size, *align);
        }
        Instruction::Store { val, ptr, ty } => {
            cg.emit_store(val, ptr, *ty);
        }
        Instruction::Load { dest, ptr, ty } => {
            cg.emit_load(dest, ptr, *ty);
        }
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            cg.emit_binop(dest, *op, lhs, rhs, *ty);
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            cg.emit_unaryop(dest, *op, src, *ty);
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            cg.emit_cmp(dest, *op, lhs, rhs, *ty);
        }
        Instruction::Call { dest, func, args, arg_types, return_type, is_variadic, num_fixed_args, struct_arg_sizes } => {
            cg.emit_call(args, arg_types, Some(func), None, *dest, *return_type, *is_variadic, *num_fixed_args, struct_arg_sizes);
        }
        Instruction::CallIndirect { dest, func_ptr, args, arg_types, return_type, is_variadic, num_fixed_args, struct_arg_sizes } => {
            cg.emit_call(args, arg_types, None, Some(func_ptr), *dest, *return_type, *is_variadic, *num_fixed_args, struct_arg_sizes);
        }
        Instruction::GlobalAddr { dest, name } => {
            cg.emit_global_addr(dest, name);
        }
        Instruction::Copy { dest, src } => {
            let is_i128_copy = match src {
                Operand::Value(v) => cg.state_ref().is_i128_value(v.0),
                Operand::Const(IrConst::I128(_)) => true,
                _ => false,
            };
            if is_i128_copy {
                cg.state().i128_values.insert(dest.0);
                cg.emit_copy_i128(dest, src);
            } else {
                cg.emit_load_operand(src);
                cg.emit_store_result(dest);
            }
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            cg.emit_cast(dest, src, *from_ty, *to_ty);
        }
        Instruction::GetElementPtr { dest, base, offset, .. } => {
            cg.emit_gep(dest, base, offset);
        }
        Instruction::Memcpy { dest, src, size } => {
            cg.emit_memcpy(dest, src, *size);
        }
        Instruction::VaArg { dest, va_list_ptr, result_ty } => {
            cg.emit_va_arg(dest, va_list_ptr, *result_ty);
        }
        Instruction::VaStart { va_list_ptr } => {
            cg.emit_va_start(va_list_ptr);
        }
        Instruction::VaEnd { va_list_ptr } => {
            cg.emit_va_end(va_list_ptr);
        }
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            cg.emit_va_copy(dest_ptr, src_ptr);
        }
        Instruction::AtomicRmw { dest, op, ptr, val, ty, ordering } => {
            cg.emit_atomic_rmw(dest, *op, ptr, val, *ty, *ordering);
        }
        Instruction::AtomicCmpxchg { dest, ptr, expected, desired, ty, success_ordering, failure_ordering, returns_bool } => {
            cg.emit_atomic_cmpxchg(dest, ptr, expected, desired, *ty, *success_ordering, *failure_ordering, *returns_bool);
        }
        Instruction::AtomicLoad { dest, ptr, ty, ordering } => {
            cg.emit_atomic_load(dest, ptr, *ty, *ordering);
        }
        Instruction::AtomicStore { ptr, val, ty, ordering } => {
            cg.emit_atomic_store(ptr, val, *ty, *ordering);
        }
        Instruction::Fence { ordering } => {
            cg.emit_fence(*ordering);
        }
        Instruction::Phi { .. } => {
            // Phi nodes are resolved before codegen by lowering to copies
        }
        Instruction::LabelAddr { dest, label } => {
            cg.emit_label_addr(dest, label);
        }
        Instruction::GetReturnF64Second { dest } => {
            cg.emit_get_return_f64_second(dest);
        }
        Instruction::SetReturnF64Second { src } => {
            cg.emit_set_return_f64_second(src);
        }
        Instruction::GetReturnF32Second { dest } => {
            cg.emit_get_return_f32_second(dest);
        }
        Instruction::SetReturnF32Second { src } => {
            cg.emit_set_return_f32_second(src);
        }
        Instruction::InlineAsm { template, outputs, inputs, clobbers, operand_types } => {
            cg.emit_inline_asm(template, outputs, inputs, clobbers, operand_types);
        }
        Instruction::X86SseOp { dest, op, dest_ptr, args } => {
            cg.emit_x86_sse_op(dest, op, dest_ptr, args);
        }
    }
}

/// Dispatch a terminator to the appropriate arch method.
fn generate_terminator(cg: &mut dyn ArchCodegen, term: &Terminator, frame_size: i64) {
    match term {
        Terminator::Return(val) => {
            cg.emit_return(val.as_ref(), frame_size);
        }
        Terminator::Branch(label) => {
            cg.emit_branch(label);
        }
        Terminator::CondBranch { cond, true_label, false_label } => {
            cg.emit_cond_branch(cond, true_label, false_label);
        }
        Terminator::IndirectBranch { target, .. } => {
            cg.emit_indirect_branch(target);
        }
        Terminator::Unreachable => {
            cg.emit_unreachable();
        }
    }
}

/// Check if an IR type is a 128-bit integer type (I128 or U128).
pub fn is_i128_type(ty: IrType) -> bool {
    matches!(ty, IrType::I128 | IrType::U128)
}

/// Shared stack space calculation: iterates over all instructions, assigns stack
/// slots for allocas and value results. Arch-specific offset direction is handled
/// by the `assign_slot` closure.
///
/// `initial_offset`: starting offset (e.g., 0 for x86, 16 for ARM/RISC-V to skip saved regs)
/// `assign_slot`: maps (current_space, raw_alloca_size, alignment) -> (slot_offset, new_space)
pub fn calculate_stack_space_common(
    state: &mut super::state::CodegenState,
    func: &IrFunction,
    initial_offset: i64,
    assign_slot: impl Fn(i64, i64, i64) -> (i64, i64),
) -> i64 {
    let mut space = initial_offset;
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, size, ty, align } = inst {
                let effective_align = *align;
                let extra = if effective_align > 16 { effective_align - 1 } else { 0 };
                let raw_size = if *size == 0 { 8 } else { *size as i64 };
                let (slot, new_space) = assign_slot(space, raw_size + extra as i64, *align as i64);
                state.value_locations.insert(dest.0, StackSlot(slot));
                state.alloca_values.insert(dest.0);
                state.alloca_types.insert(dest.0, *ty);
                if effective_align > 16 {
                    state.alloca_alignments.insert(dest.0, effective_align);
                }
                space = new_space;
            } else if let Some(dest) = inst.dest() {
                let is_i128 = matches!(inst.result_type(), Some(IrType::I128) | Some(IrType::U128));
                let slot_size = if is_i128 { 16 } else { 8 };
                let (slot, new_space) = assign_slot(space, slot_size, 0);
                state.value_locations.insert(dest.0, StackSlot(slot));
                if is_i128 {
                    state.i128_values.insert(dest.0);
                }
                space = new_space;
            }
        }
    }
    space
}

/// Find the nth alloca instruction in the entry block (used for parameter storage).
pub fn find_param_alloca(func: &IrFunction, param_idx: usize) -> Option<(Value, IrType)> {
    func.blocks.first().and_then(|block| {
        block.instructions.iter()
            .filter(|i| matches!(i, Instruction::Alloca { .. }))
            .nth(param_idx)
            .and_then(|inst| {
                if let Instruction::Alloca { dest, ty, .. } = inst {
                    Some((*dest, *ty))
                } else {
                    None
                }
            })
    })
}
