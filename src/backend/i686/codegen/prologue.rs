//! I686Codegen: prologue/epilogue and stack frame operations.

use crate::ir::ir::{IrFunction, Value};
use crate::common::types::IrType;
use crate::backend::generation::{
    is_i128_type, calculate_stack_space_common, run_regalloc_and_merge_clobbers,
    filter_available_regs, find_param_alloca, collect_inline_asm_callee_saved,
};
use crate::backend::call_emit::{ParamClass, classify_params};
use crate::emit;
use super::codegen::{
    I686Codegen, phys_reg_name, i686_constraint_to_phys, i686_clobber_to_phys,
    I686_CALLEE_SAVED, I686_CALLER_SAVED,
};
use crate::backend::regalloc::PhysReg;
use crate::backend::traits::ArchCodegen;

impl I686Codegen {
    // ---- calculate_stack_space ----

    pub(super) fn calculate_stack_space_impl(&mut self, func: &IrFunction) -> i64 {
        self.is_variadic = func.is_variadic;
        self.is_fastcall = func.is_fastcall;
        self.current_return_type = func.return_type;

        // Compute named parameter stack bytes for va_start (variadic functions).
        if func.is_variadic {
            let config = self.call_abi_config();
            let classification = crate::backend::call_emit::classify_params_full(func, &config);
            self.va_named_stack_bytes = classification.total_stack_bytes;
        }

        // Run register allocator before stack space computation.
        let mut asm_clobbered_regs: Vec<PhysReg> = Vec::new();
        collect_inline_asm_callee_saved(
            func, &mut asm_clobbered_regs,
            i686_constraint_to_phys,
            i686_clobber_to_phys,
        );
        // In PIC mode, %ebx (PhysReg(0)) is reserved as the GOT base pointer.
        if self.state.pic_mode {
            if !asm_clobbered_regs.contains(&PhysReg(0)) {
                asm_clobbered_regs.push(PhysReg(0));
            }
        }
        let available_regs = filter_available_regs(I686_CALLEE_SAVED, &asm_clobbered_regs);

        let caller_saved_regs = I686_CALLER_SAVED.to_vec();

        let (reg_assigned, cached_liveness) = run_regalloc_and_merge_clobbers(
            func, available_regs, caller_saved_regs, &asm_clobbered_regs,
            &mut self.reg_assignments, &mut self.used_callee_saved,
            false,
        );

        // In PIC mode, %ebx must be saved/restored as a callee-saved register.
        if self.state.pic_mode && !self.used_callee_saved.contains(&PhysReg(0)) {
            self.used_callee_saved.insert(0, PhysReg(0));
        }

        let callee_saved_bytes = self.used_callee_saved.len() as i64 * 4;

        calculate_stack_space_common(&mut self.state, func, callee_saved_bytes, |space, alloc_size, align| {
            let effective_align = if align > 0 { align.max(4) } else { 4 };
            let alloc = (alloc_size + 3) & !3;
            let required = space + alloc;
            let new_space = if effective_align >= 16 {
                let bias = 8i64;
                let a = effective_align;
                let rem = ((required % a) + a) % a;
                let needed = if rem <= bias { bias - rem } else { a - rem + bias };
                required + needed
            } else {
                ((required + effective_align - 1) / effective_align) * effective_align
            };
            (-new_space, new_space)
        }, &reg_assigned, cached_liveness)
    }

    // ---- aligned_frame_size ----

    pub(super) fn aligned_frame_size_impl(&self, raw_space: i64) -> i64 {
        let callee_saved_bytes = self.used_callee_saved.len() as i64 * 4;
        let raw_locals = raw_space - callee_saved_bytes;
        let fixed_overhead = callee_saved_bytes + 8;
        let needed = raw_locals + fixed_overhead;
        let aligned = (needed + 15) & !15;
        aligned - fixed_overhead
    }

    // ---- emit_prologue ----

    pub(super) fn emit_prologue_impl(&mut self, _func: &IrFunction, frame_size: i64) {
        // TODO: when omit_frame_pointer is true, skip the frame pointer setup
        // and use ESP-relative addressing for all slot accesses. This requires
        // tracking ESP offset at each instruction point and adjusting all
        // slot references accordingly.
        self.state.emit("    pushl %ebp");
        self.state.emit("    movl %esp, %ebp");

        for &reg in self.used_callee_saved.iter() {
            let name = phys_reg_name(reg);
            emit!(self.state, "    pushl %{}", name);
        }

        if self.state.pic_mode {
            debug_assert!(self.used_callee_saved.contains(&PhysReg(0)),
                "PIC mode requires ebx in used_callee_saved");
            self.state.emit("    call __x86.get_pc_thunk.bx");
            self.state.emit("    addl $_GLOBAL_OFFSET_TABLE_, %ebx");
            self.needs_pc_thunk_bx = true;
        }

        if frame_size > 0 {
            emit!(self.state, "    subl ${}, %esp", frame_size);
        }
    }

    // ---- emit_epilogue ----

    pub(super) fn emit_epilogue_impl(&mut self, _frame_size: i64) {
        let callee_saved_bytes = self.used_callee_saved.len() as i64 * 4;
        if callee_saved_bytes > 0 {
            emit!(self.state, "    leal -{}(%ebp), %esp", callee_saved_bytes);
        } else {
            self.state.emit("    movl %ebp, %esp");
        }

        for &reg in self.used_callee_saved.iter().rev() {
            let name = phys_reg_name(reg);
            emit!(self.state, "    popl %{}", name);
        }

        self.state.emit("    popl %ebp");
    }

    // ---- emit_store_params ----

    pub(super) fn emit_store_params_impl(&mut self, func: &IrFunction) {
        let config = self.call_abi_config();
        let param_classes = classify_params(func, &config);
        self.state.param_classes = param_classes.clone();
        self.state.num_params = func.params.len();
        self.state.func_is_variadic = func.is_variadic;

        self.state.param_alloca_slots = (0..func.params.len()).map(|i| {
            find_param_alloca(func, i).and_then(|(dest, ty)| {
                self.state.get_slot(dest.0).map(|slot| (slot, ty))
            })
        }).collect();

        let fastcall_reg_count = if self.is_fastcall {
            self.count_fastcall_reg_params(func)
        } else {
            0
        };
        self.fastcall_reg_param_count = fastcall_reg_count;

        if self.is_fastcall {
            let mut total_stack_bytes: usize = 0;
            for (i, _p) in func.params.iter().enumerate() {
                if i < fastcall_reg_count { continue; }
                let ty = func.params[i].ty;
                let size = match ty {
                    IrType::I64 | IrType::U64 | IrType::F64 => 8,
                    IrType::F128 => 12,
                    _ if is_i128_type(ty) => 16,
                    _ => 4,
                };
                total_stack_bytes += size;
            }
            self.fastcall_stack_cleanup = total_stack_bytes;
        } else {
            self.fastcall_stack_cleanup = 0;
        }

        let stack_base: i64 = 8;
        let mut fastcall_reg_idx = 0usize;

        for (i, _param) in func.params.iter().enumerate() {
            let class = param_classes[i];

            let (slot, ty, dest_id) = if let Some((dest, ty)) = find_param_alloca(func, i) {
                if let Some(slot) = self.state.get_slot(dest.0) {
                    (slot, ty, dest.0)
                } else {
                    if self.is_fastcall && fastcall_reg_idx < fastcall_reg_count && i < func.params.len()
                        && self.is_fastcall_reg_eligible(ty) {
                            fastcall_reg_idx += 1;
                        }
                    continue;
                }
            } else {
                if self.is_fastcall && fastcall_reg_idx < fastcall_reg_count && i < func.params.len() {
                    let param_ty = func.params[i].ty;
                    if self.is_fastcall_reg_eligible(param_ty) {
                        fastcall_reg_idx += 1;
                    }
                }
                continue;
            };

            if self.is_fastcall && fastcall_reg_idx < fastcall_reg_count && self.is_fastcall_reg_eligible(ty) {
                let src_reg_full = if fastcall_reg_idx == 0 { "%ecx" } else { "%edx" };
                // For sub-int types, sign/zero-extend to full 32-bit before
                // storing to the 4-byte SSA slot (avoids partial-write issues).
                match ty {
                    IrType::I8 => {
                        let src_byte = if fastcall_reg_idx == 0 { "%cl" } else { "%dl" };
                        emit!(self.state, "    movsbl {}, {}", src_byte, src_reg_full);
                        emit!(self.state, "    movl {}, {}(%ebp)", src_reg_full, slot.0);
                    }
                    IrType::U8 => {
                        let src_byte = if fastcall_reg_idx == 0 { "%cl" } else { "%dl" };
                        emit!(self.state, "    movzbl {}, {}", src_byte, src_reg_full);
                        emit!(self.state, "    movl {}, {}(%ebp)", src_reg_full, slot.0);
                    }
                    IrType::I16 => {
                        let src_word = if fastcall_reg_idx == 0 { "%cx" } else { "%dx" };
                        emit!(self.state, "    movswl {}, {}", src_word, src_reg_full);
                        emit!(self.state, "    movl {}, {}(%ebp)", src_reg_full, slot.0);
                    }
                    IrType::U16 => {
                        let src_word = if fastcall_reg_idx == 0 { "%cx" } else { "%dx" };
                        emit!(self.state, "    movzwl {}, {}", src_word, src_reg_full);
                        emit!(self.state, "    movl {}, {}(%ebp)", src_reg_full, slot.0);
                    }
                    _ => {
                        emit!(self.state, "    movl {}, {}(%ebp)", src_reg_full, slot.0);
                    }
                }
                fastcall_reg_idx += 1;
                continue;
            }

            let stack_offset_adjust = if self.is_fastcall { fastcall_reg_count as i64 * 4 } else { 0 };

            match class {
                ParamClass::StackScalar { offset } => {
                    let src_offset = stack_base + offset - stack_offset_adjust;
                    if ty == IrType::F64 || ty == IrType::I64 || ty == IrType::U64 {
                        emit!(self.state, "    movl {}(%ebp), %eax", src_offset);
                        emit!(self.state, "    movl %eax, {}(%ebp)", slot.0);
                        emit!(self.state, "    movl {}(%ebp), %eax", src_offset + 4);
                        emit!(self.state, "    movl %eax, {}(%ebp)", slot.0 + 4);
                    } else {
                        let load_instr = self.mov_load_for_type(ty);
                        emit!(self.state, "    {} {}(%ebp), %eax", load_instr, src_offset);
                        // Always store full 32-bit value to SSA slot. The load
                        // instruction above already sign/zero-extended sub-int
                        // types into the full eax register. Using movb/movw here
                        // would leave garbage in the upper bytes of the 4-byte
                        // slot, which gets read back later by movl.
                        emit!(self.state, "    movl %eax, {}(%ebp)", slot.0);
                    }
                }
                ParamClass::StructStack { offset, size } => {
                    let src = stack_base + offset - stack_offset_adjust;
                    let mut copied = 0usize;
                    while copied + 4 <= size {
                        emit!(self.state, "    movl {}(%ebp), %eax", src + copied as i64);
                        emit!(self.state, "    movl %eax, {}(%ebp)", slot.0 + copied as i64);
                        copied += 4;
                    }
                    while copied < size {
                        emit!(self.state, "    movb {}(%ebp), %al", src + copied as i64);
                        emit!(self.state, "    movb %al, {}(%ebp)", slot.0 + copied as i64);
                        copied += 1;
                    }
                }
                ParamClass::LargeStructStack { offset, size } => {
                    let src = stack_base + offset - stack_offset_adjust;
                    let mut copied = 0usize;
                    while copied + 4 <= size {
                        emit!(self.state, "    movl {}(%ebp), %eax", src + copied as i64);
                        emit!(self.state, "    movl %eax, {}(%ebp)", slot.0 + copied as i64);
                        copied += 4;
                    }
                    while copied < size {
                        emit!(self.state, "    movb {}(%ebp), %al", src + copied as i64);
                        emit!(self.state, "    movb %al, {}(%ebp)", slot.0 + copied as i64);
                        copied += 1;
                    }
                }
                ParamClass::F128AlwaysStack { offset } => {
                    let src = stack_base + offset - stack_offset_adjust;
                    emit!(self.state, "    fldt {}(%ebp)", src);
                    emit!(self.state, "    fstpt {}(%ebp)", slot.0);
                    self.state.f128_direct_slots.insert(dest_id);
                }
                ParamClass::I128Stack { offset } => {
                    let src = stack_base + offset - stack_offset_adjust;
                    for j in (0..16).step_by(4) {
                        emit!(self.state, "    movl {}(%ebp), %eax", src + j as i64);
                        emit!(self.state, "    movl %eax, {}(%ebp)", slot.0 + j as i64);
                    }
                }
                ParamClass::F128Stack { offset } => {
                    let src = stack_base + offset - stack_offset_adjust;
                    emit!(self.state, "    fldt {}(%ebp)", src);
                    emit!(self.state, "    fstpt {}(%ebp)", slot.0);
                    self.state.f128_direct_slots.insert(dest_id);
                }
                ParamClass::IntReg { reg_idx } => {
                    // regparm: param arrives in EAX/EDX/ECX (reg_idx 0/1/2)
                    let regparm_regs_full = ["%eax", "%edx", "%ecx"];
                    let regparm_regs_byte = ["%al", "%dl", "%cl"];
                    let regparm_regs_word = ["%ax", "%dx", "%cx"];
                    let src_full = regparm_regs_full[reg_idx];
                    match ty {
                        IrType::I8 => {
                            let src_byte = regparm_regs_byte[reg_idx];
                            emit!(self.state, "    movsbl {}, {}", src_byte, src_full);
                            emit!(self.state, "    movl {}, {}(%ebp)", src_full, slot.0);
                        }
                        IrType::U8 => {
                            let src_byte = regparm_regs_byte[reg_idx];
                            emit!(self.state, "    movzbl {}, {}", src_byte, src_full);
                            emit!(self.state, "    movl {}, {}(%ebp)", src_full, slot.0);
                        }
                        IrType::I16 => {
                            let src_word = regparm_regs_word[reg_idx];
                            emit!(self.state, "    movswl {}, {}", src_word, src_full);
                            emit!(self.state, "    movl {}, {}(%ebp)", src_full, slot.0);
                        }
                        IrType::U16 => {
                            let src_word = regparm_regs_word[reg_idx];
                            emit!(self.state, "    movzwl {}, {}", src_word, src_full);
                            emit!(self.state, "    movl {}, {}(%ebp)", src_full, slot.0);
                        }
                        _ => {
                            emit!(self.state, "    movl {}, {}(%ebp)", src_full, slot.0);
                        }
                    }
                }
                _ => {
                    // Remaining register classes (FloatReg, StructByValReg, etc.)
                    // don't apply to i686's ABI classification.
                }
            }
        }
    }

    // ---- emit_param_ref ----

    pub(super) fn emit_param_ref_impl(&mut self, dest: &Value, param_idx: usize, ty: IrType) {
        use crate::backend::call_emit::ParamClass;

        if param_idx < self.state.param_alloca_slots.len() {
            if let Some((alloca_slot, _alloca_ty)) = self.state.param_alloca_slots[param_idx] {
                if let Some(dest_slot) = self.state.get_slot(dest.0) {
                    if dest_slot.0 == alloca_slot.0 {
                        return;
                    }
                }
                if let Some(dest_slot) = self.state.get_slot(dest.0) {
                    if is_i128_type(ty) {
                        for i in (0..16).step_by(4) {
                            emit!(self.state, "    movl {}(%ebp), %eax", alloca_slot.0 + i as i64);
                            emit!(self.state, "    movl %eax, {}(%ebp)", dest_slot.0 + i as i64);
                        }
                    } else if ty == IrType::F128 {
                        emit!(self.state, "    fldt {}(%ebp)", alloca_slot.0);
                        emit!(self.state, "    fstpt {}(%ebp)", dest_slot.0);
                        self.state.f128_direct_slots.insert(dest.0);
                    } else if ty == IrType::F64 || ty == IrType::I64 || ty == IrType::U64 {
                        emit!(self.state, "    movl {}(%ebp), %eax", alloca_slot.0);
                        emit!(self.state, "    movl %eax, {}(%ebp)", dest_slot.0);
                        emit!(self.state, "    movl {}(%ebp), %eax", alloca_slot.0 + 4);
                        emit!(self.state, "    movl %eax, {}(%ebp)", dest_slot.0 + 4);
                    } else {
                        let load_instr = self.mov_load_for_type(ty);
                        emit!(self.state, "    {} {}(%ebp), %eax", load_instr, alloca_slot.0);
                        self.store_eax_to(dest);
                    }
                    return;
                }
            }
        }

        if self.is_fastcall && param_idx < self.fastcall_reg_param_count {
            if let Some(Some((slot, _slot_ty))) = self.state.param_alloca_slots.get(param_idx) {
                let load_instr = self.mov_load_for_type(ty);
                emit!(self.state, "    {} {}(%ebp), %eax", load_instr, slot.0);
                self.store_eax_to(dest);
            }
            return;
        }

        let stack_base: i64 = 8;
        let stack_offset_adjust = if self.is_fastcall { self.fastcall_reg_param_count as i64 * 4 } else { 0 };
        let param_offset = if param_idx < self.state.param_classes.len() {
            match self.state.param_classes[param_idx] {
                ParamClass::StackScalar { offset } |
                ParamClass::StructStack { offset, .. } |
                ParamClass::LargeStructStack { offset, .. } |
                ParamClass::F128AlwaysStack { offset } |
                ParamClass::I128Stack { offset } |
                ParamClass::F128Stack { offset } |
                ParamClass::LargeStructByRefStack { offset, .. } => stack_base + offset - stack_offset_adjust,
                ParamClass::IntReg { .. } => {
                    // Regparm: param was stored to its alloca slot in emit_store_params.
                    // This should have been handled by the alloca_slot path above.
                    // If we get here, just use a fallback offset.
                    stack_base + (param_idx as i64) * 4
                }
                _ => stack_base + (param_idx as i64) * 4,
            }
        } else {
            stack_base + (param_idx as i64) * 4
        };

        if is_i128_type(ty) {
            if let Some(slot) = self.state.get_slot(dest.0) {
                for i in (0..16).step_by(4) {
                    emit!(self.state, "    movl {}(%ebp), %eax", param_offset + i as i64);
                    emit!(self.state, "    movl %eax, {}(%ebp)", slot.0 + i as i64);
                }
            }
        } else if ty == IrType::F128 {
            if let Some(dest_slot) = self.state.get_slot(dest.0) {
                emit!(self.state, "    fldt {}(%ebp)", param_offset);
                emit!(self.state, "    fstpt {}(%ebp)", dest_slot.0);
                self.state.f128_direct_slots.insert(dest.0);
            }
        } else if ty == IrType::F64 || ty == IrType::I64 || ty == IrType::U64 {
            if let Some(slot) = self.state.get_slot(dest.0) {
                emit!(self.state, "    movl {}(%ebp), %eax", param_offset);
                emit!(self.state, "    movl %eax, {}(%ebp)", slot.0);
                emit!(self.state, "    movl {}(%ebp), %eax", param_offset + 4);
                emit!(self.state, "    movl %eax, {}(%ebp)", slot.0 + 4);
            }
        } else {
            let load_instr = self.mov_load_for_type(ty);
            emit!(self.state, "    {} {}(%ebp), %eax", load_instr, param_offset);
            self.store_eax_to(dest);
        }
    }

    // ---- emit_epilogue_and_ret ----

    pub(super) fn emit_epilogue_and_ret_impl(&mut self, frame_size: i64) {
        self.emit_epilogue(frame_size);
        if self.state.uses_sret {
            self.state.emit("    ret $4");
        } else if self.is_fastcall && self.fastcall_stack_cleanup > 0 {
            emit!(self.state, "    ret ${}", self.fastcall_stack_cleanup);
        } else {
            self.state.emit("    ret");
        }
    }

    // ---- store/load instr for type ----

    pub(super) fn store_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        self.mov_store_for_type(ty)
    }

    pub(super) fn load_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        self.mov_load_for_type(ty)
    }
}
