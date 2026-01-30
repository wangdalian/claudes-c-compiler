//! ArmCodegen: variadic function operations (va_arg, va_start, va_copy).

use crate::ir::ir::Value;
use crate::common::types::IrType;
use super::codegen::{ArmCodegen, callee_saved_name};

impl ArmCodegen {
    pub(super) fn emit_va_arg_impl(&mut self, dest: &Value, va_list_ptr: &Value, result_ty: IrType) {
        let is_fp = result_ty.is_float();
        let is_f128 = result_ty.is_long_double();

        if self.state.is_alloca(va_list_ptr.0) {
            if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
                self.emit_add_fp_offset("x1", slot.0);
            }
        } else if let Some(&reg) = self.reg_assignments.get(&va_list_ptr.0) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x1, {}", reg_name));
        } else if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
            self.emit_load_from_sp("x1", slot.0, "ldr");
        }

        if is_f128 {
            let label_id = self.state.next_label_id();
            let label_stack = format!(".Lva_stack_{}", label_id);
            let label_done = format!(".Lva_done_{}", label_id);

            self.state.emit("    ldrsw x2, [x1, #28]");
            self.state.emit_fmt(format_args!("    tbz x2, #63, {}", label_stack));
            self.state.emit("    ldr x3, [x1, #16]");
            self.state.emit("    add x3, x3, x2");
            self.state.emit("    add w2, w2, #16");
            self.state.emit("    str w2, [x1, #28]");
            self.state.emit("    ldr q0, [x3]");
            self.state.emit("    bl __trunctfdf2");
            self.state.emit("    fmov x0, d0");
            self.state.emit_fmt(format_args!("    b {}", label_done));

            self.state.emit_fmt(format_args!("{}:", label_stack));
            self.state.emit("    ldr x3, [x1]");
            self.state.emit("    add x3, x3, #15");
            self.state.emit("    and x3, x3, #-16");
            self.state.emit("    mov x4, x1");
            self.state.emit("    ldr q0, [x3]");
            self.state.emit("    add x3, x3, #16");
            self.state.emit("    str x3, [x4]");
            self.state.emit("    bl __trunctfdf2");
            self.state.emit("    fmov x0, d0");

            self.state.emit_fmt(format_args!("{}:", label_done));
            self.state.reg_cache.invalidate_all();
        } else if is_fp {
            let label_id = self.state.next_label_id();
            let label_stack = format!(".Lva_stack_{}", label_id);
            let label_done = format!(".Lva_done_{}", label_id);

            self.state.emit("    ldrsw x2, [x1, #28]");
            self.state.emit_fmt(format_args!("    tbz x2, #63, {}", label_stack));
            self.state.emit("    ldr x3, [x1, #16]");
            self.state.emit("    add x3, x3, x2");
            self.state.emit("    add w2, w2, #16");
            self.state.emit("    str w2, [x1, #28]");
            if result_ty == IrType::F32 {
                self.state.emit("    ldr w0, [x3]");
            } else {
                self.state.emit("    ldr x0, [x3]");
            }
            self.state.emit_fmt(format_args!("    b {}", label_done));

            self.state.emit_fmt(format_args!("{}:", label_stack));
            self.state.emit("    ldr x3, [x1]");
            if result_ty == IrType::F32 {
                self.state.emit("    ldr w0, [x3]");
            } else {
                self.state.emit("    ldr x0, [x3]");
            }
            self.state.emit("    add x3, x3, #8");
            self.state.emit("    str x3, [x1]");

            self.state.emit_fmt(format_args!("{}:", label_done));
        } else {
            let label_id = self.state.next_label_id();
            let label_stack = format!(".Lva_stack_{}", label_id);
            let label_done = format!(".Lva_done_{}", label_id);

            self.state.emit("    ldrsw x2, [x1, #24]");
            self.state.emit_fmt(format_args!("    tbz x2, #63, {}", label_stack));
            self.state.emit("    ldr x3, [x1, #8]");
            self.state.emit("    add x3, x3, x2");
            self.state.emit("    add w2, w2, #8");
            self.state.emit("    str w2, [x1, #24]");
            self.state.emit("    ldr x0, [x3]");
            self.state.emit_fmt(format_args!("    b {}", label_done));

            self.state.emit_fmt(format_args!("{}:", label_stack));
            self.state.emit("    ldr x3, [x1]");
            self.state.emit("    ldr x0, [x3]");
            self.state.emit("    add x3, x3, #8");
            self.state.emit("    str x3, [x1]");

            self.state.emit_fmt(format_args!("{}:", label_done));
        }

        if let Some(slot) = self.state.get_slot(dest.0) {
            self.emit_store_to_sp("x0", slot.0, "str");
        }
    }

    pub(super) fn emit_va_start_impl(&mut self, va_list_ptr: &Value) {
        if self.state.is_alloca(va_list_ptr.0) {
            if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
                self.emit_add_fp_offset("x0", slot.0);
            }
        } else if let Some(&reg) = self.reg_assignments.get(&va_list_ptr.0) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x0, {}", reg_name));
        } else if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
            self.emit_load_from_sp("x0", slot.0, "ldr");
        }

        let stack_offset = self.current_frame_size + self.va_named_stack_bytes as i64;
        if stack_offset <= 4095 {
            self.state.emit_fmt(format_args!("    add x1, x29, #{}", stack_offset));
        } else {
            self.load_large_imm("x1", stack_offset);
            self.state.emit("    add x1, x29, x1");
        }
        self.state.emit("    str x1, [x0]");

        let gr_top_offset = self.va_gp_save_offset + 64;
        self.emit_add_sp_offset("x1", gr_top_offset);
        self.state.emit("    str x1, [x0, #8]");

        if self.general_regs_only {
            self.state.emit("    str xzr, [x0, #16]");
        } else {
            let vr_top_offset = self.va_fp_save_offset + 128;
            self.emit_add_sp_offset("x1", vr_top_offset);
            self.state.emit("    str x1, [x0, #16]");
        }

        let gr_offs: i32 = -((8 - self.va_named_gp_count as i32) * 8);
        self.state.emit_fmt(format_args!("    mov w1, #{}", gr_offs));
        self.state.emit("    str w1, [x0, #24]");

        let vr_offs: i32 = if self.general_regs_only {
            0
        } else {
            -((8 - self.va_named_fp_count as i32) * 16)
        };
        self.state.emit_fmt(format_args!("    mov w1, #{}", vr_offs));
        self.state.emit("    str w1, [x0, #28]");
    }

    pub(super) fn emit_va_copy_impl(&mut self, dest_ptr: &Value, src_ptr: &Value) {
        if self.state.is_alloca(src_ptr.0) {
            if let Some(src_slot) = self.state.get_slot(src_ptr.0) {
                self.emit_add_fp_offset("x1", src_slot.0);
            }
        } else if let Some(&reg) = self.reg_assignments.get(&src_ptr.0) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x1, {}", reg_name));
        } else if let Some(src_slot) = self.state.get_slot(src_ptr.0) {
            self.emit_load_from_sp("x1", src_slot.0, "ldr");
        }
        if self.state.is_alloca(dest_ptr.0) {
            if let Some(dest_slot) = self.state.get_slot(dest_ptr.0) {
                self.emit_add_fp_offset("x0", dest_slot.0);
            }
        } else if let Some(&reg) = self.reg_assignments.get(&dest_ptr.0) {
            let reg_name = callee_saved_name(reg);
            self.state.emit_fmt(format_args!("    mov x0, {}", reg_name));
        } else if let Some(dest_slot) = self.state.get_slot(dest_ptr.0) {
            self.emit_load_from_sp("x0", dest_slot.0, "ldr");
        }
        self.state.emit("    ldp x2, x3, [x1]");
        self.state.emit("    stp x2, x3, [x0]");
        self.state.emit("    ldp x2, x3, [x1, #16]");
        self.state.emit("    stp x2, x3, [x0, #16]");
    }

    pub(super) fn emit_va_arg_struct_impl(&mut self, _dest_ptr: &Value, _va_list_ptr: &Value, _size: usize) {
        panic!("VaArgStruct should not be emitted for ARM64 target");
    }
}
