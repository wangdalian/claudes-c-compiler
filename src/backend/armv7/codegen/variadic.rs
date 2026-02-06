//! ARMv7 variadic function support (va_start, va_arg, va_copy).
//!
//! On ARMv7 AAPCS, va_list is a simple char* pointer to the stack area
//! where variadic arguments are stored. This is much simpler than the
//! AArch64 implementation which uses a complex struct.

use crate::ir::reexports::Value;
use crate::common::types::IrType;
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    /// Emit prologue code to save register args for variadic functions.
    pub(super) fn emit_variadic_prologue(&mut self, func: &crate::ir::reexports::IrFunction) {
        // Save remaining register args (r0-r3) that weren't used by named params
        // to the stack, just above the frame pointer.
        // Named args already consumed some registers in emit_store_params.
        // We need to save the rest so va_arg can find them.
        let named_reg_count = func.params.len().min(4);
        if named_reg_count < 4 {
            let remaining = 4 - named_reg_count;
            // Save remaining args above fp
            for i in 0..remaining {
                let reg_idx = named_reg_count + i;
                let param_regs = ["r0", "r1", "r2", "r3"];
                let offset = 8 + reg_idx * 4; // Above saved fp+lr
                emit!(self.state, "    str {}, [r11, #{}]", param_regs[reg_idx], offset);
            }
        }
        self.va_named_reg_count = named_reg_count;
    }

    pub(super) fn emit_va_start_impl(&mut self, ap: &Value) {
        // va_list = pointer to first variadic arg on the stack
        // After named args in registers, variadic args start at [fp + 8 + named_regs*4]
        let offset = 8 + self.va_named_reg_count * 4;
        if offset <= 255 {
            emit!(self.state, "    add r0, r11, #{}", offset);
        } else {
            self.load_imm32_to_reg("r0", offset as u32);
            self.state.emit("    add r0, r11, r0");
        }
        // Store the pointer to the va_list variable
        self.load_ptr_to_reg("r1", ap);
        self.state.emit("    str r0, [r1]");
    }

    pub(super) fn emit_va_arg_impl(&mut self, dest: &Value, ap: &Value, ty: IrType) {
        let size = ty.size().max(4); // Minimum 4 bytes (word-aligned)
        let align = if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) { 8 } else { 4 };

        // Load current va_list pointer
        self.load_ptr_to_reg("r1", ap);
        self.state.emit("    ldr r2, [r1]");

        // Align pointer if needed (for 64-bit types)
        if align > 4 {
            // r2 = (r2 + 7) & ~7
            self.state.emit("    add r2, r2, #7");
            self.state.emit("    bic r2, r2, #7");
        }

        // Load value from current position
        if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) {
            self.state.emit("    ldr r0, [r2]");
            self.state.emit("    ldr r3, [r2, #4]");
            self.store_r0_to(dest);
            // Store upper half
            if let Some(slot) = self.state.get_slot(dest.0) {
                let hi_offset = slot.0 + 4;
                emit!(self.state, "    str r3, [r11, #{}]", hi_offset);
            }
        } else {
            match ty.size() {
                1 => {
                    if ty.is_signed() {
                        self.state.emit("    ldrsb r0, [r2]");
                    } else {
                        self.state.emit("    ldrb r0, [r2]");
                    }
                }
                2 => {
                    if ty.is_signed() {
                        self.state.emit("    ldrsh r0, [r2]");
                    } else {
                        self.state.emit("    ldrh r0, [r2]");
                    }
                }
                _ => {
                    self.state.emit("    ldr r0, [r2]");
                }
            }
            self.store_r0_to(dest);
        }

        // Advance pointer — r0 is not clobbered, cache stays valid
        emit!(self.state, "    add r2, r2, #{}", size);
        // Store updated pointer back
        self.state.emit("    str r2, [r1]");
    }

    pub(super) fn emit_va_copy_impl(&mut self, dest: &Value, src: &Value) {
        // va_copy: just copy the pointer value
        self.load_ptr_to_reg("r1", src);
        self.state.emit("    ldr r0, [r1]");
        self.load_ptr_to_reg("r1", dest);
        self.state.emit("    str r0, [r1]");
    }
}
