//! ARMv7 memory operations (load/store).
//!
//! 64-bit types (I64, U64, F64) need special handling on ARM32 because
//! the accumulator pair (r0:r1) holds two 32-bit halves.

use crate::ir::reexports::{Operand, Value};
use crate::common::types::IrType;
use crate::backend::state::{StackSlot, SlotAddr};
use crate::backend::traits::ArchCodegen;
use crate::emit;
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_store_impl(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) {
            let addr = self.state.resolve_slot_addr(ptr.0);
            // Load both halves of the 64-bit value
            self.load_wide_to_r0_r1(val);
            if let Some(addr) = addr {
                match addr {
                    SlotAddr::OverAligned(slot, id) => {
                        // Save r0:r1, compute address, then store
                        self.state.emit("    push {r0, r1}");
                        self.emit_alloca_aligned_addr(slot, id);
                        // r12 = aligned address
                        self.state.emit("    pop {r0, r1}");
                        self.state.emit("    str r0, [r12]");
                        self.state.emit("    str r1, [r12, #4]");
                    }
                    SlotAddr::Direct(slot) => {
                        let slot_ref = self.slot_ref(slot);
                        emit!(self.state, "    str r0, {}", slot_ref);
                        let hi_offset = slot.0 + 4;
                        if hi_offset == 0 {
                            self.state.emit("    str r1, [r11]");
                        } else {
                            emit!(self.state, "    str r1, [r11, #{}]", hi_offset);
                        }
                    }
                    SlotAddr::Indirect(slot) => {
                        // Save r0:r1, load pointer, then store through it
                        self.state.emit("    push {r0, r1}");
                        self.emit_load_ptr_from_slot(slot, ptr.0);
                        // r12 = pointer
                        self.state.emit("    pop {r0, r1}");
                        self.state.emit("    str r0, [r12]");
                        self.state.emit("    str r1, [r12, #4]");
                    }
                }
            }
            self.state.reg_cache.invalidate_all();
            return;
        }
        crate::backend::traits::emit_store_default(self, val, ptr, ty);
    }

    pub(super) fn emit_load_impl(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) {
            let addr = self.state.resolve_slot_addr(ptr.0);
            if let Some(addr) = addr {
                match addr {
                    SlotAddr::OverAligned(slot, id) => {
                        self.emit_alloca_aligned_addr(slot, id);
                        // r12 = aligned address
                        self.state.emit("    ldr r0, [r12]");
                        self.state.emit("    ldr r1, [r12, #4]");
                    }
                    SlotAddr::Direct(slot) => {
                        let slot_ref = self.slot_ref(slot);
                        emit!(self.state, "    ldr r0, {}", slot_ref);
                        let hi_offset = slot.0 + 4;
                        if hi_offset == 0 {
                            self.state.emit("    ldr r1, [r11]");
                        } else {
                            emit!(self.state, "    ldr r1, [r11, #{}]", hi_offset);
                        }
                    }
                    SlotAddr::Indirect(slot) => {
                        self.emit_load_ptr_from_slot(slot, ptr.0);
                        // r12 = pointer
                        self.state.emit("    ldr r0, [r12]");
                        self.state.emit("    ldr r1, [r12, #4]");
                    }
                }
                self.store_r0_r1_to(dest);
            }
            self.state.reg_cache.invalidate_all();
            return;
        }
        crate::backend::traits::emit_load_default(self, dest, ptr, ty);
    }
}
