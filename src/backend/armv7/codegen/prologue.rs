//! ARMv7 function prologue and epilogue generation.
//!
//! Stack frame layout (growing downward):
//! ```text
//! High addresses (caller's frame)
//!   +----------------------------------+
//!   | Caller's stack arguments         |
//!   +----------------------------------+
//!   | Saved lr                         |  [fp + 4]
//!   | Saved fp (r11)                   |  [fp + 0] = r11
//!   +----------------------------------+
//!   | Callee-saved registers           |
//!   | (r4-r10 as needed, via PUSH)     |
//!   +----------------------------------+
//!   | Local variables / spill slots    |
//!   +----------------------------------+  <-- sp
//! Low addresses
//! ```

use crate::ir::reexports::{IrFunction, Value};
use crate::common::types::IrType;
use crate::backend::state::StackSlot;
use crate::backend::generation::{
    calculate_stack_space_common, run_regalloc_and_merge_clobbers,
    find_param_alloca,
};
use crate::backend::call_abi::{ParamClass, classify_params};
use crate::{emit};
use super::emit::{Armv7Codegen, ARMV7_CALLEE_SAVED, ARMV7_CALLER_SAVED, phys_reg_name};
use crate::backend::traits::ArchCodegen;

impl Armv7Codegen {
    pub(super) fn calculate_stack_space_impl(&mut self, func: &IrFunction) -> i64 {
        // Run register allocation
        let available_regs = ARMV7_CALLEE_SAVED.to_vec();
        let caller_saved_regs = ARMV7_CALLER_SAVED.to_vec();

        let (reg_assigned, cached_liveness) = run_regalloc_and_merge_clobbers(
            func, available_regs, caller_saved_regs, &[],
            &mut self.reg_assignments, &mut self.used_callee_saved,
            false,
        );

        // Check if function is variadic
        self.is_variadic = func.is_variadic;

        let callee_saved_bytes = self.used_callee_saved.len() as i64 * 4;

        // Calculate stack layout using common function
        calculate_stack_space_common(&mut self.state, func, callee_saved_bytes, |space, alloc_size, align| {
            let effective_align = if align > 0 { align.max(4) } else { 4 };
            let alloc = (alloc_size + 3) & !3; // 4-byte align allocations
            let new_space = space + alloc;
            // ARM requires 8-byte stack alignment
            let offset = -(new_space);
            (new_space, offset)
        }, &reg_assigned, &self.used_callee_saved.clone(), cached_liveness, true)
    }

    pub(super) fn emit_prologue_impl(&mut self, func: &IrFunction, frame_size: i64) {
        let aligned_frame = self.aligned_frame_size_impl(frame_size);

        // Push frame pointer and link register
        self.state.emit("    push {r11, lr}");
        if self.state.emit_cfi {
            self.state.emit("    .cfi_def_cfa_offset 8");
            self.state.emit("    .cfi_offset lr, -4");
            self.state.emit("    .cfi_offset r11, -8");
        }
        self.state.emit("    mov r11, sp");
        if self.state.emit_cfi {
            self.state.emit("    .cfi_def_cfa_register r11");
        }

        // Push callee-saved registers
        if !self.used_callee_saved.is_empty() {
            let regs: Vec<&str> = self.used_callee_saved.iter()
                .map(|r| phys_reg_name(*r))
                .collect();
            emit!(self.state, "    push {{{}}}", regs.join(", "));
        }

        // Allocate stack space for locals
        if aligned_frame > 0 {
            if aligned_frame <= 255 {
                emit!(self.state, "    sub sp, sp, #{}", aligned_frame);
            } else {
                self.load_imm32_to_reg("r12", aligned_frame as u32);
                self.state.emit("    sub sp, sp, r12");
            }
        }

        // Store variadic register args if needed
        if self.is_variadic {
            self.emit_variadic_prologue(func);
        }

        // Store parameters
        self.emit_store_params_impl(func);

        // Set return type
        self.current_return_type = func.return_type;
    }

    pub(super) fn emit_epilogue_impl(&mut self, frame_size: i64) {
        let aligned_frame = self.aligned_frame_size_impl(frame_size);

        // Deallocate stack space
        if aligned_frame > 0 {
            self.state.emit("    mov sp, r11");
        }

        // Restore callee-saved registers and return
        if !self.used_callee_saved.is_empty() {
            // We need to subtract the callee-saved space from sp first
            let callee_saved_bytes = self.used_callee_saved.len() * 4;
            emit!(self.state, "    sub sp, r11, #{}", callee_saved_bytes);
            let regs: Vec<&str> = self.used_callee_saved.iter()
                .map(|r| phys_reg_name(*r))
                .collect();
            emit!(self.state, "    pop {{{}}}", regs.join(", "));
        }

        self.state.emit("    pop {r11, lr}");
        self.state.emit("    bx lr");
    }

    pub(super) fn emit_store_params_impl(&mut self, func: &IrFunction) {
        // AAPCS: first 4 integer args in r0-r3, rest on stack
        let param_regs = ["r0", "r1", "r2", "r3"];
        let mut reg_idx = 0;

        // Build param alloca slots for param_ref optimization
        self.state.param_alloca_slots = (0..func.params.len()).map(|i| {
            find_param_alloca(func, i).and_then(|(dest, ty)| {
                self.state.get_slot(dest.0).map(|slot| (slot, ty))
            })
        }).collect();

        self.state.num_params = func.params.len();
        self.state.func_is_variadic = func.is_variadic;

        for (i, param) in func.params.iter().enumerate() {
            let ty = param.ty;
            let size = ty.size();

            // Find the alloca slot for this parameter
            let alloca = find_param_alloca(func, i);
            let val_id = alloca.map(|(v, _)| v.0);

            if let Some(val_id) = val_id {
                if let Some(slot) = self.state.get_slot(val_id) {
                    if reg_idx < 4 {
                        if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) {
                            // 64-bit: needs two registers (or one reg + stack)
                            // Align to even register
                            if reg_idx % 2 != 0 { reg_idx += 1; }
                            if reg_idx + 1 < 4 {
                                let slot_ref = self.slot_ref(slot);
                                emit!(self.state, "    str {}, {}", param_regs[reg_idx], slot_ref);
                                let hi_offset = slot.0 + 4;
                                emit!(self.state, "    str {}, [r11, #{}]", param_regs[reg_idx + 1], hi_offset);
                                reg_idx += 2;
                            } else {
                                // Partially in register, partially on stack
                                if reg_idx < 4 {
                                    let slot_ref = self.slot_ref(slot);
                                    emit!(self.state, "    str {}, {}", param_regs[reg_idx], slot_ref);
                                    reg_idx += 1;
                                }
                                // High word is on caller's stack
                                let stack_offset = 8 + (i.saturating_sub(4)) * 4;
                                let hi_offset = slot.0 + 4;
                                emit!(self.state, "    ldr r12, [r11, #{}]", stack_offset);
                                emit!(self.state, "    str r12, [r11, #{}]", hi_offset);
                            }
                        } else {
                            // 32-bit or smaller: one register
                            let slot_ref = self.slot_ref(slot);
                            emit!(self.state, "    str {}, {}", param_regs[reg_idx], slot_ref);
                            reg_idx += 1;
                        }
                    } else {
                        // Parameter is on the caller's stack
                        let stack_offset = 8 + (reg_idx.saturating_sub(4)) * 4;
                        let slot_ref = self.slot_ref(slot);
                        emit!(self.state, "    ldr r12, [r11, #{}]", stack_offset);
                        emit!(self.state, "    str r12, {}", slot_ref);
                        if size > 4 {
                            let hi_offset = slot.0 + 4;
                            emit!(self.state, "    ldr r12, [r11, #{}]", stack_offset + 4);
                            emit!(self.state, "    str r12, [r11, #{}]", hi_offset);
                        }
                    }
                } else if let Some(preg) = self.reg_assignments.get(&val_id) {
                    // Value is in a callee-saved register
                    if reg_idx < 4 {
                        let dest_name = phys_reg_name(*preg);
                        emit!(self.state, "    mov {}, {}", dest_name, param_regs[reg_idx]);
                        reg_idx += 1;
                    }
                } else {
                    if reg_idx < 4 { reg_idx += 1; }
                }
            } else {
                if reg_idx < 4 { reg_idx += 1; }
            }
        }

        self.va_named_reg_count = reg_idx.min(4);
        self.va_named_stack_bytes = if reg_idx > 4 { (reg_idx - 4) * 4 } else { 0 };
    }

    pub(super) fn emit_param_ref_impl(&mut self, dest: &Value, param_idx: usize, ty: IrType) {
        // Check if param was stored to an alloca slot
        if param_idx < self.state.param_alloca_slots.len() {
            if let Some((alloca_slot, _alloca_ty)) = self.state.param_alloca_slots[param_idx] {
                if let Some(dest_slot) = self.state.get_slot(dest.0) {
                    if dest_slot.0 == alloca_slot.0 {
                        // The param value is already in the alloca slot
                        if let Some(phys) = self.reg_assignments.get(&dest.0).copied() {
                            let reg = phys_reg_name(phys);
                            let slot_ref = self.slot_ref(alloca_slot);
                            emit!(self.state, "    ldr r0, {}", slot_ref);
                            emit!(self.state, "    mov {}, r0", reg);
                            self.state.reg_cache.invalidate_acc();
                        }
                        return;
                    }
                }
            }
        }

        let param_regs = ["r0", "r1", "r2", "r3"];
        if param_idx < 4 {
            emit!(self.state, "    mov r0, {}", param_regs[param_idx]);
        } else {
            let stack_offset = 8 + (param_idx - 4) * 4;
            emit!(self.state, "    ldr r0, [r11, #{}]", stack_offset);
        }
        self.store_r0_to(dest);
    }

    pub(super) fn emit_alloca_aligned_addr_impl(&mut self, slot: StackSlot, _val_id: u32) {
        // Load the runtime-aligned pointer from the slot into r12 (scratch)
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    ldr r12, {}", slot_ref);
    }

    pub(super) fn emit_alloca_aligned_addr_to_acc_impl(&mut self, slot: StackSlot, _val_id: u32) {
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    ldr r0, {}", slot_ref);
    }
}
