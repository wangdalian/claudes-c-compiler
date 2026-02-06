//! ARMv7 AAPCS calling convention implementation.

use crate::backend::call_abi::{CallAbiConfig, CallArgClass};
use crate::ir::reexports::{Operand, Value, IrConst};
use crate::common::types::IrType;
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn call_abi_config_impl(&self) -> CallAbiConfig {
        CallAbiConfig {
            max_int_regs: 4,
            max_float_regs: 0, // ARMv7 hard-float uses r0-r3 for args in our simplified model
            align_i128_pairs: true,
            f128_in_fp_regs: false,
            f128_in_gp_pairs: false,
            variadic_floats_in_gp: true,
            large_struct_by_ref: false,
            use_sysv_struct_classification: false,
            use_riscv_float_struct_classification: false,
            allow_struct_split_reg_stack: false,
            align_struct_pairs: false,
            sret_uses_dedicated_reg: false,
        }
    }

    pub(super) fn emit_call_compute_stack_space_impl(
        &self,
        arg_classes: &[CallArgClass],
        arg_types: &[IrType],
    ) -> usize {
        let mut stack_bytes = 0usize;
        for (i, class) in arg_classes.iter().enumerate() {
            if let CallArgClass::Stack = class {
                let ty = if i < arg_types.len() { arg_types[i] } else { IrType::I32 };
                if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) {
                    stack_bytes += 8;
                } else {
                    stack_bytes += 4;
                }
            }
        }
        // Align to 8 bytes
        (stack_bytes + 7) & !7
    }

    pub(super) fn emit_call_stack_args_impl(
        &mut self,
        args: &[Operand],
        arg_classes: &[CallArgClass],
        arg_types: &[IrType],
        stack_arg_space: usize,
        fptr_spill_size: usize,
        f128_temp_space: usize,
    ) -> i64 {
        // fptr was already spilled via pre-decrement push in emit_call_spill_fptr,
        // so only allocate stack_arg_space here.
        if stack_arg_space > 0 {
            if stack_arg_space <= 255 {
                emit!(self.state, "    sub sp, sp, #{}", stack_arg_space);
            } else {
                self.load_imm32_to_reg("r12", stack_arg_space as u32);
                self.state.emit("    sub sp, sp, r12");
            }
        }

        // Store stack arguments
        let mut stack_offset = 0usize;
        for (i, class) in arg_classes.iter().enumerate() {
            if let CallArgClass::Stack = class {
                let ty = if i < arg_types.len() { arg_types[i] } else { IrType::I32 };
                if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) {
                    // Wide type: load both halves first, then store
                    self.load_wide_to_r0_r1(&args[i]);
                    if stack_offset == 0 {
                        self.state.emit("    str r0, [sp]");
                    } else {
                        emit!(self.state, "    str r0, [sp, #{}]", stack_offset);
                    }
                    emit!(self.state, "    str r1, [sp, #{}]", stack_offset + 4);
                    stack_offset += 8;
                } else {
                    self.operand_to_r0(&args[i]);
                    if stack_offset == 0 {
                        self.state.emit("    str r0, [sp]");
                    } else {
                        emit!(self.state, "    str r0, [sp, #{}]", stack_offset);
                    }
                    stack_offset += 4;
                }
            }
        }
        // Return total SP adjustment including fptr spill (already pushed)
        (stack_arg_space + fptr_spill_size + f128_temp_space) as i64
    }

    pub(super) fn emit_call_reg_args_impl(
        &mut self,
        args: &[Operand],
        arg_classes: &[CallArgClass],
        arg_types: &[IrType],
        _total_sp_adjust: i64,
        _f128_temp_space: usize,
        _stack_arg_space: usize,
        _struct_arg_riscv_float_classes: &[Option<crate::common::types::RiscvFloatClass>],
    ) {
        let reg_names = ["r0", "r1", "r2", "r3"];

        // Collect register assignments first to avoid clobbering
        let mut reg_args: Vec<(usize, usize)> = Vec::new(); // (arg_idx, reg_idx)
        for (i, class) in arg_classes.iter().enumerate() {
            if let CallArgClass::IntReg { reg_idx, .. } = class {
                reg_args.push((i, *reg_idx));
            }
        }

        // Load in reverse order to avoid clobbering r0
        for &(arg_idx, reg_idx) in reg_args.iter().rev() {
            let ty = if arg_idx < arg_types.len() { arg_types[arg_idx] } else { IrType::I32 };
            if reg_idx < 4 {
                match &args[arg_idx] {
                    Operand::Const(c) => {
                        match c {
                            IrConst::I32(v) => {
                                self.load_imm32_to_reg(reg_names[reg_idx], *v as u32);
                            }
                            IrConst::I64(v) => {
                                self.load_imm32_to_reg(reg_names[reg_idx], *v as u32);
                                if reg_idx + 1 < 4 {
                                    self.load_imm32_to_reg(reg_names[reg_idx + 1], (*v >> 32) as u32);
                                }
                            }
                            IrConst::Zero => {
                                emit!(self.state, "    mov {}, #0", reg_names[reg_idx]);
                            }
                            _ => {
                                self.load_const_to_r0(c);
                                if reg_idx != 0 {
                                    emit!(self.state, "    mov {}, r0", reg_names[reg_idx]);
                                }
                            }
                        }
                    }
                    Operand::Value(v) => {
                        if let Some(preg) = self.reg_assignments.get(&v.0) {
                            let src = super::emit::phys_reg_name(*preg);
                            emit!(self.state, "    mov {}, {}", reg_names[reg_idx], src);
                        } else if let Some(slot) = self.state.get_slot(v.0) {
                            let slot_ref = self.slot_ref(slot);
                            emit!(self.state, "    ldr {}, {}", reg_names[reg_idx], slot_ref);
                            // 64-bit: load upper half too
                            if matches!(ty, IrType::I64 | IrType::U64 | IrType::F64) && reg_idx + 1 < 4 {
                                let hi_offset = slot.0 + 4;
                                emit!(self.state, "    ldr {}, [r11, #{}]", reg_names[reg_idx + 1], hi_offset);
                            }
                        } else {
                            // Value is in the accumulator (immediately-consumed)
                            if reg_idx != 0 {
                                emit!(self.state, "    mov {}, r0", reg_names[reg_idx]);
                            }
                        }
                    }
                }
            }
        }
    }

    pub(super) fn emit_call_instruction_impl(
        &mut self,
        name: Option<&str>,
        func_ptr: Option<&Operand>,
        indirect: bool,
        stack_arg_space: usize,
    ) {
        if let Some(name) = name {
            emit!(self.state, "    bl {}", name);
        } else if indirect {
            // Load function pointer from spill slot (above stack args on the stack)
            let spill_offset = stack_arg_space;
            if spill_offset == 0 {
                self.state.emit("    ldr r12, [sp]");
            } else if spill_offset <= 4095 {
                emit!(self.state, "    ldr r12, [sp, #{}]", spill_offset);
            } else {
                self.load_imm32_to_reg("r12", spill_offset as u32);
                self.state.emit("    add r12, sp, r12");
                self.state.emit("    ldr r12, [r12]");
            }
            self.state.emit("    blx r12");
        } else if let Some(fptr) = func_ptr {
            self.operand_to_r0(fptr);
            self.state.emit("    blx r0");
        }
    }

    pub(super) fn emit_call_cleanup_impl(
        &mut self,
        stack_arg_space: usize,
        f128_temp_space: usize,
        indirect: bool,
    ) {
        let fptr_spill = if indirect { 8usize } else { 0 };
        let total = stack_arg_space + f128_temp_space + fptr_spill;
        if total > 0 {
            if total <= 255 {
                emit!(self.state, "    add sp, sp, #{}", total);
            } else {
                self.load_imm32_to_reg("r12", total as u32);
                self.state.emit("    add sp, sp, r12");
            }
        }
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_call_store_result_impl(&mut self, dest: &Value, return_type: IrType) {
        if matches!(return_type, IrType::I64 | IrType::U64 | IrType::F64) {
            self.store_r0_r1_to(dest);
        } else if return_type != IrType::Void {
            self.store_r0_to(dest);
        }
    }

    pub(super) fn emit_call_spill_fptr_impl(&mut self, func_ptr: &Operand) {
        self.operand_to_r0(func_ptr);
        // Push fptr onto stack with pre-decrement (8-byte aligned for AAPCS)
        self.state.emit("    str r0, [sp, #-8]!");
    }

    pub(super) fn emit_call_fptr_spill_size_impl(&self) -> usize {
        8
    }

    pub(super) fn emit_call_f128_pre_convert_impl(
        &mut self,
        _args: &[Operand],
        _arg_classes: &[CallArgClass],
        _arg_types: &[IrType],
        _stack_arg_space: usize,
    ) -> usize {
        0 // F128 not used on ARMv7
    }
}
