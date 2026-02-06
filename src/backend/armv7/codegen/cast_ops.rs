//! ARMv7 type conversion (cast) operations.

use crate::ir::reexports::{Operand, Value};
use crate::common::types::IrType;
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_cast_impl(
        &mut self,
        dest: &Value,
        src: &Operand,
        from_ty: IrType,
        to_ty: IrType,
    ) {
        if from_ty == to_ty {
            self.operand_to_r0(src);
            self.store_r0_to(dest);
            return;
        }

        // Handle wide (64-bit) source types
        let from_wide = matches!(from_ty, IrType::I64 | IrType::U64 | IrType::F64);
        let to_wide = matches!(to_ty, IrType::I64 | IrType::U64 | IrType::F64);

        // Float-to-float conversions
        if from_ty == IrType::F32 && to_ty == IrType::F64 {
            self.operand_to_r0(src);
            self.state.emit("    vmov s0, r0");
            self.state.emit("    vcvt.f64.f32 d0, s0");
            self.state.emit("    vmov r0, r1, d0");
            self.store_r0_r1_to(dest);
            return;
        }
        if from_ty == IrType::F64 && to_ty == IrType::F32 {
            self.load_wide_to_r0_r1(src);
            self.state.emit("    vmov d0, r0, r1");
            self.state.emit("    vcvt.f32.f64 s0, d0");
            self.state.emit("    vmov r0, s0");
            self.store_r0_to(dest);
            return;
        }

        // Int-to-float conversions
        if !from_ty.is_float() && to_ty == IrType::F32 {
            let from_wide = matches!(from_ty, IrType::I64 | IrType::U64);
            if from_wide {
                // I64/U64→F32: VFP can't convert from 64-bit int, use library call
                self.load_wide_to_r0_r1(src);
                if from_ty.is_signed() {
                    self.state.emit("    bl __aeabi_l2f");
                } else {
                    self.state.emit("    bl __aeabi_ul2f");
                }
                self.state.reg_cache.invalidate_all();
                self.store_r0_to(dest);
            } else {
                // I32 or smaller → F32: use VFP vcvt
                self.operand_to_r0(src);
                if from_ty.size() < 4 && from_ty.is_signed() {
                    match from_ty.size() {
                        1 => self.state.emit("    sxtb r0, r0"),
                        2 => self.state.emit("    sxth r0, r0"),
                        _ => {}
                    }
                }
                self.state.emit("    vmov s0, r0");
                if from_ty.is_signed() || from_ty.size() < 4 {
                    self.state.emit("    vcvt.f32.s32 s0, s0");
                } else {
                    self.state.emit("    vcvt.f32.u32 s0, s0");
                }
                self.state.emit("    vmov r0, s0");
                self.store_r0_to(dest);
            }
            return;
        }
        if !from_ty.is_float() && to_ty == IrType::F64 {
            let from_wide = matches!(from_ty, IrType::I64 | IrType::U64);
            if from_wide {
                // I64/U64→F64: VFP can't convert from 64-bit int, use library call
                self.load_wide_to_r0_r1(src);
                if from_ty.is_signed() {
                    self.state.emit("    bl __aeabi_l2d");
                } else {
                    self.state.emit("    bl __aeabi_ul2d");
                }
                self.state.reg_cache.invalidate_all();
                self.store_r0_r1_to(dest);
            } else {
                // I32 or smaller → F64: use VFP vcvt
                self.operand_to_r0(src);
                if from_ty.size() < 4 && from_ty.is_signed() {
                    match from_ty.size() {
                        1 => self.state.emit("    sxtb r0, r0"),
                        2 => self.state.emit("    sxth r0, r0"),
                        _ => {}
                    }
                }
                self.state.emit("    vmov s0, r0");
                if from_ty.is_signed() || from_ty.size() < 4 {
                    self.state.emit("    vcvt.f64.s32 d0, s0");
                } else {
                    self.state.emit("    vcvt.f64.u32 d0, s0");
                }
                self.state.emit("    vmov r0, r1, d0");
                self.store_r0_r1_to(dest);
            }
            return;
        }

        // Float-to-int conversions
        if from_ty == IrType::F32 && !to_ty.is_float() {
            let to_wide = matches!(to_ty, IrType::I64 | IrType::U64);
            if to_wide {
                // F32→I64/U64: VFP can't convert to 64-bit int, use library call
                self.operand_to_r0(src);
                if to_ty.is_signed() {
                    self.state.emit("    bl __aeabi_f2lz");
                } else {
                    self.state.emit("    bl __aeabi_f2ulz");
                }
                self.state.reg_cache.invalidate_all();
                self.store_r0_r1_to(dest);
            } else {
                // F32→I32 or smaller: use VFP vcvt
                self.operand_to_r0(src);
                self.state.emit("    vmov s0, r0");
                if to_ty.is_signed() {
                    self.state.emit("    vcvt.s32.f32 s0, s0");
                } else {
                    self.state.emit("    vcvt.u32.f32 s0, s0");
                }
                self.state.emit("    vmov r0, s0");
                self.store_r0_to(dest);
            }
            return;
        }
        if from_ty == IrType::F64 && !to_ty.is_float() {
            let to_wide = matches!(to_ty, IrType::I64 | IrType::U64);
            if to_wide {
                // F64→I64/U64: VFP can't convert to 64-bit int, use library call
                self.load_wide_to_r0_r1(src);
                if to_ty.is_signed() {
                    self.state.emit("    bl __aeabi_d2lz");
                } else {
                    self.state.emit("    bl __aeabi_d2ulz");
                }
                self.state.reg_cache.invalidate_all();
                self.store_r0_r1_to(dest);
            } else {
                // F64→I32 or smaller: use VFP vcvt
                self.load_wide_to_r0_r1(src);
                self.state.emit("    vmov d0, r0, r1");
                if to_ty.is_signed() {
                    self.state.emit("    vcvt.s32.f64 s0, d0");
                } else {
                    self.state.emit("    vcvt.u32.f64 s0, d0");
                }
                self.state.emit("    vmov r0, s0");
                self.store_r0_to(dest);
            }
            return;
        }

        // F128 casts (long double = double on ARM32)
        if from_ty == IrType::F128 || to_ty == IrType::F128 {
            // F128 is treated as F64 on ARM32
            let actual_from = if from_ty == IrType::F128 { IrType::F64 } else { from_ty };
            let actual_to = if to_ty == IrType::F128 { IrType::F64 } else { to_ty };
            self.emit_cast_impl(dest, src, actual_from, actual_to);
            return;
        }

        // Integer-to-integer conversions (widening/narrowing)
        if from_wide && !to_wide {
            // 64-bit to 32-bit: just take lower half
            self.operand_to_r0(src);
            // Truncate to target width
            match to_ty.size() {
                1 => self.state.emit("    and r0, r0, #0xff"),
                2 => {
                    self.load_imm32_to_reg("r1", 0xffff);
                    self.state.emit("    and r0, r0, r1");
                }
                _ => {}
            }
            self.store_r0_to(dest);
            return;
        }
        if !from_wide && to_wide {
            // 32-bit to 64-bit: sign/zero extend
            self.operand_to_r0(src);
            if from_ty.is_signed() {
                // Sign extend to 32-bit first
                match from_ty.size() {
                    1 => self.state.emit("    sxtb r0, r0"),
                    2 => self.state.emit("    sxth r0, r0"),
                    _ => {}
                }
                self.state.emit("    asr r1, r0, #31");
            } else {
                match from_ty.size() {
                    1 => self.state.emit("    and r0, r0, #0xff"),
                    2 => {
                        self.load_imm32_to_reg("r1", 0xffff);
                        self.state.emit("    and r0, r0, r1");
                    }
                    _ => {}
                }
                self.state.emit("    mov r1, #0");
            }
            self.store_r0_r1_to(dest);
            return;
        }
        if from_wide && to_wide {
            // 64-to-64: just copy
            self.load_wide_to_r0_r1(src);
            self.store_r0_r1_to(dest);
            return;
        }

        // Narrow-to-narrow integer conversion
        self.operand_to_r0(src);
        let from_size = from_ty.size();
        let to_size = to_ty.size();

        if to_size < from_size {
            // Truncate
            match to_size {
                1 => self.state.emit("    and r0, r0, #0xff"),
                2 => {
                    self.load_imm32_to_reg("r1", 0xffff);
                    self.state.emit("    and r0, r0, r1");
                }
                _ => {}
            }
        } else if to_size > from_size {
            // Extend
            if from_ty.is_signed() {
                match from_size {
                    1 => self.state.emit("    sxtb r0, r0"),
                    2 => self.state.emit("    sxth r0, r0"),
                    _ => {}
                }
            } else {
                match from_size {
                    1 => self.state.emit("    uxtb r0, r0"),
                    2 => self.state.emit("    uxth r0, r0"),
                    _ => {}
                }
            }
        }
        self.store_r0_to(dest);
    }
}
