//! ARMv7 floating-point operations using VFPv3.

use crate::backend::cast::FloatOp;
use crate::ir::reexports::{Operand, Value};
use crate::common::types::IrType;
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    /// Float binary operation - full implementation with dest, op, lhs, rhs, ty.
    /// This is used by the emit_binop override when it detects a float type.
    pub(super) fn emit_float_binop_full(
        &mut self,
        dest: &Value,
        op: FloatOp,
        lhs: &Operand,
        rhs: &Operand,
        ty: IrType,
    ) {
        if ty == IrType::F64 {
            // Load lhs to d0
            self.load_wide_to_r0_r1(lhs);
            self.state.emit("    vmov d0, r0, r1");
            // Load rhs to d1
            self.load_wide_to_r0_r1(rhs);
            self.state.emit("    vmov d1, r0, r1");
            // Perform operation
            let mnemonic = match op {
                FloatOp::Add => "vadd.f64",
                FloatOp::Sub => "vsub.f64",
                FloatOp::Mul => "vmul.f64",
                FloatOp::Div => "vdiv.f64",
            };
            emit!(self.state, "    {} d0, d0, d1", mnemonic);
            // Store result
            self.state.emit("    vmov r0, r1, d0");
            self.store_r0_r1_to(dest);
        } else {
            // F32
            self.operand_to_r0(lhs);
            self.state.emit("    vmov s0, r0");
            self.operand_to_r0(rhs);
            self.state.emit("    vmov s1, r0");
            let mnemonic = match op {
                FloatOp::Add => "vadd.f32",
                FloatOp::Sub => "vsub.f32",
                FloatOp::Mul => "vmul.f32",
                FloatOp::Div => "vdiv.f32",
            };
            emit!(self.state, "    {} s0, s0, s1", mnemonic);
            self.state.emit("    vmov r0, s0");
            self.store_r0_to(dest);
        }
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_float_neg_impl(&mut self, ty: IrType) {
        if ty == IrType::F64 {
            // Negate by flipping sign bit in upper word
            self.state.emit("    eor r1, r1, #0x80000000");
        } else {
            // F32: flip sign bit
            self.state.emit("    eor r0, r0, #0x80000000");
        }
    }

    pub(super) fn emit_f128_cmp_impl(
        &mut self,
        dest: &Value,
        op: crate::ir::reexports::IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
    ) {
        // F128 comparison via soft-float library call
        // __aeabi_dcmpeq, etc. - delegate to double for now (long double = double on ARM32)
        self.emit_float_cmp_impl(dest, op, lhs, rhs, IrType::F64);
    }

    pub(super) fn emit_f128_neg_impl(&mut self, dest: &Value, src: &Operand) {
        // F128 = double on ARM32, just flip sign bit
        self.load_wide_to_r0_r1(src);
        self.state.emit("    eor r1, r1, #0x80000000");
        self.store_r0_r1_to(dest);
    }
}
