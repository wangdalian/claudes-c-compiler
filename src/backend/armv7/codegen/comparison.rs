//! ARMv7 integer and float comparison operations.

use crate::ir::reexports::{IrCmpOp, Operand, Value};
use crate::common::types::IrType;
use crate::{emit};
use super::emit::{Armv7Codegen, armv7_int_cond_code, armv7_float_cond_code};
use crate::backend::generation::is_i128_type;

impl Armv7Codegen {
    pub(super) fn emit_int_cmp_impl(
        &mut self,
        dest: &Value,
        op: IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
        _ty: IrType,
    ) {
        self.operand_to_r0(lhs);
        self.state.emit("    mov r2, r0");
        self.operand_to_r0(rhs);
        self.state.emit("    cmp r2, r0");
        let cc = armv7_int_cond_code(op);
        emit!(self.state, "    mov{} r0, #1", cc);
        let inv_cc = super::emit::armv7_invert_cond_code(cc);
        emit!(self.state, "    mov{} r0, #0", inv_cc);
        self.state.reg_cache.invalidate_all();
        self.store_r0_to(dest);
    }

    pub(super) fn emit_float_cmp_impl(
        &mut self,
        dest: &Value,
        op: IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
        ty: IrType,
    ) {
        self.emit_float_cmp_insn(lhs, rhs, ty);
        let cc = armv7_float_cond_code(op);
        emit!(self.state, "    mov{} r0, #1", cc);
        let inv_cc = super::emit::armv7_invert_cond_code(cc);
        emit!(self.state, "    mov{} r0, #0", inv_cc);
        self.state.reg_cache.invalidate_all();
        self.store_r0_to(dest);
    }

    /// Emit the comparison instructions for floats (vcmp + vmrs), setting APSR flags.
    /// This is shared between emit_float_cmp_impl and emit_fused_cmp_branch_impl.
    fn emit_float_cmp_insn(&mut self, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if ty == IrType::F64 {
            self.load_wide_to_r0_r1(lhs);
            self.state.emit("    vmov d0, r0, r1");
            self.load_wide_to_r0_r1(rhs);
            self.state.emit("    vmov d1, r0, r1");
            self.state.emit("    vcmp.f64 d0, d1");
        } else {
            self.operand_to_r0(lhs);
            self.state.emit("    vmov s0, r0");
            self.operand_to_r0(rhs);
            self.state.emit("    vmov s1, r0");
            self.state.emit("    vcmp.f32 s0, s1");
        }
        self.state.emit("    vmrs APSR_nzcv, FPSCR");
    }

    /// Emit the comparison instructions for 64-bit integers, setting APSR flags.
    /// Uses cmp high, cmpeq low pattern.
    fn emit_i64_cmp_insn(&mut self, lhs: &Operand, rhs: &Operand) {
        self.load_wide_to_r0_r1(lhs);
        self.state.emit("    push {r0, r1}");
        self.load_wide_to_r0_r1(rhs);
        self.state.emit("    mov r2, r0");
        self.state.emit("    mov r3, r1");
        self.state.emit("    pop {r0, r1}");
        // Compare high words first, then low words if equal
        self.state.emit("    cmp r1, r3");
        self.state.emit("    cmpeq r0, r2");
    }

    /// Fused compare-and-branch: emit comparison + conditional branch directly,
    /// without storing boolean result to a stack slot.
    pub(super) fn emit_fused_cmp_branch_impl(
        &mut self,
        op: IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
        ty: IrType,
        true_label: &str,
        false_label: &str,
    ) {
        if ty == IrType::F128 || ty == IrType::F64 || ty == IrType::F32 {
            // Float comparison (F128 = F64 on ARM32)
            let cmp_ty = if ty == IrType::F128 { IrType::F64 } else { ty };
            self.emit_float_cmp_insn(lhs, rhs, cmp_ty);
            let cc = armv7_float_cond_code(op);
            emit!(self.state, "    b{} {}", cc, true_label);
            emit!(self.state, "    b {}", false_label);
        } else if matches!(ty, IrType::I64 | IrType::U64) || is_i128_type(ty) {
            // 64-bit integer comparison
            self.emit_i64_cmp_insn(lhs, rhs);
            let cc = armv7_int_cond_code(op);
            emit!(self.state, "    b{} {}", cc, true_label);
            emit!(self.state, "    b {}", false_label);
        } else {
            // 32-bit integer comparison
            self.operand_to_r0(lhs);
            self.state.emit("    mov r2, r0");
            self.operand_to_r0(rhs);
            self.state.emit("    cmp r2, r0");
            let cc = armv7_int_cond_code(op);
            emit!(self.state, "    b{} {}", cc, true_label);
            emit!(self.state, "    b {}", false_label);
        }
        self.state.reg_cache.invalidate_all();
    }
}
