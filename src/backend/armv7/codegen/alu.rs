//! ARMv7 integer ALU operations.

use crate::ir::reexports::{IrBinOp, Operand, Value};
use crate::common::types::IrType;
use crate::{emit};
use super::emit::{Armv7Codegen, armv7_alu_mnemonic};

impl Armv7Codegen {
    pub(super) fn emit_int_binop_impl(
        &mut self,
        dest: &Value,
        op: IrBinOp,
        lhs: &Operand,
        rhs: &Operand,
        ty: IrType,
    ) {
        match op {
            IrBinOp::Add | IrBinOp::Sub | IrBinOp::And | IrBinOp::Or | IrBinOp::Xor => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                let mnemonic = armv7_alu_mnemonic(op);
                emit!(self.state, "    {} r0, r2, r0", mnemonic);
            }
            IrBinOp::Mul => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    mul r0, r2, r0");
            }
            IrBinOp::SDiv => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    mov r1, r0");
                self.state.emit("    mov r0, r2");
                self.state.emit("    bl __aeabi_idiv");
            }
            IrBinOp::UDiv => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    mov r1, r0");
                self.state.emit("    mov r0, r2");
                self.state.emit("    bl __aeabi_uidiv");
            }
            IrBinOp::SRem => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    mov r1, r0");
                self.state.emit("    mov r0, r2");
                self.state.emit("    bl __aeabi_idivmod");
                self.state.emit("    mov r0, r1"); // remainder in r1
            }
            IrBinOp::URem => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    mov r1, r0");
                self.state.emit("    mov r0, r2");
                self.state.emit("    bl __aeabi_uidivmod");
                self.state.emit("    mov r0, r1"); // remainder in r1
            }
            IrBinOp::Shl => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    lsl r0, r2, r0");
            }
            IrBinOp::LShr => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    lsr r0, r2, r0");
            }
            IrBinOp::AShr => {
                self.operand_to_r0(lhs);
                self.state.emit("    mov r2, r0");
                self.operand_to_r0(rhs);
                self.state.emit("    asr r0, r2, r0");
            }
        }
        // Mask result to type width
        match ty.size() {
            1 => self.state.emit("    and r0, r0, #0xff"),
            2 => {
                self.load_imm32_to_reg("r1", 0xffff);
                self.state.emit("    and r0, r0, r1");
            }
            _ => {}
        }
        self.state.reg_cache.invalidate_all();
        self.store_r0_to(dest);
    }

    pub(super) fn emit_int_neg_impl(&mut self, _ty: IrType) {
        self.state.emit("    rsb r0, r0, #0");
    }

    pub(super) fn emit_int_not_impl(&mut self, ty: IrType) {
        self.state.emit("    mvn r0, r0");
        match ty.size() {
            1 => self.state.emit("    and r0, r0, #0xff"),
            2 => {
                self.load_imm32_to_reg("r1", 0xffff);
                self.state.emit("    and r0, r0, r1");
            }
            _ => {}
        }
    }

    pub(super) fn emit_int_clz_impl(&mut self, ty: IrType) {
        self.state.emit("    clz r0, r0");
        let type_bits = (ty.size() * 8) as i32;
        if type_bits < 32 {
            let adjust = 32 - type_bits;
            emit!(self.state, "    sub r0, r0, #{}", adjust);
        }
    }

    pub(super) fn emit_int_ctz_impl(&mut self, ty: IrType) {
        self.state.emit("    rbit r0, r0");
        self.state.emit("    clz r0, r0");
        let type_bits = ty.size() * 8;
        if type_bits < 32 {
            emit!(self.state, "    cmp r0, #{}", type_bits);
            emit!(self.state, "    movgt r0, #{}", type_bits);
        }
    }

    pub(super) fn emit_int_popcount_impl(&mut self, _ty: IrType) {
        self.state.emit("    bl __popcountsi2");
    }

    pub(super) fn emit_int_bswap_impl(&mut self, ty: IrType) {
        self.state.emit("    rev r0, r0");
        match ty.size() {
            2 => self.state.emit("    lsr r0, r0, #16"),
            1 => {} // No-op for byte
            _ => {}
        }
    }
}
