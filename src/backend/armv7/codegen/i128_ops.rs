//! ARMv7 64-bit integer operations (I64/U64 are "wide" types on 32-bit).
//! On ARMv7, 64-bit values use register pairs (r0:r1 = lo:hi).
//! The i128_binop/cmp implementations here handle 64-bit on ARM32.

use crate::ir::reexports::{IrBinOp, IrCmpOp, Operand, Value};
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_i128_binop_impl(
        &mut self,
        dest: &Value,
        op: IrBinOp,
        lhs: &Operand,
        rhs: &Operand,
    ) {
        // Load lhs to r0:r1
        self.load_wide_to_r0_r1(lhs);
        self.state.emit("    push {r0, r1}"); // Save lhs
        // Load rhs to r0:r1
        self.load_wide_to_r0_r1(rhs);
        self.state.emit("    mov r2, r0"); // r2:r3 = rhs
        self.state.emit("    mov r3, r1");
        self.state.emit("    pop {r0, r1}"); // r0:r1 = lhs

        match op {
            IrBinOp::Add => {
                self.state.emit("    adds r0, r0, r2");
                self.state.emit("    adc r1, r1, r3");
            }
            IrBinOp::Sub => {
                self.state.emit("    subs r0, r0, r2");
                self.state.emit("    sbc r1, r1, r3");
            }
            IrBinOp::And => {
                self.state.emit("    and r0, r0, r2");
                self.state.emit("    and r1, r1, r3");
            }
            IrBinOp::Or => {
                self.state.emit("    orr r0, r0, r2");
                self.state.emit("    orr r1, r1, r3");
            }
            IrBinOp::Xor => {
                self.state.emit("    eor r0, r0, r2");
                self.state.emit("    eor r1, r1, r3");
            }
            IrBinOp::Mul => {
                // 64-bit multiply: result_lo = lo*lo, result_hi = hi*lo + lo*hi + carry
                self.state.emit("    push {r4, r5}");
                self.state.emit("    umull r4, r5, r0, r2"); // r4:r5 = lo * lo
                self.state.emit("    mla r5, r0, r3, r5");    // r5 += lo * rhs_hi
                self.state.emit("    mla r5, r1, r2, r5");    // r5 += lhs_hi * lo
                self.state.emit("    mov r0, r4");
                self.state.emit("    mov r1, r5");
                self.state.emit("    pop {r4, r5}");
            }
            IrBinOp::SDiv => {
                // Use runtime library (signed)
                self.state.emit("    bl __aeabi_ldivmod");
                // Result in r0:r1
            }
            IrBinOp::UDiv => {
                // Use runtime library (unsigned)
                self.state.emit("    bl __aeabi_uldivmod");
                // Result in r0:r1
            }
            IrBinOp::SRem => {
                self.state.emit("    bl __aeabi_ldivmod");
                self.state.emit("    mov r0, r2"); // Remainder in r2:r3
                self.state.emit("    mov r1, r3");
            }
            IrBinOp::URem => {
                self.state.emit("    bl __aeabi_uldivmod");
                self.state.emit("    mov r0, r2"); // Remainder in r2:r3
                self.state.emit("    mov r1, r3");
            }
            IrBinOp::Shl => {
                // Shift left 64-bit by r2 bits (only low word of shift amount)
                self.state.emit("    bl __aeabi_llsl");
            }
            IrBinOp::LShr => {
                self.state.emit("    bl __aeabi_llsr");
            }
            IrBinOp::AShr => {
                self.state.emit("    bl __aeabi_lasr");
            }
        }

        self.store_r0_r1_to(dest);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_i128_cmp_impl(
        &mut self,
        dest: &Value,
        op: IrCmpOp,
        lhs: &Operand,
        rhs: &Operand,
    ) {
        // Load lhs to r0:r1, rhs to r2:r3
        self.load_wide_to_r0_r1(lhs);
        self.state.emit("    push {r0, r1}");
        self.load_wide_to_r0_r1(rhs);
        self.state.emit("    mov r2, r0");
        self.state.emit("    mov r3, r1");
        self.state.emit("    pop {r0, r1}");

        // Compare: first compare high words, then low words
        self.state.emit("    cmp r1, r3"); // Compare high words
        self.state.emit("    cmpeq r0, r2"); // If equal, compare low words

        let cc = super::emit::armv7_int_cond_code(op);
        let inv_cc = super::emit::armv7_invert_cond_code(cc);
        emit!(self.state, "    mov{} r0, #1", cc);
        emit!(self.state, "    mov{} r0, #0", inv_cc);

        self.state.reg_cache.invalidate_all();
        self.store_r0_to(dest);
    }
}
