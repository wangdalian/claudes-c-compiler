//! ARMv7 atomic operations using LDREX/STREX.

use crate::ir::reexports::{AtomicOrdering, AtomicRmwOp, Operand, Value};
use crate::common::types::IrType;
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    fn emit_barrier(&mut self, ordering: AtomicOrdering) {
        match ordering {
            AtomicOrdering::Relaxed => {}
            _ => self.state.emit("    dmb ish"),
        }
    }

    /// Get the correct LDREX variant for the given type size.
    fn ldrex_for_size(size: usize) -> &'static str {
        match size {
            1 => "ldrexb",
            2 => "ldrexh",
            _ => "ldrex",
        }
    }

    /// Get the correct STREX variant for the given type size.
    fn strex_for_size(size: usize) -> &'static str {
        match size {
            1 => "strexb",
            2 => "strexh",
            _ => "strex",
        }
    }

    pub(super) fn emit_atomic_load_impl(
        &mut self, dest: &Value, ptr: &Operand, ty: IrType, ordering: AtomicOrdering,
    ) {
        self.operand_to_r0(ptr);
        self.state.emit("    mov r1, r0");
        self.emit_barrier(ordering);
        match ty.size() {
            1 => self.state.emit("    ldrb r0, [r1]"),
            2 => self.state.emit("    ldrh r0, [r1]"),
            4 => self.state.emit("    ldr r0, [r1]"),
            _ => self.state.emit("    ldr r0, [r1]"),
        }
        self.emit_barrier(ordering);
        self.store_r0_to(dest);
    }

    pub(super) fn emit_atomic_store_impl(
        &mut self, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering,
    ) {
        self.operand_to_r0(val);
        self.state.emit("    mov r2, r0");
        self.operand_to_r0(ptr);
        self.state.emit("    mov r1, r0");
        self.emit_barrier(ordering);
        match ty.size() {
            1 => self.state.emit("    strb r2, [r1]"),
            2 => self.state.emit("    strh r2, [r1]"),
            4 => self.state.emit("    str r2, [r1]"),
            _ => self.state.emit("    str r2, [r1]"),
        }
        self.emit_barrier(ordering);
    }

    pub(super) fn emit_atomic_rmw_impl(
        &mut self, dest: &Value, op: AtomicRmwOp, ptr: &Operand, val: &Operand,
        ty: IrType, ordering: AtomicOrdering,
    ) {
        self.operand_to_r0(val);
        self.state.emit("    mov r3, r0"); // r3 = value to apply
        self.operand_to_r0(ptr);
        self.state.emit("    mov r1, r0"); // r1 = pointer

        let size = ty.size();
        let ldrex = Self::ldrex_for_size(size);
        let strex = Self::strex_for_size(size);

        let label = self.state.next_label_id();
        emit!(self.state, ".Latomic_rmw_{}:", label);
        emit!(self.state, "    {} r0, [r1]", ldrex);  // r0 = old value
        self.state.emit("    mov r2, r0");              // r2 = old value (preserve)

        match op {
            AtomicRmwOp::Add => self.state.emit("    add r0, r0, r3"),
            AtomicRmwOp::Sub => self.state.emit("    sub r0, r0, r3"),
            AtomicRmwOp::And => self.state.emit("    and r0, r0, r3"),
            AtomicRmwOp::Or  => self.state.emit("    orr r0, r0, r3"),
            AtomicRmwOp::Xor => self.state.emit("    eor r0, r0, r3"),
            AtomicRmwOp::Xchg => self.state.emit("    mov r0, r3"),
            AtomicRmwOp::Nand => {
                self.state.emit("    and r0, r0, r3");
                self.state.emit("    mvn r0, r0");
            }
            AtomicRmwOp::TestAndSet => {
                // test_and_set: *ptr = 1, return old value
                self.state.emit("    mov r0, #1");
            }
        }
        emit!(self.state, "    {} r12, r0, [r1]", strex);
        self.state.emit("    cmp r12, #0");
        emit!(self.state, "    bne .Latomic_rmw_{}", label);
        self.emit_barrier(ordering);

        // Return old value
        self.state.emit("    mov r0, r2");
        self.store_r0_to(dest);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_atomic_cmpxchg_impl(
        &mut self, dest: &Value, ptr: &Operand, expected: &Operand, desired: &Operand,
        ty: IrType, _success_ordering: AtomicOrdering, _failure_ordering: AtomicOrdering,
        _returns_bool: bool,
    ) {
        self.operand_to_r0(expected);
        self.state.emit("    mov r2, r0"); // r2 = expected
        self.operand_to_r0(desired);
        self.state.emit("    mov r3, r0"); // r3 = desired
        self.operand_to_r0(ptr);
        self.state.emit("    mov r1, r0"); // r1 = pointer

        let size = ty.size();
        let ldrex = Self::ldrex_for_size(size);
        let strex = Self::strex_for_size(size);

        let label = self.state.next_label_id();
        let done_label = self.state.next_label_id();

        emit!(self.state, ".Lcmpxchg_{}:", label);
        emit!(self.state, "    {} r0, [r1]", ldrex);
        self.state.emit("    cmp r0, r2");
        emit!(self.state, "    bne .Lcmpxchg_fail_{}", done_label);
        emit!(self.state, "    {} r12, r3, [r1]", strex);
        self.state.emit("    cmp r12, #0");
        emit!(self.state, "    bne .Lcmpxchg_{}", label);
        self.state.emit("    dmb ish");
        // Success: return expected value
        self.state.emit("    mov r0, r2");
        emit!(self.state, "    b .Lcmpxchg_done_{}", done_label);

        emit!(self.state, ".Lcmpxchg_fail_{}:", done_label);
        self.state.emit("    clrex");
        // Failure: return loaded value
        // r0 already has the loaded value

        emit!(self.state, ".Lcmpxchg_done_{}:", done_label);
        self.store_r0_to(dest);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_fence_impl(&mut self, ordering: AtomicOrdering) {
        match ordering {
            AtomicOrdering::Relaxed => {}
            _ => self.state.emit("    dmb ish"),
        }
    }
}
