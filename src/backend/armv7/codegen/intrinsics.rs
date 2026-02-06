//! ARMv7 intrinsic operations.

use crate::ir::reexports::{IntrinsicOp, Operand, Value};
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_intrinsic_impl(
        &mut self,
        dest: &Option<Value>,
        op: &IntrinsicOp,
        _dest_ptr: &Option<Value>,
        args: &[Operand],
    ) {
        match op {
            // Memory fences
            IntrinsicOp::Lfence | IntrinsicOp::Mfence | IntrinsicOp::Sfence => {
                self.state.emit("    dmb ish");
            }
            IntrinsicOp::Pause => {
                self.state.emit("    yield");
            }
            IntrinsicOp::Clflush => {
                // No direct equivalent on ARM; use data memory barrier
                if !args.is_empty() {
                    self.operand_to_r0(&args[0]);
                    self.state.emit("    @ clflush not available on ARM");
                }
            }
            IntrinsicOp::ReturnAddress => {
                if let Some(dest) = dest {
                    self.state.emit("    mov r0, lr");
                    self.store_r0_to(dest);
                }
            }
            IntrinsicOp::FrameAddress => {
                if let Some(dest) = dest {
                    self.state.emit("    mov r0, r11");
                    self.store_r0_to(dest);
                }
            }
            IntrinsicOp::ThreadPointer => {
                if let Some(dest) = dest {
                    // ARM thread pointer is in CP15 c13
                    self.state.emit("    mrc p15, 0, r0, c13, c0, 3");
                    self.store_r0_to(dest);
                }
            }
            IntrinsicOp::SqrtF32 => {
                if let Some(dest) = dest {
                    if !args.is_empty() {
                        self.operand_to_r0(&args[0]);
                        self.state.emit("    vmov s0, r0");
                        self.state.emit("    vsqrt.f32 s0, s0");
                        self.state.emit("    vmov r0, s0");
                        self.store_r0_to(dest);
                    }
                }
            }
            IntrinsicOp::SqrtF64 => {
                if let Some(dest) = dest {
                    if !args.is_empty() {
                        self.load_wide_to_r0_r1(&args[0]);
                        self.state.emit("    vmov d0, r0, r1");
                        self.state.emit("    vsqrt.f64 d0, d0");
                        self.state.emit("    vmov r0, r1, d0");
                        self.store_r0_r1_to(dest);
                    }
                }
            }
            IntrinsicOp::FabsF32 => {
                if let Some(dest) = dest {
                    if !args.is_empty() {
                        self.operand_to_r0(&args[0]);
                        self.state.emit("    vmov s0, r0");
                        self.state.emit("    vabs.f32 s0, s0");
                        self.state.emit("    vmov r0, s0");
                        self.store_r0_to(dest);
                    }
                }
            }
            IntrinsicOp::FabsF64 => {
                if let Some(dest) = dest {
                    if !args.is_empty() {
                        self.load_wide_to_r0_r1(&args[0]);
                        self.state.emit("    vmov d0, r0, r1");
                        self.state.emit("    vabs.f64 d0, d0");
                        self.state.emit("    vmov r0, r1, d0");
                        self.store_r0_r1_to(dest);
                    }
                }
            }
            _ => {
                // Unsupported intrinsic, emit a comment
                emit!(self.state, "    @ unsupported intrinsic: {:?}", op);
                if let Some(dest) = dest {
                    self.state.emit("    mov r0, #0");
                    self.store_r0_to(dest);
                }
            }
        }
    }
}
