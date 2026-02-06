//! ARMv7 global address and TLS handling.

use crate::ir::reexports::Value;
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_global_addr_impl(&mut self, dest: &Value, name: &str) {
        if self.state.pic_mode {
            // PIC: load via GOT
            emit!(self.state, "    ldr r0, .LGOT_{}", name);
            emit!(self.state, ".LPIC_{}:", name);
            self.state.emit("    add r0, pc, r0");
            self.state.emit("    ldr r0, [r0]");
        } else {
            // Non-PIC: load absolute address using movw/movt
            emit!(self.state, "    movw r0, #:lower16:{}", name);
            emit!(self.state, "    movt r0, #:upper16:{}", name);
        }
        self.store_r0_to(dest);
    }

    pub(super) fn emit_tls_global_addr_impl(&mut self, dest: &Value, name: &str) {
        // TLS access via __tls_get_addr (General Dynamic model)
        emit!(self.state, "    ldr r0, .LTLSGD_{}", name);
        self.state.emit("    bl __tls_get_addr");
        self.store_r0_to(dest);
    }
}
