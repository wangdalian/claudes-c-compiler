//! I686Codegen: F128 negation.

use crate::ir::ir::{Operand, Value};
use crate::emit;
use super::codegen::I686Codegen;

impl I686Codegen {
    pub(super) fn emit_f128_neg_impl(&mut self, dest: &Value, src: &Operand) {
        self.emit_f128_load_to_x87(src);
        self.state.emit("    fchs");
        if let Some(slot) = self.state.get_slot(dest.0) {
            emit!(self.state, "    fstpt {}(%ebp)", slot.0);
            self.state.f128_direct_slots.insert(dest.0);
        }
        self.state.reg_cache.invalidate_acc();
    }
}
