//! ARMv7 return value handling.

use crate::ir::reexports::Operand;
use crate::common::types::IrType;
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_return_impl(&mut self, val: Option<&Operand>, frame_size: i64) {
        if let Some(val) = val {
            let ret_ty = self.current_return_type;
            if matches!(ret_ty, IrType::I64 | IrType::U64 | IrType::F64 | IrType::F128) {
                self.load_wide_to_r0_r1(val);
            } else if ret_ty != IrType::Void {
                self.operand_to_r0(val);
            }
        }
        self.emit_epilogue_and_ret_impl(frame_size);
    }
}
