//! Overflow-checking arithmetic builtins for the lowerer.
//!
//! Handles __builtin_{s,u}{add,sub,mul}{,l,ll}_overflow and the generic
//! __builtin_{add,sub,mul}_overflow variants. These builtins perform an
//! arithmetic operation, store the result through a pointer, and return
//! 1 (bool true) if the operation overflowed.

use crate::frontend::parser::ast::*;
use crate::ir::ir::*;
use crate::common::types::{AddressSpace, IrType, CType, target_int_ir_type, target_is_32bit};
use super::lowering::Lowerer;

impl Lowerer {
    /// Lower __builtin_{add,sub,mul}_overflow(a, b, result_ptr) and type-specific variants.
    ///
    /// These builtins perform an arithmetic operation, store the result through
    /// the pointer, and return 1 (bool true) if the operation overflowed.
    pub(super) fn lower_overflow_builtin(&mut self, name: &str, args: &[Expr], op: IrBinOp) -> Option<Operand> {
        if args.len() < 3 {
            return Some(Operand::Const(IrConst::I64(0)));
        }

        // Determine the result type from the builtin name or from the pointer argument type.
        // Type-specific variants: __builtin_sadd_overflow (signed int), __builtin_uaddll_overflow (unsigned long long)
        // Generic variants: __builtin_add_overflow - type from pointer arg
        let (result_ir_ty, is_signed) = self.overflow_result_type(name, &args[2]);

        // Detect generic variants: __builtin_{add,sub,mul}_overflow
        // (as opposed to type-specific like __builtin_sadd_overflow, __builtin_uaddll_overflow)
        let is_generic = name == "__builtin_add_overflow"
            || name == "__builtin_sub_overflow"
            || name == "__builtin_mul_overflow";

        // Lower the two operands
        let lhs_raw = self.lower_expr(&args[0]);
        let rhs_raw = self.lower_expr(&args[1]);

        // For generic variants, operands may have narrower/different types than the
        // result. Cast each operand respecting its own signedness (sign-extend for
        // signed, zero-extend for unsigned) so that e.g. (int)-16 is correctly
        // represented in u128.
        let lhs_src_ctype = self.expr_ctype(&args[0]);
        let rhs_src_ctype = self.expr_ctype(&args[1]);
        let (lhs_val, rhs_val) = if is_generic {
            let lhs_src_ir = IrType::from_ctype(&lhs_src_ctype);
            let rhs_src_ir = IrType::from_ctype(&rhs_src_ctype);
            let lhs = if lhs_src_ir != result_ir_ty {
                Operand::Value(self.emit_cast_val(lhs_raw, lhs_src_ir, result_ir_ty))
            } else {
                lhs_raw
            };
            let rhs = if rhs_src_ir != result_ir_ty {
                Operand::Value(self.emit_cast_val(rhs_raw, rhs_src_ir, result_ir_ty))
            } else {
                rhs_raw
            };
            (lhs, rhs)
        } else {
            (lhs_raw, rhs_raw)
        };

        // Get the result pointer
        let result_ptr_val = self.lower_expr(&args[2]);
        let result_ptr = self.operand_to_value(result_ptr_val);

        // Perform the operation in the result type (operands are now properly widened)
        let result = self.emit_binop_val(op, lhs_val, rhs_val, result_ir_ty);

        // Store the (possibly truncated/wrapped) result
        self.emit(Instruction::Store { val: Operand::Value(result), ptr: result_ptr, ty: result_ir_ty,
         seg_override: AddressSpace::Default });

        // Compute the overflow flag.
        // For generic variants with signed source(s) and unsigned result type, we use
        // infinite-precision semantics: a negative mathematical result can't fit in an
        // unsigned type. We detect this by checking the sign bit of the result.
        let any_source_signed = is_generic
            && (lhs_src_ctype.is_signed() || rhs_src_ctype.is_signed());
        let overflow = if is_signed {
            self.compute_signed_overflow(op, lhs_val, rhs_val, result, result_ir_ty)
        } else if any_source_signed {
            // Generic variant with signed source(s) and unsigned result.
            // When all source types are narrower than the result type, both
            // operands fit in the signed variant of the result type, so the
            // operation in the unsigned result type is exact (no unsigned wrapping
            // can occur). Overflow only happens if the mathematical result is
            // negative (detected via sign bit).
            // When a source is as wide as the result type, unsigned wrapping CAN
            // occur, so we must also check for standard unsigned overflow.
            let lhs_src_ir = IrType::from_ctype(&lhs_src_ctype);
            let rhs_src_ir = IrType::from_ctype(&rhs_src_ctype);
            let sources_narrower = lhs_src_ir.size() < result_ir_ty.size()
                && rhs_src_ir.size() < result_ir_ty.size();
            if sources_narrower {
                // All sources narrower: only negative result can overflow unsigned
                self.check_result_sign_bit(result, result_ir_ty)
            } else {
                // At least one source is as wide as result: check both conditions
                let sign_ov = self.check_result_sign_bit(result, result_ir_ty);
                let unsigned_ov = self.compute_unsigned_overflow(
                    op, lhs_val, rhs_val, result, result_ir_ty,
                );
                self.emit_binop_val(
                    IrBinOp::Or,
                    Operand::Value(sign_ov),
                    Operand::Value(unsigned_ov),
                    result_ir_ty,
                )
            }
        } else {
            self.compute_unsigned_overflow(op, lhs_val, rhs_val, result, result_ir_ty)
        };

        Some(Operand::Value(overflow))
    }

    /// Determine the result IrType and signedness for an overflow builtin.
    fn overflow_result_type(&self, name: &str, result_ptr_expr: &Expr) -> (IrType, bool) {
        // Check type-specific variants first
        // __builtin_sadd_overflow -> signed int
        // __builtin_saddl_overflow -> signed long
        // __builtin_saddll_overflow -> signed long long
        // __builtin_uadd_overflow -> unsigned int
        // etc.
        if name.starts_with("__builtin_s") && name != "__builtin_sub_overflow" {
            // Signed type-specific variant
            let is_signed = true;
            let ty = if name.ends_with("ll_overflow") {
                IrType::I64
            } else if name.ends_with("l_overflow") {
                target_int_ir_type() // long: I64 on LP64, I32 on ILP32
            } else {
                IrType::I32
            };
            return (ty, is_signed);
        }
        if name.starts_with("__builtin_u") {
            // Unsigned type-specific variant
            let is_signed = false;
            let ty = if name.ends_with("ll_overflow") {
                IrType::U64
            } else if name.ends_with("l_overflow") {
                if target_is_32bit() { IrType::U32 } else { IrType::U64 } // unsigned long
            } else {
                IrType::U32
            };
            return (ty, is_signed);
        }

        // Generic variant: determine type from the pointer argument's pointee type
        let result_ctype = self.expr_ctype(result_ptr_expr);
        if let CType::Pointer(pointee, _) = &result_ctype {
            let is_signed = pointee.is_signed();
            let ir_ty = IrType::from_ctype(pointee);
            return (ir_ty, is_signed);
        }

        // If we can't determine the type (e.g., &local), try to get the
        // pointee from the expression structure
        if let Expr::AddressOf(inner, _) = result_ptr_expr {
            let inner_ctype = self.expr_ctype(inner);
            let is_signed = inner_ctype.is_signed();
            let ir_ty = IrType::from_ctype(&inner_ctype);
            return (ir_ty, is_signed);
        }

        // Default to signed long if we can't determine
        (target_int_ir_type(), true)
    }

    /// Compute signed overflow flag for add/sub/mul.
    fn compute_signed_overflow(
        &mut self,
        op: IrBinOp,
        lhs: Operand,
        rhs: Operand,
        result: Value,
        ty: IrType,
    ) -> Value {
        let bits = (ty.size() * 8) as i64;
        match op {
            IrBinOp::Add => {
                // Signed add overflow: ((result ^ lhs) & (result ^ rhs)) < 0
                // i.e., overflow if both operands have same sign and result has different sign
                let xor_lhs = self.emit_binop_val(IrBinOp::Xor, Operand::Value(result), lhs, ty);
                let xor_rhs = self.emit_binop_val(IrBinOp::Xor, Operand::Value(result), rhs, ty);
                let and_val = self.emit_binop_val(IrBinOp::And, Operand::Value(xor_lhs), Operand::Value(xor_rhs), ty);
                // Check sign bit: shift right by (bits-1) and check if non-zero
                let shifted = self.emit_binop_val(IrBinOp::AShr, Operand::Value(and_val), Operand::Const(IrConst::I64(bits - 1)), ty);
                // The shifted value is all-ones (-1) if overflow, all-zeros (0) if not
                // Convert to 0/1 by AND with 1
                self.emit_binop_val(IrBinOp::And, Operand::Value(shifted), Operand::Const(IrConst::I64(1)), ty)
            }
            IrBinOp::Sub => {
                // Signed sub overflow: ((lhs ^ rhs) & (result ^ lhs)) < 0
                // overflow if operands have different signs and result sign differs from lhs
                let xor_ops = self.emit_binop_val(IrBinOp::Xor, lhs, rhs, ty);
                let xor_res = self.emit_binop_val(IrBinOp::Xor, Operand::Value(result), lhs, ty);
                let and_val = self.emit_binop_val(IrBinOp::And, Operand::Value(xor_ops), Operand::Value(xor_res), ty);
                let shifted = self.emit_binop_val(IrBinOp::AShr, Operand::Value(and_val), Operand::Const(IrConst::I64(bits - 1)), ty);
                self.emit_binop_val(IrBinOp::And, Operand::Value(shifted), Operand::Const(IrConst::I64(1)), ty)
            }
            IrBinOp::Mul => {
                // For signed multiply overflow, we widen to double width, multiply, then
                // check if the result fits in the original width by checking that
                // sign-extending the truncated result gives back the full result.
                let wide_ty = Self::double_width_type(ty, true);
                let lhs_wide = self.emit_cast_val(lhs, ty, wide_ty);
                let rhs_wide = self.emit_cast_val(rhs, ty, wide_ty);
                let wide_result = self.emit_binop_val(IrBinOp::Mul, Operand::Value(lhs_wide), Operand::Value(rhs_wide), wide_ty);
                // Truncate back to original width
                let truncated = self.emit_cast_val(Operand::Value(wide_result), wide_ty, ty);
                // Sign-extend truncated back to wide type
                let sign_extended = self.emit_cast_val(Operand::Value(truncated), ty, wide_ty);
                // Overflow if wide_result != sign_extended
                self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(wide_result), Operand::Value(sign_extended), wide_ty)
            }
            _ => unreachable!("overflow only for add/sub/mul"),
        }
    }

    /// Compute unsigned overflow flag for add/sub/mul.
    fn compute_unsigned_overflow(
        &mut self,
        op: IrBinOp,
        lhs: Operand,
        rhs: Operand,
        result: Value,
        ty: IrType,
    ) -> Value {
        // Make sure we're comparing in unsigned type
        let unsigned_ty = ty.to_unsigned();
        match op {
            IrBinOp::Add => {
                // Unsigned add overflow: result < lhs (wrapping means the sum is smaller)
                self.emit_cmp_val(IrCmpOp::Ult, Operand::Value(result), lhs, unsigned_ty)
            }
            IrBinOp::Sub => {
                // Unsigned sub overflow: lhs < rhs (borrow occurred)
                self.emit_cmp_val(IrCmpOp::Ult, lhs, rhs, unsigned_ty)
            }
            IrBinOp::Mul => {
                // Unsigned multiply overflow: widen to double width, multiply, check upper half is zero
                let wide_ty = Self::double_width_type(ty, false);
                let lhs_wide = self.emit_cast_val(lhs, unsigned_ty, wide_ty);
                let rhs_wide = self.emit_cast_val(rhs, unsigned_ty, wide_ty);
                let wide_result = self.emit_binop_val(IrBinOp::Mul, Operand::Value(lhs_wide), Operand::Value(rhs_wide), wide_ty);
                // Truncate back to original width
                let truncated = self.emit_cast_val(Operand::Value(wide_result), wide_ty, unsigned_ty);
                // Zero-extend truncated back to wide type
                let zero_extended = self.emit_cast_val(Operand::Value(truncated), unsigned_ty, wide_ty);
                // Overflow if wide_result != zero_extended
                self.emit_cmp_val(IrCmpOp::Ne, Operand::Value(wide_result), Operand::Value(zero_extended), wide_ty)
            }
            _ => unreachable!("overflow only for add/sub/mul"),
        }
    }

    /// Check if the sign bit of `result` is set when interpreted in `ty`.
    /// Returns a value that is non-zero (1) if the sign bit is set, 0 otherwise.
    /// Used to detect overflow when a signed source produces a negative result
    /// that is being stored into an unsigned result type.
    fn check_result_sign_bit(&mut self, result: Value, ty: IrType) -> Value {
        let bits = (ty.size() * 8) as i64;
        // Arithmetic shift right by (bits-1): all-ones if negative, all-zeros if positive
        let shifted = self.emit_binop_val(
            IrBinOp::AShr,
            Operand::Value(result),
            Operand::Const(IrConst::I64(bits - 1)),
            ty,
        );
        // AND with 1 to get 0 or 1
        self.emit_binop_val(
            IrBinOp::And,
            Operand::Value(shifted),
            Operand::Const(IrConst::I64(1)),
            ty,
        )
    }

    /// Get the double-width integer type for overflow multiply detection.
    fn double_width_type(ty: IrType, signed: bool) -> IrType {
        match ty {
            IrType::I8 | IrType::U8 => if signed { IrType::I16 } else { IrType::U16 },
            IrType::I16 | IrType::U16 => if signed { IrType::I32 } else { IrType::U32 },
            IrType::I32 | IrType::U32 => if signed { IrType::I64 } else { IrType::U64 },
            IrType::I64 | IrType::U64 => if signed { IrType::I128 } else { IrType::U128 },
            // For 128-bit, we can't widen further; use a different strategy
            // (but this is extremely rare in practice)
            _ => if signed { IrType::I128 } else { IrType::U128 },
        }
    }
}
