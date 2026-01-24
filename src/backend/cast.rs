//! Shared cast and float operation classification.
//!
//! All three backends use the same decision logic to determine what kind of cast
//! to emit — only the actual machine instructions differ. By classifying the cast
//! once in shared code, we eliminate duplicated Ptr-normalization and F128-reduction
//! logic from each backend.

use crate::common::types::IrType;
use crate::ir::ir::IrBinOp;

/// Classification of type casts. All three backends use the same control flow
/// to decide which kind of cast to emit; only the actual instructions differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// No conversion needed (same type, or Ptr <-> I64/U64, or F128 <-> F64).
    Noop,
    /// Float to signed integer (from_ty is F32 or F64).
    FloatToSigned { from_f64: bool },
    /// Float to unsigned integer (from_ty is F32 or F64).
    FloatToUnsigned { from_f64: bool, to_u64: bool },
    /// Signed integer to float (to_ty is F32 or F64).
    SignedToFloat { to_f64: bool },
    /// Unsigned integer to float. `from_u64` indicates U64 source needing
    /// special overflow handling on x86.
    UnsignedToFloat { to_f64: bool, from_u64: bool },
    /// Float-to-float conversion (F32 <-> F64).
    FloatToFloat { widen: bool },
    /// Integer widening: sign- or zero-extend a smaller type to a larger one.
    IntWiden { from_ty: IrType, to_ty: IrType },
    /// Integer narrowing: truncate a larger type to a smaller one.
    IntNarrow { to_ty: IrType },
    /// Same-size signed-to-unsigned (need to mask/clear upper bits).
    SignedToUnsignedSameSize { to_ty: IrType },
}

/// Classify a cast between two IR types. This captures the shared decision logic
/// that all three backends use identically. Backends then match on the returned
/// `CastKind` to emit architecture-specific instructions.
///
/// Handles Ptr normalization (Ptr treated as U64) and F128 reduction (F128 treated
/// as F64 for computation purposes) before classification.
pub fn classify_cast(from_ty: IrType, to_ty: IrType) -> CastKind {
    if from_ty == to_ty {
        return CastKind::Noop;
    }

    // F128 (long double) is computed as F64. Treat F128 <-> F64 as noop,
    // and F128 <-> other as F64 <-> other.
    if from_ty == IrType::F128 || to_ty == IrType::F128 {
        let effective_from = if from_ty == IrType::F128 { IrType::F64 } else { from_ty };
        let effective_to = if to_ty == IrType::F128 { IrType::F64 } else { to_ty };
        if effective_from == effective_to {
            return CastKind::Noop;
        }
        return classify_cast(effective_from, effective_to);
    }

    // Ptr is equivalent to U64 on all 64-bit targets.
    if (from_ty == IrType::Ptr || to_ty == IrType::Ptr) && !from_ty.is_float() && !to_ty.is_float() {
        let effective_from = if from_ty == IrType::Ptr { IrType::U64 } else { from_ty };
        let effective_to = if to_ty == IrType::Ptr { IrType::U64 } else { to_ty };
        if effective_from == effective_to || (effective_from.size() == 8 && effective_to.size() == 8) {
            return CastKind::Noop;
        }
        return classify_cast(effective_from, effective_to);
    }

    // Float-to-int
    if from_ty.is_float() && !to_ty.is_float() {
        let is_unsigned_dest = to_ty.is_unsigned() || to_ty == IrType::Ptr;
        let from_f64 = from_ty == IrType::F64;
        if is_unsigned_dest {
            let to_u64 = to_ty == IrType::U64 || to_ty == IrType::Ptr;
            return CastKind::FloatToUnsigned { from_f64, to_u64 };
        } else {
            return CastKind::FloatToSigned { from_f64 };
        }
    }

    // Int-to-float
    if !from_ty.is_float() && to_ty.is_float() {
        let is_unsigned_src = from_ty.is_unsigned();
        let to_f64 = to_ty == IrType::F64;
        if is_unsigned_src {
            let from_u64 = from_ty == IrType::U64;
            return CastKind::UnsignedToFloat { to_f64, from_u64 };
        } else {
            return CastKind::SignedToFloat { to_f64 };
        }
    }

    // Float-to-float
    if from_ty.is_float() && to_ty.is_float() {
        let widen = from_ty == IrType::F32 && to_ty == IrType::F64;
        return CastKind::FloatToFloat { widen };
    }

    // Integer-to-integer
    let from_size = from_ty.size();
    let to_size = to_ty.size();

    if from_size == to_size {
        if from_ty.is_signed() && to_ty.is_unsigned() {
            return CastKind::SignedToUnsignedSameSize { to_ty };
        }
        return CastKind::Noop;
    }

    if to_size > from_size {
        return CastKind::IntWiden { from_ty, to_ty };
    }

    CastKind::IntNarrow { to_ty }
}

/// Float arithmetic operations that all three backends support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Classify a binary operation on floats. Returns None if the operation is not
/// meaningful on floats (e.g., bitwise And, Or, Xor, shifts, integer remainder).
pub fn classify_float_binop(op: IrBinOp) -> Option<FloatOp> {
    match op {
        IrBinOp::Add => Some(FloatOp::Add),
        IrBinOp::Sub => Some(FloatOp::Sub),
        IrBinOp::Mul => Some(FloatOp::Mul),
        IrBinOp::SDiv | IrBinOp::UDiv => Some(FloatOp::Div),
        _ => None,
    }
}
