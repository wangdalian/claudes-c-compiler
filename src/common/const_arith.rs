/// Shared constant-expression arithmetic helpers.
///
/// Used by both `sema::const_eval` and `ir::lowering::const_eval` for
/// compile-time integer arithmetic with proper C width/signedness semantics.

/// Wrap an i64 result to 32-bit width if `is_32bit` is true, otherwise return as-is.
/// This handles the C semantics of truncating arithmetic results to `int` width.
#[inline]
pub fn wrap_result(v: i64, is_32bit: bool) -> i64 {
    if is_32bit { v as i32 as i64 } else { v }
}

/// Perform an unsigned binary operation, handling 32-bit vs 64-bit width.
/// Converts operands to the appropriate unsigned type, applies the operation,
/// and sign-extends the result back to i64.
#[inline]
pub fn unsigned_op(l: i64, r: i64, is_32bit: bool, op: fn(u64, u64) -> u64) -> i64 {
    if is_32bit {
        op(l as u32 as u64, r as u32 as u64) as u32 as i64
    } else {
        op(l as u64, r as u64) as i64
    }
}

/// Convert a boolean to i64 (1 for true, 0 for false).
#[inline]
pub fn bool_to_i64(b: bool) -> i64 {
    if b { 1 } else { 0 }
}
