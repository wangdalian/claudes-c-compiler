Task: Fix int128 unary ops and increment/decrement truncating to 64 bits
Status: Complete
Branch: master

Problem:
Three related bugs caused __int128 operations to silently truncate results
to 64 bits, producing wrong values for large 128-bit integers:

1. Unary negation (-) and bitwise NOT (~) of int128 values used IrType::I64
   for the operation type, making the backend emit 64-bit negation/NOT
   instead of 128-bit. This caused -(int128)x to lose the high 64 bits.

2. Increment (++) and decrement (--) of int128 values used IrConst::I64(1)
   as the step and IrType::I64 as the operation type, causing the add/sub
   to operate on only 64 bits.

3. ConstHashKey::to_hash_key() truncated I128 constants to i64, causing
   GVN (Global Value Numbering) to potentially merge distinct I128 constants
   that share the same low 64 bits.

These bugs caused PostgreSQL's numeric test suite to fail (138/216 tests)
because the sqrt_var() function's Karatsuba algorithm uses int128 arithmetic
with decrement (s_int128--) in its correction step. The truncated decrement
produced wrong sqrt results, which cascaded to wrong ln/log/exp results.

Fix:
- expr.rs: Use ty (the actual expression type) instead of IrType::I64 for
  128-bit unary neg and bitwise NOT operations
- expr.rs: Return IrConst::I128(1) with the correct 128-bit type for
  int128 increment/decrement step values
- ir.rs: Add I128(i128) variant to ConstHashKey enum and use it for
  I128 constants instead of truncating to I64

Testing:
- New regression test: int128-unary-neg-wide (covers neg, NOT, inc, dec)
- PostgreSQL numeric test now passes (was failing 138/216)
- All 100 unit tests pass
- ~98.2% compiler test suite pass rate (no regressions)
