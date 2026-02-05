Fix i686 codegen README.md

Issues to fix:
1. ALU file description incorrectly says F64 negation uses "sign-bit XOR" -
   F64 negation actually uses x87 fchs (handled before emit_float_neg is called)
2. CallAbiConfig pseudo-code shows wrong/missing fields vs actual struct
3. Minor cleanup for consistency
