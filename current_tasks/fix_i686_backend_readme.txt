Thoroughly verify and rewrite the i686 backend README (src/backend/i686/README.md).
The current README was written as part of a batch docs rewrite and has never been
individually verified against the actual source code. Checking every claim against
emit.rs, prologue.rs, calls.rs, alu.rs, comparison.rs, memory.rs, i128_ops.rs,
casts.rs, returns.rs, float_ops.rs, globals.rs, variadic.rs, atomics.rs,
intrinsics.rs, inline_asm.rs, asm_emitter.rs, and peephole.rs.
