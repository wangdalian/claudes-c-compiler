# RISC-V 64 Backend

Code generation targeting the RISC-V 64-bit (RV64GC) architecture.

## Structure

- `codegen/` - Code generation implementation
  - `codegen.rs` - Main `RiscvCodegen` struct implementing the `ArchCodegen` trait. Handles instruction selection, stack frame management (s0 frame pointer), calling convention (a0-a7 for integer args, fa0-fa7 for FP), varargs, and register allocation integration.
  - `inline_asm.rs` - RISC-V inline assembly constraint classification and template substitution (%0, %[name], %lo/%hi operands).
  - `atomics.rs` - Sub-word atomic RMW/CAS via LR.W/SC.W loops with bit masking, and software implementations of CLZ, CTZ, BSWAP, POPCOUNT builtins.

## Register Allocation

The RISC-V backend includes a linear scan register allocator that assigns callee-saved registers to frequently-used IR values. This is the first backend with register allocation; x86 and ARM remain stack-only.

**Allocated registers**: s1, s7-s11 (6 registers). s0 is the frame pointer; s2-s6 are reserved as temporaries for `emit_call_reg_args`.

**Strategy**: Write-through — values are always stored to both the stack slot and the callee-saved register. Reads check the register first (`operand_to_t0`). This ensures safety since some codegen paths bypass `store_t0_to`.

**Disabled for**: functions with loops (back-edges in CFG), inline assembly, atomic operations, float/i128/long-double types. Only BinOp, UnaryOp, Cmp, Cast, Load, and GEP results are eligible.
