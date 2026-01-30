# i686 Backend

Code generation targeting 32-bit x86 (i686) with the cdecl calling convention.

## Key Differences from x86-64

- **ILP32 type model**: pointers are 4 bytes, `long` is 4 bytes, `size_t` is `unsigned int`
- **cdecl calling convention**: all arguments passed on the stack by default. With `-mregparm=N` (N=1..3), the first N integer args are passed in EAX, EDX, ECX. Return values in `eax` (32-bit) or `eax:edx` (64-bit). Also supports `__attribute__((fastcall))` (first two DWORD args in ECX/EDX)
- **No native 64-bit arithmetic**: 64-bit operations are split into 32-bit register pairs (`eax:edx`). Division-by-constant strength reduction is disabled because the generated 64-bit multiply sequences cannot be executed correctly
- **x87 FPU for long double**: same as x86-64, F128 operations use the x87 FPU stack
- **Limited register pool**: 6 general-purpose registers (eax, ecx, edx, ebx, esi, edi) with ebx/esi/edi callee-saved

## Structure

- `codegen/` - Code generation implementation
  - `codegen.rs` - Main `I686Codegen` struct implementing the `ArchCodegen` trait. The largest backend file due to 64-bit operation splitting (i128 ops require 4-register sequences)
  - `asm_emitter.rs` - `InlineAsmEmitter` trait implementation for i686 inline assembly
  - `inline_asm.rs` - i686 inline assembly template substitution and register formatting

## Known Limitations

- mbedtls project test fails (pre-existing)
- Division-by-constant pass disabled (generates 64-bit arithmetic the backend can't handle)
