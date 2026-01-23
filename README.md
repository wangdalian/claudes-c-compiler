# CCC - C Compiler Collection

A C compiler written from scratch in Rust, targeting x86-64, AArch64, and RISC-V 64.

## Status

**Basic compilation pipeline functional.** ~68% of x86-64 tests passing.

### Working Features
- Preprocessor with `#include` file resolution (system headers, -I paths, include guards, #pragma once)
- Recursive descent parser with typedef tracking
- Type-aware IR lowering and code generation
- Optimization passes (constant folding, DCE, GVN, algebraic simplification)
- Three backend targets with correct ABI handling

### Test Results (10% sample, ratio 10)
- x86-64: ~68.6% passing (2051/2991)
- AArch64: ~71.5% passing (2051/2869)
- RISC-V 64: ~65.1% passing (1862/2861)

### What Works
- `int main() { return N; }` for any integer N
- `printf()` with string literal arguments (via libc linking)
- Basic arithmetic (`+`, `-`, `*`, `/`, `%`)
- Local variable declarations and assignments
- `if`/`else`, `while`, `for`, `do-while` control flow
- Function calls with up to 6/8 arguments
- Comparison operators
- **Type-aware code generation**: correct-sized load/store instructions for all types
  - `char` uses byte operations (movb/strb/sb)
  - `short` uses 16-bit operations (movw/strh/sh)
  - `int` uses 32-bit operations (movl/str w/sw)
  - `long`/pointers use 64-bit operations (movq/str x/sd)
  - Sign extension on loads for smaller types
- Array declarations with correct element sizes (4 bytes for int[], 1 for char[], etc.)
- Array subscript read/write (`arr[i]`, `arr[i] = val`)
- Array initializer lists (`int arr[] = {1, 2, 3}`)
- Pointer dereference assignment (`*p = val`, `val = *p`)
- Address-of operator (`&x`, `&arr[i]`)
- Compound assignment on arrays/pointers (`arr[i] += val`, `*p -= val`)
- Pre/post increment/decrement on arrays/pointers (`arr[i]++`, `++(*p)`)
- Short-circuit evaluation for `&&` and `||`
- Proper `sizeof` for basic types and arrays
- 32-bit arithmetic operations for `int` type (addl/subl/imull/idivl on x86)
- **Switch statements**: proper dispatch via if-else comparison chain
  - Case with break, fallthrough, default
  - Nested switch statements
  - Constant case expressions (integer literals, char literals, arithmetic)
- **Global variables**: declarations with initializers, arrays, zero-initialized (.bss)
  - Global scalar initializers (`int x = 42;`)
  - Global array initializers (`int arr[5] = {1, 2, 3, 4, 5};`)
  - Global pointer-to-string (`char *msg = "hello";`)
  - Read/write access to globals from any function
  - Constant expression evaluation for initializers

### Recent Additions
- **CType::Bool and _Bool normalization**: Added `CType::Bool` variant to distinguish `_Bool` from
  `unsigned char` at the type level. Stores to `_Bool` lvalues through pointer dereference,
  array subscript, struct/union member access, compound assignment, and increment/decrement
  now correctly normalize values to 0 or 1.
- **#include file resolution**: Full `#include` support reading actual system headers from
  `/usr/include`, with search path support (-I), include guard recognition, #pragma once,
  block comment stripping (C-style `/* */` comments no longer confuse the preprocessor),
  recursive include processing, and GCC-compatible predefined macros.
- **Static local variables**: `static` locals are emitted as globals with mangled names
  (e.g., `func.varname`), preserving values across function calls. Works with
  scalars, arrays, and initializers. Storage class tracking in parser and AST.
- **Typedef tracking**: parser correctly registers typedef names from `typedef` declarations
  (both top-level and local), enabling cast expressions like `(mytype)expr` with user-defined types
- **Built-in type names**: standard C type names (`size_t`, `int32_t`, `FILE`, etc.) pre-seeded
  for correct parsing without full header inclusion
- **Cast expression lowering**: emits proper IR Cast instructions for type-narrowing casts
- **_Complex type handling**: parses and skips `_Complex`/`__complex__` type modifier
- **Inline asm skipping**: parses and skips `asm`/`__asm__` statements and expressions
- **GCC extension keywords**: `__volatile__`, `__const__`, `__inline__`, `__restrict__`,
  `__signed__`, `__noreturn__` recognized as their standard equivalents

### What's Not Yet Implemented
- Parser support for GNU C extensions in system headers (`__attribute__`, `__asm__` renames)
- Floating point
- Full cast semantics (truncation/sign-extension in some cases)
- Inline assembly (parsed but skipped)
- Native assembler/linker (currently uses gcc)

## Building

```bash
cargo build --release
# Produces: target/release/ccc (x86), ccc-arm, ccc-riscv
```

## Usage

```bash
target/release/ccc -o output input.c       # x86-64
target/release/ccc-arm -o output input.c   # AArch64
target/release/ccc-riscv -o output input.c # RISC-V 64

# GCC-compatible flags: -S, -c, -E, -O0..3, -g, -D, -I
```

## Architecture

```
src/
  frontend/              C source → AST
    preprocessor/        Macro expansion, #include, #ifdef
    lexer/               Tokenization with source locations
    parser/              Recursive descent, produces AST
    sema/                Semantic analysis, symbol table

  ir/                    Target-independent SSA IR
    ir.rs                Core data structures (IrModule, Instructions, BasicBlock)
    lowering/            AST → alloca-based IR (split into expr/stmt/lvalue/types/structs)
    mem2reg/             SSA promotion (stub)

  passes/                Optimization: constant_fold, dce, gvn, simplify

  backend/               IR → assembly → object → executable
    common.rs            Shared data emission, assembler/linker invocation
    x86/codegen/         x86-64 instruction selection (SysV ABI)
    arm/codegen/         AArch64 instruction selection (AAPCS64)
    riscv/codegen/       RISC-V 64 instruction selection

  common/                Shared types (CType, IrType), symbol table, diagnostics
  driver/                CLI argument parsing, pipeline orchestration
```

Each subdirectory has its own README.md explaining the design and relationships.

## Testing

```bash
python3 /verify/verify_compiler.py --compiler target/release/ccc --arch x86 --ratio 10  # Quick (10%)
python3 /verify/verify_compiler.py --compiler target/release/ccc --arch x86              # Full suite
```
