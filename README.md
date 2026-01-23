# CCC - C Compiler Collection

A C compiler written in Rust, targeting x86-64, AArch64, and RISC-V 64.

## Status

**Basic compilation pipeline functional.** Supports:
- Preprocessor (macros, conditionals, built-in headers)
- Lexer with source locations, parser with typedef tracking
- IR lowering with type-aware operations
- Code generation for x86-64, AArch64, and RISC-V 64
- Assembly and linking via system tools (gcc/gas)

### Test Results (1% sample)
- x86-64: ~25% passing
- AArch64: ~12% passing
- RISC-V 64: ~17% passing

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
- Full `#include` resolution (only built-in headers for now)
- Floating point
- Full cast semantics (truncation/sign-extension in some cases)
- Inline assembly (parsed but skipped)
- Native assembler/linker (currently uses gcc)

## Building

```bash
cargo build
# Produces: target/debug/ccc, ccc-x86, ccc-arm, ccc-riscv
```

## Usage

```bash
# Compile C to x86-64 executable
target/debug/ccc -o output input.c

# Compile for AArch64
target/debug/ccc-arm -o output input.c

# Compile for RISC-V 64
target/debug/ccc-riscv -o output input.c
```

## Architecture

```
src/
  frontend/
    preprocessor/       Macro expansion, conditionals, built-in headers
    lexer/              Tokenize C source with source locations
    parser/             Recursive descent parser, produces AST
    sema/               Semantic analysis (TODO: type checking)

  ir/
    ir.rs               IR definition (instructions, basic blocks, values)
    lowering/           AST -> alloca-based IR
    mem2reg/            TODO: promote allocas to SSA

  passes/              TODO: optimization passes (constant fold, DCE, etc.)

  backend/
    x86/
      codegen/          IR -> x86-64 assembly (stack-based allocation)
      assembler/        Assembly -> object file (via gcc -c)
      linker/           Object files -> executable (via gcc)
    arm/
      codegen/          IR -> AArch64 assembly
      assembler/        via aarch64-linux-gnu-gcc
      linker/           via aarch64-linux-gnu-gcc
    riscv/
      codegen/          IR -> RISC-V 64 assembly
      assembler/        via riscv64-linux-gnu-gcc
      linker/           via riscv64-linux-gnu-gcc

  common/
    types.rs            CType, IrType
    symbol_table.rs     Scoped name resolution
    source.rs           Span, SourceLocation, SourceManager
    error.rs            Diagnostic with span

  driver/               CLI argument parsing, pipeline orchestration
```

## Running Tests

```bash
python3 /verify/verify_compiler.py --compiler target/debug/ccc-x86 --ratio 100
python3 /verify/verify_compiler.py --compiler target/debug/ccc-arm --ratio 100
python3 /verify/verify_compiler.py --compiler target/debug/ccc-riscv --ratio 100
```
