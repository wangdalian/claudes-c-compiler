# CCC - C Compiler Collection

A C compiler written in Rust, targeting x86-64, AArch64, and RISC-V 64.

## Status

**Initial scaffold complete.** The basic compilation pipeline is functional:
- Lexer, preprocessor (strip-only), parser, semantic analysis (stub)
- IR lowering (AST -> alloca-based IR)
- Code generation for x86-64, AArch64, and RISC-V 64
- Assembly and linking via system tools (gcc/gas)

### Test Results (1% sample)
- x86-64: ~16% passing
- AArch64: ~11% passing
- RISC-V 64: ~17% passing

### What Works
- `int main() { return N; }` for any integer N
- `printf()` with string literal arguments (via libc linking)
- Basic arithmetic (`+`, `-`, `*`, `/`, `%`)
- Local variable declarations and assignments
- `if`/`else`, `while`, `for`, `do-while` control flow
- Function calls with up to 6/8 arguments
- Comparison operators
- Array declarations, subscript read/write (`arr[i]`, `arr[i] = val`)
- Array initializer lists (`int arr[] = {1, 2, 3}`)
- Pointer dereference assignment (`*p = val`, `val = *p`)
- Address-of operator (`&x`, `&arr[i]`)
- Compound assignment on arrays/pointers (`arr[i] += val`, `*p -= val`)
- Pre/post increment/decrement on arrays/pointers (`arr[i]++`, `++(*p)`)
- Short-circuit evaluation for `&&` and `||`
- Proper `sizeof` for basic types and arrays

### What's Not Yet Implemented
- Preprocessor (macros, includes, conditionals)
- Type checking (sema is a stub)
- Structs, unions, enums (parsed but not lowered)
- Switch statements (stub)
- Floating point
- Global variables
- Type casts
- Native assembler/linker (currently uses gcc)
- Optimization passes

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
    preprocessor/       Strip preprocessor directives (TODO: full expansion)
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
