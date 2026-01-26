# IR (Intermediate Representation)

The IR is a low-level SSA-style representation used between the frontend AST and backend code generation.

## Structure

An `IrModule` contains:
- **Functions** (`IrFunction`) - each with typed parameters and a list of basic blocks
- **Globals** (`IrGlobal`) - global variable definitions with initializers
- **String literals** - collected during lowering for the `.rodata` section

Each basic block has a list of `Instruction`s and a `Terminator` (branch, conditional branch, return, unreachable).

## Modules

- **ir.rs** - Core IR data structures: `IrModule`, `IrFunction`, `BasicBlock`, `Instruction`, `Terminator`, `Operand`, `Value`, `IrConst`. `IrConst` provides type-safe conversion helpers (`to_i64()`, `to_f64()`, `is_nonzero()`, `coerce_to()`, `cast_float_to_target()`, `cast_long_double_to_target()`, `push_le_bytes()`, `from_i64()`, `zero()`), eliminating manual match-on-variant patterns throughout the codebase. The `LongDouble(f64, [u8; 16])` variant carries both an f64 approximation (for quick arithmetic/comparisons) and the raw x87 80-bit bytes (for precise codegen and data emission).
- **analysis.rs** - Shared CFG and dominator tree analysis: `build_label_map`, `build_cfg`, `compute_dominators` (Cooper-Harvey-Kennedy), `compute_dominance_frontiers`, `build_dom_tree_children`. Used by mem2reg and optimization passes.
- **lowering/** - AST → IR lowering (see lowering/README.md)
- **mem2reg/** - SSA promotion pass: promotes alloca/load/store to SSA values with phi insertion

## Key Design Decisions

- Instructions use an alloca-based memory model (like LLVM's `-O0`). The mem2reg pass promotes stack allocations to SSA registers with phi insertion.
- Values are identified by numeric IDs (`Value(u32)`), not names. This simplifies the IR and avoids name collision issues.
- The IR is target-independent: it uses abstract types (I8, I16, I32, I64, Ptr) and delegates ABI details to the backend.
