# IR (Intermediate Representation)

The IR is a low-level SSA-style representation used between the frontend AST and backend code generation.

## Structure

An `IrModule` contains:
- **Functions** (`IrFunction`) - each with typed parameters and a list of basic blocks
- **Globals** (`IrGlobal`) - global variable definitions with initializers
- **String literals** - collected during lowering for the `.rodata` section

Each basic block has a list of `Instruction`s and a `Terminator` (branch, conditional branch, return, unreachable).

## Modules

The core IR types were split from a single monolithic `ir.rs` into focused submodules. All types are re-exported through `ir.rs`, so consumers use `crate::ir::ir::*` unchanged.

| File | Purpose |
|------|---------|
| **ir.rs** | Re-export hub — imports and re-exports all submodule types |
| **constants.rs** | `IrConst`, `ConstHashKey`, float encoding (`f64_to_f128_bytes`, `f64_to_x87_bytes`) |
| **ops.rs** | `IrBinOp`, `IrUnaryOp`, `IrCmpOp`, `AtomicRmwOp`, `AtomicOrdering` with eval methods |
| **intrinsics.rs** | `IntrinsicOp` — 80+ SIMD/crypto/math intrinsic variants with `is_pure()` |
| **instruction.rs** | `Instruction` (30+ variants), `Terminator`, `BasicBlock`, `BlockId`, `Value`, `Operand`, `CallInfo` |
| **module.rs** | `IrModule`, `IrFunction`, `IrParam`, `IrGlobal`, `GlobalInit` |
| **analysis.rs** | CFG and dominator tree: `build_cfg`, `compute_dominators` (Cooper-Harvey-Kennedy), `compute_dominance_frontiers` |
| **lowering/** | AST → IR lowering (see lowering/README.md) |
| **mem2reg/** | SSA promotion: promotes alloca/load/store to SSA values with phi insertion |

## Key Design Decisions

- Instructions use an alloca-based memory model (like LLVM's `-O0`). The mem2reg pass promotes stack allocations to SSA registers with phi insertion.
- Values are identified by numeric IDs (`Value(u32)`), not names. This simplifies the IR and avoids name collision issues.
- The IR is target-independent: it uses abstract types (I8, I16, I32, I64, Ptr) and delegates ABI details to the backend.
- `IrConst::LongDouble(f64, [u8; 16])` carries both an f64 approximation (for quick arithmetic) and raw f128 bytes (for precise codegen). Architecture-specific emission uses `push_le_bytes_x86` (x87 format) or `push_le_bytes_riscv` (IEEE binary128).
- `Instruction::for_each_used_value()` is the canonical value visitor — all passes should use it instead of duplicating the match block.
