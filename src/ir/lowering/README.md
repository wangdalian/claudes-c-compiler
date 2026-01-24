# IR Lowering

Translates the parsed AST into alloca-based IR. This is the largest module because it
handles every C language construct. The `mem2reg` pass later promotes allocas to SSA.

## Module Organization

| File | Responsibility |
|---|---|
| `lowering.rs` | `Lowerer` struct, `lower()` entry point, `lower_function`, scope management, `DeclAnalysis` |
| `stmt.rs` | Statement dispatch (`lower_stmt`), `lower_local_decl`, `emit_struct_init`, control flow |
| `stmt_init.rs` | Local variable init helpers: expr-init, list-init, extern/func-decl handling |
| `stmt_return.rs` | Return statement: sret, two-reg struct, complex decomposition, scalar returns |
| `expr.rs` | Expression lowering: binary/unary ops, casts, ternary, sizeof, pointer arithmetic |
| `expr_builtins.rs` | `__builtin_*` intrinsics (fpclassify, clz, ctz, bswap, popcount, etc.) |
| `expr_atomics.rs` | `__atomic_*` and `__sync_*` operations via table-driven dispatch |
| `expr_calls.rs` | Function call lowering with ABI handling (sret, complex args, struct passing) |
| `expr_assign.rs` | Assignment/compound-assign, bitfield read-modify-write, type promotions |
| `expr_types.rs` | Expression type inference (`get_expr_type`, `expr_ctype`) |
| `lvalue.rs` | L-value resolution and array address computation |
| `types.rs` | `TypeSpecifier` to `IrType`/`CType`, sizeof/alignof |
| `structs.rs` | Struct/union layout cache, field offset resolution |
| `complex.rs` | `_Complex` arithmetic, assignment, and conversions |
| `global_init.rs` | Global initializer dispatch: routes to byte or compound path based on pointer content |
| `global_init_bytes.rs` | Byte-level global init serialization; shared `drill_designators` for nested designator resolution |
| `global_init_compound.rs` | Relocation-aware global init for structs/arrays containing pointer fields |
| `const_eval.rs` | Compile-time constant expression evaluation |
| `pointer_analysis.rs` | Tracks expressions producing struct addresses vs packed data |
| `ref_collection.rs` | Pre-pass to collect referenced static/extern symbols |

## Architecture

The `Lowerer` processes a `TranslationUnit` in multiple passes:

1. **Pass 0**: Collect typedefs so function signatures can resolve aliased types
2. **Pass 1**: Register function signatures (return types, param types, sret/variadic)
3. **Pass 2**: Lower each function body and global initializer into IR

### Key Types

- **`VarInfo`** - Shared type metadata (ty, elem_size, is_array, struct_layout, etc.)
  embedded in both `LocalInfo` and `GlobalInfo` via `Deref`
- **`DeclAnalysis`** - Computed once per declaration, bundles all type properties.
  Used by both local and global lowering to avoid duplicating type analysis
- **`FunctionMeta`** - Per-function metadata (return types, param types, sret info)
- **`ScopeFrame`** - Records additions/shadows per block scope. `pop_scope()` undoes
  changes in O(changes) rather than cloning entire HashMaps

### Key Helpers

- `shadow_local_for_scope(name)` - Remove local and track for scope restoration
- `register_block_func_meta(name, ...)` - Register function metadata for block-scope declarations
- `lower_return_expr(e)` - Handles all return conventions (sret, two-reg, complex, scalar)
- `lower_local_init_expr(...)` / `lower_local_init_list(...)` - Dispatch init by type

### Global Initialization Subsystem

The global init code handles C's complex initialization rules for global/static variables.
The key architectural decision is the **two-path split** based on whether a struct contains
pointer fields:

```
global_init.rs: lower_struct_global_init()
    ├── struct_init_has_addr_fields() → true  → global_init_compound.rs (relocation-aware)
    └── struct_init_has_addr_fields() → false → global_init_bytes.rs (flat byte array)
```

**Why two paths?** Pointer fields need `GlobalAddr` entries that become linker relocations
(e.g., `const char *s = "hello"` needs a `.quad .L.str` directive). Non-pointer structs can
be fully serialized to a `Vec<u8>` byte buffer, which is simpler and more efficient.

**Shared helpers** (in `global_init_bytes.rs`):
- `drill_designators(designators, start_ty)` — Walks a chain of field/index designators
  to resolve the target type and byte offset. Used by both paths for nested designators
  like `.u.keyword` or `[3].field.subfield`. This is the single implementation point for
  designator chain resolution, avoiding duplication across the byte and compound paths.
- `resolve_struct_init_field_idx()` — Maps a positional or designated initializer to its
  target struct field index.
- `write_const_to_bytes()`, `write_bitfield_to_bytes()` — Low-level byte buffer operations.

The **compound path** (`global_init_compound.rs`) also handles struct arrays with pointer
fields via a hybrid approach: serializes scalar fields to bytes, collects pointer relocations
separately, then merges them into a single `GlobalInit::Compound` vector.

## Design Decisions

- **Alloca-based**: Every local gets an `Alloca`. `mem2reg` promotes to SSA later.
- **Scope stack**: Push/pop `ScopeFrame`s instead of cloning HashMaps at block boundaries.
- **Complex ABI**: `_Complex double` decomposes to two FP registers, `_Complex float`
  packs into I64, larger types use sret. Handled in `stmt_return.rs`.
- **Short-circuit**: `&&`/`||` use conditional branches, not boolean ops.

## Relationship to Other Modules

```
parser/AST + sema/types  →  lowering  →  ir::Module  →  mem2reg → passes → codegen
```
