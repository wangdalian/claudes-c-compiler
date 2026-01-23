# Optimization Passes

SSA-based optimization passes that improve the IR before code generation.

## Available Passes

- **constant_fold.rs** - Evaluates constant expressions at compile time (e.g., `3 + 4` -> `7`)
- **dce.rs** - Dead code elimination: removes instructions whose results are never used
- **gvn.rs** - Global value numbering: eliminates redundant computations (common subexpression elimination)
- **simplify.rs** - Algebraic simplification: identity removal (`x + 0` -> `x`), strength reduction (`x * 2` -> `x << 1`), boolean simplification

## Pass Pipeline

Passes run in a fixed pipeline with iteration count based on `-O` level:

- `-O0`: No passes run
- `-O1`: 1 iteration (simplify, constant fold, DCE; no GVN)
- `-O2`: 2 iterations (adds GVN/CSE)
- `-O3`: 3 iterations

Each iteration runs: simplify -> constant fold -> GVN (O2+) -> DCE.

## Adding New Passes

Each pass implements `fn run(module: &mut IrModule)`. Register new passes in
`mod.rs::run_passes()` at the appropriate pipeline position.
