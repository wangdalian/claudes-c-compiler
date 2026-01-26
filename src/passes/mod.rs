//! Optimization passes for the IR.
//!
//! This module contains various optimization passes that transform the IR
//! to produce better code.
//!
//! All optimization levels (-O0 through -O3, -Os, -Oz) run the same full set
//! of passes. While the compiler is still maturing, having separate tiers
//! creates hard-to-find bugs where code works at one level but breaks at
//! another. We always run all passes to maximize test coverage of the
//! optimizer and catch issues early.

pub mod cfg_simplify;
pub mod constant_fold;
pub mod copy_prop;
pub mod dce;
pub mod gvn;
pub mod if_convert;
pub mod inline;
pub mod ipcp;
pub mod licm;
pub mod narrow;
pub mod simplify;

use crate::common::fx_hash::FxHashSet;
use crate::ir::ir::{Instruction, IrModule, GlobalInit};

/// Run all optimization passes on the module.
///
/// The pass pipeline is:
/// 1. CFG simplification (remove dead blocks, thread jump chains, simplify branches)
/// 2. Copy propagation (replace uses of copies with original values)
/// 3. Algebraic simplification (strength reduction)
/// 4. Constant folding (evaluate const exprs at compile time)
/// 5. GVN / CSE (dominator-based value numbering, eliminates redundant
///    BinOp, UnaryOp, Cmp, Cast, and GetElementPtr across all dominated blocks)
/// 6. LICM (hoist loop-invariant code to preheaders)
/// 7. If-conversion (convert branch+phi diamonds to Select)
/// 8. Copy propagation (clean up copies from GVN/simplify/LICM)
/// 9. Dead code elimination (remove dead instructions)
/// 10. CFG simplification (clean up after DCE may have made blocks dead)
/// 11. Dead static function elimination (remove unreferenced internal-linkage functions)
///
/// All optimization levels run the same pipeline with the same number of
/// iterations. The `opt_level` parameter is accepted for API compatibility
/// but currently ignored -- all levels behave identically. This is intentional:
/// while the compiler is still maturing, running all optimizations at every
/// level maximizes test coverage and avoids bugs that only surface at specific
/// optimization tiers.
// TODO: Restore per-level optimization tiers once the compiler is stable enough
// to warrant differentiated behavior (e.g., -O0 skipping passes for faster builds).
pub fn run_passes(module: &mut IrModule, _opt_level: u32) {
    // Debug support: set CCC_DISABLE_PASSES=pass1,pass2,... to skip specific passes.
    // Useful for bisecting miscompilation bugs to a specific pass.
    // Pass names: cfg, copyprop, simplify, constfold, gvn, licm, ifconv, dce, all
    let disabled = std::env::var("CCC_DISABLE_PASSES").unwrap_or_default();
    if disabled.contains("all") {
        return;
    }

    // Phase 0: Function inlining (before the optimization loop).
    // Inline small static/static-inline functions so that subsequent passes
    // can see through constant-returning helpers and eliminate dead branches.
    // This is critical for kernel code patterns like IS_ENABLED() guards.
    if !disabled.contains("inline") {
        inline::run(module);
    }

    // Always run 3 iterations of the full pipeline. The early-exit check below
    // will skip remaining iterations if no changes were made.
    let iterations = 3;

    for iter in 0..iterations {
        let mut changes = 0usize;

        // Phase 1: CFG simplification (remove dead blocks, thread jump chains,
        // simplify redundant conditional branches). Running early eliminates
        // dead code before other passes waste time analyzing it.
        if !disabled.contains("cfg") {
            changes += cfg_simplify::run(module);
        }

        // Phase 2: Copy propagation (early - propagate copies from phi elimination
        // and lowering so subsequent passes see through them)
        if !disabled.contains("copyprop") {
            changes += copy_prop::run(module);
        }

        // Phase 2b: Integer narrowing (widen-operate-narrow => direct narrow operation)
        // Must run after copy propagation so widening casts are visible,
        // and before other optimizations to reduce instruction count.
        if !disabled.contains("narrow") {
            changes += narrow::run(module);
        }

        // Phase 3: Algebraic simplification (x+0 => x, x*1 => x, etc.)
        if !disabled.contains("simplify") {
            changes += simplify::run(module);
        }

        // Phase 4: Constant folding (evaluate const exprs at compile time)
        if !disabled.contains("constfold") {
            changes += constant_fold::run(module);
        }

        // Phase 5: GVN / Common Subexpression Elimination (dominator-based)
        // Eliminates redundant computations both within and across basic blocks.
        if !disabled.contains("gvn") {
            changes += gvn::run(module);
        }

        // Phase 6: LICM - hoist loop-invariant code to preheaders.
        // Runs after scalar opts have simplified expressions, so we can
        // identify more invariants. Particularly helps inner loops with
        // redundant index computations (e.g., i*n in matrix multiply).
        if !disabled.contains("licm") {
            changes += licm::run(module);
        }

        // Phase 7: If-conversion - convert simple branch+phi diamonds to Select
        // instructions. Runs after scalar optimizations have simplified the CFG,
        // enabling cmov/csel emission instead of branches for simple conditionals.
        if !disabled.contains("ifconv") {
            changes += if_convert::run(module);
        }

        // Phase 8: Copy propagation again (clean up copies created by GVN/simplify)
        if !disabled.contains("copyprop") {
            changes += copy_prop::run(module);
        }

        // Phase 9: Dead code elimination (clean up dead instructions including dead copies)
        if !disabled.contains("dce") {
            changes += dce::run(module);
        }

        // Phase 10: CFG simplification again (DCE + constant folding may have
        // simplified conditions, creating dead blocks or redundant branches)
        if !disabled.contains("cfg") {
            changes += cfg_simplify::run(module);
        }

        // Phase 10.5: Interprocedural constant return propagation (IPCP).
        // After the first iteration of intra-procedural optimizations, static
        // functions that always return a constant should have been simplified to
        // `return const`. Now we can replace calls to those functions with the
        // constant value, enabling subsequent DCE/CFG passes to eliminate dead
        // branches that reference undefined symbols.
        // Run after iteration 0 (first pass) so returns have been simplified,
        // and the result feeds into iteration 1 for cleanup.
        if iter == 0 && !disabled.contains("ipcp") {
            changes += ipcp::run(module);
        }

        // Early exit: if no passes changed anything, additional iterations are useless
        if changes == 0 {
            break;
        }
    }

    // Phase 11: Dead static function elimination.
    // After all optimizations, remove internal-linkage (static) functions that are
    // never referenced by any other function or global initializer. This is critical
    // for `static inline` functions from headers: after intra-procedural optimizations
    // eliminate dead code paths (e.g., `if (1 || expr)` removes the else branch),
    // some static inline callees may become completely unreferenced and can be removed.
    // Without this, the dead functions may reference undefined external symbols
    // (e.g., kernel's `___siphash_aligned` calling `__siphash_aligned` which doesn't
    // exist on x86 where CONFIG_HAVE_EFFICIENT_UNALIGNED_ACCESS is set).
    eliminate_dead_static_functions(module);
}

/// Remove internal-linkage (static) functions that have no callers in the module.
///
/// After optimization passes eliminate dead code paths, some static inline functions
/// from headers may no longer be called. Keeping them would waste code size and may
/// cause linker errors if they reference symbols that don't exist (because the
/// calling path was dead code that GCC would have inlined away).
fn eliminate_dead_static_functions(module: &mut IrModule) {
    // Collect all function names referenced from any function body or global initializer.
    let mut referenced: FxHashSet<String> = FxHashSet::default();

    for func in &module.functions {
        if func.is_declaration {
            continue;
        }
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::Call { func: callee, .. } => {
                        referenced.insert(callee.clone());
                    }
                    Instruction::GlobalAddr { name, .. } => {
                        // Function pointer taken
                        referenced.insert(name.clone());
                    }
                    Instruction::InlineAsm { .. } => {
                        // Inline asm may reference symbols we can't analyze,
                        // but function references in inline asm use GlobalAddr
                    }
                    _ => {}
                }
            }
        }
    }

    // Also check global initializers for function references
    for global in &module.globals {
        collect_global_init_refs(&global.init, &mut referenced);
    }

    // Also check aliases (alias targets must be kept)
    for (_, target, _) in &module.aliases {
        referenced.insert(target.clone());
    }

    // Also check constructors and destructors
    for ctor in &module.constructors {
        referenced.insert(ctor.clone());
    }
    for dtor in &module.destructors {
        referenced.insert(dtor.clone());
    }

    // Remove static (internal linkage) functions that are unreferenced.
    // Non-static functions must be kept because they have external linkage
    // and may be called from other translation units.
    module.functions.retain(|func| {
        // Keep declarations (extern prototypes)
        if func.is_declaration {
            return true;
        }
        // Keep non-static functions (external linkage)
        if !func.is_static {
            return true;
        }
        // Keep referenced static functions
        if referenced.contains(&func.name) {
            return true;
        }
        // This is an unreferenced static function - remove it
        false
    });

    // Filter symbol_attrs to only include symbols that are actually referenced
    // by the emitted code. Without this, unused extern declarations (e.g. from
    // headers) with visibility attributes (e.g. from #pragma GCC visibility
    // push(hidden)) generate .hidden directives that create undefined hidden
    // symbol entries in the object file, causing linker errors.
    module.symbol_attrs.retain(|(name, _, _)| {
        referenced.contains(name)
    });
}

/// Collect function name references from a global initializer.
fn collect_global_init_refs(init: &GlobalInit, refs: &mut FxHashSet<String>) {
    match init {
        GlobalInit::GlobalAddr(name) | GlobalInit::GlobalAddrOffset(name, _) => {
            refs.insert(name.clone());
        }
        GlobalInit::Compound(fields) => {
            for field in fields {
                collect_global_init_refs(field, refs);
            }
        }
        _ => {}
    }
}
