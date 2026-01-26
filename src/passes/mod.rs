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

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::ir::{Instruction, IrModule, IrFunction, GlobalInit, Operand, Value};

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

        // Re-run mem2reg after inlining. The inliner creates parameter allocas
        // in the callee's entry block, which becomes a non-entry block in the
        // caller. These store/load pairs through stack slots must be promoted
        // to SSA values so that constant arguments (e.g., feature bit numbers)
        // propagate through to inline asm "i" (immediate) constraints.
        // Without this, expressions like `1 << (bit & 7)` in _static_cpu_has()
        // cannot be constant-folded and inline asm emits $0 placeholders.
        crate::ir::mem2reg::promote_allocas(module);

        // After mem2reg, run scalar optimization passes so that arithmetic
        // on inlined constants is fully resolved before we try to resolve
        // inline asm symbols. The simplify pass reduces expressions like
        // cast chains, and constant folding evaluates the final values.
        // Two rounds of constant_fold+copy_prop are needed: the first resolves
        // simple arithmetic, simplify reduces remaining expressions, and the
        // second round folds the simplified results to constants.
        constant_fold::run(module);
        copy_prop::run(module);
        simplify::run(module);
        constant_fold::run(module);
        copy_prop::run(module);

        // Resolve InlineAsm input symbols that became resolvable after inlining.
        // When a callee like _static_cpu_has(596) is inlined, its "i" constraint
        // operand `&boot_cpu_data.x86_capability[bit >> 3]` becomes a GlobalAddr +
        // constant GEP chain. This pass traces the def chain and sets input_symbols
        // to "boot_cpu_data+74" so the backend can emit correct symbol references.
        resolve_inline_asm_symbols(module);
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

/// Resolve InlineAsm input symbols by tracing Value operands back through their
/// def chain to find GlobalAddr + constant GEP patterns. This is needed after
/// inlining: callee functions like `_static_cpu_has(u16 bit)` have "i" constraint
/// operands like `&boot_cpu_data.x86_capability[bit >> 3]` that couldn't be resolved
/// at lowering time (bit was a parameter). After inlining with a constant argument
/// and running mem2reg + constant folding, the IR contains a chain like:
///   v1 = GlobalAddr @boot_cpu_data
///   v2 = GEP v1, Const(74)
/// This function recognizes that pattern and sets `input_symbols[i] = "boot_cpu_data+74"`.
fn resolve_inline_asm_symbols(module: &mut IrModule) {
    for func in &mut module.functions {
        if func.is_declaration || func.blocks.is_empty() {
            continue;
        }
        resolve_asm_symbols_in_function(func);
    }
}

fn resolve_asm_symbols_in_function(func: &mut IrFunction) {
    // Build a map from Value ID to its defining instruction for fast lookup.
    let mut value_defs: FxHashMap<u32, DefInfo> = FxHashMap::default();
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::GlobalAddr { dest, name } => {
                    value_defs.insert(dest.0, DefInfo::GlobalAddr(name.clone()));
                }
                Instruction::GetElementPtr { dest, base, offset, .. } => {
                    value_defs.insert(dest.0, DefInfo::Gep(*base, offset.clone()));
                }
                Instruction::BinOp { dest, op: crate::ir::ir::IrBinOp::Add, lhs, rhs, .. } => {
                    value_defs.insert(dest.0, DefInfo::Add(lhs.clone(), rhs.clone()));
                }
                Instruction::Cast { dest, src, .. } => {
                    value_defs.insert(dest.0, DefInfo::Cast(src.clone()));
                }
                Instruction::Copy { dest, src } => {
                    value_defs.insert(dest.0, DefInfo::Cast(src.clone()));
                }
                _ => {}
            }
        }
    }

    // Now scan InlineAsm instructions and try to resolve unresolved input_symbols.
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            if let Instruction::InlineAsm { inputs, input_symbols, .. } = inst {
                for (i, (constraint, operand, _)) in inputs.iter().enumerate() {
                    // Only process "i" constraint inputs that don't have a symbol yet
                    if i >= input_symbols.len() { break; }
                    if input_symbols[i].is_some() { continue; }
                    // Only for immediate-only constraints
                    if !crate::backend::inline_asm::constraint_is_immediate_only(constraint) {
                        continue;
                    }
                    // Try to resolve the operand to a symbol+offset
                    if let Operand::Value(v) = operand {
                        if let Some(sym) = try_resolve_global_symbol(v, &value_defs) {
                            input_symbols[i] = Some(sym);
                        }
                    }
                }
            }
        }
    }
}

/// Information about a value's defining instruction, used for symbol resolution.
enum DefInfo {
    GlobalAddr(String),
    Gep(Value, Operand),
    Add(Operand, Operand),
    Cast(Operand),
}

/// Try to resolve a Value to a global symbol + constant offset string.
/// Returns e.g., "boot_cpu_data" or "boot_cpu_data+74".
fn try_resolve_global_symbol(val: &Value, defs: &FxHashMap<u32, DefInfo>) -> Option<String> {
    let (name, offset) = try_resolve_global_with_offset(val, defs, 0)?;
    if offset == 0 {
        Some(name)
    } else if offset > 0 {
        Some(format!("{}+{}", name, offset))
    } else {
        Some(format!("{}{}", name, offset))
    }
}

/// Recursively trace a Value back through its def chain to find
/// GlobalAddr + constant offsets (from GEP, Add, Cast/Copy).
fn try_resolve_global_with_offset(val: &Value, defs: &FxHashMap<u32, DefInfo>, accum_offset: i64) -> Option<(String, i64)> {
    let def = defs.get(&val.0)?;
    match def {
        DefInfo::GlobalAddr(name) => Some((name.clone(), accum_offset)),
        DefInfo::Gep(base, offset) => {
            let off = match offset {
                Operand::Const(c) => c.to_i64()?,
                Operand::Value(_) => {
                    // Non-constant GEP offset - can't resolve to symbol+offset
                    return None;
                }
            };
            try_resolve_global_with_offset(base, defs, accum_offset + off)
        }
        DefInfo::Add(lhs, rhs) => {
            // Pattern: base + const_offset or const_offset + base
            match (lhs, rhs) {
                (Operand::Value(base), Operand::Const(c)) => {
                    let off = c.to_i64()?;
                    try_resolve_global_with_offset(base, defs, accum_offset + off)
                }
                (Operand::Const(c), Operand::Value(base)) => {
                    let off = c.to_i64()?;
                    try_resolve_global_with_offset(base, defs, accum_offset + off)
                }
                _ => None,
            }
        }
        DefInfo::Cast(src) => {
            // Look through casts and copies
            match src {
                Operand::Value(v) => try_resolve_global_with_offset(v, defs, accum_offset),
                _ => None,
            }
        }
    }
}
