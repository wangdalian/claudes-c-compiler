//! Optimization passes for the IR.
//!
//! This module contains various optimization passes that transform the IR
//! to produce better code. Passes are organized by optimization level:
//!
//! - O0: No optimization (skip all passes)
//! - O1: Basic optimizations (constant folding, simplification, copy propagation, DCE, CFG simplification)
//! - O2: Standard optimizations (O1 + GVN/CSE, repeated passes)
//! - O3: Aggressive optimizations (O2 + more iterations)

pub mod cfg_simplify;
pub mod constant_fold;
pub mod copy_prop;
pub mod dce;
pub mod gvn;
pub mod simplify;

use crate::ir::ir::IrModule;

/// Run all optimization passes on the module based on the optimization level.
///
/// The pass pipeline is:
/// 1. CFG simplification (remove dead blocks, thread jump chains, simplify branches)
/// 2. Copy propagation (replace uses of copies with original values)
/// 3. Algebraic simplification (strength reduction)
/// 4. Constant folding (evaluate const exprs at compile time)
/// 5. GVN / CSE (local value numbering, O2+)
/// 6. Copy propagation (clean up copies from GVN/simplification)
/// 7. Dead code elimination (remove dead instructions)
/// 8. CFG simplification (clean up after DCE may have made blocks dead)
///
/// Higher optimization levels run more iterations of this pipeline.
pub fn run_passes(module: &mut IrModule, opt_level: u32) {
    if opt_level == 0 {
        return; // No optimization at O0
    }

    // Determine number of pipeline iterations based on opt level
    let iterations = match opt_level {
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 1,
    };

    for _ in 0..iterations {
        let mut changes = 0usize;

        // Phase 1: CFG simplification (remove dead blocks, thread jump chains,
        // simplify redundant conditional branches). Running early eliminates
        // dead code before other passes waste time analyzing it.
        changes += cfg_simplify::run(module);

        // Phase 2: Copy propagation (early - propagate copies from phi elimination
        // and lowering so subsequent passes see through them)
        changes += copy_prop::run(module);

        // Phase 3: Algebraic simplification (x+0 => x, x*1 => x, etc.)
        changes += simplify::run(module);

        // Phase 4: Constant folding (evaluate const exprs at compile time)
        changes += constant_fold::run(module);

        // Phase 5: GVN / Common Subexpression Elimination
        if opt_level >= 2 {
            changes += gvn::run(module);
        }

        // Phase 6: Copy propagation again (clean up copies created by GVN/simplify)
        changes += copy_prop::run(module);

        // Phase 7: Dead code elimination (clean up dead instructions including dead copies)
        changes += dce::run(module);

        // Phase 8: CFG simplification again (DCE + constant folding may have
        // simplified conditions, creating dead blocks or redundant branches)
        changes += cfg_simplify::run(module);

        // Early exit: if no passes changed anything, additional iterations are useless
        if changes == 0 {
            break;
        }
    }
}
