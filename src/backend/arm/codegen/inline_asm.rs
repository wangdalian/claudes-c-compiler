//! AArch64 inline assembly template substitution and register formatting.
//!
//! This module handles operand substitution in inline assembly templates
//! (e.g., `%0`, `%[name]`, `%w0`, `%x0`) and register formatting with
//! w/x/s/d/q modifiers for ARM targets. It also contains helpers for
//! atomic exclusive access instructions (ldxr/stxr) and atomic RMW operations.

use crate::ir::ir::*;
use crate::common::types::IrType;
use crate::backend::state::CodegenState;
use super::codegen::ArmCodegen;

impl ArmCodegen {
    pub(super) fn substitute_asm_operands_static(line: &str, op_regs: &[String], op_names: &[Option<String>]) -> String {
        let mut result = String::new();
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '%' && i + 1 < chars.len() {
                i += 1;
                // Check for %% (literal %)
                if chars[i] == '%' {
                    result.push('%');
                    i += 1;
                    continue;
                }
                // Check for modifier: w, x, h, b, s, d, q
                let mut modifier = None;
                if chars[i] == 'w' || chars[i] == 'x' || chars[i] == 'h' || chars[i] == 'b'
                    || chars[i] == 's' || chars[i] == 'd' || chars[i] == 'q'
                {
                    // Check if next char is digit or [, meaning this is a modifier
                    if i + 1 < chars.len() && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '[') {
                        modifier = Some(chars[i]);
                        i += 1;
                    }
                }

                if chars[i] == '[' {
                    // Named operand: %[name] or %w[name]
                    i += 1;
                    let name_start = i;
                    while i < chars.len() && chars[i] != ']' {
                        i += 1;
                    }
                    let name: String = chars[name_start..i].iter().collect();
                    if i < chars.len() { i += 1; } // skip ]

                    // Look up by name in operands
                    let mut found = false;
                    for (idx, op_name) in op_names.iter().enumerate() {
                        if let Some(ref n) = op_name {
                            if n == &name {
                                result.push_str(&Self::format_reg_static(&op_regs[idx], modifier));
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        // Fallback: emit raw
                        result.push('%');
                        if let Some(m) = modifier { result.push(m); }
                        result.push('[');
                        result.push_str(&name);
                        result.push(']');
                    }
                } else if chars[i].is_ascii_digit() {
                    // Positional operand: %0, %1, %w2, etc.
                    let mut num = 0usize;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        num = num * 10 + (chars[i] as usize - '0' as usize);
                        i += 1;
                    }
                    if num < op_regs.len() {
                        result.push_str(&Self::format_reg_static(&op_regs[num], modifier));
                    } else {
                        result.push_str(&format!("x{}", num));
                    }
                } else {
                    // Not a recognized pattern, emit as-is
                    result.push('%');
                    if let Some(m) = modifier { result.push(m); }
                    result.push(chars[i]);
                    i += 1;
                }
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }
        result
    }

    pub(super) fn format_reg_static(reg: &str, modifier: Option<char>) -> String {
        match modifier {
            Some('w') => {
                // Convert x-register to w-register
                if reg.starts_with('x') {
                    format!("w{}", &reg[1..])
                } else {
                    reg.to_string()
                }
            }
            Some('x') => {
                // Force x-register
                if reg.starts_with('w') {
                    format!("x{}", &reg[1..])
                } else {
                    reg.to_string()
                }
            }
            _ => reg.to_string(),
        }
    }

    /// Get the exclusive load/store instructions and register prefix for a type,
    /// with appropriate acquire/release semantics based on ordering.
    /// - Relaxed: ldxr/stxr (no ordering)
    /// - Acquire: ldaxr/stxr (acquire on load)
    /// - Release: ldxr/stlxr (release on store)
    /// - AcqRel/SeqCst: ldaxr/stlxr (acquire on load, release on store)
    pub(super) fn exclusive_instrs(ty: IrType, ordering: AtomicOrdering) -> (&'static str, &'static str, &'static str) {
        let need_acquire = matches!(ordering, AtomicOrdering::Acquire | AtomicOrdering::AcqRel | AtomicOrdering::SeqCst);
        let need_release = matches!(ordering, AtomicOrdering::Release | AtomicOrdering::AcqRel | AtomicOrdering::SeqCst);
        match ty {
            IrType::I8 | IrType::U8 => (
                if need_acquire { "ldaxrb" } else { "ldxrb" },
                if need_release { "stlxrb" } else { "stxrb" },
                "w",
            ),
            IrType::I16 | IrType::U16 => (
                if need_acquire { "ldaxrh" } else { "ldxrh" },
                if need_release { "stlxrh" } else { "stxrh" },
                "w",
            ),
            IrType::I32 | IrType::U32 => (
                if need_acquire { "ldaxr" } else { "ldxr" },
                if need_release { "stlxr" } else { "stxr" },
                "w",
            ),
            _ => (
                if need_acquire { "ldaxr" } else { "ldxr" },
                if need_release { "stlxr" } else { "stxr" },
                "x",
            ),
        }
    }

    /// Emit the arithmetic operation for an atomic RMW.
    pub(super) fn emit_atomic_op_arm(state: &mut CodegenState, op: AtomicRmwOp, dest_reg: &str, old_reg: &str, val_reg: &str) {
        match op {
            AtomicRmwOp::Add => state.emit_fmt(format_args!("    add {}, {}, {}", dest_reg, old_reg, val_reg)),
            AtomicRmwOp::Sub => state.emit_fmt(format_args!("    sub {}, {}, {}", dest_reg, old_reg, val_reg)),
            AtomicRmwOp::And => state.emit_fmt(format_args!("    and {}, {}, {}", dest_reg, old_reg, val_reg)),
            AtomicRmwOp::Or  => state.emit_fmt(format_args!("    orr {}, {}, {}", dest_reg, old_reg, val_reg)),
            AtomicRmwOp::Xor => state.emit_fmt(format_args!("    eor {}, {}, {}", dest_reg, old_reg, val_reg)),
            AtomicRmwOp::Nand => {
                state.emit_fmt(format_args!("    and {}, {}, {}", dest_reg, old_reg, val_reg));
                state.emit_fmt(format_args!("    mvn {}, {}", dest_reg, dest_reg));
            }
            AtomicRmwOp::Xchg | AtomicRmwOp::TestAndSet => {
                // Handled separately in emit_atomic_rmw
                state.emit_fmt(format_args!("    mov {}, {}", dest_reg, val_reg));
            }
        }
    }
}
