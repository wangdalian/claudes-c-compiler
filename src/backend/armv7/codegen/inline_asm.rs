//! ARMv7 inline assembly support.

use crate::ir::reexports::{BlockId, Operand, Value};
use crate::common::types::IrType;
use crate::{emit};
use super::emit::Armv7Codegen;

impl Armv7Codegen {
    pub(super) fn emit_inline_asm_impl(
        &mut self,
        template: &str,
        outputs: &[(String, Value, Option<String>)],
        inputs: &[(String, Operand, Option<String>)],
        clobbers: &[String],
        _operand_types: &[IrType],
        _goto_labels: &[(String, BlockId)],
        _input_symbols: &[Option<String>],
    ) {
        // Save clobbered registers
        let mut saved_regs = Vec::new();
        for clobber in clobbers {
            let reg = clobber.trim_matches(|c| c == '{' || c == '}' || c == '~');
            if !reg.is_empty() && reg != "memory" && reg != "cc" {
                saved_regs.push(reg.to_string());
            }
        }
        if !saved_regs.is_empty() {
            emit!(self.state, "    push {{{}}}", saved_regs.join(", "));
        }

        // Load input operands
        for (constraint, op, _sym) in inputs.iter() {
            let reg = match constraint.as_str() {
                "r" | "{r0}" => "r0",
                "{r1}" => "r1",
                "{r2}" => "r2",
                "{r3}" => "r3",
                _ => "r0",
            };
            self.operand_to_r0(op);
            if reg != "r0" {
                emit!(self.state, "    mov {}, r0", reg);
            }
        }

        // Emit the template (simple substitution)
        let asm_text = template
            .replace("\\n", "\n")
            .replace("\\t", "\t");
        for line in asm_text.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                emit!(self.state, "    {}", trimmed);
            }
        }

        // Store output operands
        for (constraint, dest, _label) in outputs {
            let reg = match constraint.as_str() {
                "=r" | "={r0}" => "r0",
                "={r1}" => "r1",
                _ => "r0",
            };
            if reg != "r0" {
                emit!(self.state, "    mov r0, {}", reg);
            }
            self.store_r0_to(dest);
        }

        // Restore clobbered registers
        if !saved_regs.is_empty() {
            emit!(self.state, "    pop {{{}}}", saved_regs.join(", "));
        }

        self.state.reg_cache.invalidate_all();
    }
}
