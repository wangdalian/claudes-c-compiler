//! Native AArch64 assembler.
//!
//! Parses `.s` assembly text (as emitted by the AArch64 codegen) and produces
//! ELF `.o` object files, removing the dependency on `aarch64-linux-gnu-gcc`
//! for assembly.
//!
//! Architecture:
//! - `parser.rs`     – Tokenize + parse assembly text into `AsmStatement` items
//! - `encoder.rs`    – Encode AArch64 instructions into 32-bit machine words
//! - `elf_writer.rs` – Write ELF object files with sections, symbols, and relocations

pub mod parser;
pub mod encoder;
pub mod elf_writer;

use std::collections::HashMap;
use parser::{parse_asm, AsmStatement, Operand, AsmDirective, DataValue};
use elf_writer::ElfWriter;

/// Assemble AArch64 assembly text into an ELF object file.
///
/// This is the main entry point, called when MY_ASM is set to use the
/// built-in assembler instead of the external `aarch64-linux-gnu-gcc`.
pub fn assemble(asm_text: &str, output_path: &str) -> Result<(), String> {
    let statements = parse_asm(asm_text)?;
    let statements = resolve_numeric_labels(&statements);
    let mut writer = ElfWriter::new();
    writer.process_statements(&statements)?;
    writer.write_elf(output_path)?;
    Ok(())
}

/// Check if a label name is a GNU assembler numeric label (e.g., "1", "42").
fn is_numeric_label(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit())
}

/// Check if a string is a numeric forward/backward reference like "1f" or "2b".
/// Returns Some((number_str, is_forward)) if it is, None otherwise.
fn parse_numeric_ref(name: &str) -> Option<(&str, bool)> {
    if name.len() < 2 {
        return None;
    }
    let last = name.as_bytes()[name.len() - 1];
    let num_part = &name[..name.len() - 1];
    if !num_part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    match last {
        b'f' | b'F' => Some((num_part, true)),
        b'b' | b'B' => Some((num_part, false)),
        _ => None,
    }
}

/// Resolve a numeric label reference to its unique name.
fn resolve_numeric_name(
    name: &str,
    current_idx: usize,
    defs: &HashMap<String, Vec<(usize, String)>>,
) -> Option<String> {
    let (num, is_forward) = parse_numeric_ref(name)?;
    let def_list = defs.get(num)?;

    if is_forward {
        def_list.iter()
            .find(|(idx, _)| *idx > current_idx)
            .map(|(_, name)| name.clone())
    } else {
        def_list.iter()
            .rev()
            .find(|(idx, _)| *idx < current_idx)
            .map(|(_, name)| name.clone())
    }
}

/// Resolve numeric local labels (1:, 2:, etc.) and their references (1f, 1b)
/// into unique internal label names.
///
/// GNU assembler numeric labels can be defined multiple times. Each forward
/// reference `Nf` refers to the next definition of `N`, and each backward
/// reference `Nb` refers to the most recent definition of `N`.
fn resolve_numeric_labels(statements: &[AsmStatement]) -> Vec<AsmStatement> {
    // First pass: find all numeric label definitions and assign unique names.
    let mut defs: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    let mut unique_counter: HashMap<String, usize> = HashMap::new();

    for (i, stmt) in statements.iter().enumerate() {
        if let AsmStatement::Label(name) = stmt {
            if is_numeric_label(name) {
                let count = unique_counter.entry(name.clone()).or_insert(0);
                let unique_name = format!(".Lnum_{}_{}", name, *count);
                *count += 1;
                defs.entry(name.clone()).or_default().push((i, unique_name));
            }
        }
    }

    // If no numeric labels found, return original
    if defs.is_empty() {
        return statements.to_vec();
    }

    // Second pass: resolve all references
    let mut result = Vec::with_capacity(statements.len());
    for (i, stmt) in statements.iter().enumerate() {
        match stmt {
            AsmStatement::Label(name) if is_numeric_label(name) => {
                if let Some(def_list) = defs.get(name) {
                    if let Some((_, unique_name)) = def_list.iter().find(|(idx, _)| *idx == i) {
                        result.push(AsmStatement::Label(unique_name.clone()));
                        continue;
                    }
                }
                result.push(stmt.clone());
            }
            AsmStatement::Instruction { mnemonic, operands, raw_operands } => {
                let new_ops: Vec<Operand> = operands.iter().map(|op| {
                    resolve_numeric_operand(op, i, &defs)
                }).collect();
                result.push(AsmStatement::Instruction {
                    mnemonic: mnemonic.clone(),
                    operands: new_ops,
                    raw_operands: raw_operands.clone(),
                });
            }
            AsmStatement::Directive(dir) => {
                result.push(AsmStatement::Directive(resolve_numeric_directive(dir, i, &defs)));
            }
            _ => result.push(stmt.clone()),
        }
    }

    result
}

/// Resolve numeric label references in a single operand.
fn resolve_numeric_operand(
    op: &Operand,
    current_idx: usize,
    defs: &HashMap<String, Vec<(usize, String)>>,
) -> Operand {
    match op {
        Operand::Symbol(name) => {
            if let Some(resolved) = resolve_numeric_name(name, current_idx, defs) {
                Operand::Symbol(resolved)
            } else {
                op.clone()
            }
        }
        Operand::Label(name) => {
            if let Some(resolved) = resolve_numeric_name(name, current_idx, defs) {
                Operand::Label(resolved)
            } else {
                op.clone()
            }
        }
        Operand::SymbolOffset(name, off) => {
            if let Some(resolved) = resolve_numeric_name(name, current_idx, defs) {
                Operand::SymbolOffset(resolved, *off)
            } else {
                op.clone()
            }
        }
        _ => op.clone(),
    }
}

/// Resolve numeric label references in data directives.
fn resolve_numeric_directive(
    dir: &AsmDirective,
    current_idx: usize,
    defs: &HashMap<String, Vec<(usize, String)>>,
) -> AsmDirective {
    match dir {
        AsmDirective::Long(vals) => {
            AsmDirective::Long(resolve_numeric_data_values(vals, current_idx, defs))
        }
        AsmDirective::Quad(vals) => {
            AsmDirective::Quad(resolve_numeric_data_values(vals, current_idx, defs))
        }
        _ => dir.clone(),
    }
}

/// Resolve numeric label references in data values.
fn resolve_numeric_data_values(
    vals: &[DataValue],
    current_idx: usize,
    defs: &HashMap<String, Vec<(usize, String)>>,
) -> Vec<DataValue> {
    vals.iter().map(|v| {
        match v {
            DataValue::Symbol(name) => {
                if let Some(resolved) = resolve_numeric_name(name, current_idx, defs) {
                    DataValue::Symbol(resolved)
                } else {
                    v.clone()
                }
            }
            DataValue::SymbolOffset(name, off) => {
                if let Some(resolved) = resolve_numeric_name(name, current_idx, defs) {
                    DataValue::SymbolOffset(resolved, *off)
                } else {
                    v.clone()
                }
            }
            DataValue::SymbolDiff(a, b) => {
                let new_a = resolve_numeric_name(a, current_idx, defs).unwrap_or_else(|| a.clone());
                let new_b = resolve_numeric_name(b, current_idx, defs).unwrap_or_else(|| b.clone());
                DataValue::SymbolDiff(new_a, new_b)
            }
            DataValue::SymbolDiffAddend(a, b, add) => {
                let new_a = resolve_numeric_name(a, current_idx, defs).unwrap_or_else(|| a.clone());
                let new_b = resolve_numeric_name(b, current_idx, defs).unwrap_or_else(|| b.clone());
                DataValue::SymbolDiffAddend(new_a, new_b, *add)
            }
            _ => v.clone(),
        }
    }).collect()
}
