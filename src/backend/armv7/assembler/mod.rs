//! Native ARMv7 assembler.
//!
//! Parses `.s` assembly text (as emitted by the ARMv7 codegen) and produces
//! ELF `.o` object files, removing the dependency on `arm-linux-gnueabihf-gcc`
//! for assembly.
//!
//! Architecture:
//! - `parser.rs`     – Tokenize + parse assembly text into `AsmStatement` items
//! - `encoder/`      – Encode ARM instructions into 32-bit machine words
//! - `elf_writer.rs` – Write ELF32 object files with sections, symbols, and relocations

pub mod parser;
pub mod encoder;
pub mod elf_writer;

use std::collections::HashMap;
use parser::{parse_asm, AsmStatement};
use elf_writer::ElfWriter;

/// Assemble ARMv7 assembly text into an ELF object file.
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

/// Check if a string is a numeric forward/backward reference.
fn parse_numeric_ref(name: &str) -> Option<(&str, bool)> {
    if name.len() < 2 { return None; }
    let last = name.as_bytes()[name.len() - 1];
    let num_part = &name[..name.len() - 1];
    if !num_part.bytes().all(|b| b.is_ascii_digit()) { return None; }
    match last {
        b'f' | b'F' => Some((num_part, true)),
        b'b' | b'B' => Some((num_part, false)),
        _ => None,
    }
}

fn resolve_numeric_name(
    name: &str, current_idx: usize, defs: &HashMap<String, Vec<(usize, String)>>
) -> Option<String> {
    let (num, is_forward) = parse_numeric_ref(name)?;
    let def_list = defs.get(num)?;
    if is_forward {
        def_list.iter().find(|(idx, _)| *idx > current_idx).map(|(_, name)| name.clone())
    } else {
        def_list.iter().rev().find(|(idx, _)| *idx < current_idx).map(|(_, name)| name.clone())
    }
}

fn resolve_numeric_labels(statements: &[AsmStatement]) -> Vec<AsmStatement> {
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

    if defs.is_empty() { return statements.to_vec(); }

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
                let new_ops: Vec<parser::Operand> = operands.iter().map(|op| {
                    match op {
                        parser::Operand::Symbol(name) => {
                            if let Some(resolved) = resolve_numeric_name(name, i, &defs) {
                                parser::Operand::Symbol(resolved)
                            } else { op.clone() }
                        }
                        parser::Operand::Label(name) => {
                            if let Some(resolved) = resolve_numeric_name(name, i, &defs) {
                                parser::Operand::Label(resolved)
                            } else { op.clone() }
                        }
                        _ => op.clone(),
                    }
                }).collect();
                result.push(AsmStatement::Instruction {
                    mnemonic: mnemonic.clone(),
                    operands: new_ops,
                    raw_operands: raw_operands.clone(),
                });
            }
            _ => result.push(stmt.clone()),
        }
    }
    result
}
