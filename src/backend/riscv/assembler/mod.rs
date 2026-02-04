//! Native RISC-V assembler.
//!
//! Parses `.s` assembly text (as emitted by the RISC-V codegen) and produces
//! ELF `.o` object files, removing the dependency on `riscv64-linux-gnu-gcc`
//! for assembly.
//!
//! Architecture:
//! - `parser.rs`     – Tokenize + parse assembly text into `AsmStatement` items
//! - `encoder.rs`    – Encode RISC-V instructions into 32-bit machine words
//! - `compress.rs`   – RV64C compressed instruction support (32-bit → 16-bit)
//! - `elf_writer.rs` – Write ELF object files with sections, symbols, and relocations

pub mod parser;
pub mod encoder;
pub mod compress;
pub mod elf_writer;

use parser::parse_asm;
use elf_writer::ElfWriter;

/// Assemble RISC-V assembly text into an ELF object file.
///
/// This is the default assembler (used when the `gcc_assembler` feature is disabled).
pub fn assemble(asm_text: &str, output_path: &str) -> Result<(), String> {
    let statements = parse_asm(asm_text)?;
    let mut writer = ElfWriter::new();
    writer.process_statements(&statements)?;
    writer.write_elf(output_path)?;
    Ok(())
}
