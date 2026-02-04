//! ELF object file writer for AArch64.
//!
//! Takes parsed assembly statements and produces an ELF .o (relocatable) file
//! with proper sections, symbols, and relocations for AArch64/ELF64.

#![allow(dead_code)]

use std::collections::HashMap;
use super::parser::{AsmStatement, AsmDirective, SectionDirective, SymbolKind, SizeExpr, DataValue, Operand};
use super::encoder::{encode_instruction, EncodeResult, RelocType};

const EM_AARCH64: u16 = 183;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ELFOSABI_NONE: u8 = 0;
const ET_REL: u16 = 1;

// Section types
const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHT_NOTE: u32 = 7;

// Section flags
const SHF_WRITE: u64 = 1;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const SHF_MERGE: u64 = 0x10;
const SHF_STRINGS: u64 = 0x20;
const SHF_INFO_LINK: u64 = 0x40;
const SHF_TLS: u64 = 0x400;
const SHF_GROUP: u64 = 0x200;

// Symbol bindings
const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;

// Symbol types
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;
const STT_TLS: u8 = 6;

// Symbol visibility
const STV_DEFAULT: u8 = 0;
const STV_HIDDEN: u8 = 2;
const STV_PROTECTED: u8 = 3;
const STV_INTERNAL: u8 = 1;

/// An ELF section being built.
struct Section {
    name: String,
    sh_type: u32,
    sh_flags: u64,
    data: Vec<u8>,
    sh_addralign: u64,
    sh_entsize: u64,
    sh_link: u32,
    sh_info: u32,
    /// Relocations for this section
    relocs: Vec<ElfReloc>,
}

/// A relocation entry.
struct ElfReloc {
    offset: u64,
    reloc_type: u32,
    symbol_name: String,
    addend: i64,
}

/// A symbol entry.
struct ElfSymbol {
    name: String,
    value: u64,
    size: u64,
    binding: u8,
    sym_type: u8,
    visibility: u8,
    section_name: String,
}

/// The ELF writer state machine.
pub struct ElfWriter {
    /// Current section we're emitting into
    current_section: String,
    /// All sections being built
    sections: HashMap<String, Section>,
    /// Section order (for deterministic output)
    section_order: Vec<String>,
    /// Symbol table
    symbols: Vec<ElfSymbol>,
    /// Local labels -> (section, offset) for branch resolution
    labels: HashMap<String, (String, u64)>,
    /// Pending relocations that reference labels (need fixup)
    pending_branch_relocs: Vec<PendingReloc>,
    /// Pending symbol differences to resolve after all labels are known
    pending_sym_diffs: Vec<PendingSymDiff>,
    /// Current alignment state
    current_align: u64,
    /// Symbols that have been declared .globl
    global_symbols: HashMap<String, bool>,
    /// Symbols declared .weak
    weak_symbols: HashMap<String, bool>,
    /// Symbol types from .type directives
    symbol_types: HashMap<String, u8>,
    /// Symbol sizes from .size directives
    symbol_sizes: HashMap<String, u64>,
    /// Symbol visibility from .hidden/.protected/.internal
    symbol_visibility: HashMap<String, u8>,
}

struct PendingReloc {
    section: String,
    offset: u64,
    reloc_type: u32,
    symbol: String,
    addend: i64,
}

/// A pending symbol difference to be resolved after all labels are known.
struct PendingSymDiff {
    /// Section containing the data directive
    section: String,
    /// Offset within that section where the value should be written
    offset: u64,
    /// The positive symbol (A in A - B)
    sym_a: String,
    /// The negative symbol (B in A - B)
    sym_b: String,
    /// Extra addend
    extra_addend: i64,
    /// Size in bytes (4 or 8)
    size: usize,
}

impl ElfWriter {
    pub fn new() -> Self {
        Self {
            current_section: String::new(),
            sections: HashMap::new(),
            section_order: Vec::new(),
            symbols: Vec::new(),
            labels: HashMap::new(),
            pending_branch_relocs: Vec::new(),
            pending_sym_diffs: Vec::new(),
            current_align: 4,
            global_symbols: HashMap::new(),
            weak_symbols: HashMap::new(),
            symbol_types: HashMap::new(),
            symbol_sizes: HashMap::new(),
            symbol_visibility: HashMap::new(),
        }
    }

    fn ensure_section(&mut self, name: &str, sh_type: u32, sh_flags: u64, align: u64) {
        if !self.sections.contains_key(name) {
            self.sections.insert(name.to_string(), Section {
                name: name.to_string(),
                sh_type,
                sh_flags,
                data: Vec::new(),
                sh_addralign: align,
                sh_entsize: 0,
                sh_link: 0,
                sh_info: 0,
                relocs: Vec::new(),
            });
            self.section_order.push(name.to_string());
        }
    }

    fn current_offset(&self) -> u64 {
        self.sections.get(&self.current_section)
            .map(|s| s.data.len() as u64)
            .unwrap_or(0)
    }

    fn emit_bytes(&mut self, bytes: &[u8]) {
        if let Some(section) = self.sections.get_mut(&self.current_section) {
            section.data.extend_from_slice(bytes);
        }
    }

    fn emit_u32_le(&mut self, val: u32) {
        self.emit_bytes(&val.to_le_bytes());
    }

    fn add_reloc(&mut self, reloc_type: u32, symbol: String, addend: i64) {
        let offset = self.current_offset();
        let section = self.current_section.clone();
        if let Some(s) = self.sections.get_mut(&section) {
            s.relocs.push(ElfReloc {
                offset,
                reloc_type,
                symbol_name: symbol,
                addend,
            });
        }
    }

    /// Resolve pending symbol differences after all labels are known.
    fn resolve_sym_diffs(&mut self) -> Result<(), String> {
        let pending = std::mem::take(&mut self.pending_sym_diffs);
        for diff in &pending {
            let sym_a_info = self.labels.get(&diff.sym_a).cloned();
            let sym_b_info = self.labels.get(&diff.sym_b).cloned();

            match (sym_a_info, sym_b_info) {
                (Some((sec_a, off_a)), Some((sec_b, off_b))) => {
                    if sec_a == sec_b {
                        // Same section: resolve at assembly time by patching the data
                        let value = (off_a as i64) - (off_b as i64) + diff.extra_addend;
                        if let Some(section) = self.sections.get_mut(&diff.section) {
                            let off = diff.offset as usize;
                            if diff.size == 4 && off + 4 <= section.data.len() {
                                section.data[off..off + 4].copy_from_slice(&(value as i32).to_le_bytes());
                            } else if diff.size == 8 && off + 8 <= section.data.len() {
                                section.data[off..off + 8].copy_from_slice(&value.to_le_bytes());
                            }
                        }
                    } else {
                        // Cross-section: emit R_AARCH64_PREL32 referencing sym_a's section symbol
                        // PREL32 computes: S + A - P
                        // We want: (sec_a_base + off_a) - (sec_b_base + off_b) + extra_addend
                        // P = diff.section base + diff.offset
                        // If diff.section == sec_b:
                        //   want = (sec_a_base + off_a) - (sec_b_base + off_b) + extra_addend
                        //   PREL32 = sec_a_base + A - (sec_b_base + diff.offset)
                        //   so A = off_a + diff.offset - off_b + extra_addend
                        let addend = off_a as i64 + diff.offset as i64 - off_b as i64 + diff.extra_addend;
                        if let Some(section) = self.sections.get_mut(&diff.section) {
                            section.relocs.push(ElfReloc {
                                offset: diff.offset,
                                reloc_type: RelocType::Prel32.elf_type(),
                                symbol_name: sec_a.clone(),
                                addend,
                            });
                        }
                    }
                }
                _ => {
                    // Forward-referenced or external symbols: emit PREL32 with symbol name
                    // This shouldn't happen for jump tables, but handle gracefully
                    if let Some(section) = self.sections.get_mut(&diff.section) {
                        section.relocs.push(ElfReloc {
                            offset: diff.offset,
                            reloc_type: RelocType::Prel32.elf_type(),
                            symbol_name: diff.sym_a.clone(),
                            addend: diff.extra_addend,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn align_to(&mut self, align: u64) {
        if align <= 1 {
            return;
        }
        if let Some(section) = self.sections.get_mut(&self.current_section) {
            let current = section.data.len() as u64;
            let aligned = (current + align - 1) & !(align - 1);
            let padding = (aligned - current) as usize;
            if section.sh_flags & SHF_EXECINSTR != 0 {
                // NOP-fill for code sections: AArch64 NOP = 0xD503201F (little-endian)
                let nop: [u8; 4] = [0x1f, 0x20, 0x03, 0xd5];
                let full_nops = padding / 4;
                let remainder = padding % 4;
                for _ in 0..full_nops {
                    section.data.extend_from_slice(&nop);
                }
                // Fill any remaining bytes (shouldn't happen if alignment is >= 4)
                section.data.extend(std::iter::repeat(0u8).take(remainder));
            } else {
                section.data.extend(std::iter::repeat(0u8).take(padding));
            }
            if align > section.sh_addralign {
                section.sh_addralign = align;
            }
        }
    }

    /// Process all parsed assembly statements.
    pub fn process_statements(&mut self, statements: &[AsmStatement]) -> Result<(), String> {
        for stmt in statements {
            self.process_statement(stmt)?;
        }
        // Resolve symbol differences first (needs all labels to be known)
        self.resolve_sym_diffs()?;
        self.resolve_local_branches()?;
        Ok(())
    }

    fn process_statement(&mut self, stmt: &AsmStatement) -> Result<(), String> {
        match stmt {
            AsmStatement::Empty => Ok(()),

            AsmStatement::Label(name) => {
                // Record label position
                let section = self.current_section.clone();
                let offset = self.current_offset();
                self.labels.insert(name.clone(), (section, offset));
                Ok(())
            }

            AsmStatement::Directive(dir) => {
                self.process_directive(dir)
            }

            AsmStatement::Instruction { mnemonic, operands, raw_operands } => {
                self.process_instruction(mnemonic, operands, raw_operands)
            }
        }
    }

    fn process_directive(&mut self, dir: &AsmDirective) -> Result<(), String> {
        match dir {
            AsmDirective::Section(sec) => {
                self.process_section_directive(sec);
                Ok(())
            }

            AsmDirective::Global(sym) => {
                self.global_symbols.insert(sym.clone(), true);
                Ok(())
            }

            AsmDirective::Weak(sym) => {
                self.weak_symbols.insert(sym.clone(), true);
                Ok(())
            }

            AsmDirective::Hidden(sym) => {
                self.symbol_visibility.insert(sym.clone(), STV_HIDDEN);
                Ok(())
            }

            AsmDirective::Protected(sym) => {
                self.symbol_visibility.insert(sym.clone(), STV_PROTECTED);
                Ok(())
            }

            AsmDirective::Internal(sym) => {
                self.symbol_visibility.insert(sym.clone(), STV_INTERNAL);
                Ok(())
            }

            AsmDirective::SymbolType(sym, kind) => {
                let st = match kind {
                    SymbolKind::Function => STT_FUNC,
                    SymbolKind::Object => STT_OBJECT,
                    SymbolKind::TlsObject => STT_TLS,
                    SymbolKind::NoType => STT_NOTYPE,
                };
                self.symbol_types.insert(sym.clone(), st);
                Ok(())
            }

            AsmDirective::Size(sym, expr) => {
                match expr {
                    SizeExpr::CurrentMinusSymbol(label) => {
                        if let Some((section, label_offset)) = self.labels.get(label) {
                            if *section == self.current_section {
                                let current = self.current_offset();
                                let size = current - label_offset;
                                self.symbol_sizes.insert(sym.clone(), size);
                            }
                        }
                    }
                    SizeExpr::Constant(size) => {
                        self.symbol_sizes.insert(sym.clone(), *size);
                    }
                }
                Ok(())
            }

            AsmDirective::Align(bytes) => {
                self.align_to(*bytes);
                Ok(())
            }

            AsmDirective::Balign(bytes) => {
                self.align_to(*bytes);
                Ok(())
            }

            AsmDirective::Byte(vals) => {
                self.emit_bytes(vals);
                Ok(())
            }

            AsmDirective::Short(vals) => {
                for val in vals {
                    self.emit_bytes(&(*val as u16).to_le_bytes());
                }
                Ok(())
            }

            AsmDirective::Long(vals) => {
                self.emit_data_values(vals, 4)
            }

            AsmDirective::Quad(vals) => {
                self.emit_data_values(vals, 8)
            }

            AsmDirective::Zero(size, fill) => {
                self.emit_bytes(&vec![*fill; *size]);
                Ok(())
            }

            AsmDirective::Asciz(bytes) => {
                // Already includes null terminator from parser
                self.emit_bytes(bytes);
                Ok(())
            }

            AsmDirective::Ascii(bytes) => {
                self.emit_bytes(bytes);
                Ok(())
            }

            AsmDirective::Comm(sym, size, align) => {
                self.symbols.push(ElfSymbol {
                    name: sym.clone(),
                    value: *align, // For COMMON, value = alignment
                    size: *size,
                    binding: STB_GLOBAL,
                    sym_type: STT_OBJECT,
                    visibility: STV_DEFAULT,
                    section_name: "*COM*".to_string(),
                });
                Ok(())
            }

            AsmDirective::Local(_) => {
                // Symbols are local by default
                Ok(())
            }

            AsmDirective::Set(_name, _value) => {
                // TODO: implement properly
                Ok(())
            }

            AsmDirective::Cfi | AsmDirective::Ignored => Ok(()),
        }
    }

    /// Process a section directive with already-parsed fields.
    fn process_section_directive(&mut self, sec: &SectionDirective) {
        let sec_name = &sec.name;

        // Determine section type from parsed type string
        let sh_type = match sec.section_type.as_deref() {
            Some("@nobits") => SHT_NOBITS,
            Some("@note") => SHT_NOTE,
            _ => {
                // Default type based on section name
                if sec_name == ".bss" || sec_name.starts_with(".bss.") || sec_name.starts_with(".tbss") {
                    SHT_NOBITS
                } else {
                    SHT_PROGBITS
                }
            }
        };

        // Determine section flags from parsed flags string
        let mut sh_flags = 0u64;
        if let Some(flags) = &sec.flags {
            if flags.contains('a') { sh_flags |= SHF_ALLOC; }
            if flags.contains('w') { sh_flags |= SHF_WRITE; }
            if flags.contains('x') { sh_flags |= SHF_EXECINSTR; }
            if flags.contains('M') { sh_flags |= SHF_MERGE; }
            if flags.contains('S') { sh_flags |= SHF_STRINGS; }
            if flags.contains('T') { sh_flags |= SHF_TLS; }
            if flags.contains('G') { sh_flags |= SHF_GROUP; }
        }

        // Set default flags based on section name if none specified
        if sh_flags == 0 {
            sh_flags = default_section_flags(sec_name);
        }

        let align = if sec_name == ".text" { 4 } else { 1 };
        self.ensure_section(sec_name, sh_type, sh_flags, align);
        self.current_section = sec_name.clone();
    }

    /// Emit typed data values (Long or Quad) with proper relocations.
    fn emit_data_values(&mut self, vals: &[DataValue], size: usize) -> Result<(), String> {
        for val in vals {
            match val {
                DataValue::Integer(v) => {
                    if size == 4 {
                        self.emit_bytes(&(*v as u32).to_le_bytes());
                    } else {
                        self.emit_bytes(&(*v as u64).to_le_bytes());
                    }
                }
                DataValue::Symbol(sym) => {
                    let reloc_type = if size == 4 {
                        RelocType::Abs32.elf_type()
                    } else {
                        RelocType::Abs64.elf_type()
                    };
                    self.add_reloc(reloc_type, sym.clone(), 0);
                    if size == 4 {
                        self.emit_bytes(&0u32.to_le_bytes());
                    } else {
                        self.emit_bytes(&0u64.to_le_bytes());
                    }
                }
                DataValue::SymbolOffset(sym, addend) => {
                    let reloc_type = if size == 4 {
                        RelocType::Abs32.elf_type()
                    } else {
                        RelocType::Abs64.elf_type()
                    };
                    self.add_reloc(reloc_type, sym.clone(), *addend);
                    if size == 4 {
                        self.emit_bytes(&0u32.to_le_bytes());
                    } else {
                        self.emit_bytes(&0u64.to_le_bytes());
                    }
                }
                DataValue::SymbolDiff(sym_a, sym_b) => {
                    self.record_sym_diff(sym_a, sym_b, 0, size);
                }
                DataValue::SymbolDiffAddend(sym_a, sym_b, addend) => {
                    self.record_sym_diff(sym_a, sym_b, *addend, size);
                }
            }
        }
        Ok(())
    }

    /// Record a pending symbol difference for deferred resolution.
    fn record_sym_diff(&mut self, sym_a: &str, sym_b: &str, extra_addend: i64, size: usize) {
        let section = self.current_section.clone();
        let offset = self.current_offset();
        self.pending_sym_diffs.push(PendingSymDiff {
            section,
            offset,
            sym_a: sym_a.to_string(),
            sym_b: sym_b.to_string(),
            extra_addend,
            size,
        });
        // Emit placeholder bytes
        if size == 4 {
            self.emit_bytes(&0u32.to_le_bytes());
        } else {
            self.emit_bytes(&0u64.to_le_bytes());
        }
    }

    fn process_instruction(&mut self, mnemonic: &str, operands: &[Operand], raw_operands: &str) -> Result<(), String> {
        // Make sure we're in a text section
        if self.current_section.is_empty() {
            self.ensure_section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 4);
            self.current_section = ".text".to_string();
        }

        match encode_instruction(mnemonic, operands, raw_operands) {
            Ok(EncodeResult::Word(word)) => {
                self.emit_u32_le(word);
                Ok(())
            }
            Ok(EncodeResult::WordWithReloc { word, reloc }) => {
                // Check if it's a local label reference we can resolve
                let is_local = reloc.symbol.starts_with(".L") || reloc.symbol.starts_with(".l");

                if is_local {
                    // Store pending reloc - will be resolved after all labels are known
                    let offset = self.current_offset();
                    self.pending_branch_relocs.push(PendingReloc {
                        section: self.current_section.clone(),
                        offset,
                        reloc_type: reloc.reloc_type.elf_type(),
                        symbol: reloc.symbol.clone(),
                        addend: reloc.addend,
                    });
                    self.emit_u32_le(word);
                } else {
                    self.add_reloc(reloc.reloc_type.elf_type(), reloc.symbol, reloc.addend);
                    self.emit_u32_le(word);
                }
                Ok(())
            }
            Ok(EncodeResult::Words(words)) => {
                for word in words {
                    self.emit_u32_le(word);
                }
                Ok(())
            }
            Ok(EncodeResult::Skip) => Ok(()),
            Err(e) => {
                return Err(e);
            }
        }
    }

    /// Resolve local branch labels to PC-relative offsets.
    fn resolve_local_branches(&mut self) -> Result<(), String> {
        for reloc in &self.pending_branch_relocs {
            let (target_section, target_offset) = self.labels.get(&reloc.symbol)
                .ok_or_else(|| format!("undefined local label: {}", reloc.symbol))?
                .clone();

            if target_section != reloc.section {
                // Cross-section reference - convert local label to section symbol + offset
                // The linker doesn't know about local labels, so reference the section symbol
                // with addend = label offset within its section
                if let Some(section) = self.sections.get_mut(&reloc.section) {
                    section.relocs.push(ElfReloc {
                        offset: reloc.offset,
                        reloc_type: reloc.reloc_type,
                        symbol_name: target_section.clone(),
                        addend: target_offset as i64 + reloc.addend,
                    });
                }
                continue;
            }

            let pc_offset = (target_offset as i64) - (reloc.offset as i64) + reloc.addend;

            if let Some(section) = self.sections.get_mut(&reloc.section) {
                let instr_offset = reloc.offset as usize;
                if instr_offset + 4 > section.data.len() {
                    continue;
                }

                let mut word = u32::from_le_bytes([
                    section.data[instr_offset],
                    section.data[instr_offset + 1],
                    section.data[instr_offset + 2],
                    section.data[instr_offset + 3],
                ]);

                match reloc.reloc_type {
                    282 | 283 => {
                        // R_AARCH64_JUMP26 / R_AARCH64_CALL26
                        let imm26 = ((pc_offset >> 2) as u32) & 0x3FFFFFF;
                        word |= imm26;
                    }
                    280 => {
                        // R_AARCH64_CONDBR19
                        let imm19 = ((pc_offset >> 2) as u32) & 0x7FFFF;
                        word |= imm19 << 5;
                    }
                    279 => {
                        // R_AARCH64_TSTBR14
                        let imm14 = ((pc_offset >> 2) as u32) & 0x3FFF;
                        word |= imm14 << 5;
                    }
                    _ => {
                        // Unknown reloc type for local branch - leave as external
                        section.relocs.push(ElfReloc {
                            offset: reloc.offset,
                            reloc_type: reloc.reloc_type,
                            symbol_name: reloc.symbol.clone(),
                            addend: reloc.addend,
                        });
                        continue;
                    }
                }

                section.data[instr_offset..instr_offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        Ok(())
    }

    /// Resolve local label references in data relocations.
    ///
    /// When a data directive like `.xword .Lstr0` references a local label
    /// in a different section, the local label won't be in the symbol table.
    /// We need to convert these to section_symbol + offset_of_label_in_section,
    /// just like GCC's assembler does.
    fn resolve_local_data_relocs(&mut self) {
        let labels = &self.labels;
        for sec_name in &self.section_order.clone() {
            if let Some(section) = self.sections.get_mut(sec_name) {
                for reloc in &mut section.relocs {
                    // Check if this references a local label (starts with .L or .l)
                    if (reloc.symbol_name.starts_with(".L") || reloc.symbol_name.starts_with(".l"))
                        && !reloc.symbol_name.is_empty()
                    {
                        if let Some((label_section, label_offset)) = labels.get(&reloc.symbol_name) {
                            // Convert to section symbol + addend
                            reloc.addend += *label_offset as i64;
                            reloc.symbol_name = label_section.clone();
                        }
                    }
                }
            }
        }
    }

    /// Write the final ELF object file.
    pub fn write_elf(&mut self, output_path: &str) -> Result<(), String> {
        // Resolve local label references in data relocations before building symbol table
        self.resolve_local_data_relocs();
        // Build the symbol table from labels and directives
        self.build_symbol_table();

        let mut elf = Vec::new();

        // ── Collect section metadata ──
        // Section layout: NULL, then content sections, then rela sections, then .symtab, .strtab, .shstrtab

        let mut shstrtab = StringTable::new();
        let mut strtab = StringTable::new();

        // Add the null string
        shstrtab.add("");
        strtab.add("");

        // Content sections
        let content_sections: Vec<String> = self.section_order.clone();

        // Build symbol table entries
        let mut sym_entries: Vec<SymEntry> = Vec::new();
        // First entry is always NULL
        sym_entries.push(SymEntry {
            st_name: 0,
            st_info: 0,
            st_other: 0,
            st_shndx: 0,
            st_value: 0,
            st_size: 0,
        });

        // Section symbols (one per content section)
        for (i, sec_name) in content_sections.iter().enumerate() {
            strtab.add(sec_name);
            sym_entries.push(SymEntry {
                st_name: strtab.offset_of(sec_name),
                st_info: (STB_LOCAL << 4) | STT_SECTION,
                st_other: 0,
                st_shndx: (i + 1) as u16, // section index (1-based, 0 is NULL)
                st_value: 0,
                st_size: 0,
            });
        }

        // Local symbols (from labels, excluding .L* which are local labels)
        let mut local_syms: Vec<&ElfSymbol> = Vec::new();
        let mut global_syms: Vec<&ElfSymbol> = Vec::new();

        for sym in &self.symbols {
            if sym.binding == STB_LOCAL {
                local_syms.push(sym);
            } else {
                global_syms.push(sym);
            }
        }

        let first_global_idx = sym_entries.len() + local_syms.len();

        for sym in &local_syms {
            let name_offset = strtab.add(&sym.name);
            let shndx = section_index(&sym.section_name, &content_sections);
            sym_entries.push(SymEntry {
                st_name: name_offset,
                st_info: (sym.binding << 4) | sym.sym_type,
                st_other: sym.visibility,
                st_shndx: shndx,
                st_value: sym.value,
                st_size: sym.size,
            });
        }

        for sym in &global_syms {
            let name_offset = strtab.add(&sym.name);
            let shndx = section_index(&sym.section_name, &content_sections);
            sym_entries.push(SymEntry {
                st_name: name_offset,
                st_info: (sym.binding << 4) | sym.sym_type,
                st_other: sym.visibility,
                st_shndx: shndx,
                st_value: sym.value,
                st_size: sym.size,
            });
        }

        // Add shstrtab names
        shstrtab.add("");
        for sec_name in &content_sections {
            shstrtab.add(sec_name);
        }
        shstrtab.add(".symtab");
        shstrtab.add(".strtab");
        shstrtab.add(".shstrtab");

        // Build rela section names and add them
        let mut rela_sections: Vec<String> = Vec::new();
        for sec_name in &content_sections {
            if let Some(section) = self.sections.get(sec_name) {
                if !section.relocs.is_empty() {
                    let rela_name = format!(".rela{}", sec_name);
                    shstrtab.add(&rela_name);
                    rela_sections.push(rela_name);
                }
            }
        }

        // ── Calculate layout ──
        let ehdr_size = 64usize; // ELF64 header
        let mut offset = ehdr_size;

        // Content section offsets
        let mut section_offsets: Vec<usize> = Vec::new();
        for sec_name in &content_sections {
            let section = self.sections.get(sec_name).unwrap();
            let align = section.sh_addralign.max(1) as usize;
            offset = (offset + align - 1) & !(align - 1);
            section_offsets.push(offset);
            if section.sh_type != SHT_NOBITS {
                offset += section.data.len();
            }
        }

        // Rela section offsets
        let mut rela_offsets: Vec<usize> = Vec::new();
        for sec_name in &content_sections {
            if let Some(section) = self.sections.get(sec_name) {
                if !section.relocs.is_empty() {
                    offset = (offset + 7) & !7; // 8-byte align
                    rela_offsets.push(offset);
                    offset += section.relocs.len() * 24; // sizeof(Elf64_Rela) = 24
                }
            }
        }

        // Symtab offset
        offset = (offset + 7) & !7;
        let symtab_offset = offset;
        let symtab_size = sym_entries.len() * 24; // sizeof(Elf64_Sym) = 24
        offset += symtab_size;

        // Strtab offset
        let strtab_offset = offset;
        let strtab_data = strtab.data();
        offset += strtab_data.len();

        // Shstrtab offset
        let shstrtab_offset = offset;
        let shstrtab_data = shstrtab.data();
        offset += shstrtab_data.len();

        // Section headers offset (align to 8)
        offset = (offset + 7) & !7;
        let shdr_offset = offset;

        // Section header count:
        // 1 (NULL) + content_sections.len() + rela_sections.len() + 3 (.symtab, .strtab, .shstrtab)
        let num_sections = 1 + content_sections.len() + rela_sections.len() + 3;
        let shstrtab_idx = num_sections - 1;

        // ── Write ELF header ──
        elf.extend_from_slice(&[0x7f, b'E', b'L', b'F']); // e_ident magic
        elf.push(ELFCLASS64);
        elf.push(ELFDATA2LSB);
        elf.push(EV_CURRENT);
        elf.push(ELFOSABI_NONE);
        elf.extend_from_slice(&[0u8; 8]); // padding
        elf.extend_from_slice(&ET_REL.to_le_bytes()); // e_type
        elf.extend_from_slice(&EM_AARCH64.to_le_bytes()); // e_machine
        elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_entry
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
        elf.extend_from_slice(&(shdr_offset as u64).to_le_bytes()); // e_shoff
        elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        elf.extend_from_slice(&(ehdr_size as u16).to_le_bytes()); // e_ehsize
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
        elf.extend_from_slice(&(num_sections as u16).to_le_bytes()); // e_shnum
        elf.extend_from_slice(&(shstrtab_idx as u16).to_le_bytes()); // e_shstrndx

        assert_eq!(elf.len(), ehdr_size);

        // ── Write content section data ──
        for (i, sec_name) in content_sections.iter().enumerate() {
            let section = self.sections.get(sec_name).unwrap();
            // Pad to alignment
            while elf.len() < section_offsets[i] {
                elf.push(0);
            }
            if section.sh_type != SHT_NOBITS {
                elf.extend_from_slice(&section.data);
            }
        }

        // ── Write rela section data ──
        let symtab_shndx = 1 + content_sections.len() + rela_sections.len(); // index of .symtab in section header table
        let mut rela_idx = 0;
        for sec_name in &content_sections {
            if let Some(section) = self.sections.get(sec_name) {
                if !section.relocs.is_empty() {
                    while elf.len() < rela_offsets[rela_idx] {
                        elf.push(0);
                    }
                    for reloc in &section.relocs {
                        // Find symbol index
                        let sym_idx = self.find_symbol_index(&reloc.symbol_name, &sym_entries, &strtab, &content_sections);
                        // Write Elf64_Rela
                        elf.extend_from_slice(&reloc.offset.to_le_bytes());
                        let r_info = ((sym_idx as u64) << 32) | (reloc.reloc_type as u64);
                        elf.extend_from_slice(&r_info.to_le_bytes());
                        elf.extend_from_slice(&reloc.addend.to_le_bytes());
                    }
                    rela_idx += 1;
                }
            }
        }

        // ── Write symtab ──
        while elf.len() < symtab_offset {
            elf.push(0);
        }
        for sym in &sym_entries {
            elf.extend_from_slice(&sym.st_name.to_le_bytes());
            elf.push(sym.st_info);
            elf.push(sym.st_other);
            elf.extend_from_slice(&sym.st_shndx.to_le_bytes());
            elf.extend_from_slice(&sym.st_value.to_le_bytes());
            elf.extend_from_slice(&sym.st_size.to_le_bytes());
        }

        // ── Write strtab ──
        assert_eq!(elf.len(), strtab_offset);
        elf.extend_from_slice(&strtab_data);

        // ── Write shstrtab ──
        assert_eq!(elf.len(), shstrtab_offset);
        elf.extend_from_slice(&shstrtab_data);

        // ── Write section headers ──
        while elf.len() < shdr_offset {
            elf.push(0);
        }

        // SHT_NULL entry
        write_shdr(&mut elf, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);

        // Content sections
        for (i, sec_name) in content_sections.iter().enumerate() {
            let section = self.sections.get(sec_name).unwrap();
            let sh_name = shstrtab.offset_of(sec_name);
            let sh_offset = if section.sh_type == SHT_NOBITS { 0 } else { section_offsets[i] as u64 };
            let sh_size = section.data.len() as u64;
            write_shdr(&mut elf, sh_name, section.sh_type, section.sh_flags,
                       0, sh_offset, sh_size, 0, 0, section.sh_addralign, section.sh_entsize);
        }

        // Rela sections
        rela_idx = 0;
        for (i, sec_name) in content_sections.iter().enumerate() {
            if let Some(section) = self.sections.get(sec_name) {
                if !section.relocs.is_empty() {
                    let rela_name = format!(".rela{}", sec_name);
                    let sh_name = shstrtab.offset_of(&rela_name);
                    let sh_offset = rela_offsets[rela_idx] as u64;
                    let sh_size = (section.relocs.len() * 24) as u64;
                    let sh_link = symtab_shndx as u32;
                    let sh_info = (i + 1) as u32; // Index of the section these relocs apply to
                    write_shdr(&mut elf, sh_name, SHT_RELA, SHF_INFO_LINK,
                               0, sh_offset, sh_size, sh_link, sh_info, 8, 24);
                    rela_idx += 1;
                }
            }
        }

        // .symtab
        let symtab_name = shstrtab.offset_of(".symtab");
        let strtab_shndx = symtab_shndx + 1;
        write_shdr(&mut elf, symtab_name, SHT_SYMTAB, 0,
                   0, symtab_offset as u64, symtab_size as u64,
                   strtab_shndx as u32, first_global_idx as u32, 8, 24);

        // .strtab
        let strtab_name = shstrtab.offset_of(".strtab");
        write_shdr(&mut elf, strtab_name, SHT_STRTAB, 0,
                   0, strtab_offset as u64, strtab_data.len() as u64, 0, 0, 1, 0);

        // .shstrtab
        let shstrtab_name = shstrtab.offset_of(".shstrtab");
        write_shdr(&mut elf, shstrtab_name, SHT_STRTAB, 0,
                   0, shstrtab_offset as u64, shstrtab_data.len() as u64, 0, 0, 1, 0);

        // Write to file
        std::fs::write(output_path, &elf)
            .map_err(|e| format!("failed to write ELF file: {}", e))?;

        Ok(())
    }

    /// Build the symbol table from collected labels and directives.
    fn build_symbol_table(&mut self) {
        // Collect all defined labels as symbols
        let labels = self.labels.clone();
        for (name, (section, offset)) in &labels {
            // Skip local labels (.L*)
            if name.starts_with(".L") || name.starts_with(".l") {
                continue;
            }

            let binding = if self.weak_symbols.contains_key(name) {
                STB_WEAK
            } else if self.global_symbols.contains_key(name) {
                STB_GLOBAL
            } else {
                STB_LOCAL
            };

            let sym_type = self.symbol_types.get(name).copied().unwrap_or(STT_NOTYPE);
            let size = self.symbol_sizes.get(name).copied().unwrap_or(0);
            let visibility = self.symbol_visibility.get(name).copied().unwrap_or(STV_DEFAULT);

            self.symbols.push(ElfSymbol {
                name: name.clone(),
                value: *offset,
                size,
                binding,
                sym_type,
                visibility,
                section_name: section.clone(),
            });
        }

        // Add undefined symbols (referenced but not defined)
        let mut referenced: HashMap<String, bool> = HashMap::new();
        for sec in self.sections.values() {
            for reloc in &sec.relocs {
                if !reloc.symbol_name.starts_with(".L") && !reloc.symbol_name.starts_with(".l") {
                    referenced.insert(reloc.symbol_name.clone(), true);
                }
            }
        }

        let defined: HashMap<String, bool> = self.symbols.iter()
            .map(|s| (s.name.clone(), true))
            .collect();

        for (name, _) in &referenced {
            // Skip section names - they already have section symbols in the symtab
            if self.sections.contains_key(name) {
                continue;
            }
            if !defined.contains_key(name) {
                let binding = if self.weak_symbols.contains_key(name) {
                    STB_WEAK
                } else {
                    STB_GLOBAL
                };
                let sym_type = self.symbol_types.get(name).copied().unwrap_or(STT_NOTYPE);
                let visibility = self.symbol_visibility.get(name).copied().unwrap_or(STV_DEFAULT);

                self.symbols.push(ElfSymbol {
                    name: name.clone(),
                    value: 0,
                    size: 0,
                    binding,
                    sym_type,
                    visibility,
                    section_name: "*UND*".to_string(),
                });
            }
        }
    }

    fn find_symbol_index(&self, name: &str, sym_entries: &[SymEntry], strtab: &StringTable, content_sections: &[String]) -> u32 {
        // First check if it's a section symbol
        for (i, sec_name) in content_sections.iter().enumerate() {
            if sec_name == name {
                return (i + 1) as u32; // +1 for NULL entry
            }
        }

        // Then check named symbols
        let name_offset = strtab.offset_of(name);
        for (i, entry) in sym_entries.iter().enumerate() {
            if entry.st_name == name_offset && entry.st_info & 0xF != STT_SECTION {
                return i as u32;
            }
        }

        // Not found - return 0 (undefined)
        0
    }
}

/// Write an Elf64_Shdr entry.
fn write_shdr(
    buf: &mut Vec<u8>,
    sh_name: u32, sh_type: u32, sh_flags: u64,
    sh_addr: u64, sh_offset: u64, sh_size: u64,
    sh_link: u32, sh_info: u32, sh_addralign: u64, sh_entsize: u64,
) {
    buf.extend_from_slice(&sh_name.to_le_bytes());
    buf.extend_from_slice(&sh_type.to_le_bytes());
    buf.extend_from_slice(&sh_flags.to_le_bytes());
    buf.extend_from_slice(&sh_addr.to_le_bytes());
    buf.extend_from_slice(&sh_offset.to_le_bytes());
    buf.extend_from_slice(&sh_size.to_le_bytes());
    buf.extend_from_slice(&sh_link.to_le_bytes());
    buf.extend_from_slice(&sh_info.to_le_bytes());
    buf.extend_from_slice(&sh_addralign.to_le_bytes());
    buf.extend_from_slice(&sh_entsize.to_le_bytes());
}

/// A symbol table entry for writing.
struct SymEntry {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}

/// Simple string table builder.
struct StringTable {
    data: Vec<u8>,
    offsets: HashMap<String, u32>,
}

impl StringTable {
    fn new() -> Self {
        Self {
            data: vec![0], // Start with null byte
            offsets: HashMap::new(),
        }
    }

    fn add(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.offsets.get(s) {
            return offset;
        }
        if s.is_empty() {
            return 0;
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);
        self.offsets.insert(s.to_string(), offset);
        offset
    }

    fn offset_of(&self, s: &str) -> u32 {
        self.offsets.get(s).copied().unwrap_or(0)
    }

    fn data(&self) -> Vec<u8> {
        self.data.clone()
    }
}

/// Map section name to its section header index, handling special names.
fn section_index(section_name: &str, content_sections: &[String]) -> u16 {
    if section_name == "*COM*" {
        0xFFF2u16 // SHN_COMMON
    } else if section_name == "*UND*" || section_name.is_empty() {
        0u16 // SHN_UNDEF
    } else {
        content_sections.iter().position(|s| s == section_name)
            .map(|i| (i + 1) as u16)
            .unwrap_or(0)
    }
}

/// Return default section flags based on section name.
fn default_section_flags(name: &str) -> u64 {
    if name == ".text" || name.starts_with(".text.") {
        SHF_ALLOC | SHF_EXECINSTR
    } else if name == ".data" || name.starts_with(".data.") {
        SHF_ALLOC | SHF_WRITE
    } else if name == ".bss" || name.starts_with(".bss.") {
        SHF_ALLOC | SHF_WRITE
    } else if name == ".rodata" || name.starts_with(".rodata.") {
        SHF_ALLOC
    } else if name.starts_with(".note") {
        SHF_ALLOC
    } else if name.starts_with(".tdata") {
        SHF_ALLOC | SHF_WRITE | SHF_TLS
    } else if name.starts_with(".tbss") {
        SHF_ALLOC | SHF_WRITE | SHF_TLS
    } else if name.starts_with(".init") || name.starts_with(".fini") {
        SHF_ALLOC | SHF_EXECINSTR
    } else {
        0
    }
}

