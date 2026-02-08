//! ARMv7 ELF object file writer.
//!
//! Uses the shared `elf::object_writer` infrastructure to produce ELFCLASS32
//! EM_ARM relocatable object files.

use std::collections::HashMap;
use crate::backend::elf::*;
use super::parser::{AsmStatement, AsmDirective, DataValue, Operand, MemOffset};
use super::encoder;

// ARM relocation types
const R_ARM_ABS32: u32 = 2;
const R_ARM_REL32: u32 = 3;
const R_ARM_CALL: u32 = 28;
const R_ARM_JUMP24: u32 = 29;
const R_ARM_MOVW_ABS_NC: u32 = 43;
const R_ARM_MOVT_ABS: u32 = 44;
const R_ARM_GOT32: u32 = 26;
const R_ARM_PLT32: u32 = 27;

pub struct ElfWriter {
    sections: Vec<ObjSection>,
    symbols: Vec<ObjSymbol>,
    current_section: usize,
    /// Map from label name to (section_index, offset_in_section)
    labels: HashMap<String, (usize, u64)>,
    /// Set/equ symbol definitions
    set_symbols: HashMap<String, String>,
    /// Section name -> index in sections vec
    section_map: HashMap<String, usize>,
    /// Section stack for pushsection/popsection
    section_stack: Vec<usize>,
    /// Pending branch relocations to resolve
    pending_branches: Vec<PendingBranch>,
}

struct PendingBranch {
    section_idx: usize,
    offset: usize,
    target: String,
    is_link: bool,
    cond: u32,
}

impl ElfWriter {
    pub fn new() -> Self {
        let mut writer = Self {
            sections: Vec::new(),
            symbols: Vec::new(),
            current_section: 0,
            labels: HashMap::new(),
            set_symbols: HashMap::new(),
            section_map: HashMap::new(),
            section_stack: Vec::new(),
            pending_branches: Vec::new(),
        };
        // Create default .text section
        writer.get_or_create_section(".text", SHT_PROGBITS, (SHF_ALLOC | SHF_EXECINSTR) as u64, 4);
        writer
    }

    fn get_or_create_section(&mut self, name: &str, sh_type: u32, flags: u64, align: u64) -> usize {
        if let Some(&idx) = self.section_map.get(name) {
            return idx;
        }
        let idx = self.sections.len();
        self.sections.push(ObjSection {
            name: name.to_string(),
            sh_type,
            sh_flags: flags,
            data: Vec::new(),
            sh_addralign: align,
            relocs: Vec::new(),
            comdat_group: None,
        });
        self.section_map.insert(name.to_string(), idx);
        idx
    }

    fn current_section_mut(&mut self) -> &mut ObjSection {
        &mut self.sections[self.current_section]
    }

    fn current_offset(&self) -> usize {
        self.sections[self.current_section].data.len()
    }

    fn emit_word(&mut self, word: u32) {
        let bytes = word.to_le_bytes();
        self.current_section_mut().data.extend_from_slice(&bytes);
    }

    fn switch_section_by_name(&mut self, name: &str) {
        // Parse section flags from the name
        let (sec_name, sh_type, flags) = parse_section_spec(name);
        self.current_section = self.get_or_create_section(&sec_name, sh_type, flags, 4);
    }

    pub fn process_statements(&mut self, statements: &[AsmStatement]) -> Result<(), String> {
        for stmt in statements {
            self.process_statement(stmt)?;
        }
        self.resolve_branches()?;
        Ok(())
    }

    fn process_statement(&mut self, stmt: &AsmStatement) -> Result<(), String> {
        match stmt {
            AsmStatement::Label(name) => {
                let sec = self.current_section;
                let offset = self.current_offset() as u64;
                self.labels.insert(name.clone(), (sec, offset));
                // Update existing symbol (from .globl/.type directives) or create new local
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    // Preserve binding and sym_type from .globl/.type directives
                    sym.section_name = self.sections[sec].name.clone();
                    sym.value = offset;
                } else {
                    self.symbols.push(ObjSymbol {
                        name: name.clone(),
                        section_name: self.sections[sec].name.clone(),
                        value: offset,
                        size: 0,
                        binding: STB_LOCAL,
                        sym_type: STT_NOTYPE,
                        visibility: STV_DEFAULT,
                    });
                }
            }
            AsmStatement::Instruction { mnemonic, operands, raw_operands } => {
                self.encode_instruction(mnemonic, operands, raw_operands)?;
            }
            AsmStatement::Directive(dir) => {
                self.process_directive(dir)?;
            }
            AsmStatement::Empty => {}
        }
        Ok(())
    }

    fn process_directive(&mut self, dir: &AsmDirective) -> Result<(), String> {
        match dir {
            AsmDirective::Section(spec) => self.switch_section_by_name(spec),
            AsmDirective::PushSection(spec) => {
                self.section_stack.push(self.current_section);
                self.switch_section_by_name(spec);
            }
            AsmDirective::PopSection => {
                if let Some(prev) = self.section_stack.pop() {
                    self.current_section = prev;
                }
            }
            AsmDirective::Previous => {
                // Switch to previous section (simplified: just use .text)
                if let Some(&idx) = self.section_map.get(".text") {
                    self.current_section = idx;
                }
            }
            AsmDirective::Text => {
                self.current_section = self.get_or_create_section(".text", SHT_PROGBITS, (SHF_ALLOC | SHF_EXECINSTR) as u64, 4);
            }
            AsmDirective::Data => {
                self.current_section = self.get_or_create_section(".data", SHT_PROGBITS, (SHF_ALLOC | SHF_WRITE) as u64, 4);
            }
            AsmDirective::Bss => {
                self.current_section = self.get_or_create_section(".bss", SHT_NOBITS, (SHF_ALLOC | SHF_WRITE) as u64, 4);
            }
            AsmDirective::Global(name) => {
                // Update or create global symbol
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    sym.binding = STB_GLOBAL;
                } else {
                    self.symbols.push(ObjSymbol {
                        name: name.clone(),
                        section_name: String::new(),
                        value: 0,
                        size: 0,
                        binding: STB_GLOBAL,
                        sym_type: STT_NOTYPE,
                        visibility: STV_DEFAULT,
                    });
                }
            }
            AsmDirective::Local(name) => {
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    sym.binding = STB_LOCAL;
                }
            }
            AsmDirective::Hidden(name) => {
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    sym.visibility = STV_HIDDEN;
                } else {
                    self.symbols.push(ObjSymbol {
                        name: name.clone(), section_name: String::new(), value: 0, size: 0,
                        binding: STB_GLOBAL, sym_type: STT_NOTYPE, visibility: STV_HIDDEN,
                    });
                }
            }
            AsmDirective::Protected(name) => {
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    sym.visibility = STV_PROTECTED;
                }
            }
            AsmDirective::Weak(name) => {
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    sym.binding = STB_WEAK;
                } else {
                    self.symbols.push(ObjSymbol {
                        name: name.clone(), section_name: String::new(), value: 0, size: 0,
                        binding: STB_WEAK, sym_type: STT_NOTYPE, visibility: STV_DEFAULT,
                    });
                }
            }
            AsmDirective::Type(name, ty) => {
                let st_type = if ty.contains("function") { STT_FUNC }
                              else if ty.contains("object") { STT_OBJECT }
                              else { STT_NOTYPE };
                if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                    sym.sym_type = st_type;
                } else {
                    self.symbols.push(ObjSymbol {
                        name: name.clone(), section_name: String::new(), value: 0, size: 0,
                        binding: STB_GLOBAL, sym_type: st_type, visibility: STV_DEFAULT,
                    });
                }
            }
            AsmDirective::Size(name, expr) => {
                // Try to compute size from ".-name" pattern
                if expr.starts_with(".-") {
                    let label = &expr[2..];
                    if let Some(&(sec, start_offset)) = self.labels.get(label) {
                        let cur_offset = self.current_offset() as u64;
                        let size = cur_offset - start_offset;
                        if let Some(sym) = self.symbols.iter_mut().find(|s| s.name == *name) {
                            sym.size = size;
                        }
                    }
                }
            }
            AsmDirective::Byte(vals) => {
                for val in vals {
                    match val {
                        DataValue::Integer(v) => self.current_section_mut().data.push(*v as u8),
                        DataValue::Symbol(name) => {
                            let offset = self.current_offset() as u64;
                            let sec_name = self.sections[self.current_section].name.clone();
                            self.sections[self.current_section].relocs.push(ObjReloc {
                                offset, reloc_type: R_ARM_ABS32,
                                symbol_name: name.clone(), addend: 0,
                            });
                            self.current_section_mut().data.push(0);
                        }
                        _ => self.current_section_mut().data.push(0),
                    }
                }
            }
            AsmDirective::Short(vals) => {
                for val in vals {
                    match val {
                        DataValue::Integer(v) => {
                            self.current_section_mut().data.extend_from_slice(&(*v as u16).to_le_bytes());
                        }
                        _ => {
                            self.current_section_mut().data.extend_from_slice(&[0, 0]);
                        }
                    }
                }
            }
            AsmDirective::Word(vals) | AsmDirective::Long(vals) => {
                for val in vals {
                    self.emit_data_value_32(val);
                }
            }
            AsmDirective::Quad(vals) => {
                for val in vals {
                    match val {
                        DataValue::Integer(v) => {
                            self.current_section_mut().data.extend_from_slice(&(*v as u64).to_le_bytes());
                        }
                        DataValue::Symbol(name) => {
                            let offset = self.current_offset() as u64;
                            let sec_name = self.sections[self.current_section].name.clone();
                            self.sections[self.current_section].relocs.push(ObjReloc {
                                offset, reloc_type: R_ARM_ABS32,
                                symbol_name: name.clone(), addend: 0,
                            });
                            self.current_section_mut().data.extend_from_slice(&[0; 8]);
                        }
                        _ => self.current_section_mut().data.extend_from_slice(&[0; 8]),
                    }
                }
            }
            AsmDirective::Ascii(strings) => {
                for s in strings {
                    self.current_section_mut().data.extend_from_slice(s);
                }
            }
            AsmDirective::Asciz(strings) | AsmDirective::String(strings) => {
                for s in strings {
                    self.current_section_mut().data.extend_from_slice(s);
                    self.current_section_mut().data.push(0); // NUL terminator
                }
            }
            AsmDirective::Zero(n) => {
                self.current_section_mut().data.extend(std::iter::repeat(0u8).take(*n));
            }
            AsmDirective::Space(n) => {
                self.current_section_mut().data.extend(std::iter::repeat(0u8).take(*n));
            }
            AsmDirective::Fill(count, size, val) => {
                for _ in 0..*count {
                    for _ in 0..*size {
                        self.current_section_mut().data.push(*val);
                    }
                }
            }
            AsmDirective::Align(n) => {
                let align = if *n <= 4 { 1 << n } else { *n };
                self.align_section(align);
            }
            AsmDirective::Balign(n) => self.align_section(*n),
            AsmDirective::P2Align(n) => self.align_section(1 << n),
            AsmDirective::Comm(name, size, align) => {
                self.symbols.push(ObjSymbol {
                    name: name.clone(),
                    section_name: "*COM*".to_string(), // COMMON
                    value: *align as u64,
                    size: *size as u64,
                    binding: STB_GLOBAL,
                    sym_type: STT_OBJECT,
                    visibility: STV_DEFAULT,
                });
            }
            AsmDirective::Set(name, expr) | AsmDirective::Equiv(name, expr) => {
                self.set_symbols.insert(name.clone(), expr.clone());
            }
            AsmDirective::Inst(word) => {
                self.emit_word(*word);
            }
            // CFI directives - ignored for now (no .eh_frame generation)
            AsmDirective::CfiStartproc | AsmDirective::CfiEndproc |
            AsmDirective::CfiDefCfa(_, _) | AsmDirective::CfiDefCfaOffset(_) |
            AsmDirective::CfiDefCfaRegister(_) | AsmDirective::CfiOffset(_, _) |
            AsmDirective::CfiRestore(_) | AsmDirective::CfiRememberState |
            AsmDirective::CfiRestoreState | AsmDirective::CfiSections(_) => {}
            // ARM-specific directives - mostly ignored for basic .o generation
            AsmDirective::Syntax(_) | AsmDirective::Arch(_) | AsmDirective::Fpu(_) |
            AsmDirective::EabiAttribute(_) | AsmDirective::Code(_) |
            AsmDirective::Thumb | AsmDirective::ThumbFunc |
            AsmDirective::Fnstart | AsmDirective::Fnend | AsmDirective::Cantunwind |
            AsmDirective::Handlerdata | AsmDirective::Personality(_) |
            AsmDirective::PersonalityIndex(_) | AsmDirective::Pad(_) |
            AsmDirective::Save(_) | AsmDirective::Vsave(_) | AsmDirective::Setfp(_, _, _) |
            AsmDirective::ArmMapping | AsmDirective::ThumbMapping | AsmDirective::DataMapping |
            AsmDirective::PatchableFunctionEntry(_) => {}
            // Others
            AsmDirective::File(_) | AsmDirective::Ident(_) | AsmDirective::Loc(_) |
            AsmDirective::Ltorg | AsmDirective::Pool => {}
        }
        Ok(())
    }

    fn emit_data_value_32(&mut self, val: &DataValue) {
        match val {
            DataValue::Integer(v) => {
                self.current_section_mut().data.extend_from_slice(&(*v as u32).to_le_bytes());
            }
            DataValue::Symbol(name) => {
                let offset = self.current_offset() as u64;
                self.sections[self.current_section].relocs.push(ObjReloc {
                    offset, reloc_type: R_ARM_ABS32,
                    symbol_name: name.clone(), addend: 0,
                });
                self.current_section_mut().data.extend_from_slice(&[0; 4]);
            }
            DataValue::SymbolOffset(name, off) => {
                let offset = self.current_offset() as u64;
                self.sections[self.current_section].relocs.push(ObjReloc {
                    offset, reloc_type: R_ARM_ABS32,
                    symbol_name: name.clone(), addend: *off,
                });
                self.current_section_mut().data.extend_from_slice(&[0; 4]);
            }
            DataValue::SymbolDiff(a, b) => {
                // Try to resolve if both are in the same section
                let va = self.labels.get(a).map(|&(_, off)| off as i64);
                let vb = self.labels.get(b).map(|&(_, off)| off as i64);
                if let (Some(va), Some(vb)) = (va, vb) {
                    let diff = va - vb;
                    self.current_section_mut().data.extend_from_slice(&(diff as u32).to_le_bytes());
                } else {
                    self.current_section_mut().data.extend_from_slice(&[0; 4]);
                }
            }
            _ => {
                self.current_section_mut().data.extend_from_slice(&[0; 4]);
            }
        }
    }

    fn align_section(&mut self, align: usize) {
        if align <= 1 { return; }
        let cur = self.current_offset();
        let pad = (align - (cur % align)) % align;
        for _ in 0..pad {
            self.current_section_mut().data.push(0);
        }
        let sec = &mut self.sections[self.current_section];
        if (align as u64) > sec.sh_addralign {
            sec.sh_addralign = align as u64;
        }
    }

    fn encode_instruction(&mut self, mnemonic: &str, operands: &[Operand], raw: &str) -> Result<(), String> {
        let (base, cond_str, set_flags) = encoder::parse_mnemonic(mnemonic);
        let cond = encoder::encode_condition(cond_str);

        match base {
            // Data processing
            "add" | "sub" | "rsb" | "adc" | "sbc" | "rsc" |
            "and" | "orr" | "eor" | "bic" => {
                self.encode_dp3(base, cond, set_flags, operands)?;
            }
            "mov" | "mvn" => {
                self.encode_mov(base, cond, set_flags, operands)?;
            }
            "cmp" | "cmn" | "tst" | "teq" => {
                self.encode_cmp(base, cond, operands)?;
            }
            // Shifts
            "lsl" | "lsr" | "asr" | "ror" => {
                self.encode_shift(base, cond, set_flags, operands)?;
            }
            // Branches
            "b" => {
                self.encode_branch(cond, false, operands)?;
            }
            "bl" => {
                self.encode_branch(cond, true, operands)?;
            }
            "bx" => {
                if let Some(Operand::Reg(rm)) = operands.first() {
                    self.emit_word(encoder::encode_bx(cond, encoder::encode_reg(rm)));
                }
            }
            "blx" => {
                if let Some(Operand::Reg(rm)) = operands.first() {
                    self.emit_word(encoder::encode_blx_reg(cond, encoder::encode_reg(rm)));
                } else {
                    // BLX label - treat as BL
                    self.encode_branch(cond, true, operands)?;
                }
            }
            // Load/Store
            "ldr" | "str" | "ldrb" | "strb" => {
                self.encode_load_store(base, cond, operands)?;
            }
            "ldrh" | "strh" | "ldrsb" | "ldrsh" => {
                self.encode_halfword_ls(base, cond, operands)?;
            }
            // Multiply
            "mul" => {
                if operands.len() >= 3 {
                    let rd = self.get_reg(&operands[0]);
                    let rn = self.get_reg(&operands[1]);
                    let rm = self.get_reg(&operands[2]);
                    self.emit_word(encoder::encode_multiply(cond, rd, 0, rm, rn, false, set_flags));
                }
            }
            "mla" => {
                if operands.len() >= 4 {
                    let rd = self.get_reg(&operands[0]);
                    let rn = self.get_reg(&operands[1]);
                    let rm = self.get_reg(&operands[2]);
                    let ra = self.get_reg(&operands[3]);
                    self.emit_word(encoder::encode_mla(cond, rd, rn, rm, ra, set_flags));
                }
            }
            "umull" => {
                if operands.len() >= 4 {
                    let rdlo = self.get_reg(&operands[0]);
                    let rdhi = self.get_reg(&operands[1]);
                    let rn = self.get_reg(&operands[2]);
                    let rm = self.get_reg(&operands[3]);
                    self.emit_word(encoder::encode_umull(cond, rdlo, rdhi, rn, rm, set_flags));
                }
            }
            // Block transfer
            "push" => {
                if let Some(Operand::RegList(regs)) = operands.first() {
                    let mask = encoder::encode_reg_list(regs);
                    self.emit_word(encoder::encode_block_transfer(cond, false, true, false, true, 13, mask));
                }
            }
            "pop" => {
                if let Some(Operand::RegList(regs)) = operands.first() {
                    let mask = encoder::encode_reg_list(regs);
                    self.emit_word(encoder::encode_block_transfer(cond, true, false, true, true, 13, mask));
                }
            }
            // Special
            "nop" => self.emit_word(encoder::encode_nop(cond)),
            "dmb" | "dsb" | "isb" => {
                let opt = if let Some(Operand::Symbol(s)) = operands.first() { s.as_str() } else { "ish" };
                self.emit_word(encoder::encode_barrier(base, opt));
            }
            "clz" => {
                if operands.len() >= 2 {
                    let rd = self.get_reg(&operands[0]);
                    let rm = self.get_reg(&operands[1]);
                    self.emit_word(encoder::encode_clz(cond, rd, rm));
                }
            }
            "rbit" => {
                if operands.len() >= 2 {
                    let rd = self.get_reg(&operands[0]);
                    let rm = self.get_reg(&operands[1]);
                    self.emit_word(encoder::encode_rbit(cond, rd, rm));
                }
            }
            "rev" => {
                if operands.len() >= 2 {
                    let rd = self.get_reg(&operands[0]);
                    let rm = self.get_reg(&operands[1]);
                    self.emit_word(encoder::encode_rev(cond, rd, rm));
                }
            }
            "sxtb" | "sxth" | "uxtb" | "uxth" => {
                if operands.len() >= 2 {
                    let rd = self.get_reg(&operands[0]);
                    let rm = self.get_reg(&operands[1]);
                    self.emit_word(encoder::encode_extend(cond, base, rd, rm));
                }
            }
            "movw" => {
                if operands.len() >= 2 {
                    let rd = self.get_reg(&operands[0]);
                    match &operands[1] {
                        Operand::Imm(v) => {
                            self.emit_word(encoder::encode_movw(cond, rd, *v as u32));
                        }
                        Operand::Reloc(reloc_type, sym) if reloc_type == "lower16" => {
                            let offset = self.current_offset() as u64;
                            self.sections[self.current_section].relocs.push(ObjReloc {
                                offset, reloc_type: R_ARM_MOVW_ABS_NC,
                                symbol_name: sym.clone(), addend: 0,
                            });
                            self.emit_word(encoder::encode_movw(cond, rd, 0));
                        }
                        _ => self.emit_word(encoder::encode_movw(cond, rd, 0)),
                    }
                }
            }
            "movt" => {
                if operands.len() >= 2 {
                    let rd = self.get_reg(&operands[0]);
                    match &operands[1] {
                        Operand::Imm(v) => {
                            self.emit_word(encoder::encode_movt(cond, rd, *v as u32));
                        }
                        Operand::Reloc(reloc_type, sym) if reloc_type == "upper16" => {
                            let offset = self.current_offset() as u64;
                            self.sections[self.current_section].relocs.push(ObjReloc {
                                offset, reloc_type: R_ARM_MOVT_ABS,
                                symbol_name: sym.clone(), addend: 0,
                            });
                            self.emit_word(encoder::encode_movt(cond, rd, 0));
                        }
                        _ => self.emit_word(encoder::encode_movt(cond, rd, 0)),
                    }
                }
            }
            "ldrex" | "ldrexb" | "ldrexh" => {
                if operands.len() >= 2 {
                    let rt = self.get_reg(&operands[0]);
                    let rn = match &operands[1] {
                        Operand::Mem { base, .. } => encoder::encode_reg(base),
                        Operand::Reg(r) => encoder::encode_reg(r),
                        _ => 0,
                    };
                    let word = match mnemonic {
                        "ldrexb" => encoder::encode_ldrexb(cond, rt, rn),
                        "ldrexh" => encoder::encode_ldrexh(cond, rt, rn),
                        _ => encoder::encode_ldrex(cond, rt, rn),
                    };
                    self.emit_word(word);
                }
            }
            "strex" | "strexb" | "strexh" => {
                if operands.len() >= 3 {
                    let rd = self.get_reg(&operands[0]);
                    let rt = self.get_reg(&operands[1]);
                    let rn = match &operands[2] {
                        Operand::Mem { base, .. } => encoder::encode_reg(base),
                        Operand::Reg(r) => encoder::encode_reg(r),
                        _ => 0,
                    };
                    let word = match mnemonic {
                        "strexb" => encoder::encode_strexb(cond, rd, rt, rn),
                        "strexh" => encoder::encode_strexh(cond, rd, rt, rn),
                        _ => encoder::encode_strex(cond, rd, rt, rn),
                    };
                    self.emit_word(word);
                }
            }
            "clrex" => self.emit_word(encoder::encode_clrex()),
            "bkpt" => {
                let imm = if let Some(Operand::Imm(v)) = operands.first() { *v as u32 } else { 0 };
                self.emit_word((cond << 28) | 0x01200070 | ((imm & 0xFFF0) << 4) | (imm & 0xF));
            }
            // VFP arithmetic: vadd/vsub/vmul/vdiv .f32/.f64
            "vadd.f32" | "vsub.f32" | "vmul.f32" | "vdiv.f32" |
            "vadd.f64" | "vsub.f64" | "vmul.f64" | "vdiv.f64" => {
                self.encode_vfp_arith(mnemonic, cond, &operands)?;
            }
            // VFP conversions
            "vcvt.f32.f64" | "vcvt.f64.f32" | "vcvt.s32.f32" | "vcvt.u32.f32" |
            "vcvt.f32.s32" | "vcvt.f32.u32" | "vcvt.s32.f64" | "vcvt.u32.f64" |
            "vcvt.f64.s32" | "vcvt.f64.u32" => {
                self.encode_vfp_cvt(mnemonic, cond, &operands)?;
            }
            // VFP compare
            "vcmp.f32" | "vcmp.f64" => {
                self.encode_vfp_cmp(mnemonic, cond, &operands)?;
            }
            // VFP negate/abs
            "vneg.f32" | "vneg.f64" | "vabs.f32" | "vabs.f64" => {
                self.encode_vfp_unary(mnemonic, cond, &operands)?;
            }
            // VFP register transfer / system
            "vmov" => {
                self.encode_vmov(cond, &operands)?;
            }
            "vmrs" => {
                self.emit_word(encoder::encode_vmrs_apsr(cond));
            }
            // VFP memory (vldr/vstr/vpush/vpop) - use NOP for now
            "vldr" | "vstr" | "vpush" | "vpop" => {
                self.emit_word(encoder::encode_nop(cond));
            }
            "pld" => {
                // PLD: cache prefetch (hint, can be NOP)
                self.emit_word(0xF5D0F000);
            }
            _ => {
                // Unknown instruction: emit NOP
                self.emit_word(encoder::encode_nop(cond));
            }
        }
        Ok(())
    }

    fn encode_dp3(&mut self, mnemonic: &str, cond: u32, set_flags: bool, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 3 { return Err(format!("{}: need 3 operands", mnemonic)); }
        let rd = self.get_reg(&operands[0]);
        let rn = self.get_reg(&operands[1]);
        let opcode = encoder::dp_opcode(mnemonic);

        match &operands[2] {
            Operand::Imm(v) => {
                if let Some((imm8, rot)) = encoder::encode_arm_imm(*v as u32) {
                    let operand2 = (rot << 8) | imm8;
                    self.emit_word(encoder::encode_data_proc(cond, opcode, set_flags, rn, rd, operand2, true));
                } else {
                    return Err(format!("{}: immediate {} not encodable", mnemonic, v));
                }
            }
            Operand::Reg(rm) => {
                let rm_val = encoder::encode_reg(rm);
                self.emit_word(encoder::encode_data_proc(cond, opcode, set_flags, rn, rd, rm_val, false));
            }
            _ => {
                self.emit_word(encoder::encode_nop(cond));
            }
        }
        Ok(())
    }

    fn encode_mov(&mut self, mnemonic: &str, cond: u32, set_flags: bool, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 { return Err(format!("{}: need 2 operands", mnemonic)); }
        let rd = self.get_reg(&operands[0]);
        let opcode = encoder::dp_opcode(mnemonic);

        match &operands[1] {
            Operand::Imm(v) => {
                if let Some((imm8, rot)) = encoder::encode_arm_imm(*v as u32) {
                    let operand2 = (rot << 8) | imm8;
                    self.emit_word(encoder::encode_data_proc(cond, opcode, set_flags, 0, rd, operand2, true));
                } else {
                    return Err(format!("{}: immediate {} not encodable", mnemonic, v));
                }
            }
            Operand::Reg(rm) => {
                let rm_val = encoder::encode_reg(rm);
                self.emit_word(encoder::encode_data_proc(cond, opcode, set_flags, 0, rd, rm_val, false));
            }
            _ => {
                self.emit_word(encoder::encode_nop(cond));
            }
        }
        Ok(())
    }

    fn encode_cmp(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 { return Err(format!("{}: need 2 operands", mnemonic)); }
        let rn = self.get_reg(&operands[0]);
        let opcode = encoder::dp_opcode(mnemonic);

        match &operands[1] {
            Operand::Imm(v) => {
                if let Some((imm8, rot)) = encoder::encode_arm_imm(*v as u32) {
                    let operand2 = (rot << 8) | imm8;
                    self.emit_word(encoder::encode_data_proc(cond, opcode, true, rn, 0, operand2, true));
                } else {
                    return Err(format!("{}: immediate {} not encodable", mnemonic, v));
                }
            }
            Operand::Reg(rm) => {
                let rm_val = encoder::encode_reg(rm);
                self.emit_word(encoder::encode_data_proc(cond, opcode, true, rn, 0, rm_val, false));
            }
            _ => {
                self.emit_word(encoder::encode_nop(cond));
            }
        }
        Ok(())
    }

    fn encode_shift(&mut self, mnemonic: &str, cond: u32, set_flags: bool, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 3 { return Err(format!("{}: need 3 operands", mnemonic)); }
        let rd = self.get_reg(&operands[0]);
        let rm = self.get_reg(&operands[1]);
        let shift_type: u32 = match mnemonic {
            "lsl" => 0b00,
            "lsr" => 0b01,
            "asr" => 0b10,
            "ror" => 0b11,
            _ => 0,
        };
        let s = if set_flags { 1 } else { 0 };

        match &operands[2] {
            Operand::Imm(v) => {
                let imm5 = (*v as u32) & 0x1F;
                let operand2 = rm | (shift_type << 5) | (imm5 << 7);
                self.emit_word((cond << 28) | (0b1101 << 21) | (s << 20) | (rd << 12) | operand2);
            }
            Operand::Reg(rs) => {
                let rs_val = encoder::encode_reg(rs);
                let operand2 = rm | (shift_type << 5) | (1 << 4) | (rs_val << 8);
                self.emit_word((cond << 28) | (0b1101 << 21) | (s << 20) | (rd << 12) | operand2);
            }
            _ => self.emit_word(encoder::encode_nop(cond)),
        }
        Ok(())
    }

    fn encode_branch(&mut self, cond: u32, link: bool, operands: &[Operand]) -> Result<(), String> {
        if operands.is_empty() { return Err("branch: need target".to_string()); }
        let target = match &operands[0] {
            Operand::Symbol(s) | Operand::Label(s) => s.clone(),
            _ => return Err("branch: need symbol target".to_string()),
        };

        // Check if target is a local label we can resolve
        if let Some(&(target_sec, target_off)) = self.labels.get(&target) {
            if target_sec == self.current_section {
                let current_off = self.current_offset() as i64;
                let offset = (target_off as i64 - current_off - 8) as i32;
                self.emit_word(encoder::encode_branch(cond, link, offset));
                return Ok(());
            }
        }

        // Can't resolve now, add pending branch and relocation
        let offset = self.current_offset();
        self.pending_branches.push(PendingBranch {
            section_idx: self.current_section,
            offset,
            target: target.clone(),
            is_link: link,
            cond,
        });
        // Emit placeholder
        let reloc_type = if link { R_ARM_CALL } else { R_ARM_JUMP24 };
        self.sections[self.current_section].relocs.push(ObjReloc {
            offset: offset as u64,
            reloc_type,
            symbol_name: target,
            addend: 0,
        });
        self.emit_word(encoder::encode_branch(cond, link, 0)); // zero addend: linker handles PC+8
        Ok(())
    }

    fn encode_load_store(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 { return Err(format!("{}: need at least 2 operands", mnemonic)); }
        let rd = self.get_reg(&operands[0]);
        let is_load = mnemonic.starts_with("ldr");
        let is_byte = mnemonic.ends_with('b');

        match &operands[1] {
            Operand::Mem { base, offset, writeback } => {
                let rn = encoder::encode_reg(base);
                let wb = *writeback;
                match offset {
                    MemOffset::None => {
                        self.emit_word(encoder::encode_load_store(cond, is_load, is_byte, true, true, wb, rn, rd, 0, true));
                    }
                    MemOffset::Imm(v) => {
                        let (add, abs_off) = if *v >= 0 { (true, *v as u32) } else { (false, (-*v) as u32) };
                        self.emit_word(encoder::encode_load_store(cond, is_load, is_byte, true, add, wb, rn, rd, abs_off & 0xFFF, true));
                    }
                    MemOffset::Reg(rm) => {
                        let rm_val = encoder::encode_reg(rm);
                        self.emit_word(encoder::encode_load_store(cond, is_load, is_byte, true, true, wb, rn, rd, rm_val, false));
                    }
                    _ => self.emit_word(encoder::encode_nop(cond)),
                }
            }
            Operand::Symbol(sym) => {
                // LDR Rd, =symbol or LDR Rd, symbol
                let offset = self.current_offset() as u64;
                self.sections[self.current_section].relocs.push(ObjReloc {
                    offset, reloc_type: R_ARM_ABS32,
                    symbol_name: sym.clone(), addend: 0,
                });
                // LDR from PC-relative literal pool
                self.emit_word(encoder::encode_load_store(cond, true, false, true, true, false, 15, rd, 0, true));
            }
            _ => self.emit_word(encoder::encode_nop(cond)),
        }
        Ok(())
    }

    fn encode_halfword_ls(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 { return Err(format!("{}: need at least 2 operands", mnemonic)); }
        let rd = self.get_reg(&operands[0]);
        let is_load = mnemonic.starts_with("ldr");
        let sign = mnemonic.contains("s");
        let half = mnemonic.contains("h");

        if let Operand::Mem { base, offset, writeback } = &operands[1] {
            let rn = encoder::encode_reg(base);
            let wb = *writeback;
            match offset {
                MemOffset::None => {
                    self.emit_word(encoder::encode_halfword_load_store(cond, is_load, sign, half, true, true, wb, true, rn, rd, 0));
                }
                MemOffset::Imm(v) => {
                    let (add, abs_off) = if *v >= 0 { (true, *v as u32) } else { (false, (-*v) as u32) };
                    self.emit_word(encoder::encode_halfword_load_store(cond, is_load, sign, half, true, add, wb, true, rn, rd, abs_off & 0xFF));
                }
                _ => self.emit_word(encoder::encode_nop(cond)),
            }
        }
        Ok(())
    }

    // ============================================================
    // VFP instruction encoding helpers
    // ============================================================

    /// Get a floating-point register number from an operand.
    fn get_fp_reg(op: &Operand) -> Result<(bool, u32), String> {
        match op {
            Operand::Reg(name) => encoder::parse_fp_reg(name)
                .ok_or_else(|| format!("expected FP register, got '{}'", name)),
            _ => Err("expected FP register operand".to_string()),
        }
    }

    /// Encode vadd/vsub/vmul/vdiv .f32/.f64
    fn encode_vfp_arith(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 3 {
            return Err(format!("{}: need 3 operands (Fd, Fn, Fm)", mnemonic));
        }
        let is_double = mnemonic.ends_with(".f64");
        let op = if mnemonic.starts_with("vadd") { encoder::VfpArithOp::Add }
            else if mnemonic.starts_with("vsub") { encoder::VfpArithOp::Sub }
            else if mnemonic.starts_with("vmul") { encoder::VfpArithOp::Mul }
            else { encoder::VfpArithOp::Div };
        let (_, fd) = Self::get_fp_reg(&operands[0])?;
        let (_, fn_) = Self::get_fp_reg(&operands[1])?;
        let (_, fm) = Self::get_fp_reg(&operands[2])?;
        self.emit_word(encoder::encode_vfp_arith(cond, op, is_double, fd, fn_, fm));
        Ok(())
    }

    /// Encode vcvt variants
    fn encode_vfp_cvt(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 {
            return Err(format!("{}: need 2 operands", mnemonic));
        }
        let (_, fd) = Self::get_fp_reg(&operands[0])?;
        let (_, fm) = Self::get_fp_reg(&operands[1])?;
        let word = match mnemonic {
            "vcvt.f64.f32" => encoder::encode_vcvt_f2f(cond, true, fd, fm),
            "vcvt.f32.f64" => encoder::encode_vcvt_f2f(cond, false, fd, fm),
            "vcvt.f32.s32" => encoder::encode_vcvt_int_to_fp(cond, false, true, fd, fm),
            "vcvt.f32.u32" => encoder::encode_vcvt_int_to_fp(cond, false, false, fd, fm),
            "vcvt.f64.s32" => encoder::encode_vcvt_int_to_fp(cond, true, true, fd, fm),
            "vcvt.f64.u32" => encoder::encode_vcvt_int_to_fp(cond, true, false, fd, fm),
            "vcvt.s32.f32" => encoder::encode_vcvt_fp_to_int(cond, false, true, fd, fm),
            "vcvt.u32.f32" => encoder::encode_vcvt_fp_to_int(cond, false, false, fd, fm),
            "vcvt.s32.f64" => encoder::encode_vcvt_fp_to_int(cond, true, true, fd, fm),
            "vcvt.u32.f64" => encoder::encode_vcvt_fp_to_int(cond, true, false, fd, fm),
            _ => return Err(format!("unknown vcvt variant: {}", mnemonic)),
        };
        self.emit_word(word);
        Ok(())
    }

    /// Encode vcmp.f32/vcmp.f64
    fn encode_vfp_cmp(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 {
            return Err(format!("{}: need 2 operands", mnemonic));
        }
        let is_double = mnemonic == "vcmp.f64";
        let (_, fd) = Self::get_fp_reg(&operands[0])?;
        let (_, fm) = Self::get_fp_reg(&operands[1])?;
        self.emit_word(encoder::encode_vcmp(cond, is_double, fd, fm));
        Ok(())
    }

    /// Encode vneg/vabs .f32/.f64
    fn encode_vfp_unary(&mut self, mnemonic: &str, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 {
            return Err(format!("{}: need 2 operands", mnemonic));
        }
        let is_double = mnemonic.ends_with(".f64");
        let (_, fd) = Self::get_fp_reg(&operands[0])?;
        let (_, fm) = Self::get_fp_reg(&operands[1])?;
        let word = if mnemonic.starts_with("vneg") {
            encoder::encode_vneg(cond, is_double, fd, fm)
        } else {
            encoder::encode_vabs(cond, is_double, fd, fm)
        };
        self.emit_word(word);
        Ok(())
    }

    /// Encode vmov instruction (multiple variants).
    /// Supported forms:
    ///   vmov Sn, Rt       (GP → single)
    ///   vmov Rt, Sn       (single → GP)
    ///   vmov Dm, Rt, Rt2  (GP pair → double)
    ///   vmov Rt, Rt2, Dm  (double → GP pair)
    fn encode_vmov(&mut self, cond: u32, operands: &[Operand]) -> Result<(), String> {
        if operands.len() < 2 {
            return Err("vmov: need at least 2 operands".to_string());
        }

        // Determine which vmov variant based on operand types
        let op0_name = match &operands[0] { Operand::Reg(n) => n.as_str(), _ => "" };
        let op1_name = match &operands[1] { Operand::Reg(n) => n.as_str(), _ => "" };

        let op0_is_gp = encoder::is_gp_reg(op0_name);
        let op0_fp = encoder::parse_fp_reg(op0_name);
        let op1_is_gp = encoder::is_gp_reg(op1_name);
        let op1_fp = encoder::parse_fp_reg(op1_name);

        if let Some((false, sn)) = op0_fp {
            if op1_is_gp {
                // vmov Sn, Rt (GP → single-precision)
                let rt = encoder::encode_reg(op1_name);
                self.emit_word(encoder::encode_vmov_core_single(cond, true, sn, rt));
                return Ok(());
            }
        }

        if op0_is_gp {
            if let Some((false, sn)) = op1_fp {
                // vmov Rt, Sn (single-precision → GP)
                let rt = encoder::encode_reg(op0_name);
                self.emit_word(encoder::encode_vmov_core_single(cond, false, sn, rt));
                return Ok(());
            }
        }

        if let Some((true, dm)) = op0_fp {
            // vmov Dm, Rt, Rt2 (GP pair → double)
            if operands.len() >= 3 {
                let rt_name = match &operands[1] { Operand::Reg(n) => n.as_str(), _ => "" };
                let rt2_name = match &operands[2] { Operand::Reg(n) => n.as_str(), _ => "" };
                let rt = encoder::encode_reg(rt_name);
                let rt2 = encoder::encode_reg(rt2_name);
                self.emit_word(encoder::encode_vmov_core_double(cond, true, dm, rt, rt2));
                return Ok(());
            }
        }

        if op0_is_gp && operands.len() >= 3 {
            let op2_name = match &operands[2] { Operand::Reg(n) => n.as_str(), _ => "" };
            if let Some((true, dm)) = encoder::parse_fp_reg(op2_name) {
                // vmov Rt, Rt2, Dm (double → GP pair)
                let rt = encoder::encode_reg(op0_name);
                let rt2 = encoder::encode_reg(op1_name);
                self.emit_word(encoder::encode_vmov_core_double(cond, false, dm, rt, rt2));
                return Ok(());
            }
        }

        // Fallback: emit NOP for unsupported vmov variants
        self.emit_word(encoder::encode_nop(cond));
        Ok(())
    }

    fn get_reg(&self, op: &Operand) -> u32 {
        match op {
            Operand::Reg(name) => encoder::encode_reg(name),
            _ => 0,
        }
    }

    fn resolve_branches(&mut self) -> Result<(), String> {
        // Collect resolved branch info (section_idx, offset, reloc_type) to remove relocs after
        let mut resolved: Vec<(usize, usize, u32)> = Vec::new();

        for branch in &self.pending_branches {
            if let Some(&(target_sec, target_off)) = self.labels.get(&branch.target) {
                if target_sec == branch.section_idx {
                    let offset = (target_off as i64 - branch.offset as i64 - 8) as i32;
                    let word = encoder::encode_branch(branch.cond, branch.is_link, offset);
                    let data = &mut self.sections[branch.section_idx].data;
                    let bytes = word.to_le_bytes();
                    data[branch.offset..branch.offset + 4].copy_from_slice(&bytes);
                    // Mark for relocation removal
                    let reloc_type = if branch.is_link { R_ARM_CALL } else { R_ARM_JUMP24 };
                    resolved.push((branch.section_idx, branch.offset, reloc_type));
                }
            }
        }

        // Remove relocations for locally resolved branches to prevent
        // the linker from double-applying the offset
        for (sec_idx, offset, reloc_type) in &resolved {
            self.sections[*sec_idx].relocs.retain(|r| {
                !(r.offset == *offset as u64 && r.reloc_type == *reloc_type)
            });
        }

        Ok(())
    }

    pub fn write_elf(&self, output_path: &str) -> Result<(), String> {
        let config = ElfConfig {
            e_machine: EM_ARM,
            e_flags: 0x05000000, // EF_ARM_ABI_VER5
            elf_class: ELFCLASS32,
            force_rela: false,
        };

        // Convert Vec<ObjSection> to HashMap<String, ObjSection> + section_order
        let mut section_map = HashMap::new();
        let mut section_order = Vec::new();
        for sec in &self.sections {
            section_order.push(sec.name.clone());
            section_map.insert(sec.name.clone(), ObjSection {
                name: sec.name.clone(),
                sh_type: sec.sh_type,
                sh_flags: sec.sh_flags,
                data: sec.data.clone(),
                sh_addralign: sec.sh_addralign,
                relocs: sec.relocs.clone(),
                comdat_group: sec.comdat_group.clone(),
            });
        }

        let elf_bytes = write_relocatable_object(
            &config,
            &section_order,
            &section_map,
            &self.symbols,
        )?;
        std::fs::write(output_path, &elf_bytes)
            .map_err(|e| format!("failed to write {}: {}", output_path, e))
    }
}

fn parse_section_spec(spec: &str) -> (String, u32, u64) {
    let parts: Vec<&str> = spec.split(',').collect();
    let name = parts[0].trim().trim_matches('"');

    let mut flags: u64 = 0;
    let mut sh_type = SHT_PROGBITS;

    if name == ".text" || name.starts_with(".text.") {
        flags = (SHF_ALLOC | SHF_EXECINSTR) as u64;
    } else if name == ".data" || name.starts_with(".data.") {
        flags = (SHF_ALLOC | SHF_WRITE) as u64;
    } else if name == ".bss" || name.starts_with(".bss.") {
        flags = (SHF_ALLOC | SHF_WRITE) as u64;
        sh_type = SHT_NOBITS;
    } else if name == ".rodata" || name.starts_with(".rodata.") {
        flags = SHF_ALLOC as u64;
    } else if name.starts_with(".note") {
        flags = SHF_ALLOC as u64;
        sh_type = SHT_NOTE;
    }

    // Parse flags string if present
    if parts.len() > 1 {
        let flag_str = parts[1].trim().trim_matches('"');
        for ch in flag_str.chars() {
            match ch {
                'a' => flags |= SHF_ALLOC as u64,
                'w' => flags |= SHF_WRITE as u64,
                'x' => flags |= SHF_EXECINSTR as u64,
                'M' => flags |= SHF_MERGE as u64,
                'S' => flags |= SHF_STRINGS as u64,
                'G' => flags |= SHF_GROUP as u64,
                'T' => flags |= SHF_TLS as u64,
                _ => {}
            }
        }
    }
    if parts.len() > 2 {
        let type_str = parts[2].trim().trim_matches(|c| c == '@' || c == '%' || c == '"');
        match type_str {
            "progbits" => sh_type = SHT_PROGBITS,
            "nobits" => sh_type = SHT_NOBITS,
            "note" => sh_type = SHT_NOTE,
            "init_array" => sh_type = SHT_INIT_ARRAY,
            "fini_array" => sh_type = SHT_FINI_ARRAY,
            _ => {}
        }
    }

    (name.to_string(), sh_type, flags)
}
