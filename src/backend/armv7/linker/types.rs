//! ELF32 types and constants for the ARMv7 linker.
//!
//! Contains all ARMv7/ELF32-specific constants, structures, and the linker symbol
//! types used throughout the linking process. Closely mirrors the i686 linker's types.

use std::collections::HashMap;

// Re-export shared ELF constants
pub(super) use crate::backend::elf::{
    ELF_MAGIC, ELFCLASS32, ELFDATA2LSB, EV_CURRENT,
    ET_EXEC, ET_DYN, ET_REL, EM_ARM,
    PT_LOAD, PT_DYNAMIC, PT_INTERP, PT_PHDR, PT_TLS, PT_GNU_STACK, PT_GNU_EH_FRAME,
    SHT_NULL, SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_RELA,
    SHT_NOBITS, SHT_REL, SHT_DYNSYM, SHT_GROUP,
    SHT_INIT_ARRAY, SHT_FINI_ARRAY,
    STB_LOCAL, STB_GLOBAL, STB_WEAK,
    STT_OBJECT, STT_FUNC, STT_SECTION, STT_FILE, STT_TLS, STT_GNU_IFUNC,
    STV_DEFAULT,
    SHN_UNDEF, SHN_ABS, SHN_COMMON,
    PF_X, PF_W, PF_R,
    read_u16, read_u32, read_cstr, read_i32,
    parse_archive_members, parse_thin_archive_members, is_thin_archive,
    parse_linker_script_entries, LinkerScriptEntry,
    LinkerSymbolAddresses, get_standard_linker_symbols,
};

// ── ELF32-specific constants ──────────────────────────────────────────────────

pub(super) const SHT_NOTE: u32 = 7;
#[allow(dead_code)]
pub(super) const SHT_GNU_HASH: u32 = 0x6ffffff6;
#[allow(dead_code)]
pub(super) const SHT_GNU_VERSYM_CONST: u32 = 0x6fffffff;
#[allow(dead_code)]
pub(super) const SHT_GNU_VERNEED: u32 = 0x6ffffffe;

// Section flags (u32 for ELF32)
pub(super) const SHF_WRITE: u32 = 0x1;
pub(super) const SHF_ALLOC: u32 = 0x2;
pub(super) const SHF_EXECINSTR: u32 = 0x4;
#[allow(dead_code)]
pub(super) const SHF_MERGE: u32 = 0x10;
#[allow(dead_code)]
pub(super) const SHF_STRINGS: u32 = 0x20;
pub(super) const SHF_GROUP: u32 = 0x200;
pub(super) const SHF_TLS: u32 = 0x400;

// ARM relocation types
pub(super) const R_ARM_NONE: u32 = 0;
pub(super) const R_ARM_ABS32: u32 = 2;
pub(super) const R_ARM_REL32: u32 = 3;
pub(super) const R_ARM_GLOB_DAT: u32 = 21;
pub(super) const R_ARM_JUMP_SLOT: u32 = 22;
pub(super) const R_ARM_RELATIVE: u32 = 23;
pub(super) const R_ARM_GOT32: u32 = 26;
pub(super) const R_ARM_PLT32: u32 = 27;
pub(super) const R_ARM_CALL: u32 = 28;
pub(super) const R_ARM_JUMP24: u32 = 29;
pub(super) const R_ARM_TARGET1: u32 = 38;
pub(super) const R_ARM_V4BX: u32 = 40;
pub(super) const R_ARM_PREL31: u32 = 42;
pub(super) const R_ARM_MOVW_ABS_NC: u32 = 43;
pub(super) const R_ARM_MOVT_ABS: u32 = 44;
pub(super) const R_ARM_MOVW_PREL_NC: u32 = 45;
pub(super) const R_ARM_MOVT_PREL: u32 = 46;
pub(super) const R_ARM_GOT_BREL: u32 = 26;
pub(super) const R_ARM_GOTOFF32: u32 = 24;
pub(super) const R_ARM_COPY: u32 = 20;
pub(super) const R_ARM_TLS_GD32: u32 = 104;
pub(super) const R_ARM_TLS_IE32: u32 = 107;
pub(super) const R_ARM_TLS_LE32: u32 = 108;
#[allow(dead_code)]
pub(super) const R_ARM_TLS_DTPMOD32: u32 = 17;
#[allow(dead_code)]
pub(super) const R_ARM_TLS_DTPOFF32: u32 = 18;
#[allow(dead_code)]
pub(super) const R_ARM_TLS_TPOFF32: u32 = 19;
pub(super) const R_ARM_IRELATIVE: u32 = 160;

// Dynamic tags (i32 for ELF32)
pub(super) const DT_NULL: i32 = 0;
pub(super) const DT_NEEDED: i32 = 1;
pub(super) const DT_PLTRELSZ: i32 = 2;
pub(super) const DT_PLTGOT: i32 = 3;
pub(super) const DT_STRTAB: i32 = 5;
pub(super) const DT_SYMTAB: i32 = 6;
pub(super) const DT_STRSZ: i32 = 10;
pub(super) const DT_SYMENT: i32 = 11;
pub(super) const DT_INIT: i32 = 12;
pub(super) const DT_FINI: i32 = 13;
pub(super) const DT_SONAME: i32 = 14;
pub(super) const DT_REL_TAG: i32 = 17;
pub(super) const DT_RELSZ: i32 = 18;
pub(super) const DT_RELENT: i32 = 19;
pub(super) const DT_PLTREL: i32 = 20;
pub(super) const DT_DEBUG: i32 = 21;
pub(super) const DT_TEXTREL: i32 = 22;
pub(super) const DT_JMPREL: i32 = 23;
pub(super) const DT_INIT_ARRAY: i32 = 25;
pub(super) const DT_FINI_ARRAY: i32 = 26;
pub(super) const DT_INIT_ARRAYSZ: i32 = 27;
pub(super) const DT_FINI_ARRAYSZ: i32 = 28;
pub(super) const DT_FLAGS: i32 = 30;
pub(super) const DT_GNU_HASH_TAG: i32 = 0x6ffffef5u32 as i32;
pub(super) const DT_VERNEED: i32 = 0x6ffffffe_u32 as i32;
pub(super) const DT_VERNEEDNUM: i32 = 0x6fffffff_u32 as i32;
pub(super) const DT_VERSYM: i32 = 0x6ffffff0_u32 as i32;

// ARM ELF flags
pub(super) const EF_ARM_ABI_VER5: u32 = 0x05000000;
pub(super) const EF_ARM_ABI_FLOAT_HARD: u32 = 0x00000400;

pub(super) const PAGE_SIZE: u32 = 0x10000;
pub(super) const BASE_ADDR: u32 = 0x10000;
pub(super) const INTERP: &[u8] = b"/lib/ld-linux-armhf.so.3\0";
pub(super) const PLT_ENTRY_SIZE: u32 = 16;

// ── ELF32 structures ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(super) struct Elf32Sym {
    pub name: u32,
    pub value: u32,
    pub size: u32,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
}

#[allow(dead_code)]
impl Elf32Sym {
    pub fn binding(&self) -> u8 { self.info >> 4 }
    pub fn sym_type(&self) -> u8 { self.info & 0xf }
}

#[derive(Clone, Debug)]
pub(super) struct Elf32Shdr {
    pub name: u32,
    pub sh_type: u32,
    pub flags: u32,
    #[allow(dead_code)]
    pub addr: u32,
    pub offset: u32,
    pub size: u32,
    pub link: u32,
    pub info: u32,
    pub addralign: u32,
    pub entsize: u32,
}

// ── Input object types ────────────────────────────────────────────────────────

/// A parsed section from an input .o file.
#[derive(Clone)]
pub(super) struct InputSection {
    pub name: String,
    pub sh_type: u32,
    pub flags: u32,
    pub data: Vec<u8>,
    pub align: u32,
    /// Relocations: (offset, rel_type, sym_idx_in_input, addend)
    pub relocations: Vec<(u32, u32, u32, i32)>,
    /// Index in the input file's section header table.
    pub input_index: usize,
    #[allow(dead_code)]
    pub entsize: u32,
    #[allow(dead_code)]
    pub link: u32,
    pub info: u32,
}

/// A parsed symbol from an input .o file.
#[derive(Clone, Debug)]
pub(super) struct InputSymbol {
    pub name: String,
    pub value: u32,
    pub size: u32,
    pub binding: u8,
    pub sym_type: u8,
    #[allow(dead_code)]
    pub visibility: u8,
    pub section_index: u16,
}

/// A parsed input file.
pub(super) struct InputObject {
    pub sections: Vec<InputSection>,
    pub symbols: Vec<InputSymbol>,
    pub filename: String,
}

// ── Linker state types ────────────────────────────────────────────────────────

/// Resolved symbol info in the linker.
#[derive(Clone, Debug)]
pub(super) struct LinkerSymbol {
    pub address: u32,
    pub size: u32,
    pub sym_type: u8,
    pub binding: u8,
    #[allow(dead_code)]
    pub visibility: u8,
    pub is_defined: bool,
    pub needs_plt: bool,
    pub needs_got: bool,
    pub output_section: usize,
    pub section_offset: u32,
    pub plt_index: usize,
    pub got_index: usize,
    pub is_dynamic: bool,
    pub dynlib: String,
    pub needs_copy: bool,
    pub copy_addr: u32,
    pub version: Option<String>,
    pub uses_textrel: bool,
}

/// A merged output section.
pub(super) struct OutputSection {
    pub name: String,
    pub sh_type: u32,
    pub flags: u32,
    pub data: Vec<u8>,
    pub align: u32,
    pub addr: u32,
    pub file_offset: u32,
}

/// Maps (object_index, section_index) -> (output_section_index, offset_in_output).
pub(super) type SectionMap = HashMap<(usize, usize), (usize, u32)>;

// ── Helpers ──────────────────────────────────────────────────────────────────

pub(super) fn align_up(value: u32, align: u32) -> u32 {
    if align == 0 { return value; }
    (value + align - 1) & !(align - 1)
}

/// Append an ELF32 dynamic entry (8 bytes: tag + value).
pub(super) fn push_dyn(data: &mut Vec<u8>, tag: i32, val: u32) {
    data.extend_from_slice(&tag.to_le_bytes());
    data.extend_from_slice(&val.to_le_bytes());
}

/// Determine the output section name for an input section.
pub(super) fn output_section_name(name: &str, flags: u32, sh_type: u32) -> Option<String> {
    if sh_type == SHT_NULL || sh_type == SHT_SYMTAB || sh_type == SHT_STRTAB
        || sh_type == SHT_REL || sh_type == SHT_RELA || sh_type == SHT_GROUP {
        return None;
    }
    if name == ".note.GNU-stack" || name == ".comment" {
        return None;
    }
    // ARM mapping symbols section
    if name == ".ARM.attributes" || name.starts_with(".ARM.exidx") || name.starts_with(".ARM.extab") {
        return None;
    }
    if name.starts_with(".text") || name == ".init" || name == ".fini" {
        if name == ".init" { return Some(".init".to_string()); }
        if name == ".fini" { return Some(".fini".to_string()); }
        return Some(".text".to_string());
    }
    if name.starts_with(".rodata") {
        return Some(".rodata".to_string());
    }
    if name == ".eh_frame" {
        return Some(".eh_frame".to_string());
    }
    if name == ".tbss" || name.starts_with(".tbss.") {
        return Some(".tbss".to_string());
    }
    if name == ".tdata" || name.starts_with(".tdata.") {
        return Some(".tdata".to_string());
    }
    if flags & SHF_TLS != 0 {
        return if sh_type == SHT_NOBITS { Some(".tbss".to_string()) }
            else { Some(".tdata".to_string()) };
    }
    if name.starts_with(".data") {
        return Some(".data".to_string());
    }
    if name.starts_with(".bss") || sh_type == SHT_NOBITS {
        return Some(".bss".to_string());
    }
    if name == ".init_array" || name.starts_with(".init_array.") {
        return Some(".init_array".to_string());
    }
    if name == ".fini_array" || name.starts_with(".fini_array.") {
        return Some(".fini_array".to_string());
    }
    if name.starts_with(".note.") && sh_type == SHT_NOTE {
        return Some(".note".to_string());
    }
    if name.starts_with(".tm_clone_table") {
        return Some(".data".to_string());
    }
    if flags & SHF_ALLOC != 0 {
        if crate::backend::linker_common::is_valid_c_identifier_for_section(name) {
            return Some(name.to_string());
        }
        if flags & SHF_EXECINSTR != 0 { return Some(".text".to_string()); }
        if flags & SHF_WRITE != 0 {
            return if sh_type == SHT_NOBITS { Some(".bss".to_string()) }
                else { Some(".data".to_string()) };
        }
        return Some(".rodata".to_string());
    }
    None
}
