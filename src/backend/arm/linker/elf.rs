//! ELF64 parsing for the AArch64 linker.
//!
//! Reads ELF relocatable object files (.o), static archives (.a), and
//! shared libraries (.so), extracting sections, symbols, and relocations.

// ── ELF64 constants ────────────────────────────────────────────────────

pub const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
pub const ELFCLASS64: u8 = 2;
pub const ELFDATA2LSB: u8 = 1;
pub const ET_REL: u16 = 1;
pub const ET_EXEC: u16 = 2;
pub const ET_DYN: u16 = 3;
pub const EM_AARCH64: u16 = 183;

// Section header types
pub const SHT_NULL: u32 = 0;
pub const SHT_PROGBITS: u32 = 1;
pub const SHT_SYMTAB: u32 = 2;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_RELA: u32 = 4;
pub const SHT_DYNAMIC: u32 = 6;
pub const SHT_NOTE: u32 = 7;
pub const SHT_NOBITS: u32 = 8;
pub const SHT_REL: u32 = 9;
pub const SHT_DYNSYM: u32 = 11;
pub const SHT_INIT_ARRAY: u32 = 14;
pub const SHT_FINI_ARRAY: u32 = 15;
pub const SHT_GROUP: u32 = 17;

// Section header flags
pub const SHF_WRITE: u64 = 0x1;
pub const SHF_ALLOC: u64 = 0x2;
pub const SHF_EXECINSTR: u64 = 0x4;
pub const SHF_TLS: u64 = 0x400;
pub const SHF_EXCLUDE: u64 = 0x80000000;

// Symbol binding
pub const STB_LOCAL: u8 = 0;
pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;

// Symbol type
pub const STT_NOTYPE: u8 = 0;
pub const STT_OBJECT: u8 = 1;
pub const STT_FUNC: u8 = 2;
pub const STT_SECTION: u8 = 3;
pub const STT_FILE: u8 = 4;
pub const STT_COMMON: u8 = 5;
pub const STT_TLS: u8 = 6;
pub const STT_GNU_IFUNC: u8 = 10;

// Special section indices
pub const SHN_UNDEF: u16 = 0;
pub const SHN_ABS: u16 = 0xfff1;
pub const SHN_COMMON: u16 = 0xfff2;

// Program header types
pub const PT_LOAD: u32 = 1;
pub const PT_NOTE: u32 = 4;
pub const PT_TLS: u32 = 7;
pub const PT_GNU_STACK: u32 = 0x6474e551;

// Program header flags
pub const PF_X: u32 = 0x1;
pub const PF_W: u32 = 0x2;
pub const PF_R: u32 = 0x4;

// ── AArch64 relocation types ───────────────────────────────────────────

pub const R_AARCH64_NONE: u32 = 0;
pub const R_AARCH64_ABS64: u32 = 257;     // S + A
pub const R_AARCH64_ABS32: u32 = 258;     // S + A (32-bit)
pub const R_AARCH64_ABS16: u32 = 259;     // S + A (16-bit)
pub const R_AARCH64_PREL64: u32 = 260;    // S + A - P
pub const R_AARCH64_PREL32: u32 = 261;    // S + A - P
pub const R_AARCH64_PREL16: u32 = 262;    // S + A - P
pub const R_AARCH64_ADR_PREL_PG_HI21: u32 = 275;  // Page(S+A) - Page(P)
pub const R_AARCH64_ADR_PREL_LO21: u32 = 274;     // S + A - P
pub const R_AARCH64_ADD_ABS_LO12_NC: u32 = 277;   // (S + A) & 0xFFF
pub const R_AARCH64_LDST8_ABS_LO12_NC: u32 = 278;
pub const R_AARCH64_LDST16_ABS_LO12_NC: u32 = 284;
pub const R_AARCH64_LDST32_ABS_LO12_NC: u32 = 285;
pub const R_AARCH64_LDST64_ABS_LO12_NC: u32 = 286;
pub const R_AARCH64_LDST128_ABS_LO12_NC: u32 = 299;
pub const R_AARCH64_JUMP26: u32 = 282;    // S + A - P (26-bit B)
pub const R_AARCH64_CALL26: u32 = 283;    // S + A - P (26-bit BL)
pub const R_AARCH64_MOVW_UABS_G0_NC: u32 = 264;
pub const R_AARCH64_MOVW_UABS_G1_NC: u32 = 265;
pub const R_AARCH64_MOVW_UABS_G2_NC: u32 = 266;
pub const R_AARCH64_MOVW_UABS_G3: u32 = 267;
pub const R_AARCH64_MOVW_UABS_G0: u32 = 263;
pub const R_AARCH64_ADR_GOT_PAGE: u32 = 311;
pub const R_AARCH64_LD64_GOT_LO12_NC: u32 = 312;
pub const R_AARCH64_CONDBR19: u32 = 280;
pub const R_AARCH64_TSTBR14: u32 = 279;

// ── Data structures ────────────────────────────────────────────────────

/// Parsed ELF section header
#[derive(Debug, Clone)]
pub struct SectionHeader {
    pub name: String,
    pub sh_type: u32,
    pub flags: u64,
    pub offset: u64,
    pub size: u64,
    pub link: u32,
    pub info: u32,
    pub addralign: u64,
    pub entsize: u64,
}

/// Parsed ELF symbol
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
}

impl Symbol {
    pub fn binding(&self) -> u8 { self.info >> 4 }
    pub fn sym_type(&self) -> u8 { self.info & 0xf }
    pub fn is_undefined(&self) -> bool { self.shndx == SHN_UNDEF }
    pub fn is_global(&self) -> bool { self.binding() == STB_GLOBAL }
    pub fn is_weak(&self) -> bool { self.binding() == STB_WEAK }
    pub fn is_local(&self) -> bool { self.binding() == STB_LOCAL }
}

/// Parsed ELF relocation with addend
#[derive(Debug, Clone)]
pub struct Rela {
    pub offset: u64,
    pub sym_idx: u32,
    pub rela_type: u32,
    pub addend: i64,
}

/// Parsed ELF object file
#[derive(Debug)]
pub struct ElfObject {
    pub sections: Vec<SectionHeader>,
    pub symbols: Vec<Symbol>,
    pub section_data: Vec<Vec<u8>>,
    /// Relocations indexed by the section they apply to
    pub relocations: Vec<Vec<Rela>>,
    pub source_name: String,
}

// ── Reading helpers ────────────────────────────────────────────────────

pub fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

pub fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

pub fn read_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
    ])
}

pub fn read_i64(data: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
    ])
}

fn read_string(data: &[u8], offset: usize) -> String {
    if offset >= data.len() { return String::new(); }
    let end = data[offset..].iter().position(|&b| b == 0).unwrap_or(data.len() - offset);
    String::from_utf8_lossy(&data[offset..offset + end]).to_string()
}

// ── Writing helpers ────────────────────────────────────────────────────

pub fn w16(buf: &mut [u8], off: usize, val: u16) {
    if off + 2 <= buf.len() { buf[off..off + 2].copy_from_slice(&val.to_le_bytes()); }
}

pub fn w32(buf: &mut [u8], off: usize, val: u32) {
    if off + 4 <= buf.len() { buf[off..off + 4].copy_from_slice(&val.to_le_bytes()); }
}

pub fn w64(buf: &mut [u8], off: usize, val: u64) {
    if off + 8 <= buf.len() { buf[off..off + 8].copy_from_slice(&val.to_le_bytes()); }
}

pub fn write_bytes(buf: &mut [u8], off: usize, data: &[u8]) {
    let end = off + data.len();
    if end <= buf.len() { buf[off..end].copy_from_slice(data); }
}

// ── ELF parsing ────────────────────────────────────────────────────────

/// Parse an ELF64 relocatable object file (.o) for AArch64
pub fn parse_object(data: &[u8], source_name: &str) -> Result<ElfObject, String> {
    if data.len() < 64 {
        return Err(format!("{}: file too small for ELF header", source_name));
    }
    if data[0..4] != ELF_MAGIC {
        return Err(format!("{}: not an ELF file", source_name));
    }
    if data[4] != ELFCLASS64 {
        return Err(format!("{}: not 64-bit ELF", source_name));
    }
    if data[5] != ELFDATA2LSB {
        return Err(format!("{}: not little-endian ELF", source_name));
    }
    let e_type = read_u16(data, 16);
    if e_type != ET_REL {
        return Err(format!("{}: not a relocatable object (type={})", source_name, e_type));
    }
    let e_machine = read_u16(data, 18);
    if e_machine != EM_AARCH64 {
        return Err(format!("{}: not AArch64 (machine={})", source_name, e_machine));
    }

    let e_shoff = read_u64(data, 40) as usize;
    let e_shentsize = read_u16(data, 58) as usize;
    let e_shnum = read_u16(data, 60) as usize;
    let e_shstrndx = read_u16(data, 62) as usize;

    if e_shoff == 0 || e_shnum == 0 {
        return Err(format!("{}: no section headers", source_name));
    }

    // Parse section headers
    let mut sections = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let off = e_shoff + i * e_shentsize;
        if off + e_shentsize > data.len() {
            return Err(format!("{}: section header {} out of bounds", source_name, i));
        }
        sections.push(SectionHeader {
            name: String::new(),
            sh_type: read_u32(data, off + 4),
            flags: read_u64(data, off + 8),
            offset: read_u64(data, off + 24),
            size: read_u64(data, off + 32),
            link: read_u32(data, off + 40),
            info: read_u32(data, off + 44),
            addralign: read_u64(data, off + 48),
            entsize: read_u64(data, off + 56),
        });
    }

    // Read section name string table
    if e_shstrndx < sections.len() {
        let shstrtab = &sections[e_shstrndx];
        let strtab_off = shstrtab.offset as usize;
        let strtab_size = shstrtab.size as usize;
        if strtab_off + strtab_size <= data.len() {
            let strtab_data = &data[strtab_off..strtab_off + strtab_size];
            let name_idxs: Vec<u32> = (0..e_shnum).map(|i| {
                read_u32(data, e_shoff + i * e_shentsize)
            }).collect();
            for (i, sec) in sections.iter_mut().enumerate() {
                sec.name = read_string(strtab_data, name_idxs[i] as usize);
            }
        }
    }

    // Read section data
    let mut section_data = Vec::with_capacity(e_shnum);
    for sec in &sections {
        if sec.sh_type == SHT_NOBITS || sec.size == 0 {
            section_data.push(Vec::new());
        } else {
            let start = sec.offset as usize;
            let end = start + sec.size as usize;
            if end > data.len() {
                return Err(format!("{}: section '{}' data out of bounds", source_name, sec.name));
            }
            section_data.push(data[start..end].to_vec());
        }
    }

    // Find symbol table
    let mut symbols = Vec::new();
    for i in 0..sections.len() {
        if sections[i].sh_type == SHT_SYMTAB {
            let strtab_idx = sections[i].link as usize;
            let strtab_data = if strtab_idx < section_data.len() {
                &section_data[strtab_idx]
            } else {
                continue;
            };
            let sym_data = &section_data[i];
            let sym_count = sym_data.len() / 24;
            for j in 0..sym_count {
                let off = j * 24;
                if off + 24 > sym_data.len() { break; }
                let name_idx = read_u32(sym_data, off);
                symbols.push(Symbol {
                    name: read_string(strtab_data, name_idx as usize),
                    info: sym_data[off + 4],
                    other: sym_data[off + 5],
                    shndx: read_u16(sym_data, off + 6),
                    value: read_u64(sym_data, off + 8),
                    size: read_u64(sym_data, off + 16),
                });
            }
            break;
        }
    }

    // Parse relocations
    let mut relocations = vec![Vec::new(); e_shnum];
    for i in 0..sections.len() {
        if sections[i].sh_type == SHT_RELA {
            let target_sec = sections[i].info as usize;
            let rela_data = &section_data[i];
            let rela_count = rela_data.len() / 24;
            let mut relas = Vec::with_capacity(rela_count);
            for j in 0..rela_count {
                let off = j * 24;
                if off + 24 > rela_data.len() { break; }
                let r_info = read_u64(rela_data, off + 8);
                relas.push(Rela {
                    offset: read_u64(rela_data, off),
                    sym_idx: (r_info >> 32) as u32,
                    rela_type: (r_info & 0xffffffff) as u32,
                    addend: read_i64(rela_data, off + 16),
                });
            }
            if target_sec < relocations.len() {
                relocations[target_sec] = relas;
            }
        }
    }

    Ok(ElfObject {
        sections,
        symbols,
        section_data,
        relocations,
        source_name: source_name.to_string(),
    })
}

/// Parse an archive (.a) file, returning (member_name, data_offset, size) for each member.
pub fn parse_archive_members(data: &[u8]) -> Result<Vec<(String, usize, usize)>, String> {
    if data.len() < 8 || &data[0..8] != b"!<arch>\n" {
        return Err("not a valid archive file".to_string());
    }

    let mut members = Vec::new();
    let mut pos = 8;
    let mut extended_names: Option<&[u8]> = None;

    while pos + 60 <= data.len() {
        let name_raw = &data[pos..pos + 16];
        let size_str = std::str::from_utf8(&data[pos + 48..pos + 58]).unwrap_or("").trim();
        let magic = &data[pos + 58..pos + 60];
        if magic != b"`\n" { break; }

        let size: usize = size_str.parse().unwrap_or(0);
        let data_start = pos + 60;
        let name_str = std::str::from_utf8(name_raw).unwrap_or("").trim_end();

        if name_str == "/" || name_str == "/SYM64/" {
            // Symbol table - skip
        } else if name_str == "//" {
            extended_names = Some(&data[data_start..data_start + size]);
        } else {
            let member_name = if name_str.starts_with('/') {
                if let Some(ext) = extended_names {
                    let name_off: usize = name_str[1..].trim_end_matches('/').parse().unwrap_or(0);
                    if name_off < ext.len() {
                        let end = ext[name_off..].iter()
                            .position(|&b| b == b'/' || b == b'\n' || b == 0)
                            .unwrap_or(ext.len() - name_off);
                        String::from_utf8_lossy(&ext[name_off..name_off + end]).to_string()
                    } else {
                        name_str.to_string()
                    }
                } else {
                    name_str.to_string()
                }
            } else {
                name_str.trim_end_matches('/').to_string()
            };

            if data_start + size <= data.len() {
                members.push((member_name, data_start, size));
            }
        }

        pos = data_start + size;
        if pos % 2 != 0 { pos += 1; }
    }

    Ok(members)
}

/// Parse a linker script (e.g., libc.so text file with GROUP directive)
pub fn parse_linker_script(content: &str) -> Option<Vec<String>> {
    let group_start = content.find("GROUP")?;
    let paren_start = content[group_start..].find('(')?;
    let paren_end = content[group_start..].find(')')?;
    let inside = &content[group_start + paren_start + 1..group_start + paren_end];

    let mut paths = Vec::new();
    let mut in_as_needed = false;
    for token in inside.split_whitespace() {
        match token {
            "AS_NEEDED" => { in_as_needed = true; continue; }
            "(" => continue,
            ")" => { in_as_needed = false; continue; }
            _ => {}
        }
        if token.starts_with('/') || token.ends_with(".so") || token.ends_with(".a") ||
           token.contains(".so.") {
            if !in_as_needed {
                paths.push(token.to_string());
            }
        }
    }

    if paths.is_empty() { None } else { Some(paths) }
}
