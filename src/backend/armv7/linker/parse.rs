//! ELF32 ARM object file parsing.
//!
//! Handles parsing of relocatable ELF32 .o files, regular archives (.a),
//! and thin archives. Based on the i686 parser, adapted for ARM.

use std::collections::HashMap;

use super::types::*;

/// Parse an ELF32 relocatable object file.
pub(super) fn parse_elf32(data: &[u8], filename: &str) -> Result<InputObject, String> {
    if data.len() < 52 {
        return Err(format!("{}: too small for ELF header", filename));
    }
    if data[0..4] != ELF_MAGIC {
        return Err(format!("{}: not an ELF file", filename));
    }
    if data[4] != ELFCLASS32 {
        return Err(format!("{}: not ELF32", filename));
    }
    if data[5] != ELFDATA2LSB {
        return Err(format!("{}: not little-endian", filename));
    }
    let e_type = read_u16(data, 16);
    if e_type != ET_REL {
        return Err(format!("{}: not a relocatable object (type={})", filename, e_type));
    }
    let e_machine = read_u16(data, 18);
    if e_machine != EM_ARM {
        return Err(format!("{}: not ARM (machine={})", filename, e_machine));
    }

    let e_shoff = read_u32(data, 32) as usize;
    let e_shentsize = read_u16(data, 46) as usize;
    let e_shnum = read_u16(data, 48) as usize;
    let e_shstrndx = read_u16(data, 50) as usize;

    // Parse section headers
    let mut shdrs = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let off = e_shoff + i * e_shentsize;
        shdrs.push(Elf32Shdr {
            name: read_u32(data, off),
            sh_type: read_u32(data, off + 4),
            flags: read_u32(data, off + 8),
            addr: read_u32(data, off + 12),
            offset: read_u32(data, off + 16),
            size: read_u32(data, off + 20),
            link: read_u32(data, off + 24),
            info: read_u32(data, off + 28),
            addralign: read_u32(data, off + 32),
            entsize: read_u32(data, off + 36),
        });
    }

    // Read section name string table
    let shstrtab = &data[shdrs[e_shstrndx].offset as usize..
                         shdrs[e_shstrndx].offset as usize + shdrs[e_shstrndx].size as usize];

    // Find symtab + strtab
    let mut symtab_idx = None;
    for (i, sh) in shdrs.iter().enumerate() {
        if sh.sh_type == SHT_SYMTAB { symtab_idx = Some(i); break; }
    }

    let (sym_data, strtab_data) = if let Some(si) = symtab_idx {
        let sym_sh = &shdrs[si];
        let str_sh = &shdrs[sym_sh.link as usize];
        (
            &data[sym_sh.offset as usize .. sym_sh.offset as usize + sym_sh.size as usize],
            &data[str_sh.offset as usize .. str_sh.offset as usize + str_sh.size as usize],
        )
    } else {
        let empty: &[u8] = &[];
        (empty, empty)
    };

    // Parse symbols
    let mut symbols = Vec::new();
    let sym_entsize = 16;
    let num_syms = sym_data.len() / sym_entsize;
    for i in 0..num_syms {
        let off = i * sym_entsize;
        let name_idx = read_u32(sym_data, off);
        let value = read_u32(sym_data, off + 4);
        let size = read_u32(sym_data, off + 8);
        let info = sym_data[off + 12];
        let other = sym_data[off + 13];
        let shndx = read_u16(sym_data, off + 14);
        let mut name = read_cstr(strtab_data, name_idx as usize).to_string();
        // Some toolchains emit symbols with @PLT suffix; strip it for resolution
        if name.ends_with("@PLT") {
            name.truncate(name.len() - 4);
        }
        symbols.push(InputSymbol {
            name,
            value,
            size,
            binding: info >> 4,
            sym_type: info & 0xf,
            visibility: other & 3,
            section_index: shndx,
        });
    }

    // Build relocation map: target_section_index -> Vec of relocs
    let mut reloc_map: HashMap<usize, Vec<(u32, u32, u32, i32)>> = HashMap::new();
    for sh in &shdrs {
        if sh.sh_type == SHT_REL {
            let target = sh.info as usize;
            let rel_data = &data[sh.offset as usize .. sh.offset as usize + sh.size as usize];
            let entsize = if sh.entsize > 0 { sh.entsize as usize } else { 8 };
            let count = rel_data.len() / entsize;
            let relocs = reloc_map.entry(target).or_default();
            for i in 0..count {
                let off = i * entsize;
                let r_offset = read_u32(rel_data, off);
                let r_info = read_u32(rel_data, off + 4);
                let sym_idx = r_info >> 8;
                let rel_type = r_info & 0xff;
                // ARM uses SHT_REL (no explicit addend); addend is implicit in code
                relocs.push((r_offset, rel_type, sym_idx, 0));
            }
        } else if sh.sh_type == SHT_RELA {
            let target = sh.info as usize;
            let rel_data = &data[sh.offset as usize .. sh.offset as usize + sh.size as usize];
            let entsize = if sh.entsize > 0 { sh.entsize as usize } else { 12 };
            let count = rel_data.len() / entsize;
            let relocs = reloc_map.entry(target).or_default();
            for i in 0..count {
                let off = i * entsize;
                let r_offset = read_u32(rel_data, off);
                let r_info = read_u32(rel_data, off + 4);
                let r_addend = read_i32(rel_data, off + 8);
                let sym_idx = r_info >> 8;
                let rel_type = r_info & 0xff;
                relocs.push((r_offset, rel_type, sym_idx, r_addend));
            }
        }
    }

    // Build sections
    let mut sections = Vec::new();
    for (i, sh) in shdrs.iter().enumerate() {
        let name = read_cstr(shstrtab, sh.name as usize).to_string();
        // Only include sections that contain data or BSS
        match sh.sh_type {
            SHT_PROGBITS | SHT_NOBITS | SHT_INIT_ARRAY | SHT_FINI_ARRAY | SHT_NOTE | SHT_GROUP => {}
            _ => continue,
        }

        let sec_data = if sh.sh_type == SHT_NOBITS {
            vec![0u8; sh.size as usize]
        } else {
            data[sh.offset as usize .. sh.offset as usize + sh.size as usize].to_vec()
        };

        let relocs = reloc_map.remove(&i).unwrap_or_default();

        sections.push(InputSection {
            name,
            sh_type: sh.sh_type,
            flags: sh.flags,
            data: sec_data,
            align: sh.addralign.max(1),
            relocations: relocs,
            input_index: i,
            entsize: sh.entsize,
            link: sh.link,
            info: sh.info,
        });
    }

    Ok(InputObject { sections, symbols, filename: filename.to_string() })
}
