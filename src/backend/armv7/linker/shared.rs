//! Shared library (.so) emission for the ARMv7 linker.
//!
//! Produces ELF32 shared libraries (ET_DYN) with PLT/GOT, PIC relocations,
//! and `.dynamic` section.

use std::collections::HashMap;
use std::path::Path;

use super::types::*;
use super::emit::{layout_section, layout_custom_sections, layout_tls, build_plt_arm};
use super::DynStrTab;
use crate::backend::linker_common;

/// Discover NEEDED shared library dependencies for a shared library build.
pub(super) fn resolve_dynamic_symbols_for_shared(
    inputs: &[InputObject],
    global_symbols: &HashMap<String, LinkerSymbol>,
    needed_sonames: &mut Vec<String>,
    lib_paths: &[String],
) {
    let mut undefined: Vec<String> = Vec::new();
    for obj in inputs.iter() {
        for sym in &obj.symbols {
            if sym.binding == STB_LOCAL { continue; }
            if sym.section_index == SHN_UNDEF && !sym.name.is_empty()
                && !global_symbols.get(&sym.name).map(|gs| gs.is_defined).unwrap_or(false)
                && !undefined.contains(&sym.name)
            {
                undefined.push(sym.name.clone());
            }
        }
    }
    if undefined.is_empty() { return; }

    // Search system libraries for undefined symbols
    let lib_names = ["libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2",
                     "librt.so.1", "libgcc_s.so.1", "ld-linux-armhf.so.3"];
    for lib_name in &lib_names {
        for dir in lib_paths {
            let candidate = format!("{}/{}", dir, lib_name);
            if Path::new(&candidate).exists() {
                if !needed_sonames.contains(&lib_name.to_string()) {
                    needed_sonames.push(lib_name.to_string());
                }
                break;
            }
        }
    }
}

/// Emit an ELF32 shared library.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_shared_library_32(
    inputs: &[InputObject],
    global_symbols: &mut HashMap<String, LinkerSymbol>,
    output_sections: &mut Vec<OutputSection>,
    section_name_to_idx: &HashMap<String, usize>,
    section_map: &SectionMap,
    needed_sonames: &[String],
    output_path: &str,
    soname: Option<String>,
) -> Result<(), String> {
    // Build dynsym with all global symbols exported
    let mut dynstr = DynStrTab::new();
    let _ = dynstr.add("");

    let mut needed_offsets: Vec<u32> = Vec::new();
    for sn in needed_sonames {
        needed_offsets.push(dynstr.add(sn));
    }

    let soname_offset = soname.as_ref().map(|s| dynstr.add(s));

    let mut dynsym_entries: Vec<Elf32Sym> = Vec::new();
    dynsym_entries.push(Elf32Sym { name: 0, value: 0, size: 0, info: 0, other: 0, shndx: 0 });
    let mut dynsym_names: Vec<String> = Vec::new();

    // Export all global/weak defined symbols
    let mut export_syms: Vec<String> = global_symbols.iter()
        .filter(|(_, s)| s.is_defined && s.binding != STB_LOCAL)
        .map(|(n, _)| n.clone())
        .collect();
    export_syms.sort();

    for name in &export_syms {
        let name_off = dynstr.add(name);
        let sym = &global_symbols[name];
        dynsym_entries.push(Elf32Sym {
            name: name_off, value: 0, size: sym.size,
            info: (sym.binding << 4) | sym.sym_type, other: 0,
            shndx: 1, // Will be fixed
        });
        dynsym_names.push(name.clone());
    }

    let dynstr_data = dynstr.as_bytes().to_vec();

    // Layout sections with base address 0 for shared library
    let ehdr_size: u32 = 52;
    let phdr_size: u32 = 32;
    let num_phdrs: u32 = 6;
    let headers_size = ehdr_size + num_phdrs * phdr_size;

    let mut file_offset = headers_size;
    let mut vaddr = headers_size;

    // Text
    let (text_vaddr, _text_size) = layout_section(".text", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_section(".init", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_section(".fini", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_custom_sections(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, SHF_EXECINSTR);

    // Rodata
    file_offset = align_up(file_offset, PAGE_SIZE);
    vaddr = align_up(vaddr, PAGE_SIZE);
    layout_section(".rodata", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_section(".eh_frame", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_custom_sections(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 0);

    // Data
    file_offset = align_up(file_offset, PAGE_SIZE);
    vaddr = align_up(vaddr, PAGE_SIZE);
    let (tls_addr, tls_file_offset, tls_file_size, tls_mem_size, tls_align) =
        layout_tls(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr);

    layout_section(".data", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_section(".init_array", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_section(".fini_array", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_custom_sections(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, SHF_WRITE);

    // Assign symbol addresses
    for (name, sym) in global_symbols.iter_mut() {
        if sym.output_section < output_sections.len() {
            sym.address = output_sections[sym.output_section].addr + sym.section_offset;
        }
    }

    // Update dynsym values
    for (i, entry) in dynsym_entries.iter_mut().enumerate() {
        if i == 0 { continue; }
        let name = &dynsym_names[i - 1];
        if let Some(sym) = global_symbols.get(name) {
            entry.value = sym.address;
        }
    }

    // Dynamic section
    let mut dynamic_data: Vec<u8> = Vec::new();
    for &off in &needed_offsets {
        push_dyn(&mut dynamic_data, DT_NEEDED, off);
    }
    if let Some(off) = soname_offset {
        push_dyn(&mut dynamic_data, DT_SONAME, off);
    }
    push_dyn(&mut dynamic_data, DT_STRTAB, 0); // placeholder
    push_dyn(&mut dynamic_data, DT_SYMTAB, 0); // placeholder
    push_dyn(&mut dynamic_data, DT_STRSZ, dynstr_data.len() as u32);
    push_dyn(&mut dynamic_data, DT_SYMENT, 16);
    push_dyn(&mut dynamic_data, DT_NULL, 0);

    // Build output
    let dynsym_off = file_offset;
    let dynsym_size = (dynsym_entries.len() * 16) as u32;
    file_offset += dynsym_size;
    vaddr += dynsym_size;

    let dynstr_off = file_offset;
    file_offset += dynstr_data.len() as u32;
    vaddr += dynstr_data.len() as u32;

    let dynamic_off = align_up(file_offset, 4);
    file_offset = dynamic_off + dynamic_data.len() as u32;

    let total = file_offset as usize;
    let mut output = vec![0u8; total];

    // ELF header
    output[0..4].copy_from_slice(&ELF_MAGIC);
    output[4] = ELFCLASS32;
    output[5] = ELFDATA2LSB;
    output[6] = EV_CURRENT;
    output[16..18].copy_from_slice(&ET_DYN.to_le_bytes());
    output[18..20].copy_from_slice(&EM_ARM.to_le_bytes());
    output[20..24].copy_from_slice(&1u32.to_le_bytes());
    output[24..28].copy_from_slice(&text_vaddr.to_le_bytes());
    output[28..32].copy_from_slice(&ehdr_size.to_le_bytes());
    output[32..36].copy_from_slice(&0u32.to_le_bytes());
    output[36..40].copy_from_slice(&(EF_ARM_ABI_VER5 | EF_ARM_ABI_FLOAT_HARD).to_le_bytes());
    output[40..42].copy_from_slice(&(ehdr_size as u16).to_le_bytes());
    output[42..44].copy_from_slice(&32u16.to_le_bytes());
    output[44..46].copy_from_slice(&(num_phdrs as u16).to_le_bytes());
    output[46..48].copy_from_slice(&40u16.to_le_bytes());

    // Program headers
    let mut phdr_off = ehdr_size as usize;
    let write_ph = |output: &mut Vec<u8>, off: &mut usize, pt: u32, poff: u32, vaddr: u32, fsz: u32, msz: u32, flags: u32, align: u32| {
        if *off + 32 > output.len() { *off += 32; return; }
        output[*off..*off+4].copy_from_slice(&pt.to_le_bytes());
        output[*off+4..*off+8].copy_from_slice(&poff.to_le_bytes());
        output[*off+8..*off+12].copy_from_slice(&vaddr.to_le_bytes());
        output[*off+12..*off+16].copy_from_slice(&vaddr.to_le_bytes());
        output[*off+16..*off+20].copy_from_slice(&fsz.to_le_bytes());
        output[*off+20..*off+24].copy_from_slice(&msz.to_le_bytes());
        output[*off+24..*off+28].copy_from_slice(&flags.to_le_bytes());
        output[*off+28..*off+32].copy_from_slice(&align.to_le_bytes());
        *off += 32;
    };

    write_ph(&mut output, &mut phdr_off, PT_PHDR, ehdr_size, ehdr_size,
        num_phdrs * phdr_size, num_phdrs * phdr_size, PF_R, 4);
    write_ph(&mut output, &mut phdr_off, PT_LOAD, 0, 0, total as u32, total as u32, PF_R | PF_X, PAGE_SIZE);
    write_ph(&mut output, &mut phdr_off, PT_LOAD, 0, 0, total as u32, total as u32, PF_R | PF_W, PAGE_SIZE);
    write_ph(&mut output, &mut phdr_off, PT_DYNAMIC, dynamic_off, dynamic_off,
        dynamic_data.len() as u32, dynamic_data.len() as u32, PF_R | PF_W, 4);
    write_ph(&mut output, &mut phdr_off, PT_GNU_STACK, 0, 0, 0, 0, PF_R | PF_W, 0x10);
    write_ph(&mut output, &mut phdr_off, PT_LOAD, 0, 0, headers_size, headers_size, PF_R, PAGE_SIZE);

    // Write section data
    for sec in output_sections.iter() {
        if sec.data.is_empty() || sec.file_offset == 0 || sec.sh_type == SHT_NOBITS { continue; }
        let off = sec.file_offset as usize;
        let end = off + sec.data.len();
        if end <= output.len() {
            output[off..end].copy_from_slice(&sec.data);
        }
    }

    // Write dynsym
    let mut off = dynsym_off as usize;
    for entry in &dynsym_entries {
        if off + 16 > output.len() { break; }
        output[off..off+4].copy_from_slice(&entry.name.to_le_bytes());
        output[off+4..off+8].copy_from_slice(&entry.value.to_le_bytes());
        output[off+8..off+12].copy_from_slice(&entry.size.to_le_bytes());
        output[off+12] = entry.info;
        output[off+13] = entry.other;
        output[off+14..off+16].copy_from_slice(&entry.shndx.to_le_bytes());
        off += 16;
    }

    // Write dynstr
    let off = dynstr_off as usize;
    if off + dynstr_data.len() <= output.len() {
        output[off..off + dynstr_data.len()].copy_from_slice(&dynstr_data);
    }

    // Write dynamic
    let off = dynamic_off as usize;
    if off + dynamic_data.len() <= output.len() {
        output[off..off + dynamic_data.len()].copy_from_slice(&dynamic_data);
    }

    std::fs::write(output_path, &output)
        .map_err(|e| format!("failed to write {}: {}", output_path, e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(output_path, std::fs::Permissions::from_mode(0o755));
    }

    Ok(())
}
