//! Executable emission for the ARMv7 linker.
//!
//! Phase 10: lays out segments, assigns addresses, applies relocations,
//! builds PLT/GOT/dynamic sections, and writes the final ELF32 executable.

use std::collections::{HashMap, BTreeSet};

use super::types::*;
use super::reloc::{self, RelocContext, resolve_got_reloc};
use super::DynStrTab;
use crate::backend::linker_common;

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_executable(
    inputs: &[InputObject],
    output_sections: &mut Vec<OutputSection>,
    section_name_to_idx: &HashMap<String, usize>,
    section_map: &SectionMap,
    global_symbols: &mut HashMap<String, LinkerSymbol>,
    _sym_resolution: &HashMap<(usize, usize), String>,
    _dynlib_syms: &HashMap<String, (String, u8, u32, Option<String>, bool, u8)>,
    plt_symbols: &[String],
    got_dyn_symbols: &[String],
    got_local_symbols: &[String],
    num_plt: usize,
    num_got_slots: usize,
    ifunc_symbols: &[String],
    is_static: bool,
    is_nostdlib: bool,
    _needed_libs_param: &[&str],
    output_path: &str,
) -> Result<(), String> {
    let num_ifunc = ifunc_symbols.len();

    // ── Build dynamic symbol/string tables ────────────────────────────────
    let mut needed_libs: Vec<String> = Vec::new();
    if !is_static && !is_nostdlib {
        needed_libs.push("libc.so.6".to_string());
    }
    for sym in global_symbols.values() {
        if sym.is_dynamic && !sym.dynlib.is_empty() && !needed_libs.contains(&sym.dynlib) {
            needed_libs.push(sym.dynlib.clone());
        }
    }

    let mut dynstr = DynStrTab::new();
    let _ = dynstr.add("");
    let mut needed_offsets: Vec<u32> = Vec::new();
    for lib in &needed_libs {
        needed_offsets.push(dynstr.add(lib));
    }

    // Build dynsym entries
    let mut dynsym_entries: Vec<Elf32Sym> = Vec::new();
    dynsym_entries.push(Elf32Sym { name: 0, value: 0, size: 0, info: 0, other: 0, shndx: 0 });
    let mut dynsym_map: HashMap<String, usize> = HashMap::new();
    let mut dynsym_names: Vec<String> = Vec::new();

    for name in plt_symbols {
        let idx = dynsym_entries.len();
        let name_off = dynstr.add(name);
        let (bind, stype) = if let Some(sym) = global_symbols.get(name) {
            (sym.binding, if sym.sym_type != 0 { sym.sym_type } else { STT_FUNC })
        } else {
            (STB_GLOBAL, STT_FUNC)
        };
        dynsym_entries.push(Elf32Sym {
            name: name_off, value: 0, size: 0,
            info: (bind << 4) | stype, other: 0, shndx: SHN_UNDEF,
        });
        dynsym_map.insert(name.clone(), idx);
        dynsym_names.push(name.clone());
    }

    for name in got_dyn_symbols {
        if dynsym_map.contains_key(name) { continue; }
        let idx = dynsym_entries.len();
        let name_off = dynstr.add(name);
        let sym = &global_symbols[name];
        dynsym_entries.push(Elf32Sym {
            name: name_off, value: 0, size: sym.size,
            info: (sym.binding << 4) | sym.sym_type, other: 0, shndx: SHN_UNDEF,
        });
        dynsym_map.insert(name.clone(), idx);
        dynsym_names.push(name.clone());
    }

    // Copy-reloc symbols
    let mut copy_syms: Vec<String> = global_symbols.iter()
        .filter(|(_, s)| s.needs_copy && s.is_dynamic)
        .map(|(n, _)| n.clone())
        .collect();
    copy_syms.sort();
    for name in &copy_syms {
        if dynsym_map.contains_key(name) { continue; }
        let idx = dynsym_entries.len();
        let name_off = dynstr.add(name);
        let sym = &global_symbols[name];
        dynsym_entries.push(Elf32Sym {
            name: name_off, value: 0, size: sym.size,
            info: (STB_GLOBAL << 4) | STT_OBJECT, other: 0, shndx: SHN_UNDEF,
        });
        dynsym_map.insert(name.clone(), idx);
        dynsym_names.push(name.clone());
    }

    // Build version tables
    let mut lib_versions: HashMap<String, BTreeSet<String>> = HashMap::new();
    for name in &dynsym_names {
        if let Some(gs) = global_symbols.get(name) {
            if gs.is_dynamic {
                if let Some(ref ver) = gs.version {
                    lib_versions.entry(gs.dynlib.clone()).or_default().insert(ver.clone());
                }
            }
        }
    }

    let mut ver_index_map: HashMap<(String, String), u16> = HashMap::new();
    let mut ver_idx: u16 = 2;
    let mut lib_ver_list: Vec<(String, Vec<String>)> = Vec::new();
    let mut sorted_libs: Vec<String> = lib_versions.keys().cloned().collect();
    sorted_libs.sort();
    for lib in &sorted_libs {
        let vers: Vec<String> = lib_versions[lib].iter().cloned().collect();
        for v in &vers {
            ver_index_map.insert((lib.clone(), v.clone()), ver_idx);
            ver_idx += 1;
        }
        lib_ver_list.push((lib.clone(), vers));
    }

    // Build dynstr
    let mut dynstr2 = DynStrTab::new();
    let _ = dynstr2.add("");
    for lib in &needed_libs { dynstr2.add(lib); }
    for name in plt_symbols { dynstr2.add(name); }
    for name in got_dyn_symbols { dynstr2.add(name); }
    for name in &copy_syms { dynstr2.add(name); }
    for (_, vers) in &lib_ver_list {
        for v in vers { dynstr2.add(v); }
    }
    let dynstr_data = dynstr2.as_bytes().to_vec();

    let mut needed_offsets2: Vec<u32> = Vec::new();
    for lib in &needed_libs {
        needed_offsets2.push(dynstr2.get_offset(lib));
    }
    for (i, entry) in dynsym_entries.iter_mut().enumerate() {
        if i == 0 { continue; }
        let name = &dynsym_names[i - 1];
        entry.name = dynstr2.get_offset(name);
    }

    // Build versym
    let mut versym_data: Vec<u8> = Vec::new();
    for (i, _) in dynsym_entries.iter().enumerate() {
        if i == 0 {
            versym_data.extend_from_slice(&0u16.to_le_bytes());
        } else {
            let sym_name = if i - 1 < dynsym_names.len() { &dynsym_names[i - 1] } else { "" };
            if let Some(gs) = global_symbols.get(sym_name) {
                if gs.is_dynamic && !gs.dynlib.is_empty() {
                    if let Some(ref ver) = gs.version {
                        let idx = ver_index_map.get(&(gs.dynlib.clone(), ver.clone()))
                            .copied().unwrap_or(1);
                        versym_data.extend_from_slice(&idx.to_le_bytes());
                    } else {
                        versym_data.extend_from_slice(&1u16.to_le_bytes());
                    }
                } else {
                    versym_data.extend_from_slice(&0u16.to_le_bytes());
                }
            } else {
                versym_data.extend_from_slice(&0u16.to_le_bytes());
            }
        }
    }

    // Build verneed
    let mut verneed_data: Vec<u8> = Vec::new();
    let mut verneed_count: u32 = 0;
    for (lib_i, (lib, vers)) in lib_ver_list.iter().enumerate() {
        if !needed_libs.contains(lib) { continue; }
        let lib_name_off = dynstr2.get_offset(lib);
        let is_last_lib = lib_i == lib_ver_list.len() - 1;
        verneed_data.extend_from_slice(&1u16.to_le_bytes());
        verneed_data.extend_from_slice(&(vers.len() as u16).to_le_bytes());
        verneed_data.extend_from_slice(&lib_name_off.to_le_bytes());
        verneed_data.extend_from_slice(&16u32.to_le_bytes());
        let next_off = if is_last_lib { 0u32 } else { 16 + vers.len() as u32 * 16 };
        verneed_data.extend_from_slice(&next_off.to_le_bytes());
        verneed_count += 1;
        for (v_i, ver) in vers.iter().enumerate() {
            let ver_name_off = dynstr2.get_offset(ver);
            let v_idx = ver_index_map[&(lib.clone(), ver.clone())];
            let is_last_ver = v_i == vers.len() - 1;
            verneed_data.extend_from_slice(&linker_common::sysv_hash(ver.as_bytes()).to_le_bytes());
            verneed_data.extend_from_slice(&0u16.to_le_bytes());
            verneed_data.extend_from_slice(&v_idx.to_le_bytes());
            verneed_data.extend_from_slice(&ver_name_off.to_le_bytes());
            let vna_next: u32 = if is_last_ver { 0 } else { 16 };
            verneed_data.extend_from_slice(&vna_next.to_le_bytes());
        }
    }

    // ── Layout ────────────────────────────────────────────────────────────
    let ehdr_size: u32 = 52;
    let phdr_size: u32 = 32;
    let has_tls_sections = output_sections.iter()
        .any(|s| s.flags & SHF_TLS != 0 && s.flags & SHF_ALLOC != 0);

    let mut num_phdrs: u32 = 1; // PHDR
    if !is_static { num_phdrs += 1; } // INTERP
    num_phdrs += 3; // LOAD x3 (text R|X, rodata R, data R|W)
    if !is_static { num_phdrs += 1; } // DYNAMIC
    num_phdrs += 1; // GNU_STACK
    // Note: PT_GNU_EH_FRAME is intentionally omitted — we don't generate
    // .eh_frame_hdr. Emitting a PT_GNU_EH_FRAME with p_vaddr=0 causes glibc's
    // startup code to crash (SIGSEGV at si_addr=0x10) when it tries to
    // dereference the null pointer.
    if has_tls_sections { num_phdrs += 1; }

    let headers_size = ehdr_size + num_phdrs * phdr_size;
    let mut file_offset = headers_size;
    let mut vaddr = BASE_ADDR + headers_size;

    // Interp section
    if !is_static {
        let interp_off = file_offset;
        let interp_vaddr = vaddr;
        file_offset += INTERP.len() as u32;
        vaddr += INTERP.len() as u32;
        // Store interp for later use
        let _ = (interp_off, interp_vaddr);
    }

    // Layout text segment
    let (init_vaddr, init_size) = layout_section(".init", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);

    // PLT goes in the text segment (executable code)
    let plt_header_size: u32 = 20; // ARM PLT0
    let plt_entry_size: u32 = PLT_ENTRY_SIZE;
    let plt_size = if num_plt > 0 { plt_header_size + (num_plt as u32) * plt_entry_size } else { 0 };
    file_offset = align_up(file_offset, 4);
    vaddr = align_up(vaddr, 4);
    let plt_vaddr = vaddr;
    let plt_off = file_offset;
    file_offset += plt_size;
    vaddr += plt_size;

    let (text_vaddr, text_size) = layout_section(".text", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    let (fini_vaddr, fini_size) = layout_section(".fini", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_custom_sections(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, SHF_EXECINSTR);

    let text_seg_end_vaddr = vaddr;

    // Page-align for rodata segment with congruence:
    // Linux requires p_offset % p_align == p_vaddr % p_align for PT_LOAD segments.
    file_offset = align_up(file_offset, PAGE_SIZE);
    vaddr = align_up(vaddr, PAGE_SIZE);
    vaddr = (vaddr & !0xfff) | (file_offset & 0xfff);
    if vaddr <= text_seg_end_vaddr {
        vaddr = align_up(text_seg_end_vaddr, PAGE_SIZE) | (file_offset & 0xfff);
    }
    let rodata_seg_start = vaddr;

    let (_rodata_vaddr, _rodata_size) = layout_section(".rodata", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    let (eh_frame_vaddr, eh_frame_size) = layout_section(".eh_frame", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    let (_note_vaddr, _note_size) = layout_section(".note", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_custom_sections(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 0);

    // Dynsym, dynstr, versym, verneed, hash for dynamic linking
    let mut dynsym_vaddr = 0u32;
    let mut dynsym_off = 0u32;
    let mut dynstr_vaddr = 0u32;
    let mut hash_vaddr = 0u32;
    let mut versym_vaddr = 0u32;
    let mut verneed_vaddr = 0u32;

    if !is_static {
        file_offset = align_up(file_offset, 4);
        vaddr = align_up(vaddr, 4);
        dynsym_vaddr = vaddr;
        dynsym_off = file_offset;
        let dynsym_size = (dynsym_entries.len() * 16) as u32;
        file_offset += dynsym_size;
        vaddr += dynsym_size;

        file_offset = align_up(file_offset, 4);
        vaddr = align_up(vaddr, 4);
        dynstr_vaddr = vaddr;
        file_offset += dynstr_data.len() as u32;
        vaddr += dynstr_data.len() as u32;

        // Hash table (sysv)
        let hash_data = build_sysv_hash(&dynsym_entries, &dynsym_names);
        file_offset = align_up(file_offset, 4);
        vaddr = align_up(vaddr, 4);
        hash_vaddr = vaddr;
        file_offset += hash_data.len() as u32;
        vaddr += hash_data.len() as u32;

        // Versym
        if !versym_data.is_empty() {
            file_offset = align_up(file_offset, 2);
            vaddr = align_up(vaddr, 2);
            versym_vaddr = vaddr;
            file_offset += versym_data.len() as u32;
            vaddr += versym_data.len() as u32;
        }

        // Verneed
        if !verneed_data.is_empty() {
            file_offset = align_up(file_offset, 4);
            vaddr = align_up(vaddr, 4);
            verneed_vaddr = vaddr;
            file_offset += verneed_data.len() as u32;
            vaddr += verneed_data.len() as u32;
        }
    }

    // .rel.plt - relocation entries for PLT (R_ARM_JUMP_SLOT)
    let rel_plt_size = if !is_static && num_plt > 0 { (num_plt as u32) * 8 } else { 0 };
    let (rel_plt_off, rel_plt_vaddr) = if rel_plt_size > 0 {
        file_offset = align_up(file_offset, 4);
        vaddr = align_up(vaddr, 4);
        let off = file_offset;
        let va = vaddr;
        file_offset += rel_plt_size;
        vaddr += rel_plt_size;
        (off, va)
    } else {
        (file_offset, vaddr)
    };

    // .rel.dyn - relocation entries for dynamic symbols (R_ARM_GLOB_DAT, R_ARM_COPY)
    // Count entries needed
    let num_rel_dyn = if !is_static {
        got_dyn_symbols.len() + global_symbols.values().filter(|s| s.needs_copy).count()
    } else { 0 };
    let rel_dyn_size = (num_rel_dyn as u32) * 8;
    let (rel_dyn_off, rel_dyn_vaddr) = if rel_dyn_size > 0 {
        file_offset = align_up(file_offset, 4);
        vaddr = align_up(vaddr, 4);
        let off = file_offset;
        let va = vaddr;
        file_offset += rel_dyn_size;
        vaddr += rel_dyn_size;
        (off, va)
    } else {
        (file_offset, vaddr)
    };

    let rodata_seg_end = vaddr;

    // Page-align for data segment with congruence
    file_offset = align_up(file_offset, PAGE_SIZE);
    vaddr = align_up(vaddr, PAGE_SIZE);
    vaddr = (vaddr & !0xfff) | (file_offset & 0xfff);
    if vaddr <= rodata_seg_end {
        vaddr = align_up(rodata_seg_end, PAGE_SIZE) | (file_offset & 0xfff);
    }
    let data_seg_start = vaddr;

    // TLS sections
    let (tls_addr, tls_file_offset, tls_file_size, tls_mem_size, tls_align) =
        layout_tls(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr);

    // GOT (.got + .got.plt)
    let got_reserved: usize = 1; // Reserved GOT entries (GOT[0] = _DYNAMIC)
    let gotplt_reserved: u32 = 3;
    // num_got_slots already accounts for TLS GD symbols needing 2 slots each
    let got_size = ((got_reserved + num_got_slots) * 4) as u32;
    let gotplt_size = (gotplt_reserved + num_plt as u32) * 4;

    file_offset = align_up(file_offset, 4);
    vaddr = align_up(vaddr, 4);
    let got_vaddr = vaddr;
    let got_off = file_offset;
    file_offset += got_size;
    vaddr += got_size;

    let gotplt_vaddr = vaddr;
    let gotplt_off = file_offset;
    file_offset += gotplt_size;
    vaddr += gotplt_size;

    let got_base = got_vaddr;

    // Data sections (PLT was already placed in text segment above)
    let (data_vaddr, data_size) = layout_section(".data", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    let (init_array_vaddr, init_array_size) = layout_section(".init_array", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    let (fini_array_vaddr, fini_array_size) = layout_section(".fini_array", section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, 4);
    layout_custom_sections(section_name_to_idx, output_sections, &mut file_offset, &mut vaddr, SHF_WRITE);

    // Dynamic section
    let num_dyn_entries = if !is_static {
        count_dynamic_entries(&needed_libs, init_vaddr, init_size, fini_vaddr, fini_size,
            init_array_size, fini_array_size, num_plt, num_rel_dyn,
            versym_data.len() as u32, verneed_data.len() as u32, 0)
    } else { 0 };
    let dynamic_size = num_dyn_entries * 8;

    file_offset = align_up(file_offset, 4);
    vaddr = align_up(vaddr, 4);
    let dynamic_vaddr = if !is_static { vaddr } else { 0 };
    let dynamic_off = file_offset;
    file_offset += dynamic_size;
    vaddr += dynamic_size;

    // BSS
    let bss_vaddr = vaddr;
    if let Some(&idx) = section_name_to_idx.get(".bss") {
        let a = output_sections[idx].align.max(4);
        vaddr = align_up(vaddr, a);
        output_sections[idx].addr = vaddr;
        output_sections[idx].file_offset = file_offset;
        vaddr += output_sections[idx].data.len() as u32;
    }

    // Copy relocations in BSS
    let mut copy_offset = vaddr;
    for name in &copy_syms {
        if let Some(sym) = global_symbols.get_mut(name) {
            let align = if sym.size >= 8 { 8 } else if sym.size >= 4 { 4 } else { 1 };
            copy_offset = align_up(copy_offset, align);
            sym.copy_addr = copy_offset;
            sym.address = copy_offset; // Must set address for relocations to resolve correctly
            sym.is_defined = true;
            copy_offset += sym.size.max(4); // Minimum 4 bytes allocation
        }
    }

    let data_seg_end = copy_offset;

    // IRELATIVE relocations for IFUNC in static mode
    let rel_iplt_size = (num_ifunc * 8) as u32;
    let rel_iplt_vaddr = if num_ifunc > 0 && dynamic_vaddr >= rel_iplt_size {
        dynamic_vaddr - rel_iplt_size
    } else {
        0
    };

    // ── Assign symbol addresses ──────────────────────────────────────────
    assign_symbol_addresses(
        global_symbols, output_sections,
        got_base, plt_vaddr, plt_header_size, plt_entry_size,
        bss_vaddr, data_seg_end, data_seg_start,
        text_seg_end_vaddr, dynamic_vaddr, is_static,
        init_array_vaddr, init_array_size,
        fini_array_vaddr, fini_array_size,
        rel_iplt_vaddr, rel_iplt_size,
    );

    // ── Debug: print key symbol info for startup debugging ──────────────
    for name in &["_start", "__libc_start_main", "main", "__libc_start_call_main", "_init", "_fini", "printf"] {
        if let Some(gs) = global_symbols.get(*name) {
            eprintln!("debug sym: {} addr=0x{:x} is_thumb={} sym_type={} is_defined={} is_abs={} output_section={}",
                name, gs.address, gs.is_thumb, gs.sym_type, gs.is_defined, gs.is_abs, gs.output_section);
        }
    }

    // ── Apply relocations ────────────────────────────────────────────────
    let mut reloc_ctx = RelocContext {
        global_symbols,
        output_sections,
        section_map,
        got_base,
        got_vaddr,
        gotplt_vaddr,
        got_reserved,
        gotplt_reserved,
        plt_vaddr,
        plt_header_size,
        plt_entry_size,
        num_plt,
        tls_addr,
        tls_mem_size,
        has_tls: has_tls_sections,
    };
    let _text_relocs = reloc::apply_relocations(inputs, &mut reloc_ctx)?;

    let output_sections = reloc_ctx.output_sections;
    let global_symbols = reloc_ctx.global_symbols;

    // Entry point — check if _start is a Thumb function (bit 0 of original st_value)
    // ARM ELF: kernel uses bit 0 of e_entry to determine starting execution mode.
    let start_is_thumb = inputs.iter().any(|obj| {
        obj.symbols.iter().any(|sym| sym.name == "_start" && (sym.sym_type == STT_FUNC || sym.sym_type == STT_GNU_IFUNC) && (sym.value & 1) != 0)
    });
    let entry = global_symbols.get("_start")
        .map(|s| if start_is_thumb { s.address | 1 } else { s.address })
        .unwrap_or(text_vaddr);

    // ── Build GOT ────────────────────────────────────────────────────────
    let mut got_data = vec![0u8; got_size as usize];
    // GOT[0] = dynamic section address
    if !is_static && got_data.len() >= 4 {
        got_data[0..4].copy_from_slice(&dynamic_vaddr.to_le_bytes());
    }
    resolve_got_reloc(
        &mut got_data, got_reserved,
        got_dyn_symbols, got_local_symbols, global_symbols,
        has_tls_sections, tls_addr, tls_mem_size,
    );

    let mut gotplt_data = vec![0u8; gotplt_size as usize];
    if !is_static && gotplt_data.len() >= 4 {
        gotplt_data[0..4].copy_from_slice(&dynamic_vaddr.to_le_bytes());
    }
    // GOT.PLT entries: each points to PLT entry + 6 (lazy binding stub)
    for i in 0..num_plt {
        let off = (gotplt_reserved as usize + i) * 4;
        if off + 4 <= gotplt_data.len() {
            let val = plt_vaddr + plt_header_size + (i as u32) * plt_entry_size + 12;
            gotplt_data[off..off+4].copy_from_slice(&val.to_le_bytes());
        }
    }

    // Patch dynsym entries for copy-reloc symbols with resolved addresses
    for name in &copy_syms {
        if let Some(sym) = global_symbols.get(name) {
            if let Some(&idx) = dynsym_map.get(name) {
                dynsym_entries[idx].value = sym.copy_addr;
                dynsym_entries[idx].shndx = 1; // Mark as defined (non-SHN_UNDEF)
            }
        }
    }

    // ── Build PLT ────────────────────────────────────────────────────────
    let plt_data = build_plt_arm(num_plt, plt_vaddr, plt_header_size, plt_entry_size,
        gotplt_vaddr, gotplt_reserved);

    // ── Build .rel.plt (R_ARM_JUMP_SLOT entries for each PLT symbol) ────
    let mut rel_plt_data: Vec<u8> = Vec::new();
    if !is_static {
        for (i, name) in plt_symbols.iter().enumerate() {
            let gotplt_entry_addr = gotplt_vaddr + (gotplt_reserved + i as u32) * 4;
            let dynsym_idx = dynsym_map.get(name).copied().unwrap_or(0) as u32;
            let r_info = (dynsym_idx << 8) | R_ARM_JUMP_SLOT;
            rel_plt_data.extend_from_slice(&gotplt_entry_addr.to_le_bytes());
            rel_plt_data.extend_from_slice(&r_info.to_le_bytes());
        }
    }

    // ── Build .rel.dyn (R_ARM_GLOB_DAT and R_ARM_COPY entries) ──────────
    let mut rel_dyn_data: Vec<u8> = Vec::new();
    if !is_static {
        // GLOB_DAT entries for dynamic GOT symbols
        for (i, name) in got_dyn_symbols.iter().enumerate() {
            let got_entry_addr = got_vaddr + ((got_reserved + i) as u32) * 4;
            let dynsym_idx = dynsym_map.get(name).copied().unwrap_or(0) as u32;
            let r_info = (dynsym_idx << 8) | R_ARM_GLOB_DAT;
            rel_dyn_data.extend_from_slice(&got_entry_addr.to_le_bytes());
            rel_dyn_data.extend_from_slice(&r_info.to_le_bytes());
        }
        // COPY entries for symbols that need copy relocation
        for (name, sym) in global_symbols.iter() {
            if sym.needs_copy {
                let dynsym_idx = dynsym_map.get(name).copied().unwrap_or(0) as u32;
                let r_info = (dynsym_idx << 8) | R_ARM_COPY;
                rel_dyn_data.extend_from_slice(&sym.copy_addr.to_le_bytes());
                rel_dyn_data.extend_from_slice(&r_info.to_le_bytes());
            }
        }
    }

    // ── Build dynamic section ────────────────────────────────────────────
    let mut dynamic_data: Vec<u8> = Vec::new();
    if !is_static {
        for &off in &needed_offsets2 {
            push_dyn(&mut dynamic_data, DT_NEEDED, off);
        }
        push_dyn(&mut dynamic_data, DT_STRTAB, dynstr_vaddr);
        push_dyn(&mut dynamic_data, DT_SYMTAB, dynsym_vaddr);
        push_dyn(&mut dynamic_data, DT_STRSZ, dynstr_data.len() as u32);
        push_dyn(&mut dynamic_data, DT_SYMENT, 16);
        if init_vaddr != 0 && init_size > 0 { push_dyn(&mut dynamic_data, DT_INIT, init_vaddr); }
        if fini_vaddr != 0 && fini_size > 0 { push_dyn(&mut dynamic_data, DT_FINI, fini_vaddr); }
        if init_array_size > 0 {
            push_dyn(&mut dynamic_data, DT_INIT_ARRAY, init_array_vaddr);
            push_dyn(&mut dynamic_data, DT_INIT_ARRAYSZ, init_array_size);
        }
        if fini_array_size > 0 {
            push_dyn(&mut dynamic_data, DT_FINI_ARRAY, fini_array_vaddr);
            push_dyn(&mut dynamic_data, DT_FINI_ARRAYSZ, fini_array_size);
        }
        push_dyn(&mut dynamic_data, DT_PLTGOT, gotplt_vaddr);
        push_dyn(&mut dynamic_data, DT_DEBUG, 0);
        if num_plt > 0 {
            push_dyn(&mut dynamic_data, DT_PLTRELSZ, rel_plt_size);
            push_dyn(&mut dynamic_data, DT_PLTREL, 17); // DT_REL
            push_dyn(&mut dynamic_data, DT_JMPREL, rel_plt_vaddr);
        }
        if num_rel_dyn > 0 {
            push_dyn(&mut dynamic_data, 17, rel_dyn_vaddr); // DT_REL
            push_dyn(&mut dynamic_data, 18, rel_dyn_size);  // DT_RELSZ
            push_dyn(&mut dynamic_data, 19, 8);             // DT_RELENT
        }
        if hash_vaddr != 0 {
            push_dyn(&mut dynamic_data, 4, hash_vaddr); // DT_HASH
        }
        if versym_vaddr != 0 {
            push_dyn(&mut dynamic_data, DT_VERSYM, versym_vaddr);
        }
        if verneed_vaddr != 0 {
            push_dyn(&mut dynamic_data, DT_VERNEED, verneed_vaddr);
            push_dyn(&mut dynamic_data, DT_VERNEEDNUM, verneed_count);
        }
        push_dyn(&mut dynamic_data, DT_NULL, 0);
    }

    // ── Write ELF output ────────────────────────────────────────────────
    let total_file_size = file_offset as usize;
    let mut output = vec![0u8; total_file_size];

    // Write ELF header
    write_elf_header(&mut output, entry, ehdr_size, num_phdrs);

    // Write program headers
    let mut phdr_offset = ehdr_size as usize;

    // PHDR
    write_phdr32(&mut output, &mut phdr_offset, PT_PHDR, ehdr_size, BASE_ADDR + ehdr_size, BASE_ADDR + ehdr_size,
        num_phdrs * phdr_size, num_phdrs * phdr_size, PF_R, 4);

    // INTERP
    if !is_static {
        let interp_off = headers_size;
        let interp_vaddr = BASE_ADDR + headers_size;
        write_phdr32(&mut output, &mut phdr_offset, PT_INTERP, interp_off, interp_vaddr, interp_vaddr,
            INTERP.len() as u32, INTERP.len() as u32, PF_R, 1);
    }

    // Text LOAD segment
    write_phdr32(&mut output, &mut phdr_offset, PT_LOAD, 0, BASE_ADDR, BASE_ADDR,
        text_seg_end_vaddr - BASE_ADDR, text_seg_end_vaddr - BASE_ADDR, PF_R | PF_X, PAGE_SIZE);

    // Rodata LOAD segment
    let rodata_file_off = align_up(text_seg_end_vaddr - BASE_ADDR, PAGE_SIZE);
    write_phdr32(&mut output, &mut phdr_offset, PT_LOAD,
        rodata_file_off, rodata_seg_start, rodata_seg_start,
        rodata_seg_end - rodata_seg_start, rodata_seg_end - rodata_seg_start, PF_R, PAGE_SIZE);

    // Data LOAD segment
    let data_file_off = align_up(rodata_file_off + rodata_seg_end - rodata_seg_start, PAGE_SIZE);
    let data_filesz = file_offset - data_file_off;
    let data_memsz = data_seg_end - data_seg_start;
    write_phdr32(&mut output, &mut phdr_offset, PT_LOAD,
        data_file_off, data_seg_start, data_seg_start,
        data_filesz, data_memsz, PF_R | PF_W, PAGE_SIZE);

    // DYNAMIC
    if !is_static {
        write_phdr32(&mut output, &mut phdr_offset, PT_DYNAMIC,
            dynamic_off, dynamic_vaddr, dynamic_vaddr,
            dynamic_data.len() as u32, dynamic_data.len() as u32, PF_R | PF_W, 4);
    }

    // TLS
    if has_tls_sections {
        write_phdr32(&mut output, &mut phdr_offset, PT_TLS,
            tls_file_offset, tls_addr, tls_addr,
            tls_file_size, tls_mem_size, PF_R, tls_align.max(1));
    }

    // GNU_STACK
    write_phdr32(&mut output, &mut phdr_offset, PT_GNU_STACK,
        0, 0, 0, 0, 0, PF_R | PF_W, 0x10);

    // Write interp
    if !is_static {
        let off = headers_size as usize;
        if off + INTERP.len() <= output.len() {
            output[off..off + INTERP.len()].copy_from_slice(INTERP);
        }
    }

    // Write section data
    for sec in output_sections.iter() {
        if sec.data.is_empty() || sec.file_offset == 0 { continue; }
        if sec.sh_type == SHT_NOBITS { continue; }
        let off = sec.file_offset as usize;
        let end = off + sec.data.len();
        if end <= output.len() {
            output[off..end].copy_from_slice(&sec.data);
        }
    }

    // Write GOT
    if got_size > 0 && (got_off as usize + got_data.len()) <= output.len() {
        output[got_off as usize..got_off as usize + got_data.len()].copy_from_slice(&got_data);
    }

    // Write GOT.PLT
    if gotplt_size > 0 && (gotplt_off as usize + gotplt_data.len()) <= output.len() {
        output[gotplt_off as usize..gotplt_off as usize + gotplt_data.len()].copy_from_slice(&gotplt_data);
    }

    // Write PLT
    if plt_size > 0 && (plt_off as usize + plt_data.len()) <= output.len() {
        output[plt_off as usize..plt_off as usize + plt_data.len()].copy_from_slice(&plt_data);
    }

    // Write dynamic section
    if !is_static && dynamic_data.len() > 0 {
        let off = dynamic_off as usize;
        if off + dynamic_data.len() <= output.len() {
            output[off..off + dynamic_data.len()].copy_from_slice(&dynamic_data);
        }
    }

    // Write dynsym, dynstr, hash, versym, verneed
    if !is_static {
        // dynsym
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

        // dynstr
        let dynstr_off = (dynsym_off + (dynsym_entries.len() * 16) as u32) as usize;
        let dynstr_off = align_up(dynstr_off as u32, 4) as usize;
        if dynstr_off + dynstr_data.len() <= output.len() {
            output[dynstr_off..dynstr_off + dynstr_data.len()].copy_from_slice(&dynstr_data);
        }

        // hash
        let hash_data = build_sysv_hash(&dynsym_entries, &dynsym_names);
        let hash_off = align_up((dynstr_off + dynstr_data.len()) as u32, 4) as usize;
        if hash_off + hash_data.len() <= output.len() {
            output[hash_off..hash_off + hash_data.len()].copy_from_slice(&hash_data);
        }

        // versym
        if !versym_data.is_empty() && versym_vaddr != 0 {
            let off = (versym_vaddr - BASE_ADDR) as usize;
            if off + versym_data.len() <= output.len() {
                output[off..off + versym_data.len()].copy_from_slice(&versym_data);
            }
        }

        // verneed
        if !verneed_data.is_empty() && verneed_vaddr != 0 {
            let off = (verneed_vaddr - BASE_ADDR) as usize;
            if off + verneed_data.len() <= output.len() {
                output[off..off + verneed_data.len()].copy_from_slice(&verneed_data);
            }
        }

        // Write .rel.plt
        if !rel_plt_data.is_empty() {
            let off = rel_plt_off as usize;
            if off + rel_plt_data.len() <= output.len() {
                output[off..off + rel_plt_data.len()].copy_from_slice(&rel_plt_data);
            }
        }

        // Write .rel.dyn
        if !rel_dyn_data.is_empty() {
            let off = rel_dyn_off as usize;
            if off + rel_dyn_data.len() <= output.len() {
                output[off..off + rel_dyn_data.len()].copy_from_slice(&rel_dyn_data);
            }
        }
    }

    // Write to file
    std::fs::write(output_path, &output)
        .map_err(|e| format!("failed to write {}: {}", output_path, e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(output_path, std::fs::Permissions::from_mode(0o755));
    }

    Ok(())
}

// ── Helper functions ─────────────────────────────────────────────────────────

pub(super) fn layout_section(
    name: &str,
    section_name_to_idx: &HashMap<String, usize>,
    output_sections: &mut [OutputSection],
    file_offset: &mut u32,
    vaddr: &mut u32,
    min_align: u32,
) -> (u32, u32) {
    if let Some(idx) = section_name_to_idx.get(name).copied() {
        let a = output_sections[idx].align.max(min_align);
        *file_offset = align_up(*file_offset, a);
        *vaddr = align_up(*vaddr, a);
        let sec_vaddr = *vaddr;
        let sec_size = output_sections[idx].data.len() as u32;
        output_sections[idx].addr = *vaddr;
        output_sections[idx].file_offset = *file_offset;
        *file_offset += sec_size;
        *vaddr += sec_size;
        (sec_vaddr, sec_size)
    } else {
        (0, 0)
    }
}

pub(super) fn layout_custom_sections(
    section_name_to_idx: &HashMap<String, usize>,
    output_sections: &mut [OutputSection],
    file_offset: &mut u32,
    vaddr: &mut u32,
    required_flag: u32,
) {
    let standard_sections: &[&str] = &[
        ".text", ".rodata", ".data", ".bss", ".init", ".fini",
        ".init_array", ".fini_array", ".eh_frame", ".note",
        ".tdata", ".tbss",
    ];
    let mut custom: Vec<String> = section_name_to_idx.keys()
        .filter(|name| {
            if standard_sections.contains(&name.as_str()) { return false; }
            let idx = match section_name_to_idx.get(name.as_str()) {
                Some(&i) => i,
                None => return false,
            };
            let sec = &output_sections[idx];
            if sec.flags & SHF_ALLOC == 0 { return false; }
            match required_flag {
                0 => sec.flags & SHF_WRITE == 0 && sec.flags & SHF_EXECINSTR == 0,
                f => sec.flags & f != 0 && (f != SHF_WRITE || sec.flags & SHF_EXECINSTR == 0),
            }
        })
        .cloned()
        .collect();
    custom.sort();
    for name in &custom {
        layout_section(name, section_name_to_idx, output_sections, file_offset, vaddr, 4);
    }
}

pub(super) fn layout_tls(
    section_name_to_idx: &HashMap<String, usize>,
    output_sections: &mut [OutputSection],
    file_offset: &mut u32,
    vaddr: &mut u32,
) -> (u32, u32, u32, u32, u32) {
    let mut tls_addr = 0u32;
    let mut tls_file_offset = 0u32;
    let mut tls_file_size = 0u32;
    let mut tls_mem_size = 0u32;
    let mut tls_align = 1u32;

    if let Some(&idx) = section_name_to_idx.get(".tdata") {
        let a = output_sections[idx].align.max(4);
        *file_offset = align_up(*file_offset, a);
        *vaddr = align_up(*vaddr, a);
        output_sections[idx].addr = *vaddr;
        output_sections[idx].file_offset = *file_offset;
        tls_addr = *vaddr;
        tls_file_offset = *file_offset;
        tls_align = a;
        let sz = output_sections[idx].data.len() as u32;
        tls_file_size = sz;
        tls_mem_size = sz;
        *file_offset += sz;
        *vaddr += sz;
    }
    if let Some(&idx) = section_name_to_idx.get(".tbss") {
        let a = output_sections[idx].align.max(4);
        let aligned = align_up(tls_mem_size, a);
        if tls_addr == 0 {
            tls_addr = align_up(*vaddr, a);
            tls_file_offset = *file_offset;
            tls_align = a;
        }
        output_sections[idx].addr = tls_addr + aligned;
        output_sections[idx].file_offset = *file_offset;
        tls_mem_size = aligned + output_sections[idx].data.len() as u32;
        if a > tls_align { tls_align = a; }
    }
    tls_mem_size = align_up(tls_mem_size, tls_align);
    (tls_addr, tls_file_offset, tls_file_size, tls_mem_size, tls_align)
}

pub(super) fn build_plt_arm(
    num_plt: usize, plt_vaddr: u32, plt_header_size: u32, plt_entry_size: u32,
    gotplt_vaddr: u32, gotplt_reserved: u32,
) -> Vec<u8> {
    let mut plt_data: Vec<u8> = Vec::new();
    if num_plt == 0 { return plt_data; }

    // PLT[0]: ARM resolver stub
    // str lr, [sp, #-4]!         ; push lr
    // ldr lr, [pc, #4]           ; load GOT offset
    // add lr, pc, lr             ; compute GOT address
    // ldr pc, [lr, #8]!          ; jump to resolver
    // <GOT offset word>
    let plt0_vaddr = plt_vaddr;
    let got_offset = (gotplt_vaddr as i64 - plt0_vaddr as i64 - 16) as i32;
    plt_data.extend_from_slice(&0xe52de004u32.to_le_bytes()); // push {lr}
    plt_data.extend_from_slice(&0xe59fe004u32.to_le_bytes()); // ldr lr, [pc, #4]
    plt_data.extend_from_slice(&0xe08fe00eu32.to_le_bytes()); // add lr, pc, lr
    plt_data.extend_from_slice(&0xe5bef008u32.to_le_bytes()); // ldr pc, [lr, #8]!
    plt_data.extend_from_slice(&(got_offset as u32).to_le_bytes());

    // PLT[N]
    for i in 0..num_plt {
        let entry_vaddr = plt_vaddr + plt_header_size + (i as u32) * plt_entry_size;
        let got_entry = gotplt_vaddr + (gotplt_reserved + i as u32) * 4;
        let offset = (got_entry as i64 - entry_vaddr as i64 - 12) as i32;

        // ldr ip, [pc, #4]       ; load GOT entry offset
        // add ip, pc, ip         ; compute GOT entry address
        // ldr pc, [ip]           ; indirect jump
        // <offset word>
        plt_data.extend_from_slice(&0xe59fc004u32.to_le_bytes());
        plt_data.extend_from_slice(&0xe08fc00cu32.to_le_bytes());
        plt_data.extend_from_slice(&0xe59cf000u32.to_le_bytes());
        plt_data.extend_from_slice(&(offset as u32).to_le_bytes());
    }

    plt_data
}

fn count_dynamic_entries(
    needed_libs: &[String],
    init_vaddr: u32, init_size: u32,
    fini_vaddr: u32, fini_size: u32,
    init_array_size: u32, fini_array_size: u32,
    num_plt: usize, num_rel_dyn: usize,
    versym_size: u32, verneed_size: u32,
    num_text_relocs: usize,
) -> u32 {
    let mut n: u32 = needed_libs.len() as u32;
    n += 5; // STRTAB, SYMTAB, STRSZ, SYMENT, PLTGOT
    n += 1; // HASH
    if init_vaddr != 0 && init_size > 0 { n += 1; }
    if fini_vaddr != 0 && fini_size > 0 { n += 1; }
    if init_array_size > 0 { n += 2; }
    if fini_array_size > 0 { n += 2; }
    n += 1; // DEBUG
    if num_plt > 0 { n += 3; }
    if num_rel_dyn > 0 { n += 3; }
    if versym_size > 0 { n += 1; }  // DT_VERSYM
    if verneed_size > 0 { n += 2; } // DT_VERNEED + DT_VERNEEDNUM
    if num_text_relocs > 0 { n += 1; }
    n += 1; // DT_NULL
    n
}

fn assign_symbol_addresses(
    global_symbols: &mut HashMap<String, LinkerSymbol>,
    output_sections: &[OutputSection],
    got_base: u32,
    plt_vaddr: u32, plt_header_size: u32, plt_entry_size: u32,
    bss_vaddr: u32, data_seg_vaddr_end: u32, data_seg_vaddr_start: u32,
    text_seg_vaddr_end: u32, dynamic_vaddr: u32, is_static: bool,
    init_array_vaddr: u32, init_array_size: u32,
    fini_array_vaddr: u32, fini_array_size: u32,
    rel_iplt_vaddr: u32, rel_iplt_size: u32,
) {
    global_symbols.entry("_GLOBAL_OFFSET_TABLE_".to_string()).or_insert(LinkerSymbol {
        address: got_base, size: 0, sym_type: STT_OBJECT, binding: STB_LOCAL,
        visibility: STV_DEFAULT, is_defined: true, needs_plt: false, needs_got: false,
        needs_tls_gd: false,
        is_thumb: false, is_abs: true,
        output_section: usize::MAX, section_offset: 0, plt_index: 0, got_index: 0,
        is_dynamic: false, dynlib: String::new(), needs_copy: false, copy_addr: 0,
        version: None, uses_textrel: false,
    });
    if let Some(sym) = global_symbols.get_mut("_GLOBAL_OFFSET_TABLE_") {
        sym.address = got_base;
        sym.is_defined = true;
    }

    let linker_addrs = LinkerSymbolAddresses {
        base_addr: BASE_ADDR as u64,
        got_addr: got_base as u64,
        dynamic_addr: if is_static { 0 } else { dynamic_vaddr as u64 },
        bss_addr: bss_vaddr as u64,
        bss_size: (data_seg_vaddr_end - bss_vaddr) as u64,
        text_end: text_seg_vaddr_end as u64,
        data_start: data_seg_vaddr_start as u64,
        init_array_start: init_array_vaddr as u64,
        init_array_size: init_array_size as u64,
        fini_array_start: fini_array_vaddr as u64,
        fini_array_size: fini_array_size as u64,
        preinit_array_start: 0,
        preinit_array_size: 0,
        rela_iplt_start: rel_iplt_vaddr as u64,
        rela_iplt_size: rel_iplt_size as u64,
    };
    let standard_syms = get_standard_linker_symbols(&linker_addrs);
    let linker_sym_map: HashMap<&str, u64> = standard_syms.iter()
        .filter(|s| !s.name.starts_with("__rela_iplt"))
        .map(|s| (s.name, s.value))
        .collect();

    for (name, sym) in global_symbols.iter_mut() {
        if sym.is_dynamic {
            if sym.needs_plt {
                sym.address = plt_vaddr + plt_header_size + (sym.plt_index as u32) * plt_entry_size;
            }
            continue;
        }
        if sym.is_abs {
            // SHN_ABS symbols: their address IS their absolute value,
            // stored in section_offset (= sym.value from the input ELF).
            // No section allocation needed — just use the value directly.
            sym.address = sym.section_offset;
        } else if sym.output_section < output_sections.len() {
            sym.address = output_sections[sym.output_section].addr + sym.section_offset;
        }
        if let Some(&value) = linker_sym_map.get(name.as_str()) {
            sym.address = value as u32;
            if name == "__dso_handle" { sym.is_defined = true; }
        }
        match name.as_str() {
            "__bss_start__" => sym.address = bss_vaddr,
            "edata" => sym.address = bss_vaddr,
            "end" | "__end__" => sym.address = data_seg_vaddr_end,
            "__rel_iplt_start" => sym.address = rel_iplt_vaddr,
            "__rel_iplt_end" => sym.address = rel_iplt_vaddr + rel_iplt_size,
            // ARM .ARM.exidx section boundary symbols — since we exclude
            // .ARM.exidx from output, both point to the same address (empty range).
            "__exidx_start" | "__exidx_end" => {
                sym.address = text_seg_vaddr_end;
                sym.is_defined = true;
            }
            _ => {}
        }
        if let Some(sec_name) = name.strip_prefix("__start_") {
            if linker_common::is_valid_c_identifier_for_section(sec_name) {
                if let Some(sec) = output_sections.iter().find(|s| s.name == sec_name) {
                    sym.address = sec.addr;
                    sym.is_defined = true;
                }
            }
        } else if let Some(sec_name) = name.strip_prefix("__stop_") {
            if linker_common::is_valid_c_identifier_for_section(sec_name) {
                if let Some(sec) = output_sections.iter().find(|s| s.name == sec_name) {
                    sym.address = sec.addr + sec.data.len() as u32;
                    sym.is_defined = true;
                }
            }
        }
    }
}

fn write_elf_header(output: &mut [u8], entry_point: u32, ehdr_size: u32, num_phdrs: u32) {
    output[0..4].copy_from_slice(&ELF_MAGIC);
    output[4] = ELFCLASS32;
    output[5] = ELFDATA2LSB;
    output[6] = EV_CURRENT;
    output[7] = 0;
    output[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
    output[18..20].copy_from_slice(&EM_ARM.to_le_bytes());
    output[20..24].copy_from_slice(&1u32.to_le_bytes());
    output[24..28].copy_from_slice(&entry_point.to_le_bytes());
    output[28..32].copy_from_slice(&ehdr_size.to_le_bytes());
    output[32..36].copy_from_slice(&0u32.to_le_bytes());
    output[36..40].copy_from_slice(&(EF_ARM_ABI_VER5 | EF_ARM_ABI_FLOAT_HARD).to_le_bytes());
    output[40..42].copy_from_slice(&(ehdr_size as u16).to_le_bytes());
    output[42..44].copy_from_slice(&32u16.to_le_bytes());
    output[44..46].copy_from_slice(&(num_phdrs as u16).to_le_bytes());
    output[46..48].copy_from_slice(&0u16.to_le_bytes());  // e_shentsize = 0 (no section headers)
    output[48..50].copy_from_slice(&0u16.to_le_bytes());  // e_shnum = 0
    output[50..52].copy_from_slice(&0u16.to_le_bytes());  // e_shstrndx = 0
}

fn write_phdr32(output: &mut [u8], offset: &mut usize, p_type: u32, p_offset: u32,
    p_vaddr: u32, p_paddr: u32, p_filesz: u32, p_memsz: u32, p_flags: u32, p_align: u32)
{
    let off = *offset;
    if off + 32 > output.len() { *offset += 32; return; }
    output[off..off+4].copy_from_slice(&p_type.to_le_bytes());
    output[off+4..off+8].copy_from_slice(&p_offset.to_le_bytes());
    output[off+8..off+12].copy_from_slice(&p_vaddr.to_le_bytes());
    output[off+12..off+16].copy_from_slice(&p_paddr.to_le_bytes());
    output[off+16..off+20].copy_from_slice(&p_filesz.to_le_bytes());
    output[off+20..off+24].copy_from_slice(&p_memsz.to_le_bytes());
    output[off+24..off+28].copy_from_slice(&p_flags.to_le_bytes());
    output[off+28..off+32].copy_from_slice(&p_align.to_le_bytes());
    *offset += 32;
}

fn build_sysv_hash(dynsym_entries: &[Elf32Sym], dynsym_names: &[String]) -> Vec<u8> {
    let nbuckets = if dynsym_names.is_empty() { 1 } else { dynsym_names.len().next_power_of_two() as u32 };
    let nchains = dynsym_entries.len() as u32;

    let mut data = Vec::new();
    data.extend_from_slice(&nbuckets.to_le_bytes());
    data.extend_from_slice(&nchains.to_le_bytes());

    let mut buckets = vec![0u32; nbuckets as usize];
    let mut chains = vec![0u32; nchains as usize];

    for (i, name) in dynsym_names.iter().enumerate() {
        let sym_idx = (i + 1) as u32;
        let hash = linker_common::sysv_hash(name.as_bytes());
        let bucket = hash % nbuckets;
        chains[sym_idx as usize] = buckets[bucket as usize];
        buckets[bucket as usize] = sym_idx;
    }

    for b in &buckets { data.extend_from_slice(&b.to_le_bytes()); }
    for c in &chains { data.extend_from_slice(&c.to_le_bytes()); }

    data
}
