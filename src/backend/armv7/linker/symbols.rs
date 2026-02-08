//! Symbol resolution for the ARMv7 linker.
//!
//! Phases 6-9: global symbol resolution, COMMON symbol allocation,
//! PLT/GOT marking, undefined symbol checking, PLT/GOT list building,
//! and IFUNC collection. Closely follows the i686 linker's symbol logic.

use std::collections::HashMap;

use super::types::*;
use crate::backend::linker_common;

pub(super) fn resolve_symbols(
    inputs: &[InputObject],
    _output_sections: &[OutputSection],
    section_map: &SectionMap,
    dynlib_syms: &HashMap<String, (String, u8, u32, Option<String>, bool, u8)>,
) -> (HashMap<String, LinkerSymbol>, HashMap<(usize, usize), String>) {
    let mut global_symbols: HashMap<String, LinkerSymbol> = HashMap::new();
    let mut sym_resolution: HashMap<(usize, usize), String> = HashMap::new();

    // First pass: collect definitions
    for (obj_idx, obj) in inputs.iter().enumerate() {
        for (sym_idx, sym) in obj.symbols.iter().enumerate() {
            if sym.name.is_empty() || sym.sym_type == STT_FILE || sym.sym_type == STT_SECTION {
                continue;
            }
            if sym.section_index == SHN_UNDEF { continue; }

            let (out_sec_idx, sec_offset) = if sym.section_index != SHN_ABS && sym.section_index != SHN_COMMON {
                section_map.get(&(obj_idx, sym.section_index as usize))
                    .copied().unwrap_or((usize::MAX, 0))
            } else {
                (usize::MAX, 0)
            };

            // ARM ELF: Thumb function symbols have bit 0 set in st_value.
            // Strip it for correct address computation; interworking is handled
            // separately in relocation processing.
            let sym_val = if sym.sym_type == STT_FUNC { sym.value & !1 } else { sym.value };

            let new_sym = LinkerSymbol {
                address: 0,
                size: sym.size,
                sym_type: sym.sym_type,
                binding: sym.binding,
                visibility: sym.visibility,
                is_defined: true,
                needs_plt: false,
                needs_got: false,
                needs_tls_gd: false,
                is_thumb: sym.sym_type == STT_FUNC && (sym.value & 1) != 0,
                is_abs: sym.section_index == SHN_ABS,
                output_section: out_sec_idx,
                section_offset: sec_offset + sym_val,
                plt_index: 0,
                got_index: 0,
                is_dynamic: false,
                dynlib: String::new(),
                needs_copy: false,
                copy_addr: 0,
                version: None,
                uses_textrel: false,
            };

            match global_symbols.get(&sym.name) {
                None => {
                    // STB_LOCAL symbols are inserted as fallback definitions when
                    // no entry exists yet. glibc's static archives contain cross-
                    // object references that resolve through local symbols.
                    // Relocations to local symbols are resolved directly via
                    // section_map in resolve_sym_value, so this fallback is only
                    // used for cross-object name lookups.
                    global_symbols.insert(sym.name.clone(), new_sym);
                }
                Some(existing) => {
                    // Local symbols must not override any existing entry.
                    // They have file scope only and should not shadow globals
                    // or weaks from other objects.
                    if sym.binding == STB_LOCAL {
                        // Already have a definition; keep it.
                    } else if sym.binding == STB_GLOBAL && (existing.binding == STB_WEAK || existing.binding == STB_LOCAL)
                        || (!existing.is_defined && new_sym.is_defined)
                        // Prefer a definition in a valid output section over one
                        // whose section was discarded (COMMON or orphaned).
                        // This handles the case where a COMMON/tentative definition
                        // was inserted first but a real definition (in .data/.bss)
                        // from a later object should take precedence.
                        || (existing.is_defined && new_sym.is_defined
                            && existing.output_section == usize::MAX
                            && new_sym.output_section != usize::MAX)
                    {
                        global_symbols.insert(sym.name.clone(), new_sym);
                    }
                }
            }

            sym_resolution.insert((obj_idx, sym_idx), sym.name.clone());
        }
    }

    // Second pass: undefined symbols (look in dynamic libs)
    for (obj_idx, obj) in inputs.iter().enumerate() {
        for (sym_idx, sym) in obj.symbols.iter().enumerate() {
            if sym.name.is_empty() || sym.sym_type == STT_FILE || sym.sym_type == STT_SECTION {
                continue;
            }
            if sym.section_index != SHN_UNDEF { continue; }
            if global_symbols.get(&sym.name).map(|s| s.is_defined).unwrap_or(false) {
                sym_resolution.insert((obj_idx, sym_idx), sym.name.clone());
                continue;
            }
            if let Some((lib, stype, size, ver, _is_default, bind)) = dynlib_syms.get(&sym.name) {
                let new_sym = LinkerSymbol {
                    address: 0,
                    size: *size,
                    sym_type: *stype,
                    binding: *bind,
                    visibility: STV_DEFAULT,
                    is_defined: false,
                    needs_plt: false,
                    needs_got: false,
                    needs_tls_gd: false,
                    is_thumb: false,
                    is_abs: false,
                    output_section: usize::MAX,
                    section_offset: 0,
                    plt_index: 0,
                    got_index: 0,
                    is_dynamic: true,
                    dynlib: lib.clone(),
                    needs_copy: *stype == STT_OBJECT && *size > 0,
                    copy_addr: 0,
                    version: ver.clone(),
                    uses_textrel: false,
                };
                global_symbols.insert(sym.name.clone(), new_sym);
            } else if !global_symbols.contains_key(&sym.name) {
                global_symbols.insert(sym.name.clone(), LinkerSymbol {
                    address: 0, size: 0, sym_type: sym.sym_type,
                    binding: sym.binding, visibility: sym.visibility,
                    is_defined: false, needs_plt: false, needs_got: false,
                    needs_tls_gd: false, is_thumb: false, is_abs: false,
                    output_section: usize::MAX, section_offset: 0,
                    plt_index: 0, got_index: 0, is_dynamic: false,
                    dynlib: String::new(), needs_copy: false, copy_addr: 0,
                    version: None, uses_textrel: false,
                });
            }
            sym_resolution.insert((obj_idx, sym_idx), sym.name.clone());
        }
    }

    (global_symbols, sym_resolution)
}

/// Allocate COMMON symbols and orphaned defined symbols into .bss.
///
/// COMMON symbols (SHN_COMMON) are tentative definitions that need space
/// allocated in .bss. Additionally, some symbols may be defined in input
/// sections that were discarded (e.g., COMDAT deduplication, filtered
/// sections like .ARM.exidx). These "orphaned" symbols still need valid
/// addresses if they're referenced (e.g., via GOT), so we allocate them
/// to .bss as a fallback.
pub(super) fn allocate_common_symbols(
    inputs: &[InputObject],
    output_sections: &mut Vec<OutputSection>,
    section_name_to_idx: &mut HashMap<String, usize>,
    global_symbols: &mut HashMap<String, LinkerSymbol>,
) {
    let mut commons: Vec<(String, u32, u32)> = Vec::new(); // name, size, align
    for obj in inputs {
        for sym in &obj.symbols {
            if sym.section_index == SHN_COMMON && !sym.name.is_empty()
                && sym.binding != STB_LOCAL
            {
                // COMMON symbols are inserted with is_defined=true but output_section=usize::MAX
                // (since SHN_COMMON != SHN_UNDEF). They need allocation if they haven't been
                // overridden by a real definition from another object.
                if let Some(gs) = global_symbols.get(&sym.name) {
                    if gs.output_section == usize::MAX && gs.is_defined && !gs.is_dynamic {
                        // Avoid duplicate allocation if same symbol appears in multiple objects
                        if !commons.iter().any(|(n, _, _)| n == &sym.name) {
                            let align = sym.value.max(1);
                            commons.push((sym.name.clone(), sym.size, align));
                        }
                    }
                }
            }
        }
    }

    // Also collect orphaned defined symbols whose input sections were discarded.
    // These symbols have output_section == usize::MAX (section not in section_map)
    // but are still referenced via GOT or relocations. Allocate them to .bss
    // so they get valid (zero-initialized) addresses instead of NULL.
    // Skip SHN_ABS symbols (is_abs=true) — they have absolute addresses and
    // don't need section allocation.
    let mut orphaned: Vec<(String, u32, u32)> = Vec::new();
    for (name, gs) in global_symbols.iter() {
        if gs.output_section == usize::MAX && gs.is_defined && !gs.is_dynamic
            && !gs.is_abs
            && !commons.iter().any(|(cn, _, _)| cn == name)
        {
            // Use size from the symbol, minimum 4 bytes, alignment 4
            let size = gs.size.max(4);
            orphaned.push((name.clone(), size, 4));
        }
    }
    orphaned.sort_by(|a, b| a.0.cmp(&b.0));
    commons.extend(orphaned);

    if commons.is_empty() { return; }

    let bss_idx = if let Some(&idx) = section_name_to_idx.get(".bss") {
        idx
    } else {
        let idx = output_sections.len();
        section_name_to_idx.insert(".bss".into(), idx);
        output_sections.push(OutputSection {
            name: ".bss".into(), sh_type: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE, data: Vec::new(),
            align: 4, addr: 0, file_offset: 0,
        });
        idx
    };

    for (name, size, align) in commons {
        let bss = &mut output_sections[bss_idx];
        let cur_size = bss.data.len() as u32;
        let aligned = align_up(cur_size, align);
        bss.data.resize(aligned as usize + size as usize, 0);
        bss.align = bss.align.max(align);

        if let Some(sym) = global_symbols.get_mut(&name) {
            sym.is_defined = true;
            sym.output_section = bss_idx;
            sym.section_offset = aligned;
        }
    }
}

/// Mark PLT/GOT needs for symbols.
pub(super) fn mark_plt_got_needs(
    inputs: &[InputObject],
    global_symbols: &mut HashMap<String, LinkerSymbol>,
    is_static: bool,
) {
    for obj in inputs {
        for sec in &obj.sections {
            for &(_, rel_type, sym_idx, _) in &sec.relocations {
                let sym_idx = sym_idx as usize;
                if sym_idx >= obj.symbols.len() { continue; }
                let sym = &obj.symbols[sym_idx];
                if sym.name.is_empty() { continue; }

                let gs = match global_symbols.get_mut(&sym.name) {
                    Some(s) => s,
                    None => continue,
                };

                match rel_type {
                    R_ARM_PC24 | R_ARM_CALL | R_ARM_JUMP24 | R_ARM_PLT32
                    | R_ARM_THM_CALL | R_ARM_THM_JUMP24 => {
                        if gs.is_dynamic && !is_static {
                            gs.needs_plt = true;
                        }
                    }
                    R_ARM_GOT32 | R_ARM_GOT_BREL | R_ARM_TLS_IE32 => {
                        gs.needs_got = true;
                    }
                    R_ARM_TLS_GD32 => {
                        gs.needs_got = true;
                        gs.needs_tls_gd = true;
                    }
                    _ => {}
                }
            }
        }
    }

    // IFUNC symbols always need PLT+GOT
    if !is_static {
        let ifunc_names: Vec<String> = global_symbols.iter()
            .filter(|(_, s)| s.sym_type == STT_GNU_IFUNC)
            .map(|(n, _)| n.clone())
            .collect();
        for name in ifunc_names {
            if let Some(sym) = global_symbols.get_mut(&name) {
                sym.needs_plt = true;
                sym.needs_got = true;
            }
        }
    }
}

/// Check for undefined non-weak symbols.
pub(super) fn check_undefined_symbols(
    global_symbols: &HashMap<String, LinkerSymbol>,
) -> Result<(), String> {
    for (name, sym) in global_symbols {
        if !sym.is_defined && !sym.is_dynamic && sym.binding != STB_WEAK {
            if linker_common::is_linker_defined_symbol(name) { continue; }
            return Err(format!("undefined reference to `{}'", name));
        }
    }
    Ok(())
}

/// Build PLT and GOT symbol lists.
pub(super) fn build_plt_got_lists(
    global_symbols: &mut HashMap<String, LinkerSymbol>,
) -> (Vec<String>, Vec<String>, Vec<String>, usize, usize) {
    let mut plt_symbols: Vec<String> = Vec::new();
    let mut got_dyn_symbols: Vec<String> = Vec::new();
    let mut got_local_symbols: Vec<String> = Vec::new();

    // PLT symbols
    let mut plt_names: Vec<String> = global_symbols.iter()
        .filter(|(_, s)| s.needs_plt)
        .map(|(n, _)| n.clone())
        .collect();
    plt_names.sort();

    for (i, name) in plt_names.iter().enumerate() {
        if let Some(sym) = global_symbols.get_mut(name) {
            sym.plt_index = i;
        }
        plt_symbols.push(name.clone());
    }

    // GOT symbols (dynamic)
    // Use a running slot offset to account for TLS GD symbols that need 2 slots.
    let mut got_dyn_names: Vec<String> = global_symbols.iter()
        .filter(|(_, s)| s.needs_got && s.is_dynamic)
        .map(|(n, _)| n.clone())
        .collect();
    got_dyn_names.sort();

    let mut dyn_slot_offset: usize = 0;
    for name in got_dyn_names.iter() {
        if let Some(sym) = global_symbols.get_mut(name) {
            sym.got_index = dyn_slot_offset;
            dyn_slot_offset += if sym.needs_tls_gd { 2 } else { 1 };
        }
        got_dyn_symbols.push(name.clone());
    }

    // GOT symbols (local)
    let mut got_local_names: Vec<String> = global_symbols.iter()
        .filter(|(_, s)| s.needs_got && !s.is_dynamic)
        .map(|(n, _)| n.clone())
        .collect();
    got_local_names.sort();

    let mut local_slot_offset: usize = dyn_slot_offset;
    for name in got_local_names.iter() {
        if let Some(sym) = global_symbols.get_mut(name) {
            sym.got_index = local_slot_offset;
            local_slot_offset += if sym.needs_tls_gd { 2 } else { 1 };
        }
        got_local_symbols.push(name.clone());
    }

    let num_plt = plt_symbols.len();
    // Total GOT slots (not symbol count — GD symbols use 2 slots each)
    let num_got_slots = local_slot_offset;

    (plt_symbols, got_dyn_symbols, got_local_symbols, num_plt, num_got_slots)
}

/// Collect IFUNC symbols for static linking.
pub(super) fn collect_ifunc_symbols(
    global_symbols: &HashMap<String, LinkerSymbol>,
    is_static: bool,
) -> Vec<String> {
    if !is_static { return Vec::new(); }
    global_symbols.iter()
        .filter(|(_, s)| s.sym_type == STT_GNU_IFUNC && s.is_defined)
        .map(|(n, _)| n.clone())
        .collect()
}
