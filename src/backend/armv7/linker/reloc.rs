//! ARM relocation application for the ARMv7 linker.
//!
//! Applies relocations from input objects to merged output sections.
//! All ARM relocation types are handled here.

use std::collections::HashMap;

use super::types::*;

/// Context for relocation application.
pub(super) struct RelocContext<'a> {
    pub global_symbols: &'a HashMap<String, LinkerSymbol>,
    pub output_sections: &'a mut Vec<OutputSection>,
    pub section_map: &'a SectionMap,
    pub got_base: u32,
    pub got_vaddr: u32,
    pub gotplt_vaddr: u32,
    pub got_reserved: usize,
    pub gotplt_reserved: u32,
    #[allow(dead_code)]
    pub plt_vaddr: u32,
    #[allow(dead_code)]
    pub plt_header_size: u32,
    #[allow(dead_code)]
    pub plt_entry_size: u32,
    pub num_plt: usize,
    pub tls_addr: u32,
    pub tls_mem_size: u32,
    pub has_tls: bool,
}

/// Apply all relocations from input objects to the output sections.
pub(super) fn apply_relocations(
    inputs: &[InputObject],
    ctx: &mut RelocContext,
) -> Result<Vec<(u32, String)>, String> {
    let mut text_relocs: Vec<(u32, String)> = Vec::new();
    for (obj_idx, obj) in inputs.iter().enumerate() {
        for sec in &obj.sections {
            if sec.relocations.is_empty() { continue; }

            let _out_name = match output_section_name(&sec.name, sec.flags, sec.sh_type) {
                Some(n) => n,
                None => continue,
            };
            let (out_sec_idx, sec_base_offset) = match ctx.section_map.get(&(obj_idx, sec.input_index)) {
                Some(&v) => v,
                None => continue,
            };

            for &(rel_offset, rel_type, sym_idx, addend) in &sec.relocations {
                let tr = apply_one_reloc(
                    obj_idx, obj, sec, out_sec_idx, sec_base_offset,
                    rel_offset, rel_type, sym_idx, addend,
                    ctx,
                )?;
                if let Some(t) = tr {
                    text_relocs.push(t);
                }
            }
        }
    }
    Ok(text_relocs)
}

fn apply_one_reloc(
    obj_idx: usize,
    obj: &InputObject,
    _sec: &InputSection,
    out_sec_idx: usize,
    sec_base_offset: u32,
    rel_offset: u32,
    rel_type: u32,
    sym_idx: u32,
    addend: i32,
    ctx: &mut RelocContext,
) -> Result<Option<(u32, String)>, String> {
    let sym = if (sym_idx as usize) < obj.symbols.len() {
        &obj.symbols[sym_idx as usize]
    } else {
        return Ok(None);
    };

    // Resolve symbol value
    let (sym_value, sym_name) = resolve_sym_value(obj_idx, sym, ctx)?;

    let out_sec = &ctx.output_sections[out_sec_idx];
    let patch_addr = out_sec.addr + sec_base_offset + rel_offset;
    let patch_off = (sec_base_offset + rel_offset) as usize;

    if rel_type == R_ARM_NONE || rel_type == R_ARM_V4BX {
        return Ok(None);
    }

    // Read implicit addend from the instruction
    let out_data = &ctx.output_sections[out_sec_idx].data;
    if patch_off + 4 > out_data.len() {
        return Ok(None);
    }
    let insn_word = read_u32_le(out_data, patch_off);

    let result: u32 = match rel_type {
        R_ARM_ABS32 | R_ARM_TARGET1 => {
            let implicit_addend = insn_word as i32;
            (sym_value as i32).wrapping_add(implicit_addend).wrapping_add(addend) as u32
        }

        R_ARM_REL32 => {
            let implicit_addend = insn_word as i32;
            (sym_value as i32)
                .wrapping_add(implicit_addend)
                .wrapping_add(addend)
                .wrapping_sub(patch_addr as i32) as u32
        }

        R_ARM_CALL | R_ARM_JUMP24 | R_ARM_PLT32 => {
            // BL/B: bits [23:0] are signed offset >> 2, PC = patch_addr + 8
            let target = if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                if gs.needs_plt {
                    ctx.plt_vaddr + ctx.plt_header_size + (gs.plt_index as u32) * ctx.plt_entry_size
                } else {
                    sym_value
                }
            } else {
                sym_value
            };
            let pc = patch_addr + 8; // ARM PC offset
            let rel = (target as i64 - pc as i64) >> 2;
            let imm24 = (rel as u32) & 0x00FFFFFF;
            (insn_word & 0xFF000000) | imm24
        }

        R_ARM_MOVW_ABS_NC => {
            let val = sym_value.wrapping_add(addend as u32) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_MOVT_ABS => {
            let val = (sym_value.wrapping_add(addend as u32) >> 16) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_MOVW_PREL_NC => {
            let val = sym_value.wrapping_add(addend as u32).wrapping_sub(patch_addr) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_MOVT_PREL => {
            let val = (sym_value.wrapping_add(addend as u32).wrapping_sub(patch_addr) >> 16) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_PREL31 => {
            let implicit_addend = ((insn_word as i32) << 1) >> 1; // sign-extend 31 bits
            let val = (sym_value as i32)
                .wrapping_add(implicit_addend)
                .wrapping_add(addend)
                .wrapping_sub(patch_addr as i32);
            (insn_word & 0x80000000) | ((val as u32) & 0x7FFFFFFF)
        }

        R_ARM_GOT_BREL | R_ARM_GOT32 => {
            if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                let got_entry_off = (ctx.got_reserved + gs.got_index) as u32 * 4;
                got_entry_off
            } else {
                0
            }
        }

        R_ARM_GOTOFF32 => {
            sym_value.wrapping_sub(ctx.got_vaddr)
        }

        R_ARM_TLS_LE32 => {
            if ctx.has_tls {
                sym_value.wrapping_sub(ctx.tls_addr).wrapping_add(ctx.tls_mem_size)
            } else {
                sym_value
            }
        }

        R_ARM_TLS_IE32 => {
            // GOT-relative offset to the TLS IE entry
            if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                let got_entry_off = (ctx.got_reserved + gs.got_index) as u32 * 4;
                let pc = patch_addr + 8;
                (ctx.got_vaddr + got_entry_off).wrapping_sub(pc)
            } else {
                0
            }
        }

        _ => {
            eprintln!("warning: unsupported ARM relocation type {} for symbol '{}' at 0x{:x}",
                rel_type, sym_name, patch_addr);
            return Ok(None);
        }
    };

    // Write result
    let out_data = &mut ctx.output_sections[out_sec_idx].data;
    write_u32_le(out_data, patch_off, result);

    // Check for text relocations
    if let Some(gs) = ctx.global_symbols.get(&sym_name) {
        if gs.uses_textrel {
            return Ok(Some((patch_addr, sym_name)));
        }
    }

    Ok(None)
}

fn resolve_sym_value(
    obj_idx: usize,
    sym: &InputSymbol,
    ctx: &RelocContext,
) -> Result<(u32, String), String> {
    if sym.name.is_empty() || sym.sym_type == STT_SECTION {
        // Section symbol: resolve to section address
        if sym.section_index != SHN_UNDEF {
            if let Some(&(out_sec_idx, sec_offset)) = ctx.section_map.get(&(obj_idx, sym.section_index as usize)) {
                if out_sec_idx < ctx.output_sections.len() {
                    return Ok((ctx.output_sections[out_sec_idx].addr + sec_offset + sym.value, sym.name.clone()));
                }
            }
        }
        return Ok((0, sym.name.clone()));
    }

    if let Some(gs) = ctx.global_symbols.get(&sym.name) {
        if gs.needs_copy && gs.copy_addr != 0 {
            return Ok((gs.copy_addr, sym.name.clone()));
        }
        return Ok((gs.address, sym.name.clone()));
    }

    // Weak undefined
    if sym.binding == STB_WEAK {
        return Ok((0, sym.name.clone()));
    }

    Ok((0, sym.name.clone()))
}

/// Resolve GOT-related relocations: fill GOT entries with resolved addresses.
pub(super) fn resolve_got_reloc(
    got_data: &mut [u8],
    got_reserved: usize,
    got_dyn_symbols: &[String],
    got_local_symbols: &[String],
    global_symbols: &HashMap<String, LinkerSymbol>,
) {
    for (i, name) in got_dyn_symbols.iter().enumerate() {
        let off = (got_reserved + i) * 4;
        if off + 4 > got_data.len() { continue; }
        let val = global_symbols.get(name).map(|s| s.address).unwrap_or(0);
        write_u32_le(got_data, off, val);
    }
    let base = got_reserved + got_dyn_symbols.len();
    for (i, name) in got_local_symbols.iter().enumerate() {
        let off = (base + i) * 4;
        if off + 4 > got_data.len() { continue; }
        let val = global_symbols.get(name).map(|s| s.address).unwrap_or(0);
        write_u32_le(got_data, off, val);
    }
}

fn encode_movw_movt(insn: u32, val: u32) -> u32 {
    let imm12 = val & 0xFFF;
    let imm4 = (val >> 12) & 0xF;
    (insn & 0xFFF0F000) | (imm4 << 16) | imm12
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    if offset + 4 > data.len() { return 0; }
    u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]])
}

fn write_u32_le(data: &mut [u8], offset: usize, val: u32) {
    if offset + 4 > data.len() { return; }
    let bytes = val.to_le_bytes();
    data[offset..offset+4].copy_from_slice(&bytes);
}
