//! ARM relocation application for the ARMv7 linker.
//!
//! Applies relocations from input objects to merged output sections.
//! All ARM relocation types are handled here.

use std::collections::HashMap;

use super::types::*;

/// Context for relocation application.
#[allow(dead_code)]
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
    let debug_lsm = if let Some(start) = ctx.global_symbols.get("__libc_start_main") {
        patch_addr >= start.address && patch_addr < start.address + 0x200
    } else {
        false
    };

    if rel_type == R_ARM_NONE || rel_type == R_ARM_V4BX {
        return Ok(None);
    }

    // 16-bit Thumb relocations only need 2 bytes — handle before the 4-byte check
    if rel_type == R_ARM_THM_JUMP11 || rel_type == R_ARM_THM_JUMP8 {
        let out_data = &ctx.output_sections[out_sec_idx].data;
        if patch_off + 2 > out_data.len() { return Ok(None); }
        let insn16 = read_u16_le(out_data, patch_off);
        let pc = patch_addr + 4; // Thumb PC offset
        // Extract implicit addend from the instruction (REL: addend is in the insn)
        let implicit_addend = if rel_type == R_ARM_THM_JUMP11 {
            // B (T2): imm11 is a signed halfword offset → sign-extend and shift left 1
            let imm11 = (insn16 & 0x7FF) as i32;
            ((imm11 << 21) >> 21) << 1 // sign-extend from 11 bits, then ×2
        } else {
            // B<cond> (T1): imm8 is a signed halfword offset → sign-extend and shift left 1
            let imm8 = (insn16 & 0xFF) as i32;
            ((imm8 << 24) >> 24) << 1 // sign-extend from 8 bits, then ×2
        };
        let offset = (sym_value as i64) + (implicit_addend as i64) + (addend as i64) - (pc as i64);
        let result16 = if rel_type == R_ARM_THM_JUMP11 {
            let imm11 = ((offset >> 1) as u32) & 0x7FF;
            (insn16 & 0xF800) | (imm11 as u16)
        } else {
            let imm8 = ((offset >> 1) as u32) & 0xFF;
            (insn16 & 0xFF00) | (imm8 as u16)
        };
        let out_data = &mut ctx.output_sections[out_sec_idx].data;
        write_u16_le(out_data, patch_off, result16);
        return Ok(None);
    }

    // Read implicit addend from the instruction (32-bit relocations)
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

        R_ARM_PC24 | R_ARM_CALL | R_ARM_JUMP24 | R_ARM_PLT32 => {
            // ARM BL/B: bits [23:0] are signed offset >> 2
            // Extract implicit addend: sign-extend 24-bit field, shift left 2
            let implicit_addend = ((insn_word as i32) << 8 >> 8) << 2;
            let mut target_is_thumb = false;
            let base_target = if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                if gs.needs_plt {
                    // PLT stubs are ARM mode
                    ctx.plt_vaddr + ctx.plt_header_size + (gs.plt_index as u32) * ctx.plt_entry_size
                } else {
                    target_is_thumb = gs.is_thumb;
                    sym_value
                }
            } else {
                sym_value
            };
            let target = (base_target as i64) + (implicit_addend as i64) + (addend as i64);
            // ARM ELF ABI: result = ((S + A) | T) – P
            // The implicit addend A already includes the pipeline offset cancellation
            // (-8 for ARM). The hardware adds PC (= P + 8) to the encoded offset,
            // so using P (not P + 8) as the base gives the correct result:
            //   encoded = ((S + A) - P) >> 2
            //   hardware_target = (P + 8) + (encoded << 2) = S + A + 8
            //   With A = -8: target = S  ✓
            let relocation = target - (patch_addr as i64);  // (S + A) - P

            if target_is_thumb && rel_type == R_ARM_CALL {
                // ARM BL → BLX: switch to Thumb mode
                // BLX has H bit (bit 24) = bit 1 of offset, and cond = 0b1111
                eprintln!("debug reloc: ARM_CALL BL→BLX for '{}' at 0x{:x} → target 0x{:x}",
                    sym_name, patch_addr, target);
                let h_bit = ((relocation >> 1) & 1) as u32;
                let imm24 = ((relocation >> 2) as u32) & 0x00FFFFFF;
                0xFA000000 | (h_bit << 24) | imm24
            } else {
                let rel = relocation >> 2;
                let imm24 = (rel as u32) & 0x00FFFFFF;
                (insn_word & 0xFF000000) | imm24
            }
        }

        R_ARM_THM_CALL | R_ARM_THM_JUMP24 => {
            // Thumb-2 BL/B.W: 32-bit instruction stored as two LE halfwords
            // Extract implicit addend from the instruction's encoded offset
            let implicit_addend = decode_thm_branch(insn_word);
            let mut target_is_arm = false;
            let base_target = if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                if gs.needs_plt {
                    // PLT stubs are always ARM mode
                    target_is_arm = true;
                    ctx.plt_vaddr + ctx.plt_header_size + (gs.plt_index as u32) * ctx.plt_entry_size
                } else {
                    // Target is ARM mode if it's a function-like symbol that is NOT Thumb.
                    // Check STT_FUNC and STT_GNU_IFUNC (glibc IFUNC for memcpy/memset/etc.).
                    // For unknown sym_type (STT_NOTYPE), also check if the address has
                    // bit 0 set — if it does, the target is Thumb despite lacking type info.
                    if gs.is_thumb {
                        // Definitely Thumb — keep BL
                    } else if gs.sym_type == STT_FUNC || gs.sym_type == STT_GNU_IFUNC {
                        // Function-type symbol without Thumb bit → ARM mode
                        target_is_arm = true;
                    } else if (sym_value & 1) != 0 {
                        // Address has bit 0 set → Thumb (even without STT_FUNC)
                        // This catches assembly functions with STT_NOTYPE that have
                        // Thumb bit set in their value but weren't caught by is_thumb
                    } else if gs.is_defined && gs.output_section != usize::MAX {
                        // Defined non-function symbol without Thumb bit → assume ARM
                        target_is_arm = true;
                    }
                    sym_value
                }
            } else {
                sym_value
            };
            let target = (base_target as i64) + (implicit_addend as i64) + (addend as i64);
            // For R_ARM_THM_CALL: if target is ARM mode, convert BL to BLX
            // BLX clears bit 12 in lower halfword and requires 4-byte aligned target
            let use_blx = rel_type == R_ARM_THM_CALL && target_is_arm;
            // ARM ELF ABI: result = ((S + A) | T) – P
            // The implicit addend A (decoded from the instruction) already includes
            // the pipeline offset cancellation (-4 for Thumb). The hardware adds
            // PC (= P + 4) to the encoded offset, so using P (not P + 4) as the
            // base gives the correct result:
            //   encoded = (S + A) - P
            //   hardware_target = (P + 4) + encoded = S + A + 4
            //   With A = -4: target = S  ✓
            //
            // For BLX, the hardware uses Align(PC, 4) instead of PC. BFD handles
            // this by adjusting: offset = ((S + A) - P + 2) & ~3
            let relocation = target - (patch_addr as i64);  // (S + A) - P
            let offset = if use_blx {
                (relocation + 2) & !3_i64  // BFD adjustment for Align(PC, 4)
            } else {
                relocation
            };
            if use_blx {
                eprintln!("debug reloc: THM_CALL BL→BLX for '{}' at 0x{:x} → target 0x{:x}",
                    sym_name, patch_addr, target);
            }
            encode_thm_branch_ex(insn_word, offset as i32, use_blx)
        }

        R_ARM_BASE_PREL => {
            // B(S) + A - P, where B(S) is the GOT base address
            let implicit_addend = insn_word as i32;
            (ctx.got_vaddr as i32)
                .wrapping_add(implicit_addend)
                .wrapping_add(addend)
                .wrapping_sub(patch_addr as i32) as u32
        }

        R_ARM_MOVW_ABS_NC => {
            let implicit = decode_movw_movt(insn_word);
            let val = sym_value.wrapping_add(implicit).wrapping_add(addend as u32) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_MOVT_ABS => {
            let implicit = decode_movw_movt(insn_word);
            let val = (sym_value.wrapping_add(implicit).wrapping_add(addend as u32) >> 16) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_MOVW_PREL_NC => {
            let implicit = decode_movw_movt(insn_word);
            let val = sym_value.wrapping_add(implicit).wrapping_add(addend as u32).wrapping_sub(patch_addr) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_MOVT_PREL => {
            let implicit = decode_movw_movt(insn_word);
            let val = (sym_value.wrapping_add(implicit).wrapping_add(addend as u32).wrapping_sub(patch_addr) >> 16) & 0xFFFF;
            encode_movw_movt(insn_word, val)
        }

        R_ARM_THM_MOVW_ABS_NC => {
            let implicit = decode_thm_movw_movt(insn_word);
            let val = sym_value.wrapping_add(implicit).wrapping_add(addend as u32) & 0xFFFF;
            encode_thm_movw_movt(insn_word, val)
        }

        R_ARM_THM_MOVT_ABS => {
            let implicit = decode_thm_movw_movt(insn_word);
            let val = (sym_value.wrapping_add(implicit).wrapping_add(addend as u32) >> 16) & 0xFFFF;
            encode_thm_movw_movt(insn_word, val)
        }

        R_ARM_THM_MOVW_PREL_NC => {
            let implicit = decode_thm_movw_movt(insn_word);
            let val = sym_value.wrapping_add(implicit).wrapping_add(addend as u32).wrapping_sub(patch_addr) & 0xFFFF;
            encode_thm_movw_movt(insn_word, val)
        }

        R_ARM_THM_MOVT_PREL => {
            let implicit = decode_thm_movw_movt(insn_word);
            let val = (sym_value.wrapping_add(implicit).wrapping_add(addend as u32).wrapping_sub(patch_addr) >> 16) & 0xFFFF;
            encode_thm_movw_movt(insn_word, val)
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
            // ARM uses relocation number 26 for both GOT32 and GOT_BREL.
            // In glibc's static startup, this value is typically added to the
            // GOT base to form the GOT entry address, so we emit the offset
            // from GOT base: GOT(S) + A - GOT_ORG.
            if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                let got_entry_off = (ctx.got_reserved + gs.got_index) as u32 * 4;
                let implicit_addend = insn_word as i32;
                (got_entry_off as i32)
                    .wrapping_add(implicit_addend)
                    .wrapping_add(addend) as u32
            } else {
                0
            }
        }

        R_ARM_GOT_PREL | R_ARM_TARGET2 => {
            // GOT(S) + A - P: PC-relative offset to the GOT entry for symbol S.
            // On Linux ARM, R_ARM_TARGET2 resolves to R_ARM_GOT_PREL.
            // This is used extensively by glibc's PIC code (compiled with -fpic)
            // to compute addresses of GOT entries via:
            //   ldr r3, [pc, #N]    ; load GOT_PREL result from literal pool
            //   add r3, pc           ; r3 = address of GOT entry
            let implicit_addend = insn_word as i32;
            if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                let got_entry_off = (ctx.got_reserved + gs.got_index) as u32 * 4;
                let got_entry_addr = ctx.got_vaddr + got_entry_off;
                let result = (got_entry_addr as i32)
                    .wrapping_add(implicit_addend)
                    .wrapping_add(addend)
                    .wrapping_sub(patch_addr as i32) as u32;
                // Targeted debug: show GOT_PREL relocations inside __libc_start_main
                if let Some(start) = ctx.global_symbols.get("__libc_start_main") {
                    if patch_addr >= start.address && patch_addr < start.address + 0x200 {
                        eprintln!(
                            "debug got_prel: sym='{}' patch=0x{:x} got_entry=0x{:x} result=0x{:x} implicit_addend={} addend={}",
                            sym_name, patch_addr, got_entry_addr, result, implicit_addend, addend
                        );
                    }
                }
                result
            } else {
                // Symbol not in global_symbols — might be a local symbol.
                // Compute GOT entry address from sym_value.
                eprintln!("warning: R_ARM_GOT_PREL for '{}' has no global_symbols entry at 0x{:x}",
                    sym_name, patch_addr);
                insn_word
            }
        }

        R_ARM_GOTOFF32 => {
            // S + A - GOT_ORG
            let implicit_addend = insn_word as i32;
            (sym_value as i32)
                .wrapping_add(implicit_addend)
                .wrapping_add(addend)
                .wrapping_sub(ctx.got_vaddr as i32) as u32
        }

        R_ARM_TLS_LE32 => {
            // S + A - tp, ARM TLS variant 1: tp = tls_addr - TCB_SIZE
            // So result = S + A - tls_addr + 8 (TCB_SIZE = 8 on ARM32)
            let implicit_addend = insn_word as i32;
            if ctx.has_tls {
                (sym_value as i32)
                    .wrapping_add(implicit_addend)
                    .wrapping_add(addend)
                    .wrapping_sub(ctx.tls_addr as i32)
                    .wrapping_add(8i32) as u32
            } else {
                (sym_value as i32)
                    .wrapping_add(implicit_addend)
                    .wrapping_add(addend) as u32
            }
        }

        R_ARM_TLS_GD32 => {
            // GOT(S) + A - P (data-class relocation, points to TLS descriptor in GOT)
            let implicit_addend = insn_word as i32;
            if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                let got_entry_off = (ctx.got_reserved + gs.got_index) as u32 * 4;
                (ctx.got_vaddr + got_entry_off)
                    .wrapping_add(implicit_addend as u32)
                    .wrapping_add(addend as u32)
                    .wrapping_sub(patch_addr)
            } else {
                0
            }
        }

        R_ARM_TLS_IE32 => {
            // GOT(S) + A - P (data-class relocation, no PC pipeline adjust)
            let implicit_addend = insn_word as i32;
            if let Some(gs) = ctx.global_symbols.get(&sym_name) {
                let got_entry_off = (ctx.got_reserved + gs.got_index) as u32 * 4;
                (ctx.got_vaddr + got_entry_off)
                    .wrapping_add(implicit_addend as u32)
                    .wrapping_add(addend as u32)
                    .wrapping_sub(patch_addr)
            } else {
                0
            }
        }

        _ => {
            if debug_lsm {
                eprintln!(
                    "debug lsm reloc: unsupported type={} sym='{}' patch=0x{:x}",
                    rel_type, sym_name, patch_addr
                );
            }
            eprintln!("warning: unsupported ARM relocation type {} for symbol '{}' at 0x{:x}",
                rel_type, sym_name, patch_addr);
            return Ok(None);
        }
    };

    // Write result (for 32-bit relocations)
    let out_data = &mut ctx.output_sections[out_sec_idx].data;
    write_u32_le(out_data, patch_off, result);

    if debug_lsm {
        eprintln!(
            "debug lsm reloc: type={} sym='{}' patch=0x{:x} insn=0x{:08x} addend={} result=0x{:x}",
            rel_type, sym_name, patch_addr, insn_word, addend, result
        );
    }

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

    // STB_LOCAL symbols: resolve directly from the input object's section context.
    // Do NOT look up in global_symbols to avoid name collisions with
    // same-named local symbols from other objects (e.g., static functions).
    if sym.binding == STB_LOCAL {
        if sym.section_index != SHN_UNDEF && sym.section_index != SHN_ABS
            && sym.section_index != SHN_COMMON
        {
            if let Some(&(out_sec_idx, sec_offset)) = ctx.section_map.get(&(obj_idx, sym.section_index as usize)) {
                if out_sec_idx < ctx.output_sections.len() {
                    // Strip Thumb bit for correct address computation
                    let sym_val = if sym.sym_type == STT_FUNC || sym.sym_type == STT_GNU_IFUNC { sym.value & !1 } else { sym.value };
                    return Ok((ctx.output_sections[out_sec_idx].addr + sec_offset + sym_val, sym.name.clone()));
                }
            }
        }
        if sym.section_index == SHN_ABS {
            return Ok((sym.value, sym.name.clone()));
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
/// For TLS symbols (STT_TLS), the GOT entry must contain the TP (thread pointer)
/// offset rather than the absolute address. ARM uses TLS variant 1 where:
///   - TP points to the Thread Control Block (TCB)
///   - TLS data starts at TP + sizeof(tcbhead_t) = TP + 8
///   - tp_offset = sym.address - tls_addr + 8
///
/// For TLS GD (General Dynamic) symbols, 2 consecutive GOT slots are used:
///   GOT[n]   = module_id (1 for the executable)
///   GOT[n+1] = TLS offset within the module (sym.address - tls_addr)
pub(super) fn resolve_got_reloc(
    got_data: &mut [u8],
    got_reserved: usize,
    got_dyn_symbols: &[String],
    got_local_symbols: &[String],
    global_symbols: &HashMap<String, LinkerSymbol>,
    has_tls: bool,
    tls_addr: u32,
    tls_mem_size: u32,
) {
    // Helper: fill a single symbol's GOT entry/entries using its got_index.
    // got_index already accounts for multi-slot symbols (set by build_plt_got_lists).
    let fill_got_entry = |got_data: &mut [u8], name: &str, gs: &LinkerSymbol| {
        let off = (got_reserved + gs.got_index) * 4;
        if has_tls && gs.sym_type == STT_TLS {
            if gs.needs_tls_gd {
                // TLS GD descriptor: 2 consecutive slots
                // Slot 0: module ID (1 for the executable)
                if off + 4 <= got_data.len() {
                    write_u32_le(got_data, off, 1);
                }
                // Slot 1: TLS offset within the module
                let off2 = off + 4;
                if off2 + 4 <= got_data.len() {
                    let tls_off = gs.address as i32 - tls_addr as i32;
                    write_u32_le(got_data, off2, tls_off as u32);
                }
            } else {
                // TLS IE: single slot with TP offset
                // ARM TLS variant 1: tp_offset = S - tls_addr + TCB_SIZE
                // TCB_SIZE = sizeof(tcbhead_t) = 2 * sizeof(void*) = 8 on ARM32
                if off + 4 <= got_data.len() {
                    let tpoff = gs.address as i32 - tls_addr as i32 + 8;
                    write_u32_le(got_data, off, tpoff as u32);
                }
            }
        } else {
            // Regular symbol: absolute address
            // For Thumb functions, set bit 0 so BLX via GOT switches to Thumb mode
            if off + 4 <= got_data.len() {
                let addr = if (gs.sym_type == STT_FUNC || gs.sym_type == STT_GNU_IFUNC) && gs.is_thumb {
                    gs.address | 1
                } else {
                    gs.address
                };
                write_u32_le(got_data, off, addr);
                if name == "__rel_iplt_start" || name == "__rel_iplt_end" || gs.sym_type == STT_GNU_IFUNC {
                    eprintln!(
                        "debug got: '{}' off=0x{:x} value=0x{:x} is_thumb={} sym_type={}",
                        name, off, addr, gs.is_thumb, gs.sym_type
                    );
                }
                if gs.is_defined && gs.address == 0 {
                    eprintln!("warning: GOT entry for '{}' resolved to 0", name);
                }
            }
        }
    };

    for name in got_dyn_symbols.iter() {
        if let Some(gs) = global_symbols.get(name) {
            fill_got_entry(got_data, name, gs);
        } else {
            eprintln!("warning: GOT symbol '{}' not found in global_symbols", name);
        }
    }
    for name in got_local_symbols.iter() {
        if let Some(gs) = global_symbols.get(name) {
            fill_got_entry(got_data, name, gs);
            // Warn about zero-address GOT entries for defined non-weak symbols
            if gs.address == 0 && gs.is_defined && gs.binding != STB_WEAK
                && gs.sym_type != STT_TLS
            {
                eprintln!("warning: GOT entry for defined symbol '{}' has address 0 \
                    (output_section={}, section_offset=0x{:x})",
                    name, gs.output_section, gs.section_offset);
            }
        } else {
            eprintln!("warning: GOT symbol '{}' not found in global_symbols", name);
        }
    }
}

/// Decode the 16-bit immediate from an ARM MOVW/MOVT instruction.
/// Format: imm4 at bits [19:16], imm12 at bits [11:0].
fn decode_movw_movt(insn: u32) -> u32 {
    let imm12 = insn & 0xFFF;
    let imm4 = (insn >> 16) & 0xF;
    (imm4 << 12) | imm12
}

fn encode_movw_movt(insn: u32, val: u32) -> u32 {
    let imm12 = val & 0xFFF;
    let imm4 = (val >> 12) & 0xF;
    (insn & 0xFFF0F000) | (imm4 << 16) | imm12
}

/// Decode the implicit addend from a Thumb-2 BL/B.W branch instruction.
/// Returns the signed byte offset encoded in the instruction.
fn decode_thm_branch(insn_word: u32) -> i32 {
    let upper = insn_word & 0xFFFF;
    let lower = (insn_word >> 16) & 0xFFFF;
    let s = (upper >> 10) & 1;
    let imm10 = upper & 0x3FF;
    let j1 = (lower >> 13) & 1;
    let j2 = (lower >> 11) & 1;
    let imm11 = lower & 0x7FF;
    // I1 = NOT(J1 XOR S), I2 = NOT(J2 XOR S)
    let i1 = ((j1 ^ s) ^ 1) & 1;
    let i2 = ((j2 ^ s) ^ 1) & 1;
    // Reconstruct: S:I1:I2:imm10:imm11:0 (25-bit signed)
    let raw = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1);
    // Sign-extend from 25 bits
    ((raw as i32) << 7) >> 7
}

/// Encode a Thumb-2 BL/B.W/BLX branch instruction with the given byte offset.
///
/// Thumb-2 32-bit branch format (stored as two LE halfwords):
///   upper (insn_word[15:0]):  11110 S imm10
///   lower (insn_word[31:16]): 1x J1 1 J2 imm11
///
/// The offset encodes as: S:I1:I2:imm10:imm11:0 (25-bit signed, in bytes)
/// where I1 = NOT(J1 XOR S), I2 = NOT(J2 XOR S)
///
/// When `use_blx` is true, converts BL to BLX for Thumb-to-ARM interworking:
/// - Clears bit 12 of lower halfword (BL has bit12=1, BLX has bit12=0)
/// - BLX target must be 4-byte aligned; bit 1 of the offset encodes in H (bit 0 of imm11)
fn encode_thm_branch_ex(insn_word: u32, offset: i32, use_blx: bool) -> u32 {
    let upper = insn_word & 0xFFFF;
    let lower = (insn_word >> 16) & 0xFFFF;

    let s = if offset < 0 { 1u32 } else { 0u32 };
    let uoffset = (offset as u32) >> 1; // Remove trailing zero bit (halfword aligned)
    let imm11 = uoffset & 0x7FF;
    let imm10 = (uoffset >> 11) & 0x3FF;
    let i2 = (uoffset >> 21) & 1;
    let i1 = (uoffset >> 22) & 1;
    // J1 = NOT(I1 XOR S), J2 = NOT(I2 XOR S)
    let j1 = ((i1 ^ s) ^ 1) & 1;
    let j2 = ((i2 ^ s) ^ 1) & 1;

    // Preserve opcode bits: upper[15:11]=11110
    let new_upper = (upper & 0xF800) | (s << 10) | imm10;

    if use_blx {
        // BLX: lower halfword has bit14=1, bit12=0 (vs BL which has bit14=1, bit12=1)
        // For BLX, imm11's bit 0 is the H bit (bit 1 of the byte offset)
        let new_lower = 0xC000 | (j1 << 13) | (j2 << 11) | imm11;
        new_upper | (new_lower << 16)
    } else {
        // BL or B.W: preserve original lower opcode bits (bit15, bit14, bit12)
        let new_lower = (lower & 0xD000) | (j1 << 13) | (j2 << 11) | imm11;
        new_upper | (new_lower << 16)
    }
}

/// Decode the 16-bit immediate from a Thumb-2 MOVW/MOVT instruction.
/// Format (two LE halfwords): upper has i at bit[10], imm4 at bits[3:0];
/// lower has imm3 at bits[14:12], imm8 at bits[7:0].
/// The 16-bit immediate = imm4:i:imm3:imm8.
fn decode_thm_movw_movt(insn_word: u32) -> u32 {
    let upper = insn_word & 0xFFFF;
    let lower = (insn_word >> 16) & 0xFFFF;
    let imm8 = lower & 0xFF;
    let imm3 = (lower >> 12) & 0x7;
    let i = (upper >> 10) & 1;
    let imm4 = upper & 0xF;
    (imm4 << 12) | (i << 11) | (imm3 << 8) | imm8
}

/// Encode a Thumb-2 MOVW/MOVT instruction with a 16-bit immediate value.
///
/// Thumb-2 MOVW/MOVT format (stored as two LE halfwords):
///   upper (insn_word[15:0]):  11110 i 10 xxxx imm4
///   lower (insn_word[31:16]): 0 imm3 Rd(4) imm8
///
/// The 16-bit immediate = imm4:i:imm3:imm8
fn encode_thm_movw_movt(insn_word: u32, val: u32) -> u32 {
    let upper = insn_word & 0xFFFF;
    let lower = (insn_word >> 16) & 0xFFFF;

    let imm8 = val & 0xFF;
    let imm3 = (val >> 8) & 0x7;
    let i = (val >> 11) & 1;
    let imm4 = (val >> 12) & 0xF;

    // upper: preserve everything except bit[10] (i) and bits[3:0] (imm4)
    let new_upper = (upper & 0xFBF0) | (i << 10) | imm4;
    // lower: preserve bit[15] and Rd (bits[11:8]), modify imm3 (bits[14:12]) and imm8 (bits[7:0])
    let new_lower = (lower & 0x8F00) | (imm3 << 12) | imm8;

    new_upper | (new_lower << 16)
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    if offset + 2 > data.len() { return 0; }
    u16::from_le_bytes([data[offset], data[offset+1]])
}

fn write_u16_le(data: &mut [u8], offset: usize, val: u16) {
    if offset + 2 > data.len() { return; }
    let bytes = val.to_le_bytes();
    data[offset..offset+2].copy_from_slice(&bytes);
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
