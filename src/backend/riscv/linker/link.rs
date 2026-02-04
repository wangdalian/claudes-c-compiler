//! RISC-V 64-bit ELF linker: symbol resolution, relocation, and ELF executable emission.

use std::collections::{HashMap, HashSet};
use super::elf_read::*;

// ── ELF executable constants ────────────────────────────────────────────────

const ET_EXEC: u16 = 2;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;
const PT_NOTE: u32 = 4;
const PT_TLS: u32 = 7;
const PT_GNU_EH_FRAME: u32 = 0x6474e550;
const PT_GNU_STACK: u32 = 0x6474e551;
const PT_GNU_RELRO: u32 = 0x6474e552;
const PT_RISCV_ATTRIBUTES: u32 = 0x70000003;

const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_PLTRELSZ: i64 = 2;
const DT_PLTGOT: i64 = 3;
const DT_HASH: i64 = 4;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_DEBUG: i64 = 21;
const DT_JMPREL: i64 = 23;
const DT_PLTREL: i64 = 20;
const DT_FLAGS: i64 = 30;
const DT_GNU_HASH: i64 = 0x6ffffef5;
const DT_VERSYM: i64 = 0x6ffffff0;
const DT_VERNEED: i64 = 0x6ffffffe;
const DT_VERNEEDNUM: i64 = 0x6fffffff;
const DT_INIT_ARRAY: i64 = 25;
const DT_INIT_ARRAYSZ: i64 = 27;
const DT_FINI_ARRAY: i64 = 26;
const DT_FINI_ARRAYSZ: i64 = 28;
const DT_PREINIT_ARRAY: i64 = 32;
const DT_PREINIT_ARRAYSZ: i64 = 33;

// RISC-V relocation types
const R_RISCV_32: u32 = 1;
const R_RISCV_64: u32 = 2;
const R_RISCV_BRANCH: u32 = 16;
const R_RISCV_JAL: u32 = 17;
const R_RISCV_CALL_PLT: u32 = 19;
const R_RISCV_GOT_HI20: u32 = 20;
const R_RISCV_TLS_GOT_HI20: u32 = 21;
const R_RISCV_TLS_GD_HI20: u32 = 22;
const R_RISCV_PCREL_HI20: u32 = 23;
const R_RISCV_PCREL_LO12_I: u32 = 24;
const R_RISCV_PCREL_LO12_S: u32 = 25;
const R_RISCV_HI20: u32 = 26;
const R_RISCV_LO12_I: u32 = 27;
const R_RISCV_LO12_S: u32 = 28;
const R_RISCV_TPREL_HI20: u32 = 29;
const R_RISCV_TPREL_LO12_I: u32 = 30;
const R_RISCV_TPREL_LO12_S: u32 = 31;
const R_RISCV_TPREL_ADD: u32 = 32;
const R_RISCV_ADD8: u32 = 33;
const R_RISCV_ADD16: u32 = 34;
const R_RISCV_ADD32: u32 = 35;
const R_RISCV_ADD64: u32 = 36;
const R_RISCV_SUB8: u32 = 37;
const R_RISCV_SUB16: u32 = 38;
const R_RISCV_SUB32: u32 = 39;
const R_RISCV_SUB64: u32 = 40;
const R_RISCV_RELAX: u32 = 51;
const R_RISCV_SET6: u32 = 53;
const R_RISCV_SUB6: u32 = 52;
const R_RISCV_SET8: u32 = 54;
const R_RISCV_SET16: u32 = 55;
const R_RISCV_SET32: u32 = 56;
const R_RISCV_32_PCREL: u32 = 57;

const PAGE_SIZE: u64 = 0x1000;
const BASE_ADDR: u64 = 0x10000;

const INTERP: &[u8] = b"/lib/ld-linux-riscv64-lp64d.so.1\0";

/// A merged input section with its assigned virtual address.
struct MergedSection {
    name: String,
    sh_type: u32,
    sh_flags: u64,
    data: Vec<u8>,
    vaddr: u64,
    align: u64,
}

/// Represents a global symbol's definition.
#[derive(Clone, Debug)]
struct GlobalSym {
    /// Virtual address of the symbol.
    value: u64,
    size: u64,
    binding: u8,
    sym_type: u8,
    visibility: u8,
    /// True if this symbol is defined in an input object file.
    defined: bool,
    /// True if this symbol needs a PLT/GOT entry (from a shared library).
    needs_plt: bool,
    /// Index into the PLT (if needs_plt).
    plt_idx: usize,
    /// GOT offset (relative to GOT base) if this symbol has a GOT entry.
    got_offset: Option<u64>,
    /// Section index in the merged section list, if defined locally.
    section_idx: Option<usize>,
}

/// A pending relocation from an input object file, remapped to merged sections.
struct PendingReloc {
    /// Index of the merged section this relocation applies to.
    target_section: usize,
    /// Byte offset within the merged section.
    offset: u64,
    reloc_type: u32,
    /// Name of the symbol referenced.
    symbol_name: String,
    addend: i64,
}

/// Dynamic symbol entry for the output .dynsym.
struct DynSym {
    name: String,
    name_offset: u32,
    value: u64,
    size: u64,
    info: u8,
    other: u8,
    shndx: u16,
}

/// Link object files into a RISC-V ELF executable.
///
/// `object_files`: paths to .o files, .a archives, and linker flags (-l, -L, etc.)
/// `output_path`: path for the output executable
/// `user_args`: additional linker flags from the user
pub fn link_to_executable(
    object_files: &[&str],
    output_path: &str,
    user_args: &[String],
) -> Result<(), String> {
    let is_static = user_args.iter().any(|a| a == "-static");
    let is_nostdlib = user_args.iter().any(|a| a == "-nostdlib");

    // Collect all input files including CRT objects
    let mut all_inputs: Vec<String> = Vec::new();
    let mut lib_search_paths: Vec<String> = Vec::new();
    let mut needed_libs: Vec<String> = Vec::new();

    // Find GCC and CRT directories
    let gcc_dir = find_gcc_dir();
    let crt_dir = find_crt_dir_rv();

    if !is_nostdlib {
        // Add CRT startup objects
        if let Some(ref crt) = crt_dir {
            all_inputs.push(format!("{}/crt1.o", crt));
        }
        if let Some(ref gcc) = gcc_dir {
            all_inputs.push(format!("{}/crti.o", gcc));
            all_inputs.push(format!("{}/crtbegin.o", gcc));
        }
    }

    // Add user object files
    for obj in object_files {
        all_inputs.push(obj.to_string());
    }

    // Parse user args for -L, -l, bare .o/.a files, and -Wl, flags
    let mut i = 0;
    let args: Vec<&str> = user_args.iter().map(|s| s.as_str()).collect();
    while i < args.len() {
        let arg = args[i];
        if let Some(rest) = arg.strip_prefix("-L") {
            if rest.is_empty() && i + 1 < args.len() {
                i += 1;
                lib_search_paths.push(args[i].to_string());
            } else {
                lib_search_paths.push(rest.to_string());
            }
        } else if let Some(rest) = arg.strip_prefix("-l") {
            let libname = if rest.is_empty() && i + 1 < args.len() {
                i += 1;
                args[i]
            } else {
                rest
            };
            needed_libs.push(libname.to_string());
        } else if let Some(wl) = arg.strip_prefix("-Wl,") {
            // Parse -Wl, flags
            for part in wl.split(',') {
                if let Some(l) = part.strip_prefix("-L") {
                    lib_search_paths.push(l.to_string());
                }
            }
        } else if !arg.starts_with('-') && std::path::Path::new(arg).exists() {
            // Bare file path: .o object file, .a static archive, or other input file
            // These come from linker_ordered_items when the driver passes pre-existing .o/.a files
            all_inputs.push(arg.to_string());
        }
        i += 1;
    }

    // Add system library search paths
    if let Some(ref gcc) = gcc_dir {
        lib_search_paths.push(gcc.clone());
    }
    if let Some(ref crt) = crt_dir {
        lib_search_paths.push(crt.clone());
    }
    lib_search_paths.push("/usr/riscv64-linux-gnu/lib".into());
    lib_search_paths.push("/usr/lib/riscv64-linux-gnu".into());
    lib_search_paths.push("/lib/riscv64-linux-gnu".into());

    // Add default libraries
    if !is_nostdlib {
        needed_libs.extend(["gcc", "gcc_s", "c", "m"].iter().map(|s| s.to_string()));
    }

    // Add CRT finalization objects
    if !is_nostdlib {
        if let Some(ref gcc) = gcc_dir {
            all_inputs.push(format!("{}/crtend.o", gcc));
            all_inputs.push(format!("{}/crtn.o", gcc));
        }
    }

    // ── Phase 1: Read all input object files ────────────────────────────

    let mut input_objs: Vec<(String, ObjFile)> = Vec::new();

    for path in &all_inputs {
        if !std::path::Path::new(path).exists() {
            // Silently skip missing files for flexibility
            continue;
        }
        let data = std::fs::read(path)
            .map_err(|e| format!("Cannot read {}: {}", path, e))?;

        if data.len() >= 8 && &data[0..8] == b"!<arch>\n" {
            // Archive: extract members
            let members = parse_archive(&data)
                .map_err(|e| format!("{}: {}", path, e))?;
            input_objs.extend(members);
        } else if data.len() >= 4 && &data[0..4] == b"\x7fELF" {
            let obj = parse_obj(&data)
                .map_err(|e| format!("{}: {}", path, e))?;
            input_objs.push((path.clone(), obj));
        }
        // Skip other file types silently (linker scripts etc.)
    }

    // ── Phase 1b: Resolve -l libraries ──────────────────────────────────

    // Build a set of defined symbols from input objects
    let mut defined_syms: HashSet<String> = HashSet::new();
    let mut undefined_syms: HashSet<String> = HashSet::new();

    for (_, obj) in &input_objs {
        for sym in &obj.symbols {
            if sym.section_idx != SHN_UNDEF && sym.binding != STB_LOCAL && !sym.name.is_empty() {
                defined_syms.insert(sym.name.clone());
            }
        }
    }
    for (_, obj) in &input_objs {
        for sym in &obj.symbols {
            if sym.section_idx == SHN_UNDEF && !sym.name.is_empty() && sym.binding != STB_LOCAL {
                if !defined_syms.contains(&sym.name) {
                    undefined_syms.insert(sym.name.clone());
                }
            }
        }
    }

    // ── Phase 1b: Discover shared library symbols ─────────────────────

    let mut shared_lib_syms: HashMap<String, SharedSymInfo> = HashMap::new();
    let mut actual_needed_libs: Vec<String> = Vec::new();

    if !is_static {
        for libname in &needed_libs {
            // -l:filename means search for exact filename
            let so_name = if let Some(exact) = libname.strip_prefix(':') {
                exact.to_string()
            } else {
                format!("lib{}.so", libname)
            };
            for dir in &lib_search_paths {
                let path = format!("{}/{}", dir, so_name);
                if std::path::Path::new(&path).exists() {
                    let data = match std::fs::read(&path) {
                        Ok(d) => d,
                        Err(_) => continue,
                    };
                    if data.starts_with(b"/* GNU ld script") || data.starts_with(b"OUTPUT_FORMAT") || data.starts_with(b"GROUP") {
                        let text = String::from_utf8_lossy(&data);
                        for token in text.split_whitespace() {
                            let token = token.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
                            if token.contains(".so") && (token.starts_with('/') || token.starts_with("lib")) {
                                let actual_path = if token.starts_with('/') {
                                    token.to_string()
                                } else {
                                    format!("{}/{}", dir, token)
                                };
                                if let Ok(syms) = read_shared_lib_symbols(&actual_path) {
                                    for si in syms {
                                        shared_lib_syms.insert(si.name.clone(), si);
                                    }
                                }
                                // Also try to pull in _nonshared.a from linker script
                                if actual_path.ends_with("_nonshared.a") {
                                    if let Ok(archive_data) = std::fs::read(&actual_path) {
                                        if archive_data.len() >= 8 && &archive_data[0..8] == b"!<arch>\n" {
                                            if let Ok(members) = parse_archive(&archive_data) {
                                                resolve_archive_members(
                                                    members, &mut input_objs,
                                                    &mut defined_syms, &mut undefined_syms,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if data.len() >= 4 && &data[0..4] == b"\x7fELF" {
                        if let Ok(syms) = read_shared_lib_symbols(&path) {
                            for si in syms {
                                shared_lib_syms.insert(si.name.clone(), si);
                            }
                        }
                    }
                    let versioned = find_versioned_soname(dir, libname);
                    if let Some(soname) = versioned {
                        if !actual_needed_libs.contains(&soname) {
                            actual_needed_libs.push(soname);
                        }
                    }
                    break;
                }
            }
        }
    }

    // Treat symbols available from shared libs as "defined" for archive resolution
    // This prevents pulling in static versions of functions that the dynamic linker will provide
    let shared_defined: HashSet<String> = shared_lib_syms.keys()
        .filter(|s| undefined_syms.contains(*s))
        .cloned()
        .collect();
    for s in &shared_defined {
        undefined_syms.remove(s);
    }

    // ── Phase 1c: Resolve remaining undefined symbols from .a archives ──

    let mut resolved_archives: HashSet<String> = HashSet::new();
    for libname in &needed_libs {
        // -l:filename means search for exact filename
        let archive_name = if let Some(exact) = libname.strip_prefix(':') {
            exact.to_string()
        } else {
            format!("lib{}.a", libname)
        };
        for dir in &lib_search_paths {
            let path = format!("{}/{}", dir, archive_name);
            if resolved_archives.contains(&path) {
                continue;
            }
            if std::path::Path::new(&path).exists() {
                let data = match std::fs::read(&path) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                if data.len() >= 8 && &data[0..8] == b"!<arch>\n" {
                    if let Ok(members) = parse_archive(&data) {
                        resolve_archive_members(
                            members, &mut input_objs,
                            &mut defined_syms, &mut undefined_syms,
                        );
                    }
                    resolved_archives.insert(path);
                }
                break;
            }
        }
    }

    // Restore shared-lib-defined symbols back into undefined_syms
    // (they'll be resolved through PLT/GOT later)
    for s in shared_defined {
        if !defined_syms.contains(&s) {
            undefined_syms.insert(s);
        }
    }

    // ── Phase 2: Merge sections ─────────────────────────────────────────

    // Collect all allocatable sections, grouped by output section name.
    // Track where each input section maps to in the merged output.
    struct InputSecRef {
        obj_idx: usize,
        sec_idx: usize,
        merged_sec_idx: usize,
        offset_in_merged: u64,
    }

    let mut merged_sections: Vec<MergedSection> = Vec::new();
    let mut merged_map: HashMap<String, usize> = HashMap::new();
    let mut input_sec_refs: Vec<InputSecRef> = Vec::new();

    // Canonical output section ordering
    let section_order = |name: &str, flags: u64| -> u64 {
        match name {
            ".text" => 100,
            ".rodata" => 200,
            ".eh_frame_hdr" => 250,
            ".eh_frame" => 260,
            ".preinit_array" => 500,
            ".init_array" => 510,
            ".fini_array" => 520,
            ".data" => 600,
            ".sdata" => 650,
            ".bss" | ".sbss" => 700,
            _ if flags & SHF_EXECINSTR != 0 => 150,
            _ if flags & SHF_WRITE == 0 => 300,
            _ => 600,
        }
    };

    // Determine the output section name for an input section.
    fn output_section_name(name: &str, sh_type: u32, sh_flags: u64) -> Option<String> {
        // Skip non-alloc sections (except .riscv.attributes which we handle separately)
        if sh_flags & SHF_ALLOC == 0 && sh_type != SHT_RISCV_ATTRIBUTES {
            return None;
        }
        // Skip group sections
        if sh_type == SHT_GROUP {
            return None;
        }
        // Map .text.* -> .text, .data.* -> .data, etc.
        if let Some(rest) = name.strip_prefix(".text.") {
            let _ = rest;
            return Some(".text".into());
        }
        if let Some(rest) = name.strip_prefix(".rodata.") {
            let _ = rest;
            return Some(".rodata".into());
        }
        if let Some(rest) = name.strip_prefix(".data.rel.ro") {
            let _ = rest;
            return Some(".data".into());
        }
        if let Some(rest) = name.strip_prefix(".data.") {
            let _ = rest;
            return Some(".data".into());
        }
        if let Some(rest) = name.strip_prefix(".bss.") {
            let _ = rest;
            return Some(".bss".into());
        }
        if let Some(rest) = name.strip_prefix(".sdata.") {
            let _ = rest;
            return Some(".sdata".into());
        }
        if let Some(rest) = name.strip_prefix(".sbss.") {
            let _ = rest;
            return Some(".sbss".into());
        }
        if name == ".init_array" || name.starts_with(".init_array.") {
            return Some(".init_array".into());
        }
        if name == ".fini_array" || name.starts_with(".fini_array.") {
            return Some(".fini_array".into());
        }
        if name == ".preinit_array" || name.starts_with(".preinit_array.") {
            return Some(".preinit_array".into());
        }
        // Keep other sections as-is
        Some(name.to_string())
    }

    for (obj_idx, (_, obj)) in input_objs.iter().enumerate() {
        for (sec_idx, sec) in obj.sections.iter().enumerate() {
            if sec.name.is_empty() || sec.sh_type == SHT_SYMTAB || sec.sh_type == SHT_STRTAB
                || sec.sh_type == SHT_RELA || sec.sh_type == SHT_GROUP
            {
                continue;
            }

            let out_name = match output_section_name(&sec.name, sec.sh_type, sec.sh_flags) {
                Some(n) => n,
                None => continue,
            };

            // Skip .riscv.attributes and .note.* for merging (handled separately)
            if out_name == ".riscv.attributes" || out_name.starts_with(".note.") {
                // We still need to track these for the output
                if out_name == ".note.GNU-stack" {
                    continue; // Skip stack notes, we generate our own
                }
                // For .riscv.attributes, keep the first one
                if !merged_map.contains_key(&out_name) {
                    let idx = merged_sections.len();
                    merged_map.insert(out_name.clone(), idx);
                    merged_sections.push(MergedSection {
                        name: out_name.clone(),
                        sh_type: sec.sh_type,
                        sh_flags: sec.sh_flags,
                        data: sec.data.clone(),
                        vaddr: 0,
                        align: sec.sh_addralign.max(1),
                    });
                    input_sec_refs.push(InputSecRef {
                        obj_idx,
                        sec_idx,
                        merged_sec_idx: idx,
                        offset_in_merged: 0,
                    });
                }
                continue;
            }

            let merged_idx = if let Some(&idx) = merged_map.get(&out_name) {
                idx
            } else {
                let idx = merged_sections.len();
                let is_bss = sec.sh_type == SHT_NOBITS || out_name == ".bss" || out_name == ".sbss";
                merged_map.insert(out_name.clone(), idx);
                merged_sections.push(MergedSection {
                    name: out_name.clone(),
                    sh_type: if is_bss { SHT_NOBITS } else { sec.sh_type },
                    sh_flags: sec.sh_flags,
                    data: Vec::new(),
                    vaddr: 0,
                    align: sec.sh_addralign.max(1),
                });
                idx
            };

            let ms = &mut merged_sections[merged_idx];
            // Update flags (union of all inputs)
            ms.sh_flags |= sec.sh_flags & (SHF_WRITE | SHF_ALLOC | SHF_EXECINSTR | SHF_TLS);
            ms.align = ms.align.max(sec.sh_addralign.max(1));

            // Align data
            let align = sec.sh_addralign.max(1) as usize;
            let cur_len = ms.data.len();
            let aligned = (cur_len + align - 1) & !(align - 1);
            if aligned > cur_len {
                ms.data.resize(aligned, 0);
            }
            let offset_in_merged = ms.data.len() as u64;

            // Append section data
            if sec.sh_type == SHT_NOBITS {
                // For BSS, just extend with zeros
                ms.data.resize(ms.data.len() + sec.data.len(), 0);
            } else {
                ms.data.extend_from_slice(&sec.data);
            }

            input_sec_refs.push(InputSecRef {
                obj_idx,
                sec_idx,
                merged_sec_idx: merged_idx,
                offset_in_merged,
            });
        }
    }

    // ── Phase 3: Build global symbol table ──────────────────────────────

    // Map: (obj_idx, sec_idx) -> (merged_sec_idx, offset_in_merged)
    let mut sec_mapping: HashMap<(usize, usize), (usize, u64)> = HashMap::new();
    for r in &input_sec_refs {
        sec_mapping.insert((r.obj_idx, r.sec_idx), (r.merged_sec_idx, r.offset_in_merged));
    }

    let mut global_syms: HashMap<String, GlobalSym> = HashMap::new();

    // First pass: define symbols
    for (obj_idx, (_, obj)) in input_objs.iter().enumerate() {
        for sym in &obj.symbols {
            if sym.name.is_empty() || sym.binding == STB_LOCAL {
                continue;
            }
            if sym.section_idx == SHN_UNDEF {
                // Just register as undefined if not yet defined
                global_syms.entry(sym.name.clone()).or_insert_with(|| GlobalSym {
                    value: 0,
                    size: 0,
                    binding: sym.binding,
                    sym_type: sym.sym_type,
                    visibility: sym.visibility,
                    defined: false,
                    needs_plt: false,
                    plt_idx: 0,
                    got_offset: None,
                    section_idx: None,
                });
                continue;
            }

            let sec_idx = sym.section_idx as usize;
            let (merged_idx, offset) = match sec_mapping.get(&(obj_idx, sec_idx)) {
                Some(&v) => v,
                None => {
                    if sym.section_idx == SHN_ABS {
                        // Absolute symbol
                        let entry = global_syms.entry(sym.name.clone()).or_insert_with(|| GlobalSym {
                            value: sym.value,
                            size: sym.size,
                            binding: sym.binding,
                            sym_type: sym.sym_type,
                            visibility: sym.visibility,
                            defined: true,
                            needs_plt: false,
                            plt_idx: 0,
                            got_offset: None,
                            section_idx: None,
                        });
                        if !entry.defined || (entry.binding == STB_WEAK && sym.binding == STB_GLOBAL) {
                            entry.value = sym.value;
                            entry.size = sym.size;
                            entry.binding = sym.binding;
                            entry.sym_type = sym.sym_type;
                            entry.defined = true;
                        }
                        continue;
                    }
                    if sym.section_idx == SHN_COMMON {
                        // COMMON symbol: allocate in .bss
                        let bss_idx = *merged_map.entry(".bss".into()).or_insert_with(|| {
                            let idx = merged_sections.len();
                            merged_sections.push(MergedSection {
                                name: ".bss".into(),
                                sh_type: SHT_NOBITS,
                                sh_flags: SHF_ALLOC | SHF_WRITE,
                                data: Vec::new(),
                                vaddr: 0,
                                align: 8,
                            });
                            idx
                        });
                        let ms = &mut merged_sections[bss_idx];
                        let align = sym.value.max(1) as usize; // st_value is alignment for COMMON
                        let cur = ms.data.len();
                        let aligned = (cur + align - 1) & !(align - 1);
                        ms.data.resize(aligned, 0);
                        let off = ms.data.len() as u64;
                        ms.data.resize(ms.data.len() + sym.size as usize, 0);
                        ms.align = ms.align.max(align as u64);

                        let entry = global_syms.entry(sym.name.clone()).or_insert_with(|| GlobalSym {
                            value: off, size: sym.size, binding: sym.binding,
                            sym_type: STT_OBJECT, visibility: sym.visibility,
                            defined: true, needs_plt: false, plt_idx: 0,
                            got_offset: None, section_idx: Some(bss_idx),
                        });
                        if !entry.defined || (entry.binding == STB_WEAK && sym.binding == STB_GLOBAL) {
                            entry.value = off; // Will be fixed up when vaddr is assigned
                            entry.size = sym.size.max(entry.size); // Use largest size
                            entry.binding = sym.binding;
                            entry.defined = true;
                            entry.section_idx = Some(bss_idx);
                        }
                        continue;
                    }
                    continue;
                }
            };

            let entry = global_syms.entry(sym.name.clone()).or_insert_with(|| GlobalSym {
                value: 0, size: sym.size, binding: sym.binding,
                sym_type: sym.sym_type, visibility: sym.visibility,
                defined: false, needs_plt: false, plt_idx: 0,
                got_offset: None, section_idx: None,
            });

            // Don't override a strong definition with a weak one
            if entry.defined && entry.binding == STB_GLOBAL && sym.binding == STB_WEAK {
                continue;
            }

            entry.value = offset + sym.value;
            entry.size = sym.size;
            entry.binding = sym.binding;
            entry.sym_type = sym.sym_type;
            entry.visibility = sym.visibility;
            entry.defined = true;
            entry.section_idx = Some(merged_idx);
        }
    }



    // Mark symbols that need PLT/GOT (undefined function symbols found in shared libs)
    // Data symbols (STT_OBJECT) get COPY relocations instead
    let mut plt_symbols: Vec<String> = Vec::new();
    let mut copy_symbols: Vec<(String, u64)> = Vec::new(); // (name, size) for R_RISCV_COPY
    for (name, sym) in global_syms.iter_mut() {
        if !sym.defined {
            if let Some(shlib_sym) = shared_lib_syms.get(name) {
                if shlib_sym.sym_type == STT_OBJECT {
                    // Data symbol: will use COPY relocation
                    copy_symbols.push((name.clone(), shlib_sym.size));
                } else {
                    // Function symbol: use PLT
                    sym.needs_plt = true;
                    sym.plt_idx = plt_symbols.len();
                    plt_symbols.push(name.clone());
                }
            }
        }
    }

    // Also identify symbols that need GOT entries (referenced via GOT_HI20)
    let mut got_symbols: Vec<String> = Vec::new();
    let mut tls_got_symbols: HashSet<String> = HashSet::new();
    {
        let mut got_set: HashSet<String> = HashSet::new();
        for (_obj_idx, (_, obj)) in input_objs.iter().enumerate() {
            for (_sec_idx, relocs) in &obj.relocs {
                for reloc in relocs {
                    if reloc.reloc_type == R_RISCV_GOT_HI20
                        || reloc.reloc_type == R_RISCV_TLS_GOT_HI20
                        || reloc.reloc_type == R_RISCV_TLS_GD_HI20
                    {
                        let sym = &obj.symbols[reloc.symbol_idx as usize];
                        let name = if sym.sym_type == STT_SECTION {
                            let sec = &obj.sections[sym.section_idx as usize];
                            sec.name.clone()
                        } else {
                            sym.name.clone()
                        };
                        if !name.is_empty() && !got_set.contains(&name) {
                            got_set.insert(name.clone());
                            got_symbols.push(name.clone());
                        }
                        // Track TLS GOT symbols so we can fill them with TP offsets
                        if reloc.reloc_type == R_RISCV_TLS_GOT_HI20
                            || reloc.reloc_type == R_RISCV_TLS_GD_HI20
                        {
                            tls_got_symbols.insert(name);
                        }
                    }
                }
            }
        }
    }

    // ── Phase 4: Layout sections and assign virtual addresses ───────────

    // Sort sections by canonical order
    let mut sec_indices: Vec<usize> = (0..merged_sections.len()).collect();
    sec_indices.sort_by_key(|&i| {
        let ms = &merged_sections[i];
        section_order(&ms.name, ms.sh_flags)
    });

    // We need to know which sections go in which segment.
    // Segment 0: RX (read-execute) - .text, .rodata, .eh_frame, etc.
    // Segment 1: RW (read-write) - .data, .bss, .dynamic, .got, etc.

    // First, compute sizes for generated sections
    let plt_entry_size: u64 = 16;
    let plt_header_size: u64 = 32; // PLT[0] stub
    let plt_size = if is_static || plt_symbols.is_empty() { 0 } else { plt_header_size + plt_symbols.len() as u64 * plt_entry_size };
    let got_plt_entries = if is_static { 0 } else { plt_symbols.len() + 2 }; // 2 reserved entries + 1 per PLT symbol (RISC-V: [0]=resolver, [1]=link_map)
    let got_plt_size = got_plt_entries as u64 * 8;
    let got_size = got_symbols.len() as u64 * 8;

    // Compute interp size (not needed for static binaries)
    let interp_size = if is_static { 0 } else { INTERP.len() as u64 };

    // Dynamic section needs: estimate entry count
    // Each entry is 16 bytes (d_tag + d_val)
    let dynamic_size = if is_static {
        0
    } else {
        let dyn_entry_count = 30 + actual_needed_libs.len(); // generous estimate
        dyn_entry_count as u64 * 16
    };

    // Compute rela.dyn size (for COPY relocations)
    let rela_dyn_size = if is_static { 0 } else { copy_symbols.len() as u64 * 24 };

    // Compute rela.plt size
    let rela_plt_size = if is_static { 0 } else { plt_symbols.len() as u64 * 24 };

    // We'll build the ELF as:
    // File offset 0: ELF header (64 bytes)
    // File offset 64: program headers (56 * num_phdrs)
    // For dynamic: LOAD segment 0 (RX): .interp, .gnu.hash, .dynsym, .dynstr, .versym, .verneed, .rela.plt, .plt, .text, .rodata, .eh_frame
    //              LOAD segment 1 (RW): .init_array, .fini_array, .dynamic, .got, .got.plt, .data, .sdata, .bss
    // For static:  LOAD segment 0 (RX): .text, .rodata, .eh_frame
    //              LOAD segment 1 (RW): .init_array, .fini_array, .got, .data, .sdata, .bss

    // Check if we have TLS sections
    let has_tls = merged_sections.iter().any(|ms| ms.sh_flags & SHF_TLS != 0);

    // Estimate phdr count
    // Dynamic: PHDR, INTERP, LOAD(RX), LOAD(RW), DYNAMIC, NOTE, GNU_EH_FRAME, GNU_STACK, GNU_RELRO, RISCV_ATTR [, TLS]
    // Static: LOAD(RX), LOAD(RW), NOTE, GNU_EH_FRAME, GNU_STACK, RISCV_ATTR [, TLS]
    let num_phdrs = if is_static {
        let base = 6; // LOAD(RX), LOAD(RW), NOTE, GNU_EH_FRAME, GNU_STACK, RISCV_ATTR
        if has_tls { base + 1 } else { base }
    } else {
        if has_tls { 11 } else { 10 } // PHDR, INTERP, LOAD(RX), LOAD(RW), DYNAMIC, NOTE, GNU_EH_FRAME, GNU_STACK, GNU_RELRO, RISCV_ATTR [, TLS]
    };
    let phdr_size = num_phdrs * 56u64;
    let headers_size = 64 + phdr_size;

    // Start laying out the RX segment
    let mut file_offset = headers_size;
    let mut vaddr = BASE_ADDR + headers_size;

    // Dynamic linking sections (interp, dynsym, dynstr, gnu.hash, plt, rela.plt, etc.)
    // are only needed for dynamically-linked binaries.
    let mut interp_offset = 0u64;
    let mut interp_vaddr = 0u64;
    let mut gnu_hash_vaddr = 0u64;
    let mut gnu_hash_offset = 0u64;
    let mut dynsym_vaddr = 0u64;
    let mut dynsym_offset = 0u64;
    let mut dynstr_vaddr = 0u64;
    let mut dynstr_offset = 0u64;
    let mut versym_vaddr = 0u64;
    let mut versym_offset = 0u64;
    let mut verneed_vaddr = 0u64;
    let mut verneed_offset = 0u64;
    let mut rela_dyn_vaddr = 0u64;
    let mut rela_dyn_offset = 0u64;
    let mut rela_plt_vaddr = 0u64;
    let mut rela_plt_offset = 0u64;
    let mut plt_vaddr = 0u64;
    let mut plt_offset = 0u64;

    let mut dynstr_data = vec![0u8]; // Leading null (minimum)
    let mut dynsym_data = vec![0u8; 24]; // null entry (minimum)
    let mut gnu_hash_data = Vec::new();
    let mut versym_data: Vec<u8> = Vec::new();
    let mut verneed_data: Vec<u8> = Vec::new();
    let mut needed_lib_offsets: Vec<u32> = Vec::new();
    let mut dynsym_names: Vec<String> = Vec::new();
    let copy_sym_names: Vec<String> = copy_symbols.iter().map(|(n, _)| n.clone()).collect();

    if !is_static {
        // .interp
        interp_offset = file_offset;
        interp_vaddr = vaddr;
        file_offset += interp_size;
        vaddr += interp_size;

        // Build .gnu.hash, .dynsym, .dynstr for dynamic symbols
        // dynsym contains: PLT symbols first, then COPY symbols
        dynsym_names = plt_symbols.clone();
        dynsym_names.extend(copy_sym_names.iter().cloned());

        // Build dynstr
        dynstr_data = vec![0u8]; // Leading null
        let mut dynstr_offsets: HashMap<String, u32> = HashMap::new();
        for name in &dynsym_names {
            let off = dynstr_data.len() as u32;
            dynstr_offsets.insert(name.clone(), off);
            dynstr_data.extend_from_slice(name.as_bytes());
            dynstr_data.push(0);
        }
        // Add needed library names
        for lib in &actual_needed_libs {
            let off = dynstr_data.len() as u32;
            needed_lib_offsets.push(off);
            dynstr_data.extend_from_slice(lib.as_bytes());
            dynstr_data.push(0);
        }
        // No version strings needed - unversioned symbols match default versions.

        // Build dynsym (null entry + PLT symbols as FUNC + COPY symbols as OBJECT)
        // COPY symbol entries will be patched after layout to set st_value, st_size, st_shndx
        dynsym_data = vec![0u8; 24]; // null entry
        let copy_sym_set: HashSet<String> = copy_sym_names.iter().cloned().collect();
        for (_i, name) in dynsym_names.iter().enumerate() {
            let mut entry = [0u8; 24];
            let name_off = dynstr_offsets.get(name).copied().unwrap_or(0);
            entry[0..4].copy_from_slice(&name_off.to_le_bytes());
            if copy_sym_set.contains(name) {
                // COPY-relocated data symbol - st_info: GLOBAL OBJECT
                entry[4] = (STB_GLOBAL << 4) | STT_OBJECT;
                // st_shndx, st_value, st_size will be patched later
            } else {
                // PLT function symbol - st_info: GLOBAL FUNC
                entry[4] = (STB_GLOBAL << 4) | STT_FUNC;
            }
            // st_other: DEFAULT
            entry[5] = STV_DEFAULT;
            // st_shndx: UND (patched later for COPY symbols)
            entry[6..8].copy_from_slice(&0u16.to_le_bytes());
            // st_value: 0 (patched later for COPY symbols)
            // st_size: 0 (patched later for COPY symbols)
            dynsym_data.extend_from_slice(&entry);
        }

        // Build .gnu.hash (minimal)
        gnu_hash_data = build_gnu_hash(dynsym_names.len());

        // Skip .gnu.version and .gnu.version_r for simplicity.
        // The dynamic linker handles unversioned symbols fine - they just match
        // the default version of the symbol in the shared library.

        // Layout generated RX sections:
        // .gnu.hash
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        gnu_hash_vaddr = vaddr;
        gnu_hash_offset = file_offset;
        file_offset += gnu_hash_data.len() as u64;
        vaddr += gnu_hash_data.len() as u64;

        // .dynsym
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        dynsym_vaddr = vaddr;
        dynsym_offset = file_offset;
        file_offset += dynsym_data.len() as u64;
        vaddr += dynsym_data.len() as u64;

        // .dynstr
        dynstr_vaddr = vaddr;
        dynstr_offset = file_offset;
        file_offset += dynstr_data.len() as u64;
        vaddr += dynstr_data.len() as u64;

        // .gnu.version
        vaddr = align_up(vaddr, 2);
        file_offset = align_up(file_offset, 2);
        versym_vaddr = vaddr;
        versym_offset = file_offset;
        file_offset += versym_data.len() as u64;
        vaddr += versym_data.len() as u64;

        // .gnu.version_r
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        verneed_vaddr = vaddr;
        verneed_offset = file_offset;
        file_offset += verneed_data.len() as u64;
        vaddr += verneed_data.len() as u64;

        // .rela.dyn (for COPY relocations)
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        rela_dyn_vaddr = vaddr;
        rela_dyn_offset = file_offset;
        file_offset += rela_dyn_size;
        vaddr += rela_dyn_size;

        // .rela.plt
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        rela_plt_vaddr = vaddr;
        rela_plt_offset = file_offset;
        file_offset += rela_plt_size;
        vaddr += rela_plt_size;

        // .plt
        vaddr = align_up(vaddr, 16);
        file_offset = align_up(file_offset, 16);
        plt_vaddr = vaddr;
        plt_offset = file_offset;
        file_offset += plt_size;
        vaddr += plt_size;
    }

    // Now layout user code/data sections in the RX segment
    // .text section
    let mut section_vaddrs: Vec<u64> = vec![0; merged_sections.len()];
    let mut section_offsets: Vec<u64> = vec![0; merged_sections.len()];

    // Assign vaddrs to RX sections (alloc, non-write)
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.sh_flags & SHF_ALLOC == 0 {
            continue;
        }
        if ms.sh_flags & SHF_WRITE != 0 {
            continue; // RW sections go in segment 1
        }
        if ms.name == ".riscv.attributes" || ms.name.starts_with(".note.") {
            continue; // Handled separately
        }

        let align = ms.align.max(1);
        vaddr = align_up(vaddr, align);
        file_offset = align_up(file_offset, align);

        section_vaddrs[si] = vaddr;
        section_offsets[si] = file_offset;

        let size = ms.data.len() as u64;
        file_offset += size;
        vaddr += size;
    }

    let rx_segment_end_vaddr = vaddr;
    let rx_segment_end_offset = file_offset;

    // Start RW segment on a new page
    file_offset = align_up(file_offset, PAGE_SIZE);
    vaddr = align_up(vaddr, PAGE_SIZE);
    // Ensure file_offset and vaddr are congruent mod PAGE_SIZE
    if (vaddr % PAGE_SIZE) != (file_offset % PAGE_SIZE) {
        vaddr = align_up(vaddr, PAGE_SIZE) + (file_offset % PAGE_SIZE);
    }

    let rw_segment_start_vaddr = vaddr;
    let rw_segment_start_offset = file_offset;

    // Layout RW sections: .init_array, .fini_array, .preinit_array first (for RELRO)
    let init_array_sections = [".preinit_array", ".init_array", ".fini_array"];
    let mut init_array_vaddrs: HashMap<String, (u64, u64)> = HashMap::new(); // name -> (vaddr, size)

    for sect_name in &init_array_sections {
        if let Some(&si) = merged_map.get(*sect_name) {
            let ms = &merged_sections[si];
            let align = ms.align.max(8);
            vaddr = align_up(vaddr, align);
            file_offset = align_up(file_offset, align);
            section_vaddrs[si] = vaddr;
            section_offsets[si] = file_offset;
            init_array_vaddrs.insert(sect_name.to_string(), (vaddr, ms.data.len() as u64));
            let size = ms.data.len() as u64;
            file_offset += size;
            vaddr += size;
        }
    }

    // .dynamic (only for dynamic binaries)
    let mut dynamic_vaddr = 0u64;
    let mut dynamic_offset = 0u64;
    if !is_static {
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        dynamic_vaddr = vaddr;
        dynamic_offset = file_offset;
        file_offset += dynamic_size;
        vaddr += dynamic_size;
    }

    // .got (regular GOT for pcrel GOT references - needed even in static builds)
    vaddr = align_up(vaddr, 8);
    file_offset = align_up(file_offset, 8);
    let got_vaddr = vaddr;
    let got_offset = file_offset;
    file_offset += got_size;
    vaddr += got_size;

    // Assign GOT offsets to symbols
    for (i, name) in got_symbols.iter().enumerate() {
        if let Some(sym) = global_syms.get_mut(name) {
            sym.got_offset = Some(i as u64 * 8);
        }
    }

    // RELRO covers everything up to here (init/fini arrays, .dynamic, .got)
    // We need .got.plt and .data/.sdata OUTSIDE the RELRO region.
    // The dynamic linker mprotects the RELRO region to read-only, so .got.plt must not be in it.
    let relro_end_offset = file_offset;
    let relro_end_vaddr = vaddr;

    // .got.plt (only for dynamic binaries)
    let mut got_plt_vaddr = 0u64;
    let mut got_plt_offset = 0u64;
    if !is_static {
        vaddr = align_up(vaddr, 8);
        file_offset = align_up(file_offset, 8);
        got_plt_vaddr = vaddr;
        got_plt_offset = file_offset;
        file_offset += got_plt_size;
        vaddr += got_plt_size;

        // Assign PLT symbol GOT.PLT addresses
        for (i, name) in plt_symbols.iter().enumerate() {
            if let Some(sym) = global_syms.get_mut(name) {
                // GOT.PLT entry is at got_plt_vaddr + (2 + i) * 8
                sym.value = plt_vaddr + plt_header_size + i as u64 * plt_entry_size;
            }
        }
    }

    // Remaining RW sections (.data, .sdata, etc.) - skip TLS sections
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.sh_flags & SHF_ALLOC == 0 {
            continue;
        }
        if ms.sh_flags & SHF_WRITE == 0 {
            continue; // Already handled
        }
        // Skip sections we already placed
        if init_array_sections.contains(&ms.name.as_str()) {
            continue;
        }
        if ms.name == ".bss" || ms.name == ".sbss" {
            continue; // BSS goes last
        }
        // Skip TLS sections - they'll be placed separately
        if ms.sh_flags & SHF_TLS != 0 {
            continue;
        }

        let align = ms.align.max(1);
        vaddr = align_up(vaddr, align);
        file_offset = align_up(file_offset, align);
        section_vaddrs[si] = vaddr;
        section_offsets[si] = file_offset;

        let size = ms.data.len() as u64;
        file_offset += size;
        vaddr += size;
    }

    // TLS sections (.tdata, .tbss) - placed in RW segment with tracked boundaries
    let mut tls_vaddr = 0u64;
    let mut tls_offset = 0u64;
    let mut tls_filesz = 0u64;
    let mut tls_memsz = 0u64;
    let mut tls_align = 1u64;

    if has_tls {
        // .tdata first (initialized TLS data)
        for &si in &sec_indices {
            let ms = &merged_sections[si];
            if ms.sh_flags & SHF_TLS == 0 || ms.sh_type == SHT_NOBITS {
                continue;
            }
            let align = ms.align.max(1);
            vaddr = align_up(vaddr, align);
            file_offset = align_up(file_offset, align);
            if tls_vaddr == 0 {
                tls_vaddr = vaddr;
                tls_offset = file_offset;
            }
            tls_align = tls_align.max(align);
            section_vaddrs[si] = vaddr;
            section_offsets[si] = file_offset;
            let size = ms.data.len() as u64;
            file_offset += size;
            vaddr += size;
            tls_filesz += size;
        }
        tls_memsz = tls_filesz;

        // .tbss (uninitialized TLS data - no file space, but contributes to memsz)
        for &si in &sec_indices {
            let ms = &merged_sections[si];
            if ms.sh_flags & SHF_TLS == 0 || ms.sh_type != SHT_NOBITS {
                continue;
            }
            let align = ms.align.max(1);
            vaddr = align_up(vaddr, align);
            if tls_vaddr == 0 {
                tls_vaddr = vaddr;
                tls_offset = file_offset;
            }
            tls_align = tls_align.max(align);
            section_vaddrs[si] = vaddr;
            section_offsets[si] = file_offset;
            let size = ms.data.len() as u64;
            vaddr += size;
            tls_memsz += size;
        }
    }

    // .bss and .sbss (NOBITS - no file space) - skip TLS .tbss
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.name != ".bss" && ms.name != ".sbss" {
            continue;
        }
        if ms.sh_flags & SHF_TLS != 0 {
            continue; // Already handled above
        }
        let align = ms.align.max(1);
        vaddr = align_up(vaddr, align);
        section_vaddrs[si] = vaddr;
        section_offsets[si] = file_offset; // BSS shares file offset with end of data
        vaddr += ms.data.len() as u64;
    }

    // Allocate space in .bss for COPY-relocated symbols from shared libs
    // These are data objects (like stdout) that need R_RISCV_COPY relocations
    let mut copy_sym_addrs: HashMap<String, (u64, u64)> = HashMap::new(); // name -> (vaddr, size)
    for (name, size) in &copy_symbols {
        let sz = if *size > 0 { *size } else { 8 }; // default to pointer-size if unknown
        let align = sz.min(8); // align to size or 8, whichever is smaller
        vaddr = align_up(vaddr, align);
        copy_sym_addrs.insert(name.clone(), (vaddr, sz));
        // Mark this symbol as defined with the .bss address
        if let Some(gs) = global_syms.get_mut(name) {
            gs.defined = true;
            gs.value = vaddr;
            gs.size = sz;
            gs.sym_type = STT_OBJECT;
        }
        vaddr += sz;
    }

    // Also extend .bss to include COPY symbols
    if !copy_symbols.is_empty() {
        if let Some(&bss_idx) = merged_map.get(".bss") {
            let bss_start = section_vaddrs[bss_idx];
            let new_bss_size = vaddr - bss_start;
            merged_sections[bss_idx].data.resize(new_bss_size as usize, 0);
        }
    }

    let rw_segment_end_vaddr = vaddr;
    let rw_segment_filesz = file_offset - rw_segment_start_offset;
    let rw_segment_memsz = rw_segment_end_vaddr - rw_segment_start_vaddr;

    // ── Phase 5: Fix up symbol values with final vaddrs ─────────────────

    for (_, sym) in global_syms.iter_mut() {
        if sym.defined {
            if let Some(si) = sym.section_idx {
                sym.value += section_vaddrs[si];
            }
        }
    }

    // Define linker-provided symbols
    let sdata_vaddr = merged_map.get(".sdata").map(|&i| section_vaddrs[i]).unwrap_or(0);
    let data_vaddr = merged_map.get(".data").map(|&i| section_vaddrs[i]).unwrap_or(sdata_vaddr);
    let bss_vaddr = merged_map.get(".bss").map(|&i| section_vaddrs[i]).unwrap_or(0);
    let bss_end = merged_map.get(".bss")
        .map(|&i| section_vaddrs[i] + merged_sections[i].data.len() as u64)
        .unwrap_or(bss_vaddr);

    let mut define_linker_sym = |name: &str, value: u64, binding: u8| {
        let entry = global_syms.entry(name.to_string()).or_insert_with(|| GlobalSym {
            value: 0, size: 0, binding, sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT, defined: false, needs_plt: false,
            plt_idx: 0, got_offset: None, section_idx: None,
        });
        if !entry.defined {
            entry.value = value;
            entry.defined = true;
            entry.binding = binding;
        }
    };

    // __global_pointer$ = .sdata + 0x800 (conventional GP for RISC-V small data)
    let gp_value = if sdata_vaddr != 0 {
        sdata_vaddr + 0x800
    } else {
        // If no .sdata, put GP after .data
        data_vaddr + 0x800
    };
    define_linker_sym("__global_pointer$", gp_value, STB_GLOBAL);

    // GOT / dynamic
    define_linker_sym("_GLOBAL_OFFSET_TABLE_", got_plt_vaddr, STB_GLOBAL);
    define_linker_sym("_DYNAMIC", dynamic_vaddr, STB_GLOBAL);

    // Data boundaries
    let edata_val = if sdata_vaddr != 0 {
        sdata_vaddr + merged_map.get(".sdata").map(|&i| merged_sections[i].data.len() as u64).unwrap_or(0)
    } else {
        data_vaddr + merged_map.get(".data").map(|&i| merged_sections[i].data.len() as u64).unwrap_or(0)
    };
    define_linker_sym("_edata", edata_val, STB_GLOBAL);
    define_linker_sym("__bss_start", bss_vaddr, STB_GLOBAL);
    define_linker_sym("_end", bss_end, STB_GLOBAL);
    define_linker_sym("__BSS_END__", bss_end, STB_GLOBAL);
    define_linker_sym("__SDATA_BEGIN__", sdata_vaddr, STB_GLOBAL);
    define_linker_sym("__DATA_BEGIN__", data_vaddr, STB_GLOBAL);
    define_linker_sym("data_start", data_vaddr, STB_WEAK);
    define_linker_sym("__data_start", data_vaddr, STB_GLOBAL);
    define_linker_sym("__dso_handle", data_vaddr, STB_GLOBAL);
    // _IO_stdin_used - conventionally in .rodata, a marker that libc uses
    let rodata_vaddr = merged_map.get(".rodata").map(|&i| section_vaddrs[i]).unwrap_or(0);
    define_linker_sym("_IO_stdin_used", rodata_vaddr, STB_GLOBAL);

    // Array boundary symbols (init_array, fini_array, preinit_array)
    let init_start = init_array_vaddrs.get(".init_array").map(|&(v, _)| v).unwrap_or(0);
    let init_end = init_array_vaddrs.get(".init_array").map(|&(v, s)| v + s).unwrap_or(0);
    let fini_start = init_array_vaddrs.get(".fini_array").map(|&(v, _)| v).unwrap_or(0);
    let fini_end = init_array_vaddrs.get(".fini_array").map(|&(v, s)| v + s).unwrap_or(0);
    let preinit_start = init_array_vaddrs.get(".preinit_array").map(|&(v, _)| v).unwrap_or(0);
    let preinit_end = init_array_vaddrs.get(".preinit_array").map(|&(v, s)| v + s).unwrap_or(0);
    define_linker_sym("__init_array_start", init_start, STB_GLOBAL);
    define_linker_sym("__init_array_end", init_end, STB_GLOBAL);
    define_linker_sym("__fini_array_start", fini_start, STB_GLOBAL);
    define_linker_sym("__fini_array_end", fini_end, STB_GLOBAL);
    define_linker_sym("__preinit_array_start", preinit_start, STB_GLOBAL);
    define_linker_sym("__preinit_array_end", preinit_end, STB_GLOBAL);

    // ELF header address
    define_linker_sym("__ehdr_start", BASE_ADDR, STB_GLOBAL);

    // Relocation boundaries (for IPLT, usually empty for non-PIE)
    define_linker_sym("__rela_iplt_start", 0, STB_GLOBAL);
    define_linker_sym("__rela_iplt_end", 0, STB_GLOBAL);

    // Weak symbols for optional features
    define_linker_sym("_ITM_registerTMCloneTable", 0, STB_WEAK);
    define_linker_sym("_ITM_deregisterTMCloneTable", 0, STB_WEAK);
    define_linker_sym("__pthread_initialize_minimal", 0, STB_WEAK);
    define_linker_sym("_dl_rtld_map", 0, STB_WEAK);
    define_linker_sym("__gcc_personality_v0", 0, STB_WEAK);
    define_linker_sym("_Unwind_Resume", 0, STB_WEAK);
    define_linker_sym("_Unwind_ForcedUnwind", 0, STB_WEAK);
    define_linker_sym("_Unwind_GetCFA", 0, STB_WEAK);

    // Patch dynsym entries for COPY-relocated symbols with their final addresses
    for (i, name) in dynsym_names.iter().enumerate() {
        if let Some(&(addr, size)) = copy_sym_addrs.get(name) {
            let entry_off = (i + 1) * 24; // +1 for null entry
            // st_shndx: use a non-zero section index to indicate defined
            // We use 0xFFF1 (SHN_ABS) as it signifies the symbol is defined
            // but the dynamic linker looks at st_value for the address.
            dynsym_data[entry_off + 6..entry_off + 8].copy_from_slice(&0xFFF1u16.to_le_bytes());
            // st_value
            dynsym_data[entry_off + 8..entry_off + 16].copy_from_slice(&addr.to_le_bytes());
            // st_size
            dynsym_data[entry_off + 16..entry_off + 24].copy_from_slice(&size.to_le_bytes());
        }
    }

    // Build local symbol table for relocation resolution
    // Map: (obj_idx, local_sym_idx) -> (vaddr)
    let mut local_sym_vaddrs: Vec<Vec<u64>> = Vec::new();
    for (obj_idx, (_, obj)) in input_objs.iter().enumerate() {
        let mut sym_vaddrs = vec![0u64; obj.symbols.len()];
        for (si, sym) in obj.symbols.iter().enumerate() {
            if sym.section_idx == SHN_UNDEF || sym.section_idx == SHN_ABS {
                if sym.section_idx == SHN_ABS {
                    sym_vaddrs[si] = sym.value;
                }
                continue;
            }
            if sym.section_idx == SHN_COMMON {
                // COMMON symbols were already placed in .bss
                if let Some(gs) = global_syms.get(&sym.name) {
                    sym_vaddrs[si] = gs.value;
                }
                continue;
            }
            let sec_idx = sym.section_idx as usize;
            if let Some(&(merged_idx, offset)) = sec_mapping.get(&(obj_idx, sec_idx)) {
                sym_vaddrs[si] = section_vaddrs[merged_idx] + offset + sym.value;
            }
        }
        local_sym_vaddrs.push(sym_vaddrs);
    }

    // ── Phase 6: Apply relocations ──────────────────────────────────────

    // We need to collect all relocations and apply them to the merged section data
    for (obj_idx, (obj_name, obj)) in input_objs.iter().enumerate() {
        for (&sec_idx, relocs) in &obj.relocs {
            let (merged_idx, sec_offset) = match sec_mapping.get(&(obj_idx, sec_idx)) {
                Some(&v) => v,
                None => continue,
            };

            let ms_vaddr = section_vaddrs[merged_idx];

            for reloc in relocs {
                let offset = sec_offset + reloc.offset;
                let p = ms_vaddr + offset; // PC (address of the relocation site)

                // Resolve symbol value
                let sym_idx = reloc.symbol_idx as usize;
                if sym_idx >= obj.symbols.len() {
                    continue;
                }
                let sym = &obj.symbols[sym_idx];

                let s = if sym.sym_type == STT_SECTION {
                    // Section symbol: value is the section's vaddr
                    if (sym.section_idx as usize) < obj.sections.len() {
                        if let Some(&(mi, mo)) = sec_mapping.get(&(obj_idx, sym.section_idx as usize)) {
                            section_vaddrs[mi] + mo
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                } else if !sym.name.is_empty() && sym.binding != STB_LOCAL {
                    // Global symbol
                    if let Some(gs) = global_syms.get(&sym.name) {
                        if gs.needs_plt && matches!(reloc.reloc_type, R_RISCV_CALL_PLT) {
                            // Use PLT address
                            gs.value
                        } else if gs.defined {
                            gs.value
                        } else if gs.needs_plt {
                            gs.value
                        } else {
                            // Undefined symbol - might be weak
                            0
                        }
                    } else {
                        0
                    }
                } else {
                    // Local symbol
                    local_sym_vaddrs[obj_idx][sym_idx]
                };

                let a = reloc.addend;
                let data = &mut merged_sections[merged_idx].data;
                let off = offset as usize;

                match reloc.reloc_type {
                    R_RISCV_RELAX => {
                        // Linker relaxation hint - skip for now
                        continue;
                    }
                    R_RISCV_64 => {
                        let val = (s as i64 + a) as u64;
                        if off + 8 <= data.len() {
                            data[off..off + 8].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_32 => {
                        let val = (s as i64 + a) as u32;
                        if off + 4 <= data.len() {
                            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_PCREL_HI20 => {
                        let target = s as i64 + a;
                        let pc = p as i64;
                        let offset_val = target - pc;
                        patch_u_type(data, off, offset_val as u32);
                    }
                    R_RISCV_PCREL_LO12_I => {
                        // The symbol points to the AUIPC instruction
                        // We need to find the hi20 relocation that it references
                        // and compute the low 12 bits of that relocation's value
                        let auipc_addr = s as i64 + a;

                        // Find the hi20 relocation at the auipc address
                        let hi_val = find_hi20_value(
                            obj, obj_idx, sec_idx, &sec_mapping, &section_vaddrs,
                            &local_sym_vaddrs, &global_syms, auipc_addr as u64,
                            sec_offset, got_vaddr, &got_symbols, got_plt_vaddr,
                        );
                        patch_i_type(data, off, hi_val as u32);
                    }
                    R_RISCV_PCREL_LO12_S => {
                        let auipc_addr = s as i64 + a;
                        let hi_val = find_hi20_value(
                            obj, obj_idx, sec_idx, &sec_mapping, &section_vaddrs,
                            &local_sym_vaddrs, &global_syms, auipc_addr as u64,
                            sec_offset, got_vaddr, &got_symbols, got_plt_vaddr,
                        );
                        patch_s_type(data, off, hi_val as u32);
                    }
                    R_RISCV_GOT_HI20 => {
                        // Symbol needs GOT entry
                        let sym_name = if sym.sym_type == STT_SECTION {
                            obj.sections[sym.section_idx as usize].name.clone()
                        } else {
                            sym.name.clone()
                        };

                        let got_entry_vaddr = if let Some(gs) = global_syms.get(&sym_name) {
                            if let Some(got_off) = gs.got_offset {
                                got_vaddr + got_off
                            } else {
                                // PLT symbol: use GOT.PLT
                                let plt_idx = gs.plt_idx;
                                got_plt_vaddr + (2 + plt_idx) as u64 * 8
                            }
                        } else {
                            // Find in got_symbols
                            if let Some(idx) = got_symbols.iter().position(|n| n == &sym_name) {
                                got_vaddr + idx as u64 * 8
                            } else {
                                0
                            }
                        };

                        let offset_val = got_entry_vaddr as i64 + a - p as i64;
                        patch_u_type(data, off, offset_val as u32);
                    }
                    R_RISCV_CALL_PLT => {
                        // AUIPC + JALR pair (8 bytes)
                        let target = if !sym.name.is_empty() && sym.binding != STB_LOCAL {
                            if let Some(gs) = global_syms.get(&sym.name) {
                                if gs.needs_plt {
                                    gs.value as i64
                                } else {
                                    gs.value as i64
                                }
                            } else {
                                s as i64
                            }
                        } else {
                            s as i64
                        };
                        let offset_val = target + a - p as i64;
                        let hi = ((offset_val + 0x800) >> 12) & 0xFFFFF;
                        let lo = offset_val & 0xFFF;
                        // Patch AUIPC (first 4 bytes)
                        if off + 8 <= data.len() {
                            let auipc = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            let auipc = (auipc & 0xFFF) | ((hi as u32) << 12);
                            data[off..off + 4].copy_from_slice(&auipc.to_le_bytes());
                            // Patch JALR (next 4 bytes)
                            let jalr = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
                            let jalr = (jalr & 0x000FFFFF) | ((lo as u32) << 20);
                            data[off + 4..off + 8].copy_from_slice(&jalr.to_le_bytes());
                        }
                    }
                    R_RISCV_BRANCH => {
                        let target = s as i64 + a;
                        let offset_val = target - p as i64;
                        patch_b_type(data, off, offset_val as u32);
                    }
                    R_RISCV_JAL => {
                        let target = s as i64 + a;
                        let offset_val = target - p as i64;
                        patch_j_type(data, off, offset_val as u32);
                    }
                    R_RISCV_HI20 => {
                        let val = (s as i64 + a) as u32;
                        let hi = (val.wrapping_add(0x800)) & 0xFFFFF000;
                        if off + 4 <= data.len() {
                            let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            let insn = (insn & 0xFFF) | hi;
                            data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
                        }
                    }
                    R_RISCV_LO12_I => {
                        let val = (s as i64 + a) as u32;
                        let lo = val & 0xFFF;
                        patch_i_type(data, off, lo);
                    }
                    R_RISCV_LO12_S => {
                        let val = (s as i64 + a) as u32;
                        let lo = val & 0xFFF;
                        patch_s_type(data, off, lo);
                    }
                    R_RISCV_TPREL_HI20 => {
                        // TLS Local-Exec: compute offset from TP
                        // TP points to start of TLS block; offset = symbol_vaddr - tls_base_vaddr
                        let val = (s as i64 + a - tls_vaddr as i64) as u32;
                        let hi = (val.wrapping_add(0x800)) & 0xFFFFF000;
                        if off + 4 <= data.len() {
                            let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            let insn = (insn & 0xFFF) | hi;
                            data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
                        }
                    }
                    R_RISCV_TPREL_LO12_I => {
                        let val = (s as i64 + a - tls_vaddr as i64) as u32;
                        patch_i_type(data, off, val & 0xFFF);
                    }
                    R_RISCV_TPREL_LO12_S => {
                        let val = (s as i64 + a - tls_vaddr as i64) as u32;
                        patch_s_type(data, off, val & 0xFFF);
                    }
                    R_RISCV_TPREL_ADD => {
                        // Hint for linker relaxation - no patching needed
                    }
                    R_RISCV_ADD8 => {
                        if off < data.len() {
                            data[off] = data[off].wrapping_add((s as i64 + a) as u8);
                        }
                    }
                    R_RISCV_ADD16 => {
                        if off + 2 <= data.len() {
                            let cur = u16::from_le_bytes(data[off..off + 2].try_into().unwrap());
                            let val = cur.wrapping_add((s as i64 + a) as u16);
                            data[off..off + 2].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_ADD32 => {
                        if off + 4 <= data.len() {
                            let cur = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            let val = cur.wrapping_add((s as i64 + a) as u32);
                            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_ADD64 => {
                        if off + 8 <= data.len() {
                            let cur = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                            let val = cur.wrapping_add((s as i64 + a) as u64);
                            data[off..off + 8].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_SUB8 => {
                        if off < data.len() {
                            data[off] = data[off].wrapping_sub((s as i64 + a) as u8);
                        }
                    }
                    R_RISCV_SUB16 => {
                        if off + 2 <= data.len() {
                            let cur = u16::from_le_bytes(data[off..off + 2].try_into().unwrap());
                            let val = cur.wrapping_sub((s as i64 + a) as u16);
                            data[off..off + 2].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_SUB32 => {
                        if off + 4 <= data.len() {
                            let cur = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            let val = cur.wrapping_sub((s as i64 + a) as u32);
                            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_SUB64 => {
                        if off + 8 <= data.len() {
                            let cur = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                            let val = cur.wrapping_sub((s as i64 + a) as u64);
                            data[off..off + 8].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_SET6 => {
                        if off < data.len() {
                            data[off] = (data[off] & 0xC0) | (((s as i64 + a) as u8) & 0x3F);
                        }
                    }
                    R_RISCV_SUB6 => {
                        if off < data.len() {
                            let cur = data[off] & 0x3F;
                            let val = cur.wrapping_sub((s as i64 + a) as u8) & 0x3F;
                            data[off] = (data[off] & 0xC0) | val;
                        }
                    }
                    R_RISCV_SET8 => {
                        if off < data.len() {
                            data[off] = (s as i64 + a) as u8;
                        }
                    }
                    R_RISCV_SET16 => {
                        if off + 2 <= data.len() {
                            let val = (s as i64 + a) as u16;
                            data[off..off + 2].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_SET32 => {
                        if off + 4 <= data.len() {
                            let val = (s as i64 + a) as u32;
                            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_32_PCREL => {
                        if off + 4 <= data.len() {
                            let val = ((s as i64 + a) - p as i64) as u32;
                            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                    R_RISCV_TLS_GOT_HI20 | R_RISCV_TLS_GD_HI20 => {
                        // TLS GOT reference: auipc should point to the GOT entry
                        // that holds the TP offset for this TLS symbol
                        let sym_name = if sym.sym_type == STT_SECTION {
                            obj.sections[sym.section_idx as usize].name.clone()
                        } else {
                            sym.name.clone()
                        };

                        let got_entry_vaddr = if let Some(gs) = global_syms.get(&sym_name) {
                            if let Some(got_off) = gs.got_offset {
                                got_vaddr + got_off
                            } else if let Some(idx) = got_symbols.iter().position(|n| n == &sym_name) {
                                got_vaddr + idx as u64 * 8
                            } else {
                                0
                            }
                        } else if let Some(idx) = got_symbols.iter().position(|n| n == &sym_name) {
                            got_vaddr + idx as u64 * 8
                        } else {
                            0
                        };

                        let offset_val = got_entry_vaddr as i64 + a - p as i64;
                        patch_u_type(data, off, offset_val as u32);
                    }
                    _ => {
                        // Unknown relocation type - skip silently
                    }
                }
            }
        }
    }

    // ── Phase 7: Build GOT data ─────────────────────────────────────────

    let mut got_data = vec![0u8; got_size as usize];
    for (i, name) in got_symbols.iter().enumerate() {
        let val = if let Some(gs) = global_syms.get(name) {
            if tls_got_symbols.contains(name) {
                // TLS GOT entry: store the TP offset (symbol_vaddr - tls_vaddr)
                // For Initial-Exec TLS model, the GOT entry holds the offset from TP
                gs.value.wrapping_sub(tls_vaddr)
            } else {
                gs.value
            }
        } else {
            0
        };
        let off = i * 8;
        if off + 8 <= got_data.len() {
            got_data[off..off + 8].copy_from_slice(&val.to_le_bytes());
        }
    }

    // Build GOT.PLT data
    // RISC-V GOT.PLT layout: [0]=resolver (set by ld.so), [1]=link_map (set by ld.so), [2+]=PLT entries
    let mut got_plt_data = vec![0u8; got_plt_size as usize];
    // got_plt[0] = 0 (reserved for resolver, filled by dynamic linker)
    // got_plt[1] = 0 (reserved for link_map, filled by dynamic linker)
    // got_plt[2..] = PLT[0] address (lazy binding stub)
    for i in 0..plt_symbols.len() {
        let off = (2 + i) * 8;
        // Initially point back to PLT[0] for lazy resolution
        let plt0 = plt_vaddr;
        if off + 8 <= got_plt_data.len() {
            got_plt_data[off..off + 8].copy_from_slice(&plt0.to_le_bytes());
        }
    }

    // ── Phase 8: Build PLT ──────────────────────────────────────────────

    let mut plt_data = vec![0u8; plt_size as usize];
    if !plt_symbols.is_empty() {
        // PLT[0] header: resolve stub
        // auipc t2, %pcrel_hi(.got.plt)
        // sub t1, t1, t3          ; calculate offset into .got.plt
        // ld t3, %pcrel_lo(...)t2 ; load resolver address
        // addi t1, t1, -plt_header_size ; adjust offset
        // addi t0, t2, %pcrel_lo(...) ; got.plt address
        // srli t1, t1, 1          ; byte offset -> entry index (each PLT entry = 16 bytes, /2 = index*8)
        // ld t0, 8(t0)            ; load link_map (got.plt[1])
        // jr t3                   ; jump to resolver (got.plt[2])

        let plt0_addr = plt_vaddr;
        let got_plt_rel = got_plt_vaddr as i64 - plt0_addr as i64;
        let hi = ((got_plt_rel + 0x800) >> 12) as u32;
        let lo = (got_plt_rel & 0xFFF) as u32;

        // RISC-V PLT header: GOT.PLT[0]=resolver, GOT.PLT[1]=link_map
        // auipc t2, hi  (t2 = x7)
        let insn0 = 0x00000397 | (hi << 12); // auipc t2
        plt_data[0..4].copy_from_slice(&insn0.to_le_bytes());
        // sub t1, t1, t3  (t1=x6, t3=x28)
        let insn1 = 0x41c30333u32; // sub t1, t1, t3
        plt_data[4..8].copy_from_slice(&insn1.to_le_bytes());
        // ld t3, lo(t2)   (load got.plt[0] = resolver)
        let insn2 = 0x0003be03u32 | (lo << 20); // ld t3, lo(t2)
        plt_data[8..12].copy_from_slice(&insn2.to_le_bytes());
        // addi t1, t1, -(plt_header_size + 12)
        // The +12 accounts for jalr's return address being 12 bytes past PLT entry start
        let neg_hdr = (-((plt_header_size + 12) as i32)) as u32;
        let insn3 = 0x00030313u32 | ((neg_hdr & 0xFFF) << 20); // addi t1, t1, -hdr
        plt_data[12..16].copy_from_slice(&insn3.to_le_bytes());
        // addi t0, t2, lo (t0 = got.plt base)
        let insn4 = 0x00038293u32 | (lo << 20); // addi t0, t2, lo
        plt_data[16..20].copy_from_slice(&insn4.to_le_bytes());
        // srli t1, t1, 1
        let insn5 = 0x00135313u32; // srli t1, t1, 1
        plt_data[20..24].copy_from_slice(&insn5.to_le_bytes());
        // ld t0, 8(t0) -> load link_map from got.plt[1]
        let insn6 = 0x0082b283u32; // ld t0, 8(t0)
        plt_data[24..28].copy_from_slice(&insn6.to_le_bytes());
        // jr t3
        let insn7 = 0x000e0067u32; // jalr x0, 0(t3)
        plt_data[28..32].copy_from_slice(&insn7.to_le_bytes());

        // PLT entries
        for (i, _name) in plt_symbols.iter().enumerate() {
            let entry_off = (plt_header_size + i as u64 * plt_entry_size) as usize;
            let entry_addr = plt_vaddr + entry_off as u64;
            let got_entry_addr = got_plt_vaddr + (2 + i) as u64 * 8;

            let rel = got_entry_addr as i64 - entry_addr as i64;
            let hi = ((rel + 0x800) >> 12) as u32;
            let lo = (rel & 0xFFF) as u32;

            // auipc t3, hi  (t3 = x28)
            let insn0 = 0x00000e17u32 | (hi << 12);
            plt_data[entry_off..entry_off + 4].copy_from_slice(&insn0.to_le_bytes());
            // ld t3, lo(t3)
            let insn1 = 0x000e3e03u32 | (lo << 20);
            plt_data[entry_off + 4..entry_off + 8].copy_from_slice(&insn1.to_le_bytes());
            // jalr t1, t3  (t1 = x6 saves return for PLT0)
            let insn2 = 0x000e0367u32; // jalr t1, 0(t3)
            plt_data[entry_off + 8..entry_off + 12].copy_from_slice(&insn2.to_le_bytes());
            // nop
            let insn3 = 0x00000013u32;
            plt_data[entry_off + 12..entry_off + 16].copy_from_slice(&insn3.to_le_bytes());
        }
    }

    // ── Phase 9: Build .rela.plt ────────────────────────────────────────

    let mut rela_plt_data = Vec::with_capacity(rela_plt_size as usize);
    for (i, _name) in plt_symbols.iter().enumerate() {
        let got_entry_addr = got_plt_vaddr + (2 + i) as u64 * 8;
        // R_RISCV_JUMP_SLOT = 5
        let r_info = (((i + 1) as u64) << 32) | 5; // symbol index i+1, type JUMP_SLOT
        rela_plt_data.extend_from_slice(&got_entry_addr.to_le_bytes());
        rela_plt_data.extend_from_slice(&r_info.to_le_bytes());
        rela_plt_data.extend_from_slice(&0i64.to_le_bytes()); // addend
    }

    // ── Phase 9b: Build .rela.dyn (COPY relocations) ───────────────────

    let mut rela_dyn_data = Vec::with_capacity(rela_dyn_size as usize);
    for (name, _size) in &copy_symbols {
        if let Some(&(addr, _sz)) = copy_sym_addrs.get(name) {
            // Find the dynsym index for this symbol
            let sym_idx = dynsym_names.iter().position(|n| n == name).unwrap_or(0) + 1; // +1 for null entry
            // R_RISCV_COPY = 4
            let r_info = ((sym_idx as u64) << 32) | 4;
            rela_dyn_data.extend_from_slice(&addr.to_le_bytes()); // r_offset = .bss address
            rela_dyn_data.extend_from_slice(&r_info.to_le_bytes());
            rela_dyn_data.extend_from_slice(&0i64.to_le_bytes()); // addend = 0
        }
    }

    // ── Phase 10: Build .dynamic section ────────────────────────────────

    let mut dynamic_data = Vec::new();
    if !is_static {
        let mut add_dyn = |tag: i64, val: u64| {
            dynamic_data.extend_from_slice(&tag.to_le_bytes());
            dynamic_data.extend_from_slice(&val.to_le_bytes());
        };

        for (i, lib) in actual_needed_libs.iter().enumerate() {
            let _ = lib;
            add_dyn(DT_NEEDED, needed_lib_offsets[i] as u64);
        }

        if let Some(&(va, sz)) = init_array_vaddrs.get(".preinit_array") {
            if sz > 0 {
                add_dyn(DT_PREINIT_ARRAY, va);
                add_dyn(DT_PREINIT_ARRAYSZ, sz);
            }
        }
        if let Some(&(va, sz)) = init_array_vaddrs.get(".init_array") {
            if sz > 0 {
                add_dyn(DT_INIT_ARRAY, va);
                add_dyn(DT_INIT_ARRAYSZ, sz);
            }
        }
        if let Some(&(va, sz)) = init_array_vaddrs.get(".fini_array") {
            if sz > 0 {
                add_dyn(DT_FINI_ARRAY, va);
                add_dyn(DT_FINI_ARRAYSZ, sz);
            }
        }

        add_dyn(DT_GNU_HASH, gnu_hash_vaddr);
        add_dyn(DT_STRTAB, dynstr_vaddr);
        add_dyn(DT_SYMTAB, dynsym_vaddr);
        add_dyn(DT_STRSZ, dynstr_data.len() as u64);
        add_dyn(DT_SYMENT, 24);
        add_dyn(DT_DEBUG, 0);
        add_dyn(DT_PLTGOT, got_plt_vaddr);
        add_dyn(DT_PLTRELSZ, rela_plt_size);
        add_dyn(DT_PLTREL, 7); // DT_RELA
        add_dyn(DT_JMPREL, rela_plt_vaddr);
        // DT_RELA covers both .rela.dyn and .rela.plt
        let rela_start = if rela_dyn_size > 0 { rela_dyn_vaddr } else { rela_plt_vaddr };
        let rela_total_size = rela_dyn_size + rela_plt_size;
        add_dyn(DT_RELA, rela_start);
        add_dyn(DT_RELASZ, rela_total_size);
        add_dyn(DT_RELAENT, 24);
        if !verneed_data.is_empty() {
            add_dyn(DT_VERNEED, verneed_vaddr);
            add_dyn(DT_VERNEEDNUM, 1);
            add_dyn(DT_VERSYM, versym_vaddr);
        }
        add_dyn(DT_NULL, 0);

        // Pad dynamic to declared size
        dynamic_data.resize(dynamic_size as usize, 0);
    }

    // ── Phase 11: Build .eh_frame_hdr ───────────────────────────────────

    // Simple .eh_frame_hdr: just a header with no FDE table
    // (sufficient for basic execution, won't support C++ exceptions)
    let eh_frame_vaddr = merged_map.get(".eh_frame")
        .map(|&i| section_vaddrs[i])
        .unwrap_or(0);
    let eh_frame_hdr_data = if eh_frame_vaddr > 0 {
        let mut hdr = Vec::new();
        hdr.push(1); // version
        hdr.push(0x1b); // eh_frame_ptr encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
        hdr.push(0x03); // fde_count encoding: DW_EH_PE_udata4
        hdr.push(0x3b); // table encoding: DW_EH_PE_datarel | DW_EH_PE_sdata4
        // eh_frame_ptr: will be patched with offset
        hdr.extend_from_slice(&0i32.to_le_bytes()); // placeholder
        // fde_count
        hdr.extend_from_slice(&0u32.to_le_bytes());
        hdr
    } else {
        Vec::new()
    };

    // ── Phase 12: Find entry point ──────────────────────────────────────

    let entry_point = if let Some(gs) = global_syms.get("_start") {
        gs.value
    } else if let Some(gs) = global_syms.get("main") {
        gs.value
    } else {
        // Default to start of .text
        merged_map.get(".text").map(|&i| section_vaddrs[i]).unwrap_or(BASE_ADDR)
    };

    // ── Phase 13: Find .riscv.attributes data ───────────────────────────

    let riscv_attr_data = merged_map.get(".riscv.attributes")
        .map(|&i| merged_sections[i].data.clone())
        .unwrap_or_default();
    let riscv_attr_offset = file_offset;
    let riscv_attr_size = riscv_attr_data.len() as u64;

    // ── Phase 14: Write ELF file ────────────────────────────────────────

    let mut elf = Vec::with_capacity(file_offset as usize + riscv_attr_data.len() + 4096);

    // ELF header
    elf.extend_from_slice(&[0x7f, b'E', b'L', b'F']); // magic
    elf.push(2); // ELFCLASS64
    elf.push(1); // ELFDATA2LSB
    elf.push(1); // EV_CURRENT
    elf.push(0); // ELFOSABI_NONE
    elf.extend_from_slice(&[0; 8]); // padding
    elf.extend_from_slice(&ET_EXEC.to_le_bytes()); // e_type
    elf.extend_from_slice(&EM_RISCV.to_le_bytes()); // e_machine
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&entry_point.to_le_bytes()); // e_entry
    elf.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
    // e_shoff: we'll patch this later
    let shoff_pos = elf.len();
    elf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff placeholder
    // ELF flags: RVC + double-float ABI
    elf.extend_from_slice(&0x05u32.to_le_bytes()); // e_flags
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    elf.extend_from_slice(&(num_phdrs as u16).to_le_bytes()); // e_phnum
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    // e_shnum and e_shstrndx: patched later
    let shnum_pos = elf.len();
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shnum placeholder
    let shstrndx_pos = elf.len();
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx placeholder

    // Program headers
    assert_eq!(elf.len(), 64);

    let rx_filesz = rx_segment_end_offset;
    let rx_memsz = rx_segment_end_vaddr - BASE_ADDR;

    if !is_static {
        // PHDR (only for dynamic binaries)
        write_phdr(&mut elf, 6 /* PT_PHDR */, PF_R, 64, BASE_ADDR + 64, BASE_ADDR + 64, phdr_size, phdr_size, 8);

        // INTERP
        write_phdr(&mut elf, PT_INTERP, PF_R, interp_offset, interp_vaddr, interp_vaddr, interp_size, interp_size, 1);
    }

    // RISCV_ATTRIBUTES
    write_phdr(&mut elf, PT_RISCV_ATTRIBUTES, PF_R, riscv_attr_offset, 0, 0, riscv_attr_size, riscv_attr_size, 1);

    // LOAD RX
    write_phdr(&mut elf, PT_LOAD, PF_R | PF_X, 0, BASE_ADDR, BASE_ADDR, rx_filesz, rx_memsz, PAGE_SIZE);

    // LOAD RW
    write_phdr(&mut elf, PT_LOAD, PF_R | PF_W, rw_segment_start_offset, rw_segment_start_vaddr, rw_segment_start_vaddr, rw_segment_filesz, rw_segment_memsz, PAGE_SIZE);

    if !is_static {
        // DYNAMIC
        write_phdr(&mut elf, PT_DYNAMIC, PF_R | PF_W, dynamic_offset, dynamic_vaddr, dynamic_vaddr, dynamic_data.len() as u64, dynamic_data.len() as u64, 8);
    }

    // NOTE (dummy - for .note.ABI-tag if present)
    write_phdr(&mut elf, PT_NOTE, PF_R, 0, 0, 0, 0, 0, 4);

    // GNU_EH_FRAME (dummy for now)
    write_phdr(&mut elf, PT_GNU_EH_FRAME, PF_R, 0, 0, 0, 0, 0, 4);

    // GNU_STACK
    write_phdr(&mut elf, PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 0, 0x10);

    if !is_static {
        // GNU_RELRO - covers .preinit_array, .init_array, .fini_array, .dynamic, .got
        // Must NOT cover .got.plt (needs to stay writable for lazy binding)
        let relro_filesz = relro_end_offset - rw_segment_start_offset;
        let relro_memsz = relro_end_vaddr - rw_segment_start_vaddr;
        write_phdr(&mut elf, PT_GNU_RELRO, PF_R, rw_segment_start_offset, rw_segment_start_vaddr, rw_segment_start_vaddr,
                   relro_filesz, relro_memsz, 1);
    }

    // PT_TLS - thread-local storage segment
    if has_tls {
        write_phdr(&mut elf, PT_TLS, PF_R, tls_offset, tls_vaddr, tls_vaddr,
                   tls_filesz, tls_memsz, tls_align);
    }

    // Now write section data, padding to correct offsets
    if !is_static {
        // RX segment generated sections (dynamic linking infrastructure)
        pad_to(&mut elf, interp_offset as usize);
        elf.extend_from_slice(INTERP);

        pad_to(&mut elf, gnu_hash_offset as usize);
        elf.extend_from_slice(&gnu_hash_data);

        pad_to(&mut elf, dynsym_offset as usize);
        elf.extend_from_slice(&dynsym_data);

        pad_to(&mut elf, dynstr_offset as usize);
        elf.extend_from_slice(&dynstr_data);

        pad_to(&mut elf, versym_offset as usize);
        elf.extend_from_slice(&versym_data);

        pad_to(&mut elf, verneed_offset as usize);
        elf.extend_from_slice(&verneed_data);

        if !rela_dyn_data.is_empty() {
            pad_to(&mut elf, rela_dyn_offset as usize);
            elf.extend_from_slice(&rela_dyn_data);
        }

        pad_to(&mut elf, rela_plt_offset as usize);
        elf.extend_from_slice(&rela_plt_data);

        pad_to(&mut elf, plt_offset as usize);
        elf.extend_from_slice(&plt_data);
    }

    // Write RX merged sections
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.sh_flags & SHF_ALLOC == 0 || ms.sh_flags & SHF_WRITE != 0 {
            continue;
        }
        if ms.name == ".riscv.attributes" || ms.name.starts_with(".note.") {
            continue;
        }
        if ms.data.is_empty() {
            continue;
        }
        pad_to(&mut elf, section_offsets[si] as usize);
        elf.extend_from_slice(&ms.data);
    }

    // RW segment
    pad_to(&mut elf, rw_segment_start_offset as usize);

    // Init/fini arrays
    for sect_name in &init_array_sections {
        if let Some(&si) = merged_map.get(*sect_name) {
            let ms = &merged_sections[si];
            if !ms.data.is_empty() {
                pad_to(&mut elf, section_offsets[si] as usize);
                elf.extend_from_slice(&ms.data);
            }
        }
    }

    // .dynamic (only for dynamic binaries)
    if !is_static {
        pad_to(&mut elf, dynamic_offset as usize);
        elf.extend_from_slice(&dynamic_data);
    }

    // .got (needed even in static for GOT-relative relocations)
    if got_size > 0 {
        pad_to(&mut elf, got_offset as usize);
        elf.extend_from_slice(&got_data);
    }

    // .got.plt (only for dynamic binaries)
    if !is_static {
        pad_to(&mut elf, got_plt_offset as usize);
        elf.extend_from_slice(&got_plt_data);
    }

    // RW user sections (.data, .sdata)
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.sh_flags & SHF_ALLOC == 0 || ms.sh_flags & SHF_WRITE == 0 {
            continue;
        }
        if init_array_sections.contains(&ms.name.as_str()) {
            continue;
        }
        if ms.sh_type == SHT_NOBITS || ms.name == ".bss" || ms.name == ".sbss" {
            continue;
        }
        if ms.data.is_empty() {
            continue;
        }
        pad_to(&mut elf, section_offsets[si] as usize);
        elf.extend_from_slice(&ms.data);
    }

    // .riscv.attributes (non-loadable, after all LOAD segments)
    pad_to(&mut elf, riscv_attr_offset as usize);
    elf.extend_from_slice(&riscv_attr_data);

    // Section headers (optional but useful for debugging)
    // For now, write minimal section headers
    let shdr_offset = align_up(elf.len() as u64, 8);
    pad_to(&mut elf, shdr_offset as usize);

    // Patch e_shoff in ELF header
    elf[shoff_pos..shoff_pos + 8].copy_from_slice(&shdr_offset.to_le_bytes());

    // Build section header string table
    let mut shstrtab = vec![0u8]; // null byte
    let mut shstr_offsets: HashMap<String, u32> = HashMap::new();
    let sh_names = [
        "", ".interp", ".gnu.hash", ".dynsym", ".dynstr",
        ".gnu.version", ".gnu.version_r", ".rela.dyn", ".rela.plt", ".plt",
        ".text", ".rodata", ".eh_frame",
        ".preinit_array", ".init_array", ".fini_array",
        ".dynamic", ".got", ".got.plt", ".data", ".sdata", ".bss",
        ".riscv.attributes", ".symtab", ".strtab", ".shstrtab",
    ];
    for name in &sh_names {
        if !name.is_empty() {
            let off = shstrtab.len() as u32;
            shstr_offsets.insert(name.to_string(), off);
            shstrtab.extend_from_slice(name.as_bytes());
            shstrtab.push(0);
        }
    }

    // Add any merged section names not in the standard list
    for ms in &merged_sections {
        if !ms.name.is_empty() && !shstr_offsets.contains_key(&ms.name) {
            let off = shstrtab.len() as u32;
            shstr_offsets.insert(ms.name.clone(), off);
            shstrtab.extend_from_slice(ms.name.as_bytes());
            shstrtab.push(0);
        }
    }

    // Write section headers
    // We'll write a minimal set: NULL, then key sections
    let mut section_count: u32 = 0;

    // NULL section header
    elf.extend_from_slice(&[0u8; 64]);
    section_count += 1;

    // Helper to write a section header
    fn write_shdr(elf: &mut Vec<u8>, name: u32, sh_type: u32, flags: u64,
                  addr: u64, offset: u64, size: u64, link: u32, info: u32,
                  align: u64, entsize: u64) {
        elf.extend_from_slice(&name.to_le_bytes());
        elf.extend_from_slice(&sh_type.to_le_bytes());
        elf.extend_from_slice(&flags.to_le_bytes());
        elf.extend_from_slice(&addr.to_le_bytes());
        elf.extend_from_slice(&offset.to_le_bytes());
        elf.extend_from_slice(&size.to_le_bytes());
        elf.extend_from_slice(&link.to_le_bytes());
        elf.extend_from_slice(&info.to_le_bytes());
        elf.extend_from_slice(&align.to_le_bytes());
        elf.extend_from_slice(&entsize.to_le_bytes());
    }

    let get_name = |n: &str| -> u32 { shstr_offsets.get(n).copied().unwrap_or(0) };

    let mut dynsym_shidx = 0u32;
    if !is_static {
        // .interp
        write_shdr(&mut elf, get_name(".interp"), SHT_PROGBITS, SHF_ALLOC,
                   interp_vaddr, interp_offset, interp_size, 0, 0, 1, 0);
        section_count += 1;

        // .gnu.hash
        write_shdr(&mut elf, get_name(".gnu.hash"), 0x6ffffff6, SHF_ALLOC,
                   gnu_hash_vaddr, gnu_hash_offset, gnu_hash_data.len() as u64,
                   section_count + 1, 0, 8, 0); // link to .dynsym
        section_count += 1;
        dynsym_shidx = section_count;

        // .dynsym
        write_shdr(&mut elf, get_name(".dynsym"), 11 /* SHT_DYNSYM */, SHF_ALLOC,
                   dynsym_vaddr, dynsym_offset, dynsym_data.len() as u64,
                   section_count + 1, 1, 8, 24); // link to .dynstr, info=1 (first global)
        section_count += 1;

        // .dynstr
        write_shdr(&mut elf, get_name(".dynstr"), SHT_STRTAB, SHF_ALLOC,
                   dynstr_vaddr, dynstr_offset, dynstr_data.len() as u64, 0, 0, 1, 0);
        section_count += 1;

        // .gnu.version
        write_shdr(&mut elf, get_name(".gnu.version"), 0x6fffffff /* SHT_GNU_versym */, SHF_ALLOC,
                   versym_vaddr, versym_offset, versym_data.len() as u64,
                   dynsym_shidx, 0, 2, 2);
        section_count += 1;

        // .gnu.version_r
        write_shdr(&mut elf, get_name(".gnu.version_r"), 0x6ffffffe /* SHT_GNU_verneed */, SHF_ALLOC,
                   verneed_vaddr, verneed_offset, verneed_data.len() as u64,
                   section_count - 2, 1, 8, 0); // link to .dynstr
        section_count += 1;

        // .rela.dyn (COPY relocations)
        if rela_dyn_size > 0 {
            write_shdr(&mut elf, get_name(".rela.dyn"), SHT_RELA, SHF_ALLOC,
                       rela_dyn_vaddr, rela_dyn_offset, rela_dyn_size,
                       dynsym_shidx, 0, 8, 24);
            section_count += 1;
        }

        // .rela.plt
        let _rela_plt_shidx = section_count;
        write_shdr(&mut elf, get_name(".rela.plt"), SHT_RELA, SHF_ALLOC | 0x40 /* SHF_INFO_LINK */,
                   rela_plt_vaddr, rela_plt_offset, rela_plt_size,
                   dynsym_shidx, section_count + 1, 8, 24);
        section_count += 1;

        // .plt
        write_shdr(&mut elf, get_name(".plt"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                   plt_vaddr, plt_offset, plt_size, 0, 0, 16, 16);
        section_count += 1;
    }

    // Merged sections
    for &si in &sec_indices {
        let ms = &merged_sections[si];
        if ms.sh_flags & SHF_ALLOC == 0 && ms.sh_type != SHT_RISCV_ATTRIBUTES {
            continue;
        }
        if ms.name.starts_with(".note.") {
            continue;
        }

        let sh_type = ms.sh_type;
        let size = ms.data.len() as u64;
        let offset = section_offsets[si];
        let va = section_vaddrs[si];

        write_shdr(&mut elf, get_name(&ms.name), sh_type, ms.sh_flags,
                   va, offset, size, 0, 0, ms.align, 0);
        section_count += 1;
    }

    // Generated RW sections
    if !is_static {
        // .dynamic
        write_shdr(&mut elf, get_name(".dynamic"), 6 /* SHT_DYNAMIC */, SHF_ALLOC | SHF_WRITE,
                   dynamic_vaddr, dynamic_offset, dynamic_data.len() as u64,
                   4, 0, 8, 16); // link to dynstr section (index 4)
        section_count += 1;
    }

    // .got
    if got_size > 0 {
        write_shdr(&mut elf, get_name(".got"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                   got_vaddr, got_offset, got_size, 0, 0, 8, 8);
        section_count += 1;
    }

    if !is_static {
        // .got.plt
        write_shdr(&mut elf, get_name(".got.plt"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                   got_plt_vaddr, got_plt_offset, got_plt_size, 0, 0, 8, 8);
        section_count += 1;
    }

    // .shstrtab
    let shstrtab_offset = elf.len() as u64;
    let shstrtab_shidx = section_count;
    write_shdr(&mut elf, get_name(".shstrtab"), SHT_STRTAB, 0,
               0, shstrtab_offset + 64, shstrtab.len() as u64, 0, 0, 1, 0);
    section_count += 1;

    // Append shstrtab data (its offset was set to after the last shdr)
    // Actually we need to recalculate because we're still writing shdrs
    // The shstrtab goes after all section headers

    // Patch e_shnum and e_shstrndx
    elf[shnum_pos..shnum_pos + 2].copy_from_slice(&(section_count as u16).to_le_bytes());
    elf[shstrndx_pos..shstrndx_pos + 2].copy_from_slice(&(shstrtab_shidx as u16).to_le_bytes());

    // Fix up the shstrtab offset (it should be at current position)
    let actual_shstrtab_off = elf.len() as u64;
    // Patch the shstrtab section header's sh_offset
    let shstrtab_shdr_off = shdr_offset as usize + (shstrtab_shidx as usize) * 64;
    if shstrtab_shdr_off + 64 <= elf.len() {
        elf[shstrtab_shdr_off + 24..shstrtab_shdr_off + 32]
            .copy_from_slice(&actual_shstrtab_off.to_le_bytes());
    }

    // Write shstrtab data
    elf.extend_from_slice(&shstrtab);

    // Write the file
    std::fs::write(output_path, &elf)
        .map_err(|e| format!("Failed to write output: {}", e))?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(output_path, std::fs::Permissions::from_mode(0o755));
    }

    Ok(())
}

// ── Helper functions ────────────────────────────────────────────────────

fn align_up(val: u64, align: u64) -> u64 {
    if align <= 1 {
        val
    } else {
        (val + align - 1) & !(align - 1)
    }
}

fn pad_to(buf: &mut Vec<u8>, target: usize) {
    if buf.len() < target {
        buf.resize(target, 0);
    }
}

fn write_phdr(elf: &mut Vec<u8>, p_type: u32, p_flags: u32,
              offset: u64, vaddr: u64, paddr: u64,
              filesz: u64, memsz: u64, align: u64) {
    elf.extend_from_slice(&p_type.to_le_bytes());
    elf.extend_from_slice(&p_flags.to_le_bytes());
    elf.extend_from_slice(&offset.to_le_bytes());
    elf.extend_from_slice(&vaddr.to_le_bytes());
    elf.extend_from_slice(&paddr.to_le_bytes());
    elf.extend_from_slice(&filesz.to_le_bytes());
    elf.extend_from_slice(&memsz.to_le_bytes());
    elf.extend_from_slice(&align.to_le_bytes());
}

/// Patch a U-type instruction (LUI/AUIPC) with a 20-bit immediate.
fn patch_u_type(data: &mut [u8], off: usize, value: u32) {
    if off + 4 > data.len() {
        return;
    }
    let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let hi = (value.wrapping_add(0x800)) & 0xFFFFF000;
    let insn = (insn & 0xFFF) | hi;
    data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
}

/// Patch an I-type instruction with a 12-bit immediate.
fn patch_i_type(data: &mut [u8], off: usize, value: u32) {
    if off + 4 > data.len() {
        return;
    }
    let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let imm = value & 0xFFF;
    let insn = (insn & 0x000FFFFF) | (imm << 20);
    data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
}

/// Patch an S-type instruction with a 12-bit immediate.
fn patch_s_type(data: &mut [u8], off: usize, value: u32) {
    if off + 4 > data.len() {
        return;
    }
    let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let imm = value & 0xFFF;
    let imm_hi = (imm >> 5) & 0x7F;
    let imm_lo = imm & 0x1F;
    let insn = (insn & 0x01FFF07F) | (imm_hi << 25) | (imm_lo << 7);
    data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
}

/// Patch a B-type instruction with a 13-bit PC-relative offset.
fn patch_b_type(data: &mut [u8], off: usize, value: u32) {
    if off + 4 > data.len() {
        return;
    }
    let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let imm = value;
    let bit12 = (imm >> 12) & 1;
    let bits10_5 = (imm >> 5) & 0x3F;
    let bits4_1 = (imm >> 1) & 0xF;
    let bit11 = (imm >> 11) & 1;
    let insn = (insn & 0x01FFF07F)
        | (bit12 << 31)
        | (bits10_5 << 25)
        | (bits4_1 << 8)
        | (bit11 << 7);
    data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
}

/// Patch a J-type instruction with a 21-bit PC-relative offset.
fn patch_j_type(data: &mut [u8], off: usize, value: u32) {
    if off + 4 > data.len() {
        return;
    }
    let insn = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let imm = value;
    let bit20 = (imm >> 20) & 1;
    let bits10_1 = (imm >> 1) & 0x3FF;
    let bit11 = (imm >> 11) & 1;
    let bits19_12 = (imm >> 12) & 0xFF;
    let insn = (insn & 0xFFF)
        | (bit20 << 31)
        | (bits10_1 << 21)
        | (bit11 << 20)
        | (bits19_12 << 12);
    data[off..off + 4].copy_from_slice(&insn.to_le_bytes());
}

/// Find the hi20 value for a pcrel_lo12 relocation.
/// The pcrel_lo12 references an auipc instruction; we need to find the hi20
/// relocation at that auipc and compute the full offset, then return the low 12 bits.
fn find_hi20_value(
    obj: &ObjFile,
    obj_idx: usize,
    sec_idx: usize,
    sec_mapping: &HashMap<(usize, usize), (usize, u64)>,
    section_vaddrs: &[u64],
    local_sym_vaddrs: &[Vec<u64>],
    global_syms: &HashMap<String, GlobalSym>,
    auipc_vaddr: u64,
    sec_offset: u64,
    got_vaddr: u64,
    got_symbols: &[String],
    got_plt_vaddr: u64,
) -> i64 {
    // Find the hi20 relocation targeting the same address
    if let Some(relocs) = obj.relocs.get(&sec_idx) {
        for reloc in relocs {
            let reloc_vaddr = sec_offset + reloc.offset;
            let (mi, _mo) = match sec_mapping.get(&(obj_idx, sec_idx)) {
                Some(&v) => v,
                None => continue,
            };
            let this_vaddr = section_vaddrs[mi] + reloc_vaddr;

            if this_vaddr != auipc_vaddr {
                continue;
            }

            match reloc.reloc_type {
                R_RISCV_PCREL_HI20 => {
                    let hi_sym_idx = reloc.symbol_idx as usize;
                    let sym = &obj.symbols[hi_sym_idx];
                    let s = resolve_symbol_value(sym, hi_sym_idx, obj, obj_idx, sec_mapping, section_vaddrs, local_sym_vaddrs, global_syms);
                    let target = s as i64 + reloc.addend;
                    let offset = target - auipc_vaddr as i64;
                    return offset & 0xFFF;
                }
                R_RISCV_GOT_HI20 | R_RISCV_TLS_GOT_HI20 | R_RISCV_TLS_GD_HI20 => {
                    // For GOT references, the target is the GOT entry address, not the symbol address
                    let hi_sym_idx = reloc.symbol_idx as usize;
                    let sym = &obj.symbols[hi_sym_idx];
                    let sym_name = if sym.sym_type == STT_SECTION {
                        obj.sections[sym.section_idx as usize].name.clone()
                    } else {
                        sym.name.clone()
                    };

                    let got_entry_vaddr = if let Some(gs) = global_syms.get(&sym_name) {
                        if let Some(got_off) = gs.got_offset {
                            got_vaddr + got_off
                        } else {
                            // PLT symbol: use GOT.PLT
                            let plt_idx = gs.plt_idx;
                            got_plt_vaddr + (2 + plt_idx) as u64 * 8
                        }
                    } else {
                        // Find in got_symbols list
                        if let Some(idx) = got_symbols.iter().position(|n| n == &sym_name) {
                            got_vaddr + idx as u64 * 8
                        } else {
                            0
                        }
                    };

                    let offset = got_entry_vaddr as i64 + reloc.addend - auipc_vaddr as i64;
                    return offset & 0xFFF;
                }
                _ => {}
            }
        }
    }
    0
}

fn resolve_symbol_value(
    sym: &ObjSymbol,
    sym_idx: usize,
    obj: &ObjFile,
    obj_idx: usize,
    sec_mapping: &HashMap<(usize, usize), (usize, u64)>,
    section_vaddrs: &[u64],
    local_sym_vaddrs: &[Vec<u64>],
    global_syms: &HashMap<String, GlobalSym>,
) -> u64 {
    if sym.sym_type == STT_SECTION {
        if (sym.section_idx as usize) < obj.sections.len() {
            if let Some(&(mi, mo)) = sec_mapping.get(&(obj_idx, sym.section_idx as usize)) {
                return section_vaddrs[mi] + mo;
            }
        }
        0
    } else if !sym.name.is_empty() && sym.binding != STB_LOCAL {
        global_syms.get(&sym.name).map(|gs| gs.value).unwrap_or(0)
    } else {
        // Local symbol - index by symbol index, not section index
        local_sym_vaddrs.get(obj_idx)
            .and_then(|v| v.get(sym_idx))
            .copied()
            .unwrap_or(0)
    }
}

/// Build a minimal .gnu.hash section.
fn build_gnu_hash(nsyms: usize) -> Vec<u8> {
    let mut data = Vec::new();
    // nbuckets
    let nbuckets = if nsyms == 0 { 1 } else { nsyms.next_power_of_two() } as u32;
    data.extend_from_slice(&nbuckets.to_le_bytes());
    // symoffset (first symbol index in .dynsym covered by hash)
    let symoffset = 1u32; // skip null symbol
    data.extend_from_slice(&symoffset.to_le_bytes());
    // bloom_size
    let bloom_size = 1u32;
    data.extend_from_slice(&bloom_size.to_le_bytes());
    // bloom_shift
    data.extend_from_slice(&6u32.to_le_bytes());
    // bloom filter (single zero entry = accept all)
    data.extend_from_slice(&0u64.to_le_bytes());
    // buckets (one per bucket, value = first symbol index in that bucket, or 0)
    for i in 0..nbuckets {
        if (i as usize) < nsyms {
            data.extend_from_slice(&(i + 1).to_le_bytes()); // symbol index
        } else {
            data.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    // hash values (one per symbol, with chain terminator bit set)
    for _i in 0..nsyms {
        let hash = 1u32 | 1; // Set bit 0 = end of chain
        data.extend_from_slice(&hash.to_le_bytes());
    }
    data
}

/// ELF GNU hash function.
fn elf_hash_gnu(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &b in name {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

/// Find the GCC cross-compiler library directory.
fn find_gcc_dir() -> Option<String> {
    let bases = [
        "/usr/lib/gcc-cross/riscv64-linux-gnu",
        "/usr/lib/gcc/riscv64-linux-gnu",
    ];
    let versions = ["14", "13", "12", "11", "10", "9", "8"];
    for base in &bases {
        for ver in &versions {
            let dir = format!("{}/{}", base, ver);
            if std::path::Path::new(&format!("{}/crtbegin.o", dir)).exists() {
                return Some(dir);
            }
        }
    }
    None
}

/// Find the CRT directory for RISC-V.
fn find_crt_dir_rv() -> Option<String> {
    let candidates = [
        "/usr/riscv64-linux-gnu/lib",
        "/usr/lib/riscv64-linux-gnu",
        "/lib/riscv64-linux-gnu",
    ];
    for dir in &candidates {
        if std::path::Path::new(&format!("{}/crt1.o", dir)).exists() {
            return Some(dir.to_string());
        }
    }
    None
}

/// Resolve archive members: iteratively pull in members that define currently-undefined symbols.
fn resolve_archive_members(
    members: Vec<(String, ObjFile)>,
    input_objs: &mut Vec<(String, ObjFile)>,
    defined_syms: &mut HashSet<String>,
    undefined_syms: &mut HashSet<String>,
) {
    let mut pool = members;
    loop {
        let mut added_any = false;
        let mut remaining = Vec::new();
        for (name, obj) in pool {
            let needed = obj.symbols.iter().any(|sym| {
                sym.section_idx != SHN_UNDEF
                    && sym.binding != STB_LOCAL
                    && !sym.name.is_empty()
                    && undefined_syms.contains(&sym.name)
            });
            if needed {
                for sym in &obj.symbols {
                    if sym.section_idx != SHN_UNDEF && sym.binding != STB_LOCAL && !sym.name.is_empty() {
                        defined_syms.insert(sym.name.clone());
                        undefined_syms.remove(&sym.name);
                    }
                }
                for sym in &obj.symbols {
                    if sym.section_idx == SHN_UNDEF && !sym.name.is_empty() && sym.binding != STB_LOCAL {
                        if !defined_syms.contains(&sym.name) {
                            undefined_syms.insert(sym.name.clone());
                        }
                    }
                }
                input_objs.push((name, obj));
                added_any = true;
            } else {
                remaining.push((name, obj));
            }
        }
        if !added_any || remaining.is_empty() {
            break;
        }
        pool = remaining;
    }
}

/// Find a versioned soname for a library (e.g., "c" -> "libc.so.6").
/// Returns None if only an unversioned .so is found (linker script or dev symlink).
fn find_versioned_soname(dir: &str, libname: &str) -> Option<String> {
    let pattern = format!("lib{}.so.", libname);
    let mut best: Option<String> = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().into_owned();
            if name_str.starts_with(&pattern) {
                // Prefer shorter versioned names (e.g., libc.so.6 over libc.so.6.0.0)
                if best.is_none() || name_str.len() < best.as_ref().unwrap().len() {
                    best = Some(name_str);
                }
            }
        }
    }
    best
}
