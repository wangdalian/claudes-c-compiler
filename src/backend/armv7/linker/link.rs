//! ARMv7 linker orchestration.
//!
//! Contains the two public entry points (`link_builtin` and `link_shared`) that
//! orchestrate the linking pipeline: parse arguments, load inputs, merge sections,
//! resolve symbols, build PLT/GOT, and emit the ELF32 executable or shared library.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::types::*;
use super::parse::*;
use super::sections::merge_sections;
use super::symbols::*;
use super::emit::emit_executable;
use super::shared::{resolve_dynamic_symbols_for_shared, emit_shared_library_32};

/// Built-in linker entry point with pre-resolved CRT objects and library paths.
pub fn link_builtin(
    object_files: &[&str],
    output_path: &str,
    user_args: &[String],
    lib_paths: &[&str],
    needed_libs_param: &[&str],
    crt_objects_before: &[&str],
    crt_objects_after: &[&str],
) -> Result<(), String> {
    let is_nostdlib = user_args.iter().any(|a| a == "-nostdlib");
    // Force static linking: the ARMv7 linker does not currently support
    // shared library symbol resolution (dynlib_syms is always empty).
    // Without this, the linker creates a broken half-dynamic binary that
    // has INTERP/DYNAMIC/DT_NEEDED but all symbols resolved from static
    // archives, causing the dynamic linker to load libc.so alongside the
    // statically-linked libc — conflicting TLS, GOT, and initialization
    // code leads to a segfault during CRT startup.
    let is_static = true;

    // Phase 1: Parse arguments
    let (extra_libs, extra_lib_files, extra_lib_paths, extra_objects, defsym_defs) = parse_user_args(user_args);

    let all_lib_dirs: Vec<String> = extra_lib_paths.into_iter()
        .chain(lib_paths.iter().map(|s| s.to_string()))
        .collect();

    // Phase 2: Collect input files
    let all_objects = collect_input_files(
        object_files, &extra_objects, crt_objects_before, crt_objects_after,
        is_nostdlib, is_static, lib_paths,
    );

    // Phase 3: Load library symbols
    let (dynlib_syms, static_lib_objects) = load_libraries(
        is_static, needed_libs_param, &extra_libs, &extra_lib_files, &all_lib_dirs,
    );

    // Phase 4: Parse objects
    let mut all_objs = all_objects;
    for lib_path in &static_lib_objects {
        all_objs.push(lib_path.clone());
    }
    let (inputs, _archive_pool) = load_and_parse_objects(&all_objs, &defsym_defs)?;

    // Phase 5: Merge sections
    let (mut output_sections, mut section_name_to_idx, section_map) = merge_sections(&inputs);

    // Phase 6: Resolve symbols
    let (mut global_symbols, sym_resolution) = resolve_symbols(
        &inputs, &output_sections, &section_map, &dynlib_syms,
    );

    allocate_common_symbols(&inputs, &mut output_sections, &mut section_name_to_idx, &mut global_symbols);

    // Phase 7: Mark PLT/GOT needs
    mark_plt_got_needs(&inputs, &mut global_symbols, is_static);

    for (alias, target) in &defsym_defs {
        if let Some(target_sym) = global_symbols.get(target).cloned() {
            global_symbols.insert(alias.clone(), target_sym);
        }
    }

    check_undefined_symbols(&global_symbols)?;

    // Phase 8: Build PLT/GOT
    let (plt_symbols, got_dyn_symbols, got_local_symbols, num_plt, num_got_slots) =
        build_plt_got_lists(&mut global_symbols);

    // Mark weak dynamic data symbols for textrel
    if !is_static {
        let weak_data_syms: Vec<String> = global_symbols.iter()
            .filter(|(_, s)| s.is_dynamic && s.needs_copy && s.binding == STB_WEAK
                && s.sym_type != STT_FUNC && s.sym_type != STT_GNU_IFUNC)
            .map(|(n, _)| n.clone())
            .collect();
        for name in &weak_data_syms {
            if let Some(sym) = global_symbols.get_mut(name.as_str()) {
                sym.needs_copy = false;
                sym.uses_textrel = true;
            }
        }
    }

    // Phase 9: Collect IFUNC
    let ifunc_symbols = collect_ifunc_symbols(&global_symbols, is_static);

    // Phase 10: Emit
    emit_executable(
        &inputs, &mut output_sections, &section_name_to_idx, &section_map,
        &mut global_symbols, &sym_resolution,
        &dynlib_syms, &plt_symbols, &got_dyn_symbols, &got_local_symbols,
        num_plt, num_got_slots, &ifunc_symbols,
        is_static, is_nostdlib, needed_libs_param,
        output_path,
    )
}

/// Create a shared library (.so) from ELF32 object files.
pub fn link_shared(
    object_files: &[&str],
    output_path: &str,
    user_args: &[String],
    lib_paths: &[&str],
) -> Result<(), String> {
    let mut extra_lib_paths: Vec<String> = Vec::new();
    let mut libs_to_load: Vec<String> = Vec::new();
    let mut extra_object_files: Vec<String> = Vec::new();
    let mut soname: Option<String> = None;

    let args: Vec<&str> = user_args.iter().map(|s| s.as_str()).collect();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if let Some(path) = arg.strip_prefix("-L") {
            let p = if path.is_empty() && i + 1 < args.len() { i += 1; args[i] } else { path };
            extra_lib_paths.push(p.to_string());
        } else if let Some(lib) = arg.strip_prefix("-l") {
            let l = if lib.is_empty() && i + 1 < args.len() { i += 1; args[i] } else { lib };
            libs_to_load.push(l.to_string());
        } else if let Some(wl_arg) = arg.strip_prefix("-Wl,") {
            let parts: Vec<&str> = wl_arg.split(',').collect();
            let mut j = 0;
            while j < parts.len() {
                let part = parts[j];
                if let Some(sn) = part.strip_prefix("-soname=") {
                    soname = Some(sn.to_string());
                } else if part == "-soname" && j + 1 < parts.len() {
                    j += 1; soname = Some(parts[j].to_string());
                } else if let Some(lpath) = part.strip_prefix("-L") {
                    extra_lib_paths.push(lpath.to_string());
                } else if let Some(lib) = part.strip_prefix("-l") {
                    libs_to_load.push(lib.to_string());
                }
                j += 1;
            }
        } else if arg == "-shared" || arg == "-nostdlib" || arg == "-o" {
            if arg == "-o" { i += 1; }
        } else if !arg.starts_with('-') && Path::new(arg).exists() {
            extra_object_files.push(arg.to_string());
        }
        i += 1;
    }

    let mut all_objs: Vec<String> = object_files.iter().map(|s| s.to_string()).collect();
    all_objs.extend(extra_object_files);

    let defsym_defs: Vec<(String, String)> = Vec::new();
    let (inputs, _archive_pool) = load_and_parse_objects(&all_objs, &defsym_defs)?;

    let (mut output_sections, section_name_to_idx, section_map) = merge_sections(&inputs);

    let dynlib_syms: HashMap<String, (String, u8, u32, Option<String>, bool, u8)> = HashMap::new();
    let (mut global_symbols, _sym_resolution) = resolve_symbols(
        &inputs, &output_sections, &section_map, &dynlib_syms,
    );

    let lib_path_strings: Vec<String> = lib_paths.iter().map(|s| s.to_string()).collect();
    let mut all_lib_paths: Vec<String> = extra_lib_paths;
    all_lib_paths.extend(lib_path_strings);

    let mut needed_sonames: Vec<String> = Vec::new();
    resolve_dynamic_symbols_for_shared(&inputs, &global_symbols, &mut needed_sonames, &all_lib_paths);

    emit_shared_library_32(
        &inputs, &mut global_symbols, &mut output_sections,
        &section_name_to_idx, &section_map,
        &needed_sonames, output_path, soname,
    )
}

// ── Input processing ────────────────────────────────────────────────────────

fn parse_user_args(user_args: &[String]) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<(String, String)>) {
    let mut extra_libs = Vec::new();
    let mut extra_lib_files = Vec::new();
    let mut extra_lib_paths = Vec::new();
    let mut extra_objects = Vec::new();
    let mut defsym_defs: Vec<(String, String)> = Vec::new();

    for arg in user_args {
        if arg == "-nostdlib" || arg == "-shared" || arg == "-static" || arg == "-r" {
            continue;
        } else if let Some(libarg) = arg.strip_prefix("-l") {
            if let Some(rest) = libarg.strip_prefix(':') {
                extra_lib_files.push(rest.to_string());
            } else {
                extra_libs.push(libarg.to_string());
            }
        } else if let Some(rest) = arg.strip_prefix("-L") {
            extra_lib_paths.push(rest.to_string());
        } else if arg == "-rdynamic" || arg == "--export-dynamic" {
            // accepted
        } else if let Some(wl_args) = arg.strip_prefix("-Wl,") {
            let parts: Vec<&str> = wl_args.split(',').collect();
            let mut j = 0;
            while j < parts.len() {
                let part = parts[j];
                if let Some(libarg) = part.strip_prefix("-l") {
                    if let Some(rest) = libarg.strip_prefix(':') {
                        extra_lib_files.push(rest.to_string());
                    } else {
                        extra_libs.push(libarg.to_string());
                    }
                } else if let Some(rest) = part.strip_prefix("-L") {
                    extra_lib_paths.push(rest.to_string());
                } else if let Some(defsym_arg) = part.strip_prefix("--defsym=") {
                    if let Some(eq_pos) = defsym_arg.find('=') {
                        defsym_defs.push((defsym_arg[..eq_pos].to_string(), defsym_arg[eq_pos + 1..].to_string()));
                    }
                } else if part == "--defsym" && j + 1 < parts.len() {
                    j += 1;
                    let defsym_arg = parts[j];
                    if let Some(eq_pos) = defsym_arg.find('=') {
                        defsym_defs.push((defsym_arg[..eq_pos].to_string(), defsym_arg[eq_pos + 1..].to_string()));
                    }
                }
                j += 1;
            }
        } else if !arg.starts_with('-') && Path::new(arg.as_str()).exists() {
            extra_objects.push(arg.clone());
        }
    }

    (extra_libs, extra_lib_files, extra_lib_paths, extra_objects, defsym_defs)
}

fn collect_input_files(
    object_files: &[&str],
    extra_objects: &[String],
    crt_before: &[&str],
    crt_after: &[&str],
    is_nostdlib: bool,
    is_static: bool,
    lib_paths: &[&str],
) -> Vec<String> {
    let mut all_objects = Vec::new();
    for path in crt_before {
        if Path::new(path).exists() { all_objects.push(path.to_string()); }
    }
    for obj in object_files { all_objects.push(obj.to_string()); }
    for obj in extra_objects { all_objects.push(obj.clone()); }
    for path in crt_after {
        if Path::new(path).exists() { all_objects.push(path.to_string()); }
    }
    if !is_nostdlib && !is_static {
        for dir in lib_paths {
            let path = format!("{}/libc_nonshared.a", dir);
            if Path::new(&path).exists() { all_objects.push(path); break; }
        }
    }
    all_objects
}

fn load_libraries(
    is_static: bool,
    needed_libs: &[&str],
    extra_libs: &[String],
    extra_lib_files: &[String],
    all_lib_dirs: &[String],
) -> (HashMap<String, (String, u8, u32, Option<String>, bool, u8)>, Vec<String>) {
    let dynlib_syms: HashMap<String, (String, u8, u32, Option<String>, bool, u8)> = HashMap::new();
    let mut static_lib_objects: Vec<String> = Vec::new();

    // For static linking, find archives
    let libs_to_find: Vec<String> = needed_libs.iter().map(|s| s.to_string())
        .chain(extra_libs.iter().cloned())
        .collect();

    for lib in &libs_to_find {
        let ar_filename = format!("lib{}.a", lib);
        for dir in all_lib_dirs {
            let path = format!("{}/{}", dir, ar_filename);
            if Path::new(&path).exists() {
                static_lib_objects.push(path);
                break;
            }
        }
    }

    for filename in extra_lib_files {
        for dir in all_lib_dirs {
            let path = format!("{}/{}", dir, filename);
            if Path::new(&path).exists() {
                static_lib_objects.push(path);
                break;
            }
        }
    }

    (dynlib_syms, static_lib_objects)
}

fn load_and_parse_objects(all_objects: &[String], defsym_defs: &[(String, String)]) -> Result<(Vec<InputObject>, Vec<InputObject>), String> {
    let mut inputs: Vec<InputObject> = Vec::new();
    let mut archive_pool: Vec<InputObject> = Vec::new();

    for obj_path in all_objects {
        let data = std::fs::read(obj_path)
            .map_err(|e| format!("cannot read {}: {}", obj_path, e))?;
        if data.len() >= 8 && &data[0..8] == b"!<arch>\n" {
            let members = parse_archive(&data, obj_path)?;
            for (name, mdata) in members {
                let member_name = format!("{}({})", obj_path, name);
                if let Ok(obj) = parse_elf32(&mdata, &member_name) {
                    archive_pool.push(obj);
                }
            }
        } else if is_thin_archive(&data) {
            let members = parse_thin_archive_arm(&data, obj_path)?;
            for (name, mdata) in members {
                let member_name = format!("{}({})", obj_path, name);
                if let Ok(obj) = parse_elf32(&mdata, &member_name) {
                    archive_pool.push(obj);
                }
            }
        } else if data.len() >= 4 && &data[0..4] == ELF_MAGIC {
            inputs.push(parse_elf32(&data, obj_path)?);
        } else {
            // May be a linker script
            if let Ok(text) = std::str::from_utf8(&data) {
                if let Some(entries) = parse_linker_script_entries(text) {
                    for entry in entries {
                        match entry {
                            LinkerScriptEntry::Lib(lib_path) => {
                                if Path::new(&lib_path).exists() {
                                    let lib_data = std::fs::read(&lib_path)
                                        .map_err(|e| format!("cannot read {}: {}", lib_path, e))?;
                                    if lib_data.len() >= 4 && &lib_data[0..4] == ELF_MAGIC {
                                        if let Ok(obj) = parse_elf32(&lib_data, &lib_path) {
                                            inputs.push(obj);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    resolve_archive_members(&mut inputs, &mut archive_pool, defsym_defs);

    Ok((inputs, archive_pool))
}

fn resolve_archive_members(inputs: &mut Vec<InputObject>, archive_pool: &mut Vec<InputObject>, defsym_defs: &[(String, String)]) {
    let mut defined: HashSet<String> = HashSet::new();
    let mut undefined: HashSet<String> = HashSet::new();

    for obj in inputs.iter() {
        for sym in &obj.symbols {
            if sym.name.is_empty() || sym.sym_type == STT_FILE || sym.sym_type == STT_SECTION { continue; }
            if sym.section_index != SHN_UNDEF {
                defined.insert(sym.name.clone());
            } else {
                undefined.insert(sym.name.clone());
            }
        }
    }
    undefined.retain(|s| !defined.contains(s));

    for (alias, target) in defsym_defs {
        if undefined.contains(alias) && !defined.contains(target) {
            undefined.insert(target.clone());
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < archive_pool.len() {
            let resolves = archive_pool[i].symbols.iter().any(|sym| {
                !sym.name.is_empty() && sym.sym_type != STT_FILE && sym.sym_type != STT_SECTION
                    && sym.section_index != SHN_UNDEF && undefined.contains(&sym.name)
            });
            if resolves {
                let obj = archive_pool.remove(i);
                for sym in &obj.symbols {
                    if sym.name.is_empty() || sym.sym_type == STT_FILE || sym.sym_type == STT_SECTION { continue; }
                    if sym.section_index != SHN_UNDEF {
                        defined.insert(sym.name.clone());
                        undefined.remove(&sym.name);
                    } else if !defined.contains(&sym.name) {
                        undefined.insert(sym.name.clone());
                    }
                }
                inputs.push(obj);
                changed = true;
            } else {
                i += 1;
            }
        }
    }
}

fn parse_archive(data: &[u8], archive_path: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    let members = parse_archive_members(data)?;
    Ok(members.into_iter()
        .map(|(name, offset, size)| {
            let member_data = data[offset..offset+size].to_vec();
            (name, member_data)
        })
        .collect())
}

fn parse_thin_archive_arm(data: &[u8], archive_path: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    let base_dir = Path::new(archive_path).parent().unwrap_or(Path::new("."));
    let members = parse_thin_archive_members(data)?;
    let mut result = Vec::new();
    for name in members {
        let full_path = base_dir.join(&name);
        if let Ok(mdata) = std::fs::read(&full_path) {
            result.push((name, mdata));
        }
    }
    Ok(result)
}
