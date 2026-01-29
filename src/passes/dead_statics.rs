//! Dead static function and global elimination.
//!
//! After optimization passes eliminate dead code paths, some static inline
//! functions and static const globals from headers may no longer be referenced.
//! Keeping them wastes code size and may cause linker errors if they reference
//! symbols that don't exist in this translation unit.
//!
//! Uses BFS reachability analysis from roots (non-static symbols) to find all
//! live symbols, then removes unreachable static functions and globals.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::ir::{GlobalInit, Instruction, IrModule};

/// Remove internal-linkage (static) functions and globals that are unreachable.
///
/// Uses reachability analysis from roots (non-static symbols) to find all live symbols,
/// then removes unreachable static functions and globals.
pub(crate) fn eliminate_dead_static_functions(module: &mut IrModule) {
    // Index-based reachability analysis. Instead of cloning symbol name strings
    // for hash map keys and worklist entries, we assign each unique symbol name
    // a compact integer ID and do all reachability work with u32 indices.
    // This dramatically reduces heap allocations for large translation units.

    // Phase 1: Build name-to-index mapping for all symbols.
    // Assign unique IDs to all function and global names. A symbol name can
    // refer to both a function and a global (C allows this), so we track
    // function and global indices separately per symbol ID.
    let mut name_to_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut next_id: u32 = 0;

    // Per-ID mappings: which function/global index (if any) this symbol ID maps to.
    // Uses Option<usize> since a name might be only a function, only a global, or both.
    let mut id_func_idx: Vec<Option<usize>> = Vec::new();
    let mut id_global_idx: Vec<Option<usize>> = Vec::new();

    // Map function names to IDs
    let mut func_id: Vec<u32> = Vec::with_capacity(module.functions.len());
    for (i, func) in module.functions.iter().enumerate() {
        let id = *name_to_id.entry(func.name.as_str()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id_func_idx.push(None);
            id_global_idx.push(None);
            id
        });
        id_func_idx[id as usize] = Some(i);
        func_id.push(id);
    }

    // Map global names to IDs
    let mut global_id: Vec<u32> = Vec::with_capacity(module.globals.len());
    for (i, global) in module.globals.iter().enumerate() {
        let id = *name_to_id.entry(global.name.as_str()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id_func_idx.push(None);
            id_global_idx.push(None);
            id
        });
        id_global_idx[id as usize] = Some(i);
        global_id.push(id);
    }

    // Helper: look up or create an ID for a name that may not already exist.
    // Used for references that might point to external/undeclared symbols.
    let get_or_create_id = |name: &str, name_to_id: &mut FxHashMap<&str, u32>,
                                  next_id: &mut u32| -> u32 {
        if let Some(&id) = name_to_id.get(name) {
            id
        } else {
            // External symbol not in our function/global lists; give it an ID
            // but it won't have references of its own.
            let id = *next_id;
            *next_id += 1;
            id
        }
    };

    // Phase 2: Build reference lists per function (using symbol IDs).
    let mut func_refs: Vec<Vec<u32>> = Vec::with_capacity(module.functions.len());
    for func in &module.functions {
        if func.is_declaration {
            func_refs.push(Vec::new());
            continue;
        }
        let mut refs = Vec::new();
        for block in &func.blocks {
            for inst in &block.instructions {
                collect_instruction_symbol_refs(inst, &mut name_to_id, &mut next_id, &mut refs);
            }
        }
        func_refs.push(refs);
    }

    // Build reference lists per global (from initializers).
    let mut global_refs_lists: Vec<Vec<u32>> = Vec::with_capacity(module.globals.len());
    for global in &module.globals {
        let mut id_refs = Vec::new();
        global.init.for_each_ref(&mut |name| {
            id_refs.push(get_or_create_id(name, &mut name_to_id, &mut next_id));
        });
        global_refs_lists.push(id_refs);
    }

    let total_symbols = next_id as usize;

    // Phase 3: Reachability using bit vectors for O(1) membership test.
    let mut reachable = vec![false; total_symbols];
    let mut worklist: Vec<u32> = Vec::new();

    // Roots: non-static functions
    for (i, func) in module.functions.iter().enumerate() {
        if func.is_declaration { continue; }
        if !func.is_static || func.is_used {
            let id = func_id[i] as usize;
            if !reachable[id] {
                reachable[id] = true;
                worklist.push(id as u32);
            }
        }
    }

    // Roots: non-static globals
    for (i, global) in module.globals.iter().enumerate() {
        if global.is_extern { continue; }
        if !global.is_static || global.is_common || global.is_used {
            let id = global_id[i] as usize;
            if !reachable[id] {
                reachable[id] = true;
                worklist.push(id as u32);
            }
        }
    }

    // Roots: aliases (both the alias name and its target are reachable)
    for (alias_name, target, _) in &module.aliases {
        let tid = get_or_create_id(target, &mut name_to_id, &mut next_id);
        if (tid as usize) >= reachable.len() { reachable.resize(next_id as usize, false); }
        if !reachable[tid as usize] {
            reachable[tid as usize] = true;
            worklist.push(tid);
        }
        let aid = get_or_create_id(alias_name, &mut name_to_id, &mut next_id);
        if (aid as usize) >= reachable.len() { reachable.resize(next_id as usize, false); }
        if !reachable[aid as usize] {
            reachable[aid as usize] = true;
            worklist.push(aid);
        }
    }

    // Roots: constructors and destructors
    for ctor in &module.constructors {
        let id = get_or_create_id(ctor, &mut name_to_id, &mut next_id);
        if (id as usize) >= reachable.len() { reachable.resize(next_id as usize, false); }
        if !reachable[id as usize] {
            reachable[id as usize] = true;
            worklist.push(id);
        }
    }
    for dtor in &module.destructors {
        let id = get_or_create_id(dtor, &mut name_to_id, &mut next_id);
        if (id as usize) >= reachable.len() { reachable.resize(next_id as usize, false); }
        if !reachable[id as usize] {
            reachable[id as usize] = true;
            worklist.push(id);
        }
    }

    // Ensure reachable vec is big enough for all symbols
    if reachable.len() < next_id as usize {
        reachable.resize(next_id as usize, false);
    }

    // Toplevel asm: conservatively mark static symbols whose names appear in asm
    if !module.toplevel_asm.is_empty() {
        for (i, func) in module.functions.iter().enumerate() {
            if func.is_static && !func.is_declaration {
                let fid = func_id[i] as usize;
                if !reachable[fid] {
                    for asm_str in &module.toplevel_asm {
                        if asm_str.contains(func.name.as_str()) {
                            reachable[fid] = true;
                            worklist.push(fid as u32);
                            break;
                        }
                    }
                }
            }
        }
        for (i, global) in module.globals.iter().enumerate() {
            if global.is_static && !global.is_extern {
                let gid = global_id[i] as usize;
                if !reachable[gid] {
                    for asm_str in &module.toplevel_asm {
                        if asm_str.contains(global.name.as_str()) {
                            reachable[gid] = true;
                            worklist.push(gid as u32);
                            break;
                        }
                    }
                }
            }
        }
    }

    // BFS reachability using integer worklist (no String allocation).
    // For each reachable symbol, check both its function refs and global refs
    // since a name can refer to both a function and a global in C.
    while let Some(sym_id) = worklist.pop() {
        let sid = sym_id as usize;
        // Check function references (if this symbol has a function definition)
        if sid < id_func_idx.len() {
            if let Some(fi) = id_func_idx[sid] {
                if fi < func_refs.len() {
                    for &ref_id in &func_refs[fi] {
                        let rid = ref_id as usize;
                        if rid < reachable.len() && !reachable[rid] {
                            reachable[rid] = true;
                            worklist.push(ref_id);
                        }
                    }
                }
            }
        }
        // Check global initializer references (if this symbol has a global definition)
        if sid < id_global_idx.len() {
            if let Some(gi) = id_global_idx[sid] {
                if gi < global_refs_lists.len() {
                    for &ref_id in &global_refs_lists[gi] {
                        let rid = ref_id as usize;
                        if rid < reachable.len() && !reachable[rid] {
                            reachable[rid] = true;
                            worklist.push(ref_id);
                        }
                    }
                }
            }
        }
    }

    // Phase 4: Build address_taken set (still using IDs for efficiency).
    let mut address_taken = vec![false; reachable.len()];

    for func in &module.functions {
        if func.is_declaration { continue; }
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    Instruction::GlobalAddr { name, .. } => {
                        if let Some(&id) = name_to_id.get(name.as_str()) {
                            if (id as usize) < address_taken.len() {
                                address_taken[id as usize] = true;
                            }
                        }
                    }
                    Instruction::InlineAsm { input_symbols, .. } => {
                        for s in input_symbols.iter().flatten() {
                            let base = s.split('+').next().unwrap_or(s);
                            if let Some(&id) = name_to_id.get(base) {
                                if (id as usize) < address_taken.len() {
                                    address_taken[id as usize] = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    for global in &module.globals {
        global.init.for_each_ref(&mut |name| {
            if let Some(&id) = name_to_id.get(name) {
                if (id as usize) < address_taken.len() {
                    address_taken[id as usize] = true;
                }
            }
        });
    }

    // Drop the borrow on module strings so we can mutate module below.
    drop(name_to_id);

    // Phase 5: Remove unreachable symbols using positional index lookups.
    // func_id[i] / global_id[i] map position to symbol ID (no borrows needed).
    let mut func_pos = 0usize;
    module.functions.retain(|func| {
        let pos = func_pos;
        func_pos += 1;
        if func.is_declaration { return true; }
        let id = func_id[pos] as usize;
        if func.is_static && func.is_always_inline {
            return (id < address_taken.len() && address_taken[id])
                || (id < reachable.len() && reachable[id]);
        }
        if !func.is_static { return true; }
        id < reachable.len() && reachable[id]
    });

    let mut global_pos = 0usize;
    module.globals.retain(|global| {
        let pos = global_pos;
        global_pos += 1;
        if global.is_extern { return true; }
        if !global.is_static { return true; }
        if global.is_common { return true; }
        let id = global_id[pos] as usize;
        id < reachable.len() && reachable[id]
    });

    // Phase 6: Filter symbol_attrs.
    // Visibility directives (.hidden, .protected, .internal) for unreferenced
    // symbols cause the assembler to emit undefined symbol entries, which can
    // cause linker errors (e.g., kernel's hidden vdso symbols). Only emit
    // visibility directives for symbols that are actually referenced by
    // surviving code. .weak directives for unreferenced symbols are harmless.
    {
        // Collect all symbol names referenced by surviving functions and globals.
        let mut referenced_symbols: FxHashSet<&str> = FxHashSet::default();
        for func in &module.functions {
            if func.is_declaration { continue; }
            for block in &func.blocks {
                for inst in &block.instructions {
                    match inst {
                        Instruction::Call { func: callee, .. } => {
                            referenced_symbols.insert(callee.as_str());
                        }
                        Instruction::GlobalAddr { name, .. } => {
                            referenced_symbols.insert(name.as_str());
                        }
                        Instruction::InlineAsm { input_symbols, .. } => {
                            for s in input_symbols.iter().flatten() {
                                let base = s.split('+').next().unwrap_or(s);
                                referenced_symbols.insert(base);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        for global in &module.globals {
            collect_global_init_refs_set(&global.init, &mut referenced_symbols);
        }
        // Also include names of surviving non-extern globals and non-declaration functions,
        // since they might be targets of symbol_attrs directives.
        for func in &module.functions {
            referenced_symbols.insert(func.name.as_str());
        }
        for global in &module.globals {
            referenced_symbols.insert(global.name.as_str());
        }

        module.symbol_attrs.retain(|(name, is_weak, visibility)| {
            // Always keep .weak-only directives (no visibility) — they're harmless
            if *is_weak && visibility.is_none() {
                return true;
            }
            // For entries with visibility (.hidden, .protected, .internal),
            // only keep if the symbol is actually referenced
            referenced_symbols.contains(name.as_str())
        });
    }
}

/// Extract symbol references from a single instruction into the ID list.
///
/// Shared by the reference-collection phase to avoid repeating the same
/// `match inst { Call | GlobalAddr | InlineAsm }` pattern.
fn collect_instruction_symbol_refs(
    inst: &Instruction,
    name_to_id: &mut FxHashMap<&str, u32>,
    next_id: &mut u32,
    refs: &mut Vec<u32>,
) {
    let get_or_create = |name: &str, map: &mut FxHashMap<&str, u32>, nid: &mut u32| -> u32 {
        if let Some(&id) = map.get(name) {
            id
        } else {
            let id = *nid;
            *nid += 1;
            id
        }
    };

    match inst {
        Instruction::Call { func: callee, .. } => {
            refs.push(get_or_create(callee, name_to_id, next_id));
        }
        Instruction::GlobalAddr { name, .. } => {
            refs.push(get_or_create(name, name_to_id, next_id));
        }
        Instruction::InlineAsm { input_symbols, .. } => {
            for s in input_symbols.iter().flatten() {
                let base = s.split('+').next().unwrap_or(s);
                refs.push(get_or_create(base, name_to_id, next_id));
            }
        }
        _ => {}
    }
}

/// Collect symbol references from a global initializer into a HashSet of borrowed strings.
///
/// This requires an explicit lifetime annotation since the borrowed `&str` references come
/// from the `GlobalInit`'s owned String fields, which is what `GlobalInit::for_each_ref`
/// cannot express through its closure-based API.
fn collect_global_init_refs_set<'a>(init: &'a GlobalInit, refs: &mut FxHashSet<&'a str>) {
    match init {
        GlobalInit::GlobalAddr(name) | GlobalInit::GlobalAddrOffset(name, _) => {
            refs.insert(name.as_str());
        }
        GlobalInit::GlobalLabelDiff(label1, label2, _) => {
            refs.insert(label1.as_str());
            refs.insert(label2.as_str());
        }
        GlobalInit::Compound(fields) => {
            for field in fields {
                collect_global_init_refs_set(field, refs);
            }
        }
        _ => {}
    }
}
