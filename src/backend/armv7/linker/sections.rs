//! Section merging for the ARMv7 linker.
//!
//! Phase 5: merges input sections from all objects into output sections,
//! handling COMDAT group deduplication and section type/flag assignment.

use std::collections::{HashMap, HashSet};

use super::types::*;

pub(super) fn merge_sections(
    inputs: &[InputObject],
) -> (Vec<OutputSection>, HashMap<String, usize>, SectionMap) {
    let mut output_sections: Vec<OutputSection> = Vec::new();
    let mut section_name_to_idx: HashMap<String, usize> = HashMap::new();
    let mut section_map: SectionMap = HashMap::new();
    let mut included_comdat_sections: HashSet<String> = HashSet::new();

    let comdat_skip = compute_comdat_skip(inputs);

    for (obj_idx, obj) in inputs.iter().enumerate() {
        for sec in obj.sections.iter() {
            if comdat_skip.contains(&(obj_idx, sec.input_index)) {
                continue;
            }
            let out_name = match output_section_name(&sec.name, sec.flags, sec.sh_type) {
                Some(n) => n,
                None => continue,
            };

            // COMDAT deduplication by section name
            if sec.flags & SHF_GROUP != 0 && !included_comdat_sections.insert(sec.name.clone()) {
                continue;
            }

            let out_idx = if let Some(&idx) = section_name_to_idx.get(&out_name) {
                idx
            } else {
                let idx = output_sections.len();
                let (sh_type, flags) = section_type_and_flags(&out_name, sec);
                section_name_to_idx.insert(out_name.clone(), idx);
                output_sections.push(OutputSection {
                    name: out_name,
                    sh_type,
                    flags,
                    data: Vec::new(),
                    align: 1,
                    addr: 0,
                    file_offset: 0,
                });
                idx
            };

            let out = &mut output_sections[out_idx];
            out.align = out.align.max(sec.align);

            // .init and .fini sections from CRT objects (crti.o, crtn.o) must be
            // concatenated without padding — crti.o provides the function prologue
            // and crtn.o provides the epilogue. Alignment padding would cause the
            // CPU to execute zero bytes between them, breaking the function.
            let align = if out.name == ".init" || out.name == ".fini" {
                1
            } else {
                sec.align.max(1)
            };
            let padding = align_up(out.data.len() as u32, align) as usize - out.data.len();
            out.data.extend(std::iter::repeat(0u8).take(padding));

            let offset_in_output = out.data.len() as u32;
            out.data.extend_from_slice(&sec.data);

            section_map.insert((obj_idx, sec.input_index), (out_idx, offset_in_output));
        }
    }

    (output_sections, section_name_to_idx, section_map)
}

fn compute_comdat_skip(inputs: &[InputObject]) -> HashSet<(usize, usize)> {
    let mut skip: HashSet<(usize, usize)> = HashSet::new();
    let mut seen_groups: HashSet<String> = HashSet::new();

    for (obj_idx, obj) in inputs.iter().enumerate() {
        for sec in &obj.sections {
            if sec.sh_type == SHT_GROUP {
                // The group section data starts with a flag word:
                // bit 0 (GRP_COMDAT) = COMDAT group (duplicates should be eliminated)
                if sec.data.len() < 4 { continue; }
                let flags = read_u32(&sec.data, 0);
                if flags & 1 == 0 { continue; } // Only process COMDAT groups

                // COMDAT groups are identified by their *signature symbol*, not section name
                // (all group sections are typically named ".group").
                // sec.info holds the symbol table index of the signature symbol.
                let group_signature = if (sec.info as usize) < obj.symbols.len() {
                    obj.symbols[sec.info as usize].name.clone()
                } else {
                    continue; // Skip if signature symbol index is invalid
                };
                if !seen_groups.insert(group_signature) {
                    // Duplicate group: skip all member sections
                    if sec.data.len() >= 4 {
                        let member_count = (sec.data.len() - 4) / 4;
                        for i in 0..member_count {
                            let sec_idx = read_u32(&sec.data, 4 + i * 4) as usize;
                            skip.insert((obj_idx, sec_idx));
                        }
                    }
                }
            }
        }
    }

    skip
}

fn section_type_and_flags(out_name: &str, sec: &InputSection) -> (u32, u32) {
    let sh_type = match out_name {
        ".bss" | ".tbss" => SHT_NOBITS,
        ".init_array" => SHT_INIT_ARRAY,
        ".fini_array" => SHT_FINI_ARRAY,
        ".note" => SHT_NOTE,
        _ => SHT_PROGBITS,
    };

    let mut flags = SHF_ALLOC;
    match out_name {
        ".text" | ".init" | ".fini" | ".plt" => flags |= SHF_EXECINSTR,
        ".data" | ".bss" | ".got" | ".got.plt" | ".dynamic" |
        ".init_array" | ".fini_array" => flags |= SHF_WRITE,
        ".tdata" => flags |= SHF_WRITE | SHF_TLS,
        ".tbss" => flags |= SHF_WRITE | SHF_TLS,
        _ => {
            // Inherit flags from input section
            if sec.flags & SHF_WRITE != 0 { flags |= SHF_WRITE; }
            if sec.flags & SHF_EXECINSTR != 0 { flags |= SHF_EXECINSTR; }
            if sec.flags & SHF_TLS != 0 { flags |= SHF_TLS; }
        }
    }

    (sh_type, flags)
}
