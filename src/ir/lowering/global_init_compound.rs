//! Compound (relocation-aware) global initialization.
//!
//! This module handles global variable initialization where fields may contain
//! address expressions requiring relocations. Unlike simple byte-level
//! initialization, compound initialization emits a mix of scalar byte constants
//! and `GlobalAddr` entries that represent addresses resolved at link time.
//!
//! Struct, union, and array initializers with pointer fields (e.g., string
//! literals, function pointers, address-of expressions) are lowered here using
//! `GlobalInit::Compound`, which preserves relocation information for the linker.

use crate::frontend::parser::ast::*;
use crate::ir::ir::*;
use crate::common::types::{AddressSpace, IrType, StructLayout, CType, InitFieldResolution};
use super::lowering::Lowerer;
use super::global_init_helpers as h;
use h::{push_zero_bytes, push_bytes_as_elements, push_string_as_elements};

impl Lowerer {
    /// Lower a struct global init that contains address expressions.
    /// Emits field-by-field using Compound, with padding bytes between fields.
    /// Handles flat init (where multiple items fill an array-of-pointer field),
    /// braced init, and designated init patterns.
    pub(super) fn lower_struct_global_init_compound(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
    ) -> GlobalInit {
        let fam_extra = self.compute_fam_extra_size(items, layout);
        let total_size = layout.size + fam_extra;
        let mut elements: Vec<GlobalInit> = Vec::new();
        let mut current_offset = 0usize;

        // Build a map of field_idx -> list of initializer items.
        // For array fields with flat init, multiple items may map to the same field.
        let mut field_inits: Vec<Vec<&InitializerItem>> = vec![Vec::new(); layout.fields.len()];
        let mut current_field_idx = 0usize;

        // Collect items targeting anonymous members that need synthetic sub-inits.
        // We store them separately to avoid borrow issues with field_inits.
        let mut anon_synth_items: Vec<(usize, InitializerItem)> = Vec::new();

        let mut item_idx = 0;
        while item_idx < items.len() {
            let item = &items[item_idx];

            let designator_name = h::first_field_designator(item);
            let resolution = layout.resolve_init_field(designator_name, current_field_idx, &self.types);

            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designator targets a field inside an anonymous member.
                    // Create a synthetic init item with the inner designator,
                    // preserving any remaining nested designators from the original item.
                    // e.g., for `.base.cra_name = "val"` where `base` is inside an anonymous
                    // union, the synthetic item must be `.base.cra_name = "val"` (not just `.base`).
                    let mut synth_desigs = vec![Designator::Field(inner_name.clone())];
                    if item.designators.len() > 1 {
                        synth_desigs.extend(item.designators[1..].iter().cloned());
                    }
                    let synth_item = InitializerItem {
                        designators: synth_desigs,
                        init: item.init.clone(),
                    };
                    anon_synth_items.push((*anon_field_idx, synth_item));
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    let has_designator = !item.designators.is_empty();
                    if layout.is_union && !has_designator { break; }
                    continue;
                }
                None => {
                    item_idx += 1;
                    continue;
                }
            };
            if field_idx >= layout.fields.len() {
                item_idx += 1;
                continue;
            }

            let field_ty = &layout.fields[field_idx].ty;

            // Check if this is a flat init filling an array field.
            // A string literal initializing a char array is NOT flat init - it's a single
            // complete initializer for the entire array (e.g., char c[10] = "hello").
            // However, for pointer arrays (e.g., char *s[2] = {"abc", "def"}), each
            // string literal IS a flat-init element (it initializes one pointer).
            let is_string_literal = matches!(&item.init, Initializer::Expr(Expr::StringLiteral(..)));
            if let CType::Array(elem_ty, Some(arr_size)) = field_ty {
                let is_char_array = matches!(elem_ty.as_ref(), CType::Char | CType::UChar);
                let skip_flat = is_string_literal && is_char_array;
                if matches!(&item.init, Initializer::Expr(_)) && !skip_flat {
                    // Flat init: consume up to arr_size items for this array field
                    let mut consumed = 0;
                    while consumed < *arr_size && (item_idx + consumed) < items.len() {
                        let cur_item = &items[item_idx + consumed];
                        // Stop if we hit a designator targeting a different field
                        if !cur_item.designators.is_empty() && consumed > 0 {
                            break;
                        }
                        if matches!(&cur_item.init, Initializer::List(_)) && consumed > 0 {
                            break;
                        }
                        field_inits[field_idx].push(cur_item);
                        consumed += 1;
                    }
                    item_idx += consumed;
                    current_field_idx = field_idx + 1;
                    let has_designator = !item.designators.is_empty();
                    if layout.is_union && !has_designator { break; }
                    continue;
                }
            }

            field_inits[field_idx].push(item);
            current_field_idx = field_idx + 1;
            item_idx += 1;
            let has_designator = !item.designators.is_empty();
            if layout.is_union && !has_designator { break; }
        }

        // For unions, find the one initialized field and emit only that,
        // padding to the full union size. Non-union structs emit all fields.
        if layout.is_union {
            // Find which field (if any) has an initializer
            let mut init_fi = None;
            for (i, inits) in field_inits.iter().enumerate() {
                if !inits.is_empty() {
                    init_fi = Some(i);
                    break;
                }
            }
            // Also check anon_synth_items for anonymous member designated inits
            // (e.g., .name = "x" targeting an anonymous struct inside this union)
            if init_fi.is_none() {
                for (idx, _) in &anon_synth_items {
                    init_fi = Some(*idx);
                    break;
                }
            }
            let union_size = layout.size;
            if let Some(fi) = init_fi {
                let inits = &field_inits[fi];
                let field_size = self.resolve_ctype_size(&layout.fields[fi].ty);
                if !inits.is_empty() {
                    self.emit_field_inits_compound(&mut elements, inits, &layout.fields[fi], field_size);
                } else {
                    // Field has no direct inits but may have anonymous member inits
                    let anon_items_for_fi: Vec<InitializerItem> = anon_synth_items.iter()
                        .filter(|(idx, _)| *idx == fi)
                        .map(|(_, item)| item.clone())
                        .collect();
                    if !anon_items_for_fi.is_empty() {
                        let anon_field_ty = &layout.fields[fi].ty;
                        if let Some(sub_layout) = self.get_struct_layout_for_ctype(anon_field_ty) {
                            let sub_init = self.lower_struct_global_init_compound(&anon_items_for_fi, &sub_layout);
                            Self::append_nested_compound(&mut elements, sub_init, field_size);
                        } else {
                            push_zero_bytes(&mut elements, field_size);
                        }
                    } else {
                        push_zero_bytes(&mut elements, field_size);
                    }
                }
                // Pad to full union size
                if field_size < union_size {
                    push_zero_bytes(&mut elements, union_size - field_size);
                }
                current_offset = union_size;
            } else {
                // No initialized field - zero fill entire union
                push_zero_bytes(&mut elements, union_size);
                current_offset = union_size;
            }
        } else {

        let mut fi = 0;
        while fi < layout.fields.len() {
            let field_offset = layout.fields[fi].offset;
            let field_size = self.resolve_ctype_size(&layout.fields[fi].ty);

            // Check if this is a bitfield: if so, we need to pack all bitfields
            // sharing the same storage unit into a single byte buffer.
            if layout.fields[fi].bit_offset.is_some() {
                let storage_unit_offset = field_offset;
                let storage_unit_size = field_size; // All bitfields in this unit have the same type size

                // Emit padding before this storage unit (only if unit starts after current pos)
                if storage_unit_offset > current_offset {
                    let pad = storage_unit_offset - current_offset;
                    push_zero_bytes(&mut elements, pad);
                    current_offset = storage_unit_offset;
                }

                // Allocate a byte buffer for the storage unit and pack all
                // bitfield values into it using read-modify-write.
                let mut unit_bytes = vec![0u8; storage_unit_size];

                // Process all consecutive bitfield fields sharing this storage unit offset
                while fi < layout.fields.len()
                    && layout.fields[fi].offset == storage_unit_offset
                    && layout.fields[fi].bit_offset.is_some()
                {
                    let bit_offset = layout.fields[fi].bit_offset.unwrap();
                    let bit_width = layout.fields[fi].bit_width.unwrap();
                    let field_ir_ty = IrType::from_ctype(&layout.fields[fi].ty);

                    let inits = &field_inits[fi];
                    let val = if !inits.is_empty() {
                        self.eval_init_scalar(&inits[0].init)
                    } else {
                        IrConst::I32(0) // Zero-init for missing initializers
                    };

                    // Pack this bitfield value into the storage unit buffer
                    self.write_bitfield_to_bytes(&mut unit_bytes, 0, &val, field_ir_ty, bit_offset, bit_width);
                    fi += 1;
                }

                // When the storage unit overlaps with already-written data
                // (due to align_down placement), skip the overlapping bytes.
                let skip = if current_offset > storage_unit_offset {
                    current_offset - storage_unit_offset
                } else {
                    0
                };
                // Emit only the non-overlapping portion of the storage unit
                push_bytes_as_elements(&mut elements, &unit_bytes[skip..]);
                current_offset = storage_unit_offset + storage_unit_size;
                continue;
            }

            // Emit padding before this field
            if field_offset > current_offset {
                let pad = field_offset - current_offset;
                push_zero_bytes(&mut elements, pad);
                current_offset = field_offset;
            }

            let inits = &field_inits[fi];

            // Flexible array member (FAM): use fam_extra as the actual data size
            if let CType::Array(ref elem_ty, None) = layout.fields[fi].ty {
                if fam_extra > 0 && !inits.is_empty() {
                    self.emit_fam_compound(&mut elements, inits, elem_ty, fam_extra);
                    current_offset += fam_extra;
                }
                fi += 1;
                continue;
            }

            // Check for synthetic items from anonymous member designated inits
            if inits.is_empty() {
                let anon_items_for_fi: Vec<InitializerItem> = anon_synth_items.iter()
                    .filter(|(idx, _)| *idx == fi)
                    .map(|(_, item)| item.clone())
                    .collect();
                if !anon_items_for_fi.is_empty() {
                    let anon_field_ty = &layout.fields[fi].ty;
                    if let Some(sub_layout) = self.get_struct_layout_for_ctype(anon_field_ty) {
                        let sub_init = self.lower_struct_global_init_compound(&anon_items_for_fi, &sub_layout);
                        Self::append_nested_compound(&mut elements, sub_init, field_size);
                    } else {
                        push_zero_bytes(&mut elements, field_size);
                    }
                } else {
                    push_zero_bytes(&mut elements, field_size);
                }
            } else {
                self.emit_field_inits_compound(&mut elements, inits, &layout.fields[fi], field_size);
            }
            current_offset += field_size;
            fi += 1;
        }

        } // end of else (non-union struct path)

        // Trailing padding
        if current_offset < total_size {
            push_zero_bytes(&mut elements, total_size - current_offset);
        }

        GlobalInit::Compound(elements)
    }

    /// Emit flexible array member (FAM) elements into compound init.
    /// Each FAM element may be a struct containing pointer fields that need relocations.
    fn emit_fam_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        inits: &[&InitializerItem],
        elem_ty: &CType,
        fam_data_size: usize,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);
        if elem_size == 0 { return; }

        // The FAM init should be a single item with a List initializer containing sub-items
        // (one per FAM element), e.g. .numbers = { { .nr = 0, .ns = &init_pid_ns } }
        let sub_items = if inits.len() == 1 {
            if let Initializer::List(ref list) = inits[0].init {
                list.as_slice()
            } else {
                // Single expression init for a FAM element
                let init_item = inits[0];
                self.emit_compound_field_init(elements, &init_item.init, elem_ty, elem_size, false);
                let emitted = elem_size;
                if emitted < fam_data_size {
                    push_zero_bytes(elements, fam_data_size - emitted);
                }
                return;
            }
        } else {
            // Multiple flat items - shouldn't normally happen for struct FAMs
            // but handle by emitting each as an element
            let mut emitted = 0;
            for item in inits {
                if emitted + elem_size > fam_data_size { break; }
                self.emit_compound_field_init(elements, &item.init, elem_ty, elem_size, false);
                emitted += elem_size;
            }
            if emitted < fam_data_size {
                push_zero_bytes(elements, fam_data_size - emitted);
            }
            return;
        };

        // Emit each FAM element
        let num_elems = fam_data_size / elem_size;
        let mut emitted = 0;
        for (i, sub_item) in sub_items.iter().enumerate() {
            if i >= num_elems { break; }
            // Each sub_item is an initializer for one FAM element (e.g., one struct upid)
            if let Some(sub_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                // Struct element - use sub-struct compound emission
                match &sub_item.init {
                    Initializer::List(nested_items) => {
                        self.emit_sub_struct_to_compound(elements, nested_items, &sub_layout, elem_size);
                    }
                    _ => {
                        self.emit_compound_field_init(elements, &sub_item.init, elem_ty, elem_size, false);
                    }
                }
            } else {
                self.emit_compound_field_init(elements, &sub_item.init, elem_ty, elem_size, false);
            }
            emitted += elem_size;
        }
        // Zero-fill remaining FAM elements
        if emitted < fam_data_size {
            push_zero_bytes(elements, fam_data_size - emitted);
        }
    }

    /// Emit a sub-struct's initialization into compound elements, choosing between
    /// compound (relocation-aware) and byte-level approaches based on whether any
    /// field contains address expressions.
    fn emit_sub_struct_to_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        sub_items: &[InitializerItem],
        sub_layout: &StructLayout,
        field_size: usize,
    ) {
        if self.struct_init_has_addr_fields(sub_items, sub_layout) {
            let nested = self.lower_struct_global_init_compound(sub_items, sub_layout);
            Self::append_nested_compound(elements, nested, field_size);
        } else {
            let mut bytes = vec![0u8; field_size];
            self.fill_struct_global_bytes(sub_items, sub_layout, &mut bytes, 0);
            push_bytes_as_elements(elements, &bytes);
        }
    }

    /// Append the elements from a nested GlobalInit::Compound into `elements`,
    /// padding/truncating to exactly `target_size` bytes.
    fn append_nested_compound(elements: &mut Vec<GlobalInit>, nested: GlobalInit, target_size: usize) {
        if let GlobalInit::Compound(nested_elems) = nested {
            let mut emitted = 0;
            for elem in nested_elems {
                if emitted >= target_size { break; }
                emitted += elem.byte_size();
                elements.push(elem);
            }
            while emitted < target_size {
                elements.push(GlobalInit::Scalar(IrConst::I8(0)));
                emitted += 1;
            }
        } else {
            push_zero_bytes(elements, target_size);
        }
    }

    /// Merge a byte buffer with pointer relocations into a Compound global init.
    /// Pointer offsets in `ptr_ranges` replace the corresponding byte ranges with
    /// GlobalAddr entries; all other positions become I8 scalar bytes.
    /// This is the shared finalization step for struct arrays with pointer fields.
    fn build_compound_from_bytes_and_ptrs(
        bytes: Vec<u8>,
        mut ptr_ranges: Vec<(usize, GlobalInit)>,
        total_size: usize,
    ) -> GlobalInit {
        ptr_ranges.sort_by_key(|&(off, _)| off);
        let mut elements: Vec<GlobalInit> = Vec::new();
        let mut pos = 0;
        for (ptr_off, ref addr_init) in &ptr_ranges {
            push_bytes_as_elements(&mut elements, &bytes[pos..*ptr_off]);
            elements.push(addr_init.clone());
            pos = ptr_off + 8; // pointer is 8 bytes
        }
        push_bytes_as_elements(&mut elements, &bytes[pos..total_size]);
        GlobalInit::Compound(elements)
    }

    /// Emit a single field initializer in compound (relocation-aware) mode.
    pub(super) fn emit_compound_field_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        init: &Initializer,
        field_ty: &CType,
        field_size: usize,
        field_is_pointer: bool,
    ) {
        // Complex fields: emit {real, imag} as byte pairs
        if field_ty.is_complex() {
            let mut bytes = vec![0u8; field_size];
            self.write_complex_field_to_bytes(&mut bytes, 0, field_ty, init);
            push_bytes_as_elements(elements, &bytes);
            return;
        }

        match init {
            Initializer::Expr(expr) => {
                if let Expr::StringLiteral(s, _) = expr {
                    if field_is_pointer {
                        // String literal initializing a pointer field:
                        // create a .rodata string entry and emit GlobalAddr
                        let label = self.intern_string_literal(s);
                        elements.push(GlobalInit::GlobalAddr(label));
                    } else {
                        // String literal initializing a char array field
                        push_string_as_elements(elements, s, field_size);
                    }
                } else if let Expr::AddressOf(inner, _) = expr {
                    if let Expr::CompoundLiteral(ref cl_type_spec, ref cl_init, _) = inner.as_ref() {
                        // &(compound_literal) at file scope: create anonymous global
                        let addr_init = self.create_compound_literal_global(cl_type_spec, cl_init);
                        elements.push(addr_init);
                    } else if let Some(addr_init) = self.eval_string_literal_addr_expr(expr) {
                        elements.push(addr_init);
                    } else if let Some(addr_init) = self.eval_global_addr_expr(expr) {
                        elements.push(addr_init);
                    } else {
                        push_zero_bytes(elements, field_size);
                    }
                } else if let Expr::CompoundLiteral(_, ref cl_init, _) = expr {
                    // Non-pointer compound literal: unwrap and use inner initializer
                    self.emit_compound_field_init(elements, cl_init, field_ty, field_size, field_is_pointer);
                } else {
                    // Scalar constant, string literal addr, global addr, or zero fallback
                    self.emit_expr_to_compound(elements, expr, field_size, Some(field_ty));
                }
            }
            Initializer::List(nested_items) => {
                // Check if this is an array whose elements contain pointers
                if let CType::Array(elem_ty, Some(arr_size)) = field_ty {
                    if h::type_has_pointer_elements(elem_ty, &self.types) {
                        // Distinguish: direct pointer array vs struct-with-pointer-fields array
                        if matches!(elem_ty.as_ref(), CType::Pointer(_, _) | CType::Function(_)) {
                            // Array of direct pointers: emit each element as a GlobalAddr or zero
                            self.emit_compound_ptr_array_init(elements, nested_items, elem_ty, *arr_size);
                            return;
                        }
                        // Array of structs/unions containing pointer fields:
                        // use the struct-array-with-ptrs path (byte buffer + ptr_ranges)
                        if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                            let nested = self.lower_struct_array_with_ptrs(nested_items, &elem_layout, *arr_size);
                            if let GlobalInit::Compound(nested_elems) = nested {
                                elements.extend(nested_elems);
                            } else {
                                push_zero_bytes(elements, field_size);
                            }
                            return;
                        }
                    }
                }

                // Nested struct/union: delegate to emit_sub_struct_to_compound
                // which handles compound-vs-bytes selection automatically.
                let field_ty_clone = field_ty.clone();
                if let Some(nested_layout) = self.get_struct_layout_for_ctype(&field_ty_clone) {
                    self.emit_sub_struct_to_compound(elements, nested_items, &nested_layout, field_size);
                    return;
                }

                // Non-struct field: serialize to bytes (arrays, scalars)
                let mut bytes = vec![0u8; field_size];
                if let CType::Array(ref inner_ty, Some(arr_size)) = field_ty_clone {
                    if inner_ty.is_complex() {
                        let elem_size = self.resolve_ctype_size(inner_ty);
                        self.fill_array_of_complex(nested_items, inner_ty, arr_size, elem_size, &mut bytes, 0);
                    } else if matches!(inner_ty.as_ref(), CType::Struct(_) | CType::Union(_)) {
                        // Array of structs/unions without pointer fields: use composite
                        // array filler which handles Initializer::List sub-items correctly.
                        let elem_size = self.resolve_ctype_size(inner_ty);
                        self.fill_array_of_composites(nested_items, inner_ty, arr_size, elem_size, &mut bytes, 0);
                    } else {
                        self.fill_scalar_list_to_bytes(nested_items, inner_ty, field_size, &mut bytes);
                    }
                } else {
                    self.fill_scalar_list_to_bytes(nested_items, &field_ty_clone, field_size, &mut bytes);
                }
                push_bytes_as_elements(elements, &bytes);
            }
        }
    }

    /// Emit all initializer items for a single struct/union field in compound mode.
    /// Handles the full dispatch: anonymous members, nested designators, flat array init,
    /// and single-expression fields. This is the shared logic between the union and
    /// non-union struct paths in `lower_struct_global_init_compound`.
    fn emit_field_inits_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        inits: &[&InitializerItem],
        field: &crate::common::types::StructFieldLayout,
        field_size: usize,
    ) {
        let field_is_pointer = matches!(field.ty, CType::Pointer(_, _) | CType::Function(_));

        if inits.is_empty() {
            push_zero_bytes(elements, field_size);
        } else if inits.len() == 1 {
            let item = inits[0];
            let desig_name = h::first_field_designator(item);
            let is_anon = h::is_anon_member_designator(
                desig_name, &field.name, &field.ty);

            if is_anon {
                let sub_item = InitializerItem {
                    designators: item.designators.clone(),
                    init: item.init.clone(),
                };
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &[sub_item], &sub_layout, field_size);
            } else if h::has_nested_field_designator(item) {
                self.emit_compound_nested_designator_field(
                    elements, item, &field.ty, field_size);
            } else {
                let has_array_idx_designator = item.designators.iter().any(|d| matches!(d, Designator::Index(_)));
                if has_array_idx_designator {
                    if let CType::Array(elem_ty, Some(arr_size)) = &field.ty {
                        if h::type_has_pointer_elements(elem_ty, &self.types) {
                            self.emit_compound_ptr_array_designated_init(
                                elements, &[item], elem_ty, *arr_size);
                            return;
                        }
                    }
                }
                self.emit_compound_field_init(elements, &item.init, &field.ty, field_size, field_is_pointer);
            }
        } else {
            // Multiple items targeting the same outer struct field.
            // Check if they all have nested field designators (e.g., .mmu.f1, .mmu.f2, .mmu.f3)
            // targeting sub-fields of a struct/union field.
            let all_have_nested_desig = inits.iter().all(|item| h::has_nested_field_designator(item));
            let field_is_struct = matches!(&field.ty, CType::Struct(_) | CType::Union(_));

            let is_anon_multi = field.name.is_empty() && field_is_struct;
            if all_have_nested_desig && field_is_struct {
                // Strip the outer field designator from each item and delegate
                // to sub-struct compound init with the inner designators.
                let sub_items: Vec<InitializerItem> = inits.iter().map(|item| {
                    InitializerItem {
                        designators: item.designators[1..].to_vec(),
                        init: item.init.clone(),
                    }
                }).collect();
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &sub_items, &sub_layout, field_size);
            } else if is_anon_multi {
                let sub_items: Vec<InitializerItem> = inits.iter().map(|item| {
                    InitializerItem {
                        designators: item.designators.clone(),
                        init: item.init.clone(),
                    }
                }).collect();
                let sub_layout = self.get_struct_layout_for_ctype(&field.ty)
                    .unwrap_or_else(StructLayout::empty_rc);
                self.emit_sub_struct_to_compound(elements, &sub_items, &sub_layout, field_size);
            } else {
                self.emit_compound_flat_array_init(elements, inits, &field.ty, field_size);
            }
        }
    }

    /// Emit a field whose initializer has multi-level designators targeting a sub-field
    /// within this struct/union (e.g., .bs.keyword = {"STORE", -1}).
    ///
    /// Drills through the designator chain to find the target sub-field, then emits
    /// the field data using compound (relocation-aware) initialization for the target
    /// sub-field, with zero-fill for the rest of the outer field.
    fn emit_compound_nested_designator_field(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        item: &InitializerItem,
        outer_ty: &CType,
        outer_size: usize,
    ) {
        let drill = match self.drill_designators(&item.designators[1..], outer_ty) {
            Some(d) => d,
            None => {
                push_zero_bytes(elements, outer_size);
                return;
            }
        };

        let sub_offset = drill.byte_offset;
        let current_ty = drill.target_ty;
        let sub_size = self.resolve_ctype_size(&current_ty);
        let sub_is_pointer = matches!(current_ty, CType::Pointer(_, _) | CType::Function(_));

        // Handle bitfield targets: pack the value into the storage unit bytes
        if let (Some(bo), Some(bw)) = (drill.bit_offset, drill.bit_width) {
            let ir_ty = IrType::from_ctype(&current_ty);
            let val = self.eval_init_scalar(&item.init);
            // Emit zero bytes before the storage unit
            push_zero_bytes(elements, sub_offset);
            // Create a small buffer for the storage unit, write bitfield, emit as bytes
            let storage_size = ir_ty.size();
            let mut buf = vec![0u8; storage_size];
            self.write_bitfield_to_bytes(&mut buf, 0, &val, ir_ty, bo, bw);
            push_bytes_as_elements(elements, &buf);
            // Zero-fill remaining
            let remaining = outer_size.saturating_sub(sub_offset + storage_size);
            push_zero_bytes(elements, remaining);
            return;
        }

        // Emit zero bytes before the sub-field
        push_zero_bytes(elements, sub_offset);

        // If the target is a struct/union with a list init, delegate to the
        // sub-struct emission helper which handles compound vs bytes selection.
        let handled = if let (CType::Struct(_) | CType::Union(_), Initializer::List(nested_items)) =
            (&current_ty, &item.init)
        {
            if let Some(target_layout) = self.get_struct_layout_for_ctype(&current_ty) {
                self.emit_sub_struct_to_compound(elements, nested_items, &target_layout, sub_size);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !handled {
            self.emit_compound_field_init(elements, &item.init, &current_ty, sub_size, sub_is_pointer);
        }

        // Zero-fill the remaining outer field after the sub-field
        let remaining = outer_size.saturating_sub(sub_offset + sub_size);
        push_zero_bytes(elements, remaining);
    }

    /// Emit an array-of-pointers field from a braced initializer list.
    /// Each element is either a GlobalAddr (for string literals / address expressions)
    /// or zero-filled (for missing elements).
    pub(super) fn emit_compound_ptr_array_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        items: &[InitializerItem],
        _elem_ty: &CType,
        arr_size: usize,
    ) {
        let ptr_size = 8; // 64-bit pointers
        let mut ai = 0usize;

        for item in items {
            if ai >= arr_size { break; }

            // Handle index designator: [idx] = val
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    // Zero-fill skipped elements
                    while ai < idx && ai < arr_size {
                        push_zero_bytes(elements, ptr_size);
                        ai += 1;
                    }
                }
            }
            if ai >= arr_size { break; }

            if let Initializer::Expr(ref expr) = item.init {
                self.emit_expr_to_compound(elements, expr, ptr_size, None);
            } else {
                // Nested list - zero fill this element
                push_zero_bytes(elements, ptr_size);
            }
            ai += 1;
        }

        // Zero-fill remaining elements
        while ai < arr_size {
            push_zero_bytes(elements, ptr_size);
            ai += 1;
        }
    }

    /// Emit a designated initializer for a pointer array field.
    /// Handles cases like `.a[1] = "abc"` where a is `char *a[3]`.
    /// The items may have field+index designators; we extract the index from the designators.
    pub(super) fn emit_compound_ptr_array_designated_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        items: &[&InitializerItem],
        _elem_ty: &CType,
        arr_size: usize,
    ) {
        let ptr_size = 8; // 64-bit pointers
        // Build a sparse map of which indices are initialized
        let mut index_inits: Vec<Option<&Initializer>> = vec![None; arr_size];

        for item in items {
            // Find the Index designator (may be after a Field designator)
            let idx = item.designators.iter().find_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            }).unwrap_or(0);

            if idx < arr_size {
                index_inits[idx] = Some(&item.init);
            }
        }

        // Emit each element
        for ai in 0..arr_size {
            if let Some(init) = index_inits[ai] {
                if let Initializer::Expr(ref expr) = init {
                    self.emit_expr_to_compound(elements, expr, ptr_size, None);
                } else {
                    push_zero_bytes(elements, ptr_size);
                }
            } else {
                // Uninitialized element - zero fill
                push_zero_bytes(elements, ptr_size);
            }
        }
    }

    /// Emit a flat array init for a field that has multiple items (flat init style).
    /// E.g., struct { char *s[2]; } x = { "abc", "def" };
    pub(super) fn emit_compound_flat_array_init(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        inits: &[&InitializerItem],
        field_ty: &CType,
        field_size: usize,
    ) {
        let (elem_ty, arr_size) = match field_ty {
            CType::Array(inner, Some(size)) => (inner.as_ref(), *size),
            _ => {
                // Not an array - just use first item
                if let Some(first) = inits.first() {
                    let field_is_pointer = matches!(field_ty, CType::Pointer(_, _));
                    self.emit_compound_field_init(elements, &first.init, field_ty, field_size, field_is_pointer);
                } else {
                    push_zero_bytes(elements, field_size);
                }
                return;
            }
        };

        let elem_is_pointer = h::type_has_pointer_elements(elem_ty, &self.types);
        let elem_size = self.resolve_ctype_size(elem_ty);
        let ptr_size = 8; // 64-bit

        let mut ai = 0usize;
        for item in inits {
            if ai >= arr_size { break; }

            if let Initializer::Expr(ref expr) = item.init {
                if elem_is_pointer {
                    self.emit_expr_to_compound(elements, expr, ptr_size, None);
                } else {
                    if let Some(val) = self.eval_const_expr(expr) {
                        let elem_ir_ty = IrType::from_ctype(elem_ty);
                        let coerced = val.coerce_to(elem_ir_ty);
                        self.push_const_as_bytes(elements, &coerced, elem_size);
                    } else {
                        push_zero_bytes(elements, elem_size);
                    }
                }
            } else {
                // Nested list - zero fill element
                push_zero_bytes(elements, elem_size);
            }
            ai += 1;
        }

        // Zero-fill remaining elements
        while ai < arr_size {
            push_zero_bytes(elements, elem_size);
            ai += 1;
        }
    }

    /// Resolve which struct field a positional or designated initializer targets.
    pub(super) fn resolve_struct_init_field_idx(
        &self,
        item: &InitializerItem,
        layout: &StructLayout,
        current_field_idx: usize,
    ) -> usize {
        let desig_name = h::first_field_designator(item);
        layout.resolve_init_field_idx(desig_name, current_field_idx, &self.types)
            .unwrap_or(current_field_idx)
    }

    /// Emit a single expression as a compound element.
    ///
    /// When `coerce_ty` is Some, tries const eval with type coercion first (for
    /// scalar fields with known types, including _Bool normalization). When None,
    /// tries address resolution first (for pointer fields and pointer array elements).
    ///
    /// Fallback chain: const eval / string literal -> string addr -> global addr -> zero.
    fn emit_expr_to_compound(
        &mut self,
        elements: &mut Vec<GlobalInit>,
        expr: &Expr,
        element_size: usize,
        coerce_ty: Option<&CType>,
    ) {
        // For typed scalar fields, try const eval with coercion first
        if let Some(ty) = coerce_ty {
            if let Some(val) = self.eval_const_expr(expr) {
                let coerced = if *ty == CType::Bool {
                    val.bool_normalize()
                } else {
                    val.coerce_to(IrType::from_ctype(ty))
                };
                self.push_const_as_bytes(elements, &coerced, element_size);
                return;
            }
        }
        // For pointer/address contexts, try address resolution first
        if let Expr::LabelAddr(label_name, _) = expr {
            // GCC &&label extension: emit label address as GlobalAddr
            let scoped_label = self.get_or_create_user_label(label_name);
            elements.push(GlobalInit::GlobalAddr(scoped_label.as_label()));
        } else if let Expr::StringLiteral(s, _) = expr {
            let label = self.intern_string_literal(s);
            elements.push(GlobalInit::GlobalAddr(label));
        } else if let Some(addr_init) = self.eval_string_literal_addr_expr(expr) {
            elements.push(addr_init);
        } else if let Some(addr_init) = self.eval_global_addr_expr(expr) {
            elements.push(addr_init);
        } else if coerce_ty.is_none() {
            // Untyped path: try raw const eval as last resort
            if let Some(val) = self.eval_const_expr(expr) {
                self.push_const_as_bytes(elements, &val, element_size);
            } else {
                push_zero_bytes(elements, element_size);
            }
        } else {
            push_zero_bytes(elements, element_size);
        }
    }

    /// Resolve a pointer field's initializer expression to a GlobalInit.
    /// Handles string literals, global addresses, function pointers, etc.
    pub(super) fn resolve_ptr_field_init(&mut self, expr: &Expr) -> Option<GlobalInit> {
        // GCC &&label extension: label address in a pointer field
        if let Expr::LabelAddr(label_name, _) = expr {
            let scoped_label = self.get_or_create_user_label(label_name);
            return Some(GlobalInit::GlobalAddr(scoped_label.as_label()));
        }
        // String literal: create a .rodata entry and reference it
        if let Expr::StringLiteral(s, _) = expr {
            let label = self.intern_string_literal(s);
            return Some(GlobalInit::GlobalAddr(label));
        }
        // String literal +/- offset: "str" + N
        if let Some(addr) = self.eval_string_literal_addr_expr(expr) {
            return Some(addr);
        }
        // Try as a global address expression (&x, func name, array name, etc.)
        if let Some(addr) = self.eval_global_addr_expr(expr) {
            return Some(addr);
        }
        // Integer constant 0 -> null pointer
        if let Some(val) = self.eval_const_expr(expr) {
            if let Some(v) = self.const_to_i64(&val) {
                if v == 0 {
                    return None; // Will be zero in the byte buffer
                }
            }
            // Non-zero constant pointer (unusual but possible)
            return Some(GlobalInit::Scalar(val));
        }
        None
    }

    /// Get a StructLayout for a CType if it's a struct or union.
    pub(super) fn get_struct_layout_for_ctype(&self, ty: &CType) -> Option<crate::common::types::RcLayout> {
        match ty {
            CType::Struct(key) | CType::Union(key) => {
                self.types.struct_layouts.get(&**key).cloned()
            }
            _ => None,
        }
    }

    /// Lower a (possibly multi-dimensional) array of structs where some fields are pointers.
    /// Handles multi-dimensional arrays by recursing through dimension strides.
    pub(super) fn lower_struct_array_with_ptrs_multidim(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
        total_size: usize,
        array_dim_strides: &[usize],
    ) -> GlobalInit {
        let struct_size = layout.size;
        let mut bytes = vec![0u8; total_size];
        let mut ptr_ranges: Vec<(usize, GlobalInit)> = Vec::new();

        self.fill_multidim_struct_array_with_ptrs(
            items, layout, struct_size, array_dim_strides,
            &mut bytes, &mut ptr_ranges, 0, total_size,
        );

        Self::build_compound_from_bytes_and_ptrs(bytes, ptr_ranges, total_size)
    }

    /// Recursively fill a multi-dimensional struct array into byte buffer + ptr_ranges.
    /// Similar to fill_multidim_struct_array_bytes but with pointer relocation tracking.
    fn fill_multidim_struct_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
        struct_size: usize,
        array_dim_strides: &[usize],
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
        base_offset: usize,
        region_size: usize,
    ) {
        if struct_size == 0 { return; }

        let (this_stride, remaining_strides) = if array_dim_strides.len() > 1 {
            (array_dim_strides[0], &array_dim_strides[1..])
        } else if array_dim_strides.len() == 1 {
            (array_dim_strides[0], &array_dim_strides[0..0])
        } else {
            (struct_size, &array_dim_strides[0..0])
        };

        let num_elems = if this_stride > 0 { region_size / this_stride } else { 0 };

        if h::has_array_field_designators(items) && this_stride == struct_size {
            self.fill_array_field_designator_items(
                items, layout, struct_size, num_elems, base_offset, bytes, ptr_ranges,
            );
        } else {
            // Sequential items: each item maps to one element at this_stride
            let mut current_idx = 0usize;
            for item in items {
                if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                    if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                        current_idx = idx;
                    }
                }
                if current_idx >= num_elems { current_idx += 1; continue; }

                let elem_offset = base_offset + current_idx * this_stride;

                match &item.init {
                    Initializer::List(sub_items) => {
                        if this_stride > struct_size && !remaining_strides.is_empty() {
                            // Sub-array dimension: recurse
                            self.fill_multidim_struct_array_with_ptrs(
                                sub_items, layout, struct_size, remaining_strides,
                                bytes, ptr_ranges, elem_offset, this_stride,
                            );
                        } else {
                            // Single struct element: fill its fields
                            self.fill_nested_struct_with_ptrs(
                                sub_items, layout, elem_offset, bytes, ptr_ranges,
                            );
                        }
                    }
                    Initializer::Expr(expr) => {
                        // Single expression for first field of struct element
                        if !layout.fields.is_empty() {
                            let field = &layout.fields[0];
                            let field_offset = elem_offset + field.offset;
                            self.write_expr_to_bytes_or_ptrs(
                                expr, &field.ty, field_offset,
                                field.bit_offset, field.bit_width,
                                bytes, ptr_ranges,
                            );
                        }
                    }
                }
                current_idx += 1;
            }
        }
    }

    /// Lower an array of structs where some fields are pointers.
    /// Uses byte-level serialization but with Compound for address elements.
    /// Each struct element is emitted as a mix of byte constants and pointer-sized addresses.
    pub(super) fn lower_struct_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
        num_elems: usize,
    ) -> GlobalInit {
        // We emit the entire array as a sequence of Compound elements.
        // Each element is either:
        //   - Scalar(I8) for byte-level constant data
        //   - GlobalAddr for pointer fields
        let struct_size = layout.size;
        let total_size = num_elems * struct_size;

        // Initialize with zero bytes and track pointer relocation ranges
        let mut bytes = vec![0u8; total_size];
        let mut ptr_ranges: Vec<(usize, GlobalInit)> = Vec::new();

        if h::has_array_field_designators(items) {
            // Handle [N].field = value pattern (e.g., postgres mcxt_methods[]).
            self.fill_array_field_designator_items(
                items, layout, struct_size, num_elems, 0, &mut bytes, &mut ptr_ranges,
            );
        } else {
            // Original path: items correspond 1-to-1 to array elements (no [N].field designators).
            // Each item is either a braced list for one struct element or a single expression.
            self.fill_struct_array_sequential(
                items, layout, num_elems,
                &mut bytes, &mut ptr_ranges,
            );
        }

        Self::build_compound_from_bytes_and_ptrs(bytes, ptr_ranges, total_size)
    }

    /// Fill struct fields from an initializer item list into the byte buffer and ptr_ranges.
    /// Used both for braced init lists `{ field1, field2, ... }` and for unwrapped compound
    /// literals `((struct S) { field1, field2, ... })` in struct array initializers.
    ///
    /// Handles brace elision for array fields: when an expression item (not a braced list)
    /// targets an array field, consecutive items are consumed to fill the array elements
    /// (C11 6.7.9p17-21).
    fn fill_struct_fields_from_items(
        &mut self,
        sub_items: &[InitializerItem],
        layout: &StructLayout,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let mut current_field_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < sub_items.len() {
            let sub_item = &sub_items[item_idx];
            let desig_name = h::first_field_designator(sub_item);
            let resolution = layout.resolve_init_field(desig_name, current_field_idx, &self.types);
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designator targets a field inside an anonymous struct/union member.
                    // Recursively fill the anonymous member's sub-layout.
                    let extra_desigs = if sub_item.designators.len() > 1 { &sub_item.designators[1..] } else { &[] };
                    if let Some(res) = h::resolve_anonymous_member(layout, *anon_field_idx, inner_name, &sub_item.init, extra_desigs, &self.types.struct_layouts) {
                        let anon_offset = base_offset + res.anon_offset;
                        self.fill_struct_fields_from_items(
                            &[res.sub_item], &res.sub_layout, anon_offset, bytes, ptr_ranges,
                        );
                    }
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    continue;
                }
                None => break,
            };
            let field = &layout.fields[field_idx];
            let field_offset = base_offset + field.offset;

            if h::has_nested_field_designator(sub_item) {
                self.fill_nested_designator_with_ptrs(
                    sub_item, &field.ty, field_offset,
                    bytes, ptr_ranges,
                );
                current_field_idx = field_idx + 1;
                item_idx += 1;
                continue;
            }

            // Brace elision for array fields: when an expression (not a braced list) targets
            // an array field, consume consecutive items to fill the array elements.
            if matches!(&sub_item.init, Initializer::Expr(_)) {
                if let CType::Array(ref elem_ty, Some(arr_size)) = field.ty {
                    // Don't apply flat consumption for string literals initializing char arrays
                    let is_char_array_str = matches!(elem_ty.as_ref(), CType::Char | CType::UChar)
                        && matches!(&sub_item.init, Initializer::Expr(Expr::StringLiteral(..)));
                    if !is_char_array_str {
                        let consumed = self.fill_flat_array_field_with_ptrs(
                            &sub_items[item_idx..], elem_ty, arr_size,
                            field_offset, bytes, ptr_ranges,
                        );
                        item_idx += consumed;
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                }
            }

            self.emit_struct_field_init_compound(
                sub_item, field, field_offset,
                bytes, ptr_ranges,
            );
            current_field_idx = field_idx + 1;
            item_idx += 1;
        }
    }

    /// Emit a single struct field initialization into the byte buffer and ptr_ranges.
    /// Handles pointer fields, pointer array fields, bitfields, nested structs, and scalars.
    fn emit_struct_field_init_compound(
        &mut self,
        item: &InitializerItem,
        field: &crate::common::types::StructFieldLayout,
        field_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        // Only treat as a flat pointer array if elements are direct pointers/functions,
        // NOT if elements are structs that happen to contain pointer fields.
        let is_ptr_array = matches!(field.ty, CType::Array(ref elem_ty, _)
            if matches!(elem_ty.as_ref(), CType::Pointer(_, _) | CType::Function(_)));

        if let Initializer::Expr(ref expr) = item.init {
            // For pointer arrays with a single expr, the first element is the addr
            let effective_ty = if is_ptr_array {
                // Treat as a pointer write to the first element
                &CType::Pointer(Box::new(CType::Void), AddressSpace::Default)
            } else {
                &field.ty
            };
            self.write_expr_to_bytes_or_ptrs(
                expr, effective_ty, field_offset,
                field.bit_offset, field.bit_width,
                bytes, ptr_ranges,
            );
        } else if let Initializer::List(ref inner_items) = item.init {
            if is_ptr_array {
                let arr_size = match &field.ty {
                    CType::Array(_, Some(s)) => *s,
                    _ => inner_items.len(),
                };
                let ptr_ty = CType::Pointer(Box::new(CType::Void), AddressSpace::Default);
                for (ai, inner_item) in inner_items.iter().enumerate() {
                    if ai >= arr_size { break; }
                    let elem_offset = field_offset + ai * 8;
                    if let Initializer::Expr(ref expr) = inner_item.init {
                        self.write_expr_to_bytes_or_ptrs(
                            expr, &ptr_ty, elem_offset, None, None, bytes, ptr_ranges,
                        );
                    }
                }
            } else {
                self.fill_composite_or_array_with_ptrs(
                    inner_items, &field.ty, field_offset, bytes, ptr_ranges,
                );
            }
        }
    }

    /// Fill a struct array where items map to array elements.
    /// Handles both sequential init and `[N] = {...}` designated index patterns.
    /// Each item is either a braced list `{ .field = val, ... }` or a single expression.
    fn fill_struct_array_sequential(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
        num_elems: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let struct_size = layout.size;
        // Track the current sequential index for items without designators
        let mut current_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < items.len() {
            let item = &items[item_idx];
            // Check if this item has an [N] array index designator
            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr_for_designator(idx_expr) {
                    current_idx = idx;
                }
            }
            let elem_idx = current_idx;
            if elem_idx >= num_elems {
                current_idx += 1;
                item_idx += 1;
                continue;
            }
            let base_offset = elem_idx * struct_size;

            match &item.init {
                Initializer::List(sub_items) => {
                    self.fill_struct_fields_from_items(
                        sub_items, layout, base_offset, bytes, ptr_ranges,
                    );
                    item_idx += 1;
                }
                Initializer::Expr(ref expr) => {
                    // Compound literal, e.g., ((struct Wrap) {inc_global}):
                    // unwrap it and process the inner initializer list as struct fields.
                    if let Expr::CompoundLiteral(_, ref inner_init, _) = expr {
                        if let Initializer::List(ref sub_items) = **inner_init {
                            self.fill_struct_fields_from_items(
                                sub_items, layout, base_offset, bytes, ptr_ranges,
                            );
                        } else if let Initializer::Expr(ref inner_expr) = **inner_init {
                            // Scalar compound literal (e.g., ((type){expr})):
                            // treat the inner expression as the value for the first field
                            if !layout.fields.is_empty() {
                                let field = &layout.fields[0];
                                let field_offset = base_offset + field.offset;
                                let synth_item = InitializerItem {
                                    designators: Vec::new(),
                                    init: Initializer::Expr(inner_expr.clone()),
                                };
                                self.emit_struct_field_init_compound(
                                    &synth_item, field, field_offset,
                                    bytes, ptr_ranges,
                                );
                            }
                        }
                        item_idx += 1;
                    } else {
                        // Flat initialization: consume items field-by-field for this struct.
                        // Each item corresponds to one field, not one struct element.
                        let mut current_field_idx = 0usize;
                        while item_idx < items.len() && current_field_idx < layout.fields.len() {
                            let sub_item = &items[item_idx];
                            // If this item has an array index designator, it starts a new element
                            if sub_item.designators.first().is_some() && item_idx != 0 {
                                break;
                            }
                            let field = &layout.fields[current_field_idx];
                            let field_offset = base_offset + field.offset;

                            self.emit_struct_field_init_compound(
                                sub_item, field, field_offset,
                                bytes, ptr_ranges,
                            );
                            current_field_idx += 1;
                            item_idx += 1;
                        }
                    }
                }
            }
            current_idx += 1;
        }
    }

    /// Process `[N].field = value` designated initializer items for a struct array.
    /// This is the consolidated loop that handles the `[Index(N), Field("name")]`
    /// designator pattern used by both `fill_multidim_struct_array_with_ptrs` and
    /// `lower_struct_array_with_ptrs`. Each item targets a specific array element and
    /// field within it.
    fn fill_array_field_designator_items(
        &mut self,
        items: &[InitializerItem],
        layout: &StructLayout,
        struct_size: usize,
        num_elems: usize,
        array_base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let mut current_elem_idx = 0usize;
        let mut current_field_idx = 0usize;
        for item in items {
            let mut elem_idx = current_elem_idx;
            let mut remaining_desigs_start = 0;

            if let Some(Designator::Index(ref idx_expr)) = item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    if idx != current_elem_idx { current_field_idx = 0; }
                    elem_idx = idx;
                }
                remaining_desigs_start = 1;
            }

            let mut field_desig: Option<&str> = None;
            if let Some(Designator::Field(ref name)) = item.designators.get(remaining_desigs_start) {
                field_desig = Some(name.as_str());
                remaining_desigs_start += 1;
            }

            if elem_idx >= num_elems { current_elem_idx = elem_idx + 1; continue; }

            let elem_base = array_base_offset + elem_idx * struct_size;
            let resolution = layout.resolve_init_field(field_desig, current_field_idx, &self.types);
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Designator targets a field inside an anonymous struct/union member.
                    let extra_desigs = if item.designators.len() > remaining_desigs_start { &item.designators[remaining_desigs_start..] } else { &[] };
                    if let Some(res) = h::resolve_anonymous_member(layout, *anon_field_idx, inner_name, &item.init, extra_desigs, &self.types.struct_layouts) {
                        let anon_offset = elem_base + res.anon_offset;
                        self.fill_struct_fields_from_items(
                            &[res.sub_item], &res.sub_layout, anon_offset, bytes, ptr_ranges,
                        );
                    }
                    current_elem_idx = elem_idx;
                    current_field_idx = *anon_field_idx + 1;
                    continue;
                }
                None => { current_elem_idx = elem_idx; continue; }
            };
            let field = &layout.fields[field_idx];
            let field_offset = elem_base + field.offset;

            if remaining_desigs_start < item.designators.len() {
                let remaining_item = InitializerItem {
                    designators: item.designators[remaining_desigs_start..].to_vec(),
                    init: item.init.clone(),
                };
                self.fill_nested_designator_with_ptrs(
                    &remaining_item, &field.ty, field_offset, bytes, ptr_ranges,
                );
            } else {
                self.emit_struct_field_init_compound(
                    item, field, field_offset, bytes, ptr_ranges,
                );
            }

            current_elem_idx = elem_idx;
            current_field_idx = field_idx + 1;
        }
    }

    /// Handle a multi-level designated initializer within a struct array element.
    /// Drills through the designator chain (starting from the second designator)
    /// to find the actual target sub-field, then writes integer values to the byte
    /// buffer and records pointer relocations in ptr_ranges.
    fn fill_nested_designator_with_ptrs(
        &mut self,
        item: &InitializerItem,
        outer_ty: &CType,
        outer_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        // Drill through designators starting from the second one
        let drill = match self.drill_designators(&item.designators[1..], outer_ty) {
            Some(d) => d,
            None => return,
        };

        let sub_offset = outer_offset + drill.byte_offset;
        let current_ty = drill.target_ty;

        match &item.init {
            Initializer::Expr(expr) => {
                self.write_expr_to_bytes_or_ptrs(
                    expr, &current_ty, sub_offset,
                    drill.bit_offset, drill.bit_width,
                    bytes, ptr_ranges,
                );
            }
            Initializer::List(inner_items) => {
                self.fill_composite_or_array_with_ptrs(
                    inner_items, &current_ty, sub_offset, bytes, ptr_ranges,
                );
            }
        }
    }

    /// Fill a nested struct's fields into the byte buffer and ptr_ranges,
    /// properly handling pointer/function pointer fields that need relocations.
    /// This is used when a struct field with a braced initializer list contains
    /// pointer-type sub-fields within a struct array element context.
    fn fill_nested_struct_with_ptrs(
        &mut self,
        inner_items: &[InitializerItem],
        sub_layout: &StructLayout,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let mut current_field_idx = 0usize;
        let mut item_idx = 0usize;
        while item_idx < inner_items.len() {
            let inner_item = &inner_items[item_idx];
            let desig_name = h::first_field_designator(inner_item);
            let resolution = sub_layout.resolve_init_field(desig_name, current_field_idx, &self.types);
            let field_idx = match &resolution {
                Some(InitFieldResolution::Direct(idx)) => *idx,
                Some(InitFieldResolution::AnonymousMember { anon_field_idx, inner_name }) => {
                    // Drill into anonymous member
                    let anon_field = &sub_layout.fields[*anon_field_idx];
                    let anon_offset = base_offset + anon_field.offset;
                    if let Some(anon_layout) = self.get_struct_layout_for_ctype(&anon_field.ty) {
                        let sub_item = InitializerItem {
                            designators: vec![Designator::Field(inner_name.clone())],
                            init: inner_item.init.clone(),
                        };
                        self.fill_nested_struct_with_ptrs(
                            &[sub_item], &anon_layout, anon_offset, bytes, ptr_ranges);
                    }
                    current_field_idx = *anon_field_idx + 1;
                    item_idx += 1;
                    continue;
                }
                None => break,
            };

            // Handle multi-level designators within the nested struct (e.g., .config.i = &val)
            if h::has_nested_field_designator(inner_item) {
                if field_idx < sub_layout.fields.len() {
                    let field = &sub_layout.fields[field_idx];
                    let field_abs_offset = base_offset + field.offset;

                    if let Some(drill) = self.drill_designators(&inner_item.designators[1..], &field.ty) {
                        let sub_offset = field_abs_offset + drill.byte_offset;
                        if let Initializer::Expr(ref expr) = inner_item.init {
                            self.write_expr_to_bytes_or_ptrs(
                                expr, &drill.target_ty, sub_offset,
                                drill.bit_offset, drill.bit_width,
                                bytes, ptr_ranges,
                            );
                        }
                    }
                    current_field_idx = field_idx + 1;
                    item_idx += 1;
                    continue;
                }
            }

            let field = &sub_layout.fields[field_idx];
            let field_abs_offset = base_offset + field.offset;

            // Brace elision for array fields: when an expression (not a braced list) targets
            // an array field, consume consecutive items to fill the array elements.
            if matches!(&inner_item.init, Initializer::Expr(_)) {
                if let CType::Array(ref elem_ty, Some(arr_size)) = field.ty {
                    let is_char_array_str = matches!(elem_ty.as_ref(), CType::Char | CType::UChar)
                        && matches!(&inner_item.init, Initializer::Expr(Expr::StringLiteral(..)));
                    if !is_char_array_str {
                        let consumed = self.fill_flat_array_field_with_ptrs(
                            &inner_items[item_idx..], elem_ty, arr_size,
                            field_abs_offset, bytes, ptr_ranges,
                        );
                        item_idx += consumed;
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                }
            }

            if let Initializer::Expr(ref expr) = inner_item.init {
                self.write_expr_to_bytes_or_ptrs(
                    expr, &field.ty, field_abs_offset,
                    field.bit_offset, field.bit_width,
                    bytes, ptr_ranges,
                );
            } else if let Initializer::List(ref nested_items) = inner_item.init {
                self.fill_composite_or_array_with_ptrs(
                    nested_items, &field.ty, field_abs_offset, bytes, ptr_ranges,
                );
            }
            current_field_idx = field_idx + 1;
            item_idx += 1;
        }
    }

    /// Write a scalar expression value to either the byte buffer or ptr_ranges,
    /// depending on whether the target type is a pointer/function pointer.
    /// Handles: pointer fields (resolve as GlobalAddr), bitfields, regular scalars.
    fn write_expr_to_bytes_or_ptrs(
        &mut self,
        expr: &Expr,
        ty: &CType,
        offset: usize,
        bit_offset: Option<u32>,
        bit_width: Option<u32>,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let is_ptr = matches!(ty, CType::Pointer(_, _) | CType::Function(_));
        if is_ptr {
            if let Some(addr_init) = self.resolve_ptr_field_init(expr) {
                ptr_ranges.push((offset, addr_init));
            } else if let Some(val) = self.eval_const_expr(expr) {
                let ir_ty = IrType::from_ctype(ty);
                self.write_const_to_bytes(bytes, offset, &val, ir_ty);
            }
        } else if let (Some(bo), Some(bw)) = (bit_offset, bit_width) {
            let ir_ty = IrType::from_ctype(ty);
            let val = self.eval_init_scalar(&Initializer::Expr(expr.clone()));
            self.write_bitfield_to_bytes(bytes, offset, &val, ir_ty, bo, bw);
        } else if let Expr::StringLiteral(s, _) = expr {
            // String literal initializing a char array field (e.g., {"hello", &ptr} in a struct
            // with pointer members). write_string_to_bytes handles copying the string bytes.
            if let CType::Array(ref elem, Some(arr_size)) = ty {
                if matches!(elem.as_ref(), CType::Char | CType::UChar) {
                    Self::write_string_to_bytes(bytes, offset, s, *arr_size);
                }
            }
        } else if let Some(val) = self.eval_const_expr(expr) {
            let ir_ty = IrType::from_ctype(ty);
            self.write_const_to_bytes(bytes, offset, &val, ir_ty);
        }
    }

    /// Consume consecutive expression items from `items` to flat-initialize an array field.
    /// This implements brace elision (C11 6.7.9p17-21) for the bytes+ptrs global init path:
    /// when an expression item targets an array field, subsequent items fill later array elements.
    /// Returns the number of items consumed.
    fn fill_flat_array_field_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        elem_ty: &CType,
        arr_size: usize,
        field_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) -> usize {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let mut consumed = 0usize;
        while consumed < arr_size && consumed < items.len() {
            let item = &items[consumed];
            // Stop if we hit a designator (it targets a different field)
            if !item.designators.is_empty() && consumed > 0 {
                break;
            }
            // Stop if we hit a braced list (it's a new sub-aggregate)
            if matches!(&item.init, Initializer::List(_)) && consumed > 0 {
                break;
            }
            let elem_offset = field_offset + consumed * elem_size;
            if let Initializer::Expr(ref expr) = item.init {
                // For struct elements within the array, handle recursively
                if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                    // A single expression initializes the first field of a struct element
                    if !elem_layout.fields.is_empty() {
                        let f = &elem_layout.fields[0];
                        self.write_expr_to_bytes_or_ptrs(
                            expr, &f.ty, elem_offset + f.offset,
                            f.bit_offset, f.bit_width,
                            bytes, ptr_ranges,
                        );
                    }
                } else {
                    self.write_expr_to_bytes_or_ptrs(
                        expr, elem_ty, elem_offset, None, None, bytes, ptr_ranges,
                    );
                }
            } else if let Initializer::List(ref sub_items) = item.init {
                // Braced sub-list for first element
                self.fill_composite_or_array_with_ptrs(
                    sub_items, elem_ty, elem_offset, bytes, ptr_ranges,
                );
            }
            consumed += 1;
        }
        consumed.max(1) // Always consume at least one item
    }

    /// Fill an array of scalar/pointer elements into byte buffer + ptr_ranges.
    /// For pointer types, writes address relocations to ptr_ranges.
    /// For non-pointer types, writes constant values directly to the byte buffer.
    fn fill_scalar_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        elem_ty: &CType,
        base_offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        let elem_size = self.resolve_ctype_size(elem_ty);
        let has_ptrs = h::type_has_pointer_elements(elem_ty, &self.types);
        let elem_ir_ty = IrType::from_ctype(elem_ty);
        for (ai, item) in items.iter().enumerate() {
            let elem_offset = base_offset + ai * elem_size;
            if let Initializer::Expr(ref expr) = item.init {
                if has_ptrs {
                    self.write_expr_to_bytes_or_ptrs(
                        expr, elem_ty, elem_offset, None, None, bytes, ptr_ranges,
                    );
                } else if let Some(val) = self.eval_const_expr(expr) {
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
            }
        }
    }

    /// Fill a struct/union/array field into byte buffer + ptr_ranges, choosing the
    /// pointer-aware path or plain byte path based on whether the type contains pointers.
    fn fill_composite_or_array_with_ptrs(
        &mut self,
        items: &[InitializerItem],
        field_ty: &CType,
        offset: usize,
        bytes: &mut [u8],
        ptr_ranges: &mut Vec<(usize, GlobalInit)>,
    ) {
        if let Some(sub_layout) = self.get_struct_layout_for_ctype(field_ty) {
            if sub_layout.has_pointer_fields(&self.types) {
                self.fill_nested_struct_with_ptrs(items, &sub_layout, offset, bytes, ptr_ranges);
            } else {
                self.fill_struct_global_bytes(items, &sub_layout, bytes, offset);
            }
        } else if let CType::Array(elem_ty, _) = field_ty {
            // Array field: check if elements are structs/unions with pointer fields
            if let Some(elem_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                if elem_layout.has_pointer_fields(&self.types) {
                    // Array of structs with pointer fields: handle each element
                    let struct_size = elem_layout.size;
                    for (ai, item) in items.iter().enumerate() {
                        let elem_offset = offset + ai * struct_size;
                        match &item.init {
                            Initializer::List(sub_items) => {
                                self.fill_nested_struct_with_ptrs(
                                    sub_items, &elem_layout, elem_offset, bytes, ptr_ranges,
                                );
                            }
                            Initializer::Expr(expr) => {
                                // Single expression for first field of struct element
                                if !elem_layout.fields.is_empty() {
                                    let field = &elem_layout.fields[0];
                                    let field_offset = elem_offset + field.offset;
                                    self.write_expr_to_bytes_or_ptrs(
                                        expr, &field.ty, field_offset,
                                        field.bit_offset, field.bit_width,
                                        bytes, ptr_ranges,
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // Array of structs without pointer fields: byte serialization
                    let struct_size = elem_layout.size;
                    for (ai, item) in items.iter().enumerate() {
                        let elem_offset = offset + ai * struct_size;
                        if let Initializer::List(ref sub_items) = item.init {
                            self.fill_struct_global_bytes(sub_items, &elem_layout, bytes, elem_offset);
                        }
                    }
                }
            } else {
                // Array of non-composite elements (scalars, pointers, etc.)
                self.fill_scalar_array_with_ptrs(items, elem_ty, offset, bytes, ptr_ranges);
            }
        } else {
            // Non-composite, non-array field: write scalar values
            self.fill_scalar_array_with_ptrs(items, field_ty, offset, bytes, ptr_ranges);
        }
    }
}
