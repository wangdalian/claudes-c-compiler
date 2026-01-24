//! Byte-serialization portion of global initialization.
//!
//! This module contains the byte-level serialization methods used during global
//! variable initialization lowering. It handles writing constants, bitfields,
//! complex numbers, struct layouts, and array fills into byte buffers.

use crate::frontend::parser::ast::*;
use crate::ir::ir::*;
use crate::common::types::{IrType, StructLayout, CType};
use super::lowering::Lowerer;

/// Append `count` zero bytes (as `GlobalInit::Scalar(IrConst::I8(0))`) to `elements`.
/// Used throughout global initialization to emit padding and zero-fill.
pub(super) fn push_zero_bytes(elements: &mut Vec<GlobalInit>, count: usize) {
    for _ in 0..count {
        elements.push(GlobalInit::Scalar(IrConst::I8(0)));
    }
}

/// Result of filling an array/FAM field in fill_struct_global_bytes,
/// indicating how to advance the item index.
pub(super) struct ArrayFillResult {
    new_item_idx: usize,
    /// If true, caller should `continue` (skip the default field_idx update).
    skip_update: bool,
}

impl Lowerer {
    /// Recursively fill byte buffer for struct global initialization.
    /// Returns the number of initializer items consumed.
    ///
    /// Handles nested structs/unions, arrays (including multi-dimensional), string
    /// literals, designators, flexible array members, and bitfields.
    pub(super) fn fill_struct_global_bytes(
        &self,
        items: &[InitializerItem],
        layout: &StructLayout,
        bytes: &mut [u8],
        base_offset: usize,
    ) -> usize {
        let mut item_idx = 0usize;
        let mut current_field_idx = 0usize;

        while item_idx < items.len() {
            let item = &items[item_idx];

            let designator_name = match item.designators.first() {
                Some(Designator::Field(ref name)) => Some(name.as_str()),
                _ => None,
            };
            let array_start_idx = self.extract_index_designator(item, designator_name.is_some());
            let field_idx = match layout.resolve_init_field_idx(designator_name, current_field_idx) {
                Some(idx) => idx,
                None => {
                    if designator_name.is_some() { item_idx += 1; continue; }
                    break;
                }
            };
            let field_layout = &layout.fields[field_idx];
            let field_offset = base_offset + field_layout.offset;

            let has_nested_designator = item.designators.len() > 1
                && matches!(item.designators.first(), Some(Designator::Field(_)));

            match &field_layout.ty {
                // Nested designator into struct/union: drill into sub-composite
                CType::Struct(st) if has_nested_designator => {
                    self.fill_nested_designator_composite(
                        item, &StructLayout::for_struct(&st.fields), bytes, field_offset,
                    );
                    item_idx += 1;
                }
                CType::Union(st) if has_nested_designator => {
                    self.fill_nested_designator_composite(
                        item, &StructLayout::for_union(&st.fields), bytes, field_offset,
                    );
                    item_idx += 1;
                }
                // Nested designator into array: .field[idx] = val
                // After the designated element, continue consuming non-designated
                // items for subsequent array positions (C11 6.7.9p17).
                CType::Array(elem_ty, Some(arr_size)) if has_nested_designator => {
                    let arr_size = *arr_size;
                    let desig_idx = self.fill_nested_designator_array(item, elem_ty, arr_size, bytes, field_offset);
                    item_idx += 1;
                    // Continue consuming subsequent non-designated items for positions
                    // desig_idx+1, desig_idx+2, ... within the same array field
                    let elem_size = elem_ty.size();
                    let elem_ir_ty = IrType::from_ctype(elem_ty);
                    let mut ai = desig_idx + 1;
                    while ai < arr_size && item_idx < items.len() {
                        let next_item = &items[item_idx];
                        if !next_item.designators.is_empty() {
                            break;
                        }
                        if let Initializer::Expr(ref expr) = next_item.init {
                            let val = self.eval_expr_or_zero(expr);
                            let elem_offset = field_offset + ai * elem_size;
                            self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                        } else {
                            break;
                        }
                        item_idx += 1;
                        ai += 1;
                    }
                }
                // Struct or union field (non-nested designator)
                CType::Struct(st) => {
                    let sub_layout = StructLayout::for_struct(&st.fields);
                    item_idx += self.fill_composite_field(
                        &items[item_idx..], &sub_layout, bytes, field_offset,
                    );
                }
                CType::Union(st) => {
                    let sub_layout = StructLayout::for_union(&st.fields);
                    item_idx += self.fill_composite_field(
                        &items[item_idx..], &sub_layout, bytes, field_offset,
                    );
                }
                // Fixed-size array field
                CType::Array(elem_ty, Some(arr_size)) => {
                    let advanced = self.fill_array_field(
                        items, item_idx, elem_ty, *arr_size,
                        bytes, field_offset, array_start_idx,
                    );
                    if advanced.skip_update {
                        item_idx = advanced.new_item_idx;
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                    item_idx = advanced.new_item_idx;
                }
                // Flexible array member (FAM)
                CType::Array(elem_ty, None) => {
                    let advanced = self.fill_fam_field(
                        items, item_idx, elem_ty, bytes, field_offset,
                    );
                    if advanced.skip_update {
                        item_idx = advanced.new_item_idx;
                        current_field_idx = field_idx + 1;
                        continue;
                    }
                    item_idx = advanced.new_item_idx;
                }
                // Complex field: write {real, imag} pair to bytes
                CType::ComplexFloat | CType::ComplexDouble | CType::ComplexLongDouble => {
                    self.write_complex_field_to_bytes(bytes, field_offset, &field_layout.ty, &item.init);
                    item_idx += 1;
                }
                // Scalar field (possibly bitfield)
                _ => {
                    let field_ir_ty = IrType::from_ctype(&field_layout.ty);
                    let val = self.eval_init_scalar(&item.init);
                    if let (Some(bit_offset), Some(bit_width)) = (field_layout.bit_offset, field_layout.bit_width) {
                        self.write_bitfield_to_bytes(bytes, field_offset, &val, field_ir_ty, bit_offset, bit_width);
                    } else {
                        self.write_const_to_bytes(bytes, field_offset, &val, field_ir_ty);
                    }
                    item_idx += 1;
                }
            }
            current_field_idx = field_idx + 1;

            if layout.is_union && designator_name.is_none() {
                break;
            }
        }
        item_idx
    }

    // --- fill_struct_global_bytes helpers ---

    /// Extract an array index designator from an initializer item.
    pub(super) fn extract_index_designator(&self, item: &InitializerItem, has_field_desig: bool) -> Option<usize> {
        if has_field_desig {
            item.designators.iter().find_map(|d| {
                if let Designator::Index(ref idx_expr) = d {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                } else {
                    None
                }
            })
        } else {
            match item.designators.first() {
                Some(Designator::Index(ref idx_expr)) => {
                    self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                }
                _ => None,
            }
        }
    }

    /// Evaluate an expression to a constant, defaulting to zero.
    fn eval_expr_or_zero(&self, expr: &Expr) -> IrConst {
        self.eval_const_expr(expr).unwrap_or(IrConst::I64(0))
    }

    /// Evaluate an initializer to a scalar constant (handles both Expr and brace-wrapped List).
    pub(super) fn eval_init_scalar(&self, init: &Initializer) -> IrConst {
        match init {
            Initializer::Expr(expr) => self.eval_expr_or_zero(expr),
            Initializer::List(sub_items) => {
                sub_items.first()
                    .and_then(|first| {
                        if let Initializer::Expr(expr) = &first.init {
                            self.eval_const_expr(expr)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(IrConst::I64(0))
            }
        }
    }

    /// Write a string literal into a byte buffer at the given offset, with null terminator.
    pub(super) fn write_string_to_bytes(bytes: &mut [u8], offset: usize, s: &str, max_len: usize) {
        let str_bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
        for (i, &b) in str_bytes.iter().enumerate() {
            if i >= max_len { break; }
            if offset + i < bytes.len() {
                bytes[offset + i] = b;
            }
        }
        if str_bytes.len() < max_len && offset + str_bytes.len() < bytes.len() {
            bytes[offset + str_bytes.len()] = 0;
        }
    }

    /// Handle nested designator drilling into a struct/union sub-field (e.g., .a.b = val).
    pub(super) fn fill_nested_designator_composite(
        &self, item: &InitializerItem, sub_layout: &StructLayout,
        bytes: &mut [u8], field_offset: usize,
    ) {
        let sub_item = InitializerItem {
            designators: item.designators[1..].to_vec(),
            init: item.init.clone(),
        };
        self.fill_struct_global_bytes(&[sub_item], sub_layout, bytes, field_offset);
    }

    /// Process a nested designator into an array field (e.g., .field[idx] = val).
    /// Returns the array index that was designated.
    pub(super) fn fill_nested_designator_array(
        &self, item: &InitializerItem, elem_ty: &CType, arr_size: usize,
        bytes: &mut [u8], field_offset: usize,
    ) -> usize {
        let elem_size = elem_ty.size();
        let elem_ir_ty = IrType::from_ctype(elem_ty);

        // Collect all Index and Field designators after the first (which is Field("name"))
        // For .a[1][2].b we have designators: [Field("a"), Index(1), Index(2), Field("b")]
        // After stripping Field("a"), remaining: [Index(1), Index(2), Field("b")]
        let remaining = &item.designators[1..];

        // Find first Index designator (outer array index)
        let (first_idx_pos, idx) = remaining.iter().enumerate().find_map(|(i, d)| {
            if let Designator::Index(ref idx_expr) = d {
                self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()).map(|v| (i, v))
            } else {
                None
            }
        }).unwrap_or((0, 0));

        if idx >= arr_size {
            return idx;
        }

        let elem_offset = field_offset + idx * elem_size;

        // Check for further designators after the first Index
        let after_first_idx = &remaining[first_idx_pos + 1..];

        // Collect remaining Field designators for struct drilling
        let remaining_field_desigs: Vec<_> = after_first_idx.iter()
            .filter(|d| matches!(d, Designator::Field(_)))
            .cloned()
            .collect();

        // Collect remaining Index designators for multi-dimensional array drilling
        let remaining_index_desigs: Vec<_> = after_first_idx.iter()
            .filter(|d| matches!(d, Designator::Index(_)))
            .cloned()
            .collect();

        if !remaining_field_desigs.is_empty() {
            // Drill into struct element: .a[1].b = val
            if !remaining_index_desigs.is_empty() {
                // .a[1].b[2] - remaining has both field and index
                // Build designator list with fields and indices from after_first_idx
                let sub_desigs: Vec<_> = after_first_idx.iter().cloned().collect();
                if let Some(sub_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                    let sub_item = InitializerItem {
                        designators: sub_desigs,
                        init: item.init.clone(),
                    };
                    self.fill_struct_global_bytes(&[sub_item], &sub_layout, bytes, elem_offset);
                }
            } else {
                if let Some(sub_layout) = self.get_struct_layout_for_ctype(elem_ty) {
                    let sub_item = InitializerItem {
                        designators: remaining_field_desigs,
                        init: item.init.clone(),
                    };
                    self.fill_struct_global_bytes(&[sub_item], &sub_layout, bytes, elem_offset);
                }
            }
        } else if !remaining_index_desigs.is_empty() {
            // Multi-dimensional array: .a[1][2] = val
            // elem_ty is the inner array type (e.g., float[10] for float a[3][10])
            if let CType::Array(inner_elem_ty, Some(inner_size)) = elem_ty {
                // Recursively handle the inner array with the remaining index designators
                let inner_idx = remaining_index_desigs.iter().find_map(|d| {
                    if let Designator::Index(ref idx_expr) = d {
                        self.eval_const_expr(idx_expr).and_then(|c| c.to_usize())
                    } else {
                        None
                    }
                }).unwrap_or(0);
                if inner_idx < *inner_size {
                    let inner_elem_size = inner_elem_ty.size();
                    let inner_ir_ty = IrType::from_ctype(inner_elem_ty);
                    let inner_offset = elem_offset + inner_idx * inner_elem_size;

                    // Check for even more Index designators (3D+ arrays)
                    let further_indices: Vec<_> = remaining_index_desigs[1..].to_vec();
                    if !further_indices.is_empty() {
                        if let CType::Array(deeper_elem, Some(deeper_size)) = inner_elem_ty.as_ref() {
                            // Build sub-item with remaining indices as [Field("dummy"), Index(...)]
                            // Actually we can recursively call ourselves
                            let sub_item = InitializerItem {
                                designators: {
                                    let mut d = vec![Designator::Field(String::new())]; // dummy field
                                    d.extend(further_indices);
                                    d
                                },
                                init: item.init.clone(),
                            };
                            self.fill_nested_designator_array(
                                &sub_item, inner_elem_ty, *inner_size,
                                bytes, elem_offset,
                            );
                        }
                    } else if let Initializer::Expr(ref expr) = item.init {
                        if let Expr::StringLiteral(s, _) = expr {
                            if let CType::Array(deep_inner, Some(deep_size)) = inner_elem_ty.as_ref() {
                                if matches!(deep_inner.as_ref(), CType::Char | CType::UChar) {
                                    Self::write_string_to_bytes(bytes, inner_offset, s, *deep_size);
                                } else {
                                    let val = self.eval_expr_or_zero(expr);
                                    self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                                }
                            } else {
                                let val = self.eval_expr_or_zero(expr);
                                self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                            }
                        } else {
                            let val = self.eval_expr_or_zero(expr);
                            self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                        }
                    }
                }
            }
        } else if let Initializer::Expr(ref expr) = item.init {
            // No further designators - store value at elem_offset
            // Handle string literal initializing a char array element
            // (e.g., .a[1] = "abc" where a is char a[3][10])
            if let Expr::StringLiteral(s, _) = expr {
                if let CType::Array(inner, Some(inner_size)) = elem_ty {
                    if matches!(inner.as_ref(), CType::Char | CType::UChar) {
                        Self::write_string_to_bytes(bytes, elem_offset, s, *inner_size);
                    } else {
                        let val = self.eval_expr_or_zero(expr);
                        self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                    }
                } else if matches!(elem_ty, CType::Char | CType::UChar) {
                    let val = s.chars().next().map(|c| IrConst::I8(c as u8 as i8)).unwrap_or(IrConst::I8(0));
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                } else {
                    let val = self.eval_expr_or_zero(expr);
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
            } else {
                let val = self.eval_expr_or_zero(expr);
                self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
            }
        } else if let Initializer::List(ref sub_items) = item.init {
            // Handle list init for array element (e.g., .a[1] = {1,2,3})
            match elem_ty {
                CType::Array(inner_elem_ty, Some(inner_size)) => {
                    let inner_elem_size = inner_elem_ty.size();
                    let inner_ir_ty = IrType::from_ctype(inner_elem_ty);
                    for (si, sub_item) in sub_items.iter().enumerate() {
                        if si >= *inner_size { break; }
                        if let Initializer::Expr(ref expr) = sub_item.init {
                            let val = self.eval_expr_or_zero(expr);
                            let inner_offset = elem_offset + si * inner_elem_size;
                            self.write_const_to_bytes(bytes, inner_offset, &val, inner_ir_ty);
                        }
                    }
                }
                CType::Struct(ref st) => {
                    let sub_layout = StructLayout::for_struct(&st.fields);
                    self.fill_struct_global_bytes(sub_items, &sub_layout, bytes, elem_offset);
                }
                _ => {}
            }
        }
        idx
    }

    /// Fill a composite (struct/union) field from an initializer.
    /// Returns the number of items consumed.
    pub(super) fn fill_composite_field(
        &self, items: &[InitializerItem], sub_layout: &StructLayout,
        bytes: &mut [u8], field_offset: usize,
    ) -> usize {
        match &items[0].init {
            Initializer::List(sub_items) => {
                self.fill_struct_global_bytes(sub_items, sub_layout, bytes, field_offset);
                1
            }
            Initializer::Expr(_) => {
                let consumed = self.fill_struct_global_bytes(items, sub_layout, bytes, field_offset);
                if consumed == 0 { 1 } else { consumed }
            }
        }
    }

    /// Fill a fixed-size array field. Handles string literals, arrays of composites,
    /// arrays of scalars with designators, and flat initialization.
    pub(super) fn fill_array_field(
        &self,
        items: &[InitializerItem],
        item_idx: usize,
        elem_ty: &CType,
        arr_size: usize,
        bytes: &mut [u8],
        field_offset: usize,
        array_start_idx: Option<usize>,
    ) -> ArrayFillResult {
        let item = &items[item_idx];
        let elem_size = elem_ty.size();
        let elem_ir_ty = IrType::from_ctype(elem_ty);

        match &item.init {
            Initializer::List(sub_items) => {
                // Check for brace-wrapped string literal: { "hello" }
                if self.try_fill_string_literal_init(sub_items, elem_ty, arr_size, bytes, field_offset) {
                    return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: true };
                }
                if matches!(elem_ty, CType::Struct(_) | CType::Union(_)) {
                    self.fill_array_of_composites(sub_items, elem_ty, arr_size, elem_size, bytes, field_offset);
                } else if matches!(elem_ty, CType::Array(_, Some(_))) {
                    // Multi-dimensional array with braced initializers:
                    // check if sub_items have braces (sub-array inits) or are flat values
                    let has_braced = sub_items.iter().any(|si| matches!(si.init, Initializer::List(_)));
                    if has_braced {
                        // Sub-items have braces: each braced sub-item initializes a sub-array
                        self.fill_braced_multidim_array(sub_items, elem_ty, arr_size, elem_size, bytes, field_offset);
                    } else {
                        // All flat values: flatten across all dimensions
                        let (innermost_ty, total_elems) = Self::flatten_array_type(elem_ty, arr_size);
                        if matches!(innermost_ty, CType::Struct(_) | CType::Union(_)) {
                            // Innermost type is composite: fill using struct-aware logic
                            let inner_size = innermost_ty.size();
                            let inner_ir_ty = IrType::from_ctype(&innermost_ty);
                            self.fill_flat_array_of_composites(
                                sub_items, 0, &innermost_ty, total_elems, inner_size, inner_ir_ty,
                                bytes, field_offset, 0,
                            );
                        } else {
                            let scalar_size = innermost_ty.size();
                            let scalar_ir_ty = IrType::from_ctype(&innermost_ty);
                            self.fill_array_of_scalars(sub_items, total_elems, scalar_size, scalar_ir_ty, bytes, field_offset);
                        }
                    }
                } else {
                    self.fill_array_of_scalars(sub_items, arr_size, elem_size, elem_ir_ty, bytes, field_offset);
                }
                ArrayFillResult { new_item_idx: item_idx + 1, skip_update: false }
            }
            Initializer::Expr(expr) => {
                // String literal for char array
                if let Expr::StringLiteral(s, _) = expr {
                    if matches!(elem_ty, CType::Char | CType::UChar) {
                        Self::write_string_to_bytes(bytes, field_offset, s, arr_size);
                        return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: true };
                    }
                }
                // Flat init from consecutive items
                let start_ai = array_start_idx.unwrap_or(0);
                let new_idx = if matches!(elem_ty, CType::Struct(_) | CType::Union(_)) {
                    self.fill_flat_array_of_composites(
                        items, item_idx, elem_ty, arr_size, elem_size, elem_ir_ty,
                        bytes, field_offset, start_ai,
                    )
                } else if matches!(elem_ty, CType::Array(_, Some(_))) {
                    // Multi-dimensional array: flatten all dimensions and fill scalars
                    self.fill_flat_multidim_array(
                        items, item_idx, elem_ty, arr_size,
                        bytes, field_offset, start_ai,
                    )
                } else {
                    self.fill_flat_array_of_scalars(
                        items, item_idx, arr_size, elem_size, elem_ir_ty,
                        bytes, field_offset, start_ai,
                    )
                };
                ArrayFillResult { new_item_idx: new_idx, skip_update: true }
            }
        }
    }

    /// Fill a flexible array member (FAM) field.
    pub(super) fn fill_fam_field(
        &self,
        items: &[InitializerItem],
        item_idx: usize,
        elem_ty: &CType,
        bytes: &mut [u8],
        field_offset: usize,
    ) -> ArrayFillResult {
        let elem_size = elem_ty.size();
        let elem_ir_ty = IrType::from_ctype(elem_ty);
        let is_char_fam = matches!(elem_ty, CType::Char | CType::UChar);

        match &items[item_idx].init {
            Initializer::List(sub_items) => {
                // Check if this is a braced string literal initializing a char FAM
                // e.g., .chunk = {"hello"}
                if is_char_fam && sub_items.len() == 1 && sub_items[0].designators.is_empty() {
                    if let Initializer::Expr(Expr::StringLiteral(s, _)) = &sub_items[0].init {
                        Self::write_string_to_bytes(bytes, field_offset, s, bytes.len() - field_offset);
                        return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: false };
                    }
                }
                for (ai, sub_item) in sub_items.iter().enumerate() {
                    let elem_offset = field_offset + ai * elem_size;
                    if elem_offset + elem_size > bytes.len() { break; }
                    let val = self.eval_init_scalar(&sub_item.init);
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
                ArrayFillResult { new_item_idx: item_idx + 1, skip_update: false }
            }
            Initializer::Expr(expr) => {
                // String literal directly initializing a char FAM
                // e.g., struct { char *p; char data[]; } s = {0, "hello"};
                if is_char_fam {
                    if let Expr::StringLiteral(s, _) = expr {
                        Self::write_string_to_bytes(bytes, field_offset, s, bytes.len() - field_offset);
                        return ArrayFillResult { new_item_idx: item_idx + 1, skip_update: false };
                    }
                }
                let mut ai = 0usize;
                let val = self.eval_expr_or_zero(expr);
                let elem_offset = field_offset + ai * elem_size;
                if elem_offset + elem_size <= bytes.len() {
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
                ai += 1;
                let mut new_idx = item_idx + 1;
                while new_idx < items.len() {
                    let next_item = &items[new_idx];
                    if !next_item.designators.is_empty() { break; }
                    if let Initializer::Expr(e) = &next_item.init {
                        let val = self.eval_expr_or_zero(e);
                        let elem_offset = field_offset + ai * elem_size;
                        if elem_offset + elem_size <= bytes.len() {
                            self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                        }
                        ai += 1;
                        new_idx += 1;
                    } else {
                        break;
                    }
                }
                ArrayFillResult { new_item_idx: new_idx, skip_update: true }
            }
        }
    }

    /// Try to interpret a brace-wrapped init list as a string literal for a char array.
    /// Returns true if handled.
    pub(super) fn try_fill_string_literal_init(
        &self, sub_items: &[InitializerItem], elem_ty: &CType, arr_size: usize,
        bytes: &mut [u8], field_offset: usize,
    ) -> bool {
        if sub_items.len() == 1 && sub_items[0].designators.is_empty() {
            if let Initializer::Expr(Expr::StringLiteral(s, _)) = &sub_items[0].init {
                if matches!(elem_ty, CType::Char | CType::UChar) {
                    Self::write_string_to_bytes(bytes, field_offset, s, arr_size);
                    return true;
                }
            }
        }
        false
    }

    /// Fill an array of struct/union elements from a braced sub-item list.
    pub(super) fn fill_array_of_composites(
        &self, sub_items: &[InitializerItem], elem_ty: &CType,
        arr_size: usize, elem_size: usize, bytes: &mut [u8], field_offset: usize,
    ) {
        let sub_layout = self.get_composite_layout(elem_ty);
        let mut sub_idx = 0usize;
        let mut ai = 0usize;
        while ai < arr_size && sub_idx < sub_items.len() {
            let elem_offset = field_offset + ai * elem_size;
            match &sub_items[sub_idx].init {
                Initializer::List(inner_items) => {
                    self.fill_struct_global_bytes(inner_items, &sub_layout, bytes, elem_offset);
                    sub_idx += 1;
                }
                Initializer::Expr(_) => {
                    let consumed = self.fill_struct_global_bytes(&sub_items[sub_idx..], &sub_layout, bytes, elem_offset);
                    sub_idx += consumed;
                }
            }
            ai += 1;
        }
    }

    /// Fill an array of scalar elements from a braced sub-item list (with designator support).
    pub(super) fn fill_array_of_scalars(
        &self, sub_items: &[InitializerItem], arr_size: usize,
        elem_size: usize, elem_ir_ty: IrType, bytes: &mut [u8], field_offset: usize,
    ) {
        let mut ai = 0usize;
        for sub_item in sub_items.iter() {
            if let Some(Designator::Index(ref idx_expr)) = sub_item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            let val = self.eval_init_scalar(&sub_item.init);
            self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
            ai += 1;
        }
    }

    /// Fill flat init of an array of composites from consecutive items.
    /// Returns the new item_idx.
    pub(super) fn fill_flat_array_of_composites(
        &self, items: &[InitializerItem], mut item_idx: usize,
        elem_ty: &CType, arr_size: usize, elem_size: usize, elem_ir_ty: IrType,
        bytes: &mut [u8], field_offset: usize, start_ai: usize,
    ) -> usize {
        let sub_layout = self.get_composite_layout(elem_ty);
        if matches!(elem_ty, CType::Struct(_)) {
            for ai in start_ai..arr_size {
                if item_idx >= items.len() { break; }
                let elem_offset = field_offset + ai * elem_size;
                let consumed = self.fill_struct_global_bytes(&items[item_idx..], &sub_layout, bytes, elem_offset);
                item_idx += consumed;
            }
        } else {
            // Union: take one item per element
            for ai in start_ai..arr_size {
                if item_idx >= items.len() { break; }
                let elem_offset = field_offset + ai * elem_size;
                if let Initializer::Expr(e) = &items[item_idx].init {
                    let val = self.eval_expr_or_zero(e);
                    self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                }
                item_idx += 1;
            }
        }
        item_idx
    }

    /// Fill flat init of an array of scalars from consecutive items.
    /// Returns the new item_idx.
    pub(super) fn fill_flat_array_of_scalars(
        &self, items: &[InitializerItem], item_idx: usize,
        arr_size: usize, elem_size: usize, elem_ir_ty: IrType,
        bytes: &mut [u8], field_offset: usize, start_ai: usize,
    ) -> usize {
        let mut consumed = 0usize;
        let mut ai = start_ai;
        while ai < arr_size && (item_idx + consumed) < items.len() {
            let cur_item = &items[item_idx + consumed];
            if !cur_item.designators.is_empty() && consumed > 0 { break; }
            if let Initializer::Expr(e) = &cur_item.init {
                let val = self.eval_expr_or_zero(e);
                let elem_offset = field_offset + ai * elem_size;
                self.write_const_to_bytes(bytes, elem_offset, &val, elem_ir_ty);
                consumed += 1;
                ai += 1;
            } else {
                break;
            }
        }
        item_idx + consumed.max(1)
    }

    /// Fill a multi-dimensional array from a braced initializer where sub-items are
    /// themselves braced lists (e.g., { {1,2}, {3,4} } for int arr[2][2]).
    /// Each braced sub-item initializes one element of the outermost dimension.
    pub(super) fn fill_braced_multidim_array(
        &self, sub_items: &[InitializerItem], elem_ty: &CType,
        arr_size: usize, elem_size: usize,
        bytes: &mut [u8], field_offset: usize,
    ) {
        let mut ai = 0usize;
        let mut si = 0usize;
        while ai < arr_size && si < sub_items.len() {
            let sub_item = &sub_items[si];
            // Check for designator (e.g., [1] = {...})
            if let Some(Designator::Index(ref idx_expr)) = sub_item.designators.first() {
                if let Some(idx) = self.eval_const_expr(idx_expr).and_then(|c| c.to_usize()) {
                    ai = idx;
                }
            }
            if ai >= arr_size { break; }
            let elem_offset = field_offset + ai * elem_size;
            match &sub_item.init {
                Initializer::List(inner_items) => {
                    // Recursively fill the sub-array element
                    if let CType::Array(inner_elem_ty, Some(inner_size)) = elem_ty {
                        // Check if inner items are also braced (3D+ arrays)
                        let inner_has_braced = inner_items.iter().any(|ii| matches!(ii.init, Initializer::List(_)));
                        if matches!(inner_elem_ty.as_ref(), CType::Array(_, Some(_))) && inner_has_braced {
                            self.fill_braced_multidim_array(inner_items, inner_elem_ty, *inner_size, inner_elem_ty.size(), bytes, elem_offset);
                        } else {
                            // Innermost level: flatten and fill
                            let (innermost_ty, total) = Self::flatten_array_type(inner_elem_ty, *inner_size);
                            if matches!(innermost_ty, CType::Struct(_) | CType::Union(_)) {
                                let inner_size_bytes = innermost_ty.size();
                                let inner_ir_ty = IrType::from_ctype(&innermost_ty);
                                self.fill_flat_array_of_composites(
                                    inner_items, 0, &innermost_ty, total, inner_size_bytes, inner_ir_ty,
                                    bytes, elem_offset, 0,
                                );
                            } else {
                                let scalar_size = innermost_ty.size();
                                let scalar_ir_ty = IrType::from_ctype(&innermost_ty);
                                self.fill_array_of_scalars(inner_items, total, scalar_size, scalar_ir_ty, bytes, elem_offset);
                            }
                        }
                    }
                }
                Initializer::Expr(expr) => {
                    // Flat scalar initializing a sub-array element
                    let val = self.eval_const_expr(expr).unwrap_or(IrConst::I64(0));
                    let ir_ty = IrType::from_ctype(elem_ty);
                    self.write_const_to_bytes(bytes, elem_offset, &val, ir_ty);
                }
            }
            ai += 1;
            si += 1;
        }
    }

    /// Fill a multi-dimensional array (e.g., int arr[2][2][2]) from flat scalar initializers.
    /// Computes the innermost scalar type and total scalar count, then fills sequentially.
    pub(super) fn fill_flat_multidim_array(
        &self, items: &[InitializerItem], item_idx: usize,
        elem_ty: &CType, arr_size: usize,
        bytes: &mut [u8], field_offset: usize, start_ai: usize,
    ) -> usize {
        // Compute the innermost non-array type and total element count
        let (innermost_ty, total_elems) = Self::flatten_array_type(elem_ty, arr_size);

        if matches!(innermost_ty, CType::Struct(_) | CType::Union(_)) {
            // Innermost type is composite: use struct-aware filling
            let inner_size = innermost_ty.size();
            let inner_ir_ty = IrType::from_ctype(&innermost_ty);
            self.fill_flat_array_of_composites(
                items, item_idx, &innermost_ty, total_elems, inner_size, inner_ir_ty,
                bytes, field_offset, start_ai,
            )
        } else {
            // Innermost type is scalar: fill sequentially
            let scalar_size = innermost_ty.size();
            let scalar_ir_ty = IrType::from_ctype(&innermost_ty);

            let mut consumed = 0usize;
            let mut ai = start_ai;
            while ai < total_elems && (item_idx + consumed) < items.len() {
                let cur_item = &items[item_idx + consumed];
                if !cur_item.designators.is_empty() && consumed > 0 { break; }
                if let Initializer::Expr(e) = &cur_item.init {
                    let val = self.eval_const_expr(e).unwrap_or(IrConst::I64(0));
                    let elem_offset = field_offset + ai * scalar_size;
                    if elem_offset + scalar_size <= bytes.len() {
                        self.write_const_to_bytes(bytes, elem_offset, &val, scalar_ir_ty);
                    }
                    consumed += 1;
                    ai += 1;
                } else {
                    break;
                }
            }
            item_idx + consumed.max(1)
        }
    }

    /// Flatten a multi-dimensional array type to get the innermost scalar type
    /// and total element count.
    /// E.g., Array(Array(Int, 2), 3) -> (Int, 6)
    fn flatten_array_type(elem_ty: &CType, outer_size: usize) -> (CType, usize) {
        let mut total = outer_size;
        let mut ty = elem_ty.clone();
        loop {
            match ty {
                CType::Array(inner, Some(sz)) => {
                    total *= sz;
                    ty = (*inner).clone();
                }
                _ => break,
            }
        }
        (ty, total)
    }

    /// Get the StructLayout for a composite (struct or union) CType.
    pub(super) fn get_composite_layout(&self, ty: &CType) -> StructLayout {
        match ty {
            CType::Struct(st) => StructLayout::for_struct(&st.fields),
            CType::Union(st) => StructLayout::for_union(&st.fields),
            _ => unreachable!("get_composite_layout called on non-composite type"),
        }
    }

    /// Push a constant value as individual bytes into a compound init element list.
    pub(super) fn push_const_as_bytes(&self, elements: &mut Vec<GlobalInit>, val: &IrConst, size: usize) {
        let mut bytes = Vec::with_capacity(size);
        val.push_le_bytes(&mut bytes, size);
        for b in bytes {
            elements.push(GlobalInit::Scalar(IrConst::I8(b as i8)));
        }
    }

    /// Write an IrConst value to a byte buffer at the given offset using the field's IR type.
    pub(super) fn write_const_to_bytes(&self, bytes: &mut [u8], offset: usize, val: &IrConst, ty: IrType) {
        let coerced = val.coerce_to(ty);
        let size = ty.size();
        let mut le_buf = Vec::with_capacity(size);
        coerced.push_le_bytes(&mut le_buf, size);
        for (i, &b) in le_buf.iter().enumerate() {
            if offset + i < bytes.len() {
                bytes[offset + i] = b;
            }
        }
    }

    /// Write a bitfield value into a byte buffer at the given offset.
    /// Uses read-modify-write to pack the value at the correct bit position.
    pub(super) fn write_bitfield_to_bytes(&self, bytes: &mut [u8], offset: usize, val: &IrConst, ty: IrType, bit_offset: u32, bit_width: u32) {
        let int_val = val.to_u64().unwrap_or(0);

        let size = ty.size();
        let mask = if bit_width >= 64 { u64::MAX } else { (1u64 << bit_width) - 1 };
        let field_val = (int_val & mask) << bit_offset;
        let clear_mask = !(mask << bit_offset);

        // Read current storage unit value (little-endian)
        let mut current = 0u64;
        for i in 0..size {
            if offset + i < bytes.len() {
                current |= (bytes[offset + i] as u64) << (i * 8);
            }
        }

        // Modify: clear field bits and OR in new value
        let new_val = (current & clear_mask) | field_val;

        // Write back (little-endian)
        let le = new_val.to_le_bytes();
        for i in 0..size {
            if offset + i < bytes.len() {
                bytes[offset + i] = le[i];
            }
        }
    }

    /// Write a complex field ({real, imag} pair) to a byte buffer.
    /// Evaluates the initializer as a complex constant expression and writes
    /// both components at the correct offsets within the field.
    pub(super) fn write_complex_field_to_bytes(
        &self,
        bytes: &mut [u8],
        field_offset: usize,
        complex_ctype: &CType,
        init: &Initializer,
    ) {
        let comp_size = match complex_ctype {
            CType::ComplexFloat => 4,
            CType::ComplexDouble => 8,
            CType::ComplexLongDouble => 16,
            _ => 8,
        };

        // Try to extract (real, imag) from the initializer
        let (real, imag) = match init {
            Initializer::Expr(expr) => {
                self.eval_complex_const_public(expr).unwrap_or((0.0, 0.0))
            }
            Initializer::List(items) => {
                // {real, imag} or {real} (imag defaults to 0)
                let real_val = items.first().and_then(|item| {
                    if let Initializer::Expr(e) = &item.init {
                        self.eval_const_expr(e).and_then(|c| c.to_f64())
                    } else {
                        None
                    }
                }).unwrap_or(0.0);
                let imag_val = items.get(1).and_then(|item| {
                    if let Initializer::Expr(e) = &item.init {
                        self.eval_const_expr(e).and_then(|c| c.to_f64())
                    } else {
                        None
                    }
                }).unwrap_or(0.0);
                (real_val, imag_val)
            }
        };

        // Write real part at field_offset
        match complex_ctype {
            CType::ComplexFloat => {
                let real_const = IrConst::F32(real as f32);
                let imag_const = IrConst::F32(imag as f32);
                self.write_const_to_bytes(bytes, field_offset, &real_const, IrType::F32);
                self.write_const_to_bytes(bytes, field_offset + comp_size, &imag_const, IrType::F32);
            }
            CType::ComplexLongDouble => {
                let real_const = IrConst::LongDouble(real);
                let imag_const = IrConst::LongDouble(imag);
                self.write_const_to_bytes(bytes, field_offset, &real_const, IrType::F128);
                self.write_const_to_bytes(bytes, field_offset + comp_size, &imag_const, IrType::F128);
            }
            _ => {
                // ComplexDouble
                let real_const = IrConst::F64(real);
                let imag_const = IrConst::F64(imag);
                self.write_const_to_bytes(bytes, field_offset, &real_const, IrType::F64);
                self.write_const_to_bytes(bytes, field_offset + comp_size, &imag_const, IrType::F64);
            }
        }
    }

    /// Write a struct initializer list to a byte buffer at the given base offset.
    /// Handles nested struct fields recursively.
    pub(super) fn write_struct_init_to_bytes(
        &self,
        bytes: &mut [u8],
        base_offset: usize,
        items: &[InitializerItem],
        layout: &StructLayout,
    ) {
        let mut current_field_idx = 0usize;
        for item in items {
            let designator_name = match item.designators.first() {
                Some(Designator::Field(ref name)) => Some(name.as_str()),
                _ => None,
            };
            let field_idx = match layout.resolve_init_field_idx(designator_name, current_field_idx) {
                Some(idx) => idx,
                None => break,
            };

            let field_layout = &layout.fields[field_idx];
            let field_offset = base_offset + field_layout.offset;

            // Handle nested designators (e.g., .a.j = 2): drill into sub-struct
            let has_nested_designator = item.designators.len() > 1
                && matches!(item.designators.first(), Some(Designator::Field(_)));
            if has_nested_designator {
                if let Some(sub_layout) = self.get_struct_layout_for_ctype(&field_layout.ty) {
                    let sub_item = InitializerItem {
                        designators: item.designators[1..].to_vec(),
                        init: item.init.clone(),
                    };
                    self.write_struct_init_to_bytes(bytes, field_offset, &[sub_item], &sub_layout);
                }
                current_field_idx = field_idx + 1;
                continue;
            }

            // Complex fields: use specialized handler for {real, imag} pair
            if field_layout.ty.is_complex() {
                self.write_complex_field_to_bytes(bytes, field_offset, &field_layout.ty, &item.init);
                current_field_idx = field_idx + 1;
                continue;
            }

            match &item.init {
                Initializer::Expr(expr) => {
                    if let Expr::StringLiteral(s, _) = expr {
                        // String literal initializing a char array field
                        if let CType::Array(_, Some(arr_size)) = &field_layout.ty {
                            let s_bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
                            for (i, &b) in s_bytes.iter().enumerate() {
                                if i >= *arr_size { break; }
                                if field_offset + i < bytes.len() {
                                    bytes[field_offset + i] = b;
                                }
                            }
                            if s_bytes.len() < *arr_size && field_offset + s_bytes.len() < bytes.len() {
                                bytes[field_offset + s_bytes.len()] = 0;
                            }
                        }
                    } else {
                        let val = self.eval_expr_or_zero(expr);
                        let field_ir_ty = IrType::from_ctype(&field_layout.ty);
                        if let (Some(bit_offset), Some(bit_width)) = (field_layout.bit_offset, field_layout.bit_width) {
                            self.write_bitfield_to_bytes(bytes, field_offset, &val, field_ir_ty, bit_offset, bit_width);
                        } else {
                            self.write_const_to_bytes(bytes, field_offset, &val, field_ir_ty);
                        }
                    }
                }
                Initializer::List(sub_items) => {
                    // Nested struct/union: recurse if the field is a struct type
                    if let Some(sub_layout) = self.get_struct_layout_for_ctype(&field_layout.ty) {
                        self.write_struct_init_to_bytes(bytes, field_offset, sub_items, &sub_layout);
                    } else {
                        // Array field or other nested init: try to write elements sequentially
                        if let CType::Array(ref elem_ty, _) = field_layout.ty {
                            let elem_ir_ty = IrType::from_ctype(elem_ty);
                            let elem_size = elem_ty.size();
                            for (i, sub_item) in sub_items.iter().enumerate() {
                                if let Initializer::Expr(expr) = &sub_item.init {
                                    let val = self.eval_expr_or_zero(expr);
                                    self.write_const_to_bytes(bytes, field_offset + i * elem_size, &val, elem_ir_ty);
                                }
                            }
                        } else if let Some(first) = sub_items.first() {
                            if let Initializer::Expr(expr) = &first.init {
                                let val = self.eval_expr_or_zero(expr);
                                let field_ir_ty = IrType::from_ctype(&field_layout.ty);
                                self.write_const_to_bytes(bytes, field_offset, &val, field_ir_ty);
                            }
                        }
                    }
                }
            }

            current_field_idx = field_idx + 1;

            // For unions with flat init, only the first member can be initialized.
            // Designated inits can overwrite (last designator wins).
            if layout.is_union && designator_name.is_none() {
                break;
            }
        }
    }
}
