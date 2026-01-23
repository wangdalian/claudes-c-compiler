use crate::frontend::parser::ast::*;
use crate::ir::ir::*;
use crate::common::types::{IrType, StructField, StructLayout, CType};
use super::lowering::Lowerer;

impl Lowerer {
    /// Try to evaluate a constant expression at compile time.
    pub(super) fn eval_const_expr(&self, expr: &Expr) -> Option<IrConst> {
        match expr {
            Expr::IntLiteral(val, _) => {
                Some(IrConst::I64(*val))
            }
            Expr::CharLiteral(ch, _) => {
                Some(IrConst::I32(*ch as i32))
            }
            Expr::FloatLiteral(val, _) => {
                Some(IrConst::F64(*val))
            }
            Expr::UnaryOp(UnaryOp::Plus, inner, _) => {
                // Unary plus: identity, just evaluate the inner expression
                self.eval_const_expr(inner)
            }
            Expr::UnaryOp(UnaryOp::Neg, inner, _) => {
                if let Some(val) = self.eval_const_expr(inner) {
                    match val {
                        IrConst::I64(v) => Some(IrConst::I64(-v)),
                        IrConst::I32(v) => Some(IrConst::I32(-v)),
                        IrConst::F64(v) => Some(IrConst::F64(-v)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            Expr::BinaryOp(op, lhs, rhs, _) => {
                let l = self.eval_const_expr(lhs)?;
                let r = self.eval_const_expr(rhs)?;
                self.eval_const_binop(op, &l, &r)
            }
            Expr::UnaryOp(UnaryOp::BitNot, inner, _) => {
                if let Some(val) = self.eval_const_expr(inner) {
                    match val {
                        IrConst::I64(v) => Some(IrConst::I64(!v)),
                        IrConst::I32(v) => Some(IrConst::I32(!v)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            Expr::Cast(_, inner, _) => {
                // For now, just pass through casts in constant expressions
                self.eval_const_expr(inner)
            }
            Expr::Identifier(name, _) => {
                // Look up enum constants
                if let Some(&val) = self.enum_constants.get(name) {
                    Some(IrConst::I64(val))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Evaluate a constant binary operation.
    fn eval_const_binop(&self, op: &BinOp, lhs: &IrConst, rhs: &IrConst) -> Option<IrConst> {
        let l = self.const_to_i64(lhs)?;
        let r = self.const_to_i64(rhs)?;
        let result = match op {
            BinOp::Add => l.wrapping_add(r),
            BinOp::Sub => l.wrapping_sub(r),
            BinOp::Mul => l.wrapping_mul(r),
            BinOp::Div => if r != 0 { l.wrapping_div(r) } else { return None; },
            BinOp::Mod => if r != 0 { l.wrapping_rem(r) } else { return None; },
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            BinOp::BitXor => l ^ r,
            BinOp::Shl => l.wrapping_shl(r as u32),
            BinOp::Shr => l.wrapping_shr(r as u32),
            _ => return None,
        };
        Some(IrConst::I64(result))
    }

    /// Convert an IrConst to i64.
    pub(super) fn const_to_i64(&self, c: &IrConst) -> Option<i64> {
        match c {
            IrConst::I8(v) => Some(*v as i64),
            IrConst::I16(v) => Some(*v as i64),
            IrConst::I32(v) => Some(*v as i64),
            IrConst::I64(v) => Some(*v),
            IrConst::Zero => Some(0),
            _ => None,
        }
    }

    /// Get the zero constant for a given IR type.
    pub(super) fn zero_const(&self, ty: IrType) -> IrConst {
        match ty {
            IrType::I8 | IrType::U8 => IrConst::I8(0),
            IrType::I16 | IrType::U16 => IrConst::I16(0),
            IrType::I32 | IrType::U32 => IrConst::I32(0),
            IrType::I64 | IrType::U64 | IrType::Ptr => IrConst::I64(0),
            IrType::F32 => IrConst::F32(0.0),
            IrType::F64 => IrConst::F64(0.0),
            IrType::Void => IrConst::Zero,
        }
    }

    /// Get the IR type for an expression (best-effort, based on locals/globals info).
    pub(super) fn get_expr_type(&self, expr: &Expr) -> IrType {
        match expr {
            Expr::Identifier(name, _) => {
                if self.enum_constants.contains_key(name) {
                    return IrType::I32;
                }
                if let Some(info) = self.locals.get(name) {
                    return info.ty;
                }
                if let Some(ginfo) = self.globals.get(name) {
                    return ginfo.ty;
                }
                IrType::I64
            }
            Expr::ArraySubscript(base, _, _) => {
                // For multi-dim arrays, find the root identifier
                let root_name = self.get_array_root_name(expr);
                if let Some(name) = root_name {
                    if let Some(info) = self.locals.get(&name) {
                        if info.is_array {
                            return info.ty;
                        }
                    }
                    if let Some(ginfo) = self.globals.get(&name) {
                        if ginfo.is_array {
                            return self.ir_type_for_elem_size(*ginfo.array_dim_strides.last().unwrap_or(&8));
                        }
                    }
                }
                // Fallback: check direct base
                if let Expr::Identifier(name, _) = base.as_ref() {
                    if let Some(info) = self.locals.get(name) {
                        if info.is_array {
                            return info.ty;
                        }
                    }
                }
                IrType::I64
            }
            Expr::Deref(inner, _) => {
                // Dereference of pointer - try to infer the pointed-to type
                if let Expr::Identifier(name, _) = inner.as_ref() {
                    if let Some(info) = self.locals.get(name) {
                        if info.ty == IrType::Ptr {
                            return IrType::I64; // TODO: track pointed-to type
                        }
                    }
                }
                IrType::I64
            }
            Expr::MemberAccess(base_expr, field_name, _) => {
                let (_, field_ty) = self.resolve_member_access(base_expr, field_name);
                field_ty
            }
            Expr::PointerMemberAccess(base_expr, field_name, _) => {
                let (_, field_ty) = self.resolve_pointer_member_access(base_expr, field_name);
                field_ty
            }
            _ => IrType::I64,
        }
    }

    pub(super) fn type_spec_to_ir(&self, ts: &TypeSpecifier) -> IrType {
        match ts {
            TypeSpecifier::Void => IrType::Void,
            TypeSpecifier::Char => IrType::I8,
            TypeSpecifier::UnsignedChar => IrType::U8,
            TypeSpecifier::Short => IrType::I16,
            TypeSpecifier::UnsignedShort => IrType::U16,
            TypeSpecifier::Int | TypeSpecifier::Bool => IrType::I32,
            TypeSpecifier::UnsignedInt => IrType::U32,
            TypeSpecifier::Long | TypeSpecifier::LongLong => IrType::I64,
            TypeSpecifier::UnsignedLong | TypeSpecifier::UnsignedLongLong => IrType::U64,
            TypeSpecifier::Float => IrType::F32,
            TypeSpecifier::Double => IrType::F64,
            TypeSpecifier::Pointer(_) => IrType::Ptr,
            TypeSpecifier::Array(_, _) => IrType::Ptr,
            TypeSpecifier::Struct(_, _) | TypeSpecifier::Union(_, _) => IrType::Ptr,
            TypeSpecifier::Enum(_, _) => IrType::I32,
            TypeSpecifier::TypedefName(_) => IrType::I64, // TODO: resolve typedef
            TypeSpecifier::Signed => IrType::I32,
            TypeSpecifier::Unsigned => IrType::U32,
        }
    }

    pub(super) fn sizeof_type(&self, ts: &TypeSpecifier) -> usize {
        match ts {
            TypeSpecifier::Void => 0,
            TypeSpecifier::Char | TypeSpecifier::UnsignedChar => 1,
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => 2,
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt | TypeSpecifier::Bool => 4,
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong
            | TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => 8,
            TypeSpecifier::Float => 4,
            TypeSpecifier::Double => 8,
            TypeSpecifier::Pointer(_) => 8,
            TypeSpecifier::Array(elem, Some(size_expr)) => {
                let elem_size = self.sizeof_type(elem);
                if let Expr::IntLiteral(n, _) = size_expr.as_ref() {
                    return elem_size * (*n as usize);
                }
                elem_size
            }
            TypeSpecifier::Struct(_, Some(fields)) | TypeSpecifier::Union(_, Some(fields)) => {
                let is_union = matches!(ts, TypeSpecifier::Union(_, _));
                let struct_fields: Vec<StructField> = fields.iter().map(|f| {
                    StructField {
                        name: f.name.clone().unwrap_or_default(),
                        ty: self.type_spec_to_ctype(&f.type_spec),
                        bit_width: None,
                    }
                }).collect();
                let layout = if is_union {
                    StructLayout::for_union(&struct_fields)
                } else {
                    StructLayout::for_struct(&struct_fields)
                };
                layout.size
            }
            TypeSpecifier::Struct(Some(tag), None) => {
                let key = format!("struct.{}", tag);
                self.struct_layouts.get(&key).map(|l| l.size).unwrap_or(8)
            }
            TypeSpecifier::Union(Some(tag), None) => {
                let key = format!("union.{}", tag);
                self.struct_layouts.get(&key).map(|l| l.size).unwrap_or(8)
            }
            _ => 8,
        }
    }

    /// Compute sizeof for an expression operand (sizeof expr).
    /// Returns the size in bytes of the expression's type.
    pub(super) fn sizeof_expr(&self, expr: &Expr) -> usize {
        match expr {
            // Integer literal: type int (4 bytes)
            Expr::IntLiteral(_, _) => 4,
            // Float literal: type double (8 bytes) by default in C
            Expr::FloatLiteral(_, _) => 8,
            // Char literal: type int in C (4 bytes)
            Expr::CharLiteral(_, _) => 4,
            // String literal: array of char, size = length + 1 (null terminator)
            Expr::StringLiteral(s, _) => s.len() + 1,

            // Variable: look up its alloc_size or type
            Expr::Identifier(name, _) => {
                if let Some(info) = self.locals.get(name) {
                    if info.is_array || info.is_struct {
                        return info.alloc_size;
                    }
                    return info.ty.size();
                }
                if let Some(ginfo) = self.globals.get(name) {
                    if ginfo.is_array || ginfo.is_struct {
                        for g in &self.module.globals {
                            if g.name == *name {
                                return g.size;
                            }
                        }
                    }
                    return ginfo.ty.size();
                }
                4 // default: int
            }

            // Dereference: element type size
            Expr::Deref(inner, _) => {
                if let Expr::Identifier(name, _) = inner.as_ref() {
                    if let Some(info) = self.locals.get(name) {
                        if info.elem_size > 0 {
                            return info.elem_size;
                        }
                    }
                    if let Some(ginfo) = self.globals.get(name) {
                        if ginfo.elem_size > 0 {
                            return ginfo.elem_size;
                        }
                    }
                }
                8 // TODO: better type tracking for nested derefs
            }

            // Array subscript: element type size
            Expr::ArraySubscript(base, _, _) => {
                if let Expr::Identifier(name, _) = base.as_ref() {
                    if let Some(info) = self.locals.get(name) {
                        if info.elem_size > 0 {
                            return info.elem_size;
                        }
                    }
                    if let Some(ginfo) = self.globals.get(name) {
                        if ginfo.elem_size > 0 {
                            return ginfo.elem_size;
                        }
                    }
                }
                4 // default: int element
            }

            // sizeof(sizeof(...)) -> size_t = 8 on 64-bit
            Expr::Sizeof(_, _) => 8,

            // Cast: size of the target type
            Expr::Cast(target_type, _, _) => {
                self.sizeof_type(target_type)
            }

            // Member access: member field size
            Expr::MemberAccess(base_expr, field_name, _) => {
                let (_, field_ty) = self.resolve_member_access(base_expr, field_name);
                field_ty.size()
            }
            Expr::PointerMemberAccess(base_expr, field_name, _) => {
                let (_, field_ty) = self.resolve_pointer_member_access(base_expr, field_name);
                field_ty.size()
            }

            // Address-of: pointer (8 bytes)
            Expr::AddressOf(_, _) => 8,

            // Unary operations
            Expr::UnaryOp(op, inner, _) => {
                match op {
                    UnaryOp::LogicalNot => 4, // result is int
                    UnaryOp::Neg | UnaryOp::Plus | UnaryOp::BitNot => {
                        self.sizeof_expr(inner).max(4) // integer promotion
                    }
                    UnaryOp::PreInc | UnaryOp::PreDec => self.sizeof_expr(inner),
                }
            }

            // Postfix operations preserve the operand type
            Expr::PostfixOp(_, inner, _) => self.sizeof_expr(inner),

            // Binary operations
            Expr::BinaryOp(op, lhs, rhs, _) => {
                match op {
                    // Comparison/logical: result is int
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le
                    | BinOp::Gt | BinOp::Ge | BinOp::LogicalAnd | BinOp::LogicalOr => 4,
                    // Arithmetic/bitwise: usual arithmetic conversions
                    _ => {
                        let ls = self.sizeof_expr(lhs);
                        let rs = self.sizeof_expr(rhs);
                        ls.max(rs).max(4) // integer promotion
                    }
                }
            }

            // Conditional: common type of both branches
            Expr::Conditional(_, then_e, else_e, _) => {
                let ts = self.sizeof_expr(then_e);
                let es = self.sizeof_expr(else_e);
                ts.max(es)
            }

            // Assignment: type of the left-hand side
            Expr::Assign(lhs, _, _) | Expr::CompoundAssign(_, lhs, _, _) => {
                self.sizeof_expr(lhs)
            }

            // Comma: type of the right expression
            Expr::Comma(_, rhs, _) => {
                self.sizeof_expr(rhs)
            }

            // Function call: default to int (4 bytes)
            Expr::FunctionCall(_, _, _) => 4,

            // Compound literal: size of the type
            Expr::CompoundLiteral(ts, _, _) => self.sizeof_type(ts),

            // Default
            _ => 4,
        }
    }

    /// Get the element size for a compound literal type.
    /// For arrays, returns the element size; for scalars/structs, returns the full size.
    pub(super) fn compound_literal_elem_size(&self, ts: &TypeSpecifier) -> usize {
        match ts {
            TypeSpecifier::Array(elem, _) => self.sizeof_type(elem),
            _ => self.sizeof_type(ts),
        }
    }

    /// Compute allocation info for a declaration.
    /// Returns (alloc_size, elem_size, is_array, is_pointer, array_dim_strides).
    /// For multi-dimensional arrays like int a[2][3], array_dim_strides = [12, 4]
    /// (stride for dim 0 = 3*4=12, stride for dim 1 = 4).
    pub(super) fn compute_decl_info(&self, ts: &TypeSpecifier, derived: &[DerivedDeclarator]) -> (usize, usize, bool, bool, Vec<usize>) {
        // Check for pointer declarators
        let has_pointer = derived.iter().any(|d| matches!(d, DerivedDeclarator::Pointer));
        let has_array = derived.iter().any(|d| matches!(d, DerivedDeclarator::Array(_)));

        // If both pointer and array, treat as pointer (e.g., int (*p)[5])
        if has_pointer && !has_array {
            let elem_size = self.sizeof_type(ts);
            return (8, elem_size, false, true, vec![]);
        }
        if has_pointer && has_array {
            // Pointer to array (e.g., int (*p)[5]) - treat as pointer
            let elem_size = self.sizeof_type(ts);
            return (8, elem_size, false, true, vec![]);
        }

        // Check for array declarators - collect all dimensions
        let array_dims: Vec<Option<usize>> = derived.iter().filter_map(|d| {
            if let DerivedDeclarator::Array(size_expr) = d {
                let dim = size_expr.as_ref().and_then(|e| {
                    if let Expr::IntLiteral(n, _) = e.as_ref() {
                        Some(*n as usize)
                    } else {
                        None
                    }
                });
                Some(dim)
            } else {
                None
            }
        }).collect();

        if !array_dims.is_empty() {
            let base_elem_size = self.sizeof_type(ts).max(1);

            // Also account for array dimensions in the type specifier itself
            // e.g., if type is Array(Array(Int, 3), 2) from the parser
            let type_dims = self.collect_type_array_dims(ts);

            // Combine: derived dims come first (outermost), then type dims
            let all_dims: Vec<usize> = array_dims.iter().map(|d| d.unwrap_or(256))
                .chain(type_dims.iter().copied())
                .collect();

            // Compute total size = product of all dims * base_elem_size
            let total: usize = all_dims.iter().product::<usize>() * base_elem_size;

            // Compute strides: stride[i] = product of dims[i+1..] * base_elem_size
            let mut strides = Vec::with_capacity(all_dims.len());
            for i in 0..all_dims.len() {
                let stride: usize = all_dims[i+1..].iter().product::<usize>() * base_elem_size;
                strides.push(stride);
            }

            // elem_size is the stride of the outermost dimension (for 1D compat, it's base_elem_size)
            let elem_size = if strides.len() > 1 { strides[0] } else { base_elem_size };

            return (total, elem_size, true, false, strides);
        }

        // For struct/union types, use their layout size
        if let Some(layout) = self.get_struct_layout_for_type(ts) {
            return (layout.size, 0, false, false, vec![]);
        }

        // Regular scalar - we use 8-byte slots for each stack value
        (8, 0, false, false, vec![])
    }

    /// Collect array dimensions from nested Array type specifiers.
    /// For Array(Array(Int, 3), 2), returns [2, 3] (but we skip the outermost
    /// since that comes from the derived declarator).
    fn collect_type_array_dims(&self, ts: &TypeSpecifier) -> Vec<usize> {
        let mut dims = Vec::new();
        let mut current = ts;
        loop {
            if let TypeSpecifier::Array(inner, Some(size_expr)) = current {
                if let Expr::IntLiteral(n, _) = size_expr.as_ref() {
                    dims.push(*n as usize);
                }
                current = inner.as_ref();
            } else {
                break;
            }
        }
        dims
    }

    /// Map an element size in bytes to an appropriate IrType.
    pub(super) fn ir_type_for_elem_size(&self, size: usize) -> IrType {
        match size {
            1 => IrType::I8,
            2 => IrType::I16,
            4 => IrType::I32,
            8 => IrType::I64,
            _ => IrType::I64,
        }
    }

    /// Convert a TypeSpecifier to CType (for struct layout computation).
    pub(super) fn type_spec_to_ctype(&self, ts: &TypeSpecifier) -> CType {
        match ts {
            TypeSpecifier::Void => CType::Void,
            TypeSpecifier::Char => CType::Char,
            TypeSpecifier::UnsignedChar => CType::UChar,
            TypeSpecifier::Short => CType::Short,
            TypeSpecifier::UnsignedShort => CType::UShort,
            TypeSpecifier::Int | TypeSpecifier::Signed => CType::Int,
            TypeSpecifier::UnsignedInt | TypeSpecifier::Unsigned | TypeSpecifier::Bool => CType::UInt,
            TypeSpecifier::Long => CType::Long,
            TypeSpecifier::UnsignedLong => CType::ULong,
            TypeSpecifier::LongLong => CType::LongLong,
            TypeSpecifier::UnsignedLongLong => CType::ULongLong,
            TypeSpecifier::Float => CType::Float,
            TypeSpecifier::Double => CType::Double,
            TypeSpecifier::Pointer(inner) => CType::Pointer(Box::new(self.type_spec_to_ctype(inner))),
            TypeSpecifier::Array(elem, size_expr) => {
                let elem_ctype = self.type_spec_to_ctype(elem);
                let size = size_expr.as_ref().and_then(|e| {
                    if let Expr::IntLiteral(n, _) = e.as_ref() {
                        Some(*n as usize)
                    } else {
                        None
                    }
                });
                CType::Array(Box::new(elem_ctype), size)
            }
            TypeSpecifier::Struct(name, fields) => {
                if let Some(fs) = fields {
                    let struct_fields: Vec<StructField> = fs.iter().map(|f| {
                        StructField {
                            name: f.name.clone().unwrap_or_default(),
                            ty: self.type_spec_to_ctype(&f.type_spec),
                            bit_width: None,
                        }
                    }).collect();
                    CType::Struct(crate::common::types::StructType {
                        name: name.clone(),
                        fields: struct_fields,
                    })
                } else if let Some(tag) = name {
                    // Forward reference: look up cached layout to get field info
                    let key = format!("struct.{}", tag);
                    if let Some(layout) = self.struct_layouts.get(&key) {
                        let struct_fields: Vec<StructField> = layout.fields.iter().map(|f| {
                            StructField {
                                name: f.name.clone(),
                                ty: f.ty.clone(),
                                bit_width: None,
                            }
                        }).collect();
                        CType::Struct(crate::common::types::StructType {
                            name: Some(tag.clone()),
                            fields: struct_fields,
                        })
                    } else {
                        // Unknown struct - return empty
                        CType::Struct(crate::common::types::StructType {
                            name: Some(tag.clone()),
                            fields: Vec::new(),
                        })
                    }
                } else {
                    CType::Struct(crate::common::types::StructType {
                        name: None,
                        fields: Vec::new(),
                    })
                }
            }
            TypeSpecifier::Union(name, fields) => {
                if let Some(fs) = fields {
                    let struct_fields: Vec<StructField> = fs.iter().map(|f| {
                        StructField {
                            name: f.name.clone().unwrap_or_default(),
                            ty: self.type_spec_to_ctype(&f.type_spec),
                            bit_width: None,
                        }
                    }).collect();
                    CType::Union(crate::common::types::StructType {
                        name: name.clone(),
                        fields: struct_fields,
                    })
                } else if let Some(tag) = name {
                    let key = format!("union.{}", tag);
                    if let Some(layout) = self.struct_layouts.get(&key) {
                        let struct_fields: Vec<StructField> = layout.fields.iter().map(|f| {
                            StructField {
                                name: f.name.clone(),
                                ty: f.ty.clone(),
                                bit_width: None,
                            }
                        }).collect();
                        CType::Union(crate::common::types::StructType {
                            name: Some(tag.clone()),
                            fields: struct_fields,
                        })
                    } else {
                        CType::Union(crate::common::types::StructType {
                            name: Some(tag.clone()),
                            fields: Vec::new(),
                        })
                    }
                } else {
                    CType::Union(crate::common::types::StructType {
                        name: None,
                        fields: Vec::new(),
                    })
                }
            }
            TypeSpecifier::Enum(_, _) => CType::Int, // enums are int-sized
            TypeSpecifier::TypedefName(_) => CType::Int, // TODO: resolve typedef
        }
    }

    /// Convert a CType to IrType.
    pub(super) fn ctype_to_ir(&self, ctype: &CType) -> IrType {
        IrType::from_ctype(ctype)
    }
}
