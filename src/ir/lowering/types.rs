use crate::frontend::parser::ast::*;
use crate::ir::ir::*;
use crate::common::types::{IrType, StructField, StructLayout, CType};
use crate::common::source::Span;
use super::lowering::{Lowerer, FuncSig};

impl Lowerer {

    /// Resolve a TypeSpecifier, following typedef chains.
    /// Returns the underlying TypeSpecifier (non-TypedefName).
    pub(super) fn resolve_type_spec<'a>(&'a self, ts: &'a TypeSpecifier) -> &'a TypeSpecifier {
        let mut current = ts;
        // Follow typedef chains (limit depth to prevent infinite loops)
        for _ in 0..32 {
            match current {
                TypeSpecifier::TypedefName(name) => {
                    if let Some(resolved) = self.types.typedefs.get(name) {
                        current = resolved;
                        continue;
                    }
                }
                TypeSpecifier::TypeofType(inner) => {
                    // typeof(type-name): resolve the inner type directly
                    current = inner;
                    continue;
                }
                _ => {}
            }
            break;
        }
        current
    }

    /// Resolve typeof(expr) to a concrete TypeSpecifier by analyzing the expression type.
    /// Returns a new TypeSpecifier if the input is Typeof, otherwise returns a clone of the input.
    pub(super) fn resolve_typeof(&self, ts: &TypeSpecifier) -> TypeSpecifier {
        match ts {
            TypeSpecifier::Typeof(expr) => {
                if let Some(ctype) = self.get_expr_ctype(expr) {
                    Self::ctype_to_type_spec(&ctype)
                } else {
                    TypeSpecifier::Int // fallback
                }
            }
            TypeSpecifier::TypeofType(inner) => {
                // Recursively resolve inner
                self.resolve_typeof(inner)
            }
            TypeSpecifier::TypedefName(name) => {
                if let Some(resolved) = self.types.typedefs.get(name) {
                    // Check if the typedef itself is a typeof
                    self.resolve_typeof(resolved)
                } else {
                    ts.clone()
                }
            }
            other => other.clone(),
        }
    }

    /// Convert a CType back to a TypeSpecifier (for typeof resolution).
    fn ctype_to_type_spec(ctype: &CType) -> TypeSpecifier {
        match ctype {
            CType::Void => TypeSpecifier::Void,
            CType::Bool => TypeSpecifier::Bool,
            CType::Char => TypeSpecifier::Char,
            CType::UChar => TypeSpecifier::UnsignedChar,
            CType::Short => TypeSpecifier::Short,
            CType::UShort => TypeSpecifier::UnsignedShort,
            CType::Int => TypeSpecifier::Int,
            CType::UInt => TypeSpecifier::UnsignedInt,
            CType::Long => TypeSpecifier::Long,
            CType::ULong => TypeSpecifier::UnsignedLong,
            CType::LongLong => TypeSpecifier::LongLong,
            CType::ULongLong => TypeSpecifier::UnsignedLongLong,
            CType::Int128 => TypeSpecifier::Int128,
            CType::UInt128 => TypeSpecifier::UnsignedInt128,
            CType::Float => TypeSpecifier::Float,
            CType::Double => TypeSpecifier::Double,
            CType::LongDouble => TypeSpecifier::LongDouble,
            CType::ComplexFloat => TypeSpecifier::ComplexFloat,
            CType::ComplexDouble => TypeSpecifier::ComplexDouble,
            CType::ComplexLongDouble => TypeSpecifier::ComplexLongDouble,
            CType::Pointer(inner) => TypeSpecifier::Pointer(Box::new(Self::ctype_to_type_spec(inner))),
            CType::Array(elem, size) => TypeSpecifier::Array(
                Box::new(Self::ctype_to_type_spec(elem)),
                size.map(|s| Box::new(Expr::IntLiteral(s as i64, crate::common::source::Span::dummy()))),
            ),
            CType::Struct(key) => {
                // Extract tag name from key (e.g., "struct.Foo" -> "Foo")
                if let Some(tag) = key.strip_prefix("struct.") {
                    TypeSpecifier::Struct(Some(tag.to_string()), None, false, None, None)
                } else {
                    TypeSpecifier::Int // anonymous struct fallback
                }
            }
            CType::Union(key) => {
                // Extract tag name from key (e.g., "union.Bar" -> "Bar")
                if let Some(tag) = key.strip_prefix("union.") {
                    TypeSpecifier::Union(Some(tag.to_string()), None, false, None, None)
                } else {
                    TypeSpecifier::Int // anonymous union fallback
                }
            }
            CType::Enum(et) => {
                TypeSpecifier::Enum(et.name.clone(), None)
            }
            CType::Function { .. } => TypeSpecifier::Int, // function type fallback
        }
    }

    /// Pre-populate typedef mappings for builtin/standard types.
    pub(super) fn seed_builtin_typedefs(&mut self) {
        use TypeSpecifier::*;
        let builtins: &[(&str, TypeSpecifier)] = &[
            // <stddef.h>
            ("size_t", UnsignedLong),
            ("ssize_t", Long),
            ("ptrdiff_t", Long),
            ("wchar_t", Int),
            ("wint_t", UnsignedInt),
            // <stdint.h> - exact width types
            ("int8_t", Char),
            ("int16_t", Short),
            ("int32_t", Int),
            ("int64_t", Long),
            ("uint8_t", UnsignedChar),
            ("uint16_t", UnsignedShort),
            ("uint32_t", UnsignedInt),
            ("uint64_t", UnsignedLong),
            ("intptr_t", Long),
            ("uintptr_t", UnsignedLong),
            ("intmax_t", Long),
            ("uintmax_t", UnsignedLong),
            // least types
            ("int_least8_t", Char),
            ("int_least16_t", Short),
            ("int_least32_t", Int),
            ("int_least64_t", Long),
            ("uint_least8_t", UnsignedChar),
            ("uint_least16_t", UnsignedShort),
            ("uint_least32_t", UnsignedInt),
            ("uint_least64_t", UnsignedLong),
            // fast types
            ("int_fast8_t", Char),
            ("int_fast16_t", Long),
            ("int_fast32_t", Long),
            ("int_fast64_t", Long),
            ("uint_fast8_t", UnsignedChar),
            ("uint_fast16_t", UnsignedLong),
            ("uint_fast32_t", UnsignedLong),
            ("uint_fast64_t", UnsignedLong),
            // <signal.h>
            ("sig_atomic_t", Int),
            // <time.h>
            ("time_t", Long),
            ("clock_t", Long),
            ("timer_t", Pointer(Box::new(Void))),
            ("clockid_t", Int),
            // <sys/types.h>
            ("off_t", Long),
            ("pid_t", Int),
            ("uid_t", UnsignedInt),
            ("gid_t", UnsignedInt),
            ("mode_t", UnsignedInt),
            ("dev_t", UnsignedLong),
            ("ino_t", UnsignedLong),
            ("nlink_t", UnsignedLong),
            ("blksize_t", Long),
            ("blkcnt_t", Long),
            // GNU/glibc common
            ("ulong", UnsignedLong),
            ("ushort", UnsignedShort),
            ("uint", UnsignedInt),
            ("__u8", UnsignedChar),
            ("__u16", UnsignedShort),
            ("__u32", UnsignedInt),
            ("__u64", UnsignedLong),
            ("__s8", Char),
            ("__s16", Short),
            ("__s32", Int),
            ("__s64", Long),
            // <stdarg.h> - va_list is an array type. Size varies by arch:
            //   x86-64: 24 bytes (gp_offset, fp_offset, overflow_arg_area, reg_save_area)
            //   AArch64: 32 bytes (__stack, __gr_top, __vr_top, __gr_offs, __vr_offs)
            //   RISC-V: 8 bytes (just a pointer)
            // va_list decays to a pointer when passed to functions (it's an array type).
            // (target-dependent va_list definitions are added below)
            // <locale.h>
            ("locale_t", Pointer(Box::new(Void))),
            // <pthread.h> - opaque types, treat as unsigned long or pointer
            ("pthread_t", UnsignedLong),
            ("pthread_mutex_t", Pointer(Box::new(Void))),
            ("pthread_cond_t", Pointer(Box::new(Void))),
            ("pthread_key_t", UnsignedInt),
            ("pthread_attr_t", Pointer(Box::new(Void))),
            ("pthread_once_t", Int),
            ("pthread_mutexattr_t", Pointer(Box::new(Void))),
            ("pthread_condattr_t", Pointer(Box::new(Void))),
            // <setjmp.h>
            ("jmp_buf", Pointer(Box::new(Void))),
            ("sigjmp_buf", Pointer(Box::new(Void))),
            // <stdio.h>
            ("FILE", Pointer(Box::new(Void))),
            ("fpos_t", Long),
            // <dirent.h>
            ("DIR", Pointer(Box::new(Void))),
        ];
        for (name, ts) in builtins {
            self.types.typedefs.insert(name.to_string(), ts.clone());
        }
        // Target-dependent va_list definition.
        // On RISC-V, va_list is just `void *` (a pointer passed by value).
        // On x86-64 and AArch64, va_list is an array/struct type that decays to
        // a pointer when passed to functions.
        use crate::backend::Target;
        let va_list_type = match self.target {
            Target::Riscv64 => {
                // RISC-V: va_list = void * (8 bytes, passed by value)
                Pointer(Box::new(Void))
            }
            Target::Aarch64 => {
                // AArch64: va_list is a 32-byte struct, represented as char[32]
                Array(Box::new(Char), Some(Box::new(Expr::IntLiteral(32, Span::dummy()))))
            }
            Target::X86_64 => {
                // x86-64: va_list is __va_list_tag[1], 24 bytes, represented as char[24]
                Array(Box::new(Char), Some(Box::new(Expr::IntLiteral(24, Span::dummy()))))
            }
        };
        self.types.typedefs.insert("va_list".to_string(), va_list_type.clone());
        self.types.typedefs.insert("__builtin_va_list".to_string(), va_list_type.clone());
        self.types.typedefs.insert("__gnuc_va_list".to_string(), va_list_type);
        // Also add the __u_char etc. POSIX internal names
        let posix_extras: &[(&str, TypeSpecifier)] = &[
            ("__u_char", UnsignedChar),
            ("__u_short", UnsignedShort),
            ("__u_int", UnsignedInt),
            ("__u_long", UnsignedLong),
            ("__int8_t", Char),
            ("__int16_t", Short),
            ("__int32_t", Int),
            ("__int64_t", Long),
            ("__uint8_t", UnsignedChar),
            ("__uint16_t", UnsignedShort),
            ("__uint32_t", UnsignedInt),
            ("__uint64_t", UnsignedLong),
        ];
        for (name, ts) in posix_extras {
            self.types.typedefs.insert(name.to_string(), ts.clone());
        }
    }

    /// Seed known libc math function signatures for correct calling convention.
    /// Without these, calls like atanf(1) would pass integer args in %rdi instead of %xmm0.
    /// Helper to insert a builtin function signature into func_meta.
    fn insert_builtin_sig(&mut self, name: &str, return_type: IrType, param_types: Vec<IrType>, param_ctypes: Vec<CType>) {
        self.func_meta.sigs.insert(name.to_string(), FuncSig {
            return_type,
            return_ctype: None,
            param_types,
            param_ctypes,
            param_bool_flags: Vec::new(),
            is_variadic: false,
            sret_size: None,
            two_reg_ret_size: None,
            param_struct_sizes: Vec::new(),
        });
    }

    pub(super) fn seed_libc_math_functions(&mut self) {
        use IrType::*;
        // float func(float) - single-precision math
        let f_f: &[&str] = &[
            "sinf", "cosf", "tanf", "asinf", "acosf", "atanf",
            "sinhf", "coshf", "tanhf", "asinhf", "acoshf", "atanhf",
            "expf", "exp2f", "expm1f", "logf", "log2f", "log10f", "log1pf",
            "sqrtf", "cbrtf", "fabsf", "ceilf", "floorf", "roundf", "truncf",
            "rintf", "nearbyintf", "erff", "erfcf", "tgammaf", "lgammaf",
        ];
        for name in f_f {
            self.insert_builtin_sig(name, F32, vec![F32], Vec::new());
        }
        // double func(double) - double-precision math
        let d_d: &[&str] = &[
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
            "sqrt", "cbrt", "fabs", "ceil", "floor", "round", "trunc",
            "rint", "nearbyint", "erf", "erfc", "tgamma", "lgamma",
        ];
        for name in d_d {
            self.insert_builtin_sig(name, F64, vec![F64], Vec::new());
        }
        // float func(float, float) - two-arg single-precision
        let f_ff: &[&str] = &["atan2f", "powf", "fmodf", "remainderf", "copysignf", "fminf", "fmaxf", "fdimf", "hypotf"];
        for name in f_ff {
            self.insert_builtin_sig(name, F32, vec![F32, F32], Vec::new());
        }
        // double func(double, double) - two-arg double-precision
        let d_dd: &[&str] = &["atan2", "pow", "fmod", "remainder", "copysign", "fmin", "fmax", "fdim", "hypot"];
        for name in d_dd {
            self.insert_builtin_sig(name, F64, vec![F64, F64], Vec::new());
        }
        // int/long returning functions
        self.insert_builtin_sig("abs", I32, vec![I32], Vec::new());
        self.insert_builtin_sig("labs", I64, vec![I64], Vec::new());
        // float/double func(float/double, int)
        self.insert_builtin_sig("ldexpf", F32, vec![F32, I32], Vec::new());
        self.insert_builtin_sig("ldexp", F64, vec![F64, I32], Vec::new());
        self.insert_builtin_sig("scalbnf", F32, vec![F32, I32], Vec::new());
        self.insert_builtin_sig("scalbn", F64, vec![F64, I32], Vec::new());
        // Complex math functions: register return types and param_ctypes.
        // param_types left empty for complex since arg-casting uses param_ctypes for decomposition.
        self.insert_builtin_sig("cabs", F64, Vec::new(), vec![CType::ComplexDouble]);
        self.insert_builtin_sig("cabsf", F32, Vec::new(), vec![CType::ComplexFloat]);
        self.insert_builtin_sig("cabsl", F64, Vec::new(), Vec::new());
        self.insert_builtin_sig("carg", F64, Vec::new(), vec![CType::ComplexDouble]);
        self.insert_builtin_sig("cargf", F32, Vec::new(), vec![CType::ComplexFloat]);
        self.insert_builtin_sig("creal", F64, Vec::new(), vec![CType::ComplexDouble]);
        self.insert_builtin_sig("cimag", F64, Vec::new(), vec![CType::ComplexDouble]);
        self.insert_builtin_sig("crealf", F32, Vec::new(), vec![CType::ComplexFloat]);
        self.insert_builtin_sig("cimagf", F32, Vec::new(), vec![CType::ComplexFloat]);
        // Functions returning _Complex double (real in xmm0, imag in xmm1):
        let cd_cd: &[&str] = &[
            "csqrt", "cexp", "clog", "csin", "ccos", "ctan",
            "casin", "cacos", "catan", "csinh", "ccosh", "ctanh",
            "casinh", "cacosh", "catanh", "conj",
        ];
        for name in cd_cd {
            self.insert_builtin_sig(name, F64, Vec::new(), vec![CType::ComplexDouble]);
            self.types.func_return_ctypes.insert(name.to_string(), CType::ComplexDouble);
        }

        // Functions returning _Complex float (packed two F32 in I64):
        let cf_cf: &[&str] = &[
            "csqrtf", "cexpf", "clogf", "csinf", "ccosf", "ctanf",
            "casinf", "cacosf", "catanf", "csinhf", "ccoshf", "ctanhf",
            "casinhf", "cacoshf", "catanhf", "conjf",
        ];
        for name in cf_cf {
            self.insert_builtin_sig(name, F64, Vec::new(), vec![CType::ComplexFloat]);
            self.types.func_return_ctypes.insert(name.to_string(), CType::ComplexFloat);
        }

        // cpow/cpowf take two complex args
        self.insert_builtin_sig("cpow", F64, Vec::new(), vec![CType::ComplexDouble, CType::ComplexDouble]);
        self.types.func_return_ctypes.insert("cpow".to_string(), CType::ComplexDouble);
        self.insert_builtin_sig("cpowf", F64, Vec::new(), vec![CType::ComplexFloat, CType::ComplexFloat]);
        self.types.func_return_ctypes.insert("cpowf".to_string(), CType::ComplexFloat);
    }

    pub(super) fn type_spec_to_ir(&self, ts: &TypeSpecifier) -> IrType {
        let ts = self.resolve_type_spec(ts);
        match ts {
            TypeSpecifier::Void => IrType::Void,
            TypeSpecifier::Bool => IrType::U8, // _Bool is 1 byte, unsigned
            TypeSpecifier::Char => IrType::I8,
            TypeSpecifier::UnsignedChar => IrType::U8,
            TypeSpecifier::Short => IrType::I16,
            TypeSpecifier::UnsignedShort => IrType::U16,
            TypeSpecifier::Int => IrType::I32,
            TypeSpecifier::UnsignedInt => IrType::U32,
            TypeSpecifier::Long | TypeSpecifier::LongLong => IrType::I64,
            TypeSpecifier::UnsignedLong | TypeSpecifier::UnsignedLongLong => IrType::U64,
            TypeSpecifier::Int128 => IrType::I128,
            TypeSpecifier::UnsignedInt128 => IrType::U128,
            TypeSpecifier::Float => IrType::F32,
            TypeSpecifier::Double => IrType::F64,
            TypeSpecifier::LongDouble => IrType::F128,
            // Complex types are handled as aggregate (pointer to stack slot)
            TypeSpecifier::ComplexFloat | TypeSpecifier::ComplexDouble | TypeSpecifier::ComplexLongDouble => IrType::Ptr,
            TypeSpecifier::Pointer(_) => IrType::Ptr,
            TypeSpecifier::Array(_, _) => IrType::Ptr,
            TypeSpecifier::Struct(..) | TypeSpecifier::Union(..) => IrType::Ptr,
            TypeSpecifier::Enum(_, _) => IrType::I32,
            TypeSpecifier::TypedefName(_) => IrType::I64, // fallback for unresolved typedef
            TypeSpecifier::Signed => IrType::I32,
            TypeSpecifier::Unsigned => IrType::U32,
            TypeSpecifier::Typeof(_) => IrType::I64, // fallback: typeof(expr) not resolved at IR level
            TypeSpecifier::TypeofType(inner) => self.type_spec_to_ir(inner),
        }
    }

    /// Get the (size, alignment) for a scalar type specifier. Returns None for
    /// compound types (arrays, structs, unions) that need recursive computation.
    fn scalar_type_size_align(ts: &TypeSpecifier) -> Option<(usize, usize)> {
        // For scalar types, size == alignment on x86-64 (except LongDouble).
        match ts {
            TypeSpecifier::Void | TypeSpecifier::Bool => Some((1, 1)),
            TypeSpecifier::Char | TypeSpecifier::UnsignedChar => Some((1, 1)),
            TypeSpecifier::Short | TypeSpecifier::UnsignedShort => Some((2, 2)),
            TypeSpecifier::Int | TypeSpecifier::UnsignedInt
            | TypeSpecifier::Signed | TypeSpecifier::Unsigned => Some((4, 4)),
            TypeSpecifier::Long | TypeSpecifier::UnsignedLong
            | TypeSpecifier::LongLong | TypeSpecifier::UnsignedLongLong => Some((8, 8)),
            TypeSpecifier::Int128 | TypeSpecifier::UnsignedInt128 => Some((16, 16)),
            TypeSpecifier::Float => Some((4, 4)),
            TypeSpecifier::Double => Some((8, 8)),
            TypeSpecifier::LongDouble => Some((16, 16)),
            TypeSpecifier::ComplexFloat => Some((8, 4)),
            TypeSpecifier::ComplexDouble => Some((16, 8)),
            TypeSpecifier::ComplexLongDouble => Some((32, 16)),
            TypeSpecifier::Pointer(_) => Some((8, 8)),
            TypeSpecifier::Enum(_, _) => Some((4, 4)),
            TypeSpecifier::TypedefName(_) => Some((8, 8)), // fallback for unresolved typedefs
            _ => None,
        }
    }

    /// Look up a struct/union layout by tag name, returning the full layout.
    fn get_struct_union_layout_by_tag(&self, kind: &str, tag: &str) -> Option<&StructLayout> {
        let key = format!("{}.{}", kind, tag);
        self.types.struct_layouts.get(&key)
    }

    /// Get the struct/union layout for a resolved TypeSpecifier.
    /// Handles both inline field definitions and tag-only forward references.
    fn struct_union_layout(&self, ts: &TypeSpecifier) -> Option<StructLayout> {
        match ts {
            TypeSpecifier::Struct(tag, Some(fields), is_packed, pragma_pack, _) => {
                // Use cached layout for tagged structs
                if let Some(tag) = tag {
                    if let Some(layout) = self.types.struct_layouts.get(&format!("struct.{}", tag)) {
                        return Some(layout.clone());
                    }
                }
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                Some(self.compute_struct_union_layout_packed(fields, false, max_field_align))
            }
            TypeSpecifier::Union(tag, Some(fields), is_packed, pragma_pack, _) => {
                // Use cached layout for tagged unions
                if let Some(tag) = tag {
                    if let Some(layout) = self.types.struct_layouts.get(&format!("union.{}", tag)) {
                        return Some(layout.clone());
                    }
                }
                let max_field_align = if *is_packed { Some(1) } else { *pragma_pack };
                Some(self.compute_struct_union_layout_packed(fields, true, max_field_align))
            }
            TypeSpecifier::Struct(Some(tag), None, _, _, _) =>
                self.get_struct_union_layout_by_tag("struct", tag).cloned(),
            TypeSpecifier::Union(Some(tag), None, _, _, _) =>
                self.get_struct_union_layout_by_tag("union", tag).cloned(),
            _ => None,
        }
    }

    pub(super) fn compute_struct_union_layout_packed(&self, fields: &[StructFieldDecl], is_union: bool, max_field_align: Option<usize>) -> StructLayout {
        let struct_fields: Vec<StructField> = fields.iter().map(|f| {
            let bit_width = f.bit_width.as_ref().and_then(|bw| {
                self.eval_const_expr(bw).and_then(|c| c.to_u32())
            });
            let ty = self.struct_field_ctype(f);
            StructField {
                name: f.name.clone().unwrap_or_default(),
                ty,
                bit_width,
                alignment: f.alignment,
            }
        }).collect();
        if is_union {
            StructLayout::for_union(&struct_fields, &self.types)
        } else {
            StructLayout::for_struct_with_packing(&struct_fields, max_field_align, &self.types)
        }
    }

    pub(super) fn sizeof_type(&self, ts: &TypeSpecifier) -> usize {
        let ts = self.resolve_type_spec(ts);
        if let Some((size, _)) = Self::scalar_type_size_align(ts) {
            return size;
        }
        if let TypeSpecifier::Array(elem, Some(size_expr)) = ts {
            let elem_size = self.sizeof_type(elem);
            return self.expr_as_array_size(size_expr)
                .map(|n| elem_size * n as usize)
                .unwrap_or(elem_size);
        }
        self.struct_union_layout(ts).map(|l| l.size).unwrap_or(8)
    }

    /// Compute the alignment of a type in bytes (_Alignof).
    pub(super) fn alignof_type(&self, ts: &TypeSpecifier) -> usize {
        let ts = self.resolve_type_spec(ts);
        if let Some((_, align)) = Self::scalar_type_size_align(ts) {
            return align;
        }
        if let TypeSpecifier::Array(elem, _) = ts {
            return self.alignof_type(elem);
        }
        self.struct_union_layout(ts).map(|l| l.align).unwrap_or(8)
    }

    /// Collect array dimensions from derived declarators.
    /// Returns None for unsized dimensions (e.g., `int arr[]`).
    fn collect_derived_array_dims(&self, derived: &[DerivedDeclarator]) -> Vec<Option<usize>> {
        derived.iter().filter_map(|d| {
            if let DerivedDeclarator::Array(size_expr) = d {
                Some(size_expr.as_ref().and_then(|e| self.expr_as_array_size(e).map(|n| n as usize)))
            } else {
                None
            }
        }).collect()
    }

    /// Compute strides from an array of dimension sizes and a base element size.
    /// stride[i] = product(dims[i+1..]) * base_elem_size.
    /// E.g., dims=[3,4], base=4 -> strides=[16, 4].
    fn compute_strides_from_dims(dims: &[usize], base_elem_size: usize) -> Vec<usize> {
        let mut strides = Vec::with_capacity(dims.len());
        for i in 0..dims.len() {
            let stride: usize = dims[i+1..].iter().product::<usize>().max(1) * base_elem_size;
            strides.push(stride);
        }
        strides
    }

    /// For a pointer-to-array parameter type (e.g., Pointer(Array(Array(Int, 4), 3))),
    /// compute the array dimension strides for multi-dimensional subscript access.
    /// Returns strides for depth 0, 1, 2, ... where depth 0 is the outermost subscript.
    /// E.g., for int (*arr)[3][4]: strides = [3*4*4=48, 4*4=16, 4]
    /// For int (*arr)[3]: strides = [3*4=12, 4]
    pub(super) fn compute_ptr_array_strides(&self, type_spec: &TypeSpecifier) -> Vec<usize> {
        let ts = self.resolve_type_spec(type_spec);
        if let TypeSpecifier::Pointer(inner) = ts {
            // Collect dimensions from nested Array types
            let mut dims: Vec<usize> = Vec::new();
            let mut current = &*inner;
            loop {
                let resolved = self.resolve_type_spec(current);
                if let TypeSpecifier::Array(elem, size_expr) = resolved {
                    let n = size_expr.as_ref().and_then(|e| self.expr_as_array_size(e)).unwrap_or(1);
                    dims.push(n as usize);
                    current = &*elem;
                } else {
                    break;
                }
            }
            if dims.is_empty() {
                return vec![];
            }
            // Compute base element size (the innermost non-array type)
            let base_elem_size = self.sizeof_type(current);
            // Strides: [full_size, stride_for_dim_1, ..., base_elem_size]
            // full_size = product(all_dims) * base, then per-dim strides
            let full_size: usize = dims.iter().product::<usize>() * base_elem_size;
            let mut strides = vec![full_size];
            strides.extend(Self::compute_strides_from_dims(&dims, base_elem_size));
            strides
        } else {
            vec![]
        }
    }

    /// Compute allocation info for a declaration.
    /// Returns (alloc_size, elem_size, is_array, is_pointer, array_dim_strides).
    /// For multi-dimensional arrays like int a[2][3], array_dim_strides = [12, 4]
    /// (stride for dim 0 = 3*4=12, stride for dim 1 = 4).
    pub(super) fn compute_decl_info(&self, ts: &TypeSpecifier, derived: &[DerivedDeclarator]) -> (usize, usize, bool, bool, Vec<usize>) {
        let ts = self.resolve_type_spec(ts);
        // Check for pointer declarators (from derived or from the resolved type itself)
        let has_pointer = derived.iter().any(|d| matches!(d, DerivedDeclarator::Pointer))
            || matches!(ts, TypeSpecifier::Pointer(_));

        let has_array = derived.iter().any(|d| matches!(d, DerivedDeclarator::Array(_)));

        // Handle pointer and array combinations
        if has_pointer && !has_array {
            // Simple pointer: int *p, or typedef'd pointer (e.g., typedef struct Foo *FooPtr)
            let ptr_count = derived.iter().filter(|d| matches!(d, DerivedDeclarator::Pointer)).count();
            let elem_size = if let TypeSpecifier::Pointer(inner) = ts {
                // Pointer is in the type spec itself (typedef'd pointer)
                // If there are also pointer levels in derived, each derived pointer adds
                // a level of indirection (e.g., typedef int *intptr; intptr *pp -> int **pp)
                if ptr_count >= 1 {
                    8 // element is a pointer itself
                } else {
                    self.sizeof_type(self.resolve_type_spec(inner))
                }
            } else if ptr_count >= 2 {
                // Multiple pointer levels (e.g., char **p): element type is a pointer (size 8)
                8
            } else {
                self.sizeof_type(ts)
            };
            return (8, elem_size, false, true, vec![]);
        }
        if has_pointer && has_array {
            // Determine whether this is "array of pointers" or "pointer to array"
            // by looking at the LAST element of the derived list (outermost type wrapper).
            // The last element determines what the overall declaration IS:
            // - Last is Array: it's an array (of pointers to something)
            //   e.g., int *arr[3] -> derived=[Pointer, Array(3)] -> array of ptrs
            //   e.g., int (*ptrs[3])[4] -> derived=[Array(4), Pointer, Array(3)] -> array of ptrs-to-arrays
            // - Last is Pointer: it's a pointer (to an array)
            //   e.g., int (*p)[5] -> derived=[Array(5), Pointer] -> pointer to array
            let has_func_ptr = derived.iter().any(|d| matches!(d,
                DerivedDeclarator::FunctionPointer(_, _) | DerivedDeclarator::Function(_, _)));

            // If pointer is from resolved type spec (not in derived), and array is in derived,
            // this is an array of typedef'd pointers
            let ptr_pos = derived.iter().position(|d| matches!(d, DerivedDeclarator::Pointer));
            let pointer_from_type_spec = ptr_pos.is_none() && matches!(ts, TypeSpecifier::Pointer(_));

            // Check if the outermost (last) derived element is an Array
            let last_is_array = matches!(derived.last(), Some(DerivedDeclarator::Array(_)));

            if has_func_ptr || pointer_from_type_spec || last_is_array {
                // Array of pointers (or array of pointers-to-arrays, etc.)
                // Each element is a pointer (8 bytes).
                // Collect the variable's own array dimensions:
                // - For regular pointer arrays like int *arr[3] (derived=[Pointer, Array(3)]):
                //   Array dims AFTER the last Pointer are the variable's dimensions.
                // - For function pointer arrays like int (*ops[3])(int,int)
                //   (derived=[Array(3), Pointer, FunctionPointer(...)]):
                //   Array dims BEFORE the Pointer are the variable's dimensions,
                //   because the Pointer+FunctionPointer group describes the element type.
                let last_ptr_pos = derived.iter().rposition(|d| matches!(d, DerivedDeclarator::Pointer));
                let array_dims: Vec<Option<usize>> = if let Some(lpp) = last_ptr_pos {
                    // First try: collect Array dims after the last pointer
                    let after_dims: Vec<Option<usize>> = derived[lpp + 1..].iter().filter_map(|d| {
                        if let DerivedDeclarator::Array(size_expr) = d {
                            Some(size_expr.as_ref().and_then(|e| self.expr_as_array_size(e).map(|n| n as usize)))
                        } else {
                            None
                        }
                    }).collect();
                    if !after_dims.is_empty() {
                        after_dims
                    } else if has_func_ptr {
                        // For function pointer arrays, array dims come BEFORE the
                        // Pointer+FunctionPointer group (e.g., [Array(3), Pointer, FuncPtr])
                        derived[..lpp].iter().filter_map(|d| {
                            if let DerivedDeclarator::Array(size_expr) = d {
                                Some(size_expr.as_ref().and_then(|e| self.expr_as_array_size(e).map(|n| n as usize)))
                            } else {
                                None
                            }
                        }).collect()
                    } else {
                        after_dims
                    }
                } else {
                    // Pointer comes from type spec (typedef'd pointer/func ptr),
                    // not from derived list. All array dims in derived belong to
                    // the variable's own dimensions.
                    self.collect_derived_array_dims(derived)
                };
                let resolved_dims: Vec<usize> = array_dims.iter().map(|d| d.unwrap_or(256)).collect();
                let total_size: usize = resolved_dims.iter().product::<usize>() * 8;
                let strides = if resolved_dims.len() > 1 {
                    Self::compute_strides_from_dims(&resolved_dims, 8)
                } else {
                    vec![8]  // 1D pointer array: stride is just pointer size
                };
                return (total_size, 8, true, false, strides);
            }
            // Pointer to array (e.g., int (*p)[5]) - treat as pointer
            // Compute strides from the Array dims BEFORE the pointer in the derived list
            // (these are the pointed-to array dimensions)
            let array_dims: Vec<usize> = derived.iter()
                .take_while(|d| !matches!(d, DerivedDeclarator::Pointer))
                .filter_map(|d| {
                    if let DerivedDeclarator::Array(size_expr) = d {
                        Some(size_expr.as_ref()
                            .and_then(|e| self.expr_as_array_size(e).map(|n| n as usize))
                            .unwrap_or(1))
                    } else {
                        None
                    }
                }).collect();
            let base_elem_size = self.sizeof_type(ts);
            let full_array_size: usize = if array_dims.is_empty() {
                base_elem_size
            } else {
                array_dims.iter().product::<usize>() * base_elem_size
            };
            // strides[0] = full pointed-to array size, then per-dim strides
            let mut strides = vec![full_array_size];
            if !array_dims.is_empty() {
                strides.extend(Self::compute_strides_from_dims(&array_dims, base_elem_size));
            }
            let elem_size = full_array_size;
            return (8, elem_size, false, true, strides);
        }

        // If the resolved type itself is an Array (e.g., va_list = Array(Char, 24),
        // or typedef'd multi-dimensional arrays like typedef int arr_t[2][3])
        // and there are no derived array declarators, handle it as an array type.
        if !has_array && !has_pointer {
            if let TypeSpecifier::Array(_, _) = ts {
                // Collect all dimensions from nested Array types
                let all_dims = self.collect_type_array_dims(ts);
                // Find the innermost (non-array) element type
                let mut inner = ts;
                while let TypeSpecifier::Array(elem, _) = inner {
                    inner = elem.as_ref();
                }
                let base_elem_size = self.sizeof_type(inner).max(1);
                let total: usize = all_dims.iter().product::<usize>() * base_elem_size;
                let strides = Self::compute_strides_from_dims(&all_dims, base_elem_size);
                let elem_size = if strides.len() > 1 { strides[0] } else { base_elem_size };
                return (total, elem_size, true, false, strides);
            }
        }

        // Check for array declarators - collect all dimensions
        let array_dims = self.collect_derived_array_dims(derived);

        if !array_dims.is_empty() {
            // If derived declarators include Function/FunctionPointer,
            // the element type is a function pointer (8 bytes), not the return type.
            let has_func_ptr = derived.iter().any(|d| matches!(d,
                DerivedDeclarator::Function(_, _) | DerivedDeclarator::FunctionPointer(_, _)));
            // Also account for array dimensions in the type specifier itself
            // e.g., if type is Array(Array(Int, 3), 2) from the parser,
            // or from a typedef like: typedef unsigned char byte4_t[4];
            let type_dims = self.collect_type_array_dims(ts);

            let base_elem_size = if has_func_ptr {
                8 // function pointer size
            } else if !type_dims.is_empty() {
                // When ts itself is an array type (from typedef), use the innermost
                // element size, since collect_type_array_dims already extracts the
                // array dimensions which will be included in all_dims.
                let mut inner = ts;
                while let TypeSpecifier::Array(elem, _) = inner {
                    inner = elem.as_ref();
                }
                self.sizeof_type(inner).max(1)
            } else {
                self.sizeof_type(ts).max(1)
            };

            // Combine: derived dims come first (outermost), then type dims
            let all_dims: Vec<usize> = array_dims.iter().map(|d| d.unwrap_or(256))
                .chain(type_dims.iter().copied())
                .collect();

            // Compute total size = product of all dims * base_elem_size
            let total: usize = all_dims.iter().product::<usize>() * base_elem_size;

            // Compute strides: stride[i] = product of dims[i+1..] * base_elem_size
            let strides = Self::compute_strides_from_dims(&all_dims, base_elem_size);

            // elem_size is the stride of the outermost dimension (for 1D compat, it's base_elem_size)
            let elem_size = if strides.len() > 1 { strides[0] } else { base_elem_size };

            return (total, elem_size, true, false, strides);
        }

        // For struct/union types, use their layout size
        if let Some(layout) = self.get_struct_layout_for_type(ts) {
            return (layout.size, 0, false, false, vec![]);
        }

        // Regular scalar - use sizeof_type for the allocation size
        // (8 bytes for most scalars, 16 for long double)
        let scalar_size = self.sizeof_type(ts).max(8);
        (scalar_size, 0, false, false, vec![])
    }

    /// For Array(Array(Int, 3), 2), returns [2, 3] (but we skip the outermost
    /// since that comes from the derived declarator).
    fn collect_type_array_dims(&self, ts: &TypeSpecifier) -> Vec<usize> {
        let mut dims = Vec::new();
        let mut current_owned = ts.clone();
        loop {
            let resolved = self.resolve_type_spec(&current_owned);
            if let TypeSpecifier::Array(inner, Some(size_expr)) = &resolved {
                if let Some(n) = self.expr_as_array_size(size_expr) {
                    dims.push(n as usize);
                }
                current_owned = inner.as_ref().clone();
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

    /// Build the CType for a function parameter, correctly handling function
    /// pointer parameters (both explicit syntax and typedef'd).
    ///
    /// For explicit function pointer params like `void (*callback)(int, int)`,
    /// the parser sets `fptr_params` to Some(...) and we build
    /// CType::Pointer(CType::Function(...)).
    ///
    /// For typedef'd function pointer params like `lua_Alloc f`, the parser
    /// doesn't set `fptr_params` (the function pointer nature is hidden in the
    /// typedef). We detect this by checking if the resolved type spec has
    /// DerivedDeclarator::FunctionPointer info stored in function_typedefs or
    /// by checking the original typedef declaration.
    pub(super) fn param_ctype(&self, param: &crate::frontend::parser::ast::ParamDecl) -> CType {
        // Case 1: explicit function pointer syntax - fptr_params is set
        if let Some(ref fptr_params) = param.fptr_params {
            let return_ctype = self.type_spec_to_ctype(&param.type_spec);
            // Peel the outer Pointer wrapper from the return type
            // (type_spec for `int (*f)(...)` is Pointer(Int), the actual return type is Int)
            let actual_return = match return_ctype {
                CType::Pointer(inner) => *inner,
                other => other,
            };
            let param_types: Vec<(CType, Option<String>)> = fptr_params.iter()
                .map(|p| (self.type_spec_to_ctype(&p.type_spec), p.name.clone()))
                .collect();
            let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                return_type: actual_return,
                params: param_types,
                variadic: false,
            }));
            return CType::Pointer(Box::new(func_type));
        }

        // Case 2: typedef'd function pointer (e.g., lua_Alloc f)
        // Check if the parameter's type resolves to a typedef that was
        // declared as a function pointer typedef
        if let TypeSpecifier::TypedefName(tname) = &param.type_spec {
            if self.is_typedef_function_pointer(tname) {
                // Build the full function pointer CType from the typedef info
                if let Some(fptr_ctype) = self.build_function_pointer_ctype_from_typedef(tname) {
                    return fptr_ctype;
                }
            }
        }

        // Default: use the standard type_spec_to_ctype
        self.type_spec_to_ctype(&param.type_spec)
    }

    /// Check if a typedef name refers to a function pointer type.
    /// This handles typedefs like `typedef void *(*lua_Alloc)(void *, ...)`.
    fn is_typedef_function_pointer(&self, tname: &str) -> bool {
        // Check if the typedef's resolved type is a Pointer AND the original
        // typedef had a FunctionPointer derived declarator.
        // We track this via function_typedefs which stores function type info
        // for both bare function typedefs and function pointer typedefs.
        //
        // However, function_typedefs only stores bare function typedefs
        // (typedef int func_t(int)), not function pointer typedefs.
        // For function pointer typedefs, we check the resolved typedef: if it's
        // a Pointer type and the original declaration context implies function pointer.
        //
        // Heuristic: if the typedef resolves to Pointer(X) and there's also a
        // function_typedefs entry, it was a function pointer typedef. But this
        // isn't reliable. Instead, track function pointer typedefs explicitly.
        self.types.func_ptr_typedefs.contains(tname)
    }

    /// Build a CType::Pointer(CType::Function(...)) from a function pointer typedef.
    fn build_function_pointer_ctype_from_typedef(&self, tname: &str) -> Option<CType> {
        // Look up the stored function pointer typedef info
        if let Some(fti) = self.types.func_ptr_typedef_info.get(tname) {
            let return_ctype = self.type_spec_to_ctype(&fti.return_type);
            let param_types: Vec<(CType, Option<String>)> = fti.params.iter()
                .map(|p| (self.type_spec_to_ctype(&p.type_spec), p.name.clone()))
                .collect();
            let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                return_type: return_ctype,
                params: param_types,
                variadic: fti.variadic,
            }));
            return Some(CType::Pointer(Box::new(func_type)));
        }
        None
    }

    /// Convert a TypeSpecifier to CType (for struct layout computation).
    pub(super) fn type_spec_to_ctype(&self, ts: &TypeSpecifier) -> CType {
        // Before resolving, check if this is a function pointer typedef.
        // resolve_type_spec loses function type info for function pointer typedefs
        // (e.g., lua_Alloc resolves to Pointer(Void) instead of Pointer(Function(...))).
        // We intercept here to produce the correct CType.
        if let TypeSpecifier::TypedefName(tname) = ts {
            if let Some(fptr_ctype) = self.build_function_pointer_ctype_from_typedef(tname) {
                return fptr_ctype;
            }
        }
        let ts = self.resolve_type_spec(ts);
        match ts {
            TypeSpecifier::Void => CType::Void,
            TypeSpecifier::Char => CType::Char,
            TypeSpecifier::UnsignedChar => CType::UChar,
            TypeSpecifier::Short => CType::Short,
            TypeSpecifier::UnsignedShort => CType::UShort,
            TypeSpecifier::Bool => CType::Bool, // _Bool: 1-byte unsigned, normalizes to 0 or 1 on store
            TypeSpecifier::Int | TypeSpecifier::Signed => CType::Int,
            TypeSpecifier::UnsignedInt | TypeSpecifier::Unsigned => CType::UInt,
            TypeSpecifier::Long => CType::Long,
            TypeSpecifier::UnsignedLong => CType::ULong,
            TypeSpecifier::LongLong => CType::LongLong,
            TypeSpecifier::UnsignedLongLong => CType::ULongLong,
            TypeSpecifier::Int128 => CType::Int128,
            TypeSpecifier::UnsignedInt128 => CType::UInt128,
            TypeSpecifier::Float => CType::Float,
            TypeSpecifier::Double => CType::Double,
            TypeSpecifier::LongDouble => CType::LongDouble,
            TypeSpecifier::ComplexFloat => CType::ComplexFloat,
            TypeSpecifier::ComplexDouble => CType::ComplexDouble,
            TypeSpecifier::ComplexLongDouble => CType::ComplexLongDouble,
            TypeSpecifier::Pointer(inner) => CType::Pointer(Box::new(self.type_spec_to_ctype(inner))),
            TypeSpecifier::Array(elem, size_expr) => {
                let elem_ctype = self.type_spec_to_ctype(elem);
                let size = size_expr.as_ref().and_then(|e| {
                    self.expr_as_array_size(e).map(|n| n as usize)
                });
                CType::Array(Box::new(elem_ctype), size)
            }
            TypeSpecifier::Struct(name, fields, is_packed, pragma_pack, _) => {
                self.struct_or_union_to_ctype(name, fields, false, *is_packed, *pragma_pack)
            }
            TypeSpecifier::Union(name, fields, is_packed, pragma_pack, _) => {
                self.struct_or_union_to_ctype(name, fields, true, *is_packed, *pragma_pack)
            }
            TypeSpecifier::Enum(_, _) => CType::Int, // enums are int-sized
            TypeSpecifier::TypedefName(_) => CType::Int, // TODO: resolve typedef
            TypeSpecifier::Typeof(expr) => {
                // typeof(expr): get type from expression
                self.get_expr_ctype(expr).unwrap_or(CType::Int)
            }
            TypeSpecifier::TypeofType(inner_ts) => {
                // typeof(type): just use the inner type
                self.type_spec_to_ctype(inner_ts)
            }
        }
    }

    /// Convert a struct or union TypeSpecifier to CType.
    /// `is_union` selects between struct and union semantics.
    /// `is_packed` indicates __attribute__((packed)).
    /// `pragma_pack` is the #pragma pack(N) alignment, if any.
    fn struct_or_union_to_ctype(
        &self,
        name: &Option<String>,
        fields: &Option<Vec<StructFieldDecl>>,
        is_union: bool,
        is_packed: bool,
        pragma_pack: Option<usize>,
    ) -> CType {
        let prefix = if is_union { "union" } else { "struct" };
        let wrap = |key: String| -> CType {
            if is_union { CType::Union(key) } else { CType::Struct(key) }
        };
        // __attribute__((packed)) forces alignment 1; #pragma pack(N) caps to N.
        let max_field_align = if is_packed { Some(1) } else { pragma_pack };

        if let Some(fs) = fields {
            // Inline definition with fields: check cache first for named types.
            // Only return cached if it has a non-zero size (not a forward-declaration stub).
            // Forward-declared structs get cached with size 0 and must be recomputed
            // when the full definition becomes available.
            if let Some(tag) = name {
                let cache_key = format!("{}.{}", prefix, tag);
                if let Some(cached) = self.types.ctype_cache.borrow().get(&cache_key) {
                    let cached_size = cached.size();
                    if cached_size > 0 {
                        return cached.clone();
                    }
                    // Cached entry has size 0 (forward-declaration stub) - recompute below
                }
            }
            let struct_fields: Vec<StructField> = fs.iter().map(|f| {
                let bit_width = f.bit_width.as_ref().and_then(|bw| {
                    self.eval_const_expr(bw).and_then(|c| c.to_u32())
                });
                let ty = self.struct_field_ctype(f);
                StructField {
                    name: f.name.clone().unwrap_or_default(),
                    ty,
                    bit_width,
                    alignment: f.alignment,
                }
            }).collect();
            let layout = if is_union {
                StructLayout::for_union(&struct_fields, &self.types)
            } else {
                StructLayout::for_struct_with_packing(&struct_fields, max_field_align, &self.types)
            };
            let key = if let Some(tag) = name {
                format!("{}.{}", prefix, tag)
            } else {
                let id = self.types.next_anon_struct_id();
                format!("__anon_struct_{}", id)
            };
            self.types.insert_struct_layout_from_ref(&key, layout);
            self.types.invalidate_ctype_cache_from_ref(&key);
            let result = wrap(key.clone());
            self.types.ctype_cache.borrow_mut().insert(key, result.clone());
            result
        } else if let Some(tag) = name {
            let key = format!("{}.{}", prefix, tag);
            // Check cache first
            if let Some(cached) = self.types.ctype_cache.borrow().get(&key) {
                return cached.clone();
            }
            // Forward declaration: insert an empty layout if not already present
            if self.types.struct_layouts.get(&key).is_none() {
                let empty_layout = StructLayout {
                    fields: Vec::new(),
                    size: 0,
                    align: 1,
                    is_union,
                };
                self.types.insert_struct_layout_from_ref(&key, empty_layout);
            }
            let result = wrap(key.clone());
            self.types.ctype_cache.borrow_mut().insert(key, result.clone());
            result
        } else {
            // Anonymous forward declaration (no name, no fields)
            let id = self.types.next_anon_struct_id();
            let key = format!("__anon_struct_{}", id);
            let empty_layout = StructLayout {
                fields: Vec::new(),
                size: 0,
                align: 1,
                is_union,
            };
            self.types.insert_struct_layout_from_ref(&key, empty_layout);
            wrap(key)
        }
    }

    /// Get the CType for a struct field declaration, accounting for derived declarators.
    /// For simple fields (derived is empty), just converts type_spec.
    /// For complex fields (function pointers, etc.), uses build_full_ctype.
    pub(super) fn struct_field_ctype(&self, f: &StructFieldDecl) -> CType {
        if f.derived.is_empty() {
            self.type_spec_to_ctype(&f.type_spec)
        } else {
            self.build_full_ctype(&f.type_spec, &f.derived)
        }
    }

    /// Build a full CType from a TypeSpecifier and DerivedDeclarator chain.
    ///
    /// The derived list is produced by parse_declarator's combine_declarator_parts,
    /// which stores declarators outer-to-inner. For building the CType, we need to
    /// process inner-to-outer (inside-out rule).
    ///
    /// Examples (derived list -> CType):
    /// - `int **p`: [Pointer, Pointer] -> Pointer(Pointer(Int))
    /// - `int *arr[3]`: [Pointer, Array(3)] -> Array(Pointer(Int), 3)
    /// - `int (*fp)(int)`: [Pointer, FunctionPointer([int])] -> Pointer(Function(Int->Int))
    /// - `int (*fp[3])(int)`: [Array(3), Pointer, FunctionPointer([int])] -> Array(Pointer(Function(Int->Int)), 3)
    pub(super) fn build_full_ctype(&self, type_spec: &TypeSpecifier, derived: &[DerivedDeclarator]) -> CType {
        // Use type_spec_to_ctype on the ORIGINAL type spec (not resolved) so that
        // function pointer typedefs are correctly detected and expanded.
        let base = self.type_spec_to_ctype(type_spec);

        // Process derived declarators. The list is ordered outer-to-inner:
        // outermost wrapping (e.g. Array) first, innermost (closest to base type, e.g.
        // FunctionPointer) last. We need to build inside-out: first build the
        // inner type from the base, then wrap with outer layers.
        //
        // Strategy: find the innermost function pointer group first (Pointer+FunctionPointer),
        // build the function pointer type from the base, then apply remaining outer
        // wrappers (Array, Pointer).

        // Separate into prefix (outer wrappers) and suffix (Pointer+FunctionPointer core)
        // Look for the pattern: [outer...] [Pointer] [FunctionPointer]
        // where the Pointer+FunctionPointer pair is the function pointer core.
        let fptr_idx = self.find_function_pointer_core(derived);

        if let Some(fp_start) = fptr_idx {
            // Build the function pointer type from base
            let mut result = base;

            // Process from fp_start to end (the function pointer core and any
            // additional inner wrappers after it)
            let mut i = fp_start;
            while i < derived.len() {
                match &derived[i] {
                    DerivedDeclarator::Pointer => {
                        if i + 1 < derived.len() && matches!(&derived[i + 1], DerivedDeclarator::FunctionPointer(params, _) | DerivedDeclarator::Function(params, _)) {
                            let (params, variadic) = match &derived[i + 1] {
                                DerivedDeclarator::FunctionPointer(p, v) | DerivedDeclarator::Function(p, v) => (p, *v),
                                _ => unreachable!(),
                            };
                            let param_types = self.convert_param_decls_to_ctypes(params);
                            let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                                return_type: result,
                                params: param_types,
                                variadic,
                            }));
                            result = CType::Pointer(Box::new(func_type));
                            i += 2;
                        } else {
                            result = CType::Pointer(Box::new(result));
                            i += 1;
                        }
                    }
                    DerivedDeclarator::FunctionPointer(params, variadic) => {
                        let param_types = self.convert_param_decls_to_ctypes(params);
                        let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                            return_type: result,
                            params: param_types,
                            variadic: *variadic,
                        }));
                        result = CType::Pointer(Box::new(func_type));
                        i += 1;
                    }
                    DerivedDeclarator::Function(params, variadic) => {
                        let param_types = self.convert_param_decls_to_ctypes(params);
                        let func_type = CType::Function(Box::new(crate::common::types::FunctionType {
                            return_type: result,
                            params: param_types,
                            variadic: *variadic,
                        }));
                        result = func_type;
                        i += 1;
                    }
                    _ => { i += 1; }
                }
            }

            // Now apply outer wrappers (prefix before fp_start): Array, Pointer
            // These are outermost, so apply them in reverse order
            let prefix = &derived[..fp_start];
            for d in prefix.iter().rev() {
                match d {
                    DerivedDeclarator::Array(size_expr) => {
                        let size = size_expr.as_ref().and_then(|e| {
                            self.expr_as_array_size(e).map(|n| n as usize)
                        });
                        result = CType::Array(Box::new(result), size);
                    }
                    DerivedDeclarator::Pointer => {
                        result = CType::Pointer(Box::new(result));
                    }
                    _ => {}
                }
            }

            result
        } else {
            // No function pointer - simple case
            let mut result = base;
            let mut i = 0;
            while i < derived.len() {
                match &derived[i] {
                    DerivedDeclarator::Pointer => {
                        result = CType::Pointer(Box::new(result));
                        i += 1;
                    }
                    DerivedDeclarator::Array(_) => {
                        let start = i;
                        while i < derived.len() && matches!(&derived[i], DerivedDeclarator::Array(_)) {
                            i += 1;
                        }
                        for j in (start..i).rev() {
                            if let DerivedDeclarator::Array(size_expr) = &derived[j] {
                                let size = size_expr.as_ref().and_then(|e| {
                                    self.expr_as_array_size(e).map(|n| n as usize)
                                });
                                result = CType::Array(Box::new(result), size);
                            }
                        }
                    }
                    _ => { i += 1; }
                }
            }
            result
        }
    }

    /// Find the start index of the function pointer core in a derived declarator list.
    /// The function pointer core is [Pointer, FunctionPointer] or standalone FunctionPointer.
    fn find_function_pointer_core(&self, derived: &[DerivedDeclarator]) -> Option<usize> {
        // Look for Pointer followed by FunctionPointer
        for i in 0..derived.len() {
            if matches!(&derived[i], DerivedDeclarator::Pointer) {
                if i + 1 < derived.len() && matches!(&derived[i + 1], DerivedDeclarator::FunctionPointer(_, _)) {
                    return Some(i);
                }
            }
            // Standalone FunctionPointer
            if matches!(&derived[i], DerivedDeclarator::FunctionPointer(_, _)) {
                return Some(i);
            }
            // Standalone Function (for function declarations)
            if matches!(&derived[i], DerivedDeclarator::Function(_, _)) {
                return Some(i);
            }
        }
        None
    }

    /// Convert ParamDecl list to CType list for function types.
    fn convert_param_decls_to_ctypes(&self, params: &[ParamDecl]) -> Vec<(CType, Option<String>)> {
        params.iter().map(|p| {
            let ty = self.type_spec_to_ctype(&self.resolve_type_spec(&p.type_spec));
            (ty, p.name.clone())
        }).collect()
    }

}
