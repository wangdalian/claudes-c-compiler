use crate::frontend::parser::ast::*;
use crate::frontend::sema::builtins::{self, BuiltinKind};
use crate::ir::ir::*;
use crate::common::types::IrType;
use super::lowering::Lowerer;

impl Lowerer {
    pub(super) fn lower_expr(&mut self, expr: &Expr) -> Operand {
        match expr {
            Expr::IntLiteral(val, _) => {
                Operand::Const(IrConst::I64(*val))
            }
            Expr::FloatLiteral(val, _) => {
                Operand::Const(IrConst::F64(*val))
            }
            Expr::CharLiteral(ch, _) => {
                Operand::Const(IrConst::I32(*ch as i32))
            }
            Expr::StringLiteral(s, _) => {
                let label = format!(".Lstr{}", self.next_string);
                self.next_string += 1;
                self.module.string_literals.push((label.clone(), s.clone()));
                let dest = self.fresh_value();
                self.emit(Instruction::GlobalAddr { dest, name: label });
                Operand::Value(dest)
            }
            Expr::Identifier(name, _) => {
                // Check for enum constants first
                if let Some(&val) = self.enum_constants.get(name) {
                    return Operand::Const(IrConst::I64(val));
                }
                if let Some(info) = self.locals.get(name).cloned() {
                    if info.is_array {
                        // Arrays decay to pointer: return the alloca address itself
                        return Operand::Value(info.alloca);
                    }
                    let dest = self.fresh_value();
                    self.emit(Instruction::Load { dest, ptr: info.alloca, ty: info.ty });
                    Operand::Value(dest)
                } else if let Some(ginfo) = self.globals.get(name).cloned() {
                    // Global variable: get address, then load the value
                    let addr = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest: addr, name: name.clone() });
                    if ginfo.is_array {
                        // Arrays decay to pointer: address IS the pointer
                        Operand::Value(addr)
                    } else {
                        let dest = self.fresh_value();
                        self.emit(Instruction::Load { dest, ptr: addr, ty: ginfo.ty });
                        Operand::Value(dest)
                    }
                } else {
                    // Assume it's a function reference (or unknown global)
                    let dest = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest, name: name.clone() });
                    Operand::Value(dest)
                }
            }
            Expr::BinaryOp(op, lhs, rhs, _) => {
                match op {
                    BinOp::LogicalAnd => {
                        return self.lower_logical_and(lhs, rhs);
                    }
                    BinOp::LogicalOr => {
                        return self.lower_logical_or(lhs, rhs);
                    }
                    _ => {}
                }

                // Check for pointer arithmetic: ptr + int or int + ptr or ptr - int
                if matches!(op, BinOp::Add | BinOp::Sub) {
                    let lhs_is_ptr = self.expr_is_pointer(lhs);
                    let rhs_is_ptr = self.expr_is_pointer(rhs);

                    if lhs_is_ptr && !rhs_is_ptr {
                        // ptr + int or ptr - int: scale RHS by element size
                        let elem_size = self.get_pointer_elem_size_from_expr(lhs);
                        let lhs_val = self.lower_expr(lhs);
                        let rhs_val = self.lower_expr(rhs);
                        let scaled_rhs = if elem_size > 1 {
                            let scale = Operand::Const(IrConst::I64(elem_size as i64));
                            let scaled = self.fresh_value();
                            self.emit(Instruction::BinOp {
                                dest: scaled,
                                op: IrBinOp::Mul,
                                lhs: rhs_val,
                                rhs: scale,
                                ty: IrType::I64,
                            });
                            Operand::Value(scaled)
                        } else {
                            rhs_val
                        };
                        let dest = self.fresh_value();
                        let ir_op = if *op == BinOp::Add { IrBinOp::Add } else { IrBinOp::Sub };
                        self.emit(Instruction::BinOp { dest, op: ir_op, lhs: lhs_val, rhs: scaled_rhs, ty: IrType::I64 });
                        return Operand::Value(dest);
                    } else if rhs_is_ptr && !lhs_is_ptr && *op == BinOp::Add {
                        // int + ptr: scale LHS by element size
                        let elem_size = self.get_pointer_elem_size_from_expr(rhs);
                        let lhs_val = self.lower_expr(lhs);
                        let rhs_val = self.lower_expr(rhs);
                        let scaled_lhs = if elem_size > 1 {
                            let scale = Operand::Const(IrConst::I64(elem_size as i64));
                            let scaled = self.fresh_value();
                            self.emit(Instruction::BinOp {
                                dest: scaled,
                                op: IrBinOp::Mul,
                                lhs: lhs_val,
                                rhs: scale,
                                ty: IrType::I64,
                            });
                            Operand::Value(scaled)
                        } else {
                            lhs_val
                        };
                        let dest = self.fresh_value();
                        self.emit(Instruction::BinOp { dest, op: IrBinOp::Add, lhs: scaled_lhs, rhs: rhs_val, ty: IrType::I64 });
                        return Operand::Value(dest);
                    } else if lhs_is_ptr && rhs_is_ptr && *op == BinOp::Sub {
                        // ptr - ptr: result is byte difference / element size
                        let elem_size = self.get_pointer_elem_size_from_expr(lhs);
                        let lhs_val = self.lower_expr(lhs);
                        let rhs_val = self.lower_expr(rhs);
                        let diff = self.fresh_value();
                        self.emit(Instruction::BinOp { dest: diff, op: IrBinOp::Sub, lhs: lhs_val, rhs: rhs_val, ty: IrType::I64 });
                        if elem_size > 1 {
                            let scale = Operand::Const(IrConst::I64(elem_size as i64));
                            let dest = self.fresh_value();
                            self.emit(Instruction::BinOp { dest, op: IrBinOp::SDiv, lhs: Operand::Value(diff), rhs: scale, ty: IrType::I64 });
                            return Operand::Value(dest);
                        }
                        return Operand::Value(diff);
                    }
                }

                // Infer types to determine signed/unsigned and result width
                let lhs_ty = self.infer_expr_type(lhs);
                let rhs_ty = self.infer_expr_type(rhs);
                let common_ty = Self::common_type(lhs_ty, rhs_ty);
                let is_unsigned = common_ty.is_unsigned();

                let lhs_val = self.lower_expr(lhs);
                let rhs_val = self.lower_expr(rhs);
                let dest = self.fresh_value();

                match op {
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let cmp_op = match (op, is_unsigned) {
                            (BinOp::Eq, _) => IrCmpOp::Eq,
                            (BinOp::Ne, _) => IrCmpOp::Ne,
                            (BinOp::Lt, false) => IrCmpOp::Slt,
                            (BinOp::Lt, true) => IrCmpOp::Ult,
                            (BinOp::Le, false) => IrCmpOp::Sle,
                            (BinOp::Le, true) => IrCmpOp::Ule,
                            (BinOp::Gt, false) => IrCmpOp::Sgt,
                            (BinOp::Gt, true) => IrCmpOp::Ugt,
                            (BinOp::Ge, false) => IrCmpOp::Sge,
                            (BinOp::Ge, true) => IrCmpOp::Uge,
                            _ => unreachable!(),
                        };
                        self.emit(Instruction::Cmp { dest, op: cmp_op, lhs: lhs_val, rhs: rhs_val, ty: IrType::I64 });
                    }
                    _ => {
                        let ir_op = match (op, is_unsigned) {
                            (BinOp::Add, _) => IrBinOp::Add,
                            (BinOp::Sub, _) => IrBinOp::Sub,
                            (BinOp::Mul, _) => IrBinOp::Mul,
                            (BinOp::Div, false) => IrBinOp::SDiv,
                            (BinOp::Div, true) => IrBinOp::UDiv,
                            (BinOp::Mod, false) => IrBinOp::SRem,
                            (BinOp::Mod, true) => IrBinOp::URem,
                            (BinOp::BitAnd, _) => IrBinOp::And,
                            (BinOp::BitOr, _) => IrBinOp::Or,
                            (BinOp::BitXor, _) => IrBinOp::Xor,
                            (BinOp::Shl, _) => IrBinOp::Shl,
                            (BinOp::Shr, false) => IrBinOp::AShr,
                            (BinOp::Shr, true) => IrBinOp::LShr,
                            _ => unreachable!(),
                        };
                        self.emit(Instruction::BinOp { dest, op: ir_op, lhs: lhs_val, rhs: rhs_val, ty: IrType::I64 });
                    }
                }

                // For sub-64-bit types (U32, I32, etc.), insert a truncation cast
                // to ensure the result fits in the narrower type.
                // This handles unsigned int wraparound and signed int overflow behavior.
                if common_ty == IrType::U32 || common_ty == IrType::I32 {
                    match op {
                        // Comparisons always produce 0 or 1, no truncation needed
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le |
                        BinOp::Gt | BinOp::Ge | BinOp::LogicalAnd | BinOp::LogicalOr => {
                            Operand::Value(dest)
                        }
                        _ => {
                            let narrowed = self.fresh_value();
                            self.emit(Instruction::Cast {
                                dest: narrowed,
                                src: Operand::Value(dest),
                                from_ty: IrType::I64,
                                to_ty: common_ty,
                            });
                            Operand::Value(narrowed)
                        }
                    }
                } else {
                    Operand::Value(dest)
                }
            }
            Expr::UnaryOp(op, inner, _) => {
                match op {
                    UnaryOp::Plus => {
                        // Unary plus is a no-op (identity), just lower the inner expression
                        self.lower_expr(inner)
                    }
                    UnaryOp::Neg => {
                        let inner_ty = self.infer_expr_type(inner);
                        let val = self.lower_expr(inner);
                        let dest = self.fresh_value();
                        self.emit(Instruction::UnaryOp { dest, op: IrUnaryOp::Neg, src: val, ty: IrType::I64 });
                        // Truncate to sub-64-bit type if needed
                        if inner_ty == IrType::U32 || inner_ty == IrType::I32 {
                            let narrowed = self.fresh_value();
                            self.emit(Instruction::Cast {
                                dest: narrowed,
                                src: Operand::Value(dest),
                                from_ty: IrType::I64,
                                to_ty: inner_ty,
                            });
                            Operand::Value(narrowed)
                        } else {
                            Operand::Value(dest)
                        }
                    }
                    UnaryOp::BitNot => {
                        let inner_ty = self.infer_expr_type(inner);
                        let val = self.lower_expr(inner);
                        let dest = self.fresh_value();
                        self.emit(Instruction::UnaryOp { dest, op: IrUnaryOp::Not, src: val, ty: IrType::I64 });
                        // Truncate to sub-64-bit type if needed (e.g., ~uint must be 32-bit)
                        if inner_ty == IrType::U32 || inner_ty == IrType::I32 {
                            let narrowed = self.fresh_value();
                            self.emit(Instruction::Cast {
                                dest: narrowed,
                                src: Operand::Value(dest),
                                from_ty: IrType::I64,
                                to_ty: inner_ty,
                            });
                            Operand::Value(narrowed)
                        } else {
                            Operand::Value(dest)
                        }
                    }
                    UnaryOp::LogicalNot => {
                        let val = self.lower_expr(inner);
                        let dest = self.fresh_value();
                        self.emit(Instruction::Cmp { dest, op: IrCmpOp::Eq, lhs: val, rhs: Operand::Const(IrConst::I64(0)), ty: IrType::I64 });
                        Operand::Value(dest)
                    }
                    UnaryOp::PreInc | UnaryOp::PreDec => {
                        self.lower_pre_inc_dec(inner, *op)
                    }
                }
            }
            Expr::PostfixOp(op, inner, _) => {
                self.lower_post_inc_dec(inner, *op)
            }
            Expr::Assign(lhs, rhs, _) => {
                let rhs_val = self.lower_expr(rhs);
                let ty = self.get_expr_type(lhs);
                if let Some(lv) = self.lower_lvalue(lhs) {
                    self.store_lvalue_typed(&lv, rhs_val.clone(), ty);
                    return rhs_val;
                }
                // Fallback: just return the rhs value
                rhs_val
            }
            Expr::CompoundAssign(op, lhs, rhs, _) => {
                self.lower_compound_assign(op, lhs, rhs)
            }
            Expr::FunctionCall(func, args, _) => {
                // First, resolve __builtin_* functions before normal dispatch.
                if let Expr::Identifier(name, _) = func.as_ref() {
                    if let Some(builtin_info) = builtins::resolve_builtin(name) {
                        match &builtin_info.kind {
                            BuiltinKind::LibcAlias(libc_name) => {
                                let arg_vals: Vec<Operand> = args.iter().map(|a| self.lower_expr(a)).collect();
                                let dest = self.fresh_value();
                                self.emit(Instruction::Call {
                                    dest: Some(dest),
                                    func: libc_name.clone(),
                                    args: arg_vals,
                                    return_type: IrType::I64,
                                });
                                return Operand::Value(dest);
                            }
                            BuiltinKind::Identity => {
                                // __builtin_expect(x, y) -> just return x
                                if let Some(first_arg) = args.first() {
                                    return self.lower_expr(first_arg);
                                }
                                return Operand::Const(IrConst::I64(0));
                            }
                            BuiltinKind::ConstantI64(val) => {
                                return Operand::Const(IrConst::I64(*val));
                            }
                            BuiltinKind::ConstantF64(_val) => {
                                // TODO: handle float constants properly
                                return Operand::Const(IrConst::I64(0));
                            }
                            BuiltinKind::Intrinsic(_intr) => {
                                // Strip __builtin_ prefix and call the underlying function
                                let cleaned_name = name.strip_prefix("__builtin_").unwrap_or(name).to_string();
                                let arg_vals: Vec<Operand> = args.iter().map(|a| self.lower_expr(a)).collect();
                                let dest = self.fresh_value();
                                self.emit(Instruction::Call {
                                    dest: Some(dest),
                                    func: cleaned_name,
                                    args: arg_vals,
                                    return_type: IrType::I64,
                                });
                                return Operand::Value(dest);
                            }
                        }
                    }
                }

                // Normal call dispatch (direct calls and indirect function pointer calls)
                let arg_vals: Vec<Operand> = args.iter().map(|a| self.lower_expr(a)).collect();
                let dest = self.fresh_value();

                // Look up function return type for narrowing cast after the call.
                // Default to I64 if unknown (e.g., implicitly declared functions).
                let mut call_ret_ty = IrType::I64;

                match func.as_ref() {
                    Expr::Identifier(name, _) => {
                        // Check if this is a local variable (function pointer) rather
                        // than a direct function name.
                        let is_local_fptr = self.locals.contains_key(name)
                            && !self.known_functions.contains(name);
                        let is_global_fptr = !self.locals.contains_key(name)
                            && self.globals.contains_key(name)
                            && !self.known_functions.contains(name);

                        if is_local_fptr {
                            // Load the function pointer from the local variable
                            let info = self.locals.get(name).unwrap().clone();
                            let ptr_val = self.fresh_value();
                            self.emit(Instruction::Load {
                                dest: ptr_val,
                                ptr: info.alloca,
                                ty: IrType::Ptr,
                            });
                            self.emit(Instruction::CallIndirect {
                                dest: Some(dest),
                                func_ptr: Operand::Value(ptr_val),
                                args: arg_vals,
                                return_type: IrType::I64,
                            });
                        } else if is_global_fptr {
                            // Load the function pointer from a global variable
                            let addr = self.fresh_value();
                            self.emit(Instruction::GlobalAddr { dest: addr, name: name.clone() });
                            let ptr_val = self.fresh_value();
                            self.emit(Instruction::Load {
                                dest: ptr_val,
                                ptr: addr,
                                ty: IrType::Ptr,
                            });
                            self.emit(Instruction::CallIndirect {
                                dest: Some(dest),
                                func_ptr: Operand::Value(ptr_val),
                                args: arg_vals,
                                return_type: IrType::I64,
                            });
                        } else {
                            // Direct function call - look up return type
                            if let Some(&ret_ty) = self.function_return_types.get(name) {
                                call_ret_ty = ret_ty;
                            }
                            self.emit(Instruction::Call {
                                dest: Some(dest),
                                func: name.clone(),
                                args: arg_vals,
                                return_type: IrType::I64,
                            });
                        }
                    }
                    Expr::Deref(inner, _) => {
                        // (*func_ptr)(args...) - dereference is a no-op for function
                        // pointers in C, just evaluate the inner expression to get the pointer
                        let func_ptr = self.lower_expr(inner);
                        self.emit(Instruction::CallIndirect {
                            dest: Some(dest),
                            func_ptr,
                            args: arg_vals,
                            return_type: IrType::I64,
                        });
                    }
                    _ => {
                        // General expression as callee (e.g., array[i](...), struct.fptr(...))
                        // Evaluate the expression to get the function pointer value
                        let func_ptr = self.lower_expr(func);
                        self.emit(Instruction::CallIndirect {
                            dest: Some(dest),
                            func_ptr,
                            args: arg_vals,
                            return_type: IrType::I64,
                        });
                    }
                }

                // If the function returns a type narrower than I64,
                // insert a narrowing cast to properly truncate/sign-extend the result.
                if call_ret_ty != IrType::I64 && call_ret_ty != IrType::Ptr
                    && call_ret_ty != IrType::Void && call_ret_ty.is_integer()
                {
                    let narrowed = self.fresh_value();
                    self.emit(Instruction::Cast {
                        dest: narrowed,
                        src: Operand::Value(dest),
                        from_ty: IrType::I64,
                        to_ty: call_ret_ty,
                    });
                    Operand::Value(narrowed)
                } else {
                    Operand::Value(dest)
                }
            }
            Expr::Conditional(cond, then_expr, else_expr, _) => {
                let cond_val = self.lower_expr(cond);
                let result_alloca = self.fresh_value();
                self.emit(Instruction::Alloca { dest: result_alloca, ty: IrType::I64, size: 8 });

                let then_label = self.fresh_label("ternary_then");
                let else_label = self.fresh_label("ternary_else");
                let end_label = self.fresh_label("ternary_end");

                self.terminate(Terminator::CondBranch {
                    cond: cond_val,
                    true_label: then_label.clone(),
                    false_label: else_label.clone(),
                });

                self.start_block(then_label);
                let then_val = self.lower_expr(then_expr);
                self.emit(Instruction::Store { val: then_val, ptr: result_alloca, ty: IrType::I64 });
                self.terminate(Terminator::Branch(end_label.clone()));

                self.start_block(else_label);
                let else_val = self.lower_expr(else_expr);
                self.emit(Instruction::Store { val: else_val, ptr: result_alloca, ty: IrType::I64 });
                self.terminate(Terminator::Branch(end_label.clone()));

                self.start_block(end_label);
                let result = self.fresh_value();
                self.emit(Instruction::Load { dest: result, ptr: result_alloca, ty: IrType::I64 });
                Operand::Value(result)
            }
            Expr::Cast(ref target_type, inner, _) => {
                let src = self.lower_expr(inner);
                let to_ty = self.type_spec_to_ir(target_type);
                // For pointer casts or same-type casts, just pass through
                if to_ty == IrType::Ptr || to_ty == IrType::I64 {
                    src
                } else {
                    // Emit a cast instruction for type conversions
                    let dest = self.fresh_value();
                    self.emit(Instruction::Cast {
                        dest,
                        src,
                        from_ty: IrType::I64, // TODO: track actual source type
                        to_ty,
                    });
                    Operand::Value(dest)
                }
            }
            Expr::CompoundLiteral(ref type_spec, ref init, _) => {
                // Compound literal: (type){initializer}
                // Allocate a local, initialize it, and return its address or value.
                // For now, treat it as an alloca with the initializer, similar to a local var.
                let ty = self.type_spec_to_ir(type_spec);
                let size = self.sizeof_type(type_spec);
                let alloca = self.fresh_value();
                self.emit(Instruction::Alloca { dest: alloca, size, ty });
                // Initialize the compound literal - for simple scalar initializers
                match init.as_ref() {
                    Initializer::Expr(expr) => {
                        let val = self.lower_expr(expr);
                        self.emit(Instruction::Store { val, ptr: alloca, ty });
                    }
                    Initializer::List(items) => {
                        // Initialize elements
                        let elem_size = self.compound_literal_elem_size(type_spec);
                        for (i, item) in items.iter().enumerate() {
                            let val = match &item.init {
                                Initializer::Expr(expr) => self.lower_expr(expr),
                                _ => Operand::Const(IrConst::I64(0)), // TODO: nested lists
                            };
                            if i == 0 && items.len() == 1 && elem_size == size {
                                // Simple scalar in braces
                                self.emit(Instruction::Store { val, ptr: alloca, ty });
                            } else {
                                // Array/struct element: compute offset
                                let offset_val = Operand::Const(IrConst::I64((i * elem_size) as i64));
                                let elem_ptr = self.fresh_value();
                                self.emit(Instruction::GetElementPtr {
                                    dest: elem_ptr,
                                    base: alloca,
                                    offset: offset_val,
                                    ty,
                                });
                                let store_ty = if elem_size <= 1 { IrType::I8 }
                                    else if elem_size <= 2 { IrType::I16 }
                                    else if elem_size <= 4 { IrType::I32 }
                                    else { IrType::I64 };
                                self.emit(Instruction::Store { val, ptr: elem_ptr, ty: store_ty });
                            }
                        }
                    }
                }
                // Return the alloca value (address of the compound literal)
                Operand::Value(alloca)
            }
            Expr::Sizeof(arg, _) => {
                let size = match arg.as_ref() {
                    SizeofArg::Type(ts) => self.sizeof_type(ts),
                    SizeofArg::Expr(expr) => self.sizeof_expr(expr),
                };
                Operand::Const(IrConst::I64(size as i64))
            }
            Expr::AddressOf(inner, _) => {
                // Try to get the lvalue address
                if let Some(lv) = self.lower_lvalue(inner) {
                    let addr = self.lvalue_addr(&lv);
                    // For alloca values, we need to get the address of the alloca
                    // using GEP with offset 0. This produces a leaq in codegen.
                    let result = self.fresh_value();
                    self.emit(Instruction::GetElementPtr {
                        dest: result,
                        base: addr,
                        offset: Operand::Const(IrConst::I64(0)),
                        ty: IrType::Ptr,
                    });
                    return Operand::Value(result);
                }
                // Fallback for globals
                if let Expr::Identifier(name, _) = inner.as_ref() {
                    let dest = self.fresh_value();
                    self.emit(Instruction::GlobalAddr { dest, name: name.clone() });
                    return Operand::Value(dest);
                }
                // TODO: handle other cases
                let dest = self.fresh_value();
                self.emit(Instruction::GlobalAddr { dest, name: "unknown".to_string() });
                Operand::Value(dest)
            }
            Expr::Deref(inner, _) => {
                let ptr = self.lower_expr(inner);
                let dest = self.fresh_value();
                let deref_ty = self.get_pointee_type_of_expr(inner).unwrap_or(IrType::I64);
                match ptr {
                    Operand::Value(v) => {
                        self.emit(Instruction::Load { dest, ptr: v, ty: deref_ty });
                    }
                    Operand::Const(_) => {
                        let tmp = self.fresh_value();
                        self.emit(Instruction::Copy { dest: tmp, src: ptr });
                        self.emit(Instruction::Load { dest, ptr: tmp, ty: deref_ty });
                    }
                }
                Operand::Value(dest)
            }
            Expr::ArraySubscript(base, index, _) => {
                // For multi-dim arrays, a[i] where a is int[2][3] returns the
                // address of the sub-array (don't load). Only load at the innermost subscript.
                if self.subscript_result_is_array(expr) {
                    // Result is a sub-array - return its address as a pointer
                    let addr = self.compute_array_element_addr(base, index);
                    return Operand::Value(addr);
                }
                // Compute element address and load with proper element type
                let elem_ty = self.get_expr_type(expr);
                let addr = self.compute_array_element_addr(base, index);
                let dest = self.fresh_value();
                self.emit(Instruction::Load { dest, ptr: addr, ty: elem_ty });
                Operand::Value(dest)
            }
            Expr::MemberAccess(base_expr, field_name, _) => {
                // s.field -> compute address of struct base + field offset, then load
                let (field_offset, field_ty) = self.resolve_member_access(base_expr, field_name);
                let base_addr = self.get_struct_base_addr(base_expr);
                let field_addr = self.fresh_value();
                self.emit(Instruction::GetElementPtr {
                    dest: field_addr,
                    base: base_addr,
                    offset: Operand::Const(IrConst::I64(field_offset as i64)),
                    ty: field_ty,
                });
                let dest = self.fresh_value();
                self.emit(Instruction::Load { dest, ptr: field_addr, ty: field_ty });
                Operand::Value(dest)
            }
            Expr::PointerMemberAccess(base_expr, field_name, _) => {
                // p->field -> load pointer, then compute field offset, then load
                let ptr_val = self.lower_expr(base_expr);
                let base_addr = match ptr_val {
                    Operand::Value(v) => v,
                    Operand::Const(_) => {
                        let tmp = self.fresh_value();
                        self.emit(Instruction::Copy { dest: tmp, src: ptr_val });
                        tmp
                    }
                };
                let (field_offset, field_ty) = self.resolve_pointer_member_access(base_expr, field_name);
                let field_addr = self.fresh_value();
                self.emit(Instruction::GetElementPtr {
                    dest: field_addr,
                    base: base_addr,
                    offset: Operand::Const(IrConst::I64(field_offset as i64)),
                    ty: field_ty,
                });
                let dest = self.fresh_value();
                self.emit(Instruction::Load { dest, ptr: field_addr, ty: field_ty });
                Operand::Value(dest)
            }
            Expr::Comma(lhs, rhs, _) => {
                self.lower_expr(lhs);
                self.lower_expr(rhs)
            }
            Expr::StmtExpr(compound, _) => {
                // GCC statement expression: ({ stmt; ...; expr; })
                // Lower all block items; the value is the last expression
                let mut last_val = Operand::Const(IrConst::I64(0));
                for item in &compound.items {
                    match item {
                        BlockItem::Statement(stmt) => {
                            // Check if this is an expression statement (value to return)
                            if let Stmt::Expr(Some(expr)) = stmt {
                                last_val = self.lower_expr(expr);
                            } else {
                                self.lower_stmt(stmt);
                            }
                        }
                        BlockItem::Declaration(decl) => {
                            self.lower_local_decl(decl);
                        }
                    }
                }
                last_val
            }
        }
    }

    /// Lower short-circuit && (logical AND).
    /// Evaluates lhs; if false, result is 0; otherwise evaluates rhs.
    fn lower_logical_and(&mut self, lhs: &Expr, rhs: &Expr) -> Operand {
        let result_alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: result_alloca, ty: IrType::I64, size: 8 });

        let rhs_label = self.fresh_label("and_rhs");
        let end_label = self.fresh_label("and_end");

        // Evaluate lhs
        let lhs_val = self.lower_expr(lhs);

        // If lhs is false (0), short-circuit to end with result = 0
        self.emit(Instruction::Store { val: Operand::Const(IrConst::I64(0)), ptr: result_alloca, ty: IrType::I64 });
        self.terminate(Terminator::CondBranch {
            cond: lhs_val,
            true_label: rhs_label.clone(),
            false_label: end_label.clone(),
        });

        // Evaluate rhs
        self.start_block(rhs_label);
        let rhs_val = self.lower_expr(rhs);
        // Result is (rhs != 0) ? 1 : 0
        let rhs_bool = self.fresh_value();
        self.emit(Instruction::Cmp {
            dest: rhs_bool,
            op: IrCmpOp::Ne,
            lhs: rhs_val,
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::I64,
        });
        self.emit(Instruction::Store { val: Operand::Value(rhs_bool), ptr: result_alloca, ty: IrType::I64 });
        self.terminate(Terminator::Branch(end_label.clone()));

        self.start_block(end_label);
        let result = self.fresh_value();
        self.emit(Instruction::Load { dest: result, ptr: result_alloca, ty: IrType::I64 });
        Operand::Value(result)
    }

    /// Lower short-circuit || (logical OR).
    /// Evaluates lhs; if true, result is 1; otherwise evaluates rhs.
    fn lower_logical_or(&mut self, lhs: &Expr, rhs: &Expr) -> Operand {
        let result_alloca = self.fresh_value();
        self.emit(Instruction::Alloca { dest: result_alloca, ty: IrType::I64, size: 8 });

        let rhs_label = self.fresh_label("or_rhs");
        let end_label = self.fresh_label("or_end");

        // Evaluate lhs
        let lhs_val = self.lower_expr(lhs);

        // If lhs is true (nonzero), short-circuit to end with result = 1
        self.emit(Instruction::Store { val: Operand::Const(IrConst::I64(1)), ptr: result_alloca, ty: IrType::I64 });
        self.terminate(Terminator::CondBranch {
            cond: lhs_val,
            true_label: end_label.clone(),
            false_label: rhs_label.clone(),
        });

        // Evaluate rhs
        self.start_block(rhs_label);
        let rhs_val = self.lower_expr(rhs);
        // Result is (rhs != 0) ? 1 : 0
        let rhs_bool = self.fresh_value();
        self.emit(Instruction::Cmp {
            dest: rhs_bool,
            op: IrCmpOp::Ne,
            lhs: rhs_val,
            rhs: Operand::Const(IrConst::I64(0)),
            ty: IrType::I64,
        });
        self.emit(Instruction::Store { val: Operand::Value(rhs_bool), ptr: result_alloca, ty: IrType::I64 });
        self.terminate(Terminator::Branch(end_label.clone()));

        self.start_block(end_label);
        let result = self.fresh_value();
        self.emit(Instruction::Load { dest: result, ptr: result_alloca, ty: IrType::I64 });
        Operand::Value(result)
    }

    /// Lower pre-increment/decrement for any lvalue.
    pub(super) fn lower_pre_inc_dec(&mut self, inner: &Expr, op: UnaryOp) -> Operand {
        let ty = self.get_expr_type(inner);
        if let Some(lv) = self.lower_lvalue(inner) {
            let loaded = self.load_lvalue_typed(&lv, ty);
            let loaded_val = match loaded {
                Operand::Value(v) => v,
                _ => {
                    let tmp = self.fresh_value();
                    self.emit(Instruction::Copy { dest: tmp, src: loaded });
                    tmp
                }
            };
            // For pointer types, increment by element size instead of 1
            let step = if ty == IrType::Ptr {
                let elem_size = self.get_pointer_elem_size(inner);
                Operand::Const(IrConst::I64(elem_size as i64))
            } else {
                Operand::Const(IrConst::I64(1))
            };
            let result = self.fresh_value();
            let ir_op = if op == UnaryOp::PreInc { IrBinOp::Add } else { IrBinOp::Sub };
            self.emit(Instruction::BinOp {
                dest: result,
                op: ir_op,
                lhs: Operand::Value(loaded_val),
                rhs: step,
                ty: IrType::I64,
            });
            self.store_lvalue_typed(&lv, Operand::Value(result), ty);
            return Operand::Value(result);
        }
        // Fallback
        self.lower_expr(inner)
    }

    /// Lower post-increment/decrement for any lvalue.
    pub(super) fn lower_post_inc_dec(&mut self, inner: &Expr, op: PostfixOp) -> Operand {
        let ty = self.get_expr_type(inner);
        if let Some(lv) = self.lower_lvalue(inner) {
            let loaded = self.load_lvalue_typed(&lv, ty);
            let loaded_val = match loaded.clone() {
                Operand::Value(v) => v,
                _ => {
                    let tmp = self.fresh_value();
                    self.emit(Instruction::Copy { dest: tmp, src: loaded.clone() });
                    tmp
                }
            };
            // For pointer types, increment by element size instead of 1
            let step = if ty == IrType::Ptr {
                let elem_size = self.get_pointer_elem_size(inner);
                Operand::Const(IrConst::I64(elem_size as i64))
            } else {
                Operand::Const(IrConst::I64(1))
            };
            let result = self.fresh_value();
            let ir_op = if op == PostfixOp::PostInc { IrBinOp::Add } else { IrBinOp::Sub };
            self.emit(Instruction::BinOp {
                dest: result,
                op: ir_op,
                lhs: Operand::Value(loaded_val),
                rhs: step,
                ty: IrType::I64,
            });
            self.store_lvalue_typed(&lv, Operand::Value(result), ty);
            // Return original value (before inc/dec)
            return loaded;
        }
        // Fallback
        self.lower_expr(inner)
    }

    /// Lower compound assignment (+=, -=, etc.) for any lvalue.
    pub(super) fn lower_compound_assign(&mut self, op: &BinOp, lhs: &Expr, rhs: &Expr) -> Operand {
        let rhs_val = self.lower_expr(rhs);
        let ty = self.get_expr_type(lhs);
        if let Some(lv) = self.lower_lvalue(lhs) {
            let loaded = self.load_lvalue_typed(&lv, ty);
            let ir_op = match op {
                BinOp::Add => IrBinOp::Add,
                BinOp::Sub => IrBinOp::Sub,
                BinOp::Mul => IrBinOp::Mul,
                BinOp::Div => IrBinOp::SDiv,
                BinOp::Mod => IrBinOp::SRem,
                BinOp::BitAnd => IrBinOp::And,
                BinOp::BitOr => IrBinOp::Or,
                BinOp::BitXor => IrBinOp::Xor,
                BinOp::Shl => IrBinOp::Shl,
                BinOp::Shr => IrBinOp::AShr,
                _ => IrBinOp::Add,
            };
            // For pointer += and -= operations, scale the RHS by element size
            let actual_rhs = if ty == IrType::Ptr && matches!(op, BinOp::Add | BinOp::Sub) {
                let elem_size = self.get_pointer_elem_size(lhs);
                if elem_size > 1 {
                    let scale = Operand::Const(IrConst::I64(elem_size as i64));
                    let scaled = self.fresh_value();
                    self.emit(Instruction::BinOp {
                        dest: scaled,
                        op: IrBinOp::Mul,
                        lhs: rhs_val,
                        rhs: scale,
                        ty: IrType::I64,
                    });
                    Operand::Value(scaled)
                } else {
                    rhs_val
                }
            } else {
                rhs_val
            };
            let result = self.fresh_value();
            self.emit(Instruction::BinOp {
                dest: result,
                op: ir_op,
                lhs: loaded,
                rhs: actual_rhs,
                ty: IrType::I64,
            });
            self.store_lvalue_typed(&lv, Operand::Value(result), ty);
            return Operand::Value(result);
        }
        // Fallback
        rhs_val
    }

    /// Infer the IrType of an expression based on available type information.
    /// Returns I64 if the type cannot be determined. This is used to select
    /// unsigned vs signed operations and insert narrowing casts.
    pub(super) fn infer_expr_type(&self, expr: &Expr) -> IrType {
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
            Expr::IntLiteral(val, _) => {
                if *val >= 0 && *val <= i32::MAX as i64 {
                    IrType::I32
                } else {
                    IrType::I64
                }
            }
            Expr::CharLiteral(_, _) => IrType::I8,
            Expr::Cast(ref target_type, _, _) => {
                self.type_spec_to_ir(target_type)
            }
            Expr::BinaryOp(op, lhs, rhs, _) => {
                match op {
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le |
                    BinOp::Gt | BinOp::Ge | BinOp::LogicalAnd | BinOp::LogicalOr => {
                        IrType::I32 // comparison results are int
                    }
                    _ => {
                        // For arithmetic, use usual arithmetic conversions:
                        // promote both operands and return the common type
                        let lt = self.infer_expr_type(lhs);
                        let rt = self.infer_expr_type(rhs);
                        Self::common_type(lt, rt)
                    }
                }
            }
            Expr::UnaryOp(_, inner, _) => self.infer_expr_type(inner),
            Expr::FunctionCall(func, _, _) => {
                // Check function return type
                if let Expr::Identifier(name, _) = func.as_ref() {
                    if let Some(&ret_ty) = self.function_return_types.get(name.as_str()) {
                        return ret_ty;
                    }
                }
                IrType::I64
            }
            _ => IrType::I64,
        }
    }

    /// Determine the common type for binary operation (usual arithmetic conversions, simplified).
    fn common_type(a: IrType, b: IrType) -> IrType {
        // If either is I64/U64/Ptr, result is 64-bit
        if a == IrType::I64 || a == IrType::U64 || a == IrType::Ptr
            || b == IrType::I64 || b == IrType::U64 || b == IrType::Ptr
        {
            if a == IrType::U64 || b == IrType::U64 {
                return IrType::U64;
            }
            return IrType::I64;
        }
        // Both are 32-bit or narrower
        if a == IrType::U32 || b == IrType::U32 {
            return IrType::U32;
        }
        if a == IrType::I32 || b == IrType::I32 {
            return IrType::I32;
        }
        // Narrow types get promoted to int
        IrType::I32
    }
}
