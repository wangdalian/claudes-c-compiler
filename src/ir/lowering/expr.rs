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

                let lhs_val = self.lower_expr(lhs);
                let rhs_val = self.lower_expr(rhs);
                let dest = self.fresh_value();

                match op {
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let cmp_op = match op {
                            BinOp::Eq => IrCmpOp::Eq,
                            BinOp::Ne => IrCmpOp::Ne,
                            BinOp::Lt => IrCmpOp::Slt,
                            BinOp::Le => IrCmpOp::Sle,
                            BinOp::Gt => IrCmpOp::Sgt,
                            BinOp::Ge => IrCmpOp::Sge,
                            _ => unreachable!(),
                        };
                        self.emit(Instruction::Cmp { dest, op: cmp_op, lhs: lhs_val, rhs: rhs_val, ty: IrType::I64 });
                    }
                    _ => {
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
                            _ => unreachable!(),
                        };
                        self.emit(Instruction::BinOp { dest, op: ir_op, lhs: lhs_val, rhs: rhs_val, ty: IrType::I64 });
                    }
                }

                Operand::Value(dest)
            }
            Expr::UnaryOp(op, inner, _) => {
                match op {
                    UnaryOp::Plus => {
                        // Unary plus is a no-op (identity), just lower the inner expression
                        self.lower_expr(inner)
                    }
                    UnaryOp::Neg => {
                        let val = self.lower_expr(inner);
                        let dest = self.fresh_value();
                        self.emit(Instruction::UnaryOp { dest, op: IrUnaryOp::Neg, src: val, ty: IrType::I64 });
                        Operand::Value(dest)
                    }
                    UnaryOp::BitNot => {
                        let val = self.lower_expr(inner);
                        let dest = self.fresh_value();
                        self.emit(Instruction::UnaryOp { dest, op: IrUnaryOp::Not, src: val, ty: IrType::I64 });
                        Operand::Value(dest)
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
                            // Direct function call
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

                Operand::Value(dest)
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
                // TODO: determine the type the pointer points to
                let deref_ty = IrType::I64;
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
            let one = Operand::Const(IrConst::I64(1));
            let result = self.fresh_value();
            let ir_op = if op == UnaryOp::PreInc { IrBinOp::Add } else { IrBinOp::Sub };
            self.emit(Instruction::BinOp {
                dest: result,
                op: ir_op,
                lhs: Operand::Value(loaded_val),
                rhs: one,
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
            let one = Operand::Const(IrConst::I64(1));
            let result = self.fresh_value();
            let ir_op = if op == PostfixOp::PostInc { IrBinOp::Add } else { IrBinOp::Sub };
            self.emit(Instruction::BinOp {
                dest: result,
                op: ir_op,
                lhs: Operand::Value(loaded_val),
                rhs: one,
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
            let result = self.fresh_value();
            self.emit(Instruction::BinOp {
                dest: result,
                op: ir_op,
                lhs: loaded,
                rhs: rhs_val,
                ty: IrType::I64,
            });
            self.store_lvalue_typed(&lv, Operand::Value(result), ty);
            return Operand::Value(result);
        }
        // Fallback
        rhs_val
    }
}
