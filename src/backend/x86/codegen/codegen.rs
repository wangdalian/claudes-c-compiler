use crate::ir::ir::*;
use crate::common::types::IrType;
use crate::backend::common::PtrDirective;
use crate::backend::codegen_shared::*;

/// x86-64 code generator. Implements the ArchCodegen trait for the shared framework.
/// Uses System V AMD64 ABI with stack-based allocation (no register allocator yet).
pub struct X86Codegen {
    state: CodegenState,
}

impl X86Codegen {
    pub fn new() -> Self {
        Self { state: CodegenState::new() }
    }

    pub fn generate(mut self, module: &IrModule) -> String {
        generate_module(&mut self, module)
    }

    // --- x86 helper methods ---

    /// Load an operand into %rax.
    fn operand_to_rax(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I8(v) => self.state.emit(&format!("    movq ${}, %rax", *v as i64)),
                    IrConst::I16(v) => self.state.emit(&format!("    movq ${}, %rax", *v as i64)),
                    IrConst::I32(v) => self.state.emit(&format!("    movq ${}, %rax", *v as i64)),
                    IrConst::I64(v) => {
                        if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                            self.state.emit(&format!("    movq ${}, %rax", v));
                        } else {
                            self.state.emit(&format!("    movabsq ${}, %rax", v));
                        }
                    }
                    IrConst::F32(_) | IrConst::F64(_) => {
                        // TODO: float constants
                        self.state.emit("    xorq %rax, %rax");
                    }
                    IrConst::Zero => self.state.emit("    xorq %rax, %rax"),
                }
            }
            Operand::Value(v) => {
                if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        self.state.emit(&format!("    leaq {}(%rbp), %rax", slot.0));
                    } else {
                        self.state.emit(&format!("    movq {}(%rbp), %rax", slot.0));
                    }
                } else {
                    self.state.emit("    xorq %rax, %rax");
                }
            }
        }
    }

    /// Store %rax to a value's stack slot.
    fn store_rax_to(&mut self, dest: &Value) {
        if let Some(slot) = self.state.get_slot(dest.0) {
            self.state.emit(&format!("    movq %rax, {}(%rbp)", slot.0));
        }
    }

    /// Get the store instruction mnemonic for a given type.
    fn mov_store_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 | IrType::U8 => "movb",
            IrType::I16 | IrType::U16 => "movw",
            IrType::I32 | IrType::U32 => "movl",
            _ => "movq",
        }
    }

    /// Get the load instruction mnemonic for a given type.
    fn mov_load_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 => "movsbq",
            IrType::U8 => "movzbq",
            IrType::I16 => "movswq",
            IrType::U16 => "movzwq",
            IrType::I32 => "movslq",
            IrType::U32 => "movl",     // movl zero-extends to 64-bit implicitly
            _ => "movq",
        }
    }

    /// Destination register for loads. U32 uses movl which needs %eax.
    fn load_dest_reg(ty: IrType) -> &'static str {
        match ty {
            IrType::U32 => "%eax",
            _ => "%rax",
        }
    }

    /// Map base register name + type to sized sub-register.
    fn reg_for_type(base_reg: &str, ty: IrType) -> &'static str {
        let (r8, r16, r32, r64) = match base_reg {
            "rax" => ("al", "ax", "eax", "rax"),
            "rcx" => ("cl", "cx", "ecx", "rcx"),
            "rdx" => ("dl", "dx", "edx", "rdx"),
            "rdi" => ("dil", "di", "edi", "rdi"),
            "rsi" => ("sil", "si", "esi", "rsi"),
            "r8"  => ("r8b", "r8w", "r8d", "r8"),
            "r9"  => ("r9b", "r9w", "r9d", "r9"),
            _ => return "rax",
        };
        match ty {
            IrType::I8 | IrType::U8 => r8,
            IrType::I16 | IrType::U16 => r16,
            IrType::I32 | IrType::U32 => r32,
            _ => r64,
        }
    }

    /// Emit a type cast instruction sequence.
    fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType) {
        if from_ty == to_ty {
            return;
        }
        if from_ty.size() == to_ty.size() && from_ty.is_integer() && to_ty.is_integer() {
            return;
        }

        let from_size = from_ty.size();
        let to_size = to_ty.size();

        if to_size > from_size {
            if from_ty.is_unsigned() {
                match from_ty {
                    IrType::U8 => self.state.emit("    movzbq %al, %rax"),
                    IrType::U16 => self.state.emit("    movzwq %ax, %rax"),
                    IrType::U32 => self.state.emit("    movl %eax, %eax"),
                    _ => {}
                }
            } else {
                match from_ty {
                    IrType::I8 => self.state.emit("    movsbq %al, %rax"),
                    IrType::I16 => self.state.emit("    movswq %ax, %rax"),
                    IrType::I32 => self.state.emit("    cltq"),
                    _ => {}
                }
            }
        } else if to_size == from_size {
            // Same size: pointer <-> integer conversions
            match (from_ty, to_ty) {
                (IrType::I32, IrType::Ptr) | (IrType::U32, IrType::Ptr) => {
                    self.state.emit("    movl %eax, %eax");
                }
                _ => {}
            }
        } else {
            // Narrowing: truncate then sign/zero-extend back to 64-bit
            // so that the full 64-bit register has the correct narrowed value.
            // This is necessary because we store all values as 64-bit movq.
            match to_ty {
                IrType::I8 => self.state.emit("    movsbq %al, %rax"),
                IrType::U8 => self.state.emit("    movzbq %al, %rax"),
                IrType::I16 => self.state.emit("    movswq %ax, %rax"),
                IrType::U16 => self.state.emit("    movzwq %ax, %rax"),
                IrType::I32 => self.state.emit("    movslq %eax, %rax"),
                IrType::U32 => self.state.emit("    movl %eax, %eax"),
                _ => {}
            }
        }
    }
}

const X86_ARG_REGS: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

impl ArchCodegen for X86Codegen {
    fn state(&mut self) -> &mut CodegenState { &mut self.state }
    fn state_ref(&self) -> &CodegenState { &self.state }
    fn ptr_directive(&self) -> PtrDirective { PtrDirective::Quad }

    fn calculate_stack_space(&mut self, func: &IrFunction) -> i64 {
        calculate_stack_space_common(&mut self.state, func, 0, |space, alloc_size| {
            // x86 uses negative offsets from rbp
            let new_space = space + ((alloc_size + 7) & !7);
            (-new_space, new_space)
        })
    }

    fn aligned_frame_size(&self, raw_space: i64) -> i64 {
        if raw_space > 0 { (raw_space + 15) & !15 } else { 0 }
    }

    fn emit_prologue(&mut self, _func: &IrFunction, frame_size: i64) {
        self.state.emit("    pushq %rbp");
        self.state.emit("    movq %rsp, %rbp");
        if frame_size > 0 {
            self.state.emit(&format!("    subq ${}, %rsp", frame_size));
        }
    }

    fn emit_epilogue(&mut self, _frame_size: i64) {
        self.state.emit("    movq %rbp, %rsp");
        self.state.emit("    popq %rbp");
    }

    fn emit_store_params(&mut self, func: &IrFunction) {
        for (i, param) in func.params.iter().enumerate() {
            if i >= 6 || param.name.is_empty() { continue; }
            if let Some((dest, ty)) = find_param_alloca(func, i) {
                if let Some(slot) = self.state.get_slot(dest.0) {
                    let store_instr = Self::mov_store_for_type(ty);
                    let reg = Self::reg_for_type(X86_ARG_REGS[i], ty);
                    self.state.emit(&format!("    {} %{}, {}(%rbp)", store_instr, reg, slot.0));
                }
            }
        }
    }

    fn emit_load_operand(&mut self, op: &Operand) {
        self.operand_to_rax(op);
    }

    fn emit_store_result(&mut self, dest: &Value) {
        self.store_rax_to(dest);
    }

    fn emit_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        self.operand_to_rax(val);
        if let Some(slot) = self.state.get_slot(ptr.0) {
            if self.state.is_alloca(ptr.0) {
                let store_instr = Self::mov_store_for_type(ty);
                let reg = Self::reg_for_type("rax", ty);
                self.state.emit(&format!("    {} %{}, {}(%rbp)", store_instr, reg, slot.0));
            } else {
                let store_instr = Self::mov_store_for_type(ty);
                self.state.emit("    movq %rax, %rcx");
                self.state.emit(&format!("    movq {}(%rbp), %rdx", slot.0));
                let store_reg = Self::reg_for_type("rcx", ty);
                self.state.emit(&format!("    {} %{}, (%rdx)", store_instr, store_reg));
            }
        }
    }

    fn emit_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if let Some(slot) = self.state.get_slot(ptr.0) {
            let load_instr = Self::mov_load_for_type(ty);
            let dest_reg = Self::load_dest_reg(ty);
            if self.state.is_alloca(ptr.0) {
                self.state.emit(&format!("    {} {}(%rbp), {}", load_instr, slot.0, dest_reg));
            } else {
                self.state.emit(&format!("    movq {}(%rbp), %rax", slot.0));
                self.state.emit(&format!("    {} (%rax), {}", load_instr, dest_reg));
            }
            self.store_rax_to(dest);
        }
    }

    fn emit_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        self.operand_to_rax(lhs);
        self.state.emit("    pushq %rax");
        self.operand_to_rax(rhs);
        self.state.emit("    movq %rax, %rcx");
        self.state.emit("    popq %rax");

        let use_32bit = ty == IrType::I32 || ty == IrType::U32;
        let is_unsigned = ty.is_unsigned();

        match op {
            IrBinOp::Add => {
                if use_32bit {
                    self.state.emit("    addl %ecx, %eax");
                    if !is_unsigned { self.state.emit("    cltq"); }
                } else {
                    self.state.emit("    addq %rcx, %rax");
                }
            }
            IrBinOp::Sub => {
                if use_32bit {
                    self.state.emit("    subl %ecx, %eax");
                    if !is_unsigned { self.state.emit("    cltq"); }
                } else {
                    self.state.emit("    subq %rcx, %rax");
                }
            }
            IrBinOp::Mul => {
                if use_32bit {
                    self.state.emit("    imull %ecx, %eax");
                    if !is_unsigned { self.state.emit("    cltq"); }
                } else {
                    self.state.emit("    imulq %rcx, %rax");
                }
            }
            IrBinOp::SDiv => {
                if use_32bit {
                    self.state.emit("    cltd");
                    self.state.emit("    idivl %ecx");
                    self.state.emit("    cltq");
                } else {
                    self.state.emit("    cqto");
                    self.state.emit("    idivq %rcx");
                }
            }
            IrBinOp::UDiv => {
                if use_32bit {
                    self.state.emit("    xorl %edx, %edx");
                    self.state.emit("    divl %ecx");
                } else {
                    self.state.emit("    xorq %rdx, %rdx");
                    self.state.emit("    divq %rcx");
                }
            }
            IrBinOp::SRem => {
                if use_32bit {
                    self.state.emit("    cltd");
                    self.state.emit("    idivl %ecx");
                    self.state.emit("    movl %edx, %eax");
                    self.state.emit("    cltq");
                } else {
                    self.state.emit("    cqto");
                    self.state.emit("    idivq %rcx");
                    self.state.emit("    movq %rdx, %rax");
                }
            }
            IrBinOp::URem => {
                if use_32bit {
                    self.state.emit("    xorl %edx, %edx");
                    self.state.emit("    divl %ecx");
                    self.state.emit("    movl %edx, %eax");
                } else {
                    self.state.emit("    xorq %rdx, %rdx");
                    self.state.emit("    divq %rcx");
                    self.state.emit("    movq %rdx, %rax");
                }
            }
            IrBinOp::And => self.state.emit("    andq %rcx, %rax"),
            IrBinOp::Or => self.state.emit("    orq %rcx, %rax"),
            IrBinOp::Xor => self.state.emit("    xorq %rcx, %rax"),
            IrBinOp::Shl => self.state.emit("    shlq %cl, %rax"),
            IrBinOp::AShr => self.state.emit("    sarq %cl, %rax"),
            IrBinOp::LShr => self.state.emit("    shrq %cl, %rax"),
        }

        self.store_rax_to(dest);
    }

    fn emit_unaryop(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand) {
        self.operand_to_rax(src);
        match op {
            IrUnaryOp::Neg => self.state.emit("    negq %rax"),
            IrUnaryOp::Not => self.state.emit("    notq %rax"),
        }
        self.store_rax_to(dest);
    }

    fn emit_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, _ty: IrType) {
        self.operand_to_rax(lhs);
        self.state.emit("    pushq %rax");
        self.operand_to_rax(rhs);
        self.state.emit("    movq %rax, %rcx");
        self.state.emit("    popq %rax");
        self.state.emit("    cmpq %rcx, %rax");

        let set_instr = match op {
            IrCmpOp::Eq => "sete",
            IrCmpOp::Ne => "setne",
            IrCmpOp::Slt => "setl",
            IrCmpOp::Sle => "setle",
            IrCmpOp::Sgt => "setg",
            IrCmpOp::Sge => "setge",
            IrCmpOp::Ult => "setb",
            IrCmpOp::Ule => "setbe",
            IrCmpOp::Ugt => "seta",
            IrCmpOp::Uge => "setae",
        };
        self.state.emit(&format!("    {} %al", set_instr));
        self.state.emit("    movzbq %al, %rax");
        self.store_rax_to(dest);
    }

    fn emit_call(&mut self, args: &[Operand], direct_name: Option<&str>,
                 func_ptr: Option<&Operand>, dest: Option<Value>) {
        let mut stack_args: Vec<&Operand> = Vec::new();
        let mut reg_args: Vec<(&Operand, usize)> = Vec::new();

        for (i, arg) in args.iter().enumerate() {
            if i < 6 {
                reg_args.push((arg, i));
            } else {
                stack_args.push(arg);
            }
        }

        // Push stack args in reverse order
        for arg in stack_args.iter().rev() {
            self.operand_to_rax(arg);
            self.state.emit("    pushq %rax");
        }

        // Load register args
        for (arg, i) in &reg_args {
            self.operand_to_rax(arg);
            self.state.emit(&format!("    movq %rax, %{}", X86_ARG_REGS[*i]));
        }

        // Zero AL for variadic functions (SysV ABI requirement)
        self.state.emit("    xorl %eax, %eax");

        if let Some(name) = direct_name {
            self.state.emit(&format!("    call {}", name));
        } else if let Some(ptr) = func_ptr {
            self.operand_to_rax(ptr);
            self.state.emit("    movq %rax, %r10");
            self.state.emit("    xorl %eax, %eax");
            self.state.emit("    call *%r10");
        }

        if !stack_args.is_empty() {
            self.state.emit(&format!("    addq ${}, %rsp", stack_args.len() * 8));
        }

        if let Some(dest) = dest {
            self.store_rax_to(&dest);
        }
    }

    fn emit_global_addr(&mut self, dest: &Value, name: &str) {
        self.state.emit(&format!("    leaq {}(%rip), %rax", name));
        self.store_rax_to(dest);
    }

    fn emit_gep(&mut self, dest: &Value, base: &Value, offset: &Operand) {
        if let Some(slot) = self.state.get_slot(base.0) {
            if self.state.is_alloca(base.0) {
                self.state.emit(&format!("    leaq {}(%rbp), %rax", slot.0));
            } else {
                self.state.emit(&format!("    movq {}(%rbp), %rax", slot.0));
            }
        }
        self.state.emit("    pushq %rax");
        self.operand_to_rax(offset);
        self.state.emit("    movq %rax, %rcx");
        self.state.emit("    popq %rax");
        self.state.emit("    addq %rcx, %rax");
        self.store_rax_to(dest);
    }

    fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) {
        self.operand_to_rax(src);
        self.emit_cast_instrs(from_ty, to_ty);
        self.store_rax_to(dest);
    }

    fn emit_return(&mut self, val: Option<&Operand>, _frame_size: i64) {
        if let Some(val) = val {
            self.operand_to_rax(val);
        }
        self.emit_epilogue(0); // frame_size not needed for x86 epilogue
        self.state.emit("    ret");
    }

    fn emit_branch(&mut self, label: &str) {
        self.state.emit(&format!("    jmp {}", label));
    }

    fn emit_cond_branch(&mut self, cond: &Operand, true_label: &str, false_label: &str) {
        self.operand_to_rax(cond);
        self.state.emit("    testq %rax, %rax");
        self.state.emit(&format!("    jne {}", true_label));
        self.state.emit(&format!("    jmp {}", false_label));
    }

    fn emit_unreachable(&mut self) {
        self.state.emit("    ud2");
    }
}

impl Default for X86Codegen {
    fn default() -> Self {
        Self::new()
    }
}
