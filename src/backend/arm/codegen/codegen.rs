use crate::ir::ir::*;
use crate::common::types::IrType;
use crate::backend::common::PtrDirective;
use crate::backend::codegen_shared::*;

/// AArch64 code generator. Implements the ArchCodegen trait for the shared framework.
/// Uses AAPCS64 calling convention with stack-based allocation.
pub struct ArmCodegen {
    state: CodegenState,
    /// Frame size for the current function (needed for epilogue in terminators).
    current_frame_size: i64,
}

impl ArmCodegen {
    pub fn new() -> Self {
        Self {
            state: CodegenState::new(),
            current_frame_size: 0,
        }
    }

    pub fn generate(mut self, module: &IrModule) -> String {
        generate_module(&mut self, module)
    }

    // --- AArch64 large-offset helpers ---

    /// Emit a large immediate subtraction from sp. For values > 4095, uses x16 as scratch.
    fn emit_sub_sp(&mut self, n: i64) {
        if n == 0 { return; }
        if n <= 4095 {
            self.state.emit(&format!("    sub sp, sp, #{}", n));
        } else if n <= 65535 {
            self.state.emit(&format!("    mov x16, #{}", n));
            self.state.emit("    sub sp, sp, x16");
        } else {
            self.state.emit(&format!("    movz x16, #{}", n & 0xFFFF));
            if (n >> 16) & 0xFFFF != 0 {
                self.state.emit(&format!("    movk x16, #{}, lsl #16", (n >> 16) & 0xFFFF));
            }
            if (n >> 32) & 0xFFFF != 0 {
                self.state.emit(&format!("    movk x16, #{}, lsl #32", (n >> 32) & 0xFFFF));
            }
            self.state.emit("    sub sp, sp, x16");
        }
    }

    /// Emit a large immediate addition to sp.
    fn emit_add_sp(&mut self, n: i64) {
        if n == 0 { return; }
        if n <= 4095 {
            self.state.emit(&format!("    add sp, sp, #{}", n));
        } else if n <= 65535 {
            self.state.emit(&format!("    mov x16, #{}", n));
            self.state.emit("    add sp, sp, x16");
        } else {
            self.state.emit(&format!("    movz x16, #{}", n & 0xFFFF));
            if (n >> 16) & 0xFFFF != 0 {
                self.state.emit(&format!("    movk x16, #{}, lsl #16", (n >> 16) & 0xFFFF));
            }
            if (n >> 32) & 0xFFFF != 0 {
                self.state.emit(&format!("    movk x16, #{}, lsl #32", (n >> 32) & 0xFFFF));
            }
            self.state.emit("    add sp, sp, x16");
        }
    }

    /// Emit store to [sp, #offset], handling large offsets via x16.
    fn emit_store_to_sp(&mut self, reg: &str, offset: i64, instr: &str) {
        if offset >= 0 && offset <= 32760 {
            self.state.emit(&format!("    {} {}, [sp, #{}]", instr, reg, offset));
        } else {
            self.load_large_imm("x16", offset);
            self.state.emit("    add x16, sp, x16");
            self.state.emit(&format!("    {} {}, [x16]", instr, reg));
        }
    }

    /// Emit load from [sp, #offset], handling large offsets via x16.
    fn emit_load_from_sp(&mut self, reg: &str, offset: i64, instr: &str) {
        if offset >= 0 && offset <= 32760 {
            self.state.emit(&format!("    {} {}, [sp, #{}]", instr, reg, offset));
        } else {
            self.load_large_imm("x16", offset);
            self.state.emit("    add x16, sp, x16");
            self.state.emit(&format!("    {} {}, [x16]", instr, reg));
        }
    }

    /// Emit `add dest, sp, #offset` handling large offsets.
    fn emit_add_sp_offset(&mut self, dest: &str, offset: i64) {
        if offset >= 0 && offset <= 4095 {
            self.state.emit(&format!("    add {}, sp, #{}", dest, offset));
        } else {
            self.load_large_imm("x16", offset);
            self.state.emit(&format!("    add {}, sp, x16", dest));
        }
    }

    /// Load a large immediate into a register.
    fn load_large_imm(&mut self, reg: &str, val: i64) {
        if val <= 65535 {
            self.state.emit(&format!("    mov {}, #{}", reg, val));
        } else {
            self.state.emit(&format!("    movz {}, #{}", reg, val & 0xFFFF));
            if (val >> 16) & 0xFFFF != 0 {
                self.state.emit(&format!("    movk {}, #{}, lsl #16", reg, (val >> 16) & 0xFFFF));
            }
        }
    }

    /// Emit function prologue: allocate stack and save fp/lr.
    fn emit_prologue_arm(&mut self, frame_size: i64) {
        if frame_size > 0 && frame_size <= 504 {
            self.state.emit(&format!("    stp x29, x30, [sp, #-{}]!", frame_size));
        } else {
            self.emit_sub_sp(frame_size);
            self.state.emit("    stp x29, x30, [sp]");
        }
        self.state.emit("    mov x29, sp");
    }

    /// Emit function epilogue: restore fp/lr and deallocate stack.
    fn emit_epilogue_arm(&mut self, frame_size: i64) {
        if frame_size > 0 && frame_size <= 504 {
            self.state.emit(&format!("    ldp x29, x30, [sp], #{}", frame_size));
        } else {
            self.state.emit("    ldp x29, x30, [sp]");
            self.emit_add_sp(frame_size);
        }
    }

    /// Load an operand into x0.
    fn operand_to_x0(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I8(v) => self.state.emit(&format!("    mov x0, #{}", v)),
                    IrConst::I16(v) => self.state.emit(&format!("    mov x0, #{}", v)),
                    IrConst::I32(v) => {
                        if *v >= 0 && *v <= 65535 {
                            self.state.emit(&format!("    mov x0, #{}", v));
                        } else if *v < 0 && *v >= -65536 {
                            self.state.emit(&format!("    mov x0, #{}", v));
                        } else {
                            self.state.emit(&format!("    mov x0, #{}", *v as u32 & 0xffff));
                            self.state.emit(&format!("    movk x0, #{}, lsl #16", (*v as u32 >> 16) & 0xffff));
                        }
                    }
                    IrConst::I64(v) => {
                        if *v >= 0 && *v <= 65535 {
                            self.state.emit(&format!("    mov x0, #{}", v));
                        } else if *v < 0 && *v >= -65536 {
                            self.state.emit(&format!("    mov x0, #{}", v));
                        } else {
                            self.state.emit(&format!("    mov x0, #{}", *v as u64 & 0xffff));
                            if (*v as u64 >> 16) & 0xffff != 0 {
                                self.state.emit(&format!("    movk x0, #{}, lsl #16", (*v as u64 >> 16) & 0xffff));
                            }
                            if (*v as u64 >> 32) & 0xffff != 0 {
                                self.state.emit(&format!("    movk x0, #{}, lsl #32", (*v as u64 >> 32) & 0xffff));
                            }
                            if (*v as u64 >> 48) & 0xffff != 0 {
                                self.state.emit(&format!("    movk x0, #{}, lsl #48", (*v as u64 >> 48) & 0xffff));
                            }
                        }
                    }
                    IrConst::F32(_) | IrConst::F64(_) => {
                        self.state.emit("    mov x0, #0");
                    }
                    IrConst::Zero => self.state.emit("    mov x0, #0"),
                }
            }
            Operand::Value(v) => {
                if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        self.emit_add_sp_offset("x0", slot.0);
                    } else {
                        self.emit_load_from_sp("x0", slot.0, "ldr");
                    }
                } else {
                    self.state.emit("    mov x0, #0");
                }
            }
        }
    }

    /// Store x0 to a value's stack slot.
    fn store_x0_to(&mut self, dest: &Value) {
        if let Some(slot) = self.state.get_slot(dest.0) {
            self.emit_store_to_sp("x0", slot.0, "str");
        }
    }

    fn str_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 | IrType::U8 => "strb",
            IrType::I16 | IrType::U16 => "strh",
            _ => "str",
        }
    }

    fn ldr_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 => "ldrsb",
            IrType::U8 => "ldrb",
            IrType::I16 => "ldrsh",
            IrType::U16 => "ldrh",
            IrType::I32 => "ldrsw",
            IrType::U32 => "ldr",
            _ => "ldr",
        }
    }

    fn load_dest_reg(ty: IrType) -> &'static str {
        match ty {
            IrType::U8 | IrType::U16 | IrType::U32 => "w0",
            _ => "x0",
        }
    }

    /// Get the appropriate register name for a given base and type.
    fn reg_for_type(base: &str, ty: IrType) -> &'static str {
        let use_w = matches!(ty,
            IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 |
            IrType::I32 | IrType::U32
        );
        match base {
            "x0" => if use_w { "w0" } else { "x0" },
            "x1" => if use_w { "w1" } else { "x1" },
            "x2" => if use_w { "w2" } else { "x2" },
            "x3" => if use_w { "w3" } else { "x3" },
            "x4" => if use_w { "w4" } else { "x4" },
            "x5" => if use_w { "w5" } else { "x5" },
            "x6" => if use_w { "w6" } else { "x6" },
            "x7" => if use_w { "w7" } else { "x7" },
            _ => "x0",
        }
    }

    /// Emit a type cast instruction sequence for AArch64.
    fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType) {
        if from_ty == to_ty { return; }
        if from_ty.size() == to_ty.size() && from_ty.is_integer() && to_ty.is_integer() {
            return;
        }

        let from_size = from_ty.size();
        let to_size = to_ty.size();

        if to_size > from_size {
            if from_ty.is_unsigned() {
                match from_ty {
                    IrType::U8 => self.state.emit("    and x0, x0, #0xff"),
                    IrType::U16 => self.state.emit("    and x0, x0, #0xffff"),
                    IrType::U32 => self.state.emit("    mov w0, w0"),
                    _ => {}
                }
            } else {
                match from_ty {
                    IrType::I8 => self.state.emit("    sxtb x0, w0"),
                    IrType::I16 => self.state.emit("    sxth x0, w0"),
                    IrType::I32 => self.state.emit("    sxtw x0, w0"),
                    _ => {}
                }
            }
        } else if to_size < from_size {
            // Narrowing: truncate and sign/zero-extend back to 64-bit
            match to_ty {
                IrType::I8 => self.state.emit("    sxtb x0, w0"),
                IrType::U8 => self.state.emit("    and x0, x0, #0xff"),
                IrType::I16 => self.state.emit("    sxth x0, w0"),
                IrType::U16 => self.state.emit("    and x0, x0, #0xffff"),
                IrType::I32 => self.state.emit("    sxtw x0, w0"),
                IrType::U32 => self.state.emit("    mov w0, w0"),
                _ => {}
            }
        }
        // Same size: pointer <-> integer conversions (no-op on AArch64)
    }
}

const ARM_ARG_REGS: [&str; 8] = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
const ARM_TMP_REGS: [&str; 8] = ["x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16"];

impl ArchCodegen for ArmCodegen {
    fn state(&mut self) -> &mut CodegenState { &mut self.state }
    fn state_ref(&self) -> &CodegenState { &self.state }
    fn ptr_directive(&self) -> PtrDirective { PtrDirective::Xword }
    fn function_type_directive(&self) -> &'static str { "%function" }

    fn calculate_stack_space(&mut self, func: &IrFunction) -> i64 {
        calculate_stack_space_common(&mut self.state, func, 16, |space, alloc_size| {
            // ARM uses positive offsets from sp, starting at 16 (after fp/lr)
            let slot = space;
            let new_space = space + ((alloc_size + 7) & !7).max(8);
            (slot, new_space)
        })
    }

    fn aligned_frame_size(&self, raw_space: i64) -> i64 {
        (raw_space + 15) & !15
    }

    fn emit_prologue(&mut self, _func: &IrFunction, frame_size: i64) {
        self.current_frame_size = frame_size;
        self.emit_prologue_arm(frame_size);
    }

    fn emit_epilogue(&mut self, frame_size: i64) {
        self.emit_epilogue_arm(frame_size);
    }

    fn emit_store_params(&mut self, func: &IrFunction) {
        for (i, param) in func.params.iter().enumerate() {
            if i >= 8 || param.name.is_empty() { continue; }
            if let Some((dest, ty)) = find_param_alloca(func, i) {
                if let Some(slot) = self.state.get_slot(dest.0) {
                    let store_instr = Self::str_for_type(ty);
                    let reg = Self::reg_for_type(ARM_ARG_REGS[i], ty);
                    self.emit_store_to_sp(reg, slot.0, store_instr);
                }
            }
        }
    }

    fn emit_load_operand(&mut self, op: &Operand) {
        self.operand_to_x0(op);
    }

    fn emit_store_result(&mut self, dest: &Value) {
        self.store_x0_to(dest);
    }

    fn emit_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        self.operand_to_x0(val);
        if let Some(slot) = self.state.get_slot(ptr.0) {
            if self.state.is_alloca(ptr.0) {
                let store_instr = Self::str_for_type(ty);
                let reg = Self::reg_for_type("x0", ty);
                self.emit_store_to_sp(reg, slot.0, store_instr);
            } else {
                self.state.emit("    mov x1, x0");
                self.emit_load_from_sp("x2", slot.0, "ldr");
                let store_instr = Self::str_for_type(ty);
                let reg = Self::reg_for_type("x1", ty);
                self.state.emit(&format!("    {} {}, [x2]", store_instr, reg));
            }
        }
    }

    fn emit_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if let Some(slot) = self.state.get_slot(ptr.0) {
            let load_instr = Self::ldr_for_type(ty);
            let dest_reg = Self::load_dest_reg(ty);
            if self.state.is_alloca(ptr.0) {
                self.emit_load_from_sp(dest_reg, slot.0, load_instr);
            } else {
                self.emit_load_from_sp("x0", slot.0, "ldr");
                self.state.emit("    mov x1, x0");
                self.state.emit(&format!("    {} {}, [x1]", load_instr, dest_reg));
            }
            self.store_x0_to(dest);
        }
    }

    fn emit_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        self.operand_to_x0(lhs);
        self.state.emit("    mov x1, x0");
        self.operand_to_x0(rhs);
        self.state.emit("    mov x2, x0");

        let use_32bit = ty == IrType::I32 || ty == IrType::U32;
        let is_unsigned = ty.is_unsigned();

        if use_32bit {
            match op {
                IrBinOp::Add => {
                    self.state.emit("    add w0, w1, w2");
                    if !is_unsigned { self.state.emit("    sxtw x0, w0"); }
                }
                IrBinOp::Sub => {
                    self.state.emit("    sub w0, w1, w2");
                    if !is_unsigned { self.state.emit("    sxtw x0, w0"); }
                }
                IrBinOp::Mul => {
                    self.state.emit("    mul w0, w1, w2");
                    if !is_unsigned { self.state.emit("    sxtw x0, w0"); }
                }
                IrBinOp::SDiv => {
                    self.state.emit("    sdiv w0, w1, w2");
                    self.state.emit("    sxtw x0, w0");
                }
                IrBinOp::UDiv => self.state.emit("    udiv w0, w1, w2"),
                IrBinOp::SRem => {
                    self.state.emit("    sdiv w3, w1, w2");
                    self.state.emit("    msub w0, w3, w2, w1");
                    self.state.emit("    sxtw x0, w0");
                }
                IrBinOp::URem => {
                    self.state.emit("    udiv w3, w1, w2");
                    self.state.emit("    msub w0, w3, w2, w1");
                }
                IrBinOp::And => self.state.emit("    and w0, w1, w2"),
                IrBinOp::Or => self.state.emit("    orr w0, w1, w2"),
                IrBinOp::Xor => self.state.emit("    eor w0, w1, w2"),
                IrBinOp::Shl => self.state.emit("    lsl w0, w1, w2"),
                IrBinOp::AShr => self.state.emit("    asr w0, w1, w2"),
                IrBinOp::LShr => self.state.emit("    lsr w0, w1, w2"),
            }
        } else {
            match op {
                IrBinOp::Add => self.state.emit("    add x0, x1, x2"),
                IrBinOp::Sub => self.state.emit("    sub x0, x1, x2"),
                IrBinOp::Mul => self.state.emit("    mul x0, x1, x2"),
                IrBinOp::SDiv => self.state.emit("    sdiv x0, x1, x2"),
                IrBinOp::UDiv => self.state.emit("    udiv x0, x1, x2"),
                IrBinOp::SRem => {
                    self.state.emit("    sdiv x3, x1, x2");
                    self.state.emit("    msub x0, x3, x2, x1");
                }
                IrBinOp::URem => {
                    self.state.emit("    udiv x3, x1, x2");
                    self.state.emit("    msub x0, x3, x2, x1");
                }
                IrBinOp::And => self.state.emit("    and x0, x1, x2"),
                IrBinOp::Or => self.state.emit("    orr x0, x1, x2"),
                IrBinOp::Xor => self.state.emit("    eor x0, x1, x2"),
                IrBinOp::Shl => self.state.emit("    lsl x0, x1, x2"),
                IrBinOp::AShr => self.state.emit("    asr x0, x1, x2"),
                IrBinOp::LShr => self.state.emit("    lsr x0, x1, x2"),
            }
        }

        self.store_x0_to(dest);
    }

    fn emit_unaryop(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand) {
        self.operand_to_x0(src);
        match op {
            IrUnaryOp::Neg => self.state.emit("    neg x0, x0"),
            IrUnaryOp::Not => self.state.emit("    mvn x0, x0"),
        }
        self.store_x0_to(dest);
    }

    fn emit_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        self.operand_to_x0(lhs);
        self.state.emit("    mov x1, x0");
        self.operand_to_x0(rhs);
        let use_32bit = ty == IrType::I32 || ty == IrType::U32
            || ty == IrType::I8 || ty == IrType::U8
            || ty == IrType::I16 || ty == IrType::U16;
        if use_32bit {
            self.state.emit("    cmp w1, w0");
        } else {
            self.state.emit("    cmp x1, x0");
        }

        let cond = match op {
            IrCmpOp::Eq => "eq",
            IrCmpOp::Ne => "ne",
            IrCmpOp::Slt => "lt",
            IrCmpOp::Sle => "le",
            IrCmpOp::Sgt => "gt",
            IrCmpOp::Sge => "ge",
            IrCmpOp::Ult => "lo",
            IrCmpOp::Ule => "ls",
            IrCmpOp::Ugt => "hi",
            IrCmpOp::Uge => "hs",
        };
        self.state.emit(&format!("    cset x0, {}", cond));
        self.store_x0_to(dest);
    }

    fn emit_call(&mut self, args: &[Operand], direct_name: Option<&str>,
                 func_ptr: Option<&Operand>, dest: Option<Value>) {
        let num_args = args.len().min(8);

        // Load all args into temp registers first (to avoid clobbering)
        for (i, arg) in args.iter().enumerate().take(num_args) {
            self.operand_to_x0(arg);
            self.state.emit(&format!("    mov {}, x0", ARM_TMP_REGS[i]));
        }
        // Move from temps to arg registers
        for i in 0..num_args {
            self.state.emit(&format!("    mov {}, {}", ARM_ARG_REGS[i], ARM_TMP_REGS[i]));
        }

        if let Some(name) = direct_name {
            self.state.emit(&format!("    bl {}", name));
        } else if let Some(ptr) = func_ptr {
            self.operand_to_x0(ptr);
            self.state.emit("    mov x17, x0");
            self.state.emit("    blr x17");
        }

        if let Some(dest) = dest {
            self.store_x0_to(&dest);
        }
    }

    fn emit_global_addr(&mut self, dest: &Value, name: &str) {
        self.state.emit(&format!("    adrp x0, {}", name));
        self.state.emit(&format!("    add x0, x0, :lo12:{}", name));
        self.store_x0_to(dest);
    }

    fn emit_gep(&mut self, dest: &Value, base: &Value, offset: &Operand) {
        if let Some(slot) = self.state.get_slot(base.0) {
            if self.state.is_alloca(base.0) {
                self.emit_add_sp_offset("x1", slot.0);
            } else {
                self.emit_load_from_sp("x1", slot.0, "ldr");
            }
        }
        self.operand_to_x0(offset);
        self.state.emit("    add x0, x1, x0");
        self.store_x0_to(dest);
    }

    fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) {
        self.operand_to_x0(src);
        self.emit_cast_instrs(from_ty, to_ty);
        self.store_x0_to(dest);
    }

    fn emit_return(&mut self, val: Option<&Operand>, frame_size: i64) {
        if let Some(val) = val {
            self.operand_to_x0(val);
        }
        self.emit_epilogue_arm(frame_size);
        self.state.emit("    ret");
    }

    fn emit_branch(&mut self, label: &str) {
        self.state.emit(&format!("    b {}", label));
    }

    fn emit_cond_branch(&mut self, cond: &Operand, true_label: &str, false_label: &str) {
        self.operand_to_x0(cond);
        self.state.emit(&format!("    cbnz x0, {}", true_label));
        self.state.emit(&format!("    b {}", false_label));
    }

    fn emit_unreachable(&mut self) {
        self.state.emit("    brk #0");
    }
}

impl Default for ArmCodegen {
    fn default() -> Self {
        Self::new()
    }
}
