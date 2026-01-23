use crate::ir::ir::*;
use crate::common::types::IrType;
use crate::backend::common::PtrDirective;
use crate::backend::codegen_shared::*;

/// RISC-V 64 code generator. Implements the ArchCodegen trait for the shared framework.
/// Uses standard RISC-V calling convention with stack-based allocation.
pub struct RiscvCodegen {
    state: CodegenState,
    current_return_type: IrType,
    /// For variadic functions: offset from SP where the register save area starts.
    va_save_area_offset: i64,
    /// Number of named integer params for current variadic function.
    va_named_gp_count: usize,
    /// Current frame size.
    current_frame_size: i64,
}

impl RiscvCodegen {
    pub fn new() -> Self {
        Self {
            state: CodegenState::new(),
            current_return_type: IrType::I64,
            va_save_area_offset: 0,
            va_named_gp_count: 0,
            current_frame_size: 0,
        }
    }

    pub fn generate(mut self, module: &IrModule) -> String {
        generate_module(&mut self, module)
    }

    // --- RISC-V helpers ---

    /// Check if an immediate fits in a 12-bit signed field.
    fn fits_imm12(val: i64) -> bool {
        val >= -2048 && val <= 2047
    }

    /// Emit: store `reg` to `offset(s0)`, handling large offsets via t5.
    fn emit_store_to_s0(&mut self, reg: &str, offset: i64, store_instr: &str) {
        if Self::fits_imm12(offset) {
            self.state.emit(&format!("    {} {}, {}(s0)", store_instr, reg, offset));
        } else {
            self.state.emit(&format!("    li t5, {}", offset));
            self.state.emit("    add t5, s0, t5");
            self.state.emit(&format!("    {} {}, 0(t5)", store_instr, reg));
        }
    }

    /// Emit: load from `offset(s0)` into `reg`, handling large offsets via t5.
    fn emit_load_from_s0(&mut self, reg: &str, offset: i64, load_instr: &str) {
        if Self::fits_imm12(offset) {
            self.state.emit(&format!("    {} {}, {}(s0)", load_instr, reg, offset));
        } else {
            self.state.emit(&format!("    li t5, {}", offset));
            self.state.emit("    add t5, s0, t5");
            self.state.emit(&format!("    {} {}, 0(t5)", load_instr, reg));
        }
    }

    /// Emit: `dest_reg = s0 + offset`, handling large offsets.
    fn emit_addi_s0(&mut self, dest_reg: &str, offset: i64) {
        if Self::fits_imm12(offset) {
            self.state.emit(&format!("    addi {}, s0, {}", dest_reg, offset));
        } else {
            self.state.emit(&format!("    li {}, {}", dest_reg, offset));
            self.state.emit(&format!("    add {}, s0, {}", dest_reg, dest_reg));
        }
    }

    /// Emit prologue: allocate stack and save ra/s0.
    ///
    /// Stack layout (s0 points to top of frame = old sp):
    ///   s0 - 8:  saved ra
    ///   s0 - 16: saved s0
    ///   s0 - 16 - ...: local data (allocas and value slots)
    ///   sp: bottom of frame
    fn emit_prologue_riscv(&mut self, frame_size: i64) {
        // Small-frame path requires ALL immediates to fit in 12 bits:
        // -frame_size (sp adjust), frame_size-8 and frame_size-16 (save offsets),
        // and frame_size (s0 setup). Since fits_imm12 checks [-2048, 2047],
        // we check both -frame_size AND frame_size.
        if Self::fits_imm12(-frame_size) && Self::fits_imm12(frame_size) {
            // Small frame: all offsets fit in 12-bit immediates
            self.state.emit(&format!("    addi sp, sp, -{}", frame_size));
            self.state.emit(&format!("    sd ra, {}(sp)", frame_size - 8));
            self.state.emit(&format!("    sd s0, {}(sp)", frame_size - 16));
            self.state.emit(&format!("    addi s0, sp, {}", frame_size));
        } else {
            // Large frame: save ra/s0 at top of frame (s0-8, s0-16) to avoid
            // collision with local data that grows downward from s0-16.
            self.state.emit(&format!("    li t0, {}", frame_size));
            self.state.emit("    sub sp, sp, t0");
            // t0 still has frame_size; compute s0 = sp + frame_size = old_sp
            self.state.emit("    add t0, sp, t0");
            // Save ra and old s0 at top of frame (relative to new s0)
            self.state.emit("    sd ra, -8(t0)");
            self.state.emit("    sd s0, -16(t0)");
            self.state.emit("    mv s0, t0");
        }
    }

    /// Emit epilogue: restore ra/s0 and deallocate stack.
    fn emit_epilogue_riscv(&mut self, frame_size: i64) {
        if Self::fits_imm12(-frame_size) && Self::fits_imm12(frame_size) {
            // Small frame: restore from known sp offsets
            self.state.emit(&format!("    ld ra, {}(sp)", frame_size - 8));
            self.state.emit(&format!("    ld s0, {}(sp)", frame_size - 16));
            self.state.emit(&format!("    addi sp, sp, {}", frame_size));
        } else {
            // Large frame: restore from s0-relative offsets (always fit in imm12).
            // Load saved values before adjusting sp to avoid reading below sp.
            self.state.emit("    ld ra, -8(s0)");
            self.state.emit("    ld t0, -16(s0)");
            self.state.emit("    mv sp, s0");
            self.state.emit("    mv s0, t0");
        }
    }

    /// Load an operand into t0.
    fn operand_to_t0(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                match c {
                    IrConst::I8(v) => self.state.emit(&format!("    li t0, {}", v)),
                    IrConst::I16(v) => self.state.emit(&format!("    li t0, {}", v)),
                    IrConst::I32(v) => self.state.emit(&format!("    li t0, {}", v)),
                    IrConst::I64(v) => self.state.emit(&format!("    li t0, {}", v)),
                    IrConst::F32(v) => {
                        let bits = v.to_bits() as u64;
                        self.state.emit(&format!("    li t0, {}", bits as i64));
                    }
                    IrConst::F64(v) => {
                        let bits = v.to_bits();
                        self.state.emit(&format!("    li t0, {}", bits as i64));
                    }
                    // LongDouble at computation level is treated as F64
                    IrConst::LongDouble(v) => {
                        let bits = v.to_bits();
                        self.state.emit(&format!("    li t0, {}", bits as i64));
                    }
                    IrConst::Zero => self.state.emit("    li t0, 0"),
                }
            }
            Operand::Value(v) => {
                if let Some(slot) = self.state.get_slot(v.0) {
                    if self.state.is_alloca(v.0) {
                        self.emit_addi_s0("t0", slot.0);
                    } else {
                        self.emit_load_from_s0("t0", slot.0, "ld");
                    }
                } else {
                    self.state.emit("    li t0, 0");
                }
            }
        }
    }

    /// Store t0 to a value's stack slot.
    fn store_t0_to(&mut self, dest: &Value) {
        if let Some(slot) = self.state.get_slot(dest.0) {
            self.emit_store_to_s0("t0", slot.0, "sd");
        }
    }

    fn store_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 | IrType::U8 => "sb",
            IrType::I16 | IrType::U16 => "sh",
            IrType::I32 | IrType::U32 | IrType::F32 => "sw",
            _ => "sd",
        }
    }

    fn load_for_type(ty: IrType) -> &'static str {
        match ty {
            IrType::I8 => "lb",
            IrType::U8 => "lbu",
            IrType::I16 => "lh",
            IrType::U16 => "lhu",
            IrType::I32 => "lw",
            IrType::U32 | IrType::F32 => "lwu",
            _ => "ld",
        }
    }

    /// Emit a type cast instruction sequence for RISC-V 64.
    /// Emit RISC-V instructions for a type cast, using shared cast classification.
    fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType) {
        match classify_cast(from_ty, to_ty) {
            CastKind::Noop => {}

            CastKind::FloatToSigned { from_f64 } => {
                if from_f64 {
                    self.state.emit("    fmv.d.x ft0, t0");
                    self.state.emit("    fcvt.l.d t0, ft0, rtz");
                } else {
                    self.state.emit("    fmv.w.x ft0, t0");
                    self.state.emit("    fcvt.l.s t0, ft0, rtz");
                }
            }

            CastKind::FloatToUnsigned { from_f64, .. } => {
                if from_f64 {
                    self.state.emit("    fmv.d.x ft0, t0");
                    self.state.emit("    fcvt.lu.d t0, ft0, rtz");
                } else {
                    self.state.emit("    fmv.w.x ft0, t0");
                    self.state.emit("    fcvt.lu.s t0, ft0, rtz");
                }
            }

            CastKind::SignedToFloat { to_f64 } => {
                if to_f64 {
                    self.state.emit("    fcvt.d.l ft0, t0");
                    self.state.emit("    fmv.x.d t0, ft0");
                } else {
                    self.state.emit("    fcvt.s.l ft0, t0");
                    self.state.emit("    fmv.x.w t0, ft0");
                }
            }

            CastKind::UnsignedToFloat { to_f64, .. } => {
                if to_f64 {
                    self.state.emit("    fcvt.d.lu ft0, t0");
                    self.state.emit("    fmv.x.d t0, ft0");
                } else {
                    self.state.emit("    fcvt.s.lu ft0, t0");
                    self.state.emit("    fmv.x.w t0, ft0");
                }
            }

            CastKind::FloatToFloat { widen } => {
                if widen {
                    self.state.emit("    fmv.w.x ft0, t0");
                    self.state.emit("    fcvt.d.s ft0, ft0");
                    self.state.emit("    fmv.x.d t0, ft0");
                } else {
                    self.state.emit("    fmv.d.x ft0, t0");
                    self.state.emit("    fcvt.s.d ft0, ft0");
                    self.state.emit("    fmv.x.w t0, ft0");
                }
            }

            CastKind::SignedToUnsignedSameSize { .. } => {
                // On RISC-V 64, same-size signed->unsigned is a noop
            }

            CastKind::IntWiden { from_ty, .. } => {
                if from_ty.is_unsigned() {
                    match from_ty {
                        IrType::U8 => self.state.emit("    andi t0, t0, 0xff"),
                        IrType::U16 => {
                            self.state.emit("    slli t0, t0, 48");
                            self.state.emit("    srli t0, t0, 48");
                        }
                        IrType::U32 => {
                            self.state.emit("    slli t0, t0, 32");
                            self.state.emit("    srli t0, t0, 32");
                        }
                        _ => {}
                    }
                } else {
                    match from_ty {
                        IrType::I8 => {
                            self.state.emit("    slli t0, t0, 56");
                            self.state.emit("    srai t0, t0, 56");
                        }
                        IrType::I16 => {
                            self.state.emit("    slli t0, t0, 48");
                            self.state.emit("    srai t0, t0, 48");
                        }
                        IrType::I32 => self.state.emit("    sext.w t0, t0"),
                        _ => {}
                    }
                }
            }

            CastKind::IntNarrow { to_ty } => {
                match to_ty {
                    IrType::I8 => {
                        self.state.emit("    slli t0, t0, 56");
                        self.state.emit("    srai t0, t0, 56");
                    }
                    IrType::U8 => self.state.emit("    andi t0, t0, 0xff"),
                    IrType::I16 => {
                        self.state.emit("    slli t0, t0, 48");
                        self.state.emit("    srai t0, t0, 48");
                    }
                    IrType::U16 => {
                        self.state.emit("    slli t0, t0, 48");
                        self.state.emit("    srli t0, t0, 48");
                    }
                    IrType::I32 => self.state.emit("    sext.w t0, t0"),
                    IrType::U32 => {
                        self.state.emit("    slli t0, t0, 32");
                        self.state.emit("    srli t0, t0, 32");
                    }
                    _ => {}
                }
            }
        }
    }
}

const RISCV_ARG_REGS: [&str; 8] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"];

impl ArchCodegen for RiscvCodegen {
    fn state(&mut self) -> &mut CodegenState { &mut self.state }
    fn state_ref(&self) -> &CodegenState { &self.state }
    fn ptr_directive(&self) -> PtrDirective { PtrDirective::Dword }

    fn calculate_stack_space(&mut self, func: &IrFunction) -> i64 {
        let mut space = calculate_stack_space_common(&mut self.state, func, 16, |space, alloc_size| {
            // RISC-V uses negative offsets from s0 (frame pointer)
            let new_space = space + ((alloc_size + 7) & !7).max(8);
            (-(new_space as i64), new_space)
        });

        // For variadic functions, reserve space for the register save area (a0-a7 = 64 bytes)
        if func.is_variadic {
            space = (space + 7) & !7; // align
            self.va_save_area_offset = space;
            space += 64; // 8 registers * 8 bytes

            // Count named params. On RISC-V, all params in variadic functions
            // go through integer registers (a0-a7), including named float params.
            self.va_named_gp_count = func.params.len().min(8);
        }

        space
    }

    fn aligned_frame_size(&self, raw_space: i64) -> i64 {
        (raw_space + 15) & !15
    }

    fn emit_prologue(&mut self, func: &IrFunction, frame_size: i64) {
        self.current_return_type = func.return_type;
        self.current_frame_size = frame_size;
        self.emit_prologue_riscv(frame_size);
    }

    fn emit_epilogue(&mut self, frame_size: i64) {
        self.emit_epilogue_riscv(frame_size);
    }

    fn emit_store_params(&mut self, func: &IrFunction) {
        // For variadic functions: save all integer register args (a0-a7) to the save area.
        // Layout: a0 at lowest offset, a7 at highest offset, so va_arg can advance by +8.
        // save_area starts at -(va_save_area_offset + 64) and ends at -(va_save_area_offset + 8).
        if func.is_variadic {
            for i in 0..8usize {
                let offset = -(self.va_save_area_offset as i64) - 64 + (i as i64) * 8;
                self.emit_store_to_s0(RISCV_ARG_REGS[i], offset, "sd");
            }
        }

        let float_arg_regs = ["fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"];
        let mut int_reg_idx = 0usize;
        let mut float_reg_idx = 0usize;
        // Stack-passed params are at positive offsets from s0 (s0 = old sp)
        let mut stack_param_offset: i64 = 0;
        for (_i, param) in func.params.iter().enumerate() {
            let is_float = param.ty.is_float();
            let is_stack_passed = if is_float { float_reg_idx >= 8 } else { int_reg_idx >= 8 };

            if param.name.is_empty() {
                if is_stack_passed {
                    stack_param_offset += 8;
                } else if is_float {
                    float_reg_idx += 1;
                } else {
                    int_reg_idx += 1;
                }
                continue;
            }
            if let Some((dest, ty)) = find_param_alloca(func, _i) {
                if let Some(slot) = self.state.get_slot(dest.0) {
                    if is_stack_passed {
                        // Stack-passed parameter: load from positive s0 offset
                        self.emit_load_from_s0("t0", stack_param_offset, "ld");
                        let store_instr = Self::store_for_type(ty);
                        self.emit_store_to_s0("t0", slot.0, store_instr);
                        stack_param_offset += 8;
                    } else if is_float {
                        // Float params arrive in fa0-fa7 per RISC-V calling convention
                        if ty == IrType::F32 {
                            // F32 param: extract 32-bit float from fa-reg
                            self.state.emit(&format!("    fmv.x.w t0, {}", float_arg_regs[float_reg_idx]));
                        } else {
                            // F64 param: extract 64-bit double from fa-reg
                            self.state.emit(&format!("    fmv.x.d t0, {}", float_arg_regs[float_reg_idx]));
                        }
                        self.emit_store_to_s0("t0", slot.0, "sd");
                        float_reg_idx += 1;
                    } else {
                        let store_instr = Self::store_for_type(ty);
                        self.emit_store_to_s0(RISCV_ARG_REGS[int_reg_idx], slot.0, store_instr);
                        int_reg_idx += 1;
                    }
                }
            } else {
                if is_stack_passed {
                    stack_param_offset += 8;
                } else if is_float {
                    float_reg_idx += 1;
                } else {
                    int_reg_idx += 1;
                }
            }
        }
    }

    fn emit_load_operand(&mut self, op: &Operand) {
        self.operand_to_t0(op);
    }

    fn emit_store_result(&mut self, dest: &Value) {
        self.store_t0_to(dest);
    }

    fn emit_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        self.operand_to_t0(val);
        if let Some(slot) = self.state.get_slot(ptr.0) {
            if self.state.is_alloca(ptr.0) {
                let store_instr = Self::store_for_type(ty);
                self.emit_store_to_s0("t0", slot.0, store_instr);
            } else {
                self.state.emit("    mv t3, t0");
                self.emit_load_from_s0("t4", slot.0, "ld");
                let store_instr = Self::store_for_type(ty);
                self.state.emit(&format!("    {} t3, 0(t4)", store_instr));
            }
        }
    }

    fn emit_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if let Some(slot) = self.state.get_slot(ptr.0) {
            if self.state.is_alloca(ptr.0) {
                let load_instr = Self::load_for_type(ty);
                self.emit_load_from_s0("t0", slot.0, load_instr);
            } else {
                self.emit_load_from_s0("t0", slot.0, "ld");
                let load_instr = Self::load_for_type(ty);
                self.state.emit(&format!("    {} t0, 0(t0)", load_instr));
            }
            self.store_t0_to(dest);
        }
    }

    fn emit_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if ty.is_float() {
            let float_op = classify_float_binop(op).unwrap_or(FloatOp::Add);
            let mnemonic = match float_op {
                FloatOp::Add => "fadd",
                FloatOp::Sub => "fsub",
                FloatOp::Mul => "fmul",
                FloatOp::Div => "fdiv",
            };
            self.operand_to_t0(lhs);
            self.state.emit("    mv t1, t0");
            self.operand_to_t0(rhs);
            self.state.emit("    mv t2, t0");
            if ty == IrType::F64 {
                self.state.emit("    fmv.d.x ft0, t1");
                self.state.emit("    fmv.d.x ft1, t2");
                self.state.emit(&format!("    {}.d ft0, ft0, ft1", mnemonic));
                self.state.emit("    fmv.x.d t0, ft0");
            } else {
                self.state.emit("    fmv.w.x ft0, t1");
                self.state.emit("    fmv.w.x ft1, t2");
                self.state.emit(&format!("    {}.s ft0, ft0, ft1", mnemonic));
                self.state.emit("    fmv.x.w t0, ft0");
            }
            self.store_t0_to(dest);
            return;
        }

        self.operand_to_t0(lhs);
        self.state.emit("    mv t1, t0");
        self.operand_to_t0(rhs);
        self.state.emit("    mv t2, t0");

        let use_32bit = ty == IrType::I32 || ty == IrType::U32;

        if use_32bit {
            match op {
                IrBinOp::Add => self.state.emit("    addw t0, t1, t2"),
                IrBinOp::Sub => self.state.emit("    subw t0, t1, t2"),
                IrBinOp::Mul => self.state.emit("    mulw t0, t1, t2"),
                IrBinOp::SDiv => self.state.emit("    divw t0, t1, t2"),
                IrBinOp::UDiv => self.state.emit("    divuw t0, t1, t2"),
                IrBinOp::SRem => self.state.emit("    remw t0, t1, t2"),
                IrBinOp::URem => self.state.emit("    remuw t0, t1, t2"),
                IrBinOp::And => self.state.emit("    and t0, t1, t2"),
                IrBinOp::Or => self.state.emit("    or t0, t1, t2"),
                IrBinOp::Xor => self.state.emit("    xor t0, t1, t2"),
                IrBinOp::Shl => self.state.emit("    sllw t0, t1, t2"),
                IrBinOp::AShr => self.state.emit("    sraw t0, t1, t2"),
                IrBinOp::LShr => self.state.emit("    srlw t0, t1, t2"),
            }
        } else {
            match op {
                IrBinOp::Add => self.state.emit("    add t0, t1, t2"),
                IrBinOp::Sub => self.state.emit("    sub t0, t1, t2"),
                IrBinOp::Mul => self.state.emit("    mul t0, t1, t2"),
                IrBinOp::SDiv => self.state.emit("    div t0, t1, t2"),
                IrBinOp::UDiv => self.state.emit("    divu t0, t1, t2"),
                IrBinOp::SRem => self.state.emit("    rem t0, t1, t2"),
                IrBinOp::URem => self.state.emit("    remu t0, t1, t2"),
                IrBinOp::And => self.state.emit("    and t0, t1, t2"),
                IrBinOp::Or => self.state.emit("    or t0, t1, t2"),
                IrBinOp::Xor => self.state.emit("    xor t0, t1, t2"),
                IrBinOp::Shl => self.state.emit("    sll t0, t1, t2"),
                IrBinOp::AShr => self.state.emit("    sra t0, t1, t2"),
                IrBinOp::LShr => self.state.emit("    srl t0, t1, t2"),
            }
        }

        self.store_t0_to(dest);
    }

    fn emit_unaryop(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand, ty: IrType) {
        self.operand_to_t0(src);
        if ty.is_float() {
            match op {
                IrUnaryOp::Neg => {
                    if ty == IrType::F64 {
                        self.state.emit("    fmv.d.x ft0, t0");
                        self.state.emit("    fneg.d ft0, ft0");
                        self.state.emit("    fmv.x.d t0, ft0");
                    } else {
                        self.state.emit("    fmv.w.x ft0, t0");
                        self.state.emit("    fneg.s ft0, ft0");
                        self.state.emit("    fmv.x.w t0, ft0");
                    }
                }
                IrUnaryOp::Not => self.state.emit("    not t0, t0"),
                _ => {} // Clz/Ctz/Bswap/Popcount not applicable to floats
            }
        } else {
            match op {
                IrUnaryOp::Neg => self.state.emit("    neg t0, t0"),
                IrUnaryOp::Not => self.state.emit("    not t0, t0"),
                // RISC-V doesn't have native CLZ/CTZ/BSWAP/POPCOUNT without Zbb extension
                // Emit a software fallback via loop for now
                IrUnaryOp::Clz | IrUnaryOp::Ctz | IrUnaryOp::Bswap | IrUnaryOp::Popcount => {
                    // TODO: implement RISC-V bit manipulation
                    self.state.emit("    li t0, 0");
                }
            }
        }
        self.store_t0_to(dest);
    }

    fn emit_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        self.operand_to_t0(lhs);
        self.state.emit("    mv t1, t0");
        self.operand_to_t0(rhs);
        self.state.emit("    mv t2, t0");

        if ty.is_float() {
            if ty == IrType::F64 {
                self.state.emit("    fmv.d.x ft0, t1");
                self.state.emit("    fmv.d.x ft1, t2");
                match op {
                    IrCmpOp::Eq => self.state.emit("    feq.d t0, ft0, ft1"),
                    IrCmpOp::Ne => {
                        self.state.emit("    feq.d t0, ft0, ft1");
                        self.state.emit("    xori t0, t0, 1");
                    }
                    IrCmpOp::Slt | IrCmpOp::Ult => self.state.emit("    flt.d t0, ft0, ft1"),
                    IrCmpOp::Sle | IrCmpOp::Ule => self.state.emit("    fle.d t0, ft0, ft1"),
                    IrCmpOp::Sgt | IrCmpOp::Ugt => self.state.emit("    flt.d t0, ft1, ft0"),
                    IrCmpOp::Sge | IrCmpOp::Uge => self.state.emit("    fle.d t0, ft1, ft0"),
                }
            } else {
                self.state.emit("    fmv.w.x ft0, t1");
                self.state.emit("    fmv.w.x ft1, t2");
                match op {
                    IrCmpOp::Eq => self.state.emit("    feq.s t0, ft0, ft1"),
                    IrCmpOp::Ne => {
                        self.state.emit("    feq.s t0, ft0, ft1");
                        self.state.emit("    xori t0, t0, 1");
                    }
                    IrCmpOp::Slt | IrCmpOp::Ult => self.state.emit("    flt.s t0, ft0, ft1"),
                    IrCmpOp::Sle | IrCmpOp::Ule => self.state.emit("    fle.s t0, ft0, ft1"),
                    IrCmpOp::Sgt | IrCmpOp::Ugt => self.state.emit("    flt.s t0, ft1, ft0"),
                    IrCmpOp::Sge | IrCmpOp::Uge => self.state.emit("    fle.s t0, ft1, ft0"),
                }
            }
            self.store_t0_to(dest);
            return;
        }

        // For sub-64-bit types, sign/zero-extend before comparison
        let is_32bit = ty == IrType::I32 || ty == IrType::U32
            || ty == IrType::I8 || ty == IrType::U8
            || ty == IrType::I16 || ty == IrType::U16;
        if is_32bit && ty.is_unsigned() {
            self.state.emit("    slli t1, t1, 32");
            self.state.emit("    srli t1, t1, 32");
            self.state.emit("    slli t2, t2, 32");
            self.state.emit("    srli t2, t2, 32");
        } else if is_32bit {
            self.state.emit("    sext.w t1, t1");
            self.state.emit("    sext.w t2, t2");
        }

        match op {
            IrCmpOp::Eq => {
                self.state.emit("    sub t0, t1, t2");
                self.state.emit("    seqz t0, t0");
            }
            IrCmpOp::Ne => {
                self.state.emit("    sub t0, t1, t2");
                self.state.emit("    snez t0, t0");
            }
            IrCmpOp::Slt => self.state.emit("    slt t0, t1, t2"),
            IrCmpOp::Ult => self.state.emit("    sltu t0, t1, t2"),
            IrCmpOp::Sge => {
                self.state.emit("    slt t0, t1, t2");
                self.state.emit("    xori t0, t0, 1");
            }
            IrCmpOp::Uge => {
                self.state.emit("    sltu t0, t1, t2");
                self.state.emit("    xori t0, t0, 1");
            }
            IrCmpOp::Sgt => self.state.emit("    slt t0, t2, t1"),
            IrCmpOp::Ugt => self.state.emit("    sltu t0, t2, t1"),
            IrCmpOp::Sle => {
                self.state.emit("    slt t0, t2, t1");
                self.state.emit("    xori t0, t0, 1");
            }
            IrCmpOp::Ule => {
                self.state.emit("    sltu t0, t2, t1");
                self.state.emit("    xori t0, t0, 1");
            }
        }

        self.store_t0_to(dest);
    }

    fn emit_call(&mut self, args: &[Operand], arg_types: &[IrType], direct_name: Option<&str>,
                 func_ptr: Option<&Operand>, dest: Option<Value>, return_type: IrType,
                 is_variadic: bool, _num_fixed_args: usize) {
        let float_arg_regs = ["fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"];

        // Phase 1: Classify all args into register assignments or stack overflow.
        // 'I' = GP register, 'F' = FP register, 'Q' = F128 GP register pair, 'S' = stack, 'T' = F128 stack
        let mut arg_classes: Vec<char> = Vec::new();
        let mut arg_int_indices: Vec<usize> = Vec::new(); // GP reg index for 'I' or base index for 'Q'
        let mut arg_fp_indices: Vec<usize> = Vec::new();  // FP reg index for 'F'
        let mut int_idx = 0usize;
        let mut float_idx = 0usize;

        for (i, _arg) in args.iter().enumerate() {
            let is_long_double = if i < arg_types.len() {
                arg_types[i].is_long_double()
            } else {
                false
            };
            let is_float_arg = if i < arg_types.len() {
                arg_types[i].is_float()
            } else {
                matches!(args[i], Operand::Const(IrConst::F32(_) | IrConst::F64(_)))
            };

            if is_long_double {
                // Align int_idx to even for register pair
                if int_idx % 2 != 0 {
                    int_idx += 1;
                }
                if int_idx + 1 < 8 {
                    arg_classes.push('Q');
                    arg_int_indices.push(int_idx);
                    arg_fp_indices.push(0);
                    int_idx += 2;
                } else {
                    arg_classes.push('T'); // F128 stack overflow
                    arg_int_indices.push(0);
                    arg_fp_indices.push(0);
                    int_idx = 8;
                }
            } else if is_float_arg && !is_variadic && float_idx < 8 {
                arg_classes.push('F');
                arg_int_indices.push(0);
                arg_fp_indices.push(float_idx);
                float_idx += 1;
            } else if int_idx < 8 {
                arg_classes.push('I');
                arg_int_indices.push(int_idx);
                arg_fp_indices.push(0);
                int_idx += 1;
            } else {
                arg_classes.push('S');
                arg_int_indices.push(0);
                arg_fp_indices.push(0);
            }
        }

        // Phase 2: Handle F128 variable args that need __extenddftf2.
        // Call __extenddftf2 for each, save results to stack temporaries.
        // Count how many F128 variable reg args we have.
        let mut f128_var_temps: Vec<(usize, i64)> = Vec::new(); // (arg_index, sp_offset of saved lo:hi)
        let mut f128_temp_space: i64 = 0;
        for (i, arg) in args.iter().enumerate() {
            if arg_classes[i] == 'Q' {
                if let Operand::Value(_) = arg {
                    f128_temp_space += 16;
                    f128_var_temps.push((i, 0)); // offset filled below
                }
            }
        }

        if f128_temp_space > 0 {
            // Allocate temp space for F128 conversion results
            self.state.emit(&format!("    addi sp, sp, -{}", f128_temp_space));
            let mut temp_offset: i64 = 0;
            for item in &mut f128_var_temps {
                item.1 = temp_offset;
                let arg = &args[item.0];
                // Load f64 value, call __extenddftf2, save result
                self.operand_to_t0(arg);
                self.state.emit("    fmv.d.x fa0, t0");
                self.state.emit("    call __extenddftf2");
                // Save a0:a1 to temp space
                self.state.emit(&format!("    sd a0, {}(sp)", temp_offset));
                self.state.emit(&format!("    sd a1, {}(sp)", temp_offset + 8));
                temp_offset += 16;
            }
        }

        // Phase 3: Handle stack overflow args.
        let stack_args: Vec<(usize, bool)> = args.iter().enumerate()
            .filter(|(i, _)| arg_classes[*i] == 'S' || arg_classes[*i] == 'T')
            .map(|(i, _)| (i, arg_classes[i] == 'T'))
            .collect();

        let mut stack_arg_space: usize = 0;
        if !stack_args.is_empty() {
            for &(_, is_f128) in &stack_args {
                if is_f128 {
                    stack_arg_space = (stack_arg_space + 15) & !15;
                    stack_arg_space += 16;
                } else {
                    stack_arg_space += 8;
                }
            }
            stack_arg_space = (stack_arg_space + 15) & !15;
            self.state.emit(&format!("    addi sp, sp, -{}", stack_arg_space));
            let mut offset: usize = 0;
            for &(arg_i, is_f128) in &stack_args {
                if is_f128 {
                    offset = (offset + 15) & !15;
                    match &args[arg_i] {
                        Operand::Const(ref c) => {
                            let f64_val = match c {
                                IrConst::LongDouble(v) => *v,
                                IrConst::F64(v) => *v,
                                _ => c.to_f64().unwrap_or(0.0),
                            };
                            let bytes = crate::ir::ir::f64_to_f128_bytes(f64_val);
                            let lo = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
                            let hi = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
                            self.state.emit(&format!("    li t0, {}", lo));
                            self.state.emit(&format!("    sd t0, {}(sp)", offset));
                            self.state.emit(&format!("    li t0, {}", hi));
                            self.state.emit(&format!("    sd t0, {}(sp)", offset + 8));
                        }
                        Operand::Value(ref _v) => {
                            self.operand_to_t0(&args[arg_i]);
                            self.state.emit("    fmv.d.x fa0, t0");
                            self.state.emit("    call __extenddftf2");
                            self.state.emit(&format!("    sd a0, {}(sp)", offset));
                            self.state.emit(&format!("    sd a1, {}(sp)", offset + 8));
                        }
                    }
                    offset += 16;
                } else {
                    self.operand_to_t0(&args[arg_i]);
                    self.state.emit(&format!("    sd t0, {}(sp)", offset));
                    offset += 8;
                }
            }
        }

        // Phase 4: Load non-F128 register args.
        // Load all non-F128 args into their target registers.
        // Process GP args into t3-t6 first (temp), then FP args, then move GP from temp.
        let mut gp_temps: Vec<(usize, &str)> = Vec::new(); // (target_reg_idx, temp_reg)
        let temp_regs = ["t3", "t4", "t5", "t6", "s2", "s3", "s4", "s5"];
        let mut temp_i = 0usize;
        for (i, arg) in args.iter().enumerate() {
            if arg_classes[i] == 'I' {
                self.operand_to_t0(arg);
                if temp_i < temp_regs.len() {
                    self.state.emit(&format!("    mv {}, t0", temp_regs[temp_i]));
                    gp_temps.push((arg_int_indices[i], temp_regs[temp_i]));
                    temp_i += 1;
                }
            } else if arg_classes[i] == 'F' {
                let fp_i = arg_fp_indices[i];
                self.operand_to_t0(arg);
                let arg_ty = if i < arg_types.len() { Some(arg_types[i]) } else { None };
                if arg_ty == Some(IrType::F32) {
                    self.state.emit(&format!("    fmv.w.x {}, t0", float_arg_regs[fp_i]));
                } else {
                    self.state.emit(&format!("    fmv.d.x {}, t0", float_arg_regs[fp_i]));
                }
            }
        }

        // Phase 5: Move GP args from temps to actual arg registers.
        for (target_idx, temp_reg) in &gp_temps {
            self.state.emit(&format!("    mv {}, {}", RISCV_ARG_REGS[*target_idx], temp_reg));
        }

        // Phase 6: Load F128 register args into their target register pairs.
        for (i, _arg) in args.iter().enumerate() {
            if arg_classes[i] != 'Q' { continue; }
            let base_reg = arg_int_indices[i];
            match &args[i] {
                Operand::Const(ref c) => {
                    let f64_val = match c {
                        IrConst::LongDouble(v) => *v,
                        IrConst::F64(v) => *v,
                        _ => c.to_f64().unwrap_or(0.0),
                    };
                    let bytes = crate::ir::ir::f64_to_f128_bytes(f64_val);
                    let lo = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
                    let hi = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
                    self.state.emit(&format!("    li {}, {}", RISCV_ARG_REGS[base_reg], lo));
                    self.state.emit(&format!("    li {}, {}", RISCV_ARG_REGS[base_reg + 1], hi));
                }
                Operand::Value(_) => {
                    // Load from temp space (saved in Phase 2)
                    let temp_info = f128_var_temps.iter().find(|t| t.0 == i).unwrap();
                    // Adjust offset for stack_arg_space
                    let offset = temp_info.1 + stack_arg_space as i64;
                    self.state.emit(&format!("    ld {}, {}(sp)", RISCV_ARG_REGS[base_reg], offset));
                    self.state.emit(&format!("    ld {}, {}(sp)", RISCV_ARG_REGS[base_reg + 1], offset + 8));
                }
            }
        }

        // Clean up F128 temp space before the call (only if no stack overflow args below it)
        if f128_temp_space > 0 && stack_arg_space == 0 {
            self.state.emit(&format!("    addi sp, sp, {}", f128_temp_space));
        }

        if let Some(name) = direct_name {
            self.state.emit(&format!("    call {}", name));
        } else if let Some(ptr) = func_ptr {
            self.operand_to_t0(ptr);
            self.state.emit("    mv t2, t0");
            self.state.emit("    jalr ra, t2, 0");
        }

        // Clean up stack space after call
        if stack_arg_space > 0 {
            // Both stack overflow args and f128 temp space (if any) need cleanup
            let cleanup = stack_arg_space + f128_temp_space as usize;
            self.state.emit(&format!("    addi sp, sp, {}", cleanup));
        }

        if let Some(dest) = dest {
            if let Some(slot) = self.state.get_slot(dest.0) {
                if return_type == IrType::F32 {
                    // F32 return value is in fa0 as single-precision
                    self.state.emit("    fmv.x.w t0, fa0");
                    self.emit_store_to_s0("t0", slot.0, "sd");
                } else if return_type.is_float() {
                    // F64 return value is in fa0 as double-precision
                    self.state.emit("    fmv.x.d t0, fa0");
                    self.emit_store_to_s0("t0", slot.0, "sd");
                } else {
                    self.emit_store_to_s0("a0", slot.0, "sd");
                }
            }
        }
    }

    fn emit_global_addr(&mut self, dest: &Value, name: &str) {
        self.state.emit(&format!("    la t0, {}", name));
        self.store_t0_to(dest);
    }

    fn emit_gep(&mut self, dest: &Value, base: &Value, offset: &Operand) {
        if let Some(slot) = self.state.get_slot(base.0) {
            if self.state.is_alloca(base.0) {
                self.emit_addi_s0("t1", slot.0);
            } else {
                self.emit_load_from_s0("t1", slot.0, "ld");
            }
        }
        self.operand_to_t0(offset);
        self.state.emit("    add t0, t1, t0");
        self.store_t0_to(dest);
    }

    fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) {
        self.operand_to_t0(src);
        self.emit_cast_instrs(from_ty, to_ty);
        self.store_t0_to(dest);
    }

    fn emit_memcpy(&mut self, dest: &Value, src: &Value, size: usize) {
        // Load dest address into t1, src address into t2
        if let Some(dst_slot) = self.state.get_slot(dest.0) {
            if self.state.is_alloca(dest.0) {
                self.emit_addi_s0("t1", dst_slot.0);
            } else {
                self.emit_load_from_s0("t1", dst_slot.0, "ld");
            }
        }
        if let Some(src_slot) = self.state.get_slot(src.0) {
            if self.state.is_alloca(src.0) {
                self.emit_addi_s0("t2", src_slot.0);
            } else {
                self.emit_load_from_s0("t2", src_slot.0, "ld");
            }
        }
        // Inline byte-by-byte copy using a loop
        let label_id = self.state.next_label_id();
        let loop_label = format!(".Lmemcpy_loop_{}", label_id);
        let done_label = format!(".Lmemcpy_done_{}", label_id);
        self.state.emit(&format!("    li t3, {}", size));
        self.state.emit(&format!("{}:", loop_label));
        self.state.emit(&format!("    beqz t3, {}", done_label));
        self.state.emit("    lbu t4, 0(t2)");
        self.state.emit("    sb t4, 0(t1)");
        self.state.emit("    addi t1, t1, 1");
        self.state.emit("    addi t2, t2, 1");
        self.state.emit("    addi t3, t3, -1");
        self.state.emit(&format!("    j {}", loop_label));
        self.state.emit(&format!("{}:", done_label));
    }

    fn emit_va_arg(&mut self, dest: &Value, va_list_ptr: &Value, result_ty: IrType) {
        // RISC-V LP64D: va_list is just a void* (pointer to the next arg on stack).
        // Load va_list pointer address into t1
        if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
            if self.state.is_alloca(va_list_ptr.0) {
                self.emit_addi_s0("t1", slot.0);
            } else {
                self.emit_load_from_s0("t1", slot.0, "ld");
            }
        }
        // Load the current va_list pointer value (points to next arg)
        self.state.emit("    ld t2, 0(t1)");

        if result_ty.is_long_double() {
            // F128 (long double): 16 bytes, 16-byte aligned.
            // Align t2 to 16 bytes: t2 = (t2 + 15) & ~15
            self.state.emit("    addi t2, t2, 15");
            self.state.emit("    andi t2, t2, -16");
            // Load 16 bytes (f128) into a0:a1 for __trunctfdf2
            self.state.emit("    ld a0, 0(t2)");    // lo 8 bytes
            self.state.emit("    ld a1, 8(t2)");    // hi 8 bytes
            // Advance pointer by 16
            self.state.emit("    addi t2, t2, 16");
            self.state.emit("    sd t2, 0(t1)");
            // Convert f128 (in a0:a1) to f64 using __trunctfdf2
            // __trunctfdf2 on RISC-V: takes f128 in a0:a1, returns f64 in fa0
            self.state.emit("    call __trunctfdf2");
            // Move f64 result from fa0 to t0 (bit pattern)
            self.state.emit("    fmv.x.d t0, fa0");
        } else {
            // Standard 8-byte arg
            self.state.emit("    ld t0, 0(t2)");
            // Advance pointer by 8
            self.state.emit("    addi t2, t2, 8");
            self.state.emit("    sd t2, 0(t1)");
        }
        // Store result
        self.store_t0_to(dest);
    }

    fn emit_va_start(&mut self, va_list_ptr: &Value) {
        // RISC-V LP64D: va_list = pointer to first variadic arg.
        // For variadic functions, we save a0-a7 to the register save area.
        // The variadic args start after the named GP params.
        if let Some(slot) = self.state.get_slot(va_list_ptr.0) {
            if self.state.is_alloca(va_list_ptr.0) {
                self.emit_addi_s0("t0", slot.0);
            } else {
                self.emit_load_from_s0("t0", slot.0, "ld");
            }
        }

        if self.va_named_gp_count < 8 {
            // Variadic args start in the register save area, after named params.
            // a0 is at -(va_save_area_offset + 64), a1 at -(va_save_area_offset + 56), etc.
            // Variadic args start at a[named_gp_count], which is at:
            let vararg_offset = -(self.va_save_area_offset as i64) - 64 + (self.va_named_gp_count as i64) * 8;
            self.emit_addi_s0("t1", vararg_offset);
        } else {
            // All registers used by named params; variadic args are on the caller's stack.
            // s0 points to the old sp, and stack-passed args start at s0 + 0 (the return
            // address slot in the caller's frame is above, args are at positive offsets).
            self.state.emit("    mv t1, s0");
        }
        self.state.emit("    sd t1, 0(t0)");
    }

    fn emit_va_end(&mut self, _va_list_ptr: &Value) {
        // va_end is a no-op on RISC-V
    }

    fn emit_va_copy(&mut self, dest_ptr: &Value, src_ptr: &Value) {
        // Copy va_list (just 8 bytes on RISC-V - a single pointer)
        if let Some(src_slot) = self.state.get_slot(src_ptr.0) {
            if self.state.is_alloca(src_ptr.0) {
                self.emit_addi_s0("t1", src_slot.0);
            } else {
                self.emit_load_from_s0("t1", src_slot.0, "ld");
            }
        }
        self.state.emit("    ld t2, 0(t1)");
        if let Some(dest_slot) = self.state.get_slot(dest_ptr.0) {
            if self.state.is_alloca(dest_ptr.0) {
                self.emit_addi_s0("t0", dest_slot.0);
            } else {
                self.emit_load_from_s0("t0", dest_slot.0, "ld");
            }
        }
        self.state.emit("    sd t2, 0(t0)");
    }

    fn emit_return(&mut self, val: Option<&Operand>, frame_size: i64) {
        if let Some(val) = val {
            self.operand_to_t0(val);
            if self.current_return_type == IrType::F32 {
                // F32 return: bit pattern in t0, move to fa0 as single-precision
                self.state.emit("    fmv.w.x fa0, t0");
            } else if self.current_return_type.is_float() {
                // F64 return: bit pattern in t0, move to fa0 as double-precision
                self.state.emit("    fmv.d.x fa0, t0");
            } else {
                self.state.emit("    mv a0, t0");
            }
        }
        self.emit_epilogue_riscv(frame_size);
        self.state.emit("    ret");
    }

    fn emit_branch(&mut self, label: &str) {
        self.state.emit(&format!("    j {}", label));
    }

    fn emit_cond_branch(&mut self, cond: &Operand, true_label: &str, false_label: &str) {
        self.operand_to_t0(cond);
        self.state.emit(&format!("    bnez t0, {}", true_label));
        self.state.emit(&format!("    j {}", false_label));
    }

    fn emit_unreachable(&mut self) {
        self.state.emit("    ebreak");
    }

    fn emit_label_addr(&mut self, dest: &Value, label: &str) {
        // Load address of a label for computed goto (GCC &&label extension)
        self.state.emit(&format!("    la t0, {}", label));
        self.store_t0_to(dest);
    }

    fn emit_indirect_branch(&mut self, target: &Operand) {
        // Computed goto: goto *target
        self.operand_to_t0(target);
        self.state.emit("    jr t0");
    }

    fn emit_atomic_rmw(&mut self, dest: &Value, op: AtomicRmwOp, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) {
        // Load ptr into t1, val into t2
        self.operand_to_t0(ptr);
        self.state.emit("    mv t1, t0"); // t1 = ptr
        self.operand_to_t0(val);
        self.state.emit("    mv t2, t0"); // t2 = val

        let aq_rl = Self::amo_ordering(ordering);
        let suffix = Self::amo_width_suffix(ty);

        match op {
            AtomicRmwOp::Add => {
                self.state.emit(&format!("    amoadd.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
            AtomicRmwOp::Sub => {
                // No amosub; negate and use amoadd
                self.state.emit("    neg t2, t2");
                self.state.emit(&format!("    amoadd.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
            AtomicRmwOp::And => {
                self.state.emit(&format!("    amoand.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
            AtomicRmwOp::Or => {
                self.state.emit(&format!("    amoor.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
            AtomicRmwOp::Xor => {
                self.state.emit(&format!("    amoxor.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
            AtomicRmwOp::Xchg => {
                self.state.emit(&format!("    amoswap.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
            AtomicRmwOp::Nand => {
                // No amonand; use lr/sc loop
                let label_id = self.state.next_label_id();
                let loop_label = format!(".Latomic_{}", label_id);
                self.state.emit(&format!("{}:", loop_label));
                self.state.emit(&format!("    lr.{}{} t0, (t1)", suffix, aq_rl));
                self.state.emit("    and t3, t0, t2");
                self.state.emit("    not t3, t3");
                self.state.emit(&format!("    sc.{}{} t4, t3, (t1)", suffix, aq_rl));
                self.state.emit(&format!("    bnez t4, {}", loop_label));
            }
            AtomicRmwOp::TestAndSet => {
                // test_and_set: set byte to 1, return old
                self.state.emit("    li t2, 1");
                self.state.emit(&format!("    amoswap.{}{} t0, t2, (t1)", suffix, aq_rl));
            }
        }
        // Sign-extend result for sub-word types
        Self::sign_extend_riscv(&mut self.state, ty);
        self.store_t0_to(dest);
    }

    fn emit_atomic_cmpxchg(&mut self, dest: &Value, ptr: &Operand, expected: &Operand, desired: &Operand, ty: IrType, ordering: AtomicOrdering, _failure_ordering: AtomicOrdering, returns_bool: bool) {
        // t1 = ptr, t2 = expected, t3 = desired
        self.operand_to_t0(ptr);
        self.state.emit("    mv t1, t0");
        self.operand_to_t0(desired);
        self.state.emit("    mv t3, t0");
        self.operand_to_t0(expected);
        self.state.emit("    mv t2, t0");

        let aq_rl = Self::amo_ordering(ordering);
        let suffix = Self::amo_width_suffix(ty);

        let label_id = self.state.next_label_id();
        let loop_label = format!(".Lcas_loop_{}", label_id);
        let fail_label = format!(".Lcas_fail_{}", label_id);
        let done_label = format!(".Lcas_done_{}", label_id);

        self.state.emit(&format!("{}:", loop_label));
        self.state.emit(&format!("    lr.{}{} t0, (t1)", suffix, aq_rl));
        // Mask for sub-word comparison
        Self::mask_for_cmp(&mut self.state, ty);
        self.state.emit(&format!("    bne t0, t2, {}", fail_label));
        self.state.emit(&format!("    sc.{}{} t4, t3, (t1)", suffix, aq_rl));
        self.state.emit(&format!("    bnez t4, {}", loop_label));
        if returns_bool {
            self.state.emit("    li t0, 1");
        }
        self.state.emit(&format!("    j {}", done_label));
        self.state.emit(&format!("{}:", fail_label));
        if returns_bool {
            self.state.emit("    li t0, 0");
        }
        // t0 has old value if !returns_bool
        self.state.emit(&format!("{}:", done_label));
        self.store_t0_to(dest);
    }

    fn emit_atomic_load(&mut self, dest: &Value, ptr: &Operand, ty: IrType, _ordering: AtomicOrdering) {
        self.operand_to_t0(ptr);
        // Use lr for atomic load (conservative but correct)
        let suffix = Self::amo_width_suffix(ty);
        self.state.emit(&format!("    lr.{}.aq t0, (t0)", suffix));
        Self::sign_extend_riscv(&mut self.state, ty);
        self.store_t0_to(dest);
    }

    fn emit_atomic_store(&mut self, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) {
        self.operand_to_t0(val);
        self.state.emit("    mv t1, t0"); // t1 = val
        self.operand_to_t0(ptr);
        // Use amoswap with zero dest for atomic store
        let aq_rl = Self::amo_ordering(ordering);
        let suffix = Self::amo_width_suffix(ty);
        self.state.emit(&format!("    amoswap.{}{} zero, t1, (t0)", suffix, aq_rl));
    }

    fn emit_fence(&mut self, _ordering: AtomicOrdering) {
        self.state.emit("    fence rw, rw");
    }

    fn emit_inline_asm(&mut self, template: &str, outputs: &[(String, Value, Option<String>)], inputs: &[(String, Operand, Option<String>)], _clobbers: &[String], _operand_types: &[IrType]) {
        // RISC-V inline assembly support.
        // Allocate temporary registers for operands, substitute %0/%1/%[name],
        // load inputs and store outputs.

        // Scratch registers for generic "r" constraints.
        // Use t0-t6 (temporaries) and a2-a7 (args not typically in use during asm).
        let gp_scratch: &[&str] = &["t0", "t1", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4", "a5", "a6", "a7"];
        let mut scratch_idx = 0;

        // Map from specific register constraint names to RISC-V register names
        fn specific_reg(constraint: &str) -> Option<&'static str> {
            let c = constraint.trim_start_matches(|c: char| c == '=' || c == '+' || c == '&');
            match c {
                "a0" => Some("a0"),
                "a1" => Some("a1"),
                "a2" => Some("a2"),
                "a3" => Some("a3"),
                "a4" => Some("a4"),
                "a5" => Some("a5"),
                "a6" => Some("a6"),
                "a7" => Some("a7"),
                "ra" => Some("ra"),
                "t0" => Some("t0"),
                "t1" => Some("t1"),
                "t2" => Some("t2"),
                _ => None,
            }
        }

        fn is_memory_constraint(constraint: &str) -> bool {
            let c = constraint.trim_start_matches(|c: char| c == '=' || c == '+' || c == '&');
            c == "m"
        }

        fn tied_operand(constraint: &str) -> Option<usize> {
            let c = constraint.trim_start_matches(|c: char| c == '=' || c == '+' || c == '&');
            if !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit()) {
                c.parse::<usize>().ok()
            } else {
                None
            }
        }

        let total_operands = outputs.len() + inputs.len();
        let mut op_regs: Vec<String> = vec![String::new(); total_operands];
        let mut op_is_memory: Vec<bool> = vec![false; total_operands];
        let mut op_names: Vec<Option<String>> = vec![None; total_operands];
        let mut op_mem_offsets: Vec<i64> = vec![0; total_operands]; // stack offset for memory ops

        // First pass: assign specific registers and mark memory operands
        for (i, (constraint, ptr, name)) in outputs.iter().enumerate() {
            op_names[i] = name.clone();
            if let Some(reg) = specific_reg(constraint) {
                op_regs[i] = reg.to_string();
            } else if is_memory_constraint(constraint) {
                op_is_memory[i] = true;
                if let Some(slot) = self.state.get_slot(ptr.0) {
                    op_mem_offsets[i] = slot.0;
                }
            }
        }

        // Track tied inputs for deferred resolution
        let mut input_tied_to: Vec<Option<usize>> = vec![None; inputs.len()];

        for (i, (constraint, _val, name)) in inputs.iter().enumerate() {
            let op_idx = outputs.len() + i;
            op_names[op_idx] = name.clone();
            if let Some(tied_to) = tied_operand(constraint) {
                input_tied_to[i] = Some(tied_to);
                // Don't assign yet - resolve after scratch allocation
            } else if let Some(reg) = specific_reg(constraint) {
                op_regs[op_idx] = reg.to_string();
            } else if is_memory_constraint(constraint) {
                op_is_memory[op_idx] = true;
                if let Operand::Value(v) = _val {
                    if let Some(slot) = self.state.get_slot(v.0) {
                        op_mem_offsets[op_idx] = slot.0;
                    }
                }
            }
        }

        // Second pass: assign scratch registers for generic "r" operands (skip tied)
        for i in 0..total_operands {
            if op_regs[i].is_empty() && !op_is_memory[i] {
                let is_tied = if i >= outputs.len() {
                    input_tied_to[i - outputs.len()].is_some()
                } else {
                    false
                };
                if !is_tied {
                    if scratch_idx < gp_scratch.len() {
                        op_regs[i] = gp_scratch[scratch_idx].to_string();
                        scratch_idx += 1;
                    } else {
                        op_regs[i] = format!("s{}", 2 + scratch_idx - gp_scratch.len());
                        scratch_idx += 1;
                    }
                }
            }
        }

        // Third pass: resolve tied operands now that all targets have registers
        for (i, tied_to) in input_tied_to.iter().enumerate() {
            if let Some(tied_to) = tied_to {
                let op_idx = outputs.len() + i;
                if *tied_to < op_regs.len() {
                    op_regs[op_idx] = op_regs[*tied_to].clone();
                    op_is_memory[op_idx] = op_is_memory[*tied_to];
                    op_mem_offsets[op_idx] = op_mem_offsets[*tied_to];
                }
            }
        }

        // Handle "+" read-write: synthetic inputs share register with their output
        let num_plus = outputs.iter().filter(|(c,_,_)| c.contains('+')).count();
        let mut plus_idx = 0;
        for (i, (constraint, _, _)) in outputs.iter().enumerate() {
            if constraint.contains('+') {
                let plus_input_idx = outputs.len() + plus_idx;
                if plus_input_idx < total_operands {
                    op_regs[plus_input_idx] = op_regs[i].clone();
                    op_is_memory[plus_input_idx] = op_is_memory[i];
                    op_mem_offsets[plus_input_idx] = op_mem_offsets[i];
                }
                plus_idx += 1;
            }
        }

        // Build GCC operand number → internal index mapping.
        // GCC numbers: outputs first, then EXPLICIT inputs (synthetic "+" inputs are hidden).
        let num_plus = outputs.iter().filter(|(c,_,_)| c.contains('+')).count();
        let num_gcc_operands = outputs.len() + (inputs.len() - num_plus);
        let mut gcc_to_internal: Vec<usize> = Vec::with_capacity(num_gcc_operands);
        for i in 0..outputs.len() {
            gcc_to_internal.push(i);
        }
        for i in num_plus..inputs.len() {
            gcc_to_internal.push(outputs.len() + i);
        }

        // Phase 2: Load input values into their assigned registers
        for (i, (_constraint, val, _)) in inputs.iter().enumerate() {
            let op_idx = outputs.len() + i;
            if op_is_memory[op_idx] {
                continue;
            }
            let reg = &op_regs[op_idx];
            if reg.is_empty() {
                continue;
            }

            // For ALL inputs (including tied), load the input value into the register.
            // Tied operands already share the same register as their target output.
            match val {
                Operand::Const(c) => {
                    let imm = c.to_i64().unwrap_or(0);
                    self.state.emit(&format!("    li {}, {}", reg, imm));
                }
                Operand::Value(v) => {
                    if let Some(slot) = self.state.get_slot(v.0) {
                        if self.state.is_alloca(v.0) {
                            self.emit_addi_s0(reg, slot.0);
                        } else {
                            self.emit_load_from_s0(reg, slot.0, "ld");
                        }
                    }
                }
            }
        }

        // Pre-load read-write output values
        for (i, (constraint, ptr, _)) in outputs.iter().enumerate() {
            if constraint.contains('+') && !op_is_memory[i] {
                let reg = &op_regs[i].clone();
                if let Some(slot) = self.state.get_slot(ptr.0) {
                    self.emit_load_from_s0(reg, slot.0, "ld");
                }
            }
        }

        // Phase 3: Substitute operand references in template and emit
        let lines: Vec<&str> = template.split('\n').collect();
        for line in &lines {
            let line = line.trim().trim_start_matches('\t').trim();
            if line.is_empty() {
                continue;
            }
            let resolved = Self::substitute_riscv_asm_operands(line, &op_regs, &op_names, &op_is_memory, &op_mem_offsets, &gcc_to_internal);
            self.state.emit(&format!("    {}", resolved));
        }

        // Phase 4: Store output register values back to their stack slots
        for (i, (constraint, ptr, _)) in outputs.iter().enumerate() {
            if (constraint.contains('=') || constraint.contains('+')) && !op_is_memory[i] {
                let reg = &op_regs[i].clone();
                if let Some(slot) = self.state.get_slot(ptr.0) {
                    self.emit_store_to_s0(reg, slot.0, "sd");
                }
            }
        }
    }
}

impl RiscvCodegen {
    /// Substitute %0, %1, %[name] in RISC-V asm template.
    fn substitute_riscv_asm_operands(
        line: &str,
        op_regs: &[String],
        op_names: &[Option<String>],
        op_is_memory: &[bool],
        op_mem_offsets: &[i64],
        gcc_to_internal: &[usize],
    ) -> String {
        let mut result = String::new();
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '%' && i + 1 < chars.len() {
                i += 1;
                // %% -> literal %
                if chars[i] == '%' {
                    result.push('%');
                    i += 1;
                    continue;
                }

                if chars[i] == '[' {
                    // Named operand: %[name]
                    i += 1;
                    let name_start = i;
                    while i < chars.len() && chars[i] != ']' {
                        i += 1;
                    }
                    let name: String = chars[name_start..i].iter().collect();
                    if i < chars.len() { i += 1; }

                    let mut found = false;
                    for (idx, op_name) in op_names.iter().enumerate() {
                        if let Some(ref n) = op_name {
                            if n == &name {
                                if op_is_memory[idx] {
                                    result.push_str(&format!("{}(s0)", op_mem_offsets[idx]));
                                } else {
                                    result.push_str(&op_regs[idx]);
                                }
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        result.push('%');
                        result.push('[');
                        result.push_str(&name);
                        result.push(']');
                    }
                } else if chars[i].is_ascii_digit() {
                    // Positional operand: %0, %1, etc.
                    // GCC operand numbers skip synthetic "+" inputs, so map through gcc_to_internal
                    let mut num = 0usize;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        num = num * 10 + (chars[i] as usize - '0' as usize);
                        i += 1;
                    }
                    let internal_idx = if num < gcc_to_internal.len() {
                        gcc_to_internal[num]
                    } else {
                        num
                    };
                    if internal_idx < op_regs.len() {
                        if op_is_memory[internal_idx] {
                            result.push_str(&format!("{}(s0)", op_mem_offsets[internal_idx]));
                        } else {
                            result.push_str(&op_regs[internal_idx]);
                        }
                    } else {
                        result.push_str(&format!("%{}", num));
                    }
                } else {
                    // Not recognized, emit as-is
                    result.push('%');
                    result.push(chars[i]);
                    i += 1;
                }
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }
        result
    }

    /// Get the AMO ordering suffix.
    /// Get the AMO ordering suffix.
    fn amo_ordering(ordering: AtomicOrdering) -> &'static str {
        match ordering {
            AtomicOrdering::Relaxed => "",
            AtomicOrdering::Acquire => ".aq",
            AtomicOrdering::Release => ".rl",
            AtomicOrdering::AcqRel => ".aqrl",
            AtomicOrdering::SeqCst => ".aqrl",
        }
    }

    /// Get the AMO width suffix.
    fn amo_width_suffix(ty: IrType) -> &'static str {
        match ty {
            IrType::I32 | IrType::U32 | IrType::I8 | IrType::U8 | IrType::I16 | IrType::U16 => "w",
            _ => "d",
        }
    }

    /// Sign-extend result for sub-word types after atomic ops.
    fn sign_extend_riscv(state: &mut CodegenState, ty: IrType) {
        match ty {
            IrType::I8 => {
                state.emit("    slli t0, t0, 56");
                state.emit("    srai t0, t0, 56");
            }
            IrType::U8 => {
                state.emit("    andi t0, t0, 0xff");
            }
            IrType::I16 => {
                state.emit("    slli t0, t0, 48");
                state.emit("    srai t0, t0, 48");
            }
            IrType::U16 => {
                state.emit("    slli t0, t0, 48");
                state.emit("    srli t0, t0, 48");
            }
            IrType::I32 => {
                state.emit("    sext.w t0, t0");
            }
            _ => {}
        }
    }

    /// Mask values for sub-word CAS comparison.
    fn mask_for_cmp(state: &mut CodegenState, ty: IrType) {
        match ty {
            IrType::I8 | IrType::U8 => {
                state.emit("    andi t0, t0, 0xff");
            }
            IrType::I16 | IrType::U16 => {
                // Mask to 16 bits
                state.emit("    slli t0, t0, 48");
                state.emit("    srli t0, t0, 48");
            }
            _ => {}
        }
    }
}

impl Default for RiscvCodegen {
    fn default() -> Self {
        Self::new()
    }
}
