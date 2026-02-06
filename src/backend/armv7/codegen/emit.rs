//! ARMv7 (32-bit ARM) code generator. Implements the ArchCodegen trait.
//!
//! Uses the AAPCS (ARM Architecture Procedure Call Standard) calling convention:
//! - Arguments: r0-r3 (first 4 integer/pointer args), stack for the rest
//! - Return values: r0 (32-bit), r0:r1 (64-bit)
//! - Callee-saved: r4-r11 (v1-v8)
//! - Caller-saved: r0-r3, r12 (ip), r14 (lr)
//! - Frame pointer: r11 (fp), Stack pointer: r13 (sp), Link register: r14 (lr), PC: r15
//! - Floating-point: s0-s15/d0-d7 for args, VFPv3-D16 (d0-d15)
//! - Stack aligned to 8 bytes

use crate::delegate_to_impl;
use crate::backend::traits::ArchCodegen;
use crate::backend::common::PtrDirective;
use crate::backend::state::{CodegenState, StackSlot};
use crate::backend::regalloc::PhysReg;
use crate::backend::generation::is_i128_type;
use crate::backend::call_abi;
use crate::ir::reexports::{
    AtomicOrdering,
    AtomicRmwOp,
    BlockId,
    IntrinsicOp,
    IrBinOp,
    IrCmpOp,
    IrConst,
    IrFunction,
    IrUnaryOp,
    Operand,
    Value,
};
use crate::common::types::IrType;
use crate::common::fx_hash::FxHashMap;
use crate::{emit};

/// ARMv7 code generator. Implements the ArchCodegen trait for the shared framework.
/// Uses AAPCS calling convention with accumulator-based codegen (r0 as primary).
pub struct Armv7Codegen {
    pub(crate) state: CodegenState,
    pub(super) current_return_type: IrType,
    /// Whether the current function is variadic
    pub(super) is_variadic: bool,
    /// Register allocation results (callee-saved registers: r4-r10)
    pub(super) reg_assignments: FxHashMap<u32, PhysReg>,
    /// Which callee-saved registers are used and need save/restore
    pub(super) used_callee_saved: Vec<PhysReg>,
    /// Total stack bytes consumed by named parameters (for va_start computation).
    pub(super) va_named_stack_bytes: usize,
    /// Number of named args passed in registers (for variadic)
    pub(super) va_named_reg_count: usize,
}

// Callee-saved physical register indices for ARMv7
// PhysReg(0) = r4, PhysReg(1) = r5, ..., PhysReg(6) = r10
pub(super) const ARMV7_CALLEE_SAVED: &[PhysReg] = &[
    PhysReg(0), PhysReg(1), PhysReg(2), PhysReg(3),
    PhysReg(4), PhysReg(5), PhysReg(6),
];
// No caller-saved registers available for allocation (r0-r3 are scratch/args)
pub(super) const ARMV7_CALLER_SAVED: &[PhysReg] = &[];

pub(super) fn phys_reg_name(reg: PhysReg) -> &'static str {
    match reg.0 {
        0 => "r4",
        1 => "r5",
        2 => "r6",
        3 => "r7",
        4 => "r8",
        5 => "r9",
        6 => "r10",
        _ => panic!("invalid armv7 phys reg: {:?}", reg),
    }
}

/// Map IrBinOp to ARM mnemonic for simple ALU ops.
pub(super) fn armv7_alu_mnemonic(op: IrBinOp) -> &'static str {
    match op {
        IrBinOp::Add => "add",
        IrBinOp::Sub => "sub",
        IrBinOp::And => "and",
        IrBinOp::Or => "orr",
        IrBinOp::Xor => "eor",
        IrBinOp::Mul => "mul",
        _ => unreachable!("unsupported ALU op: {:?}", op),
    }
}

/// Map an IrCmpOp to its ARM condition code suffix.
pub(super) fn armv7_int_cond_code(op: IrCmpOp) -> &'static str {
    match op {
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
    }
}

/// Return the inverted ARM condition code suffix.
pub(super) fn armv7_invert_cond_code(cc: &str) -> &'static str {
    match cc {
        "eq" => "ne",
        "ne" => "eq",
        "lt" => "ge",
        "le" => "gt",
        "gt" => "le",
        "ge" => "lt",
        "lo" => "hs",
        "ls" => "hi",
        "hi" => "ls",
        "hs" => "lo",
        _ => "al",
    }
}

/// Map IrCmpOp to ARM VFP condition code.
pub(super) fn armv7_float_cond_code(op: IrCmpOp) -> &'static str {
    match op {
        IrCmpOp::Eq => "eq",
        IrCmpOp::Ne => "ne",
        IrCmpOp::Slt | IrCmpOp::Ult => "mi",
        IrCmpOp::Sle | IrCmpOp::Ule => "ls",
        IrCmpOp::Sgt | IrCmpOp::Ugt => "gt",
        IrCmpOp::Sge | IrCmpOp::Uge => "ge",
    }
}

impl Armv7Codegen {
    pub fn new() -> Self {
        Self {
            state: CodegenState::new(),
            current_return_type: IrType::Void,
            is_variadic: false,
            reg_assignments: FxHashMap::default(),
            used_callee_saved: Vec::new(),
            va_named_stack_bytes: 0,
            va_named_reg_count: 0,
        }
    }

    pub fn apply_options(&mut self, opts: &crate::backend::CodegenOptions) {
        self.state.pic_mode = opts.pic;
        self.state.emit_cfi = opts.emit_cfi;
        let _ = opts;
    }

    /// Reference to a stack slot from r11 (frame pointer).
    pub(super) fn slot_ref(&self, slot: StackSlot) -> String {
        let offset = slot.0;
        if offset == 0 {
            "[r11]".to_string()
        } else {
            format!("[r11, #{}]", offset)
        }
    }

    /// Get the slot reference for a value.
    pub(super) fn value_slot_ref(&self, val_id: u32) -> String {
        if let Some(slot) = self.state.get_slot(val_id) {
            self.slot_ref(slot)
        } else {
            panic!("no stack slot for value v{}", val_id)
        }
    }

    /// Load value into r0.
    /// Checks the reg cache first to skip redundant loads.
    pub(super) fn operand_to_r0(&mut self, op: &Operand) {
        match op {
            Operand::Const(c) => {
                self.state.reg_cache.invalidate_acc();
                self.load_const_to_r0(c);
            }
            Operand::Value(v) => {
                let is_alloca = self.state.is_alloca(v.0);
                // Check cache: skip load if value is already in r0
                if self.state.reg_cache.acc_has(v.0, is_alloca) {
                    return;
                }
                if let Some(preg) = self.reg_assignments.get(&v.0) {
                    let name = phys_reg_name(*preg);
                    emit!(self.state, "    mov r0, {}", name);
                    self.state.reg_cache.set_acc(v.0, false);
                } else if let Some(slot) = self.state.get_slot(v.0) {
                    let slot_ref = self.slot_ref(slot);
                    emit!(self.state, "    ldr r0, {}", slot_ref);
                    self.state.reg_cache.set_acc(v.0, is_alloca);
                } else {
                    // Value has no slot and no register (shouldn't happen for
                    // non-immediately-consumed operands, but be safe)
                    self.load_imm32_to_reg("r0", 0);
                    self.state.reg_cache.invalidate_acc();
                }
            }
        }
    }

    /// Store r0 to a value's stack slot.
    /// If the value has no stack slot and no register assignment (immediately-consumed),
    /// the store is skipped — the value stays in r0 (the accumulator).
    /// Always updates the reg cache to indicate r0 holds dest's value.
    pub(super) fn store_r0_to(&mut self, dest: &Value) {
        if let Some(preg) = self.reg_assignments.get(&dest.0) {
            let name = phys_reg_name(*preg);
            emit!(self.state, "    mov {}, r0", name);
        } else if let Some(slot) = self.state.get_slot(dest.0) {
            let slot_ref = self.slot_ref(slot);
            emit!(self.state, "    str r0, {}", slot_ref);
        }
        // After storing (or skipping for immediately-consumed values),
        // r0 still holds dest's value. Update the reg cache.
        self.state.reg_cache.set_acc(dest.0, false);
    }

    /// Load a constant into r0.
    pub(super) fn load_const_to_r0(&mut self, c: &IrConst) {
        match c {
            IrConst::I32(v) => {
                self.load_imm32_to_reg("r0", *v as u32);
            }
            IrConst::I8(v) => self.load_imm32_to_reg("r0", *v as i32 as u32),
            IrConst::I16(v) => self.load_imm32_to_reg("r0", *v as i32 as u32),
            IrConst::I64(v) => {
                // 64-bit on 32-bit: only lower 32 bits in r0
                self.load_imm32_to_reg("r0", *v as u32);
            }
            IrConst::I128(v) => {
                self.load_imm32_to_reg("r0", *v as u32);
            }
            IrConst::F32(v) => {
                let bits = v.to_bits();
                self.load_imm32_to_reg("r0", bits);
            }
            IrConst::F64(v) => {
                // Lower 32 bits to r0
                let bits = v.to_bits();
                self.load_imm32_to_reg("r0", bits as u32);
            }
            IrConst::LongDouble(v, _bytes) => {
                // Long double on ARM32 is treated as double
                let bits = v.to_bits();
                self.load_imm32_to_reg("r0", bits as u32);
            }
            IrConst::Zero => {
                self.state.emit("    mov r0, #0");
            }
        }
    }

    /// Load a 32-bit immediate to a register using movw/movt.
    pub(super) fn load_imm32_to_reg(&mut self, reg: &str, val: u32) {
        if val == 0 {
            emit!(self.state, "    mov {}, #0", reg);
        } else if val <= 0xFFFF {
            emit!(self.state, "    movw {}, #{}", reg, val);
        } else {
            let lo = val & 0xFFFF;
            let hi = (val >> 16) & 0xFFFF;
            emit!(self.state, "    movw {}, #{}", reg, lo);
            if hi != 0 {
                emit!(self.state, "    movt {}, #{}", reg, hi);
            }
        }
    }

    /// Load a 64-bit value (lower half to r0, upper half to r1).
    pub(super) fn load_wide_to_r0_r1(&mut self, op: &Operand) {
        match op {
            Operand::Const(IrConst::I64(v)) => {
                let lo = *v as u32;
                let hi = (*v >> 32) as u32;
                self.load_imm32_to_reg("r0", lo);
                self.load_imm32_to_reg("r1", hi);
            }
            Operand::Const(IrConst::F64(v)) => {
                let bits = v.to_bits();
                self.load_imm32_to_reg("r0", bits as u32);
                self.load_imm32_to_reg("r1", (bits >> 32) as u32);
            }
            Operand::Const(IrConst::I128(v)) => {
                let lo = *v as u32;
                let hi = (*v >> 32) as u32;
                self.load_imm32_to_reg("r0", lo);
                self.load_imm32_to_reg("r1", hi);
            }
            Operand::Value(v) => {
                // Wide values (I64/U64/F64/I128) always have stack slots on 32-bit targets
                // (excluded from immediately-consumed optimization in slot_assignment.rs).
                if let Some(slot) = self.state.get_slot(v.0) {
                    let slot_ref = self.slot_ref(slot);
                    emit!(self.state, "    ldr r0, {}", slot_ref);
                    let hi_offset = slot.0 + 4;
                    if hi_offset == 0 {
                        self.state.emit("    ldr r1, [r11]");
                    } else {
                        emit!(self.state, "    ldr r1, [r11, #{}]", hi_offset);
                    }
                } else {
                    // Fallback: shouldn't happen for wide values, but be safe
                    self.operand_to_r0(op);
                    self.state.emit("    mov r1, #0");
                }
            }
            _ => {
                self.operand_to_r0(op);
                self.state.emit("    mov r1, #0");
            }
        }
    }

    /// Store r0:r1 as a 64-bit value.
    pub(super) fn store_r0_r1_to(&mut self, dest: &Value) {
        if let Some(slot) = self.state.get_slot(dest.0) {
            let slot_ref = self.slot_ref(slot);
            emit!(self.state, "    str r0, {}", slot_ref);
            let offset = slot.0 + 4;
            if offset == 0 {
                self.state.emit("    str r1, [r11]");
            } else {
                emit!(self.state, "    str r1, [r11, #{}]", offset);
            }
        }
        // After a 64-bit store, r0 only holds the low 32 bits of dest —
        // the acc cache cannot represent this accurately, so invalidate it.
        self.state.reg_cache.invalidate_acc();
    }

    /// Emit load of a pointer-to-slot address into a register.
    pub(super) fn load_slot_addr_to_reg(&mut self, reg: &str, slot: StackSlot) {
        let offset = slot.0;
        if offset == 0 {
            emit!(self.state, "    mov {}, r11", reg);
        } else if offset > 0 && offset <= 255 {
            emit!(self.state, "    add {}, r11, #{}", reg, offset);
        } else if offset < 0 && offset >= -255 {
            emit!(self.state, "    sub {}, r11, #{}", reg, -offset);
        } else {
            self.load_imm32_to_reg(reg, offset as u32);
            emit!(self.state, "    add {}, r11, {}", reg, reg);
        }
    }

    /// Load store instruction for a given type width.
    pub(super) fn store_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        match ty.size() {
            1 => "strb",
            2 => "strh",
            4 | 8 => "str",
            _ => "str",
        }
    }

    pub(super) fn load_instr_for_type_impl(&self, ty: IrType) -> &'static str {
        match (ty.size(), ty.is_signed()) {
            (1, true) => "ldrsb",
            (1, false) => "ldrb",
            (2, true) => "ldrsh",
            (2, false) => "ldrh",
            _ => "ldr",
        }
    }

    /// Emit typed store: store r0 to [ptr_reg + offset].
    pub(super) fn emit_typed_store_impl(&mut self, ty: IrType, offset: i64, ptr_reg: &str) {
        let instr = self.store_instr_for_type_impl(ty);
        if offset == 0 {
            emit!(self.state, "    {} r0, [{}]", instr, ptr_reg);
        } else {
            emit!(self.state, "    {} r0, [{}, #{}]", instr, ptr_reg, offset);
        }
    }

    /// Emit typed load: load from [ptr_reg + offset] to r0.
    pub(super) fn emit_typed_load_impl(&mut self, ty: IrType, offset: i64, ptr_reg: &str) {
        let instr = self.load_instr_for_type_impl(ty);
        if offset == 0 {
            emit!(self.state, "    {} r0, [{}]", instr, ptr_reg);
        } else {
            emit!(self.state, "    {} r0, [{}, #{}]", instr, ptr_reg, offset);
        }
    }

    /// Load a pointer from a value's slot into a register.
    pub(super) fn load_ptr_to_reg(&mut self, reg: &str, val: &Value) {
        if let Some(preg) = self.reg_assignments.get(&val.0) {
            let name = phys_reg_name(*preg);
            if name != reg {
                emit!(self.state, "    mov {}, {}", reg, name);
            }
        } else if let Some(slot) = self.state.get_slot(val.0) {
            let slot_ref = self.slot_ref(slot);
            emit!(self.state, "    ldr {}, {}", reg, slot_ref);
        } else {
            // Value is in the accumulator (immediately-consumed), move r0 to target reg
            if reg != "r0" {
                emit!(self.state, "    mov {}, r0", reg);
            }
        }
    }

    // ---- New _impl methods for the updated trait ----

    pub(super) fn emit_typed_store_to_slot_impl(&mut self, _instr: &'static str, ty: IrType, slot: StackSlot) {
        let instr = self.store_instr_for_type_impl(ty);
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    {} r0, {}", instr, slot_ref);
    }

    pub(super) fn emit_typed_load_from_slot_impl(&mut self, _instr: &'static str, slot: StackSlot) {
        // Use the passed instruction mnemonic
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    {} r0, {}", _instr, slot_ref);
    }

    pub(super) fn emit_load_ptr_from_slot_impl(&mut self, slot: StackSlot, _val_id: u32) {
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    ldr r12, {}", slot_ref);
    }

    pub(super) fn emit_typed_store_indirect_impl(&mut self, instr: &'static str, _ty: IrType) {
        // r2 has the value (saved acc), r12 has the pointer
        emit!(self.state, "    {} r2, [r12]", instr);
    }

    pub(super) fn emit_typed_load_indirect_impl(&mut self, instr: &'static str) {
        // r12 has the pointer, load into r0
        emit!(self.state, "    {} r0, [r12]", instr);
    }

    pub(super) fn emit_add_offset_to_addr_reg_impl(&mut self, offset: i64) {
        if offset == 0 {
            return;
        }
        if offset > 0 && offset <= 255 {
            emit!(self.state, "    add r12, r12, #{}", offset);
        } else if offset < 0 && offset >= -255 {
            emit!(self.state, "    sub r12, r12, #{}", -offset);
        } else {
            // Use r3 as temp
            self.load_imm32_to_reg("r3", offset as u32);
            self.state.emit("    add r12, r12, r3");
        }
    }

    pub(super) fn emit_slot_addr_to_secondary_impl(&mut self, slot: StackSlot, is_alloca: bool, _val_id: u32) {
        if is_alloca {
            self.load_slot_addr_to_reg("r1", slot);
        } else {
            let slot_ref = self.slot_ref(slot);
            emit!(self.state, "    ldr r1, {}", slot_ref);
        }
    }

    pub(super) fn emit_add_imm_to_acc_impl(&mut self, imm: i64) {
        if imm > 0 && imm <= 255 {
            emit!(self.state, "    add r0, r0, #{}", imm);
        } else if imm < 0 && imm >= -255 {
            emit!(self.state, "    sub r0, r0, #{}", -imm);
        } else {
            self.load_imm32_to_reg("r1", imm as u32);
            self.state.emit("    add r0, r0, r1");
        }
    }

    pub(super) fn emit_round_up_acc_to_16_impl(&mut self) {
        // Round up to 16-byte alignment: r0 = (r0 + 15) & ~15
        self.state.emit("    add r0, r0, #15");
        self.state.emit("    bic r0, r0, #15");
    }

    pub(super) fn emit_sub_sp_by_acc_impl(&mut self) {
        self.state.emit("    sub sp, sp, r0");
    }

    pub(super) fn emit_mov_sp_to_acc_impl(&mut self) {
        self.state.emit("    mov r0, sp");
    }

    pub(super) fn emit_mov_acc_to_sp_impl(&mut self) {
        self.state.emit("    mov sp, r0");
    }

    pub(super) fn emit_align_acc_impl(&mut self, align: usize) {
        // r0 = (r0 + align-1) & ~(align-1)
        let mask = !(align as i32 - 1);
        if (align - 1) <= 255 {
            emit!(self.state, "    add r0, r0, #{}", align - 1);
        } else {
            self.load_imm32_to_reg("r1", (align - 1) as u32);
            self.state.emit("    add r0, r0, r1");
        }
        self.load_imm32_to_reg("r1", mask as u32);
        self.state.emit("    and r0, r0, r1");
    }

    pub(super) fn emit_memcpy_load_dest_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, _val_id: u32) {
        if is_alloca {
            self.load_slot_addr_to_reg("r0", slot);
        } else {
            let slot_ref = self.slot_ref(slot);
            emit!(self.state, "    ldr r0, {}", slot_ref);
        }
        self.state.emit("    push {r0}");
    }

    pub(super) fn emit_memcpy_load_src_addr_impl(&mut self, slot: StackSlot, is_alloca: bool, _val_id: u32) {
        if is_alloca {
            self.load_slot_addr_to_reg("r1", slot);
        } else {
            let slot_ref = self.slot_ref(slot);
            emit!(self.state, "    ldr r1, {}", slot_ref);
        }
        self.state.emit("    pop {r0}");
    }

    pub(super) fn emit_memcpy_impl_impl(&mut self, size: usize) {
        self.load_imm32_to_reg("r2", size as u32);
        self.state.emit("    bl memcpy");
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_jump_indirect_impl(&mut self) {
        self.state.emit("    bx r0");
    }

    pub(super) fn emit_float_binop_impl_impl(&mut self, mnemonic: &str, ty: IrType) {
        // r1 has lhs (secondary), r0 has rhs
        if ty == IrType::F64 {
            // For F64, need to handle as double pairs
            // This is a simplified version - full implementation would use VFP
            emit!(self.state, "    @ float64 binop: {}", mnemonic);
            self.state.emit("    bl __aeabi_dadd"); // placeholder
        } else {
            // F32: use VFP
            self.state.emit("    vmov s1, r1"); // lhs
            self.state.emit("    vmov s0, r0"); // rhs
            let vfp_mnemonic = match mnemonic {
                "fadd" => "vadd.f32",
                "fsub" => "vsub.f32",
                "fmul" => "vmul.f32",
                "fdiv" => "vdiv.f32",
                _ => "vadd.f32",
            };
            emit!(self.state, "    {} s0, s1, s0", vfp_mnemonic);
            self.state.emit("    vmov r0, s0");
        }
    }

    // ---- 128-bit accumulator pair primitives ----
    // On ARMv7, 128-bit values use r0:r1 as the accumulator pair (same as 64-bit)
    // For true 128-bit, we use r0:r1:r2:r3 but simplify to 64-bit for now

    pub(super) fn emit_sign_extend_acc_high_impl(&mut self) {
        // Sign-extend r0 into r1 (high half)
        self.state.emit("    asr r1, r0, #31");
    }

    pub(super) fn emit_zero_acc_high_impl(&mut self) {
        self.state.emit("    mov r1, #0");
    }

    pub(super) fn emit_load_acc_pair_impl(&mut self, op: &Operand) {
        self.load_wide_to_r0_r1(op);
    }

    pub(super) fn emit_store_acc_pair_impl(&mut self, dest: &Value) {
        self.store_r0_r1_to(dest);
    }

    pub(super) fn emit_store_pair_to_slot_impl(&mut self, slot: StackSlot) {
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    str r0, {}", slot_ref);
        let hi_offset = slot.0 + 4;
        if hi_offset == 0 {
            self.state.emit("    str r1, [r11]");
        } else {
            emit!(self.state, "    str r1, [r11, #{}]", hi_offset);
        }
    }

    pub(super) fn emit_load_pair_from_slot_impl(&mut self, slot: StackSlot) {
        let slot_ref = self.slot_ref(slot);
        emit!(self.state, "    ldr r0, {}", slot_ref);
        let hi_offset = slot.0 + 4;
        if hi_offset == 0 {
            self.state.emit("    ldr r1, [r11]");
        } else {
            emit!(self.state, "    ldr r1, [r11, #{}]", hi_offset);
        }
    }

    pub(super) fn emit_save_acc_pair_impl(&mut self) {
        // Save r0:r1 to scratch regs r2:r3
        self.state.emit("    mov r2, r0");
        self.state.emit("    mov r3, r1");
    }

    pub(super) fn emit_store_pair_indirect_impl(&mut self) {
        // Store saved acc pair (r2:r3) through address in r12
        self.state.emit("    str r2, [r12]");
        self.state.emit("    str r3, [r12, #4]");
    }

    pub(super) fn emit_load_pair_indirect_impl(&mut self) {
        // Load acc pair through address in r12
        self.state.emit("    ldr r0, [r12]");
        self.state.emit("    ldr r1, [r12, #4]");
    }

    pub(super) fn emit_i128_neg_impl(&mut self) {
        // Negate 64-bit: rsbs r0, r0, #0; rsc r1, r1, #0
        self.state.emit("    rsbs r0, r0, #0");
        self.state.emit("    rsc r1, r1, #0");
    }

    pub(super) fn emit_i128_not_impl(&mut self) {
        self.state.emit("    mvn r0, r0");
        self.state.emit("    mvn r1, r1");
    }

    pub(super) fn emit_i128_to_float_call_impl(&mut self, src: &Operand, from_signed: bool, to_ty: IrType) {
        self.load_wide_to_r0_r1(src);
        let func_name = match (from_signed, to_ty) {
            (true, IrType::F32) => "__aeabi_l2f",
            (true, IrType::F64) => "__aeabi_l2d",
            (false, IrType::F32) => "__aeabi_ul2f",
            (false, IrType::F64) => "__aeabi_ul2d",
            _ => "__aeabi_l2d",
        };
        emit!(self.state, "    bl {}", func_name);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_float_to_i128_call_impl(&mut self, src: &Operand, to_signed: bool, from_ty: IrType) {
        if from_ty == IrType::F64 {
            self.load_wide_to_r0_r1(src);
        } else {
            self.operand_to_r0(src);
        }
        let func_name = match (to_signed, from_ty) {
            (true, IrType::F32) => "__aeabi_f2lz",
            (true, IrType::F64) => "__aeabi_d2lz",
            (false, IrType::F32) => "__aeabi_f2ulz",
            (false, IrType::F64) => "__aeabi_d2ulz",
            _ => "__aeabi_d2lz",
        };
        emit!(self.state, "    bl {}", func_name);
        self.state.reg_cache.invalidate_all();
    }

    // ---- Return primitives ----

    pub(super) fn emit_return_i128_to_regs_impl(&mut self) {
        // r0:r1 already holds the i128 low:high value (AAPCS 64-bit return)
        // No-op on ARM32
    }

    pub(super) fn emit_return_f128_to_reg_impl(&mut self) {
        // F128 on ARM32 is treated as F64, returned in r0:r1
        // No-op, value is already in r0:r1
    }

    pub(super) fn emit_return_f32_to_reg_impl(&mut self) {
        // F32 returned in r0 (soft-float ABI), already there
    }

    pub(super) fn emit_return_f64_to_reg_impl(&mut self) {
        // F64 returned in r0:r1, need to load wide
        // Value should already be in r0:r1 from emit_load_operand
    }

    pub(super) fn emit_return_int_to_reg_impl(&mut self) {
        // Integer returned in r0, already there
    }

    pub(super) fn emit_epilogue_and_ret_impl(&mut self, frame_size: i64) {
        self.emit_epilogue_impl(frame_size);
    }

    pub(super) fn emit_cast_instrs_impl(&mut self, from_ty: IrType, to_ty: IrType) {
        let from_size = from_ty.size();
        let to_size = to_ty.size();

        if to_size < from_size {
            // Truncate
            match to_size {
                1 => self.state.emit("    and r0, r0, #0xff"),
                2 => {
                    self.load_imm32_to_reg("r1", 0xffff);
                    self.state.emit("    and r0, r0, r1");
                }
                _ => {}
            }
        } else if to_size > from_size {
            // Extend
            if from_ty.is_signed() {
                match from_size {
                    1 => self.state.emit("    sxtb r0, r0"),
                    2 => self.state.emit("    sxth r0, r0"),
                    _ => {}
                }
            } else {
                match from_size {
                    1 => self.state.emit("    uxtb r0, r0"),
                    2 => self.state.emit("    uxth r0, r0"),
                    _ => {}
                }
            }
        }
        // Float conversions handled in emit_cast_impl
    }

    // ---- i128 binary operation primitives ----

    pub(super) fn emit_i128_prep_binop_impl(&mut self, lhs: &Operand, rhs: &Operand) {
        // Load lhs to r0:r1, save, load rhs to r2:r3
        self.load_wide_to_r0_r1(lhs);
        self.state.emit("    push {r0, r1}"); // Save lhs
        self.load_wide_to_r0_r1(rhs);
        self.state.emit("    mov r2, r0"); // r2:r3 = rhs
        self.state.emit("    mov r3, r1");
        self.state.emit("    pop {r0, r1}"); // r0:r1 = lhs
    }

    pub(super) fn emit_i128_add_impl(&mut self) {
        self.state.emit("    adds r0, r0, r2");
        self.state.emit("    adc r1, r1, r3");
    }

    pub(super) fn emit_i128_sub_impl(&mut self) {
        self.state.emit("    subs r0, r0, r2");
        self.state.emit("    sbc r1, r1, r3");
    }

    pub(super) fn emit_i128_mul_impl(&mut self) {
        // 64-bit multiply: result_lo = lo*lo, result_hi = hi*lo + lo*hi + carry
        self.state.emit("    push {r4, r5}");
        self.state.emit("    umull r4, r5, r0, r2"); // r4:r5 = lo * lo
        self.state.emit("    mla r5, r0, r3, r5");    // r5 += lo * rhs_hi
        self.state.emit("    mla r5, r1, r2, r5");    // r5 += lhs_hi * lo
        self.state.emit("    mov r0, r4");
        self.state.emit("    mov r1, r5");
        self.state.emit("    pop {r4, r5}");
    }

    pub(super) fn emit_i128_and_impl(&mut self) {
        self.state.emit("    and r0, r0, r2");
        self.state.emit("    and r1, r1, r3");
    }

    pub(super) fn emit_i128_or_impl(&mut self) {
        self.state.emit("    orr r0, r0, r2");
        self.state.emit("    orr r1, r1, r3");
    }

    pub(super) fn emit_i128_xor_impl(&mut self) {
        self.state.emit("    eor r0, r0, r2");
        self.state.emit("    eor r1, r1, r3");
    }

    pub(super) fn emit_i128_shl_impl(&mut self) {
        // 64-bit shift left by r2 bits
        self.state.emit("    bl __aeabi_llsl");
    }

    pub(super) fn emit_i128_lshr_impl(&mut self) {
        self.state.emit("    bl __aeabi_llsr");
    }

    pub(super) fn emit_i128_ashr_impl(&mut self) {
        self.state.emit("    bl __aeabi_lasr");
    }

    pub(super) fn emit_i128_divrem_call_impl(&mut self, func_name: &str, lhs: &Operand, rhs: &Operand) {
        self.load_wide_to_r0_r1(lhs);
        self.state.emit("    push {r0, r1}");
        self.load_wide_to_r0_r1(rhs);
        self.state.emit("    mov r2, r0");
        self.state.emit("    mov r3, r1");
        self.state.emit("    pop {r0, r1}");
        emit!(self.state, "    bl {}", func_name);
        self.state.reg_cache.invalidate_all();
    }

    pub(super) fn emit_i128_store_result_impl(&mut self, dest: &Value) {
        self.store_r0_r1_to(dest);
    }

    pub(super) fn emit_i128_shl_const_impl(&mut self, amount: u32) {
        if amount == 0 {
            return;
        }
        if amount >= 32 {
            let shift = amount - 32;
            if shift == 0 {
                self.state.emit("    mov r1, r0");
            } else {
                emit!(self.state, "    lsl r1, r0, #{}", shift);
            }
            self.state.emit("    mov r0, #0");
        } else {
            emit!(self.state, "    lsl r1, r1, #{}", amount);
            emit!(self.state, "    orr r1, r1, r0, lsr #{}", 32 - amount);
            emit!(self.state, "    lsl r0, r0, #{}", amount);
        }
    }

    pub(super) fn emit_i128_lshr_const_impl(&mut self, amount: u32) {
        if amount == 0 {
            return;
        }
        if amount >= 32 {
            let shift = amount - 32;
            if shift == 0 {
                self.state.emit("    mov r0, r1");
            } else {
                emit!(self.state, "    lsr r0, r1, #{}", shift);
            }
            self.state.emit("    mov r1, #0");
        } else {
            emit!(self.state, "    lsr r0, r0, #{}", amount);
            emit!(self.state, "    orr r0, r0, r1, lsl #{}", 32 - amount);
            emit!(self.state, "    lsr r1, r1, #{}", amount);
        }
    }

    pub(super) fn emit_i128_ashr_const_impl(&mut self, amount: u32) {
        if amount == 0 {
            return;
        }
        if amount >= 32 {
            let shift = amount - 32;
            if shift == 0 {
                self.state.emit("    mov r0, r1");
            } else {
                emit!(self.state, "    asr r0, r1, #{}", shift);
            }
            self.state.emit("    asr r1, r1, #31");
        } else {
            emit!(self.state, "    lsr r0, r0, #{}", amount);
            emit!(self.state, "    orr r0, r0, r1, lsl #{}", 32 - amount);
            emit!(self.state, "    asr r1, r1, #{}", amount);
        }
    }

    pub(super) fn emit_i128_cmp_eq_impl(&mut self, is_ne: bool) {
        // r0:r1 = lhs, r2:r3 = rhs (from prep_cmp/prep_binop)
        self.state.emit("    eor r0, r0, r2");
        self.state.emit("    eor r1, r1, r3");
        self.state.emit("    orrs r0, r0, r1");
        if is_ne {
            self.state.emit("    movne r0, #1");
            self.state.emit("    moveq r0, #0");
        } else {
            self.state.emit("    moveq r0, #1");
            self.state.emit("    movne r0, #0");
        }
    }

    pub(super) fn emit_i128_cmp_ordered_impl(&mut self, op: IrCmpOp) {
        let is_signed = matches!(op, IrCmpOp::Slt | IrCmpOp::Sle | IrCmpOp::Sgt | IrCmpOp::Sge);

        if is_signed {
            // For signed comparison: high words use signed condition,
            // low words use unsigned condition (since low words represent
            // the unsigned magnitude portion of the 64-bit value).
            let label_id = self.state.next_label_id();
            let hi_decided = format!(".Li128_hidec_{}", label_id);
            let done = format!(".Li128_done_{}", label_id);

            // Compare high words (signed)
            self.state.emit("    cmp r1, r3");
            emit!(self.state, "    bne {}", hi_decided);

            // High words equal: compare low words with unsigned condition
            self.state.emit("    cmp r0, r2");
            let lo_cc = match op {
                IrCmpOp::Slt => "lo",  // unsigned less than
                IrCmpOp::Sle => "ls",  // unsigned less or same
                IrCmpOp::Sgt => "hi",  // unsigned higher
                IrCmpOp::Sge => "hs",  // unsigned higher or same
                _ => unreachable!(),
            };
            let lo_inv = armv7_invert_cond_code(lo_cc);
            emit!(self.state, "    mov{} r0, #1", lo_cc);
            emit!(self.state, "    mov{} r0, #0", lo_inv);
            emit!(self.state, "    b {}", done);

            // High words differ: use signed condition
            emit!(self.state, "{}:", hi_decided);
            let hi_cc = match op {
                IrCmpOp::Slt | IrCmpOp::Sle => "lt",
                IrCmpOp::Sgt | IrCmpOp::Sge => "gt",
                _ => unreachable!(),
            };
            let hi_inv = armv7_invert_cond_code(hi_cc);
            emit!(self.state, "    mov{} r0, #1", hi_cc);
            emit!(self.state, "    mov{} r0, #0", hi_inv);

            emit!(self.state, "{}:", done);
        } else {
            // Unsigned comparison: both high and low use unsigned conditions
            self.state.emit("    cmp r1, r3"); // Compare high words
            self.state.emit("    cmpeq r0, r2"); // If equal, compare low words
            let cc = armv7_int_cond_code(op);
            let inv_cc = armv7_invert_cond_code(cc);
            emit!(self.state, "    mov{} r0, #1", cc);
            emit!(self.state, "    mov{} r0, #0", inv_cc);
        }
    }

    pub(super) fn emit_i128_cmp_store_result_impl(&mut self, dest: &Value) {
        self.state.reg_cache.invalidate_all();
        self.store_r0_to(dest);
    }

    // ---- Second return value primitives ----

    pub(super) fn emit_get_return_f64_second_impl(&mut self, dest: &Value) {
        // ARM32 doesn't support returning two f64 values
        self.state.emit("    mov r0, #0");
        self.store_r0_to(dest);
    }

    pub(super) fn emit_set_return_f64_second_impl(&mut self, _src: &Operand) {
        // No-op on ARM32
    }

    pub(super) fn emit_get_return_f32_second_impl(&mut self, dest: &Value) {
        self.state.emit("    mov r0, #0");
        self.store_r0_to(dest);
    }

    pub(super) fn emit_set_return_f32_second_impl(&mut self, _src: &Operand) {
        // No-op on ARM32
    }

    pub(super) fn emit_get_return_f128_second_impl(&mut self, dest: &Value) {
        self.state.emit("    mov r0, #0");
        self.store_r0_to(dest);
    }

    pub(super) fn emit_set_return_f128_second_impl(&mut self, _src: &Operand) {
        // No-op on ARM32
    }

    pub(super) fn aligned_frame_size_impl(&self, raw_space: i64) -> i64 {
        // ARM AAPCS requires 8-byte stack alignment at public interfaces.
        // After push {r11, lr} (always 8 bytes, aligned), we push callee-saved
        // registers then allocate frame space. The sum of callee-saved bytes
        // and frame space must be a multiple of 8 so sp stays aligned.
        let callee_saved_bytes = self.used_callee_saved.len() as i64 * 4;
        let total = raw_space + callee_saved_bytes;
        let aligned_total = (total + 7) & !7;
        aligned_total - callee_saved_bytes
    }

    // ---- Switch primitives ----

    pub(super) fn emit_switch_jump_table_impl(
        &mut self,
        val: &Operand,
        cases: &[(i64, BlockId)],
        default: &BlockId,
        _ty: IrType,
    ) {
        use crate::backend::traits::build_jump_table;

        let (table, min_val, _range) = build_jump_table(cases, default);
        let max_val = cases.iter().map(|&(v, _)| v).max().expect("switch must have cases");

        self.emit_load_operand(val);

        // Range check: if val < min or val > max, branch to default
        if min_val != 0 {
            self.load_imm32_to_reg("r1", min_val as u32);
            self.state.emit("    sub r0, r0, r1");
        }
        let range = max_val - min_val;
        if range <= 255 {
            emit!(self.state, "    cmp r0, #{}", range);
        } else {
            self.load_imm32_to_reg("r1", range as u32);
            self.state.emit("    cmp r0, r1");
        }
        let default_label = default.as_label();
        emit!(self.state, "    bhi {}", default_label);

        // Load jump table entry and branch
        let table_id = self.state.next_label_id();
        emit!(self.state, "    ldr r1, =.Ljt_{}", table_id);
        self.state.emit("    ldr pc, [r1, r0, lsl #2]");

        // Emit jump table in .rodata
        emit!(self.state, "    .section .rodata");
        emit!(self.state, "    .align 2");
        emit!(self.state, ".Ljt_{}:", table_id);
        for target in &table {
            let label = target.as_label();
            emit!(self.state, "    .long {}", label);
        }
        self.state.emit("    .text");
    }

    pub(super) fn emit_switch_case_branch_impl(&mut self, case_val: i64, label: &str, _ty: IrType) {
        if case_val >= 0 && case_val <= 255 {
            emit!(self.state, "    cmp r0, #{}", case_val);
        } else {
            self.load_imm32_to_reg("r1", case_val as u32);
            self.state.emit("    cmp r0, r1");
        }
        emit!(self.state, "    beq {}", label);
    }

    // ---- Call store result helpers ----

    pub(super) fn emit_call_store_i128_result_impl(&mut self, dest: &Value) {
        // 64-bit return value in r0:r1
        self.store_r0_r1_to(dest);
    }

    pub(super) fn emit_call_store_f128_result_impl(&mut self, dest: &Value) {
        // F128 on ARM32 is double, returned in r0:r1
        self.store_r0_r1_to(dest);
    }

    pub(super) fn emit_call_move_f32_to_acc_impl(&mut self) {
        // On soft-float ARM, F32 returned in r0. Already there.
    }

    pub(super) fn emit_call_move_f64_to_acc_impl(&mut self) {
        // On soft-float ARM, F64 returned in r0:r1. r0 already has low half.
    }
}

impl ArchCodegen for Armv7Codegen {
    fn state(&mut self) -> &mut CodegenState { &mut self.state }
    fn state_ref(&self) -> &CodegenState { &self.state }

    fn ptr_directive(&self) -> PtrDirective { PtrDirective::Long }

    fn get_phys_reg_for_value(&self, val_id: u32) -> Option<PhysReg> {
        self.reg_assignments.get(&val_id).copied()
    }

    fn emit_reg_to_reg_move(&mut self, src: PhysReg, dest: PhysReg) {
        let src_name = phys_reg_name(src);
        let dest_name = phys_reg_name(dest);
        emit!(self.state, "    mov {}, {}", dest_name, src_name);
    }

    fn emit_acc_to_phys_reg(&mut self, dest: PhysReg) {
        let dest_name = phys_reg_name(dest);
        emit!(self.state, "    mov {}, r0", dest_name);
    }

    // ---- Standard trait methods ----
    fn emit_load_operand(&mut self, op: &Operand) { self.operand_to_r0(op); }
    fn emit_store_result(&mut self, dest: &Value) { self.store_r0_to(dest); }
    fn emit_save_acc(&mut self) { self.state.emit("    mov r2, r0"); }
    fn emit_add_secondary_to_acc(&mut self) { self.state.emit("    add r0, r0, r1"); }
    fn emit_acc_to_secondary(&mut self) { self.state.emit("    mov r1, r0"); }
    fn emit_memcpy_store_dest_from_acc(&mut self) { self.state.emit("    mov r2, r0"); }
    fn emit_memcpy_store_src_from_acc(&mut self) { self.state.emit("    mov r3, r0"); }
    fn current_return_type(&self) -> IrType { self.current_return_type }
    fn emit_gep_add_const_to_acc(&mut self, offset: i64) {
        if offset != 0 {
            if offset > 0 && offset <= 255 {
                emit!(self.state, "    add r0, r0, #{}", offset);
            } else if offset < 0 && offset >= -255 {
                emit!(self.state, "    sub r0, r0, #{}", -offset);
            } else {
                self.load_imm32_to_reg("r1", offset as u32);
                self.state.emit("    add r0, r0, r1");
            }
        }
    }

    fn emit_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if matches!(ty, IrType::I64 | IrType::U64) || is_i128_type(ty) {
            self.emit_i128_binop(dest, op, lhs, rhs);
            self.state.reg_cache.invalidate_all();
            return;
        }
        if ty.is_float() {
            let float_op = crate::backend::cast::classify_float_binop(op)
                .unwrap_or_else(|| panic!("unsupported float binop: {:?} on type {:?}", op, ty));
            self.emit_float_binop(dest, float_op, lhs, rhs, ty);
            return;
        }
        self.emit_int_binop(dest, op, lhs, rhs, ty);
    }

    fn emit_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if ty == IrType::F128 {
            self.emit_f128_cmp(dest, op, lhs, rhs);
            return;
        }
        if ty == IrType::F64 || ty == IrType::F32 {
            self.emit_float_cmp(dest, op, lhs, rhs, ty);
            return;
        }
        if matches!(ty, IrType::I64 | IrType::U64) || is_i128_type(ty) {
            self.emit_i128_cmp(dest, op, lhs, rhs);
            return;
        }
        self.emit_int_cmp(dest, op, lhs, rhs, ty);
    }

    fn emit_unaryop(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand, ty: IrType) {
        if op == IrUnaryOp::IsConstant {
            self.emit_load_operand(&Operand::Const(IrConst::I32(0)));
            self.emit_store_result(dest);
            return;
        }
        if ty == IrType::F128 && matches!(op, IrUnaryOp::Neg) {
            self.emit_f128_neg(dest, src);
            return;
        }
        if ty == IrType::F64 && matches!(op, IrUnaryOp::Neg) {
            // F64 negation needs both halves loaded; handle like F128 neg
            self.load_wide_to_r0_r1(src);
            self.state.emit("    eor r1, r1, #0x80000000");
            self.store_r0_r1_to(dest);
            self.state.reg_cache.invalidate_all();
            return;
        }
        if matches!(ty, IrType::I64 | IrType::U64) || is_i128_type(ty) {
            self.load_wide_to_r0_r1(src);
            match op {
                IrUnaryOp::Neg => {
                    // negate 64-bit: rsbs r0, r0, #0; rsc r1, r1, #0
                    self.state.emit("    rsbs r0, r0, #0");
                    self.state.emit("    rsc r1, r1, #0");
                }
                IrUnaryOp::Not => {
                    self.state.emit("    mvn r0, r0");
                    self.state.emit("    mvn r1, r1");
                }
                IrUnaryOp::Clz => {
                    // CLZ of 64-bit: if hi != 0, clz(hi), else 32 + clz(lo)
                    self.state.emit("    cmp r1, #0");
                    self.state.emit("    clzne r0, r1");
                    self.state.emit("    clzeq r0, r0");
                    self.state.emit("    addeq r0, r0, #32");
                    self.state.emit("    mov r1, #0");
                }
                IrUnaryOp::Ctz => {
                    // CTZ of 64-bit: if lo != 0, ctz(lo), else 32 + ctz(hi)
                    self.state.emit("    cmp r0, #0");
                    self.state.emit("    rbitne r0, r0");
                    self.state.emit("    clzne r0, r0");
                    self.state.emit("    rbiteq r0, r1");
                    self.state.emit("    clzeq r0, r0");
                    self.state.emit("    addeq r0, r0, #32");
                    self.state.emit("    mov r1, #0");
                }
                IrUnaryOp::Bswap => {
                    // Bswap 64-bit: swap halves and bswap each
                    self.state.emit("    rev r2, r0");
                    self.state.emit("    rev r0, r1");
                    self.state.emit("    mov r1, r2");
                }
                IrUnaryOp::Popcount => {
                    // Use helper function
                    self.state.emit("    bl __popcountdi2");
                    self.state.emit("    mov r1, #0");
                }
                IrUnaryOp::IsConstant => unreachable!(),
            }
            self.store_r0_r1_to(dest);
            self.state.reg_cache.invalidate_all();
            return;
        }
        self.operand_to_r0(src);
        match op {
            IrUnaryOp::Neg => {
                if ty.is_float() {
                    self.emit_float_neg(ty);
                } else {
                    self.state.emit("    rsb r0, r0, #0");
                }
            }
            IrUnaryOp::Not => {
                self.state.emit("    mvn r0, r0");
                // Mask to type width
                match ty.size() {
                    1 => self.state.emit("    and r0, r0, #0xff"),
                    2 => {
                        self.load_imm32_to_reg("r1", 0xffff);
                        self.state.emit("    and r0, r0, r1");
                    }
                    _ => {}
                }
            }
            IrUnaryOp::Clz => {
                self.state.emit("    clz r0, r0");
                let type_bits = (ty.size() * 8) as i32;
                if type_bits < 32 {
                    let adjust = 32 - type_bits;
                    emit!(self.state, "    sub r0, r0, #{}", adjust);
                }
            }
            IrUnaryOp::Ctz => {
                self.state.emit("    rbit r0, r0");
                self.state.emit("    clz r0, r0");
                let type_bits = ty.size() * 8;
                if type_bits < 32 {
                    emit!(self.state, "    cmp r0, #{}", type_bits);
                    emit!(self.state, "    movgt r0, #{}", type_bits);
                }
            }
            IrUnaryOp::Popcount => {
                self.state.emit("    bl __popcountsi2");
            }
            IrUnaryOp::Bswap => {
                self.state.emit("    rev r0, r0");
                match ty.size() {
                    2 => self.state.emit("    lsr r0, r0, #16"),
                    1 => {} // No-op for byte
                    _ => {}
                }
            }
            IrUnaryOp::IsConstant => unreachable!("handled above"),
        }
        self.state.reg_cache.invalidate_acc();
        self.store_r0_to(dest);
    }

    fn emit_call(&mut self, args: &[Operand], arg_types: &[IrType], direct_name: Option<&str>,
                 func_ptr: Option<&Operand>, dest: Option<Value>, return_type: IrType,
                 is_variadic: bool, _num_fixed_args: usize, struct_arg_sizes: &[Option<usize>],
                 struct_arg_aligns: &[Option<usize>],
                 struct_arg_classes: &[Vec<crate::common::types::EightbyteClass>],
                 struct_arg_riscv_float_classes: &[Option<crate::common::types::RiscvFloatClass>],
                 _is_sret: bool,
                 _is_fastcall: bool,
                 _ret_eightbyte_classes: &[crate::common::types::EightbyteClass]) {
        use crate::backend::call_abi::*;
        let config = self.call_abi_config();
        let arg_classes_vec = classify_call_args(args, arg_types, struct_arg_sizes, struct_arg_aligns, struct_arg_classes, struct_arg_riscv_float_classes, is_variadic, &config);
        let indirect = func_ptr.is_some() && direct_name.is_none();
        if indirect {
            self.emit_call_spill_fptr(func_ptr.expect("indirect call requires func_ptr"));
        }
        let stack_arg_space = self.emit_call_compute_stack_space(&arg_classes_vec, arg_types);
        let f128_temp_space = self.emit_call_f128_pre_convert(args, &arg_classes_vec, arg_types, stack_arg_space);
        self.state().reg_cache.invalidate_acc();
        let total_sp_adjust = self.emit_call_stack_args(args, &arg_classes_vec, arg_types, stack_arg_space,
                                                        if indirect { self.emit_call_fptr_spill_size() } else { 0 },
                                                        f128_temp_space);
        self.state().reg_cache.invalidate_acc();
        self.emit_call_reg_args(args, &arg_classes_vec, arg_types, total_sp_adjust, f128_temp_space, stack_arg_space, &[]);
        self.emit_call_instruction(direct_name, func_ptr, indirect, stack_arg_space);
        self.emit_call_cleanup(stack_arg_space, f128_temp_space, indirect);
        if let Some(dest) = dest {
            self.emit_call_store_result(&dest, return_type);
        }
    }

    fn callee_pops_bytes_for_sret(&self, _is_sret: bool) -> usize {
        0 // ARM doesn't pop sret from stack
    }

    fn function_type_directive(&self) -> &'static str { "%function" }

    // ---- Control flow ----
    fn jump_mnemonic(&self) -> &'static str { "b" }
    fn trap_instruction(&self) -> &'static str { ".inst 0xe7f000f0" }

    fn emit_branch_nonzero(&mut self, label: &str) {
        self.state.emit("    cmp r0, #0");
        emit!(self.state, "    bne {}", label);
    }

    fn emit_cond_branch_blocks(&mut self, cond: &Operand, true_block: BlockId, false_block: BlockId) {
        match cond {
            Operand::Const(IrConst::I64(v)) => {
                if *v != 0 { self.emit_branch_to_block(true_block); }
                else { self.emit_branch_to_block(false_block); }
                return;
            }
            Operand::Const(IrConst::I32(v)) => {
                if *v != 0 { self.emit_branch_to_block(true_block); }
                else { self.emit_branch_to_block(false_block); }
                return;
            }
            _ => {}
        }
        if let Operand::Value(v) = cond {
            if self.state.is_wide_value(v.0) {
                // 64-bit: OR both halves to check truthiness
                if let Some(slot) = self.state.get_slot(v.0) {
                    let slot_ref = self.slot_ref(slot);
                    emit!(self.state, "    ldr r0, {}", slot_ref);
                    let hi_offset = slot.0 + 4;
                    emit!(self.state, "    ldr r1, [r11, #{}]", hi_offset);
                } else {
                    // Wide value without slot (shouldn't happen, but be safe)
                    self.operand_to_r0(cond);
                    self.state.emit("    mov r1, #0");
                }
                self.state.emit("    orrs r0, r0, r1");
                let true_label = true_block.as_label();
                emit!(self.state, "    bne {}", true_label);
                self.emit_branch_to_block(false_block);
                return;
            }
        }
        self.emit_load_operand(cond);
        self.state.emit("    cmp r0, #0");
        let true_label = true_block.as_label();
        emit!(self.state, "    bne {}", true_label);
        self.emit_branch_to_block(false_block);
    }

    fn emit_switch(&mut self, val: &Operand, cases: &[(i64, BlockId)], default: &BlockId, ty: IrType) {
        // Check density for jump table eligibility
        use crate::backend::traits::{MIN_JUMP_TABLE_CASES, MAX_JUMP_TABLE_RANGE, MIN_JUMP_TABLE_DENSITY_PERCENT};
        let use_jump_table = if self.state_ref().no_jump_tables {
            false
        } else if cases.len() >= MIN_JUMP_TABLE_CASES {
            let min_val = cases.iter().map(|&(v, _)| v).min().expect("switch must have cases");
            let max_val = cases.iter().map(|&(v, _)| v).max().expect("switch must have cases");
            let range = (max_val - min_val + 1) as usize;
            range <= MAX_JUMP_TABLE_RANGE && cases.len() * 100 / range >= MIN_JUMP_TABLE_DENSITY_PERCENT
        } else {
            false
        };

        if use_jump_table {
            self.emit_switch_jump_table(val, cases, default, ty);
        } else {
            // Sparse: linear compare-and-branch chain
            self.emit_load_operand(val);
            for &(case_val, target) in cases {
                let label = target.as_label();
                self.emit_switch_case_branch(case_val, &label, ty);
            }
            self.emit_branch_to_block(*default);
        }
    }

    fn emit_indirect_branch(&mut self, target: &Operand) {
        self.emit_load_operand(target);
        self.state.emit("    bx r0");
    }

    fn emit_inline_asm(&mut self, template: &str, outputs: &[(String, Value, Option<String>)], inputs: &[(String, Operand, Option<String>)], clobbers: &[String], operand_types: &[IrType], goto_labels: &[(String, BlockId)], input_symbols: &[Option<String>]) {
        self.emit_inline_asm_impl(template, outputs, inputs, clobbers, operand_types, goto_labels, input_symbols);
    }

    fn emit_va_arg_struct(&mut self, dest_ptr: &Value, va_list_ptr: &Value, _size: usize) {
        // ARMv7: structs passed by value on the stack, simple pointer increment
        self.emit_va_arg_impl(dest_ptr, va_list_ptr, IrType::I32);
    }

    delegate_to_impl! {
        // prologue
        fn calculate_stack_space(&mut self, func: &IrFunction) -> i64 => calculate_stack_space_impl;
        fn aligned_frame_size(&self, raw_space: i64) -> i64 => aligned_frame_size_impl;
        fn emit_prologue(&mut self, func: &IrFunction, frame_size: i64) => emit_prologue_impl;
        fn emit_epilogue(&mut self, frame_size: i64) => emit_epilogue_impl;
        fn emit_store_params(&mut self, func: &IrFunction) => emit_store_params_impl;
        fn emit_param_ref(&mut self, dest: &Value, param_idx: usize, ty: IrType) => emit_param_ref_impl;
        fn emit_epilogue_and_ret(&mut self, frame_size: i64) => emit_epilogue_and_ret_impl;
        fn store_instr_for_type(&self, ty: IrType) -> &'static str => store_instr_for_type_impl;
        fn load_instr_for_type(&self, ty: IrType) -> &'static str => load_instr_for_type_impl;
        // memory
        fn emit_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) => emit_store_impl;
        fn emit_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) => emit_load_impl;
        fn emit_typed_store_to_slot(&mut self, instr: &'static str, ty: IrType, slot: StackSlot) => emit_typed_store_to_slot_impl;
        fn emit_typed_load_from_slot(&mut self, instr: &'static str, slot: StackSlot) => emit_typed_load_from_slot_impl;
        fn emit_load_ptr_from_slot(&mut self, slot: StackSlot, val_id: u32) => emit_load_ptr_from_slot_impl;
        fn emit_typed_store_indirect(&mut self, instr: &'static str, ty: IrType) => emit_typed_store_indirect_impl;
        fn emit_typed_load_indirect(&mut self, instr: &'static str) => emit_typed_load_indirect_impl;
        fn emit_add_offset_to_addr_reg(&mut self, offset: i64) => emit_add_offset_to_addr_reg_impl;
        fn emit_slot_addr_to_secondary(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_slot_addr_to_secondary_impl;
        fn emit_add_imm_to_acc(&mut self, imm: i64) => emit_add_imm_to_acc_impl;
        fn emit_round_up_acc_to_16(&mut self) => emit_round_up_acc_to_16_impl;
        fn emit_sub_sp_by_acc(&mut self) => emit_sub_sp_by_acc_impl;
        fn emit_mov_sp_to_acc(&mut self) => emit_mov_sp_to_acc_impl;
        fn emit_mov_acc_to_sp(&mut self) => emit_mov_acc_to_sp_impl;
        fn emit_align_acc(&mut self, align: usize) => emit_align_acc_impl;
        fn emit_alloca_aligned_addr(&mut self, slot: StackSlot, val_id: u32) => emit_alloca_aligned_addr_impl;
        fn emit_alloca_aligned_addr_to_acc(&mut self, slot: StackSlot, val_id: u32) => emit_alloca_aligned_addr_to_acc_impl;
        fn emit_memcpy_load_dest_addr(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_memcpy_load_dest_addr_impl;
        fn emit_memcpy_load_src_addr(&mut self, slot: StackSlot, is_alloca: bool, val_id: u32) => emit_memcpy_load_src_addr_impl;
        fn emit_memcpy_impl(&mut self, size: usize) => emit_memcpy_impl_impl;
        fn emit_jump_indirect(&mut self) => emit_jump_indirect_impl;
        // alu
        fn emit_float_binop(&mut self, dest: &Value, op: crate::backend::cast::FloatOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_float_binop_full;
        fn emit_float_binop_impl(&mut self, mnemonic: &str, ty: IrType) => emit_float_binop_impl_impl;
        fn emit_float_neg(&mut self, ty: IrType) => emit_float_neg_impl;
        fn emit_int_neg(&mut self, ty: IrType) => emit_int_neg_impl;
        fn emit_int_not(&mut self, ty: IrType) => emit_int_not_impl;
        fn emit_int_clz(&mut self, ty: IrType) => emit_int_clz_impl;
        fn emit_int_ctz(&mut self, ty: IrType) => emit_int_ctz_impl;
        fn emit_int_bswap(&mut self, ty: IrType) => emit_int_bswap_impl;
        fn emit_int_popcount(&mut self, ty: IrType) => emit_int_popcount_impl;
        fn emit_int_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_int_binop_impl;
        // comparison
        fn emit_float_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_float_cmp_impl;
        fn emit_f128_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand) => emit_f128_cmp_impl;
        fn emit_int_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType) => emit_int_cmp_impl;
        fn emit_fused_cmp_branch(&mut self, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType, true_label: &str, false_label: &str) => emit_fused_cmp_branch_impl;
        fn emit_f128_neg(&mut self, dest: &Value, src: &Operand) => emit_f128_neg_impl;
        // cast
        fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) => emit_cast_impl;
        fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType) => emit_cast_instrs_impl;
        // calls
        fn call_abi_config(&self) -> call_abi::CallAbiConfig => call_abi_config_impl;
        fn emit_call_f128_pre_convert(&mut self, args: &[Operand], arg_classes: &[call_abi::CallArgClass], arg_types: &[IrType], stack_arg_space: usize) -> usize => emit_call_f128_pre_convert_impl;
        fn emit_call_compute_stack_space(&self, arg_classes: &[call_abi::CallArgClass], arg_types: &[IrType]) -> usize => emit_call_compute_stack_space_impl;
        fn emit_call_stack_args(&mut self, args: &[Operand], arg_classes: &[call_abi::CallArgClass], arg_types: &[IrType], stack_arg_space: usize, fptr_spill: usize, f128_temp_space: usize) -> i64 => emit_call_stack_args_impl;
        fn emit_call_reg_args(&mut self, args: &[Operand], arg_classes: &[call_abi::CallArgClass], arg_types: &[IrType], total_sp_adjust: i64, f128_temp_space: usize, stack_arg_space: usize, struct_arg_riscv_float_classes: &[Option<crate::common::types::RiscvFloatClass>]) => emit_call_reg_args_impl;
        fn emit_call_instruction(&mut self, direct_name: Option<&str>, func_ptr: Option<&Operand>, indirect: bool, stack_arg_space: usize) => emit_call_instruction_impl;
        fn emit_call_cleanup(&mut self, stack_arg_space: usize, f128_temp_space: usize, indirect: bool) => emit_call_cleanup_impl;
        fn emit_call_store_result(&mut self, dest: &Value, return_type: IrType) => emit_call_store_result_impl;
        fn emit_call_store_i128_result(&mut self, dest: &Value) => emit_call_store_i128_result_impl;
        fn emit_call_store_f128_result(&mut self, dest: &Value) => emit_call_store_f128_result_impl;
        fn emit_call_move_f32_to_acc(&mut self) => emit_call_move_f32_to_acc_impl;
        fn emit_call_move_f64_to_acc(&mut self) => emit_call_move_f64_to_acc_impl;
        fn emit_call_spill_fptr(&mut self, func_ptr: &Operand) => emit_call_spill_fptr_impl;
        fn emit_call_fptr_spill_size(&self) -> usize => emit_call_fptr_spill_size_impl;
        // globals
        fn emit_global_addr(&mut self, dest: &Value, name: &str) => emit_global_addr_impl;
        fn emit_tls_global_addr(&mut self, dest: &Value, name: &str) => emit_tls_global_addr_impl;
        // variadic
        fn emit_va_arg(&mut self, dest: &Value, va_list_ptr: &Value, result_ty: IrType) => emit_va_arg_impl;
        fn emit_va_start(&mut self, va_list_ptr: &Value) => emit_va_start_impl;
        fn emit_va_copy(&mut self, dest_ptr: &Value, src_ptr: &Value) => emit_va_copy_impl;
        // returns
        fn emit_return(&mut self, val: Option<&Operand>, frame_size: i64) => emit_return_impl;
        fn emit_return_i128_to_regs(&mut self) => emit_return_i128_to_regs_impl;
        fn emit_return_f128_to_reg(&mut self) => emit_return_f128_to_reg_impl;
        fn emit_return_f32_to_reg(&mut self) => emit_return_f32_to_reg_impl;
        fn emit_return_f64_to_reg(&mut self) => emit_return_f64_to_reg_impl;
        fn emit_return_int_to_reg(&mut self) => emit_return_int_to_reg_impl;
        fn emit_get_return_f64_second(&mut self, dest: &Value) => emit_get_return_f64_second_impl;
        fn emit_set_return_f64_second(&mut self, src: &Operand) => emit_set_return_f64_second_impl;
        fn emit_get_return_f32_second(&mut self, dest: &Value) => emit_get_return_f32_second_impl;
        fn emit_set_return_f32_second(&mut self, src: &Operand) => emit_set_return_f32_second_impl;
        fn emit_get_return_f128_second(&mut self, dest: &Value) => emit_get_return_f128_second_impl;
        fn emit_set_return_f128_second(&mut self, src: &Operand) => emit_set_return_f128_second_impl;
        // atomics
        fn emit_atomic_rmw(&mut self, dest: &Value, op: AtomicRmwOp, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_rmw_impl;
        fn emit_atomic_cmpxchg(&mut self, dest: &Value, ptr: &Operand, expected: &Operand, desired: &Operand, ty: IrType, success_ordering: AtomicOrdering, failure_ordering: AtomicOrdering, returns_bool: bool) => emit_atomic_cmpxchg_impl;
        fn emit_atomic_load(&mut self, dest: &Value, ptr: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_load_impl;
        fn emit_atomic_store(&mut self, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering) => emit_atomic_store_impl;
        fn emit_fence(&mut self, ordering: AtomicOrdering) => emit_fence_impl;
        // i128 ops
        fn emit_sign_extend_acc_high(&mut self) => emit_sign_extend_acc_high_impl;
        fn emit_zero_acc_high(&mut self) => emit_zero_acc_high_impl;
        fn emit_load_acc_pair(&mut self, op: &Operand) => emit_load_acc_pair_impl;
        fn emit_store_acc_pair(&mut self, dest: &Value) => emit_store_acc_pair_impl;
        fn emit_store_pair_to_slot(&mut self, slot: StackSlot) => emit_store_pair_to_slot_impl;
        fn emit_load_pair_from_slot(&mut self, slot: StackSlot) => emit_load_pair_from_slot_impl;
        fn emit_save_acc_pair(&mut self) => emit_save_acc_pair_impl;
        fn emit_store_pair_indirect(&mut self) => emit_store_pair_indirect_impl;
        fn emit_load_pair_indirect(&mut self) => emit_load_pair_indirect_impl;
        fn emit_i128_neg(&mut self) => emit_i128_neg_impl;
        fn emit_i128_not(&mut self) => emit_i128_not_impl;
        fn emit_i128_to_float_call(&mut self, src: &Operand, from_signed: bool, to_ty: IrType) => emit_i128_to_float_call_impl;
        fn emit_float_to_i128_call(&mut self, src: &Operand, to_signed: bool, from_ty: IrType) => emit_float_to_i128_call_impl;
        fn emit_i128_prep_binop(&mut self, lhs: &Operand, rhs: &Operand) => emit_i128_prep_binop_impl;
        fn emit_i128_add(&mut self) => emit_i128_add_impl;
        fn emit_i128_sub(&mut self) => emit_i128_sub_impl;
        fn emit_i128_mul(&mut self) => emit_i128_mul_impl;
        fn emit_i128_and(&mut self) => emit_i128_and_impl;
        fn emit_i128_or(&mut self) => emit_i128_or_impl;
        fn emit_i128_xor(&mut self) => emit_i128_xor_impl;
        fn emit_i128_shl(&mut self) => emit_i128_shl_impl;
        fn emit_i128_lshr(&mut self) => emit_i128_lshr_impl;
        fn emit_i128_ashr(&mut self) => emit_i128_ashr_impl;
        fn emit_i128_shl_const(&mut self, amount: u32) => emit_i128_shl_const_impl;
        fn emit_i128_lshr_const(&mut self, amount: u32) => emit_i128_lshr_const_impl;
        fn emit_i128_ashr_const(&mut self, amount: u32) => emit_i128_ashr_const_impl;
        fn emit_i128_divrem_call(&mut self, func_name: &str, lhs: &Operand, rhs: &Operand) => emit_i128_divrem_call_impl;
        fn emit_i128_store_result(&mut self, dest: &Value) => emit_i128_store_result_impl;
        fn emit_i128_cmp_eq(&mut self, is_ne: bool) => emit_i128_cmp_eq_impl;
        fn emit_i128_cmp_ordered(&mut self, op: IrCmpOp) => emit_i128_cmp_ordered_impl;
        fn emit_i128_cmp_store_result(&mut self, dest: &Value) => emit_i128_cmp_store_result_impl;
        // switch
        fn emit_switch_jump_table(&mut self, val: &Operand, cases: &[(i64, BlockId)], default: &BlockId, ty: IrType) => emit_switch_jump_table_impl;
        fn emit_switch_case_branch(&mut self, case_val: i64, label: &str, ty: IrType) => emit_switch_case_branch_impl;
        // intrinsics
        fn emit_intrinsic(&mut self, dest: &Option<Value>, op: &IntrinsicOp, dest_ptr: &Option<Value>, args: &[Operand]) => emit_intrinsic_impl;
    }
}
