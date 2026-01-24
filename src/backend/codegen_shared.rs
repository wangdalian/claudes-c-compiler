//! Shared code generation framework for all backends.
//!
//! All three backends (x86-64, AArch64, RISC-V 64) use the same stack-based
//! code generation strategy: each IR value is assigned a stack slot, and
//! instructions move values through a primary accumulator register.
//!
//! This module extracts the shared instruction dispatch and function generation
//! logic. Each architecture implements `ArchCodegen` to provide its specific
//! register names, instruction mnemonics, and ABI details.

use std::collections::{HashMap, HashSet};
use crate::ir::ir::*;
use crate::common::types::IrType;
use super::common::{self, AsmOutput, PtrDirective};

/// Stack slot location for a value. Interpretation varies by arch:
/// - x86: negative offset from %rbp
/// - ARM: positive offset from sp
/// - RISC-V: negative offset from s0
#[derive(Debug, Clone, Copy)]
pub struct StackSlot(pub i64);

/// Shared codegen state, used by all backends.
pub struct CodegenState {
    pub out: AsmOutput,
    pub stack_offset: i64,
    pub value_locations: HashMap<u32, StackSlot>,
    /// Values that are allocas (their stack slot IS the data, not a pointer to data).
    pub alloca_values: HashSet<u32>,
    /// Type associated with each alloca (for type-aware loads/stores).
    pub alloca_types: HashMap<u32, IrType>,
    /// Alloca values that need runtime alignment > 16 bytes.
    pub alloca_alignments: HashMap<u32, usize>,
    /// Values that are 128-bit integers (need 16-byte copy).
    pub i128_values: HashSet<u32>,
    /// Counter for generating unique labels (e.g., memcpy loops).
    label_counter: u32,
    /// Whether position-independent code (PIC) generation is enabled.
    pub pic_mode: bool,
    /// Set of symbol names that are locally defined (not extern) and have internal
    /// linkage (static) — these can use direct addressing even in PIC mode.
    pub local_symbols: HashSet<String>,
    /// Whether the current function contains DynAlloca instructions.
    /// When true, the epilogue must restore SP from the frame pointer instead of
    /// adding back the compile-time frame size.
    pub has_dyn_alloca: bool,
}

impl CodegenState {
    pub fn new() -> Self {
        Self {
            out: AsmOutput::new(),
            stack_offset: 0,
            value_locations: HashMap::new(),
            alloca_values: HashSet::new(),
            alloca_types: HashMap::new(),
            alloca_alignments: HashMap::new(),
            i128_values: HashSet::new(),
            label_counter: 0,
            pic_mode: false,
            local_symbols: HashSet::new(),
            has_dyn_alloca: false,
        }
    }

    pub fn next_label_id(&mut self) -> u32 {
        let id = self.label_counter;
        self.label_counter += 1;
        id
    }

    /// Generate a fresh label with the given prefix.
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let id = self.next_label_id();
        format!(".L{}_{}", prefix, id)
    }

    pub fn emit(&mut self, s: &str) {
        self.out.emit(s);
    }

    pub fn reset_for_function(&mut self) {
        self.stack_offset = 0;
        self.value_locations.clear();
        self.alloca_values.clear();
        self.alloca_types.clear();
        self.alloca_alignments.clear();
        self.i128_values.clear();
        self.has_dyn_alloca = false;
    }

    /// Get the over-alignment requirement for an alloca (> 16 bytes), or None.
    pub fn alloca_over_align(&self, v: u32) -> Option<usize> {
        self.alloca_alignments.get(&v).copied()
    }

    pub fn is_alloca(&self, v: u32) -> bool {
        self.alloca_values.contains(&v)
    }

    pub fn get_slot(&self, v: u32) -> Option<StackSlot> {
        self.value_locations.get(&v).copied()
    }

    pub fn is_i128_value(&self, v: u32) -> bool {
        self.i128_values.contains(&v)
    }

    /// Returns true if the given symbol needs GOT indirection in PIC mode.
    /// A symbol needs GOT if PIC is enabled AND it's not a local (static) symbol.
    /// Local labels (starting with '.') are always PIC-safe via RIP-relative.
    pub fn needs_got(&self, name: &str) -> bool {
        if !self.pic_mode {
            return false;
        }
        // Local labels (.Lxxx) never need GOT
        if name.starts_with('.') {
            return false;
        }
        // Symbols defined with internal linkage (static) don't need GOT
        !self.local_symbols.contains(name)
    }

    /// Returns true if a function call needs PLT indirection in PIC mode.
    pub fn needs_plt(&self, name: &str) -> bool {
        self.needs_got(name)
    }
}

/// Trait that each architecture implements to provide its specific code generation.
///
/// The shared framework calls these methods during instruction dispatch.
/// Each method should emit the appropriate assembly instructions.
pub trait ArchCodegen {
    /// Mutable access to the shared codegen state.
    fn state(&mut self) -> &mut CodegenState;
    /// Immutable access to the shared codegen state.
    fn state_ref(&self) -> &CodegenState;

    /// The pointer directive for this architecture's data emission.
    fn ptr_directive(&self) -> PtrDirective;

    /// Calculate stack space and assign locations for all values in the function.
    /// Returns the raw stack space needed (before alignment).
    fn calculate_stack_space(&mut self, func: &IrFunction) -> i64;

    /// Emit function prologue (save frame pointer, allocate stack).
    fn emit_prologue(&mut self, func: &IrFunction, frame_size: i64);

    /// Emit function epilogue (restore frame pointer, deallocate stack).
    fn emit_epilogue(&mut self, frame_size: i64);

    /// Store function parameters from argument registers to their stack slots.
    fn emit_store_params(&mut self, func: &IrFunction);

    /// Load an operand into the primary accumulator register (rax/x0/t0).
    fn emit_load_operand(&mut self, op: &Operand);

    /// Store the primary accumulator to a value's stack slot.
    fn emit_store_result(&mut self, dest: &Value);

    /// Compute the runtime-aligned address of an over-aligned alloca into the
    /// pointer register (same register used by emit_load_ptr_from_slot: rcx on x86).
    fn emit_alloca_aligned_addr(&mut self, slot: StackSlot, val_id: u32);

    /// Compute aligned alloca address into the accumulator (rax/x0/a0).
    fn emit_alloca_aligned_addr_to_acc(&mut self, slot: StackSlot, val_id: u32);

    /// Emit a store instruction: store val to the address in ptr.
    /// Default implementation uses i128 pair ops and slot primitives.
    fn emit_store(&mut self, val: &Operand, ptr: &Value, ty: IrType) {
        if is_i128_type(ty) {
            self.emit_load_acc_pair(val);
            if let Some(slot) = self.state_ref().get_slot(ptr.0) {
                if self.state_ref().is_alloca(ptr.0) {
                    if self.state_ref().alloca_over_align(ptr.0).is_some() {
                        self.emit_save_acc_pair();
                        self.emit_alloca_aligned_addr(slot, ptr.0);
                        self.emit_store_pair_indirect();
                    } else {
                        self.emit_store_pair_to_slot(slot);
                    }
                } else {
                    self.emit_save_acc_pair();
                    self.emit_load_ptr_from_slot(slot);
                    self.emit_store_pair_indirect();
                }
            }
            return;
        }
        self.emit_load_operand(val);
        if let Some(slot) = self.state_ref().get_slot(ptr.0) {
            let store_instr = self.store_instr_for_type(ty);
            if self.state_ref().is_alloca(ptr.0) {
                if self.state_ref().alloca_over_align(ptr.0).is_some() {
                    self.emit_save_acc();
                    self.emit_alloca_aligned_addr(slot, ptr.0);
                    self.emit_typed_store_indirect(store_instr, ty);
                } else {
                    self.emit_typed_store_to_slot(store_instr, ty, slot);
                }
            } else {
                self.emit_save_acc();
                self.emit_load_ptr_from_slot(slot);
                self.emit_typed_store_indirect(store_instr, ty);
            }
        }
    }

    /// Emit a load instruction: load from the address in ptr to dest.
    /// Default implementation uses i128 pair ops and slot primitives.
    fn emit_load(&mut self, dest: &Value, ptr: &Value, ty: IrType) {
        if is_i128_type(ty) {
            if let Some(slot) = self.state_ref().get_slot(ptr.0) {
                if self.state_ref().is_alloca(ptr.0) {
                    if self.state_ref().alloca_over_align(ptr.0).is_some() {
                        self.emit_alloca_aligned_addr(slot, ptr.0);
                        self.emit_load_pair_indirect();
                    } else {
                        self.emit_load_pair_from_slot(slot);
                    }
                } else {
                    self.emit_load_ptr_from_slot(slot);
                    self.emit_load_pair_indirect();
                }
                self.emit_store_acc_pair(dest);
            }
            return;
        }
        if let Some(slot) = self.state_ref().get_slot(ptr.0) {
            let load_instr = self.load_instr_for_type(ty);
            if self.state_ref().is_alloca(ptr.0) {
                if self.state_ref().alloca_over_align(ptr.0).is_some() {
                    self.emit_alloca_aligned_addr(slot, ptr.0);
                    self.emit_typed_load_indirect(load_instr);
                } else {
                    self.emit_typed_load_from_slot(load_instr, slot);
                }
            } else {
                self.emit_load_ptr_from_slot(slot);
                self.emit_typed_load_indirect(load_instr);
            }
            self.emit_store_result(dest);
        }
    }

    /// Emit a binary operation. Default: dispatches float/integer ops via
    /// `emit_float_binop` and `emit_int_binop` primitives.
    fn emit_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType) {
        if ty.is_float() {
            let float_op = classify_float_binop(op)
                .unwrap_or_else(|| panic!("unsupported float binop: {:?} on type {:?}", op, ty));
            self.emit_float_binop(dest, float_op, lhs, rhs, ty);
            return;
        }
        self.emit_int_binop(dest, op, lhs, rhs, ty);
    }

    /// Emit a float binary operation (add/sub/mul/div).
    /// The default `emit_binop` dispatches here after classifying the operation.
    fn emit_float_binop(&mut self, dest: &Value, op: FloatOp, lhs: &Operand, rhs: &Operand, ty: IrType);

    /// Emit an integer binary operation (all IrBinOp variants).
    /// The default `emit_binop` dispatches here for non-float types.
    fn emit_int_binop(&mut self, dest: &Value, op: IrBinOp, lhs: &Operand, rhs: &Operand, ty: IrType);

    /// Emit a unary operation.
    /// Default dispatches i128/float/int to arch-specific primitives.
    fn emit_unaryop(&mut self, dest: &Value, op: IrUnaryOp, src: &Operand, ty: IrType) {
        if is_i128_type(ty) {
            self.emit_load_acc_pair(src);
            match op {
                IrUnaryOp::Neg => self.emit_i128_neg(),
                IrUnaryOp::Not => self.emit_i128_not(),
                _ => {} // Clz/Ctz/Bswap/Popcount not expected for 128-bit
            }
            self.emit_store_acc_pair(dest);
            return;
        }
        self.emit_load_operand(src);
        if ty.is_float() {
            match op {
                IrUnaryOp::Neg => self.emit_float_neg(ty),
                IrUnaryOp::Not => self.emit_int_not(ty),
                _ => {} // Clz/Ctz/Bswap/Popcount not applicable to floats
            }
        } else {
            match op {
                IrUnaryOp::Neg => self.emit_int_neg(ty),
                IrUnaryOp::Not => self.emit_int_not(ty),
                IrUnaryOp::Clz => self.emit_int_clz(ty),
                IrUnaryOp::Ctz => self.emit_int_ctz(ty),
                IrUnaryOp::Bswap => self.emit_int_bswap(ty),
                IrUnaryOp::Popcount => self.emit_int_popcount(ty),
            }
        }
        self.emit_store_result(dest);
    }

    /// Emit a comparison operation.
    fn emit_cmp(&mut self, dest: &Value, op: IrCmpOp, lhs: &Operand, rhs: &Operand, ty: IrType);

    /// Emit a function call (direct or indirect).
    /// `arg_types` provides the IR type of each argument for proper calling convention handling.
    /// `is_variadic` indicates the called function uses variadic arguments (affects calling convention).
    /// `num_fixed_args` is the number of named (non-variadic) parameters. For variadic calls on
    /// AArch64, arguments beyond this index must be passed in GP registers even if they are floats.
    fn emit_call(&mut self, args: &[Operand], arg_types: &[IrType], direct_name: Option<&str>,
                 func_ptr: Option<&Operand>, dest: Option<Value>, return_type: IrType,
                 is_variadic: bool, num_fixed_args: usize);

    /// Emit a global address load.
    fn emit_global_addr(&mut self, dest: &Value, name: &str);

    /// Emit a get-element-pointer (base + offset).
    /// Default: load base address to secondary reg, load offset to acc, add, store.
    fn emit_gep(&mut self, dest: &Value, base: &Value, offset: &Operand) {
        if let Some(slot) = self.state_ref().get_slot(base.0) {
            let is_alloca = self.state_ref().is_alloca(base.0);
            if is_alloca && self.state_ref().alloca_over_align(base.0).is_some() {
                // Over-aligned: compute aligned addr to acc, save to secondary
                self.emit_alloca_aligned_addr_to_acc(slot, base.0);
                self.emit_acc_to_secondary();
            } else {
                self.emit_slot_addr_to_secondary(slot, is_alloca);
            }
        }
        self.emit_load_operand(offset);
        self.emit_add_secondary_to_acc();
        self.emit_store_result(dest);
    }

    /// Move accumulator to secondary register (push on x86).
    fn emit_acc_to_secondary(&mut self);

    /// Emit architecture-specific instructions for a type cast, after the source
    /// value has been loaded into the accumulator. Does NOT load/store—only emits
    /// the conversion instructions.
    fn emit_cast_instrs(&mut self, from_ty: IrType, to_ty: IrType);

    /// Emit a type cast. Handles i128 widening/narrowing/copy using accumulator
    /// pair primitives, and delegates non-i128 casts to emit_cast_instrs.
    fn emit_cast(&mut self, dest: &Value, src: &Operand, from_ty: IrType, to_ty: IrType) {
        if is_i128_type(to_ty) && !is_i128_type(from_ty) {
            // Widening to 128-bit: load src, widen to 64-bit if needed, sign/zero extend high half
            self.emit_load_operand(src);
            if from_ty.size() < 8 {
                let widen_to = if from_ty.is_signed() { IrType::I64 } else { IrType::U64 };
                self.emit_cast_instrs(from_ty, widen_to);
            }
            if from_ty.is_signed() {
                self.emit_sign_extend_acc_high();
            } else {
                self.emit_zero_acc_high();
            }
            self.emit_store_acc_pair(dest);
            return;
        }
        if is_i128_type(from_ty) && !is_i128_type(to_ty) {
            // Narrowing from 128-bit: load pair, use low half, truncate if needed
            self.emit_load_acc_pair(src);
            if to_ty.size() < 8 {
                self.emit_cast_instrs(IrType::I64, to_ty);
            }
            self.emit_store_result(dest);
            return;
        }
        if is_i128_type(from_ty) && is_i128_type(to_ty) {
            // I128 <-> U128: same representation, just copy
            self.emit_load_acc_pair(src);
            self.emit_store_acc_pair(dest);
            return;
        }
        // Standard non-i128 cast
        self.emit_load_operand(src);
        self.emit_cast_instrs(from_ty, to_ty);
        self.emit_store_result(dest);
    }

    /// Emit a memory copy: copy `size` bytes from src address to dest address.
    /// Default: loads dest/src addresses via slot primitives, then calls emit_memcpy_impl.
    fn emit_memcpy(&mut self, dest: &Value, src: &Value, size: usize) {
        if let Some(dst_slot) = self.state_ref().get_slot(dest.0) {
            let is_alloca = self.state_ref().is_alloca(dest.0);
            if is_alloca && self.state_ref().alloca_over_align(dest.0).is_some() {
                self.emit_alloca_aligned_addr(dst_slot, dest.0);
                self.emit_memcpy_store_dest_from_acc();
            } else {
                self.emit_memcpy_load_dest_addr(dst_slot, is_alloca);
            }
        }
        if let Some(src_slot) = self.state_ref().get_slot(src.0) {
            let is_alloca = self.state_ref().is_alloca(src.0);
            if is_alloca && self.state_ref().alloca_over_align(src.0).is_some() {
                self.emit_alloca_aligned_addr(src_slot, src.0);
                self.emit_memcpy_store_src_from_acc();
            } else {
                self.emit_memcpy_load_src_addr(src_slot, is_alloca);
            }
        }
        self.emit_memcpy_impl(size);
    }

    /// Store accumulator to memcpy dest register (after computing aligned addr).
    fn emit_memcpy_store_dest_from_acc(&mut self);
    /// Store accumulator to memcpy src register (after computing aligned addr).
    fn emit_memcpy_store_src_from_acc(&mut self);

    /// Emit va_arg: extract next variadic argument from va_list and store to dest.
    fn emit_va_arg(&mut self, dest: &Value, va_list_ptr: &Value, result_ty: IrType);

    /// Emit va_start: initialize a va_list for variadic argument access.
    fn emit_va_start(&mut self, va_list_ptr: &Value);

    /// Emit va_end: clean up a va_list. No-op on all current targets (x86-64, AArch64, RISC-V).
    fn emit_va_end(&mut self, _va_list_ptr: &Value) {
        // va_end is a no-op on all supported architectures
    }

    /// Emit va_copy: copy src va_list to dest va_list.
    fn emit_va_copy(&mut self, dest_ptr: &Value, src_ptr: &Value);

    /// Emit an atomic read-modify-write operation.
    fn emit_atomic_rmw(&mut self, dest: &Value, op: AtomicRmwOp, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering);

    /// Emit an atomic compare-and-exchange operation.
    fn emit_atomic_cmpxchg(&mut self, dest: &Value, ptr: &Operand, expected: &Operand, desired: &Operand, ty: IrType, success_ordering: AtomicOrdering, failure_ordering: AtomicOrdering, returns_bool: bool);

    /// Emit an atomic load.
    fn emit_atomic_load(&mut self, dest: &Value, ptr: &Operand, ty: IrType, ordering: AtomicOrdering);

    /// Emit an atomic store.
    fn emit_atomic_store(&mut self, ptr: &Operand, val: &Operand, ty: IrType, ordering: AtomicOrdering);

    /// Emit a memory fence.
    fn emit_fence(&mut self, ordering: AtomicOrdering);

    /// Emit inline assembly.
    fn emit_inline_asm(&mut self, template: &str, outputs: &[(String, Value, Option<String>)], inputs: &[(String, Operand, Option<String>)], clobbers: &[String], operand_types: &[IrType]);

    /// Emit a return terminator. Default handles i128 pair returns, f128/f32/f64
    /// float-to-register moves, and integer returns using arch-specific primitives.
    fn emit_return(&mut self, val: Option<&Operand>, frame_size: i64) {
        if let Some(val) = val {
            let ret_ty = self.current_return_type();
            if is_i128_type(ret_ty) {
                self.emit_load_acc_pair(val);
                self.emit_return_i128_to_regs();
                self.emit_epilogue_and_ret(frame_size);
                return;
            }
            self.emit_load_operand(val);
            if ret_ty.is_long_double() {
                self.emit_return_f128_to_reg();
            } else if ret_ty == IrType::F32 {
                self.emit_return_f32_to_reg();
            } else if ret_ty.is_float() {
                self.emit_return_f64_to_reg();
            } else {
                self.emit_return_int_to_reg();
            }
        }
        self.emit_epilogue_and_ret(frame_size);
    }

    // ---- Architecture-specific instruction primitives ----
    // These small methods provide the building blocks for default implementations
    // of higher-level codegen methods, avoiding duplication across backends.

    // --- 128-bit (accumulator pair) primitives ---
    // Convention: each arch uses a register pair for 128-bit values:
    //   x86: rax:rdx, ARM: x0:x1, RISC-V: t0:t1

    /// Sign-extend the accumulator into the high half of the accumulator pair.
    /// x86: cqto (rax sign-extends into rdx), ARM: asr x1, x0, #63, RISC-V: srai t1, t0, 63
    fn emit_sign_extend_acc_high(&mut self);

    /// Zero the high half of the accumulator pair.
    /// x86: xorq %rdx, %rdx, ARM: mov x1, #0, RISC-V: li t1, 0
    fn emit_zero_acc_high(&mut self);

    /// Load an operand into the accumulator pair (rax:rdx / x0:x1 / t0:t1).
    fn emit_load_acc_pair(&mut self, op: &Operand);

    /// Store the accumulator pair to a value's 16-byte stack slot.
    fn emit_store_acc_pair(&mut self, dest: &Value);

    /// Store the accumulator pair directly to a stack slot (alloca case for i128 store).
    fn emit_store_pair_to_slot(&mut self, slot: StackSlot);

    /// Load the accumulator pair from a stack slot (alloca case for i128 load).
    fn emit_load_pair_from_slot(&mut self, slot: StackSlot);

    /// Save the accumulator pair to scratch regs before loading a pointer (i128 indirect store).
    fn emit_save_acc_pair(&mut self);

    /// Store the saved accumulator pair through the pointer now in the scratch/addr reg (i128 indirect store).
    fn emit_store_pair_indirect(&mut self);

    /// Load the accumulator pair through the pointer now in the scratch/addr reg (i128 indirect load).
    fn emit_load_pair_indirect(&mut self);

    /// Emit 128-bit negate on the accumulator pair.
    fn emit_i128_neg(&mut self);

    /// Emit 128-bit bitwise NOT on the accumulator pair.
    fn emit_i128_not(&mut self);

    // --- Return primitives ---
    // Used by the default emit_return implementation.

    /// Get the current function's return type (stored in backend struct).
    fn current_return_type(&self) -> IrType;

    /// Move the i128 accumulator pair into the ABI return registers.
    /// x86: noop (rax:rdx already correct), ARM: noop (x0:x1), RISC-V: mv a0,t0 / mv a1,t1
    fn emit_return_i128_to_regs(&mut self);

    /// Move the accumulator (containing an f128/long double f64 bit pattern) into
    /// the ABI f128 return register/format.
    /// x86: push rax, fldl, pop. ARM: fmov d0,x0 + bl __extenddftf2. RISC-V: fmv.d.x fa0,t0 + call __extenddftf2
    fn emit_return_f128_to_reg(&mut self);

    /// Move the accumulator (containing an f32 bit pattern) into the ABI float return register.
    /// x86: movd %eax,%xmm0. ARM: fmov s0,w0. RISC-V: fmv.w.x fa0,t0
    fn emit_return_f32_to_reg(&mut self);

    /// Move the accumulator (containing an f64 bit pattern) into the ABI float return register.
    /// x86: movq %rax,%xmm0. ARM: fmov d0,x0. RISC-V: fmv.d.x fa0,t0
    fn emit_return_f64_to_reg(&mut self);

    /// Move the accumulator into the ABI integer return register (if not already there).
    /// x86: noop (already rax). ARM: noop (already x0). RISC-V: mv a0,t0
    fn emit_return_int_to_reg(&mut self);

    /// Emit function epilogue and return instruction.
    fn emit_epilogue_and_ret(&mut self, frame_size: i64);

    // --- Typed store/load primitives ---
    // Used by emit_store/emit_load defaults for non-i128 paths.

    /// Return the store instruction mnemonic for a type (e.g., "movb"/"strb"/"sb").
    fn store_instr_for_type(&self, ty: IrType) -> &'static str;

    /// Return the load instruction mnemonic for a type (e.g., "movzbl"/"ldrb"/"lbu").
    fn load_instr_for_type(&self, ty: IrType) -> &'static str;

    /// Store the accumulator to a slot using a typed instruction (alloca direct store).
    fn emit_typed_store_to_slot(&mut self, instr: &'static str, ty: IrType, slot: StackSlot);

    /// Load from a slot into the accumulator using a typed instruction (alloca direct load).
    fn emit_typed_load_from_slot(&mut self, instr: &'static str, slot: StackSlot);

    /// Save the accumulator to a scratch register (for indirect store: save val, then load ptr).
    fn emit_save_acc(&mut self);

    /// Load a pointer value from a non-alloca slot into the address register.
    fn emit_load_ptr_from_slot(&mut self, slot: StackSlot);

    /// Store the saved accumulator through the address register (indirect typed store).
    fn emit_typed_store_indirect(&mut self, instr: &'static str, ty: IrType);

    /// Load through the address register into the accumulator (indirect typed load).
    fn emit_typed_load_indirect(&mut self, instr: &'static str);

    // --- GEP primitives ---

    /// Load a slot's effective address (alloca) or pointer value (non-alloca) into a secondary register.
    /// The secondary register is used to hold the base while the accumulator loads the offset.
    fn emit_slot_addr_to_secondary(&mut self, slot: StackSlot, is_alloca: bool);

    /// Add the secondary register to the accumulator (base + offset for GEP).
    fn emit_add_secondary_to_acc(&mut self);

    // --- Dynamic alloca primitives ---

    /// Round up the accumulator to 16-byte alignment: acc = (acc + 15) & -16.
    fn emit_round_up_acc_to_16(&mut self);

    /// Subtract the accumulator from the stack pointer.
    fn emit_sub_sp_by_acc(&mut self);

    /// Move the stack pointer value into the accumulator.
    fn emit_mov_sp_to_acc(&mut self);

    /// Align the accumulator to the given alignment: acc = (acc + align-1) & -align.
    fn emit_align_acc(&mut self, align: usize);

    // --- Memcpy primitives ---

    /// Load the dest address for memcpy into the arch-specific dest register.
    fn emit_memcpy_load_dest_addr(&mut self, slot: StackSlot, is_alloca: bool);

    /// Load the src address for memcpy into the arch-specific src register.
    fn emit_memcpy_load_src_addr(&mut self, slot: StackSlot, is_alloca: bool);

    /// Emit the actual copy loop/instruction for memcpy (after addresses are loaded).
    fn emit_memcpy_impl(&mut self, size: usize);

    // --- Unary operation primitives ---

    /// Emit float negation on the accumulator.
    fn emit_float_neg(&mut self, ty: IrType);

    /// Emit integer negation on the accumulator.
    fn emit_int_neg(&mut self, ty: IrType);

    /// Emit integer bitwise NOT on the accumulator.
    fn emit_int_not(&mut self, ty: IrType);

    /// Emit count-leading-zeros on the accumulator.
    fn emit_int_clz(&mut self, ty: IrType);

    /// Emit count-trailing-zeros on the accumulator.
    fn emit_int_ctz(&mut self, ty: IrType);

    /// Emit byte-swap on the accumulator.
    fn emit_int_bswap(&mut self, ty: IrType);

    /// Emit population count on the accumulator.
    fn emit_int_popcount(&mut self, ty: IrType);

    // --- Control flow primitives ---

    /// The unconditional jump mnemonic for this architecture.
    /// x86: "jmp", ARM: "b", RISC-V: "j"
    fn jump_mnemonic(&self) -> &'static str;

    /// The trap/unreachable instruction for this architecture.
    /// x86: "ud2", ARM: "brk #0", RISC-V: "ebreak"
    fn trap_instruction(&self) -> &'static str;

    /// Emit a branch-if-nonzero instruction: if the accumulator register is
    /// nonzero, branch to `label`. Does NOT emit the preceding operand load.
    /// x86: "testq %rax, %rax; jne label", ARM: "cbnz x0, label", RISC-V: "bnez t0, label"
    fn emit_branch_nonzero(&mut self, label: &str);

    /// Emit an indirect jump through the accumulator register.
    /// x86: "jmpq *%rax", ARM: "br x0", RISC-V: "jr t0"
    fn emit_jump_indirect(&mut self);

    // ---- Default implementations for simple terminators and branches ----
    // These use the primitives above to provide shared control flow logic.

    /// Emit an unconditional branch. Default uses jump_mnemonic().
    fn emit_branch(&mut self, label: &str) {
        let mnemonic = self.jump_mnemonic();
        self.state().emit(&format!("    {} {}", mnemonic, label));
    }

    /// Emit an unreachable trap instruction. Default uses trap_instruction().
    fn emit_unreachable(&mut self) {
        let trap = self.trap_instruction();
        self.state().emit(&format!("    {}", trap));
    }

    /// Emit a conditional branch: load cond, branch to true_label if nonzero,
    /// fall through to false_label. Default uses emit_load_operand + emit_branch_nonzero + emit_branch.
    fn emit_cond_branch(&mut self, cond: &Operand, true_label: &str, false_label: &str) {
        self.emit_load_operand(cond);
        self.emit_branch_nonzero(true_label);
        self.emit_branch(false_label);
    }

    /// Emit an indirect branch (computed goto: goto *addr).
    /// Default: load target into accumulator, emit indirect jump.
    fn emit_indirect_branch(&mut self, target: &Operand) {
        self.emit_load_operand(target);
        self.emit_jump_indirect();
    }

    /// Emit a label address load (GCC &&label extension for computed goto).
    /// Default delegates to emit_global_addr since on all current backends,
    /// loading a label address uses the same mechanism as loading a global symbol address.
    fn emit_label_addr(&mut self, dest: &Value, label: &str) {
        self.emit_global_addr(dest, label);
    }

    /// Emit code to capture the second F64 return value after a function call.
    /// On x86-64: read xmm1 into dest. On ARM64: read d1. On RISC-V: read fa1.
    fn emit_get_return_f64_second(&mut self, dest: &Value);

    /// Emit code to set the second F64 return value before a return.
    /// On x86-64: write xmm1. On ARM64: write d1. On RISC-V: write fa1.
    fn emit_set_return_f64_second(&mut self, src: &Operand);

    /// Emit code to capture the second F32 return value after a function call.
    /// Used for _Complex float returns on ARM64 (s1) and RISC-V (fa1 as float).
    /// x86-64 does not use this (packs both F32s into one xmm0).
    fn emit_get_return_f32_second(&mut self, dest: &Value);

    /// Emit code to set the second F32 return value before a return.
    /// Used for _Complex float returns on ARM64 (s1) and RISC-V (fa1 as float).
    fn emit_set_return_f32_second(&mut self, src: &Operand);

    /// Emit the function directive for the function type attribute.
    /// x86 uses "@function", ARM uses "%function".
    fn function_type_directive(&self) -> &'static str { "@function" }

    /// Emit dynamic stack allocation: subtract size from stack pointer,
    /// align the result, and store the pointer in dest.
    /// Default: load size, round up to 16, subtract from SP, optionally over-align.
    fn emit_dyn_alloca(&mut self, dest: &Value, size: &Operand, align: usize) {
        self.emit_load_operand(size);
        self.emit_round_up_acc_to_16();
        self.emit_sub_sp_by_acc();
        if align > 16 {
            self.emit_mov_sp_to_acc();
            self.emit_align_acc(align);
        } else {
            self.emit_mov_sp_to_acc();
        }
        self.emit_store_result(dest);
    }

    /// Emit a 128-bit value copy (src -> dest, both 16-byte stack slots).
    /// Default: truncates to 64-bit (used by ARM/RISC-V where i128 is not fully supported).
    /// x86-64 overrides this to use rax:rdx register pair.
    fn emit_copy_i128(&mut self, dest: &Value, src: &Operand) {
        self.emit_load_operand(src);
        self.emit_store_result(dest);
    }

    /// Frame size including alignment and saved registers.
    fn aligned_frame_size(&self, raw_space: i64) -> i64;

    /// Emit an X86 SSE operation. Default is no-op (non-x86 targets).
    fn emit_x86_sse_op(&mut self, _dest: &Option<Value>, _op: &X86SseOpKind, _dest_ptr: &Option<Value>, _args: &[Operand]) {}
}

/// Generate assembly for a module using the given architecture's codegen.
pub fn generate_module(cg: &mut dyn ArchCodegen, module: &IrModule) -> String {
    // Build the set of locally-defined symbols for PIC mode.
    // In PIC mode, only symbols with internal linkage (static) are guaranteed
    // not to be interposed. Non-static globals/functions need GOT/PLT because
    // they could be preempted by another shared library (ELF symbol preemption).
    // String literal labels (starting with .L) are always local.
    {
        let state = cg.state();
        for func in &module.functions {
            if func.is_static {
                state.local_symbols.insert(func.name.clone());
            }
        }
        for global in &module.globals {
            if global.is_static {
                state.local_symbols.insert(global.name.clone());
            }
        }
        // String/wide string literal labels are always local (compiler-generated .L labels)
        for (label, _) in &module.string_literals {
            state.local_symbols.insert(label.clone());
        }
        for (label, _) in &module.wide_string_literals {
            state.local_symbols.insert(label.clone());
        }
    }

    // Emit data sections
    let ptr_dir = cg.ptr_directive();
    common::emit_data_sections(&mut cg.state().out, module, ptr_dir);

    // Text section
    cg.state().emit(".section .text");
    for func in &module.functions {
        if !func.is_declaration {
            generate_function(cg, func);
        }
    }

    // Emit .init_array for constructor functions
    for ctor in &module.constructors {
        cg.state().emit("");
        cg.state().emit(".section .init_array,\"aw\",@init_array");
        cg.state().emit(".align 8");
        cg.state().emit(&format!("{} {}", ptr_dir.as_str(), ctor));
    }

    // Emit .fini_array for destructor functions
    for dtor in &module.destructors {
        cg.state().emit("");
        cg.state().emit(".section .fini_array,\"aw\",@fini_array");
        cg.state().emit(".align 8");
        cg.state().emit(&format!("{} {}", ptr_dir.as_str(), dtor));
    }

    // Emit .note.GNU-stack section to indicate non-executable stack
    cg.state().emit("");
    cg.state().emit(".section .note.GNU-stack,\"\",@progbits");

    std::mem::take(&mut cg.state().out.buf)
}

/// Generate code for a single function.
fn generate_function(cg: &mut dyn ArchCodegen, func: &IrFunction) {
    cg.state().reset_for_function();

    let type_dir = cg.function_type_directive();
    if !func.is_static {
        cg.state().emit(&format!(".globl {}", func.name));
    }
    cg.state().emit(&format!(".type {}, {}", func.name, type_dir));
    cg.state().emit(&format!("{}:", func.name));

    // Pre-scan for DynAlloca: if present, the epilogue must restore SP from
    // the frame pointer instead of adding back the compile-time frame size.
    let has_dyn_alloca = func.blocks.iter().any(|block| {
        block.instructions.iter().any(|inst| matches!(inst, Instruction::DynAlloca { .. }))
    });
    cg.state().has_dyn_alloca = has_dyn_alloca;

    // Calculate stack space and emit prologue
    let raw_space = cg.calculate_stack_space(func);
    let frame_size = cg.aligned_frame_size(raw_space);
    cg.emit_prologue(func, frame_size);

    // Store parameters
    cg.emit_store_params(func);

    // Generate basic blocks
    for block in &func.blocks {
        if block.label != "entry" {
            cg.state().emit(&format!("{}:", block.label));
        }
        for inst in &block.instructions {
            generate_instruction(cg, inst);
        }
        generate_terminator(cg, &block.terminator, frame_size);
    }

    cg.state().emit(&format!(".size {}, .-{}", func.name, func.name));
    cg.state().emit("");
}

/// Dispatch a single IR instruction to the appropriate arch method.
fn generate_instruction(cg: &mut dyn ArchCodegen, inst: &Instruction) {
    match inst {
        Instruction::Alloca { .. } => {
            // Space already allocated in prologue
        }
        Instruction::DynAlloca { dest, size, align } => {
            cg.emit_dyn_alloca(dest, size, *align);
        }
        Instruction::Store { val, ptr, ty } => {
            cg.emit_store(val, ptr, *ty);
        }
        Instruction::Load { dest, ptr, ty } => {
            cg.emit_load(dest, ptr, *ty);
        }
        Instruction::BinOp { dest, op, lhs, rhs, ty } => {
            cg.emit_binop(dest, *op, lhs, rhs, *ty);
        }
        Instruction::UnaryOp { dest, op, src, ty } => {
            cg.emit_unaryop(dest, *op, src, *ty);
        }
        Instruction::Cmp { dest, op, lhs, rhs, ty } => {
            cg.emit_cmp(dest, *op, lhs, rhs, *ty);
        }
        Instruction::Call { dest, func, args, arg_types, return_type, is_variadic, num_fixed_args } => {
            cg.emit_call(args, arg_types, Some(func), None, *dest, *return_type, *is_variadic, *num_fixed_args);
        }
        Instruction::CallIndirect { dest, func_ptr, args, arg_types, return_type, is_variadic, num_fixed_args } => {
            cg.emit_call(args, arg_types, None, Some(func_ptr), *dest, *return_type, *is_variadic, *num_fixed_args);
        }
        Instruction::GlobalAddr { dest, name } => {
            cg.emit_global_addr(dest, name);
        }
        Instruction::Copy { dest, src } => {
            // Check if this is a 128-bit value copy
            let is_i128_copy = match src {
                Operand::Value(v) => cg.state_ref().is_i128_value(v.0),
                Operand::Const(IrConst::I128(_)) => true,
                _ => false,
            };
            if is_i128_copy {
                // Mark dest as i128 too so subsequent copies from it are also 128-bit
                cg.state().i128_values.insert(dest.0);
                cg.emit_copy_i128(dest, src);
            } else {
                cg.emit_load_operand(src);
                cg.emit_store_result(dest);
            }
        }
        Instruction::Cast { dest, src, from_ty, to_ty } => {
            cg.emit_cast(dest, src, *from_ty, *to_ty);
        }
        Instruction::GetElementPtr { dest, base, offset, .. } => {
            cg.emit_gep(dest, base, offset);
        }
        Instruction::Memcpy { dest, src, size } => {
            cg.emit_memcpy(dest, src, *size);
        }
        Instruction::VaArg { dest, va_list_ptr, result_ty } => {
            cg.emit_va_arg(dest, va_list_ptr, *result_ty);
        }
        Instruction::VaStart { va_list_ptr } => {
            cg.emit_va_start(va_list_ptr);
        }
        Instruction::VaEnd { va_list_ptr } => {
            cg.emit_va_end(va_list_ptr);
        }
        Instruction::VaCopy { dest_ptr, src_ptr } => {
            cg.emit_va_copy(dest_ptr, src_ptr);
        }
        Instruction::AtomicRmw { dest, op, ptr, val, ty, ordering } => {
            cg.emit_atomic_rmw(dest, *op, ptr, val, *ty, *ordering);
        }
        Instruction::AtomicCmpxchg { dest, ptr, expected, desired, ty, success_ordering, failure_ordering, returns_bool } => {
            cg.emit_atomic_cmpxchg(dest, ptr, expected, desired, *ty, *success_ordering, *failure_ordering, *returns_bool);
        }
        Instruction::AtomicLoad { dest, ptr, ty, ordering } => {
            cg.emit_atomic_load(dest, ptr, *ty, *ordering);
        }
        Instruction::AtomicStore { ptr, val, ty, ordering } => {
            cg.emit_atomic_store(ptr, val, *ty, *ordering);
        }
        Instruction::Fence { ordering } => {
            cg.emit_fence(*ordering);
        }
        Instruction::Phi { .. } => {
            // Phi nodes are resolved before codegen by lowering them to copies
            // at the end of predecessor blocks. This case should not be reached
            // in normal operation, but is a no-op for safety.
        }
        Instruction::LabelAddr { dest, label } => {
            cg.emit_label_addr(dest, label);
        }
        Instruction::GetReturnF64Second { dest } => {
            cg.emit_get_return_f64_second(dest);
        }
        Instruction::SetReturnF64Second { src } => {
            cg.emit_set_return_f64_second(src);
        }
        Instruction::GetReturnF32Second { dest } => {
            cg.emit_get_return_f32_second(dest);
        }
        Instruction::SetReturnF32Second { src } => {
            cg.emit_set_return_f32_second(src);
        }
        Instruction::InlineAsm { template, outputs, inputs, clobbers, operand_types } => {
            cg.emit_inline_asm(template, outputs, inputs, clobbers, operand_types);
        }
        Instruction::X86SseOp { dest, op, dest_ptr, args } => {
            cg.emit_x86_sse_op(dest, op, dest_ptr, args);
        }
    }
}

/// Dispatch a terminator to the appropriate arch method.
fn generate_terminator(cg: &mut dyn ArchCodegen, term: &Terminator, frame_size: i64) {
    match term {
        Terminator::Return(val) => {
            cg.emit_return(val.as_ref(), frame_size);
        }
        Terminator::Branch(label) => {
            cg.emit_branch(label);
        }
        Terminator::CondBranch { cond, true_label, false_label } => {
            cg.emit_cond_branch(cond, true_label, false_label);
        }
        Terminator::IndirectBranch { target, .. } => {
            cg.emit_indirect_branch(target);
        }
        Terminator::Unreachable => {
            cg.emit_unreachable();
        }
    }
}

/// Check if an IR type is a 128-bit integer type (I128 or U128).
/// Used by all backends to detect 128-bit operands.
pub fn is_i128_type(ty: IrType) -> bool {
    matches!(ty, IrType::I128 | IrType::U128)
}

/// Shared stack space calculation: iterates over all instructions, assigns stack
/// slots for allocas and value results. Arch-specific offset direction is handled
/// by the `assign_slot` closure.
///
/// `initial_offset`: starting offset (e.g., 0 for x86, 16 for ARM/RISC-V to skip saved regs)
/// `assign_slot`: maps (current_space, raw_alloca_size) -> (slot_offset, new_space)
///   - raw_alloca_size is the alloca byte count (0 means pointer-sized, i.e., 8)
///   - For non-alloca values, raw_alloca_size is always 8
///   - The closure is responsible for alignment
pub fn calculate_stack_space_common(
    state: &mut CodegenState,
    func: &IrFunction,
    initial_offset: i64,
    assign_slot: impl Fn(i64, i64, i64) -> (i64, i64),
) -> i64 {
    let mut space = initial_offset;
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Instruction::Alloca { dest, size, ty, align } = inst {
                let effective_align = *align;
                // For over-aligned allocas (> 16), over-allocate by align-1 bytes
                // so runtime alignment can find a properly aligned address within the slot.
                let extra = if effective_align > 16 { effective_align - 1 } else { 0 };
                let raw_size = if *size == 0 { 8 } else { *size as i64 };
                let (slot, new_space) = assign_slot(space, raw_size + extra as i64, *align as i64);
                state.value_locations.insert(dest.0, StackSlot(slot));
                state.alloca_values.insert(dest.0);
                state.alloca_types.insert(dest.0, *ty);
                if effective_align > 16 {
                    state.alloca_alignments.insert(dest.0, effective_align);
                }
                space = new_space;
            } else if let Some(dest) = inst.dest() {
                // Use 16-byte slots for I128/U128 result types, 8 bytes for everything else
                let is_i128 = matches!(inst.result_type(), Some(IrType::I128) | Some(IrType::U128));
                let slot_size = if is_i128 { 16 } else { 8 };
                let (slot, new_space) = assign_slot(space, slot_size, 0);
                state.value_locations.insert(dest.0, StackSlot(slot));
                if is_i128 {
                    state.i128_values.insert(dest.0);
                }
                space = new_space;
            }
        }
    }
    space
}

// ============================================================================
// Shared cast classification
// ============================================================================

/// Classification of type casts. All three backends use the same control flow
/// to decide which kind of cast to emit; only the actual instructions differ.
/// By classifying the cast once in shared code, we eliminate duplicated
/// Ptr-normalization and F128-reduction logic from each backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// No conversion needed (same type, or Ptr <-> I64/U64, or F128 <-> F64).
    Noop,
    /// Float to signed integer (from_ty is F32 or F64).
    FloatToSigned { from_f64: bool },
    /// Float to unsigned integer (from_ty is F32 or F64).
    FloatToUnsigned { from_f64: bool, to_u64: bool },
    /// Signed integer to float (to_ty is F32 or F64).
    SignedToFloat { to_f64: bool },
    /// Unsigned integer to float. `from_u64` indicates U64 source needing
    /// special overflow handling on x86.
    UnsignedToFloat { to_f64: bool, from_u64: bool },
    /// Float-to-float conversion (F32 <-> F64).
    FloatToFloat { widen: bool },
    /// Integer widening: sign- or zero-extend a smaller type to a larger one.
    IntWiden { from_ty: IrType, to_ty: IrType },
    /// Integer narrowing: truncate a larger type to a smaller one.
    IntNarrow { to_ty: IrType },
    /// Same-size signed-to-unsigned (need to mask/clear upper bits).
    SignedToUnsignedSameSize { to_ty: IrType },
}

/// Classify a cast between two IR types. This captures the shared decision logic
/// that all three backends use identically. Backends then match on the returned
/// `CastKind` to emit architecture-specific instructions.
///
/// Handles Ptr normalization (Ptr treated as U64) and F128 reduction (F128 treated
/// as F64 for computation purposes) before classification.
pub fn classify_cast(from_ty: IrType, to_ty: IrType) -> CastKind {
    if from_ty == to_ty {
        return CastKind::Noop;
    }

    // F128 (long double) is computed as F64. Treat F128 <-> F64 as noop,
    // and F128 <-> other as F64 <-> other.
    if from_ty == IrType::F128 || to_ty == IrType::F128 {
        let effective_from = if from_ty == IrType::F128 { IrType::F64 } else { from_ty };
        let effective_to = if to_ty == IrType::F128 { IrType::F64 } else { to_ty };
        if effective_from == effective_to {
            return CastKind::Noop;
        }
        return classify_cast(effective_from, effective_to);
    }

    // Ptr is equivalent to U64 on all 64-bit targets.
    if (from_ty == IrType::Ptr || to_ty == IrType::Ptr) && !from_ty.is_float() && !to_ty.is_float() {
        let effective_from = if from_ty == IrType::Ptr { IrType::U64 } else { from_ty };
        let effective_to = if to_ty == IrType::Ptr { IrType::U64 } else { to_ty };
        if effective_from == effective_to || (effective_from.size() == 8 && effective_to.size() == 8) {
            return CastKind::Noop;
        }
        return classify_cast(effective_from, effective_to);
    }

    // Float-to-int
    if from_ty.is_float() && !to_ty.is_float() {
        let is_unsigned_dest = to_ty.is_unsigned() || to_ty == IrType::Ptr;
        let from_f64 = from_ty == IrType::F64;
        if is_unsigned_dest {
            let to_u64 = to_ty == IrType::U64 || to_ty == IrType::Ptr;
            return CastKind::FloatToUnsigned { from_f64, to_u64 };
        } else {
            return CastKind::FloatToSigned { from_f64 };
        }
    }

    // Int-to-float
    if !from_ty.is_float() && to_ty.is_float() {
        let is_unsigned_src = from_ty.is_unsigned();
        let to_f64 = to_ty == IrType::F64;
        if is_unsigned_src {
            let from_u64 = from_ty == IrType::U64;
            return CastKind::UnsignedToFloat { to_f64, from_u64 };
        } else {
            return CastKind::SignedToFloat { to_f64 };
        }
    }

    // Float-to-float
    if from_ty.is_float() && to_ty.is_float() {
        let widen = from_ty == IrType::F32 && to_ty == IrType::F64;
        return CastKind::FloatToFloat { widen };
    }

    // Integer-to-integer
    let from_size = from_ty.size();
    let to_size = to_ty.size();

    if from_size == to_size {
        // Same size, different signedness
        if from_ty.is_signed() && to_ty.is_unsigned() {
            return CastKind::SignedToUnsignedSameSize { to_ty };
        }
        return CastKind::Noop;
    }

    if to_size > from_size {
        return CastKind::IntWiden { from_ty, to_ty };
    }

    CastKind::IntNarrow { to_ty }
}

// ============================================================================
// Shared float operation helpers
// ============================================================================

/// Classify a binary operation on floats. Returns None if the operation is not
/// meaningful on floats (e.g., bitwise And, Or, Xor, shifts, integer remainder).
/// This prevents the latent bug where all three backends silently emit `fadd`
/// for unsupported float operations.
pub fn classify_float_binop(op: IrBinOp) -> Option<FloatOp> {
    match op {
        IrBinOp::Add => Some(FloatOp::Add),
        IrBinOp::Sub => Some(FloatOp::Sub),
        IrBinOp::Mul => Some(FloatOp::Mul),
        IrBinOp::SDiv | IrBinOp::UDiv => Some(FloatOp::Div),
        // Bitwise ops, shifts, and remainder are not meaningful for floats.
        // Falling through to Add was a latent bug in all three backends.
        _ => None,
    }
}

/// Float arithmetic operations that all three backends support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Find the nth alloca instruction in the entry block (used for parameter storage).
pub fn find_param_alloca(func: &IrFunction, param_idx: usize) -> Option<(Value, IrType)> {
    func.blocks.first().and_then(|block| {
        block.instructions.iter()
            .filter(|i| matches!(i, Instruction::Alloca { .. }))
            .nth(param_idx)
            .and_then(|inst| {
                if let Instruction::Alloca { dest, ty, .. } = inst {
                    Some((*dest, *ty))
                } else {
                    None
                }
            })
    })
}

// ============================================================================
// Shared call argument classification
// ============================================================================

/// Classification of a function call argument for register/stack assignment.
/// All three backends use the same algorithmic structure to classify arguments;
/// only the register counts and F128 handling differ. This shared classification
/// eliminates the duplicated arg-walking logic from each backend's `emit_call`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallArgClass {
    /// Integer/pointer argument in a GP register. `reg_idx` is the GP register index.
    IntReg { reg_idx: usize },
    /// Float argument in an FP register. `reg_idx` is the FP register index.
    FloatReg { reg_idx: usize },
    /// 128-bit integer in a GP register pair. `base_reg_idx` is the first register.
    I128RegPair { base_reg_idx: usize },
    /// F128 (long double) — handling is arch-specific (x87 on x86, Q-reg on ARM, GP pair on RISC-V).
    /// `reg_or_stack` indicates whether it goes in registers or overflows to stack.
    F128Reg { reg_idx: usize },
    /// Argument overflows to the stack (normal 8-byte).
    Stack,
    /// F128 argument overflows to the stack (16-byte aligned).
    F128Stack,
    /// I128 argument overflows to the stack (16-byte aligned).
    I128Stack,
}

/// ABI configuration for call argument classification.
pub struct CallAbiConfig {
    /// Maximum GP registers for arguments (x86: 6, ARM/RISC-V: 8).
    pub max_int_regs: usize,
    /// Maximum FP registers for arguments (all: 8).
    pub max_float_regs: usize,
    /// Whether i128 register pairs must be even-aligned (ARM/RISC-V: true, x86: false).
    pub align_i128_pairs: bool,
    /// Whether F128 uses FP registers (ARM: true) or always goes to stack/x87 (x86: true = stack).
    /// On RISC-V, F128 goes in GP register pairs like i128.
    pub f128_in_fp_regs: bool,
    /// Whether F128 uses GP register pairs (RISC-V: true).
    pub f128_in_gp_pairs: bool,
    /// Whether variadic float args must go in GP registers instead of FP regs (RISC-V: true, x86: false, ARM: false).
    pub variadic_floats_in_gp: bool,
}

/// Classify all arguments for a function call, returning a `CallArgClass` per argument.
/// This captures the shared classification logic used identically by all three backends.
pub fn classify_call_args(
    args: &[Operand],
    arg_types: &[IrType],
    is_variadic: bool,
    config: &CallAbiConfig,
) -> Vec<CallArgClass> {
    let mut result = Vec::with_capacity(args.len());
    let mut int_idx = 0usize;
    let mut float_idx = 0usize;

    for (i, arg) in args.iter().enumerate() {
        let arg_ty = if i < arg_types.len() { Some(arg_types[i]) } else { None };
        let is_long_double = arg_ty.map(|t| t.is_long_double()).unwrap_or(false);
        let is_i128 = arg_ty.map(|t| is_i128_type(t)).unwrap_or(false);
        let is_float = if let Some(ty) = arg_ty {
            ty.is_float()
        } else {
            matches!(arg, Operand::Const(IrConst::F32(_) | IrConst::F64(_)))
        };
        // Variadic floats go through GP registers on RISC-V
        let force_gp = is_variadic && config.variadic_floats_in_gp && is_float && !is_long_double;

        if is_i128 {
            if config.align_i128_pairs && int_idx % 2 != 0 {
                int_idx += 1;
            }
            if int_idx + 1 < config.max_int_regs {
                result.push(CallArgClass::I128RegPair { base_reg_idx: int_idx });
                int_idx += 2;
            } else {
                result.push(CallArgClass::I128Stack);
                int_idx = config.max_int_regs;
            }
        } else if is_long_double {
            if config.f128_in_fp_regs {
                // ARM: F128 goes in Q registers (FP)
                if float_idx < config.max_float_regs {
                    result.push(CallArgClass::F128Reg { reg_idx: float_idx });
                    float_idx += 1;
                } else {
                    result.push(CallArgClass::F128Stack);
                }
            } else if config.f128_in_gp_pairs {
                // RISC-V: F128 goes in GP register pairs
                if config.align_i128_pairs && int_idx % 2 != 0 {
                    int_idx += 1;
                }
                if int_idx + 1 < config.max_int_regs {
                    result.push(CallArgClass::F128Reg { reg_idx: int_idx });
                    int_idx += 2;
                } else {
                    result.push(CallArgClass::F128Stack);
                    int_idx = config.max_int_regs;
                }
            } else {
                // x86: F128 always goes to stack (x87)
                result.push(CallArgClass::F128Stack);
            }
        } else if is_float && !force_gp && float_idx < config.max_float_regs {
            result.push(CallArgClass::FloatReg { reg_idx: float_idx });
            float_idx += 1;
        } else if int_idx < config.max_int_regs {
            result.push(CallArgClass::IntReg { reg_idx: int_idx });
            int_idx += 1;
        } else {
            result.push(CallArgClass::Stack);
        }
    }

    result
}

// ============================================================================
// Shared inline assembly framework
// ============================================================================

/// Operand classification for inline asm. Each backend classifies its constraints
/// into these categories so the shared framework can orchestrate register
/// assignment, tied operand resolution, and GCC numbering.
#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperandKind {
    /// General-purpose register (e.g., x86 "r", ARM "r", RISC-V "r").
    GpReg,
    /// Floating-point register (RISC-V "f").
    FpReg,
    /// Memory operand (all arches "m").
    Memory,
    /// Specific named register (x86 "a"→"rax", RISC-V "a0", etc.).
    Specific(String),
    /// Tied to another operand by index (e.g., "0", "1").
    Tied(usize),
    /// Immediate value (RISC-V "I", "i", "n").
    Immediate,
    /// Address for atomic ops (RISC-V "A").
    Address,
    /// Zero-or-register (RISC-V "rJ", "J").
    ZeroOrReg,
}

/// Per-operand state tracked by the shared inline asm framework.
/// Backends populate arch-specific fields (mem_addr, mem_offset, imm_value)
/// during constraint classification.
#[derive(Debug, Clone)]
pub struct AsmOperand {
    pub kind: AsmOperandKind,
    pub reg: String,
    pub name: Option<String>,
    /// x86: memory address string like "offset(%rbp)".
    pub mem_addr: String,
    /// RISC-V/ARM: stack offset for memory/address operands.
    pub mem_offset: i64,
    /// Immediate value for "I"/"i" constraints.
    pub imm_value: Option<i64>,
    /// IR type of this operand, used for correctly-sized loads/stores.
    pub operand_type: IrType,
}

impl AsmOperand {
    pub fn new(kind: AsmOperandKind, name: Option<String>) -> Self {
        Self { kind, reg: String::new(), name, mem_addr: String::new(), mem_offset: 0, imm_value: None, operand_type: IrType::I64 }
    }
}

/// Trait that backends implement to provide architecture-specific inline asm behavior.
/// The shared `emit_inline_asm_common` function calls these methods to handle the
/// architecture-dependent parts of inline assembly processing.
pub trait InlineAsmEmitter {
    /// Mutable access to the codegen state (for emitting instructions).
    fn asm_state(&mut self) -> &mut CodegenState;
    /// Immutable access to the codegen state.
    fn asm_state_ref(&self) -> &CodegenState;

    /// Classify a constraint string into an AsmOperandKind, and optionally
    /// return the specific register name for Specific constraints.
    fn classify_constraint(&self, constraint: &str) -> AsmOperandKind;

    /// Set up arch-specific operand metadata after classification.
    /// Called once per operand. For memory/address operands, set mem_addr or mem_offset.
    /// For immediate operands, set imm_value. The `val` is the IR value associated
    /// with this operand (output pointer for outputs, input value for inputs).
    fn setup_operand_metadata(&self, op: &mut AsmOperand, val: &Operand, is_output: bool);

    /// Assign the next available scratch register for the given operand kind.
    /// Returns the register name. Called during Phase 1 for operands without
    /// specific register assignments.
    fn assign_scratch_reg(&mut self, kind: &AsmOperandKind) -> String;

    /// Load an input value into its assigned register. Called during Phase 2.
    fn load_input_to_reg(&mut self, op: &AsmOperand, val: &Operand, constraint: &str);

    /// Pre-load a read-write ("+") output's current value into its register.
    fn preload_readwrite_output(&mut self, op: &AsmOperand, ptr: &Value);

    /// Substitute operand references in a single template line and return the result.
    fn substitute_template_line(&self, line: &str, operands: &[AsmOperand], gcc_to_internal: &[usize], operand_types: &[IrType]) -> String;

    /// Store an output register value back to its stack slot after the asm executes.
    fn store_output_from_reg(&mut self, op: &AsmOperand, ptr: &Value, constraint: &str);

    /// Resolve memory operand addresses that require indirection (non-alloca pointers).
    /// Called during Phase 2 for Memory operands whose pointer values need to be loaded
    /// into a register for indirect addressing. Returns true if the operand was updated.
    fn resolve_memory_operand(&mut self, _op: &mut AsmOperand, _val: &Operand) -> bool {
        false // default: no resolution needed
    }

    /// Reset scratch register allocation state (called at start of each inline asm).
    fn reset_scratch_state(&mut self);
}

/// Shared inline assembly emission logic. All three backends call this from their
/// `emit_inline_asm` implementation, providing an `InlineAsmEmitter` to handle
/// arch-specific details.
///
/// The 4-phase structure is:
/// 1. Classify constraints, assign registers (specific first, then scratch), resolve ties
/// 2. Load input values into registers, pre-load read-write outputs
/// 3. Substitute operand references in template and emit
/// 4. Store output registers back to stack slots
pub fn emit_inline_asm_common(
    emitter: &mut dyn InlineAsmEmitter,
    template: &str,
    outputs: &[(String, Value, Option<String>)],
    inputs: &[(String, Operand, Option<String>)],
    operand_types: &[IrType],
) {
    emitter.reset_scratch_state();
    let total_operands = outputs.len() + inputs.len();

    // Phase 1: Classify all operands and assign registers
    let mut operands: Vec<AsmOperand> = Vec::with_capacity(total_operands);

    // Classify outputs
    for (constraint, ptr, name) in outputs {
        let kind = emitter.classify_constraint(constraint);
        let mut op = AsmOperand::new(kind, name.clone());
        // Set up specific register if classified as Specific
        if let AsmOperandKind::Specific(ref reg) = op.kind {
            op.reg = reg.clone();
        }
        let val = Operand::Value(*ptr);
        emitter.setup_operand_metadata(&mut op, &val, true);
        operands.push(op);
    }

    // Track which inputs are tied (to avoid assigning scratch regs)
    let mut input_tied_to: Vec<Option<usize>> = Vec::with_capacity(inputs.len());

    // Classify inputs
    for (constraint, val, name) in inputs {
        let kind = emitter.classify_constraint(constraint);
        let mut op = AsmOperand::new(kind.clone(), name.clone());
        if let AsmOperandKind::Specific(ref reg) = op.kind {
            op.reg = reg.clone();
        }
        if let AsmOperandKind::Tied(idx) = &kind {
            input_tied_to.push(Some(*idx));
        } else {
            input_tied_to.push(None);
        }
        emitter.setup_operand_metadata(&mut op, val, false);
        operands.push(op);
    }

    // Assign scratch registers to operands that need them (not specific, not memory,
    // not immediate, not tied)
    for i in 0..total_operands {
        if !operands[i].reg.is_empty() {
            continue;
        }
        match &operands[i].kind {
            AsmOperandKind::Memory | AsmOperandKind::Immediate => continue,
            AsmOperandKind::Tied(_) => continue,
            kind => {
                // Check if this is a tied input
                let is_tied = if i >= outputs.len() {
                    input_tied_to[i - outputs.len()].is_some()
                } else {
                    false
                };
                if !is_tied {
                    operands[i].reg = emitter.assign_scratch_reg(kind);
                }
            }
        }
    }

    // Resolve tied operands: copy register and metadata from the target operand
    for i in 0..total_operands {
        if let AsmOperandKind::Tied(tied_to) = operands[i].kind.clone() {
            if tied_to < operands.len() {
                operands[i].reg = operands[tied_to].reg.clone();
                operands[i].mem_addr = operands[tied_to].mem_addr.clone();
                operands[i].mem_offset = operands[tied_to].mem_offset;
                if matches!(operands[tied_to].kind, AsmOperandKind::Memory) {
                    operands[i].kind = AsmOperandKind::Memory;
                } else if matches!(operands[tied_to].kind, AsmOperandKind::Address) {
                    operands[i].kind = AsmOperandKind::Address;
                }
            }
        }
    }
    // Also check via input_tied_to for x86-style tied operands (not classified as Tied kind)
    for (i, tied_to) in input_tied_to.iter().enumerate() {
        if let Some(tied_to) = tied_to {
            let op_idx = outputs.len() + i;
            if *tied_to < operands.len() && operands[op_idx].reg.is_empty() {
                operands[op_idx].reg = operands[*tied_to].reg.clone();
                operands[op_idx].mem_addr = operands[*tied_to].mem_addr.clone();
                operands[op_idx].mem_offset = operands[*tied_to].mem_offset;
                if matches!(operands[*tied_to].kind, AsmOperandKind::Memory) {
                    operands[op_idx].kind = AsmOperandKind::Memory;
                }
            }
        }
    }

    // Populate operand types from the operand_types array (outputs first, then inputs)
    for (i, ty) in operand_types.iter().enumerate() {
        if i < operands.len() {
            operands[i].operand_type = *ty;
        }
    }

    // Resolve memory operand addresses that require indirection (non-alloca pointers).
    // This must happen BEFORE "+" propagation so synthetic inputs inherit resolved addresses.
    for (i, (_, ptr, _)) in outputs.iter().enumerate() {
        if matches!(operands[i].kind, AsmOperandKind::Memory) {
            let val = Operand::Value(*ptr);
            emitter.resolve_memory_operand(&mut operands[i], &val);
        }
    }

    // Handle "+" read-write constraints: synthetic inputs share the output's register.
    // The IR lowering adds synthetic inputs at the BEGINNING of the inputs list
    // (one per "+" output, in order).
    let mut plus_idx = 0;
    for (i, (constraint, _, _)) in outputs.iter().enumerate() {
        if constraint.contains('+') {
            let plus_input_idx = outputs.len() + plus_idx;
            if plus_input_idx < total_operands {
                operands[plus_input_idx].reg = operands[i].reg.clone();
                operands[plus_input_idx].kind = operands[i].kind.clone();
                operands[plus_input_idx].mem_addr = operands[i].mem_addr.clone();
                operands[plus_input_idx].mem_offset = operands[i].mem_offset;
                operands[plus_input_idx].operand_type = operands[i].operand_type;
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

    // Resolve memory operand addresses for non-synthetic input operands
    {
        let num_plus = outputs.iter().filter(|(c,_,_)| c.contains('+')).count();
        for (i, (_, val, _)) in inputs.iter().enumerate() {
            // Skip synthetic "+" inputs (they already inherited from outputs)
            if i < num_plus { continue; }
            let op_idx = outputs.len() + i;
            if matches!(operands[op_idx].kind, AsmOperandKind::Memory) {
                emitter.resolve_memory_operand(&mut operands[op_idx], val);
            }
        }
    }

    // Phase 2: Load input values into their assigned registers
    for (i, (constraint, val, _)) in inputs.iter().enumerate() {
        let op_idx = outputs.len() + i;
        match &operands[op_idx].kind {
            AsmOperandKind::Memory | AsmOperandKind::Immediate => continue,
            _ => {}
        }
        if operands[op_idx].reg.is_empty() {
            continue;
        }
        emitter.load_input_to_reg(&operands[op_idx], val, constraint);
    }

    // Pre-load read-write output values
    for (i, (constraint, ptr, _)) in outputs.iter().enumerate() {
        if constraint.contains('+') {
            if !matches!(operands[i].kind, AsmOperandKind::Memory) {
                emitter.preload_readwrite_output(&operands[i], ptr);
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
        let resolved = emitter.substitute_template_line(line, &operands, &gcc_to_internal, operand_types);
        emitter.asm_state().emit(&format!("    {}", resolved));
    }

    // Phase 4: Store output register values back to their stack slots
    for (i, (constraint, ptr, _)) in outputs.iter().enumerate() {
        if constraint.contains('=') || constraint.contains('+') {
            emitter.store_output_from_reg(&operands[i], ptr, constraint);
        }
    }
}
