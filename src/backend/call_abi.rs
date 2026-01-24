//! Shared call ABI classification and stack space computation.
//!
//! All three backends use the same algorithmic structure to classify function call
//! arguments into register/stack classes. Only the register counts and F128 handling
//! differ per architecture. This module extracts that shared classification so each
//! backend's `emit_call` only needs to handle the instruction emission.

use crate::ir::ir::{IrConst, Operand};
use crate::common::types::IrType;
use super::generation::is_i128_type;

/// Classification of a function call argument for register/stack assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallArgClass {
    /// Integer/pointer argument in a GP register. `reg_idx` is the GP register index.
    IntReg { reg_idx: usize },
    /// Float argument in an FP register. `reg_idx` is the FP register index.
    FloatReg { reg_idx: usize },
    /// 128-bit integer in a GP register pair. `base_reg_idx` is the first register.
    I128RegPair { base_reg_idx: usize },
    /// F128 (long double) — handling is arch-specific (x87 on x86, Q-reg on ARM, GP pair on RISC-V).
    F128Reg { reg_idx: usize },
    /// Small struct (<=16 bytes) passed by value in 1-2 GP registers.
    StructByValReg { base_reg_idx: usize, size: usize },
    /// Small struct (<=16 bytes) that overflows to the stack.
    StructByValStack { size: usize },
    /// Large struct (>16 bytes) passed on the stack (MEMORY class).
    LargeStructStack { size: usize },
    /// Argument overflows to the stack (normal 8-byte).
    Stack,
    /// F128 argument overflows to the stack (16-byte aligned).
    F128Stack,
    /// I128 argument overflows to the stack (16-byte aligned).
    I128Stack,
}

impl CallArgClass {
    /// Returns true if this argument is passed on the stack (any kind).
    pub fn is_stack(&self) -> bool {
        matches!(self, CallArgClass::Stack | CallArgClass::F128Stack |
                 CallArgClass::I128Stack | CallArgClass::StructByValStack { .. } |
                 CallArgClass::LargeStructStack { .. })
    }

    /// Returns the stack space consumed by this argument (0 if register).
    pub fn stack_bytes(&self) -> usize {
        match self {
            CallArgClass::F128Stack | CallArgClass::I128Stack => 16,
            CallArgClass::StructByValStack { size } | CallArgClass::LargeStructStack { size } => {
                (*size + 7) & !7 // round up to 8-byte alignment
            }
            CallArgClass::Stack => 8,
            _ => 0,
        }
    }
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
/// `struct_arg_sizes` indicates which args are struct/union by-value: Some(size) for struct
/// args, None otherwise.
pub fn classify_call_args(
    args: &[Operand],
    arg_types: &[IrType],
    struct_arg_sizes: &[Option<usize>],
    is_variadic: bool,
    config: &CallAbiConfig,
) -> Vec<CallArgClass> {
    let mut result = Vec::with_capacity(args.len());
    let mut int_idx = 0usize;
    let mut float_idx = 0usize;

    for (i, _arg) in args.iter().enumerate() {
        let struct_size = struct_arg_sizes.get(i).copied().flatten();
        let arg_ty = if i < arg_types.len() { Some(arg_types[i]) } else { None };
        let is_long_double = arg_ty.map(|t| t.is_long_double()).unwrap_or(false);
        let is_i128 = arg_ty.map(|t| is_i128_type(t)).unwrap_or(false);
        let is_float = if let Some(ty) = arg_ty {
            ty.is_float()
        } else {
            matches!(args[i], Operand::Const(IrConst::F32(_) | IrConst::F64(_)))
        };
        let force_gp = is_variadic && config.variadic_floats_in_gp && is_float && !is_long_double;

        if let Some(size) = struct_size {
            if size <= 16 {
                let regs_needed = if size <= 8 { 1 } else { 2 };
                if int_idx + regs_needed <= config.max_int_regs {
                    result.push(CallArgClass::StructByValReg { base_reg_idx: int_idx, size });
                    int_idx += regs_needed;
                } else {
                    result.push(CallArgClass::StructByValStack { size });
                    int_idx = config.max_int_regs;
                }
            } else {
                result.push(CallArgClass::LargeStructStack { size });
            }
        } else if is_i128 {
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
                if float_idx < config.max_float_regs {
                    result.push(CallArgClass::F128Reg { reg_idx: float_idx });
                    float_idx += 1;
                } else {
                    result.push(CallArgClass::F128Stack);
                }
            } else if config.f128_in_gp_pairs {
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
                result.push(CallArgClass::F128Stack);
            }
        } else if is_float && !force_gp && float_idx < config.max_float_regs {
            result.push(CallArgClass::FloatReg { reg_idx: float_idx });
            float_idx += 1;
        } else if is_float && !force_gp {
            result.push(CallArgClass::Stack);
        } else if int_idx < config.max_int_regs {
            result.push(CallArgClass::IntReg { reg_idx: int_idx });
            int_idx += 1;
        } else {
            result.push(CallArgClass::Stack);
        }
    }

    result
}

/// Compute the total stack space needed for stack-overflow arguments.
/// Returns the total bytes needed, 16-byte aligned.
/// Use this for ARM and RISC-V which pre-allocate stack space with a single SP adjustment.
pub fn compute_stack_arg_space(arg_classes: &[CallArgClass]) -> usize {
    let mut total: usize = 0;
    for cls in arg_classes {
        if !cls.is_stack() { continue; }
        if matches!(cls, CallArgClass::F128Stack | CallArgClass::I128Stack) {
            total = (total + 15) & !15;
        }
        total += cls.stack_bytes();
    }
    (total + 15) & !15
}

/// Compute the raw bytes that will be pushed onto the stack for stack arguments.
/// Unlike `compute_stack_arg_space`, this does NOT apply final 16-byte alignment,
/// because x86 uses individual `pushq` instructions and handles alignment separately.
pub fn compute_stack_push_bytes(arg_classes: &[CallArgClass]) -> usize {
    let mut total: usize = 0;
    for cls in arg_classes {
        if !cls.is_stack() { continue; }
        total += cls.stack_bytes();
    }
    total
}
