//! Shared callee-side parameter classification for `emit_store_params`.
//!
//! The caller side already has `classify_call_args` in `call_abi.rs`. This module
//! provides the analogous callee-side classification so that each backend's
//! `emit_store_params` only handles the arch-specific register store instructions,
//! not the duplicated ABI classification logic.
//!
//! The `classify_params` function walks a function's parameter list and assigns each
//! parameter to a `ParamClass` (GP register, FP register, i128 pair, struct-by-value,
//! F128, or stack-passed), using the same `CallAbiConfig` that drives `classify_call_args`.

use crate::ir::ir::IrFunction;
use crate::common::types::IrType;
use super::call_abi::CallAbiConfig;
use super::generation::is_i128_type;

/// Classification of a function parameter for `emit_store_params`.
///
/// Each variant tells the backend exactly where the parameter arrives and what
/// kind of store logic is needed, without the backend reimplementing the ABI
/// classification algorithm.
#[derive(Debug, Clone, Copy)]
pub enum ParamClass {
    /// Integer/pointer in GP register at `reg_idx`.
    IntReg { reg_idx: usize },
    /// Float/double in FP register at `reg_idx`.
    FloatReg { reg_idx: usize },
    /// i128 in aligned GP register pair starting at `base_reg_idx`.
    I128RegPair { base_reg_idx: usize },
    /// Small struct (<=16 bytes) by value in 1-2 GP registers.
    StructByValReg { base_reg_idx: usize, size: usize },
    /// F128 (long double) in FP register (ARM: Q-reg).
    F128FpReg { reg_idx: usize },
    /// F128 in GP register pair (RISC-V).
    F128GpPair { lo_reg_idx: usize, hi_reg_idx: usize },
    /// F128 always on stack (x86: x87 convention).
    F128AlwaysStack { offset: i64 },
    /// Regular scalar on the stack.
    StackScalar { offset: i64 },
    /// i128 on the stack (16-byte aligned).
    I128Stack { offset: i64 },
    /// F128 on the stack (overflow from registers).
    F128Stack { offset: i64 },
    /// Small struct that overflowed to the stack.
    StructStack { offset: i64, size: usize },
    /// Large struct (>16 bytes) passed on the stack.
    LargeStructStack { offset: i64, size: usize },
}

impl ParamClass {
    /// Returns true if this parameter is passed on the stack.
    pub fn is_stack(&self) -> bool {
        matches!(self,
            ParamClass::StackScalar { .. } | ParamClass::I128Stack { .. } |
            ParamClass::F128Stack { .. } | ParamClass::F128AlwaysStack { .. } |
            ParamClass::StructStack { .. } | ParamClass::LargeStructStack { .. }
        )
    }

    /// Returns true if this parameter arrives in a GP register (int, i128 pair, struct, or F128 GP pair).
    pub fn uses_gp_reg(&self) -> bool {
        matches!(self,
            ParamClass::IntReg { .. } | ParamClass::I128RegPair { .. } |
            ParamClass::StructByValReg { .. } | ParamClass::F128GpPair { .. }
        )
    }
}

/// Classify all parameters of a function for callee-side store emission.
///
/// Uses the same `CallAbiConfig` as `classify_call_args` to ensure caller and callee
/// agree on parameter locations. Returns one `ParamClass` per parameter.
pub fn classify_params(func: &IrFunction, config: &CallAbiConfig) -> Vec<ParamClass> {
    let mut result = Vec::with_capacity(func.params.len());
    let mut int_reg_idx = 0usize;
    let mut float_reg_idx = 0usize;
    let mut stack_offset: i64 = 0;

    for param in &func.params {
        let is_float = param.ty.is_float();
        let is_i128 = is_i128_type(param.ty);
        let is_long_double = param.ty.is_long_double();
        let struct_size = param.struct_size;

        // Struct-by-value parameters.
        if let Some(size) = struct_size {
            if size <= 16 {
                let regs_needed = if size <= 8 { 1 } else { 2 };
                if int_reg_idx + regs_needed <= config.max_int_regs {
                    result.push(ParamClass::StructByValReg {
                        base_reg_idx: int_reg_idx,
                        size,
                    });
                    int_reg_idx += regs_needed;
                } else {
                    result.push(ParamClass::StructStack {
                        offset: stack_offset,
                        size,
                    });
                    stack_offset += ((size + 7) & !7) as i64;
                    int_reg_idx = config.max_int_regs;
                }
            } else {
                result.push(ParamClass::LargeStructStack {
                    offset: stack_offset,
                    size,
                });
                stack_offset += ((size + 7) & !7) as i64;
            }
            continue;
        }

        // i128 parameters.
        if is_i128 {
            if config.align_i128_pairs && int_reg_idx % 2 != 0 {
                int_reg_idx += 1;
            }
            if int_reg_idx + 1 < config.max_int_regs {
                result.push(ParamClass::I128RegPair {
                    base_reg_idx: int_reg_idx,
                });
                int_reg_idx += 2;
            } else {
                stack_offset = (stack_offset + 15) & !15;
                result.push(ParamClass::I128Stack { offset: stack_offset });
                stack_offset += 16;
                int_reg_idx = config.max_int_regs;
            }
            continue;
        }

        // F128 / long double parameters.
        if is_long_double {
            if config.f128_in_fp_regs {
                // ARM: F128 in Q-register (uses FP register slot).
                if float_reg_idx < config.max_float_regs {
                    result.push(ParamClass::F128FpReg { reg_idx: float_reg_idx });
                    float_reg_idx += 1;
                } else {
                    stack_offset = (stack_offset + 15) & !15;
                    result.push(ParamClass::F128Stack { offset: stack_offset });
                    stack_offset += 16;
                }
            } else if config.f128_in_gp_pairs {
                // RISC-V: F128 in aligned GP pair.
                if config.align_i128_pairs && int_reg_idx % 2 != 0 {
                    int_reg_idx += 1;
                }
                if int_reg_idx + 1 < config.max_int_regs {
                    result.push(ParamClass::F128GpPair {
                        lo_reg_idx: int_reg_idx,
                        hi_reg_idx: int_reg_idx + 1,
                    });
                    int_reg_idx += 2;
                } else {
                    stack_offset = (stack_offset + 15) & !15;
                    result.push(ParamClass::F128Stack { offset: stack_offset });
                    stack_offset += 16;
                    int_reg_idx = config.max_int_regs;
                }
            } else {
                // x86: F128 always passes on the stack via x87.
                result.push(ParamClass::F128AlwaysStack { offset: stack_offset });
                stack_offset += 16;
            }
            continue;
        }

        // Float/double parameters.
        // On RISC-V variadic functions, float args go in GP registers instead of FP.
        let force_gp = config.variadic_floats_in_gp && is_float && !is_long_double;
        if is_float && !force_gp && float_reg_idx < config.max_float_regs {
            result.push(ParamClass::FloatReg { reg_idx: float_reg_idx });
            float_reg_idx += 1;
            continue;
        }
        // Float that overflowed FP registers goes to stack, not GP registers.
        if is_float && !force_gp {
            result.push(ParamClass::StackScalar { offset: stack_offset });
            stack_offset += 8;
            continue;
        }

        // GP register or stack overflow.
        if int_reg_idx < config.max_int_regs {
            result.push(ParamClass::IntReg { reg_idx: int_reg_idx });
            int_reg_idx += 1;
        } else {
            result.push(ParamClass::StackScalar { offset: stack_offset });
            stack_offset += 8;
        }
    }

    result
}
