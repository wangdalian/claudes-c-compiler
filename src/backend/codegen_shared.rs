//! Re-export shim for backwards compatibility.
//!
//! The codegen shared framework has been split into focused modules:
//! - `state`: CodegenState, StackSlot, SlotAddr
//! - `traits`: ArchCodegen trait with default implementations
//! - `generation`: generate_module, generate_function, instruction dispatch
//! - `call_abi`: CallArgClass, CallAbiConfig, classify_call_args
//! - `cast`: CastKind, classify_cast, FloatOp, classify_float_binop
//! - `inline_asm`: InlineAsmEmitter trait, AsmOperand types, emit_inline_asm_common
//!
//! This file re-exports everything so existing `use crate::backend::codegen_shared::*`
//! imports continue to work.

pub use super::state::*;
pub use super::traits::*;
pub use super::generation::*;
pub use super::call_abi::*;
pub use super::cast::*;
pub use super::inline_asm::*;
