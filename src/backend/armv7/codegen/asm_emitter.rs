//! ARMv7 inline assembly emitter (InlineAsmEmitter trait).
//! Placeholder - delegates to the basic inline_asm module.

use super::emit::Armv7Codegen;

// The ARMv7 inline asm support is handled directly in inline_asm.rs
// via emit_inline_asm_impl. The InlineAsmEmitter trait is not needed
// for the basic implementation.
pub(super) const ARM_GP_SCRATCH: &[&str] = &["r4", "r5", "r6", "r7", "r8", "r9", "r10"];
