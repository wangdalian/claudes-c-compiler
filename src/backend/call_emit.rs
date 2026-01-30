//! Re-exports from `call_abi` for backward compatibility.
//!
//! The callee-side parameter classification (`ParamClass`, `classify_params`, etc.)
//! was originally in this module. It has been unified with the caller-side classification
//! into `call_abi.rs`, which now contains the single shared classification algorithm.
//! This module re-exports the public API so that existing `use crate::backend::call_emit::*`
//! statements continue to work.

pub use super::call_abi::{
    ParamClass,
    classify_params, classify_params_full, named_params_stack_bytes,
};
