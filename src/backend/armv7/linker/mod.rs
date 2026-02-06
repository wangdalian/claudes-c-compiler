//! Native ARMv7 ELF linker.
//!
//! Links ELF32 ARM object files into executables or shared libraries.
//! Closely follows the i686 linker's architecture but adapted for ARM.
//!
//! ## Module structure
//!
//! - `types` - ELF32 constants, structures, and linker state types
//! - `parse` - ELF32 ARM object file parsing
//! - `reloc` - ARM relocation application
//! - `sections` - Section merging and COMDAT deduplication
//! - `symbols` - Symbol resolution, PLT/GOT marking, IFUNC collection
//! - `shared` - Shared library (.so) emission
//! - `emit` - Executable layout and ELF32 emission
//! - `link` - Orchestration: `link_builtin` and `link_shared` entry points

#[allow(dead_code)]
mod types;
mod parse;
mod reloc;
mod sections;
mod symbols;
mod shared;
mod emit;
mod link;

use crate::backend::linker_common;

struct DynStrTab(linker_common::DynStrTab);

impl DynStrTab {
    fn new() -> Self { Self(linker_common::DynStrTab::new()) }
    fn add(&mut self, s: &str) -> u32 { self.0.add(s) as u32 }
    fn get_offset(&self, s: &str) -> u32 { self.0.get_offset(s) as u32 }
    fn as_bytes(&self) -> &[u8] { self.0.as_bytes() }
}

#[cfg(not(feature = "gcc_linker"))]
pub use link::link_builtin;
#[cfg(not(feature = "gcc_linker"))]
pub use link::link_shared;
