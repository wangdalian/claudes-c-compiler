pub(crate) mod codegen;
#[cfg_attr(feature = "gcc_assembler", allow(dead_code))]
pub(crate) mod assembler;
#[cfg_attr(feature = "gcc_linker", allow(dead_code))]
pub(crate) mod linker;

pub(crate) use codegen::emit::ArmCodegen;
