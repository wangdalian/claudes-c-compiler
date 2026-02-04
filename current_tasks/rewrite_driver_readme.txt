Rewriting the driver README (src/driver/README.md) to be accurate and complete.

The current README is outdated - it still references the compiler as if it lacks
a native assembler/linker. The external_tools.rs section claims "If the compiler
later gains a native assembler/linker, only this file needs to change" when all
four architectures already have fully functional builtin assemblers and linkers.

Will audit all claims against the actual code and produce an accurate, comprehensive
design document covering:
- Driver struct and all configuration fields
- CLI parsing (response files, query flags, flag categories)
- Compilation pipeline with all phases
- Assembler/linker selection (MY_ASM/MY_LD)
- File type classification
- Build system compatibility features
- GCC -m16 delegation hack
- Dependency file generation
- Phase timing support
