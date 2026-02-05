Review src/backend/README.md for accuracy against actual code.

Key fixes needed:
- linker_common/ directory listing outdated (3 files listed, 18 actual)
- stack_layout/ directory listing outdated (3 files listed, 8 actual)
- assembler encoder is directory not file
- linker file descriptions for all 4 architectures outdated
- x86 peephole phase numbering (code has 7 phases, README says 8)
- section header says stack_layout.rs instead of stack_layout/
- F128SoftFloat ~45 -> ~48 methods
- i686 lacks f128.rs in codegen

Started: 2026-02-05
