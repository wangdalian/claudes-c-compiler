# x86-64 Built-in Assembler -- Design Document

## Overview

This module is a native x86-64 assembler that translates AT&T-syntax assembly
text into ELF relocatable object files (`.o`).  It replaces the external
`gcc -c` invocation that would otherwise be needed to assemble the
compiler-generated assembly, removing the hard dependency on an installed
toolchain at compile time.

The assembler handles the **subset of AT&T syntax that the compiler's code
generator actually emits**.  It is not a general-purpose GAS replacement --
it intentionally does not implement the full GNU assembler specification, but
it covers enough of the ISA and directive set to assemble real C programs
compiled by this project.

### Entry Point

```rust
pub fn assemble(asm_text: &str, output_path: &str) -> Result<(), String>
```

Called when the built-in assembler is selected (`MY_ASM` mode).  It parses the
input text, encodes instructions, performs jump relaxation, resolves internal
relocations, and writes a standards-conforming ELF64 relocatable object file.


## Architecture / Pipeline

```
                          Assembly Source Text (AT&T syntax)
                                     |
                                     v
                    +--------------------------------+
                    |        parser.rs                |
                    |  - Line-by-line tokenization    |
                    |  - Directive parsing            |
                    |  - Operand parsing              |
                    |  - String/escape handling        |
                    +--------------------------------+
                                     |
                              Vec<AsmItem>
                                     |
                                     v
                    +--------------------------------+
                    |       elf_writer.rs             |
                    |  (First Pass)                   |
                    |  - Section management           |
                    |  - Symbol table construction    |
                    |  - Label position recording     |
                    |  - Delegates to encoder.rs      |
                    |    for each Instruction item    |
                    +--------------------------------+
                          |                   |
                          v                   v
            +------------------+   +--------------------+
            |   encoder.rs     |   | elf_writer.rs      |
            | - REX prefix     |   | (Second Pass)      |
            | - ModR/M + SIB   |   | - Jump relaxation  |
            | - Displacement   |   | - Size resolution   |
            | - Immediate      |   | - Internal reloc    |
            | - Relocation     |   |   resolution        |
            |   generation     |   | - ELF byte layout   |
            +------------------+   +--------------------+
                                             |
                                     ELF .o bytes
                                             |
                                             v
                                   std::fs::write()
```

### Pass Summary

| Pass | Module | Purpose |
|------|--------|---------|
| 1a   | `parser.rs` | Tokenize and parse all lines into `AsmItem` values |
| 1b   | `elf_writer.rs` + `encoder.rs` | Walk items sequentially: switch sections, record labels, encode instructions, emit data, collect relocations |
| 2a   | `elf_writer.rs` | Relax long jumps to short form (iterative) |
| 2b   | `elf_writer.rs` | Update symbol values after relaxation |
| 2c   | `elf_writer.rs` | Resolve `.size` directives |
| 2d   | `elf_writer.rs` | Resolve same-section PC-relative relocations for local symbols |
| 2e   | `elf_writer.rs` | Serialize the complete ELF file (headers, section data, symtab, strtab, rela sections) |


## File Inventory

| File | Lines | Role |
|------|-------|------|
| `mod.rs` | ~30 | Public `assemble()` entry point; module wiring |
| `parser.rs` | ~1100 | AT&T syntax line parser; produces `Vec<AsmItem>` |
| `encoder.rs` | ~2200 | x86-64 instruction encoder; produces machine code bytes + relocations |
| `elf_writer.rs` | ~1370 | Two-pass ELF builder; section/symbol management, jump relaxation, ELF serialization |


## Key Data Structures

### `AsmItem` (parser.rs)

The fundamental intermediate representation.  Each non-empty line of assembly
maps to one `AsmItem` variant:

```
AsmItem
  |-- Section(SectionDirective)     .section / .text / .data / .bss / .rodata
  |-- Global(String)                .globl name
  |-- Weak(String)                  .weak name
  |-- Hidden(String)                .hidden name
  |-- Protected(String)             .protected name
  |-- Internal(String)              .internal name
  |-- SymbolType(String, SymbolKind)  .type name, @function/@object/@tls_object
  |-- Size(String, SizeExpr)        .size name, expr
  |-- Label(String)                 name:
  |-- Align(u32)                    .align N / .p2align N
  |-- Byte(Vec<u8>)                 .byte val, ...
  |-- Short(Vec<i16>)               .short val, ...
  |-- Long(Vec<DataValue>)          .long val, ...  (may contain symbol refs)
  |-- Quad(Vec<DataValue>)          .quad val, ...  (may contain symbol refs)
  |-- Zero(u32)                     .zero N
  |-- Asciz(Vec<u8>)                .asciz "str"  (NUL-terminated)
  |-- Ascii(Vec<u8>)                .ascii "str"
  |-- Comm(String, u64, u32)        .comm name, size, align
  |-- Set(String, String)           .set alias, target
  |-- Cfi(CfiDirective)             .cfi_* (ignored for codegen)
  |-- File(u32, String)             .file N "name" (debug; ignored)
  |-- Loc(u32, u32, u32)            .loc filenum line col (debug; ignored)
  |-- Instruction(Instruction)      encoded x86-64 instruction
  |-- OptionDirective(String)       .option (RISC-V; ignored on x86)
  |-- Empty                         blank/comment-only line
```

### `Instruction` (parser.rs)

```rust
struct Instruction {
    prefix:   Option<String>,   // "lock", "rep", "repnz", ...
    mnemonic: String,           // "movq", "addl", "ret", ...
    operands: Vec<Operand>,     // AT&T order: source(s) first, destination last
}
```

### `Operand` (parser.rs)

```
Operand
  |-- Register(Register)                 %rax, %xmm0, %st(0)
  |-- Immediate(ImmediateValue)          $42, $symbol
  |-- Memory(MemoryOperand)              disp(%base, %index, scale)
  |-- Label(String)                      target label for jmp/call
  |-- Indirect(Box<Operand>)             *%rax, *addr
```

### `MemoryOperand` (parser.rs)

```rust
struct MemoryOperand {
    segment:      Option<String>,   // "fs" or "gs"
    displacement: Displacement,     // None | Integer | Symbol | SymbolMod | ...
    base:         Option<Register>, // (%rbp), (%rip), ...
    index:        Option<Register>, // (, %rcx, 4)
    scale:        Option<u8>,       // 1, 2, 4, or 8
}
```

### `InstructionEncoder` (encoder.rs)

Stateful encoder that accumulates machine code bytes and relocations for one
instruction at a time:

```rust
struct InstructionEncoder {
    bytes:       Vec<u8>,          // encoded machine code
    relocations: Vec<Relocation>,  // generated relocation entries
    offset:      u64,              // current position in section
}
```

### `Relocation` (encoder.rs)

```rust
struct Relocation {
    offset:     u64,     // where in the section the fix-up applies
    symbol:     String,  // target symbol name
    reloc_type: u32,     // R_X86_64_* constant
    addend:     i64,     // addend for RELA-style relocations
}
```

### `Section` (elf_writer.rs)

Tracks one ELF section as it is built up:

```rust
struct Section {
    name:         String,
    section_type: u32,              // SHT_PROGBITS, SHT_NOBITS, ...
    flags:        u64,              // SHF_ALLOC | SHF_WRITE | ...
    data:         Vec<u8>,          // accumulated section bytes
    alignment:    u64,
    relocations:  Vec<ElfRelocation>,
    jumps:        Vec<JumpInfo>,    // candidates for short-form relaxation
}
```

### `SymbolInfo` (elf_writer.rs)

```rust
struct SymbolInfo {
    name:         String,
    binding:      u8,      // STB_LOCAL, STB_GLOBAL, STB_WEAK
    sym_type:     u8,      // STT_FUNC, STT_OBJECT, STT_TLS, STT_NOTYPE
    visibility:   u8,      // STV_DEFAULT, STV_HIDDEN, STV_PROTECTED, STV_INTERNAL
    section:      Option<String>,
    value:        u64,     // offset within section
    size:         u64,
    is_common:    bool,
    common_align: u32,
}
```

### `ElfWriter` (elf_writer.rs)

The top-level builder that owns all sections, symbols, labels, and pending
attributes.  It drives the first-pass processing of `AsmItem` values and the
second-pass ELF emission:

```rust
struct ElfWriter {
    sections:          Vec<Section>,
    symbols:           Vec<SymbolInfo>,
    section_map:       HashMap<String, usize>,
    symbol_map:        HashMap<String, usize>,
    current_section:   Option<usize>,
    label_positions:   HashMap<String, (usize, u64)>,  // name -> (sec_idx, offset)
    pending_globals:   Vec<String>,
    pending_weaks:     Vec<String>,
    pending_types:     HashMap<String, SymbolKind>,
    pending_sizes:     HashMap<String, SizeExpr>,
    pending_hidden:    Vec<String>,
    pending_protected: Vec<String>,
    pending_internal:  Vec<String>,
    aliases:           HashMap<String, String>,
}
```


## Processing Algorithm

### Step 1: Parsing (parser.rs)

`parse_asm()` iterates over the input text line by line (1-based numbering for
error messages) and produces a `Vec<AsmItem>`.

1. **Comment stripping** -- `#` to end-of-line, respecting quoted strings.
2. **Semicolon splitting** -- GAS allows `;` as an instruction separator on a
   single line (e.g., `rep; nop`).  The parser splits on `;` outside strings
   and parses each part independently.
3. **Line classification** -- Each trimmed part is classified as:
   - **Label** -- Ends with `:`, no spaces, not starting with `$` or `%`.
   - **Directive** -- Starts with `.`.
   - **Prefixed instruction** -- Starts with `lock`, `rep`, `repz`, `repnz`.
   - **Instruction** -- Everything else.
4. **Directive parsing** -- A large `match` on the directive name handles
   `.section`, `.globl`, `.type`, `.size`, `.align`, `.byte`/`.short`/`.long`/
   `.quad`, `.zero`, `.asciz`/`.ascii`, `.comm`, `.set`, `.cfi_*`, `.file`,
   `.loc`, and more.  Unknown directives (`.ident`, `.addrsig`, etc.) are
   silently ignored.
5. **Instruction parsing** -- The mnemonic is split from operands by
   whitespace.  Operands are split by commas (respecting parentheses for memory
   operands) and individually parsed:
   - `%name` -> Register
   - `$value` -> Immediate (integer, symbol, symbol@modifier)
   - `disp(%base, %index, scale)` -> Memory
   - Bare identifier -> Label
   - `*operand` -> Indirect
6. **Displacement parsing** -- Handles integer, plain symbol, symbol+offset,
   and symbol@modifier (GOTPCREL, GOTTPOFF, TPOFF, PLT, etc.).
7. **Integer parsing** -- Decimal, `0x` hex, `0b` binary, and leading-zero
   octal.  Handles negative values and large unsigned values via `u64` parsing.
8. **String literal parsing** -- Supports standard C escapes (`\n`, `\t`,
   `\r`, `\0`, `\\`, `\"`, `\a`, `\b`, `\f`, `\v`), octal escapes (`\NNN`),
   and hex escapes (`\xNN`).  Non-ASCII characters are UTF-8 encoded.

### Step 2: First Pass -- Item Processing (elf_writer.rs)

`ElfWriter::build()` calls `process_item()` for each `AsmItem`:

- **Section** -- Creates or switches to the named section with appropriate
  type/flags.  Well-known names (`.text`, `.data`, `.bss`, `.rodata`, `.tdata`,
  `.tbss`, `.init_array`, `.fini_array`) get default flags.  The `.section`
  directive with explicit `"flags",@type` overrides these.
- **Global/Weak/Hidden/Protected/Internal** -- Recorded as pending attributes.
  They are applied when the next label bearing that name is defined.
- **SymbolType** -- Recorded in `pending_types`.
- **Size** -- For `.-symbol` expressions, a synthetic end-label is created at
  the current section position so that the correct size can be computed after
  jump relaxation.  Stored as `SizeExpr::SymbolDiff(end_label, start_label)`.
- **Label** -- Records `(section_index, byte_offset)` in `label_positions`.
  Creates or updates a `SymbolInfo` entry, applying any pending binding, type,
  and visibility attributes.
- **Align** -- Pads the current section to the requested alignment.  Code
  sections are padded with NOP (`0x90`); data sections with zero bytes.
- **Data directives** (Byte, Short, Long, Quad, Zero, Asciz, Ascii) -- Appended
  directly to the current section's data buffer.  Symbol references in `.long`
  and `.quad` generate `R_X86_64_32` or `R_X86_64_64` relocations.
  Symbol-difference expressions (`.long .LBB3 - .Ljt_0`) generate relocations
  with a `diff_symbol` field for later resolution.
- **Comm** -- Creates a COMMON symbol (global, object type, `SHN_COMMON`).
- **Set** -- Records a symbol alias in the `aliases` map.
- **Instruction** -- Delegates to `InstructionEncoder::encode()`.  The encoded
  bytes are appended to the current section.  Relocations are copied with
  adjusted offsets.  If the instruction is a jump to a label, a `JumpInfo`
  entry is recorded for potential relaxation.

### Step 3: Instruction Encoding (encoder.rs)

The encoder translates one `Instruction` into machine code bytes.  Its main
dispatch is a large `match` on the mnemonic string (200+ arms).

**Encoding fundamentals:**

1. **Prefix emission** -- `lock` (0xF0), `rep`/`repe` (0xF3), `repnz` (0xF2).
2. **Operand-size override** -- 16-bit operations emit `0x66` before REX.
3. **REX prefix** -- Computed from operand width (W), register extension bits
   (R, X, B), and special 8-bit register requirements (spl, bpl, sil, dil).
4. **Opcode emission** -- 1-3 bytes depending on instruction.
5. **ModR/M byte** -- `(mod << 6) | (reg << 3) | rm`, encoding the addressing
   mode and register fields.
6. **SIB byte** -- Emitted when the addressing mode uses an index register,
   RSP/R12 as base, or has no base register.
   `(scale << 6) | (index << 3) | base`.
7. **Displacement** -- 0, 1, or 4 bytes.  RBP/R13 always require at least
   disp8.  Symbol displacements always use disp32 with a relocation.
8. **Immediate** -- 1, 2, 4, or 8 bytes depending on size suffix and value.

**RIP-relative addressing:**

When the base register is `%rip`, the encoder emits `mod=00, rm=101` in ModR/M
and a 32-bit displacement.  Symbol references generate `R_X86_64_PC32`
relocations with addend `-4` (to account for the displacement being relative to
the end of the instruction).  Modifiers like `@GOTPCREL` and `@GOTTPOFF`
produce the corresponding relocation types.

**Relocation generation:**

Relocations are generated for:
- RIP-relative symbol references (PC32, PLT32, GOTPCREL, GOTTPOFF)
- Absolute symbol references in immediates and displacements (32, 32S, 64)
- Call/jump targets (PLT32)
- Data symbol references in `.long`/`.quad` directives (32, 64)

### Step 4: Jump Relaxation (elf_writer.rs)

After the first pass, `relax_jumps()` attempts to shrink long-form jumps to
short form:

```
Long JMP:  E9 rel32  (5 bytes) --> Short JMP:  EB rel8  (2 bytes)  saves 3
Long Jcc:  0F 8x rel32 (6 bytes) --> Short Jcc: 7x rel8  (2 bytes)  saves 4
```

The algorithm is iterative because shrinking one jump may bring other targets
into short-jump range:

1. Build a map of label positions within the current section.
2. For each un-relaxed jump, compute the displacement as if the jump were
   already in short form (end of instruction = offset + 2).
3. If the displacement fits in a signed byte [-128, +127], mark for relaxation.
4. Process relaxations back-to-front (so byte offsets remain valid):
   - Rewrite the opcode bytes in place (Jcc: `0x70+cc`; JMP: `0xEB`).
   - Remove the extra bytes via `data.drain()`.
   - Adjust all label positions, relocation offsets, and other jump offsets
     that fall after the shrunk instruction.
   - Remove the relocation entry for this jump (displacement is now inline).
5. Repeat from step 1 until no more relaxation occurs (fixed-point).
6. After convergence, compute and patch the final short displacements.

### Step 5: Post-Relaxation Resolution (elf_writer.rs)

1. **Symbol value update** -- All symbol values are refreshed from
   `label_positions` (which were adjusted during relaxation).
2. **Size resolution** -- `.size` directives using `.-symbol` are resolved by
   computing `end_label_offset - start_label_offset` from the updated label
   positions.
3. **Internal relocation resolution** -- Same-section PC-relative relocations
   (`R_X86_64_PC32`, `R_X86_64_PLT32`) targeting local symbols (`.L*` labels
   or symbols with `STB_LOCAL` binding) are resolved immediately.  The ELF
   formula `S + A - P` is applied and the result is patched into the section
   data.  Global/weak symbols are NOT resolved here -- their relocations are
   kept for the linker to handle symbol interposition and PLT redirection.
   Symbol-difference relocations (`.long a - b`) where both symbols are in
   the same section are also resolved to a constant.

### Step 6: ELF Emission (elf_writer.rs)

The shared `write_relocatable_object()` in `backend/elf.rs` serializes the
final ELF file. The x86 `emit_elf()` converts internal data to shared
`ObjSection`/`ObjSymbol`/`ObjReloc` types and delegates to this function.

**Layout:**
```
+---------------------------------------------------+
| ELF Header (64 bytes)                             |
+---------------------------------------------------+
| Section 1 data (aligned)                          |
| Section 2 data (aligned)                          |
| ...                                               |
+---------------------------------------------------+
| .rela.text (if .text has unresolved relocations)  |
| .rela.data (if .data has unresolved relocations)  |
| ...                                               |
+---------------------------------------------------+
| .symtab (8-byte aligned)                          |
|   - Null symbol                                   |
|   - Section symbols (STT_SECTION, one per section)|
|   - Local defined symbols                         |
|   - Global and weak symbols                       |
|   - Alias symbols (.set)                          |
|   - Undefined external symbols                    |
+---------------------------------------------------+
| .strtab (symbol name strings, NUL-terminated)     |
+---------------------------------------------------+
| .shstrtab (section name strings, NUL-terminated)  |
+---------------------------------------------------+
| Section Header Table (8-byte aligned)             |
|   [0] NULL                                        |
|   [1..N] data sections                            |
|   [N+1..] .rela.* sections                        |
|   [M] .symtab                                     |
|   [M+1] .strtab                                   |
|   [M+2] .shstrtab                                 |
+---------------------------------------------------+
```

**Symbol table ordering** (required by ELF spec):
1. Null symbol (index 0)
2. Section symbols (`STT_SECTION`, one per data section, `STB_LOCAL`)
3. Local non-internal symbols
4. ---- `sh_info` boundary (first global index) ----
5. Global and weak symbols
6. Alias symbols from `.set` directives
7. Undefined external symbols (auto-created from unresolved relocations)

**Relocation symbol resolution** (in `emit_elf`):
- Named symbols: converted to `ObjReloc` with the symbol name.
- Internal labels (`.L*`): converted to section name references with the
  label's offset added to the addend (section-symbol-relative addressing).

**String table construction** (`StringTable` in `backend/elf.rs`):
A simple append-only NUL-terminated string pool with deduplication via a
HashMap.  Used for both `.strtab` (symbol names) and `.shstrtab` (section
names).


## Supported Instruction Families

The encoder covers the following instruction categories:

| Category | Instructions |
|----------|-------------|
| **Data movement** | mov (b/w/l/q), movabs, movsx (bl/bq/wl/wq/slq), movzx (bl/bq/wl/wq), lea (l/q), push, pop, xchg, cmpxchg, xadd |
| **Arithmetic** | add, sub, adc, sbb, and, or, xor, cmp (b/w/l/q), test, neg, not, inc, dec, imul (1/2/3 operand), mul, div, idiv |
| **Shifts/Rotates** | shl, shr, sar, rol, ror (b/w/l/q), shld, shrd (q) |
| **Bit manipulation** | lzcnt, tzcnt, popcnt (l/q), bswap (l/q) |
| **Conditional** | setcc (all conditions), cmovcc (l/q, all conditions) |
| **Control flow** | jmp, jcc (all conditions), call, ret; direct, indirect, and RIP-relative forms |
| **Sign extension** | cltq, cqto, cltd, cdq |
| **SSE scalar float** | movss, movsd, addsd, subsd, mulsd, divsd, addss, subss, mulss, divss, sqrtsd, sqrtss, ucomisd, ucomiss, xorpd, xorps, andpd, andps, minsd, maxsd, minss, maxss |
| **SSE packed float** | addpd, subpd, mulpd, divpd, addps, subps, mulps, divps, orpd, orps, andnpd, andnps |
| **SSE data movement** | movaps, movdqa, movdqu, movupd, movd, movq (xmm), movlpd, movhpd |
| **SSE integer SIMD** | paddw, psubw, paddd, psubd, pmulhw, pmullw, pmulld, pmuludq, pmaddwd, pcmpgtw, pcmpgtb, pcmpeqb, pcmpeqd, pcmpeqw, packssdw, packuswb, pand, pandn, por, pxor, pminub, pmaxub, pminsd, pmaxsd, pavgb, pavgw, psadbw, punpckl/h (bw/wd/dq/qdq), paddus/subs/padds (b/w) |
| **SSE shifts** | psllw, psrlw, psraw, pslld, psrld, psrad, psllq, psrlq, pslldq, psrldq (imm and xmm forms) |
| **SSE shuffles** | pshufd, pshuflw, pshufhw |
| **SSE insert/extract** | pinsrb, pinsrw, pinsrd, pinsrq, pextrb, pextrw, pextrd, pextrq |
| **SSE conversions** | cvtsd2ss, cvtss2sd, cvtsi2sdq, cvtsi2ssq, cvttsd2siq, cvttss2siq, pmovmskb |
| **AES-NI** | aesenc, aesenclast, aesdec, aesdeclast, aesimc, aeskeygenassist |
| **PCLMULQDQ** | pclmulqdq |
| **CRC32** | crc32 (b/w/l/q) |
| **Non-temporal** | movnti, movntdq, movntpd |
| **x87 FPU** | fld, fstp, fldl, flds, fstpl, fstps, fldt, fstpt, fildq, fisttpq, faddp, fsubp, fsubrp, fmulp, fdivp, fdivrp, fchs, fcomip, fucomip |
| **String ops** | movsb, movsd (string), stosb, stosd |
| **Atomics** | lock prefix, xchg, cmpxchg, xadd (all sizes) |
| **Misc** | nop, ud2, endbr64, pause, mfence, lfence, sfence, clflush, syscall-ready patterns |


## Supported Relocation Types

| Constant | Value | Usage |
|----------|-------|-------|
| `R_X86_64_NONE` | 0 | No relocation |
| `R_X86_64_64` | 1 | Absolute 64-bit address (`.quad symbol`) |
| `R_X86_64_PC32` | 2 | PC-relative 32-bit (RIP-relative addressing, `.long a - b`) |
| `R_X86_64_PLT32` | 4 | PC-relative via PLT (call/jmp targets) |
| `R_X86_64_32` | 10 | Absolute 32-bit unsigned (`.long symbol`) |
| `R_X86_64_32S` | 11 | Absolute 32-bit signed (symbol displacement in non-RIP memory) |
| `R_X86_64_GOTPCREL` | 9 | GOT entry, PC-relative (`symbol@GOTPCREL(%rip)`) |
| `R_X86_64_TPOFF32` | 23 | TLS offset from thread pointer (`symbol@TPOFF`) |
| `R_X86_64_GOTTPOFF` | 22 | TLS GOT entry, PC-relative (`symbol@GOTTPOFF(%rip)`) |


## Supported Directives

| Directive | Purpose |
|-----------|---------|
| `.section name, "flags", @type` | Switch to named section with explicit attributes |
| `.text`, `.data`, `.bss`, `.rodata` | Switch to well-known section (shorthand) |
| `.globl` / `.global` | Mark symbol as global binding |
| `.weak` | Mark symbol as weak binding |
| `.hidden`, `.protected`, `.internal` | Set symbol visibility |
| `.type name, @function/@object/@tls_object/@notype` | Set symbol type |
| `.size name, expr` | Set symbol size (supports `.-name` and constants) |
| `.align N` / `.p2align N` | Align current position (byte-count / power-of-2) |
| `.byte`, `.short`/`.value`/`.2byte`, `.long`/`.4byte`, `.quad`/`.8byte` | Emit data |
| `.zero N` / `.skip N` | Emit N zero bytes |
| `.asciz` / `.string` | Emit NUL-terminated string |
| `.ascii` | Emit string without NUL terminator |
| `.comm name, size, align` | Define common (BSS) symbol |
| `.set alias, target` | Define symbol alias |
| `.cfi_*` | CFI directives (parsed but not emitted to .eh_frame) |
| `.file`, `.loc` | Debug info directives (parsed, ignored for codegen) |


## Key Design Decisions and Trade-offs

### 1. Subset, Not Full GAS

The assembler only implements what the compiler's codegen actually produces.
This dramatically simplifies the implementation -- no macro system, no
conditional assembly, no `.if`/`.else`, no expression evaluation beyond simple
`symbol+offset`.  Unknown directives are silently ignored with `AsmItem::Empty`,
which provides forward compatibility as the codegen evolves.

### 2. Mnemonic-Suffix-Based Size Inference

Rather than building a full operand-type inference system, the encoder uses the
AT&T mnemonic suffix (`b`/`w`/`l`/`q`) to determine operand size.  This is
reliable because the compiler always emits sized mnemonics and is much simpler
than deriving size from register names.

### 3. Always-Long Encoding with Post-Hoc Relaxation

Instructions are initially encoded in their longest form (e.g., `jmp` always
uses `E9 rel32`).  The relaxation pass then shrinks eligible jumps to short
form.  This two-phase approach avoids the complexity of an iterative
encode-measure-re-encode loop during the first pass.

### 4. Iterative Jump Relaxation

The relaxation loop runs until convergence because shrinking one jump can bring
another jump's target into short range.  Relaxations are processed back-to-front
within each iteration to maintain valid byte offsets.  This is the same approach
used by production linkers (ld, lld).

### 5. Lazy Symbol Attribute Application

`.globl`, `.type`, `.hidden`, etc. are collected as "pending" attributes and
only applied when a label definition (`name:`) is encountered.  This handles
the common GAS pattern where attributes appear before the label:

```asm
.globl main
.type main, @function
main:
```

### 6. Internal Labels via Section Symbols

Labels starting with `.` (e.g., `.LBB0`, `.Lstr0`) are treated as
section-local.  In the ELF output, they are not emitted as named symbols.
Instead, relocations referencing them use the parent section's section symbol,
with the addend adjusted to include the label's offset.  This matches GAS
behavior and keeps the symbol table small.

### 7. CFI Directives Parsed but Not Emitted

The assembler parses `.cfi_startproc`, `.cfi_endproc`, `.cfi_def_cfa_offset`,
etc. into `CfiDirective` variants but does not generate `.eh_frame` section
data.  This is acceptable because the built-in linker does not require unwind
information for basic compilation, and adding `.eh_frame` generation would
substantially increase complexity.

### 8. No Instruction Shortening Beyond Jumps

The encoder does not attempt to select shorter instruction forms for arithmetic
(e.g., `add $1, %rax` could use `inc %rax`).  It faithfully encodes what the
codegen emits.  The one exception is immediate-size optimization: ALU
instructions with small immediates automatically use the sign-extended imm8
form (`0x83` instead of `0x81`), and `movq` with 32-bit-range immediates uses
`C7` instead of `movabs`.

### 9. Two Separate String Tables

The ELF file uses distinct `.strtab` (symbol names) and `.shstrtab` (section
names) string tables, each built by the same `StringTable` helper.  This is
the standard ELF convention and simplifies the section header linkage.
