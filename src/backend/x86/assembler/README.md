# x86-64 Built-in Assembler -- Design Document

## Overview

This module is a native x86-64 assembler that translates AT&T-syntax assembly
text into ELF relocatable object files (`.o`).  It replaces the external
`gcc -c` invocation that would otherwise be needed to assemble
compiler-generated assembly, removing the hard dependency on an installed
toolchain at compile time.

The assembler handles the full subset of AT&T syntax that the compiler's code
generator emits, plus enough additional coverage to assemble hand-written
assembly from projects such as musl libc.  It is not a general-purpose GAS
replacement -- it intentionally does not implement the full GNU assembler
specification -- but it covers a broad swath of the x86-64 ISA, including SSE
through SSE4.1, AVX, AVX2, BMI2, AES-NI, PCLMULQDQ, CRC32, and x87 FPU
instructions.

### Entry Point

```rust
pub fn assemble(asm_text: &str, output_path: &str) -> Result<(), String>
```

Defined in `mod.rs`.  Called when the built-in assembler is selected
(`MY_ASM` mode).  The implementation is three lines:

1. `parse_asm(asm_text)` -- parse into `Vec<AsmItem>`
2. `ElfWriter::new().build(&items)` -- encode, relax, resolve, serialize
3. `std::fs::write(output_path, &elf_bytes)` -- write the object file


## Architecture / Pipeline

```
                      Assembly Source Text (AT&T syntax)
                                   |
                                   v
                  +------------------------------------+
                  |           parser.rs                 |
                  |  1. strip_c_comments()              |
                  |  2. Split into lines                |
                  |  3. expand_rept_blocks()            |
                  |  4. For each line:                  |
                  |     - Strip # comments              |
                  |     - Split on semicolons           |
                  |     - Parse directives/instructions |
                  +------------------------------------+
                                   |
                            Vec<AsmItem>
                                   |
                                   v
                  +------------------------------------+
                  |          elf_writer.rs              |
                  |  (First Pass -- process_item)       |
                  |  - Numeric label resolution         |
                  |  - Section management               |
                  |  - Symbol table construction        |
                  |  - Label position recording         |
                  |  - Data emission                    |
                  |  - Delegates to encoder.rs          |
                  |    for each Instruction item        |
                  +------------------------------------+
                        |                   |
                        v                   v
          +--------------------+   +------------------------+
          |    encoder.rs      |   |     elf_writer.rs      |
          | - Suffix inference |   |  (Second Pass)         |
          | - REX/VEX prefix   |   |  1. relax_jumps()      |
          | - ModR/M + SIB     |   |  2. resolve_deferred   |
          | - Displacement     |   |     _skips()           |
          | - Immediate        |   |  3. resolve_deferred   |
          | - Relocation       |   |     _byte_diffs()      |
          |   generation       |   |  4. Update symbol vals |
          |                    |   |  5. Resolve .size dirs  |
          +--------------------+   |  6. resolve_internal   |
                                   |     _relocations()     |
                                   |  7. emit_elf() ->      |
                                   |     write_relocatable  |
                                   |     _object()          |
                                   +------------------------+
                                             |
                                     ELF .o bytes
                                             |
                                             v
                                   std::fs::write()
```

### Pass Summary

| Pass | Module | Purpose |
|------|--------|---------|
| 1a   | `parser.rs` | Tokenize and parse all lines into `Vec<AsmItem>` |
| 1b   | `elf_writer.rs` + `encoder.rs` | Walk items sequentially: switch sections, record labels, encode instructions, emit data, collect relocations |
| 2a   | `elf_writer.rs` | Relax long jumps to short form (iterative until convergence) |
| 2b   | `elf_writer.rs` | Resolve deferred `.skip` expressions (insert bytes, shift labels/relocations) |
| 2c   | `elf_writer.rs` | Resolve deferred byte-sized symbol diffs |
| 2d   | `elf_writer.rs` | Update symbol values from `label_positions` (post-relaxation/skip) |
| 2e   | `elf_writer.rs` | Resolve `.size` directives |
| 2f   | `elf_writer.rs` | Resolve same-section PC-relative relocations for local symbols |
| 2g   | `elf_writer.rs` | Resolve `.set` aliases, create undefined/weak symbols, convert to shared format, serialize ELF via `write_relocatable_object()` |


## File Inventory

| File | Lines | Role |
|------|-------|------|
| `mod.rs` | 30 | Public `assemble()` entry point; module wiring |
| `parser.rs` | ~1490 | AT&T syntax line parser with C-comment stripping, `.rept` expansion, expression evaluation; produces `Vec<AsmItem>` |
| `encoder.rs` | ~4060 | x86-64 instruction encoder with 300+ match arms; VEX prefix support for AVX/BMI2; suffix inference for hand-written assembly |
| `elf_writer.rs` | ~1520 | Two-pass ELF builder; section/symbol management, numeric labels, jump relaxation, deferred expression evaluation, ELF serialization |


## Key Data Structures

### `AsmItem` enum (29 variants) -- parser.rs

The fundamental intermediate representation.  Each non-empty line of assembly
maps to one `AsmItem` variant:

```
AsmItem
  |-- Section(SectionDirective)        .section / .text / .data / .bss / .rodata
  |-- Global(String)                   .globl name
  |-- Weak(String)                     .weak name
  |-- Hidden(String)                   .hidden name
  |-- Protected(String)                .protected name
  |-- Internal(String)                 .internal name
  |-- SymbolType(String, SymbolKind)   .type name, @function/@object/@tls_object/@notype
  |-- Size(String, SizeExpr)           .size name, expr
  |-- Label(String)                    name:
  |-- Align(u32)                       .align N / .p2align N / .balign N
  |-- Byte(Vec<DataValue>)             .byte val, ...
  |-- Short(Vec<DataValue>)            .short/.value/.2byte val, ...
  |-- Long(Vec<DataValue>)             .long/.4byte val, ...
  |-- Quad(Vec<DataValue>)             .quad/.8byte val, ...
  |-- Zero(u32)                        .zero N / .skip N
  |-- SkipExpr(String, u8)             .skip expr, fill  (deferred expression)
  |-- Asciz(Vec<u8>)                   .asciz "str" / .string "str"
  |-- Ascii(Vec<u8>)                   .ascii "str"
  |-- Comm(String, u64, u32)           .comm name, size, align
  |-- Set(String, String)              .set alias, target
  |-- Cfi(CfiDirective)                .cfi_* (parsed but not emitted)
  |-- File(u32, String)                .file N "name"
  |-- Loc(u32, u32, u32)               .loc filenum line col
  |-- Instruction(Instruction)         encoded x86-64 instruction
  |-- OptionDirective(String)          .option (ignored on x86)
  |-- PushSection(SectionDirective)    .pushsection
  |-- PopSection                       .popsection / .previous
  |-- Org(String, i64)                 .org expression, fill
  |-- Empty                            blank/comment-only line
```

### `SectionDirective` struct -- parser.rs

```rust
pub struct SectionDirective {
    pub name: String,
    pub flags: Option<String>,
    pub section_type: Option<String>,
    pub extra: Option<String>,
}
```

### `SymbolKind` enum -- parser.rs

```rust
pub enum SymbolKind {
    Function,
    Object,
    TlsObject,
    NoType,
}
```

### `SizeExpr` enum -- parser.rs

```rust
pub enum SizeExpr {
    Constant(u64),
    CurrentMinusSymbol(String),        // .-symbol
    SymbolDiff(String, String),        // end_label - start_label
}
```

### `DataValue` enum -- parser.rs

```rust
pub enum DataValue {
    Integer(i64),
    Symbol(String),
    SymbolOffset(String, i64),         // symbol + offset
    SymbolDiff(String, String),        // symbol_a - symbol_b
}
```

### `CfiDirective` enum -- parser.rs

```rust
pub enum CfiDirective {
    StartProc,
    EndProc,
    DefCfaOffset(i32),
    DefCfaRegister(String),
    Offset(String, i32),
    Other(String),
}
```

### `Instruction` struct -- parser.rs

```rust
pub struct Instruction {
    pub prefix: Option<String>,    // "lock", "rep", "repnz", "notrack"
    pub mnemonic: String,
    pub operands: Vec<Operand>,
}
```

### `Operand` enum -- parser.rs

```rust
pub enum Operand {
    Register(Register),            // %rax, %xmm0, %st(0)
    Immediate(ImmediateValue),     // $42, $symbol
    Memory(MemoryOperand),         // disp(%base, %index, scale)
    Label(String),                 // target label for jmp/call
    Indirect(Box<Operand>),        // *%rax, *addr
}
```

### `Register` struct -- parser.rs

```rust
pub struct Register {
    pub name: String,
}
```

### `ImmediateValue` enum -- parser.rs

```rust
pub enum ImmediateValue {
    Integer(i64),
    Symbol(String),
    SymbolMod(String, String),     // symbol@GOTPCREL, symbol@TPOFF, etc.
}
```

### `MemoryOperand` struct -- parser.rs

```rust
pub struct MemoryOperand {
    pub segment: Option<String>,   // "fs" or "gs"
    pub displacement: Displacement,
    pub base: Option<Register>,
    pub index: Option<Register>,
    pub scale: Option<u8>,         // 1, 2, 4, or 8
}
```

### `Displacement` enum -- parser.rs

```rust
pub enum Displacement {
    None,
    Integer(i64),
    Symbol(String),
    SymbolAddend(String, i64),     // symbol+offset or symbol-offset
    SymbolMod(String, String),     // symbol@GOT, symbol@GOTPCREL, etc.
    SymbolPlusOffset(String, i64), // symbol+N or symbol-N
}
```

### `InstructionEncoder` struct -- encoder.rs

Stateful encoder that accumulates machine code bytes and relocations for one
instruction at a time:

```rust
pub struct InstructionEncoder {
    pub bytes: Vec<u8>,            // encoded machine code
    pub relocations: Vec<Relocation>,
    pub offset: u64,               // current position in section
}
```

### `Relocation` struct -- encoder.rs

```rust
pub struct Relocation {
    pub offset: u64,               // where in the section the fix-up applies
    pub symbol: String,            // target symbol name
    pub reloc_type: u32,           // R_X86_64_* constant
    pub addend: i64,               // addend for RELA-style relocations
}
```

### `ElfWriter` struct (18 fields) -- elf_writer.rs

The top-level builder that owns all sections, symbols, labels, and pending
attributes.  Drives both passes:

```rust
struct ElfWriter {
    sections: Vec<Section>,
    symbols: Vec<SymbolInfo>,
    section_map: HashMap<String, usize>,
    symbol_map: HashMap<String, usize>,
    current_section: Option<usize>,
    label_positions: HashMap<String, (usize, u64)>,
    numeric_label_positions: HashMap<String, Vec<(usize, u64)>>,
    pending_globals: Vec<String>,
    pending_weaks: Vec<String>,
    pending_types: HashMap<String, SymbolKind>,
    pending_sizes: HashMap<String, SizeExpr>,
    pending_hidden: Vec<String>,
    pending_protected: Vec<String>,
    pending_internal: Vec<String>,
    aliases: HashMap<String, String>,
    section_stack: Vec<Option<usize>>,
    deferred_skips: Vec<(usize, usize, String, u8)>,
    deferred_byte_diffs: Vec<(usize, usize, String, String, usize)>,
}
```

### `Section` struct (8 fields) -- elf_writer.rs

Tracks one ELF section as it is built up:

```rust
struct Section {
    name: String,
    section_type: u32,             // SHT_PROGBITS, SHT_NOBITS, ...
    flags: u64,                    // SHF_ALLOC | SHF_WRITE | ...
    data: Vec<u8>,                 // accumulated section bytes
    alignment: u64,
    relocations: Vec<ElfRelocation>,
    jumps: Vec<JumpInfo>,          // candidates for short-form relaxation
    index: usize,                  // (dead code)
}
```

### `SymbolInfo` struct (9 fields) -- elf_writer.rs

```rust
struct SymbolInfo {
    name: String,
    binding: u8,                   // STB_LOCAL, STB_GLOBAL, STB_WEAK
    sym_type: u8,                  // STT_FUNC, STT_OBJECT, STT_TLS, STT_NOTYPE
    visibility: u8,                // STV_DEFAULT, STV_HIDDEN, STV_PROTECTED, STV_INTERNAL
    section: Option<String>,       // section name, or None for undefined/common
    value: u64,                    // offset within section
    size: u64,
    is_common: bool,
    common_align: u32,
}
```

### `JumpInfo` struct (5 fields) -- elf_writer.rs

```rust
struct JumpInfo {
    offset: usize,                 // offset of jump instruction in section data
    len: usize,                    // total instruction length (5 for JMP, 6 for Jcc)
    target: String,                // target label name
    is_conditional: bool,          // Jcc vs JMP
    relaxed: bool,                 // whether already relaxed to short form
}
```

### `ElfRelocation` struct (6 fields) -- elf_writer.rs

```rust
struct ElfRelocation {
    offset: u64,
    symbol: String,
    reloc_type: u32,
    addend: i64,
    diff_symbol: Option<String>,   // for symbol-difference relocations (.long a - b)
    patch_size: u8,                // 1 for .byte, 2 for .2byte, 4 for .long, 8 for .quad
}
```


## Processing Algorithm

### Step 1: Parsing (parser.rs)

`parse_asm()` transforms raw assembly text into `Vec<AsmItem>` through a
multi-stage pipeline:

1. **C-comment stripping** -- `strip_c_comments()` removes all `/* ... */`
   comments globally, preserving newlines so that line numbers remain correct
   for error messages.  This handles multi-line comments found in hand-written
   assembly (e.g., musl libc).

2. **Line splitting** -- The comment-stripped text is split into lines.

3. **`.rept` expansion** -- `expand_rept_blocks()` finds `.rept N` / `.endr`
   pairs and repeats the enclosed lines N times.  The repeat count is
   evaluated via `parse_integer_expr()`, supporting full arithmetic
   expressions.

4. **Per-line processing** -- For each line:
   - Strip `#`-to-end-of-line comments.
   - Split on semicolons (GAS instruction separator, e.g., `rep; nop`).
   - Parse each part independently.

5. **Line classification** -- Each trimmed part is classified as:
   - **Label** -- Ends with `:`, no spaces, not starting with `$` or `%`.
   - **Directive** -- Starts with `.`.
   - **Prefixed instruction** -- Starts with `lock`, `rep`, `repz`, `repnz`,
     or `notrack`.
   - **Instruction** -- Everything else.

6. **Directive parsing** -- A large `match` on the directive name handles
   `.section`, `.globl`, `.type`, `.size`, `.align`/`.p2align`/`.balign`,
   `.byte`/`.short`/`.value`/`.2byte`/`.long`/`.4byte`/`.quad`/`.8byte`,
   `.zero`, `.skip`, `.asciz`/`.string`/`.ascii`, `.comm`, `.set`,
   `.cfi_*`, `.file`, `.loc`, `.pushsection`, `.popsection`/`.previous`,
   `.org`, and more.  Unknown directives (`.ident`, `.addrsig`, etc.) are
   silently ignored and produce `AsmItem::Empty`.

7. **Instruction parsing** -- The mnemonic is split from operands by
   whitespace.  Operands are split by commas (respecting parentheses for
   memory operands) and individually parsed:
   - `%name` -> `Register`
   - `$value` -> `Immediate` (integer, symbol, or symbol@modifier)
   - `disp(%base, %index, scale)` -> `Memory`
   - Bare identifier -> `Label`
   - `*operand` -> `Indirect`

8. **Displacement parsing** -- Handles integer literals, plain symbol
   references, symbol+offset, and symbol@modifier forms (GOTPCREL, GOTTPOFF,
   TPOFF, PLT, etc.).

9. **Expression evaluation** -- `parse_integer_expr()` implements a full
   recursive-descent expression evaluator supporting `+`, `-`, `*`, `/`,
   `%`, `|`, `^`, `&`, `<<`, `>>`, `~`, and parentheses with proper operator
   precedence.  Used for integer expressions in data directives, alignment
   values, and `.skip`/`.rept` arguments.

10. **Integer parsing** -- Decimal, `0x` hex, `0b` binary, and leading-zero
    octal.  Handles negative values and large unsigned values via `u64`
    parsing.

11. **String literal parsing** -- Supports standard C escapes (`\n`, `\t`,
    `\r`, `\0`, `\\`, `\"`, `\a`, `\b`, `\f`, `\v`), octal escapes
    (`\NNN`), and hex escapes (`\xNN`).

### Step 2: First Pass -- Item Processing (elf_writer.rs)

`ElfWriter::build()` first resolves numeric local labels (1:, 2:, etc.) to
unique names via a shared utility, then calls `process_item()` for each
`AsmItem`:

- **Section** -- Creates or switches to the named section with appropriate
  type/flags.  Well-known names get default flags (see "Well-Known Section
  Defaults" below).  The `.section` directive with explicit `"flags",@type`
  overrides these.

- **PushSection / PopSection** -- `section_stack: Vec<Option<usize>>` saves
  and restores `current_section`.  Push saves the current section and switches
  to the new one; pop restores from the stack.  If the stack is empty, pop
  silently keeps the current section (matching GNU as behavior).

- **Global/Weak/Hidden/Protected/Internal** -- Recorded as pending attributes.
  They are applied when a label bearing that name is defined.

- **SymbolType** -- Recorded in `pending_types`.

- **Size** -- For `.-symbol` expressions, a synthetic end-label
  (`.Lsize_end_<name>`) is created at the current section position so that the
  correct size can be computed after jump relaxation.  The expression is
  stored as `SizeExpr::SymbolDiff(end_label, start_label)`.

- **Label** -- Records `(section_index, byte_offset)` in `label_positions`.
  Numeric labels (all-digit names) are also appended to
  `numeric_label_positions`.  Creates or updates a `SymbolInfo` entry,
  applying any pending binding, type, and visibility attributes.

- **Align** -- Pads the current section to the requested alignment.  Updates
  the section's `alignment` field if the new value is larger.  Code sections
  (with `SHF_EXECINSTR`) are padded with NOP (`0x90`); data sections with
  zero bytes.

- **Data directives** (Byte, Short, Long, Quad) -- Appended to the current
  section's data buffer via `emit_data_values()`.  Each `DataValue` variant
  is handled:
  - `Integer` -- Written directly in little-endian format.
  - `Symbol` -- Generates an `R_X86_64_32` (4-byte) or `R_X86_64_64`
    (8-byte) relocation.  `.set` aliases with label-difference expressions
    are resolved via `SymbolDiff`.
  - `SymbolOffset` -- Like `Symbol` but with a non-zero addend.
  - `SymbolDiff` -- For `.long a - b`: if `b` is `.` (current position),
    emits `R_X86_64_PC32`.  For byte/short-sized diffs, defers resolution
    to `deferred_byte_diffs`.  For 4-byte/8-byte diffs, records a relocation
    with `diff_symbol` set.

- **Zero** -- Extends the section data with N zero bytes.

- **Org** -- Advances the current position to the specified offset within the
  section.  Resolves the target via label lookup or numeric label resolution.
  Code sections pad with NOP; data sections pad with zero.

- **SkipExpr** -- Records a deferred skip in `deferred_skips` to be evaluated
  after all labels are known.  Used by complex expressions like those in the
  Linux kernel alternatives framework.

- **Asciz / Ascii** -- Appended directly to the current section.

- **Comm** -- Creates a COMMON symbol (`STB_GLOBAL`, `STT_OBJECT`,
  `SHN_COMMON`).

- **Set** -- Records a symbol alias in the `aliases` map.

- **Instruction** -- Delegates to `InstructionEncoder::encode()`.  The encoded
  bytes are appended to the current section.  Relocations are copied with
  adjusted offsets.  If the instruction is a jump to a label, a `JumpInfo`
  entry is recorded for potential relaxation.

- **Ignored items** -- `Cfi`, `File`, `Loc`, `OptionDirective`, and `Empty`
  are silently skipped.

### Step 3: Instruction Encoding (encoder.rs)

The encoder translates one `Instruction` into machine code bytes.  Its main
dispatch is a large `match` on the mnemonic string (300+ arms).

**Suffix inference** -- The `infer_suffix()` function (at line 181) infers
AT&T size suffixes from register operands for unsuffixed mnemonics.  This
enables hand-written assembly (e.g., musl's `.s` files) that omits the size
suffix when it can be derived from context.  Only mnemonics in a whitelist are
candidates:

```
mov, add, sub, and, or, xor, cmp, test, push, pop, lea,
shl, shr, sar, rol, ror, inc, dec, neg, not,
imul, mul, div, idiv, adc, sbb,
xchg, cmpxchg, xadd, bswap, bsf, bsr
```

**Encoding fundamentals:**

1. **Prefix emission** -- `lock` (0xF0), `rep`/`repe`/`repz` (0xF3),
   `repne`/`repnz` (0xF2), `notrack` (0x3E).
2. **Operand-size override** -- 16-bit operations emit `0x66` before REX.
3. **REX prefix** -- Computed from operand width (W), register extension bits
   (R, X, B), and special 8-bit register requirements (spl, bpl, sil, dil).
4. **VEX prefix** -- Used for AVX and BMI2 instructions.  The encoder
   constructs 2-byte or 3-byte VEX prefixes with the appropriate `vvvv`,
   `L`, `pp`, `mmmmm`, `W`, and extension bits.
5. **Opcode emission** -- 1-3 bytes depending on instruction.
6. **ModR/M byte** -- `(mod << 6) | (reg << 3) | rm`, encoding the addressing
   mode and register fields.
7. **SIB byte** -- Emitted when the addressing mode uses an index register,
   RSP/R12 as base, or has no base register.
   `(scale << 6) | (index << 3) | base`.
8. **Displacement** -- 0, 1, or 4 bytes.  RBP/R13 always require at least
   disp8.  Symbol displacements always use disp32 with a relocation.
9. **Immediate** -- 1, 2, 4, or 8 bytes depending on size suffix and value.

**RIP-relative addressing:**

When the base register is `%rip`, the encoder emits `mod=00, rm=101` in
ModR/M and a 32-bit displacement.  Symbol references generate `R_X86_64_PC32`
relocations with addend `-4` (to account for the displacement being relative
to the end of the instruction).  Modifiers like `@GOTPCREL` and `@GOTTPOFF`
produce the corresponding relocation types.

**Relocation generation:**

Relocations are generated for:
- RIP-relative symbol references (PC32, PLT32, GOTPCREL, GOTTPOFF)
- Absolute symbol references in immediates and displacements (32, 32S, 64)
- Call/jump targets (PLT32)
- Data symbol references in `.long`/`.quad` directives (32, 64)
- Thread-local storage references (TPOFF32, GOTTPOFF)
- Internal-only 8-bit PC-relative for `jrcxz`/`loop` (R_X86_64_PC8_INTERNAL)

### Step 4: Jump Relaxation (elf_writer.rs)

After the first pass, `relax_jumps()` attempts to shrink long-form jumps to
short form:

```
Long JMP:  E9 rel32   (5 bytes) --> Short JMP:  EB rel8  (2 bytes)  saves 3
Long Jcc:  0F 8x rel32 (6 bytes) --> Short Jcc: 7x rel8  (2 bytes)  saves 4
```

The algorithm is iterative because shrinking one jump can bring another
jump's target into short range:

1. Build a map of label positions within the current section.
2. For each un-relaxed jump, compute the displacement as if the jump were
   already in short form (end of instruction = offset + 2).
3. If the displacement fits in a signed byte [-128, +127], mark for
   relaxation.
4. Process relaxations **back-to-front** (so byte offsets remain valid):
   - Rewrite the opcode bytes in place (Jcc: `0x70+cc`; JMP: `0xEB`).
   - Remove the extra bytes via `data.drain()`.
   - Adjust all `label_positions`, `numeric_label_positions`, relocation
     offsets, and other jump offsets that fall after the shrunk instruction.
   - Remove the relocation entry for this jump (displacement is now inline).
5. Repeat from step 1 until no more relaxation occurs (fixed-point
   convergence).
6. After convergence, compute and patch the final short displacements.

### Step 5: Deferred Expression Resolution (elf_writer.rs)

Two categories of deferred work are resolved after jump relaxation:

**Deferred `.skip` expressions** -- `resolve_deferred_skips()` evaluates
complex `.skip` expressions that reference labels.  The expression evaluator
in the ELF writer supports:
- Arithmetic: `+`, `-`, `*`
- Comparison: `<`, `>` (returning -1 for true, 0 for false, per GNU as
  convention)
- Bitwise: `&`, `|`, `^`, `~`
- Parentheses and unary negation
- Symbol references (resolved from `label_positions`)

Skips are processed in reverse order within each section so that earlier
insertions do not invalidate the offsets of later ones.  After each insertion,
all subsequent label positions, numeric label positions, relocation offsets,
jump offsets, and deferred byte diffs in the same section are adjusted.

**Deferred byte-sized symbol diffs** -- `resolve_deferred_byte_diffs()`
resolves 1-byte and 2-byte `symbol_a - symbol_b` expressions.  These are
deferred because skip insertion can shift offsets.  Both symbols must be in the
same section; cross-section diffs are an error.

### Step 6: Post-Relaxation Updates (elf_writer.rs)

1. **Symbol value update** -- All symbol values are refreshed from
   `label_positions` (which were adjusted during relaxation and skip
   resolution).
2. **Size resolution** -- `.size` directives are resolved:
   - `Constant(v)` -- Used directly.
   - `CurrentMinusSymbol(start)` -- `section_data_len - start_offset`.
   - `SymbolDiff(end, start)` -- `end_offset - start_offset` from the
     updated label positions.

### Step 7: Internal Relocation Resolution (elf_writer.rs)

`resolve_internal_relocations()` resolves relocations that can be computed
without the linker:

- **`R_X86_64_PC8_INTERNAL`** -- Internal-only 8-bit PC-relative relocations
  for `jrcxz`/`loop` instructions.  Same-section targets are patched inline;
  these are never emitted to the ELF file.

- **`R_X86_64_PC32`** -- Same-section, local-symbol targets: the ELF formula
  `S + A - P` is applied and the result is patched into the section data.

- **`R_X86_64_PLT32`** -- Same-section, local-symbol targets: resolved
  identically to PC32.

- **`R_X86_64_32`** -- Absolute references to local symbols: `S + A` patched
  directly.

- **Symbol-difference relocations** -- Where both symbols in `.long a - b`
  are in the same section, the difference `offset(a) - offset(b)` is computed
  and patched as a constant.

Global and weak symbols are **never** resolved at this stage -- their
relocations are kept for the linker to handle symbol interposition and PLT
redirection correctly.

### Step 8: ELF Emission (elf_writer.rs)

`emit_elf()` resolves `.set` aliases into proper symbols, creates undefined
symbols for external references found in relocations, then converts the
internal data structures into the shared `ObjSection`/`ObjSymbol`/`ObjReloc`
format and delegates to `write_relocatable_object()` in `backend/elf.rs`.

**Internal label conversion:** Relocations referencing `.L*` labels are
converted to reference the parent section's section symbol, with the label's
offset baked into the addend.  This matches GAS behavior and keeps the symbol
table small.

**ELF file layout:**

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

**ELF configuration:**
- Class: ELFCLASS64
- Machine: EM_X86_64
- Flags: 0


## Instruction Encoding -- Supported Families

The encoder covers the following instruction categories (300+ match arms):

| # | Category | Instructions |
|---|----------|-------------|
| 1 | **Data movement** | mov (b/w/l/q), movabs, movsx (bl/bq/wl/wq/slq), movzx (bl/bq/wl/wq), lea (l/q), push, pop, xchg, cmpxchg, xadd, cmpxchg8b, cmpxchg16b |
| 2 | **Arithmetic** | add, sub, adc, sbb, and, or, xor, cmp, test (b/w/l/q), neg, not, inc, dec, imul (1/2/3-operand), mul, div, idiv |
| 3 | **Shifts/Rotates** | shl, shr, sar, rol, ror, rcl, rcr (b/w/l/q), shld, shrd (l/q) |
| 4 | **Sign extension** | cltq, cqto/cqo, cltd/cdq, cbw, cwd |
| 5 | **Byte swap** | bswap (l/q) |
| 6 | **Bit manipulation** | lzcnt, tzcnt, popcnt, bsf, bsr, bt, bts, btr, btc (l/q/w) |
| 7 | **Conditional set** | setcc (all 20+ conditions) |
| 8 | **Conditional move** | cmovcc (l/q, all conditions, plus suffix-less forms) |
| 9 | **Control flow** | jmp/jmpq, jcc (all conditions), call/callq, ret/retq, jrcxz, loop |
| 10 | **SSE/SSE2 scalar float** | movss, movsd, addss/sd, subss/sd, mulss/sd, divss/sd, sqrtss/sd, ucomisd, ucomiss, xorpd/ps, andpd/ps, minss/sd, maxss/sd |
| 11 | **SSE packed float** | addpd/ps, subpd/ps, mulpd/ps, divpd/ps, orpd/ps, andnpd/ps |
| 12 | **SSE data movement** | movaps, movdqa, movdqu, movlpd, movhpd, movapd, movups, movupd, movhlps, movlhps, movd, movmskpd, movmskps, movntps |
| 13 | **SSE2 integer SIMD** | paddw, psubw, paddd, psubd, pmulhw, pmaddwd, pcmpgtw/b, packssdw, packuswb, punpckl/h (bw/wd/dq/qdq), pmovmskb, pcmpeqb/d/w, pand, pandn, por, pxor, psubusb/w, paddusb/w, paddsb/w, pmuludq, pmullw, pmulld, pminub, pmaxub, pminsd, pmaxsd, pavgb/w, psadbw, paddb |
| 14 | **SSE shifts** | psllw/d/q, psrlw/d/q, psraw/d, pslldq, psrldq |
| 15 | **SSE shuffles** | pshufd, pshuflw, pshufhw, shufps, shufpd, palignr, pshufb |
| 16 | **SSE insert/extract** | pinsrb/w/d/q, pextrb/w/d/q |
| 17 | **SSE4.1** | blendvpd/ps, pblendvb, roundsd/ss/pd/ps, pblendw, blendpd/ps, dpps, dppd, ptest, pminsb, pminuw, pmaxsb, pmaxuw, pminud, pmaxud, pminsw, pmaxsw, phminposuw, packusdw, packsswb |
| 18 | **SSE unpacks** | unpcklpd/ps, unpckhpd/ps |
| 19 | **SSE non-temporal** | movnti, movntdq, movntpd, movntps |
| 20 | **SSE MXCSR** | ldmxcsr, stmxcsr |
| 21 | **AES-NI** | aesenc, aesenclast, aesdec, aesdeclast, aesimc, aeskeygenassist |
| 22 | **PCLMULQDQ** | pclmulqdq |
| 23 | **CRC32** | crc32 (b/w/l/q) |
| 24 | **SSE conversions** | cvtsd2ss, cvtss2sd, cvtsi2sdq/ssq/sdl/ssl, cvttsd2siq/ssiq/sd2sil/ss2sil, cvtsd2siq/sd2si/ss2siq/ss2si |
| 25 | **AVX data movement** | vmovdqa, vmovdqu, vmovaps, vmovapd, vmovups, vmovupd, vmovd, vmovq, vbroadcastss, vbroadcastsd |
| 26 | **AVX integer** | vpaddb/w/d/q, vpsubb/w/d/q, vpmullw, vpmulld, vpmuludq, vpcmpeqb/w/d, vpand, vpandn, vpor, vpxor, vpunpckl/h (bw/wd/dq/qdq), vpslldq, vpsrldq |
| 27 | **AVX shifts** | vpsllw/d/q, vpsrlw/d/q, vpsraw/d |
| 28 | **AVX shuffles** | vpshufd, vpshufb, vpalignr |
| 29 | **AVX float** | vaddpd/ps, vsubpd/ps, vmulpd/ps, vdivpd/ps, vxorpd/ps, vandpd/ps, vandnpd/ps, vorpd/ps |
| 30 | **AVX misc** | vpmovmskb, vpextrq, vpinsrq, vptest, vzeroupper, vzeroall |
| 31 | **BMI2** | shrx, shlx, sarx, rorx, bzhi, pext, pdep, mulx, andn, bextr (l/q plus suffix-less forms) |
| 32 | **x87 FPU** | fld, fstp, fldl, flds, fldt, fstpl, fstps, fstpt, fsts, fildq/ll/l/s, fisttpq/ll/l, fistpq/ll/l/s, fistl, fld1, fldl2e, fldlg2, fldln2, fldz, fldpi, fldl2t, faddp, fsubp, fsubrp, fmulp, fdivp, fdivrp, fchs, fadd, fmul, fsub, fdiv, faddl, fadds, fmull, fmuls, fsubl, fsubs, fdivl, fdivs, fabs, fsqrt, frndint, f2xm1, fscale, fpatan, fprem, fprem1, fyl2x, fyl2xp1, fptan, fsin, fcos, fxtract, fcomip, fucomip, fxch, fninit, fwait/wait, fnstcw/fstcw, fldcw, fnclex, fnstenv, fldenv, fnstsw (62 mnemonics total) |
| 33 | **String ops** | movsb/w/l/d/q, stosb/w/l/d/q, lodsb/w/d/q, scasb/w/d/q, cmpsb/w/d/q |
| 34 | **Standalone prefixes** | lock, rep/repe/repz, repne/repnz |
| 35 | **System** | syscall, sysenter, cpuid, rdtsc, rdtscp, int3 |
| 36 | **Misc** | nop, hlt, ud2, endbr64, pause, mfence, lfence, sfence, clflush |
| 37 | **Flags** | cld, std, clc, stc, cli, sti, sahf, lahf, pushf/pushfq, popf/popfq |
| 38 | **MMX** | emms, paddb (MMX form) |


## Supported Relocation Types

| Constant | Value | Usage |
|----------|-------|-------|
| `R_X86_64_NONE` | 0 | No relocation |
| `R_X86_64_64` | 1 | Absolute 64-bit address (`.quad symbol`) |
| `R_X86_64_PC32` | 2 | PC-relative 32-bit (RIP-relative addressing, `.long a - b`) |
| `R_X86_64_GOT32` | 3 | 32-bit GOT offset |
| `R_X86_64_PLT32` | 4 | PC-relative via PLT (`call`/`jmp` targets) |
| `R_X86_64_GOTPCREL` | 9 | GOT entry, PC-relative (`symbol@GOTPCREL(%rip)`) |
| `R_X86_64_32` | 10 | Absolute 32-bit unsigned (`.long symbol`) |
| `R_X86_64_32S` | 11 | Absolute 32-bit signed (symbol displacement in non-RIP memory) |
| `R_X86_64_TPOFF64` | 18 | 64-bit TLS offset from thread pointer |
| `R_X86_64_GOTTPOFF` | 22 | TLS GOT entry, PC-relative (`symbol@GOTTPOFF(%rip)`) |
| `R_X86_64_TPOFF32` | 23 | 32-bit TLS offset from thread pointer (`symbol@TPOFF`) |
| `R_X86_64_PC8_INTERNAL` | 0x8000_0001 | **Internal-only**: 8-bit PC-relative for `jrcxz`/`loop` (never emitted to ELF) |


## Numeric Labels

Labels like `1:`, `2:`, etc. can be defined multiple times.  Forward
references (`1f`) resolve to the next definition after the reference point;
backward references (`1b`) resolve to the most recent definition before the
reference point.

Numeric label positions are stored in:
```rust
numeric_label_positions: HashMap<String, Vec<(usize, u64)>>
```

The `resolve_numeric_label()` method scans the position list for the nearest
matching label in the correct direction within the same section.  Numeric
labels are also resolved by a shared pre-pass (`resolve_numeric_labels()`)
that converts them to unique names before the first pass begins.


## Well-Known Section Defaults

When switching to a section by its well-known name (without explicit flags),
the following defaults are applied:

| Section Name | Type | Flags |
|-------------|------|-------|
| `.text` | PROGBITS | ALLOC, EXECINSTR |
| `.data` | PROGBITS | ALLOC, WRITE |
| `.bss` | NOBITS | ALLOC, WRITE |
| `.rodata` | PROGBITS | ALLOC |
| `.tdata` | PROGBITS | ALLOC, WRITE, TLS |
| `.tbss` | NOBITS | ALLOC, WRITE, TLS |
| `.init` / `.fini` | PROGBITS | ALLOC, EXECINSTR |
| `.init_array` | INIT_ARRAY | ALLOC, WRITE |
| `.fini_array` | FINI_ARRAY | ALLOC, WRITE |
| `.text.*` | PROGBITS | ALLOC, EXECINSTR |
| `.data.*` | PROGBITS | ALLOC, WRITE |
| `.bss.*` | NOBITS | ALLOC, WRITE |
| `.rodata.*` | PROGBITS | ALLOC |
| `.note.*` | NOTE | (no flags) |

Sections created with explicit `.section name, "flags", @type` use the
provided flags/type instead of these defaults.


## Supported Directives

| Directive | Syntax | Purpose |
|-----------|--------|---------|
| `.section` | `.section name, "flags", @type` | Switch to named section with explicit attributes |
| `.text` | `.text` | Switch to `.text` section (shorthand) |
| `.data` | `.data` | Switch to `.data` section (shorthand) |
| `.bss` | `.bss` | Switch to `.bss` section (shorthand) |
| `.rodata` | `.rodata` | Switch to `.rodata` section (shorthand) |
| `.pushsection` | `.pushsection name, "flags", @type` | Push current section, switch to named section |
| `.popsection` / `.previous` | `.popsection` | Pop section stack, restore previous section |
| `.globl` / `.global` | `.globl name` | Mark symbol as global binding |
| `.weak` | `.weak name` | Mark symbol as weak binding |
| `.hidden` | `.hidden name` | Set symbol visibility to STV_HIDDEN |
| `.protected` | `.protected name` | Set symbol visibility to STV_PROTECTED |
| `.internal` | `.internal name` | Set symbol visibility to STV_INTERNAL |
| `.type` | `.type name, @function\|@object\|@tls_object\|@notype` | Set symbol type |
| `.size` | `.size name, expr` | Set symbol size (supports `.-name`, constants, `sym_a - sym_b`) |
| `.align` | `.align N` | Align to N-byte boundary (byte count) |
| `.p2align` | `.p2align N` | Align to 2^N-byte boundary (power of 2) |
| `.balign` | `.balign N` | Align to N-byte boundary (byte count) |
| `.byte` | `.byte val, ...` | Emit 1-byte values |
| `.short` / `.value` / `.2byte` | `.short val, ...` | Emit 2-byte values |
| `.long` / `.4byte` | `.long val, ...` | Emit 4-byte values |
| `.quad` / `.8byte` | `.quad val, ...` | Emit 8-byte values |
| `.zero` | `.zero N` | Emit N zero bytes |
| `.skip` | `.skip N, fill` | Skip N bytes (simple) or deferred expression |
| `.org` | `.org sym, fill` | Advance to position within section |
| `.asciz` / `.string` | `.asciz "str"` | Emit NUL-terminated string |
| `.ascii` | `.ascii "str"` | Emit string without NUL terminator |
| `.comm` | `.comm name, size, align` | Define common (BSS) symbol |
| `.set` | `.set alias, target` | Define symbol alias |
| `.rept` / `.endr` | `.rept N` ... `.endr` | Repeat block of lines N times |
| `.cfi_startproc` | `.cfi_startproc` | CFI: start of procedure (parsed, not emitted) |
| `.cfi_endproc` | `.cfi_endproc` | CFI: end of procedure (parsed, not emitted) |
| `.cfi_def_cfa_offset` | `.cfi_def_cfa_offset N` | CFI: CFA offset (parsed, not emitted) |
| `.cfi_def_cfa_register` | `.cfi_def_cfa_register reg` | CFI: CFA register (parsed, not emitted) |
| `.cfi_offset` | `.cfi_offset reg, N` | CFI: register saved at offset (parsed, not emitted) |
| `.file` | `.file N "name"` | Debug file directive (parsed, ignored) |
| `.loc` | `.loc filenum line col` | Debug location (parsed, ignored) |
| `.option` | `.option ...` | RISC-V directive (ignored on x86) |


## Key Design Decisions and Trade-offs

### 1. Subset, Not Full GAS

The assembler implements what the compiler's codegen actually produces, plus
what hand-written assembly in musl and similar projects requires.  This
dramatically simplifies the implementation -- no macro system, no conditional
assembly, no `.if`/`.else`.  Unknown directives are silently ignored with
`AsmItem::Empty`, which provides forward compatibility as the codegen evolves.

### 2. Suffix Inference for Hand-Written Assembly

Hand-written assembly (e.g., musl's `.s` files) often omits the AT&T size
suffix when it can be inferred from register operands.  The `infer_suffix()`
function handles this by looking at the first register operand and appending
the appropriate suffix (`b`/`w`/`l`/`q`).  Only a curated whitelist of
mnemonics is eligible -- this prevents incorrect inference on mnemonics where
the trailing letter is part of the name (e.g., `movsd`, `comiss`).

### 3. Always-Long Encoding with Post-Hoc Relaxation

Instructions are initially encoded in their longest form (e.g., `jmp` always
uses `E9 rel32`).  The relaxation pass then shrinks eligible jumps to short
form.  This two-phase approach avoids the complexity of an iterative
encode-measure-re-encode loop during the first pass.

### 4. Iterative Jump Relaxation

The relaxation loop runs until convergence because shrinking one jump can
bring another jump's target into short range.  Relaxations are processed
back-to-front within each iteration to maintain valid byte offsets.  This is
the same approach used by production linkers (ld, lld).

### 5. Deferred Expression Evaluation

Complex `.skip` expressions (such as those used in the Linux kernel
alternatives framework) cannot be evaluated during the first pass because
they reference labels that may not yet be defined, and whose positions may
shift during jump relaxation.  These are deferred and evaluated after both
relaxation and skip insertion are complete.  The deferred evaluator
implements a recursive-descent expression parser with proper operator
precedence.

### 6. Lazy Symbol Attribute Application

`.globl`, `.type`, `.hidden`, etc. are collected as "pending" attributes and
only applied when a label definition (`name:`) is encountered.  This handles
the common GAS pattern where attributes appear before the label:

```asm
.globl main
.type main, @function
main:
```

### 7. Internal Labels via Section Symbols

Labels starting with `.` (e.g., `.LBB0`, `.Lstr0`) are treated as
section-local.  In the ELF output, they are not emitted as named symbols.
Instead, relocations referencing them use the parent section's section symbol,
with the addend adjusted to include the label's offset.  This matches GAS
behavior and keeps the symbol table small.

### 8. CFI Directives Parsed but Not Emitted

The assembler parses `.cfi_startproc`, `.cfi_endproc`, `.cfi_def_cfa_offset`,
etc. into `CfiDirective` variants but does not generate `.eh_frame` section
data.  This is acceptable because the built-in linker does not require unwind
information for basic compilation, and adding `.eh_frame` generation would
substantially increase complexity.

### 9. No Instruction Shortening Beyond Jumps

The encoder does not attempt to select shorter instruction forms for
arithmetic (e.g., `add $1, %rax` could use `inc %rax`).  It faithfully
encodes what the codegen emits.  The one exception is immediate-size
optimization: ALU instructions with small immediates automatically use the
sign-extended imm8 form (`0x83` instead of `0x81`), and `movq` with
32-bit-range immediates uses `C7` instead of `movabs`.

### 10. Two Separate String Tables

The ELF file uses distinct `.strtab` (symbol names) and `.shstrtab` (section
names) string tables, each built by the same `StringTable` helper.  This is
the standard ELF convention and simplifies the section header linkage.

### 11. VEX Prefix Construction for AVX/BMI2

AVX and BMI2 instructions use VEX encoding rather than legacy prefixes.  The
encoder constructs the appropriate 2-byte or 3-byte VEX prefix based on the
instruction requirements (map select, operand size, vector length, register
extension bits).  This allows a single unified encoder to handle both legacy
SSE and VEX-encoded AVX instructions.
