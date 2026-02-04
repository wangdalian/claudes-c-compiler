# AArch64 Built-in Linker -- Design Document

## Overview

The built-in AArch64 linker links ELF64 relocatable object files (`.o`) and
static archives (`.a`) into statically-linked ELF64 executables for AArch64
Linux.  It replaces the external `ld` dependency when
the `gcc_linker` Cargo feature is not enabled (the default), making the
compiler fully self-hosting.

The linker implements the complete static linking pipeline: ELF object parsing,
archive member extraction, symbol resolution, section merging, virtual address
layout, GOT construction, TLS handling, IFUNC support, relocation application,
and final ELF executable emission.

The implementation spans roughly 2,060 lines of Rust across three files.

```
             AArch64 Built-in Linker
  ============================================================

  .o files    .a archives    -l libraries
       \           |           /
        v          v          v
  +------------------------------------------+
  |           elf.rs  (~70 lines)            |
  |   Type aliases + thin wrappers to        |
  |   linker_common; AArch64 reloc consts    |
  +------------------------------------------+
               |
               |  Vec<ElfObject>, HashMap<String, GlobalSymbol>
               v
  +------------------------------------------+
  |           mod.rs  (1,105 lines)          |
  |   Orchestrator: file loading, archive    |
  |   resolution, section merging, layout,   |
  |   GOT/IPLT construction, ELF emission   |
  +------------------------------------------+
               |
               |  output buffer + section map
               v
  +------------------------------------------+
  |           reloc.rs  (526 lines)          |
  |   Relocation Application: 30+ reloc     |
  |   types, TLS relaxation, GOT refs       |
  +------------------------------------------+
               |
               v
        ELF64 executable on disk
```


---

## Public Entry Point

```rust
// mod.rs
pub fn link(
    object_files: &[&str],     // Paths to .o files from the compiler
    output_path: &str,          // Output executable path
    user_args: &[String],       // Additional flags: -L, -l, -static, -nostdlib, -Wl,...
) -> Result<(), String>
```


---

## Stage 1: ELF Parsing (`elf.rs` / `linker_common`)

### Purpose

Read and decode ELF64 relocatable object files, static archives, and minimal
linker scripts.  The actual parsing logic lives in the shared `linker_common`
module; `elf.rs` provides AArch64-specific relocation constants and re-exports
shared types under local names via type aliases.

### Key Data Structures

All ELF64 types are defined in `linker_common` and re-exported:

| Type | Alias | Role |
|------|-------|------|
| `Elf64Object` | `ElfObject` | A fully parsed object file: sections, symbols, raw section data, relocations indexed by section. |
| `Elf64Section` | `SectionHeader` | Parsed `Elf64_Shdr`: name, type, flags, offset, size, link, info, alignment, entsize. |
| `Elf64Symbol` | `Symbol` | Parsed `Elf64_Sym`: name, info (binding + type), other (visibility), shndx, value, size. |
| `Elf64Rela` | `Rela` | Parsed `Elf64_Rela`: offset, sym_idx, rela_type, addend. |

### Object Parsing (`parse_object`)

Delegates to `linker_common::parse_elf64_object(data, source_name, EM_AARCH64)`.

### Archive and Linker Script Parsing

Archive parsing (`parse_archive_members`) and linker script parsing
(`parse_linker_script`) are provided by the shared `crate::backend::elf` module.


---

## Stage 2: Orchestration (`mod.rs`)

### Purpose

This is the main linker driver.  It coordinates file loading, symbol
resolution, section merging, address layout, GOT/IPLT construction, and
ELF executable emission.

### Key Data Structures

| Type | Role |
|------|------|
| `OutputSection` | A merged output section: name, type, flags, alignment, list of `InputSection` references, merged data, assigned virtual address and file offset. |
| `InputSection` | Reference to one input section: object index, section index, output offset within the merged section, size. |
| `GlobalSymbol` | A resolved global symbol: final value (address), size, info byte, defining object index, section index. |

### Constants

```
BASE_ADDR  = 0x400000     -- Base virtual address for the executable
PAGE_SIZE  = 0x10000      -- 64 KB (AArch64 linker page alignment)
```

### CRT Object Discovery

The linker automatically locates C runtime startup objects:

| Object | Purpose | Search Paths |
|--------|---------|-------------|
| `crt1.o` | Entry point (`_start`) | `/usr/aarch64-linux-gnu/lib`, `/usr/lib/aarch64-linux-gnu` |
| `crti.o` | Init function prologue | same |
| `crtbeginT.o` | Static C++ init | `/usr/lib/gcc-cross/aarch64-linux-gnu/{13,12,11}`, `/usr/lib/gcc/aarch64-linux-gnu/{13,12,11}` |
| `crtend.o` | C++ fini epilogue | GCC paths |
| `crtn.o` | Fini function epilogue | CRT paths |

CRT objects are loaded in order: `crt1.o`, `crti.o`, `crtbeginT.o` (before
user objects), then `crtend.o`, `crtn.o` (after).  Skipped if `-nostdlib` is
passed.

### Linking Algorithm -- Step by Step

```
link(object_files, output_path, user_args):

  1. ARGUMENT PARSING
     Parse user_args for -L (library paths), -l (libraries),
     -static, -nostdlib, -Wl,... options.

  2. FILE LOADING
     a. Load CRT objects (before): crt1.o, crti.o, crtbeginT.o
     b. Load user object files from object_files[]
     c. Load objects/archives/libraries from user_args
     d. Load CRT objects (after): crtend.o, crtn.o
     e. Group-load default libraries: libgcc.a, libgcc_eh.a, libc.a
        (iterate until no new symbols resolved -- handles circular deps)

  3. SYMBOL RESOLUTION
     register_symbols() for each loaded object:
       - Skip FILE and SECTION symbols
       - Defined symbols: insert or replace if existing is
         undefined or weak-vs-global
       - COMMON symbols: insert if not already present
       - Undefined symbols: insert placeholder if not present

  4. UNRESOLVED SYMBOL CHECK
     Error on undefined non-weak symbols, excluding well-known
     linker-defined names (__bss_start, _GLOBAL_OFFSET_TABLE_, etc.)

  5. SECTION MERGING (merge_sections)
     a. Map input section names to output names:
        .text.*, .text       -> .text
        .data.rel.ro*        -> .data.rel.ro
        .data.*, .data       -> .data
        .rodata.*, .rodata   -> .rodata
        .bss.*, .bss         -> .bss
        .tbss.*, .tdata.*    -> .tbss, .tdata
        .init_array.*        -> .init_array
        .fini_array.*        -> .fini_array
        .gcc_except_table.*  -> .gcc_except_table
        .eh_frame.*          -> .eh_frame
     b. For each allocatable input section, append to matching output
        section, recording alignment and size
     c. Calculate output offsets within each merged section
     d. Sort output sections: RO -> Exec -> RW(progbits) -> RW(nobits)
     e. Build section_map: (obj_idx, sec_idx) -> (out_idx, offset)

  6. COMMON SYMBOL ALLOCATION (allocate_common_symbols)
     Allocate SHN_COMMON symbols into .bss with proper alignment.

  7. ADDRESS LAYOUT AND EMISSION (emit_executable)
     Detailed below.
```

### Memory Layout

The linker produces a two-segment layout:

```
  Virtual Address Space
  ====================================================================

  0x400000 +========================+  ----+
           |  ELF Header (64 B)     |      |
           |  Program Headers       |      |
           +------------------------+      |
           |  .text                  |      |  LOAD segment 1
           |  (executable code)      |      |  RX (Read + Execute)
           +------------------------+      |
           |  .rodata               |      |
           |  (read-only data)       |      |
           +------------------------+      |
           |  .gcc_except_table     |      |
           |  .eh_frame             |      |
           +------------------------+      |
           |  [IPLT stubs]          |      |  (in RX padding gap)
           +========================+  ----+
           |  (page alignment gap)  |  <- 64 KB aligned
           +========================+  ----+
           |  .tdata                |      |
           |  (TLS initialized)     |      |
           +------------------------+      |
           |  .init_array           |      |
           |  .fini_array           |      |
           +------------------------+      |
           |  .data.rel.ro          |      |  LOAD segment 2
           |  .data                 |      |  RW (Read + Write)
           +------------------------+      |
           |  .got                  |      |  (built by linker)
           +------------------------+      |
           |  [IPLT GOT slots]     |      |
           |  [.rela.iplt entries]  |      |
           +========================+  ----+
           |  .bss                  |  (no file space, only memsize)
           |  .tbss                 |
           +========================+

  Program Headers:
    LOAD  RX: file offset 0, vaddr BASE_ADDR, filesz=rx_filesz
    LOAD  RW: file offset rw_page_offset, vaddr=rw_page_addr
    TLS:  .tdata + .tbss (if present)
    GNU_STACK: RW, no exec
```

### GOT (Global Offset Table) Construction

The linker builds a GOT for two purposes:

1. **Regular GOT entries** (`R_AARCH64_ADR_GOT_PAGE` / `R_AARCH64_LD64_GOT_LO12_NC`):
   8-byte slots containing the absolute address of the target symbol.

2. **TLS IE GOT entries** (`R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21` /
   `R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC`): 8-byte slots containing the
   TP-relative offset of the TLS variable (computed as
   `sym_addr - tls_base + 16` per AArch64 Variant 1 TLS).

The `collect_got_symbols()` function in `reloc.rs` scans all relocations
to determine which symbols need GOT entries, and what kind (`Regular` or
`TlsIE`).  GOT entries are allocated in the RW segment, 8-byte aligned.

### IFUNC / IPLT Support

The linker handles `STT_GNU_IFUNC` symbols (indirect functions whose runtime
address is determined by a resolver function):

1. **Identify IFUNC symbols** in the global symbol table.
2. **Allocate IPLT GOT slots** (one 8-byte slot per IFUNC) in the RW segment.
3. **Generate `.rela.iplt` entries** with `R_AARCH64_IRELATIVE` relocations
   pointing to the resolver function.
4. **Generate IPLT PLT stubs** in the RX gap between text and data segments.
   Each stub is 16 bytes:
   ```
   ADRP  x16, page_of(got_slot)
   LDR   x17, [x16, #lo12(got_slot)]
   BR    x17
   NOP
   ```
5. **Redirect IFUNC symbol addresses** to point to the PLT stub instead of
   the resolver.  The symbol type is changed from `STT_GNU_IFUNC` to
   `STT_FUNC`.

### Linker-Defined Symbols

The following symbols are automatically provided:

| Symbol | Value |
|--------|-------|
| `__dso_handle` | `BASE_ADDR` |
| `_DYNAMIC` | 0 (no dynamic section in static executables) |
| `_GLOBAL_OFFSET_TABLE_` | GOT base address |
| `__init_array_start` / `__init_array_end` | `.init_array` bounds |
| `__fini_array_start` / `__fini_array_end` | `.fini_array` bounds |
| `__preinit_array_start` / `__preinit_array_end` | Same as init_array start |
| `__ehdr_start` | `BASE_ADDR` |
| `__executable_start` | `BASE_ADDR` |
| `_etext` / `etext` | End of text (RX) segment |
| `__data_start` / `data_start` | Start of RW data segment |
| `_init` / `_fini` | Address of `.init` / `.fini` sections |
| `__rela_iplt_start` / `__rela_iplt_end` | IRELATIVE relocation table bounds |
| `__bss_start` / `_edata` | BSS start address |
| `_end` / `__end` | BSS end address |


---

## Stage 3: Relocation Application (`reloc.rs`)

### Purpose

After all sections have been laid out and symbol addresses are known, apply
every relocation from every input object to the output buffer.  This module
also handles TLS model relaxation and GOT-indirect references.

### Key Data Structures

| Type | Role |
|------|------|
| `TlsInfo` | TLS segment base address and total size. |
| `GotInfo` | GOT base address and a map of symbol keys to entry indices. |
| `GotEntryKind` | Whether a GOT entry is `Regular` (absolute address) or `TlsIE` (TP offset). |

### Symbol Resolution (`resolve_sym`)

```
resolve_sym(obj_idx, sym, globals, section_map, output_sections, bss_addr):
  if sym is STT_SECTION:
    return output_sections[mapped_section].addr + section_offset
  if sym.name is known linker symbol (__bss_start, _end, ...):
    return bss boundary address
  if sym.name is in globals and defined:
    return global value
  if sym is weak and undefined:
    return 0
  if sym is SHN_ABS:
    return sym.value
  otherwise:
    return mapped section addr + section offset + sym.value
```

### Supported Relocation Types

The linker handles 30+ AArch64 relocation types, organized by category:

#### Absolute Relocations

| Type | ELF # | Formula | Usage |
|------|-------|---------|-------|
| `R_AARCH64_ABS64` | 257 | S + A | 64-bit data pointer |
| `R_AARCH64_ABS32` | 258 | S + A | 32-bit data pointer |
| `R_AARCH64_ABS16` | 259 | S + A | 16-bit data value |

#### PC-Relative Relocations

| Type | ELF # | Formula | Usage |
|------|-------|---------|-------|
| `R_AARCH64_PREL64` | 260 | S + A - P | 64-bit PC-relative |
| `R_AARCH64_PREL32` | 261 | S + A - P | 32-bit PC-relative (jump tables) |
| `R_AARCH64_PREL16` | 262 | S + A - P | 16-bit PC-relative |

#### Page-Relative and Immediate Relocations

| Type | ELF # | Formula | Usage |
|------|-------|---------|-------|
| `R_AARCH64_ADR_PREL_PG_HI21` | 275 | Page(S+A) - Page(P) | ADRP instruction |
| `R_AARCH64_ADR_PREL_LO21` | 274 | S + A - P | ADR instruction |
| `R_AARCH64_ADD_ABS_LO12_NC` | 277 | (S+A) & 0xFFF | ADD :lo12: |
| `R_AARCH64_LDST8_ABS_LO12_NC` | 278 | (S+A) & 0xFFF | Byte load/store |
| `R_AARCH64_LDST16_ABS_LO12_NC` | 284 | (S+A) & 0xFFF >> 1 | Halfword load/store |
| `R_AARCH64_LDST32_ABS_LO12_NC` | 285 | (S+A) & 0xFFF >> 2 | Word load/store |
| `R_AARCH64_LDST64_ABS_LO12_NC` | 286 | (S+A) & 0xFFF >> 3 | Doubleword load/store |
| `R_AARCH64_LDST128_ABS_LO12_NC` | 299 | (S+A) & 0xFFF >> 4 | Quadword load/store |

#### Branch Relocations

| Type | ELF # | Formula | Usage |
|------|-------|---------|-------|
| `R_AARCH64_CALL26` | 283 | (S+A-P) >> 2 | BL instruction (26-bit) |
| `R_AARCH64_JUMP26` | 282 | (S+A-P) >> 2 | B instruction (26-bit) |
| `R_AARCH64_CONDBR19` | 280 | (S+A-P) >> 2 | Conditional branch (19-bit) |
| `R_AARCH64_TSTBR14` | 279 | (S+A-P) >> 2 | Test-and-branch (14-bit) |

Special: when a `CALL26`/`JUMP26` target resolves to address 0 (undefined
weak symbol), the instruction is replaced with `NOP` (0xd503201f).

#### MOVW Relocations

| Type | ELF # | Formula | Usage |
|------|-------|---------|-------|
| `R_AARCH64_MOVW_UABS_G0[_NC]` | 263/264 | (S+A) & 0xFFFF | MOVZ/MOVK bits [15:0] |
| `R_AARCH64_MOVW_UABS_G1_NC` | 265 | (S+A) >> 16 & 0xFFFF | MOVK bits [31:16] |
| `R_AARCH64_MOVW_UABS_G2_NC` | 266 | (S+A) >> 32 & 0xFFFF | MOVK bits [47:32] |
| `R_AARCH64_MOVW_UABS_G3` | 267 | (S+A) >> 48 & 0xFFFF | MOVK bits [63:48] |

#### GOT Relocations

| Type | ELF # | Description |
|------|-------|-------------|
| `R_AARCH64_ADR_GOT_PAGE` | 311 | ADRP to page containing GOT entry |
| `R_AARCH64_LD64_GOT_LO12_NC` | 312 | LDR from GOT entry (low 12 bits) |

In static linking, the GOT is a real data structure in the RW segment
populated at link time (not lazily at runtime).

#### TLS Local Exec (LE) Relocations

Used when the TLS variable is in the executable itself (most common in
static linking).  The TP (Thread Pointer) offset is computed as:

```
tp_offset = (sym_addr - tls_start_addr) + 16    // AArch64 Variant 1
```

| Type | ELF # | Description |
|------|-------|-------------|
| `R_AARCH64_TLSLE_ADD_TPREL_HI12` | 549 | ADD, high 12 bits of TP offset |
| `R_AARCH64_TLSLE_ADD_TPREL_LO12[_NC]` | 550/551 | ADD, low 12 bits of TP offset |
| `R_AARCH64_TLSLE_MOVW_TPREL_G0[_NC]` | 544/545 | MOVZ/MOVK, bits [15:0] |
| `R_AARCH64_TLSLE_MOVW_TPREL_G1[_NC]` | 546/547 | MOVK, bits [31:16] |
| `R_AARCH64_TLSLE_MOVW_TPREL_G2` | 548 | MOVK, bits [47:32] |

#### TLS Initial Exec (IE) via GOT

Instead of relaxing ADRP+LDR to MOVZ+MOVK (which can break if different
registers are used), the linker uses real GOT entries pre-populated with
TP offsets:

| Type | ELF # | Description |
|------|-------|-------------|
| `R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21` | 541 | ADRP to GOT page holding TP offset |
| `R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC` | 542 | LDR from GOT entry |

#### TLS Descriptor (TLSDESC) Relaxation to LE

For static linking, TLSDESC sequences are relaxed to direct TP-offset
computation:

| Type | ELF # | Relaxation |
|------|-------|------------|
| `R_AARCH64_TLSDESC_ADR_PAGE21` | 562 | ADRP -> MOVZ Xd, #tprel_g1, LSL #16 |
| `R_AARCH64_TLSDESC_LD64_LO12` | 563 | LDR -> MOVK Xd, #tprel_lo |
| `R_AARCH64_TLSDESC_ADD_LO12` | 564 | ADD -> NOP |
| `R_AARCH64_TLSDESC_CALL` | 569 | BLR -> NOP |

#### TLS General Dynamic (GD) Relaxation to LE

| Type | ELF # | Relaxation |
|------|-------|------------|
| `R_AARCH64_TLSGD_ADR_PAGE21` | 513 | ADRP -> MOVZ Xd, #tprel_g1, LSL #16 |
| `R_AARCH64_TLSGD_ADD_LO12_NC` | 514 | ADD -> MOVK Xd, #tprel_lo |

### Instruction Patching Helpers

The relocation module includes helpers that patch individual instruction
fields without disturbing other bits:

| Helper | Field Modified |
|--------|---------------|
| `encode_adrp()` | immhi[23:5] and immlo[30:29] of ADRP |
| `encode_adr()` | immhi[23:5] and immlo[30:29] of ADR |
| `encode_add_imm12()` | imm12[21:10] of ADD immediate |
| `encode_ldst_imm12()` | imm12[21:10] of LDR/STR, scaled by access size |
| `encode_movw()` | imm16[20:5] of MOVZ/MOVK |


---

## Archive and Library Handling

### Archive Loading Strategy

Archives are loaded using **selective extraction with iterative resolution**,
matching the behavior of traditional `ld --start-group`:

```
load_archive(data, archive_path):
  1. Parse all archive members into temporary ElfObject list
  2. Filter to AArch64 ELF objects only
  3. Iterate until stable:
     for each unloaded member:
       if member defines any currently-undefined global symbol:
         load the member (register its symbols, add to objects list)
         set changed = true
  4. Discard any remaining unextracted members
```

### Default Library Group Loading

When not `-nostdlib`, the linker loads `libgcc.a`, `libgcc_eh.a` (if
`-static`), and `libc.a` in a group-loading loop.  This handles circular
dependencies between these libraries (e.g., libc referencing libgcc exception
handling, and libgcc referencing libc memory allocation):

```
lib_group = [libgcc.a, libgcc_eh.a, libc.a]
repeat:
  prev_count = objects.len()
  for each archive in lib_group:
    load_archive(archive)  // only extracts members that resolve undefs
  if objects.len() == prev_count:
    break   // stable -- no new members pulled in
```

### Library Resolution (`resolve_lib`)

Libraries specified with `-l` are searched in order across all library paths:

1. User-specified `-L` paths
2. CRT paths (`/usr/aarch64-linux-gnu/lib`, `/usr/lib/aarch64-linux-gnu`)
3. GCC paths (versions 13, 12, 11, both native and cross)

For each directory, `lib<name>.a` is preferred over `lib<name>.so`.
The special `-l:filename` syntax searches for an exact filename.


---

## Design Decisions and Trade-offs

### 1. Static Linking Only

The linker produces static executables (`ET_EXEC`) with no dynamic linking
support.  This simplifies the implementation significantly -- no `.dynamic`
section, no PLT/GOT lazy binding, no `DT_*` tags, no `.interp` section.
Shared libraries (`.so` files) are silently skipped.

### 2. Two-Segment Layout

The output uses exactly two `PT_LOAD` segments (RX and RW) plus optional
TLS and GNU_STACK segments.  This is the minimal viable layout.  The
64 KB page alignment (`PAGE_SIZE = 0x10000`) accommodates AArch64 systems
with either 4 KB or 64 KB page sizes.

### 3. Real GOT for All GOT-Based Relocations

Rather than relaxing `ADRP+LDR` GOT sequences to `ADRP+ADD` (which would
save memory but requires verifying instruction sequences), the linker
maintains a real GOT in the RW segment.  GOT entries are populated at link
time with final addresses.  This is conservative but correct -- the `LDR`
instruction genuinely loads from memory, and converting it to `ADD` would
require instruction replacement.

### 4. TLS IE via GOT (Not MOVZ/MOVK Relaxation)

TLS Initial Exec relocations use real GOT entries containing pre-computed
TP offsets, rather than relaxing to `MOVZ+MOVK` instruction sequences.  The
relaxation approach was found to be fragile because the ADRP and LDR
instructions may use different registers, and the relaxed MOVZ+MOVK must
target the same register as the original LDR destination.

### 5. TLSDESC and TLSGD Relaxation to LE

For static linking, both TLSDESC and General Dynamic TLS access patterns are
relaxed to Local Exec.  The TLSDESC 4-instruction sequence
(ADRP + LDR + ADD + BLR) is replaced with (MOVZ + MOVK + NOP + NOP).
This is correct because in a static executable, all TLS variables are in the
executable's own TLS block.

### 6. IFUNC Handling via IPLT

GNU IFUNC symbols (where the symbol resolves to a "resolver" function that
returns the actual implementation address at runtime) are handled by
generating IPLT stubs and IRELATIVE relocations.  The glibc startup code
processes these relocations to fill the GOT slots with the actual function
addresses returned by the resolvers.

### 7. No Section Headers in Output

The output executable contains no section header table (`e_shnum = 0`).
This is valid per the ELF specification (section headers are optional for
executables) and reduces output size.  Tools like `objdump -d` still work
by following program headers.

### 8. Diagnostic Support

Setting `LINKER_DEBUG=1` enables verbose tracing of object loading, symbol
resolution, section layout, GOT allocation, and final addresses.  This is
invaluable for debugging linking failures.


---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | 1,105 | Public API, file loading, archive handling, CRT discovery, section merging, address layout, GOT/IPLT construction, ELF executable emission |
| `elf.rs` | ~70 | AArch64 relocation constants; type aliases and thin wrappers delegating to `linker_common` for ELF64 parsing |
| `reloc.rs` | 526 | Relocation application (30+ types), TLS relaxation, GOT/TLS-IE references, instruction field patching helpers |
| **Total** | **~1,700** | |
