# Driver

Orchestrates the compilation pipeline from command-line arguments through to final output.

## Module Layout

| File | Purpose |
|------|---------|
| `driver.rs` | `Driver` struct, `new()`, `run()`, compilation pipeline (`compile_to_assembly`), run modes |
| `cli.rs` | GCC-compatible CLI argument parsing (`parse_cli_args`) |
| `external_tools.rs` | External tool invocation: assembler, linker, GCC fallback, dependency files |
| `file_types.rs` | Input file classification by extension or magic bytes |

## Why This Split

The driver was originally a single monolithic file. The split follows natural seams:

- **CLI parsing** is self-contained: it reads `args`, mutates `Driver` fields, and returns. No coupling to the compilation pipeline.
- **External tool invocation** centralizes all `std::process::Command` usage. If the compiler later gains a native assembler/linker, only this file changes.
- **File type detection** is pure logic with no state dependencies.

## Compilation Pipeline

```
parse_cli_args()  ->  run()  ->  compile_to_assembly()
                        |
                        +-- PreprocessOnly: preprocess -> stdout/.i
                        +-- AssemblyOnly:   compile -> .s
                        +-- ObjectOnly:     compile -> assemble -> .o
                        +-- Full:           compile -> assemble -> link -> executable

compile_to_assembly():
  read_source -> preprocess -> lex -> parse -> sema -> lower -> mem2reg -> optimize -> phi_eliminate -> codegen
```

## CLI Compatibility

The driver accepts a broad set of GCC flags, silently ignoring unrecognized ones (e.g., unknown `-f` flags, `-m` arch flags). This allows using ccc as a drop-in replacement for GCC in build systems. Target architecture is selected by the binary name (ccc-arm, ccc-riscv, ccc-i686, or default x86-64).
