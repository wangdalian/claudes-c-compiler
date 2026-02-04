#!/usr/bin/env python3
"""Compare .o files produced by GCC assembler vs built-in assembler.

Usage:
    python3 scripts/compare_asm.py --arch arm [--before gcc] [--after builtin] [file.c ...]

This script compiles C files to .s, then assembles them with both the 'before'
assembler (default: GCC cross-compiler) and the 'after' assembler (default:
MY_ASM=builtin), and compares the .text section bytes, symbols, and relocations.

If no files are given, it uses a built-in set of test snippets.
"""

import argparse
import os
import subprocess
import sys
import tempfile

ARCH_CONFIG = {
    "arm": {
        "compiler": "ccc-arm",
        "objdump": "aarch64-linux-gnu-objdump",
        "readelf": "readelf",  # host readelf can read cross-arch ELF
    },
    "riscv": {
        "compiler": "ccc-riscv",
        "objdump": "riscv64-linux-gnu-objdump",
        "readelf": "readelf",
    },
    "x86": {
        "compiler": "ccc-x86",
        "objdump": "objdump",
        "readelf": "readelf",
    },
    "i686": {
        "compiler": "ccc-i686",
        "objdump": "objdump",
        "readelf": "readelf",
    },
}

DEFAULT_TEST_SNIPPETS = [
    # Simple function
    ("simple_add", "int add(int a, int b) { return a + b; }"),
    # Branches
    ("branches", """
int max(int a, int b) {
    if (a > b) return a;
    return b;
}
"""),
    # Function call
    ("calls", """
int square(int x) { return x * x; }
int main() { return square(5); }
"""),
    # Global variables
    ("globals", """
int g = 42;
int get_g() { return g; }
"""),
    # Loops
    ("loop", """
int sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += i;
    return s;
}
"""),
]


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return result.returncode, result.stdout, result.stderr


def get_text_bytes(objdump, obj_path):
    """Extract .text section hex bytes using objdump -d."""
    rc, stdout, stderr = run([objdump, "-d", "-j", ".text", obj_path])
    if rc != 0:
        return None, stderr
    return stdout, None


def get_readelf_info(readelf, obj_path):
    """Get section, symbol, and relocation info."""
    rc, sections, _ = run([readelf, "-S", obj_path])
    rc2, symbols, _ = run([readelf, "-s", obj_path])
    rc3, relocs, _ = run([readelf, "-r", obj_path])
    return sections, symbols, relocs


def compare_files(arch_config, compiler_path, c_file, before_env, after_env, verbose=False):
    """Compare .o produced by two assembler configurations."""
    compiler = os.path.join(compiler_path, arch_config["compiler"])
    objdump = arch_config["objdump"]
    readelf = arch_config["readelf"]

    with tempfile.TemporaryDirectory() as tmpdir:
        before_o = os.path.join(tmpdir, "before.o")
        after_o = os.path.join(tmpdir, "after.o")

        # Compile with 'before' assembler
        env_before = dict(os.environ)
        env_before.update(before_env)
        # Remove MY_ASM if it was "builtin" and before is gcc
        if "MY_ASM" not in before_env:
            env_before.pop("MY_ASM", None)
        rc1, _, err1 = run([compiler, "-c", c_file, "-o", before_o], env=env_before)
        if rc1 != 0:
            print(f"  BEFORE assembly failed: {err1.strip()}")
            return False

        # Compile with 'after' assembler
        env_after = dict(os.environ)
        env_after.update(after_env)
        rc2, _, err2 = run([compiler, "-c", c_file, "-o", after_o], env=env_after)
        if rc2 != 0:
            print(f"  AFTER assembly failed: {err2.strip()}")
            return False

        # Compare .text bytes
        before_text, err = get_text_bytes(objdump, before_o)
        if before_text is None:
            # Try host objdump
            before_text, err = get_text_bytes("objdump", before_o)
        after_text, err2 = get_text_bytes(objdump, after_o)
        if after_text is None:
            after_text, err2 = get_text_bytes("objdump", after_o)

        # Compare readelf info
        before_sec, before_sym, before_rel = get_readelf_info(readelf, before_o)
        after_sec, after_sym, after_rel = get_readelf_info(readelf, after_o)

        # Extract just the hex bytes from objdump output
        def extract_hex(text):
            if text is None:
                return []
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if ':' in line and not line.endswith(':'):
                    # Lines like "   0:    d503201f    nop"
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        hex_part = parts[1].strip()
                        lines.append(hex_part)
            return lines

        before_hex = extract_hex(before_text)
        after_hex = extract_hex(after_text)

        match = before_hex == after_hex
        total = max(len(before_hex), len(after_hex))
        matching = sum(1 for a, b in zip(before_hex, after_hex) if a == b)

        if verbose or not match:
            print(f"  Text section: {matching}/{total} instructions match", end="")
            if not match:
                print(" [MISMATCH]")
                # Show first few differences
                for i, (a, b) in enumerate(zip(before_hex, after_hex)):
                    if a != b:
                        print(f"    instr {i}: before={a}  after={b}")
                        if i > 10:
                            print(f"    ... and more differences")
                            break
            else:
                print(" [OK]")

        return match


def main():
    parser = argparse.ArgumentParser(description="Compare assembler output")
    parser.add_argument("--arch", required=True, choices=["arm", "riscv", "x86", "i686"],
                        help="Target architecture")
    parser.add_argument("--before", default="gcc",
                        help="'before' assembler: 'gcc' (default) or env var settings")
    parser.add_argument("--after", default="builtin",
                        help="'after' assembler: 'builtin' (default) or env var settings")
    parser.add_argument("--compiler-path", default="target/release",
                        help="Path to compiler binaries")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed comparison even when matching")
    parser.add_argument("files", nargs="*", help="C files to compare")
    args = parser.parse_args()

    arch_config = ARCH_CONFIG[args.arch]

    # Set up environment for before/after
    if args.before == "gcc":
        before_env = {}  # No MY_ASM = use default GCC
    else:
        before_env = {"MY_ASM": args.before}

    if args.after == "builtin":
        after_env = {"MY_ASM": "builtin"}
    else:
        after_env = {"MY_ASM": args.after}

    if args.files:
        # User-provided files
        total = 0
        passed = 0
        for f in args.files:
            print(f"Comparing: {f}")
            if compare_files(arch_config, args.compiler_path, f, before_env, after_env, args.verbose):
                passed += 1
            total += 1
        print(f"\nResults: {passed}/{total} files match")
    else:
        # Built-in test snippets
        total = 0
        passed = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, code in DEFAULT_TEST_SNIPPETS:
                c_file = os.path.join(tmpdir, f"{name}.c")
                with open(c_file, 'w') as f:
                    f.write(code)
                print(f"Test: {name}")
                if compare_files(arch_config, args.compiler_path, c_file, before_env, after_env, args.verbose):
                    passed += 1
                total += 1
        print(f"\nResults: {passed}/{total} tests match")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
