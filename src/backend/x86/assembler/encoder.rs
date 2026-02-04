//! x86-64 instruction encoder.
//!
//! Encodes parsed x86-64 instructions into machine code bytes.
//! Handles REX prefixes, ModR/M, SIB, and displacement encoding.

use super::parser::*;

/// Relocation entry for the linker to resolve.
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Offset within the section where the relocation applies.
    pub offset: u64,
    /// Symbol name to resolve.
    pub symbol: String,
    /// Relocation type (ELF R_X86_64_* constants).
    pub reloc_type: u32,
    /// Addend for the relocation.
    pub addend: i64,
}

// ELF x86-64 relocation types
#[allow(dead_code)]
pub const R_X86_64_NONE: u32 = 0;
pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_PC32: u32 = 2;
#[allow(dead_code)]
pub const R_X86_64_GOT32: u32 = 3;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_X86_64_32: u32 = 10;
pub const R_X86_64_32S: u32 = 11;
pub const R_X86_64_GOTPCREL: u32 = 9;
pub const R_X86_64_TPOFF32: u32 = 23;
pub const R_X86_64_GOTTPOFF: u32 = 22;
#[allow(dead_code)]
pub const R_X86_64_TPOFF64: u32 = 18;
/// Internal-only: 8-bit PC-relative relocation (for loop/jrcxz instructions).
/// This is resolved during assembly and never appears in the output ELF.
pub const R_X86_64_PC8_INTERNAL: u32 = 0xFF00;

/// Register encoding (3-bit register number in ModR/M and SIB).
fn reg_num(name: &str) -> Option<u8> {
    match name {
        "al" | "ax" | "eax" | "rax" | "xmm0" | "ymm0" | "st" | "st(0)" | "mm0" | "es" => Some(0),
        "cl" | "cx" | "ecx" | "rcx" | "xmm1" | "ymm1" | "st(1)" | "mm1" | "cs" => Some(1),
        "dl" | "dx" | "edx" | "rdx" | "xmm2" | "ymm2" | "st(2)" | "mm2" | "ss" => Some(2),
        "bl" | "bx" | "ebx" | "rbx" | "xmm3" | "ymm3" | "st(3)" | "mm3" | "ds" => Some(3),
        "ah" | "spl" | "sp" | "esp" | "rsp" | "xmm4" | "ymm4" | "st(4)" | "mm4" | "fs" => Some(4),
        "ch" | "bpl" | "bp" | "ebp" | "rbp" | "xmm5" | "ymm5" | "st(5)" | "mm5" | "gs" => Some(5),
        "dh" | "sil" | "si" | "esi" | "rsi" | "xmm6" | "ymm6" | "st(6)" | "mm6" => Some(6),
        "bh" | "dil" | "di" | "edi" | "rdi" | "xmm7" | "ymm7" | "st(7)" | "mm7" => Some(7),
        "r8b" | "r8w" | "r8d" | "r8" | "xmm8" | "ymm8" => Some(0),
        "r9b" | "r9w" | "r9d" | "r9" | "xmm9" | "ymm9" => Some(1),
        "r10b" | "r10w" | "r10d" | "r10" | "xmm10" | "ymm10" => Some(2),
        "r11b" | "r11w" | "r11d" | "r11" | "xmm11" | "ymm11" => Some(3),
        "r12b" | "r12w" | "r12d" | "r12" | "xmm12" | "ymm12" => Some(4),
        "r13b" | "r13w" | "r13d" | "r13" | "xmm13" | "ymm13" => Some(5),
        "r14b" | "r14w" | "r14d" | "r14" | "xmm14" | "ymm14" => Some(6),
        "r15b" | "r15w" | "r15d" | "r15" | "xmm15" | "ymm15" => Some(7),
        _ => None,
    }
}

/// Does this register need the REX.B/R/X extension bit?
fn needs_rex_ext(name: &str) -> bool {
    name.starts_with("r8") || name.starts_with("r9") || name.starts_with("r10")
        || name.starts_with("r11") || name.starts_with("r12") || name.starts_with("r13")
        || name.starts_with("r14") || name.starts_with("r15")
        || name.starts_with("xmm8") || name.starts_with("xmm9")
        || name.starts_with("xmm10") || name.starts_with("xmm11")
        || name.starts_with("xmm12") || name.starts_with("xmm13")
        || name.starts_with("xmm14") || name.starts_with("xmm15")
        || name.starts_with("ymm8") || name.starts_with("ymm9")
        || name.starts_with("ymm10") || name.starts_with("ymm11")
        || name.starts_with("ymm12") || name.starts_with("ymm13")
        || name.starts_with("ymm14") || name.starts_with("ymm15")
}

/// Is this a 64-bit GP register?
fn is_reg64(name: &str) -> bool {
    matches!(name, "rax" | "rcx" | "rdx" | "rbx" | "rsp" | "rbp" | "rsi" | "rdi"
        | "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15")
}

/// Is this a 32-bit GP register?
#[allow(dead_code)]
fn is_reg32(name: &str) -> bool {
    matches!(name, "eax" | "ecx" | "edx" | "ebx" | "esp" | "ebp" | "esi" | "edi"
        | "r8d" | "r9d" | "r10d" | "r11d" | "r12d" | "r13d" | "r14d" | "r15d")
}

/// Is this a 16-bit GP register?
#[allow(dead_code)]
fn is_reg16(name: &str) -> bool {
    matches!(name, "ax" | "cx" | "dx" | "bx" | "sp" | "bp" | "si" | "di"
        | "r8w" | "r9w" | "r10w" | "r11w" | "r12w" | "r13w" | "r14w" | "r15w")
}

/// Is this an 8-bit GP register?
#[allow(dead_code)]
fn is_reg8(name: &str) -> bool {
    matches!(name, "al" | "cl" | "dl" | "bl" | "ah" | "ch" | "dh" | "bh"
        | "spl" | "bpl" | "sil" | "dil"
        | "r8b" | "r9b" | "r10b" | "r11b" | "r12b" | "r13b" | "r14b" | "r15b")
}

/// Does this 8-bit register require REX prefix for access (spl, bpl, sil, dil)?
fn is_rex_required_8bit(name: &str) -> bool {
    matches!(name, "spl" | "bpl" | "sil" | "dil")
}

/// Is this an XMM register?
fn is_xmm(name: &str) -> bool {
    name.starts_with("xmm")
}

/// Is this an x87 FPU register?
#[allow(dead_code)]
fn is_st(name: &str) -> bool {
    name == "st" || name.starts_with("st(")
}

/// Get operand size from mnemonic suffix.
fn mnemonic_size_suffix(mnemonic: &str) -> Option<u8> {
    // Handle mnemonics that don't follow the simple suffix pattern
    match mnemonic {
        "cltq" | "cqto" | "cltd" | "cdq" | "cqo" | "ret" | "nop" | "ud2"
        | "endbr64" | "pause" | "mfence" | "lfence" | "sfence" | "clflush"
        | "syscall" | "sysenter" | "cpuid" | "rdtsc" | "rdtscp"
        | "clc" | "stc" | "cld" | "std" | "cmc"
        | "pushfq" | "popfq" | "pushf" | "popf"
        | "sahf" | "lahf" | "int3" | "hlt"
        | "fninit" | "fnstcw" | "fldcw" | "emms"
        | "leave" | "leaveq" | "fwait" | "wait"
        | "fabs" | "fxch" | "fsubp" | "fdivp" => return None,
        _ => {}
    }

    let last = mnemonic.as_bytes().last()?;
    match last {
        b'b' => Some(1),
        b'w' => Some(2),
        b'l' | b'd' => Some(4),
        b'q' => Some(8),
        _ => None,
    }
}

/// Infer operand size from register operands (for suffix-less instructions).
/// Searches in reverse order because in AT&T syntax, the destination (last operand)
/// is the most reliable indicator of the intended operation size.
/// Returns None when no register operands are present.
fn infer_size_from_ops(ops: &[Operand]) -> Option<u8> {
    for op in ops.iter().rev() {
        match op {
            Operand::Register(r) => {
                if is_reg64(&r.name) { return Some(8); }
                if is_reg32(&r.name) { return Some(4); }
                if is_reg16(&r.name) { return Some(2); }
                if is_reg8(&r.name) { return Some(1); }
            }
            _ => {}
        }
    }
    None
}

/// Is this a YMM register (AVX 256-bit)?
fn is_ymm(name: &str) -> bool {
    name.starts_with("ymm")
}

/// Get YMM/XMM register number.
fn ymm_num(name: &str) -> Option<u8> {
    let n: u8 = name.strip_prefix("ymm")?.parse().ok()?;
    if n < 16 { Some(n & 7) } else { None }
}

/// Does this YMM register need VEX.B extension?
fn ymm_needs_ext(name: &str) -> bool {
    match name {
        "ymm8" | "ymm9" | "ymm10" | "ymm11" | "ymm12" | "ymm13" | "ymm14" | "ymm15" => true,
        _ => false,
    }
}

/// Is this a segment register?
fn is_segment_reg(name: &str) -> bool {
    matches!(name, "cs" | "ds" | "es" | "fs" | "gs" | "ss")
}

/// Is this an MMX register?
fn is_mmx_reg(name: &str) -> bool {
    name.starts_with("mm") && !name.starts_with("mmx") && name.len() <= 3
}

/// Get MMX register number.
fn mmx_num(name: &str) -> Option<u8> {
    let n: u8 = name.strip_prefix("mm")?.parse().ok()?;
    if n < 8 { Some(n) } else { None }
}

/// Instruction encoding context.
pub struct InstructionEncoder {
    /// Output bytes.
    pub bytes: Vec<u8>,
    /// Relocations generated during encoding.
    pub relocations: Vec<Relocation>,
    /// Current offset within the section.
    pub offset: u64,
}

impl InstructionEncoder {
    pub fn new() -> Self {
        InstructionEncoder {
            bytes: Vec::new(),
            relocations: Vec::new(),
            offset: 0,
        }
    }

    /// Encode a single instruction and append bytes.
    pub fn encode(&mut self, instr: &Instruction) -> Result<(), String> {
        let start_len = self.bytes.len();

        // Handle prefix
        if let Some(ref prefix) = instr.prefix {
            match prefix.as_str() {
                "lock" => self.bytes.push(0xF0),
                "rep" | "repz" | "repe" => self.bytes.push(0xF3),
                "repnz" | "repne" => self.bytes.push(0xF2),
                _ => return Err(format!("unknown prefix: {}", prefix)),
            }
        }

        let result = self.encode_mnemonic(instr);

        if result.is_ok() {
            self.offset += (self.bytes.len() - start_len) as u64;
        }

        result
    }

    /// Main mnemonic dispatch.
    fn encode_mnemonic(&mut self, instr: &Instruction) -> Result<(), String> {
        let mnemonic = instr.mnemonic.as_str();
        let ops = &instr.operands;

        match mnemonic {
            // Data movement
            "movq" => {
                // Check if any operand is an XMM register - route to SSE movq
                let has_xmm = ops.iter().any(|op| matches!(op, Operand::Register(r) if is_xmm(&r.name)));
                let has_mmx = ops.iter().any(|op| matches!(op, Operand::Register(r) if is_mmx_reg(&r.name)));
                if has_xmm {
                    self.encode_movq_xmm(ops)
                } else if has_mmx {
                    self.encode_mmx_movq(ops)
                } else {
                    self.encode_mov(ops, 8)
                }
            }
            "movl" => self.encode_mov(ops, 4),
            "movw" => self.encode_mov(ops, 2),
            "movb" => self.encode_mov(ops, 1),
            "movslq" => self.encode_movsx(ops, 4, 8),
            "movsbq" => self.encode_movsx(ops, 1, 8),
            "movswq" => self.encode_movsx(ops, 2, 8),
            "movsbl" => self.encode_movsx(ops, 1, 4),
            "movsbw" => self.encode_movsx(ops, 1, 2),
            "movswl" => self.encode_movsx(ops, 2, 4),
            "movsww" => self.encode_movsx(ops, 2, 2),
            "movzbq" | "movzbl" => self.encode_movzx(ops, 1, if mnemonic == "movzbq" { 8 } else { 4 }),
            "movzbw" => self.encode_movzx(ops, 1, 2),
            "movzwq" | "movzwl" => self.encode_movzx(ops, 2, if mnemonic == "movzwq" { 8 } else { 4 }),

            // LEA
            "leaq" => self.encode_lea(ops, 8),
            "leal" => self.encode_lea(ops, 4),

            // Stack ops
            "pushq" => self.encode_push(ops),
            "popq" => self.encode_pop(ops),

            // Arithmetic
            "addq" | "addl" | "addw" | "addb" => self.encode_alu(ops, mnemonic, 0),
            "orq" | "orl" | "orw" | "orb" => self.encode_alu(ops, mnemonic, 1),
            "adcq" | "adcl" => self.encode_alu(ops, mnemonic, 2),
            "sbbq" | "sbbl" => self.encode_alu(ops, mnemonic, 3),
            "andq" | "andl" | "andw" | "andb" => self.encode_alu(ops, mnemonic, 4),
            "subq" | "subl" | "subw" | "subb" => self.encode_alu(ops, mnemonic, 5),
            "xorq" | "xorl" | "xorw" | "xorb" => self.encode_alu(ops, mnemonic, 6),
            "cmpq" | "cmpl" | "cmpw" | "cmpb" => self.encode_alu(ops, mnemonic, 7),
            "testq" | "testl" | "testw" | "testb" => self.encode_test(ops, mnemonic),

            // Multiply/divide
            "imulq" => self.encode_imul(ops, 8),
            "imull" => self.encode_imul(ops, 4),
            "mulq" => self.encode_unary_rm(ops, 4, 8),
            "divq" => self.encode_unary_rm(ops, 6, 8),
            "divl" => self.encode_unary_rm(ops, 6, 4),
            "idivq" => self.encode_unary_rm(ops, 7, 8),
            "idivl" => self.encode_unary_rm(ops, 7, 4),

            // Unary
            "negq" => self.encode_unary_rm(ops, 3, 8),
            "negl" => self.encode_unary_rm(ops, 3, 4),
            "notq" => self.encode_unary_rm(ops, 2, 8),
            "notl" => self.encode_unary_rm(ops, 2, 4),
            "incq" => self.encode_unary_rm(ops, 0, 8),
            "incl" => self.encode_unary_rm(ops, 0, 4),
            "decq" => self.encode_unary_rm(ops, 1, 8),
            "decl" => self.encode_unary_rm(ops, 1, 4),

            // Shifts
            "shlq" | "shll" | "shlw" | "shlb" => self.encode_shift(ops, mnemonic, 4),
            "shrq" | "shrl" | "shrw" | "shrb" => self.encode_shift(ops, mnemonic, 5),
            "sarq" | "sarl" | "sarw" | "sarb" => self.encode_shift(ops, mnemonic, 7),
            "rolq" | "roll" | "rolw" | "rolb" => self.encode_shift(ops, mnemonic, 0),
            "rorq" | "rorl" | "rorw" | "rorb" => self.encode_shift(ops, mnemonic, 1),

            // Double-precision shifts
            "shldq" => self.encode_double_shift(ops, 0xA4, 8),
            "shrdq" => self.encode_double_shift(ops, 0xAC, 8),

            // Sign extension
            "cltq" => { self.bytes.extend_from_slice(&[0x48, 0x98]); Ok(()) }
            "cqto" | "cqo" => { self.bytes.extend_from_slice(&[0x48, 0x99]); Ok(()) }
            "cltd" | "cdq" => { self.bytes.push(0x99); Ok(()) }

            // Byte swap
            "bswapl" => self.encode_bswap(ops, 4),
            "bswapq" => self.encode_bswap(ops, 8),

            // Bit operations
            "lzcntl" | "lzcntq" | "tzcntl" | "tzcntq" | "popcntl" | "popcntq" => {
                self.encode_bit_count(ops, mnemonic)
            }

            // Conditional set
            "sete" | "setz" | "setne" | "setnz" | "setl" | "setle" | "setg" | "setge"
            | "setb" | "setc" | "setbe" | "seta" | "setae" | "setnc" | "setnp" | "setp"
            | "sets" | "setns" | "seto" | "setno"
            | "seteb" | "setzb" | "setneb" | "setnzb" | "setlb" | "setleb" | "setgb" | "setgeb"
            | "setbb" | "setcb" | "setbeb" | "setab" | "setaeb" | "setncb" | "setnpb" | "setpb"
            | "setsb" | "setnsb" | "setob" | "setnob"
            | "setnle" | "setnge" | "setnl" | "setnleb" | "setngeb" | "setnlb"
            => self.encode_setcc(ops, mnemonic),

            // Conditional move
            "cmoveq" | "cmovneq" | "cmovlq" | "cmovleq" | "cmovgq" | "cmovgeq"
            | "cmovbq" | "cmovbeq" | "cmovaq" | "cmovaeq"
            | "cmovel" | "cmovnel" | "cmovll" | "cmovlel" | "cmovgl" | "cmovgel"
            | "cmovbl" | "cmovbel" | "cmoval" | "cmovael" => self.encode_cmovcc(ops, mnemonic),

            // Jumps (jmpq is a common AT&T alias for jmp on x86-64)
            "jmp" | "jmpq" => self.encode_jmp(ops),
            "je" | "jz" | "jne" | "jnz" | "jl" | "jle" | "jg" | "jge"
            | "jb" | "jbe" | "ja" | "jae" | "js" | "jns" | "jo" | "jno" | "jp" | "jnp" => {
                self.encode_jcc(ops, mnemonic)
            }

            // Call/return (callq/retq are common AT&T aliases on x86-64)
            "call" | "callq" => self.encode_call(ops),
            "ret" | "retq" => {
                if ops.is_empty() {
                    self.bytes.push(0xC3);
                } else if let Some(Operand::Immediate(ImmediateValue::Integer(val))) = ops.first() {
                    // ret $imm16 - pop return address and deallocate imm16 bytes
                    self.bytes.push(0xC2);
                    self.bytes.extend_from_slice(&(*val as u16).to_le_bytes());
                } else {
                    return Err("unsupported ret operand".to_string());
                }
                Ok(())
            }

            // No-ops and misc
            "nop" => { self.bytes.push(0x90); Ok(()) }
            "ud2" => { self.bytes.extend_from_slice(&[0x0F, 0x0B]); Ok(()) }
            "endbr64" => { self.bytes.extend_from_slice(&[0xF3, 0x0F, 0x1E, 0xFA]); Ok(()) }
            "pause" => { self.bytes.extend_from_slice(&[0xF3, 0x90]); Ok(()) }
            "mfence" => { self.bytes.extend_from_slice(&[0x0F, 0xAE, 0xF0]); Ok(()) }
            "lfence" => { self.bytes.extend_from_slice(&[0x0F, 0xAE, 0xE8]); Ok(()) }
            "sfence" => { self.bytes.extend_from_slice(&[0x0F, 0xAE, 0xF8]); Ok(()) }
            "clflush" => self.encode_clflush(ops),

            // String ops
            "movsb" => { self.bytes.push(0xA4); Ok(()) }
            "movsd" if ops.is_empty() => { self.bytes.push(0xA5); Ok(()) }
            "stosb" => { self.bytes.push(0xAA); Ok(()) }
            "stosd" | "stosl" => { self.bytes.push(0xAB); Ok(()) }

            // Atomic exchange
            "xchgb" | "xchgw" | "xchgl" | "xchgq" => self.encode_xchg(ops, mnemonic),

            // Lock-prefixed atomics (already have the lock prefix from the prefix handling)
            "cmpxchgb" | "cmpxchgw" | "cmpxchgl" | "cmpxchgq" => self.encode_cmpxchg(ops, mnemonic),
            "cmpxchg8b" => self.encode_cmpxchg8b(ops),
            "cmpxchg16b" => self.encode_cmpxchg16b(ops),
            "xaddb" | "xaddw" | "xaddl" | "xaddq" => self.encode_xadd(ops, mnemonic),

            // SSE/SSE2 floating-point
            "movss" => self.encode_sse_rr_rm(ops, &[0xF3, 0x0F, 0x10], &[0xF3, 0x0F, 0x11]),
            "movsd" => self.encode_sse_rr_rm(ops, &[0xF2, 0x0F, 0x10], &[0xF2, 0x0F, 0x11]),
            "movd" => self.encode_movd(ops),
            "movdqu" => self.encode_sse_rr_rm(ops, &[0xF3, 0x0F, 0x6F], &[0xF3, 0x0F, 0x7F]),
            "movupd" => self.encode_sse_rr_rm(ops, &[0x66, 0x0F, 0x10], &[0x66, 0x0F, 0x11]),

            "addsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x58]),
            "subsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x5C]),
            "mulsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x59]),
            "divsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x5E]),
            "addss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x58]),
            "subss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x5C]),
            "mulss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x59]),
            "divss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x5E]),
            "sqrtsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x51]),
            "sqrtss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x51]),
            "ucomisd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x2E]),
            "ucomiss" => self.encode_sse_op(ops, &[0x0F, 0x2E]),
            "xorpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x57]),
            "xorps" => self.encode_sse_op(ops, &[0x0F, 0x57]),
            "andpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x54]),
            "andps" => self.encode_sse_op(ops, &[0x0F, 0x54]),

            // SSE conversions
            "cvtsd2ss" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x5A]),
            "cvtss2sd" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x5A]),
            "cvtsi2sdq" => self.encode_sse_cvt_gp_to_xmm(ops, &[0xF2, 0x0F, 0x2A], 8),
            "cvtsi2ssq" => self.encode_sse_cvt_gp_to_xmm(ops, &[0xF3, 0x0F, 0x2A], 8),
            "cvttsd2siq" => self.encode_sse_cvt_xmm_to_gp(ops, &[0xF2, 0x0F, 0x2C], 8),
            "cvttss2siq" => self.encode_sse_cvt_xmm_to_gp(ops, &[0xF3, 0x0F, 0x2C], 8),

            // SSE2 integer SIMD
            "paddw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xFD]),
            "psubw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xF9]),
            "paddd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xFE]),
            "psubd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xFA]),
            "pmulhw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE5]),
            "pmaddwd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xF5]),
            "pcmpgtw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x65]),
            "pcmpgtb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x64]),
            "packssdw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x6B]),
            "packuswb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x67]),
            "punpcklbw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x60]),
            "punpckhbw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x68]),
            "punpcklwd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x61]),
            "punpckhwd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x69]),
            "pmovmskb" => self.encode_sse_cvt_xmm_to_gp(ops, &[0x66, 0x0F, 0xD7], 4),

            // Additional SSE2 packed integer operations
            "pcmpeqb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x74]),
            "pcmpeqd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x76]),
            "pcmpeqw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x75]),
            "pand" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xDB]),
            "pandn" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xDF]),
            "por" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xEB]),
            "pxor" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xEF]),
            "psubusb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xD8]),
            "psubusw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xD9]),
            "psubsb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE8]),
            "psubsw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE9]),
            "paddusb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xDC]),
            "paddusw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xDD]),
            "paddsb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xEC]),
            "paddsw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xED]),
            "pmuludq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xF4]),
            "pmullw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xD5]),
            "pmulld" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x40]),
            "pminub" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xDA]),
            "pmaxub" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xDE]),
            "pminsd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x39]),
            "pmaxsd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x3D]),
            "pavgb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE0]),
            "pavgw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE3]),
            "psadbw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xF6]),
            "punpcklqdq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x6C]),
            "punpckhqdq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x6D]),
            "punpckldq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x62]),
            "punpckhdq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x6A]),

            // SSE packed float operations
            "addpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x58]),
            "subpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x5C]),
            "mulpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x59]),
            "divpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x5E]),
            "addps" => self.encode_sse_op(ops, &[0x0F, 0x58]),
            "subps" => self.encode_sse_op(ops, &[0x0F, 0x5C]),
            "mulps" => self.encode_sse_op(ops, &[0x0F, 0x59]),
            "divps" => self.encode_sse_op(ops, &[0x0F, 0x5E]),
            "orpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x56]),
            "orps" => self.encode_sse_op(ops, &[0x0F, 0x56]),
            "andnpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x55]),
            "andnps" => self.encode_sse_op(ops, &[0x0F, 0x55]),
            "minsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x5D]),
            "maxsd" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x5F]),
            "minss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x5D]),
            "maxss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x5F]),

            // Additional SSE data movement
            "movaps" => self.encode_sse_rr_rm(ops, &[0x0F, 0x28], &[0x0F, 0x29]),
            "movdqa" => self.encode_sse_rr_rm(ops, &[0x66, 0x0F, 0x6F], &[0x66, 0x0F, 0x7F]),
            "movlpd" => self.encode_sse_rr_rm(ops, &[0x66, 0x0F, 0x12], &[0x66, 0x0F, 0x13]),
            "movhpd" => self.encode_sse_rr_rm(ops, &[0x66, 0x0F, 0x16], &[0x66, 0x0F, 0x17]),

            // SSE shifts with immediate
            "psllw" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xF1], 6, &[0x66, 0x0F, 0x71]),
            "psrlw" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xD1], 2, &[0x66, 0x0F, 0x71]),
            "psraw" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xE1], 4, &[0x66, 0x0F, 0x71]),
            "pslld" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xF2], 6, &[0x66, 0x0F, 0x72]),
            "psrld" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xD2], 2, &[0x66, 0x0F, 0x72]),
            "psrad" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xE2], 4, &[0x66, 0x0F, 0x72]),
            "psllq" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xF3], 6, &[0x66, 0x0F, 0x73]),
            "psrlq" => self.encode_sse_shift(ops, &[0x66, 0x0F, 0xD3], 2, &[0x66, 0x0F, 0x73]),
            "pslldq" => self.encode_sse_shift_imm_only(ops, 7, &[0x66, 0x0F, 0x73]),
            "psrldq" => self.encode_sse_shift_imm_only(ops, 3, &[0x66, 0x0F, 0x73]),

            // SSE shuffles
            "pshufd" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x70]),
            "pshuflw" => self.encode_sse_op_imm8(ops, &[0xF2, 0x0F, 0x70]),
            "pshufhw" => self.encode_sse_op_imm8(ops, &[0xF3, 0x0F, 0x70]),

            // SSE insert/extract
            "pinsrw" => self.encode_sse_insert(ops, &[0x66, 0x0F, 0xC4]),
            "pextrw" => self.encode_sse_extract(ops, &[0x66, 0x0F, 0xC5]),
            "pinsrd" => self.encode_sse_insert(ops, &[0x66, 0x0F, 0x3A, 0x22]),
            "pextrd" => self.encode_sse_extract(ops, &[0x66, 0x0F, 0x3A, 0x16]),
            "pinsrb" => self.encode_sse_insert(ops, &[0x66, 0x0F, 0x3A, 0x20]),
            "pextrb" => self.encode_sse_extract(ops, &[0x66, 0x0F, 0x3A, 0x14]),
            "pinsrq" => self.encode_sse_insert(ops, &[0x66, 0x0F, 0x3A, 0x22]),
            "pextrq" => self.encode_sse_extract(ops, &[0x66, 0x0F, 0x3A, 0x16]),

            // AES-NI
            "aesenc" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0xDC]),
            "aesenclast" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0xDD]),
            "aesdec" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0xDE]),
            "aesdeclast" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0xDF]),
            "aesimc" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0xDB]),
            "aeskeygenassist" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0xDF]),

            // PCLMULQDQ
            "pclmulqdq" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x44]),

            // Non-temporal stores
            "movnti" => self.encode_movnti(ops),
            "movntdq" => self.encode_sse_store(ops, &[0x66, 0x0F, 0xE7]),
            "movntpd" => self.encode_sse_store(ops, &[0x66, 0x0F, 0x2B]),

            // CRC32
            "crc32b" => self.encode_crc32(ops, 1),
            "crc32w" => self.encode_crc32(ops, 2),
            "crc32l" => self.encode_crc32(ops, 4),
            "crc32q" => self.encode_crc32(ops, 8),

            // x87 FPU
            "fldt" => self.encode_x87_mem(ops, &[0xDB], 5),
            "fstpt" => self.encode_x87_mem(ops, &[0xDB], 7),
            "fldl" => self.encode_x87_mem(ops, &[0xDD], 0),
            "flds" => self.encode_x87_mem(ops, &[0xD9], 0),
            "fstpl" => self.encode_x87_mem(ops, &[0xDD], 3),
            "fstps" => self.encode_x87_mem(ops, &[0xD9], 3),
            "fildq" => self.encode_x87_mem(ops, &[0xDF], 5),
            "fildl" => self.encode_x87_mem(ops, &[0xDB], 0),
            "fisttpq" => self.encode_x87_mem(ops, &[0xDD], 1),
            "fistpl" => self.encode_x87_mem(ops, &[0xDB], 3),
            "fistpq" => self.encode_x87_mem(ops, &[0xDF], 7),
            "fistpll" => self.encode_x87_mem(ops, &[0xDF], 7), // alias for fistpq
            "faddp" => { self.bytes.extend_from_slice(&[0xDE, 0xC1]); Ok(()) }
            // Note: AT&T syntax swaps the meaning of fsub/fsubr and fdiv/fdivr
            // relative to Intel mnemonics for the *p (pop) forms.
            // GAS: fsubp = DE E1, fsubrp = DE E9, fdivp = DE F1, fdivrp = DE F9
            "fsubp" => { self.bytes.extend_from_slice(&[0xDE, 0xE1]); Ok(()) }
            "fsubrp" => { self.bytes.extend_from_slice(&[0xDE, 0xE9]); Ok(()) }
            "fmulp" => { self.bytes.extend_from_slice(&[0xDE, 0xC9]); Ok(()) }
            "fdivp" => { self.bytes.extend_from_slice(&[0xDE, 0xF1]); Ok(()) }
            "fdivrp" => { self.bytes.extend_from_slice(&[0xDE, 0xF9]); Ok(()) }
            "fchs" => { self.bytes.extend_from_slice(&[0xD9, 0xE0]); Ok(()) }
            "fabs" => { self.bytes.extend_from_slice(&[0xD9, 0xE1]); Ok(()) }
            "fninit" => { self.bytes.extend_from_slice(&[0xDB, 0xE3]); Ok(()) }
            "fnstcw" => self.encode_x87_mem(ops, &[0xD9], 7),
            "fldcw" => self.encode_x87_mem(ops, &[0xD9], 5),
            "fcomip" => self.encode_fcomip(ops),
            "fucomip" => self.encode_fucomip(ops),
            "fld" => self.encode_fld_st(ops),
            "fstp" => self.encode_fstp_st(ops),
            "fxch" => self.encode_fxch(ops),
            "fild" => self.encode_x87_mem(ops, &[0xDB], 0),
            "fistp" => self.encode_x87_mem(ops, &[0xDB], 3),
            "faddl" => self.encode_x87_mem(ops, &[0xDC], 0),
            "fadds" => self.encode_x87_mem(ops, &[0xD8], 0),
            "fsubl" => self.encode_x87_mem(ops, &[0xDC], 4),
            "fsubs" => self.encode_x87_mem(ops, &[0xD8], 4),
            "fmull" => self.encode_x87_mem(ops, &[0xDC], 1),
            "fmuls" => self.encode_x87_mem(ops, &[0xD8], 1),
            "fdivl" => self.encode_x87_mem(ops, &[0xDC], 6),
            "fdivs" => self.encode_x87_mem(ops, &[0xD8], 6),
            "fdivrl" => self.encode_x87_mem(ops, &[0xDC], 7),
            "fsubrl" => self.encode_x87_mem(ops, &[0xDC], 5),
            "fstl" => self.encode_x87_mem(ops, &[0xDD], 2),
            "fsts" => self.encode_x87_mem(ops, &[0xD9], 2),
            "fsubp" => self.encode_fsubp(ops),
            "fdivp" => self.encode_fdivp(ops),
            "fcompl" => self.encode_x87_mem(ops, &[0xDC], 3),
            "fcomps" => self.encode_x87_mem(ops, &[0xD8], 3),
            "fcoml" => self.encode_x87_mem(ops, &[0xDC], 2),
            "fcoms" => self.encode_x87_mem(ops, &[0xD8], 2),
            "fisttpl" => self.encode_x87_mem(ops, &[0xDB], 1),
            "fnstenv" => self.encode_x87_mem(ops, &[0xD9], 6),
            "fldenv" => self.encode_x87_mem(ops, &[0xD9], 4),
            "fnsave" => self.encode_x87_mem(ops, &[0xDD], 6),
            "frstor" => self.encode_x87_mem(ops, &[0xDD], 4),
            "emms" => { self.bytes.extend_from_slice(&[0x0F, 0x77]); Ok(()) }

            // Flag manipulation
            "clc" => { self.bytes.push(0xF8); Ok(()) }
            "stc" => { self.bytes.push(0xF9); Ok(()) }
            "cmc" => { self.bytes.push(0xF5); Ok(()) }
            "cld" => { self.bytes.push(0xFC); Ok(()) }
            "std" => { self.bytes.push(0xFD); Ok(()) }
            "sahf" => { self.bytes.push(0x9E); Ok(()) }
            "lahf" => { self.bytes.push(0x9F); Ok(()) }

            // Flag push/pop
            "pushfq" | "pushf" => { self.bytes.push(0x9C); Ok(()) }
            "popfq" | "popf" => { self.bytes.push(0x9D); Ok(()) }

            // System instructions
            "syscall" => { self.bytes.extend_from_slice(&[0x0F, 0x05]); Ok(()) }
            "sysenter" => { self.bytes.extend_from_slice(&[0x0F, 0x34]); Ok(()) }
            "cpuid" => { self.bytes.extend_from_slice(&[0x0F, 0xA2]); Ok(()) }
            "rdtsc" => { self.bytes.extend_from_slice(&[0x0F, 0x31]); Ok(()) }
            "rdtscp" => { self.bytes.extend_from_slice(&[0x0F, 0x01, 0xF9]); Ok(()) }

            // Breakpoint
            "int3" => { self.bytes.push(0xCC); Ok(()) }
            "int" => self.encode_int(ops),
            "hlt" => { self.bytes.push(0xF4); Ok(()) }

            // String operations
            "movsq" => { self.bytes.extend_from_slice(&[0x48, 0xA5]); Ok(()) }
            "movsw" => { self.bytes.extend_from_slice(&[0x66, 0xA5]); Ok(()) }
            "movsl" => { self.bytes.push(0xA5); Ok(()) }
            "stosq" => { self.bytes.extend_from_slice(&[0x48, 0xAB]); Ok(()) }
            "stosw" => { self.bytes.extend_from_slice(&[0x66, 0xAB]); Ok(()) }
            "lodsb" => { self.bytes.push(0xAC); Ok(()) }
            "lodsl" => { self.bytes.push(0xAD); Ok(()) }
            "lodsq" => { self.bytes.extend_from_slice(&[0x48, 0xAD]); Ok(()) }
            "scasb" => { self.bytes.push(0xAE); Ok(()) }
            "scasl" => { self.bytes.push(0xAF); Ok(()) }
            "scasq" => { self.bytes.extend_from_slice(&[0x48, 0xAF]); Ok(()) }
            "cmpsb" => { self.bytes.push(0xA6); Ok(()) }
            "cmpsw" => { self.bytes.extend_from_slice(&[0x66, 0xA7]); Ok(()) }
            "cmpsl" => { self.bytes.push(0xA7); Ok(()) }
            "cmpsq" => { self.bytes.extend_from_slice(&[0x48, 0xA7]); Ok(()) }

            // rep prefix as standalone mnemonic (followed by string op)
            "rep" => {
                // rep is a prefix; when it appears as a standalone mnemonic,
                // just emit the prefix byte - the next instruction will be the string op
                self.bytes.push(0xF3);
                Ok(())
            }
            "repe" | "repz" => {
                self.bytes.push(0xF3);
                Ok(())
            }
            "repnz" | "repne" => {
                self.bytes.push(0xF2);
                Ok(())
            }
            "lock" => {
                self.bytes.push(0xF0);
                Ok(())
            }

            // Bit scan
            "bsfl" => self.encode_bitscan(ops, &[0x0F, 0xBC], 4),
            "bsfq" => self.encode_bitscan(ops, &[0x0F, 0xBC], 8),
            "bsrl" => self.encode_bitscan(ops, &[0x0F, 0xBD], 4),
            "bsrq" => self.encode_bitscan(ops, &[0x0F, 0xBD], 8),
            "bsfw" => self.encode_bitscan(ops, &[0x0F, 0xBC], 2),
            "bsrw" => self.encode_bitscan(ops, &[0x0F, 0xBD], 2),

            // Bit test
            "btl" => self.encode_bt(ops, 0, 4),
            "btq" => self.encode_bt(ops, 0, 8),
            "btsl" => self.encode_bt(ops, 5, 4),
            "btsq" => self.encode_bt(ops, 5, 8),
            "btrl" => self.encode_bt(ops, 6, 4),
            "btrq" => self.encode_bt(ops, 6, 8),
            "btcl" => self.encode_bt(ops, 7, 4),
            "btcq" => self.encode_bt(ops, 7, 8),
            "btw" => self.encode_bt(ops, 0, 2),
            "btsw" => self.encode_bt(ops, 5, 2),
            "btrw" => self.encode_bt(ops, 6, 2),
            "btcw" => self.encode_bt(ops, 7, 2),

            // Unsigned multiply (one-operand form)
            "mull" => self.encode_unary_rm(ops, 4, 4),
            "mulw" => self.encode_unary_rm(ops, 4, 2),
            "mulb" => self.encode_unary_rm(ops, 4, 1),

            // Missing unary operations
            "negw" => self.encode_unary_rm(ops, 3, 2),
            "negb" => self.encode_unary_rm(ops, 3, 1),
            "notw" => self.encode_unary_rm(ops, 2, 2),
            "notb" => self.encode_unary_rm(ops, 2, 1),
            "incw" => self.encode_unary_rm(ops, 0, 2),
            "incb" => self.encode_unary_rm(ops, 0, 1),
            "decw" => self.encode_unary_rm(ops, 1, 2),
            "decb" => self.encode_unary_rm(ops, 1, 1),
            // Missing divide operations
            "divw" => self.encode_unary_rm(ops, 6, 2),
            "divb" => self.encode_unary_rm(ops, 6, 1),
            "idivw" => self.encode_unary_rm(ops, 7, 2),
            "idivb" => self.encode_unary_rm(ops, 7, 1),
            "imulw" => self.encode_imul(ops, 2),
            "imulb" => self.encode_unary_rm(ops, 5, 1),

            // Missing shifts on memory
            "rclq" | "rcll" | "rclw" | "rclb" => self.encode_shift(ops, mnemonic, 2),
            "rcrq" | "rcrl" | "rcrw" | "rcrb" => self.encode_shift(ops, mnemonic, 3),

            // Double-precision shifts (32-bit and with cl)
            "shldl" => self.encode_double_shift(ops, 0xA4, 4),
            "shrdl" => self.encode_double_shift(ops, 0xAC, 4),
            "shldw" => self.encode_double_shift(ops, 0xA4, 2),
            "shrdw" => self.encode_double_shift(ops, 0xAC, 2),

            // Byte swap (without suffix, infer from register)
            "bswap" => self.encode_bswap_infer(ops),

            // Missing conditional move variations
            "cmovzq" | "cmovzl" | "cmovzw" => self.encode_cmovcc(ops, mnemonic),
            "cmovnzq" | "cmovnzl" | "cmovnzw" => self.encode_cmovcc(ops, mnemonic),
            "cmovcq" | "cmovcl" | "cmovcw" => self.encode_cmovcc(ops, mnemonic),
            "cmovncq" | "cmovncl" | "cmovncw" => self.encode_cmovcc(ops, mnemonic),
            "cmovsq" | "cmovsl" | "cmovsw" => self.encode_cmovcc(ops, mnemonic),
            "cmovnsq" | "cmovnsl" | "cmovnsw" => self.encode_cmovcc(ops, mnemonic),
            "cmovoq" | "cmovol" | "cmovow" => self.encode_cmovcc(ops, mnemonic),
            "cmovnoq" | "cmovnol" | "cmovnow" => self.encode_cmovcc(ops, mnemonic),
            "cmovpq" | "cmovpl" | "cmovpw" => self.encode_cmovcc(ops, mnemonic),
            "cmovnpq" | "cmovnpl" | "cmovnpw" => self.encode_cmovcc(ops, mnemonic),
            "cmovnael" | "cmovnaeq" => self.encode_cmovcc(ops, mnemonic),
            "cmovnal" | "cmovnaq" => self.encode_cmovcc(ops, mnemonic),

            // Missing conditional jumps
            "jc" => self.encode_jcc(ops, "jb"),  // jc = jb (carry)
            "jnc" => self.encode_jcc(ops, "jae"), // jnc = jae (no carry)

            // Additional SSE3/SSSE3/SSE4 instructions
            "pshufb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x00]),
            "phaddw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x01]),
            "phaddd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x02]),
            "pmaddubsw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x04]),
            "palignr" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x0F]),
            "pabsb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x1C]),
            "pabsw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x1D]),
            "pabsd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x1E]),
            "pminsb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x38]),
            "pminuw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x3A]),
            "pminud" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x3B]),
            "pmaxsb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x3C]),
            "pmaxuw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x3E]),
            "pmaxud" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x3F]),
            "pminsw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xEA]),
            "pmaxsw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xEE]),
            "blendvpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x15]),
            "pblendw" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x0E]),
            "dpps" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x40]),
            "dppd" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x41]),
            "roundsd" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x0B]),
            "roundss" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x0A]),
            "roundpd" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x09]),
            "roundps" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x08]),
            "pcmpistri" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x63]),
            "pcmpistrm" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x62]),
            "pcmpestri" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x61]),
            "pcmpestrm" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0x3A, 0x60]),
            "pcmpgtd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x66]),
            "pcmpgtq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x37]),
            "paddb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xFC]),
            "paddq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xD4]),
            "psubb" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xF8]),
            "psubq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xFB]),
            "pmulhuw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE4]),
            "pmuldq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x28]),
            "pmovsxbw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x20]),
            "pmovsxwd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x23]),
            "pmovsxdq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x25]),
            "pmovzxbw" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x30]),
            "pmovzxwd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x33]),
            "pmovzxdq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x38, 0x35]),

            // SSE conversions (32-bit integer forms)
            "cvtsi2sdl" => self.encode_sse_cvt_gp_to_xmm(ops, &[0xF2, 0x0F, 0x2A], 4),
            "cvtsi2ssl" => self.encode_sse_cvt_gp_to_xmm(ops, &[0xF3, 0x0F, 0x2A], 4),
            "cvttsd2sil" => self.encode_sse_cvt_xmm_to_gp(ops, &[0xF2, 0x0F, 0x2C], 4),
            "cvttss2sil" => self.encode_sse_cvt_xmm_to_gp(ops, &[0xF3, 0x0F, 0x2C], 4),
            "cvtsd2sil" => self.encode_sse_cvt_xmm_to_gp(ops, &[0xF2, 0x0F, 0x2D], 4),
            "cvtsd2siq" => self.encode_sse_cvt_xmm_to_gp(ops, &[0xF2, 0x0F, 0x2D], 8),

            // Additional SSE comparisons
            "cmpsd" => self.encode_sse_op_imm8(ops, &[0xF2, 0x0F, 0xC2]),
            "cmpss" => self.encode_sse_op_imm8(ops, &[0xF3, 0x0F, 0xC2]),
            "cmppd" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0xC2]),
            "cmpps" => self.encode_sse_op_imm8(ops, &[0x0F, 0xC2]),

            // Additional SSE data movement
            "movups" => self.encode_sse_rr_rm(ops, &[0x0F, 0x10], &[0x0F, 0x11]),
            "movhlps" => self.encode_sse_op(ops, &[0x0F, 0x12]),
            "movlhps" => self.encode_sse_op(ops, &[0x0F, 0x16]),
            "unpcklpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x14]),
            "unpckhpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x15]),
            "unpcklps" => self.encode_sse_op(ops, &[0x0F, 0x14]),
            "unpckhps" => self.encode_sse_op(ops, &[0x0F, 0x15]),
            "shufpd" => self.encode_sse_op_imm8(ops, &[0x66, 0x0F, 0xC6]),
            "shufps" => self.encode_sse_op_imm8(ops, &[0x0F, 0xC6]),

            // Non-temporal stores
            "movntps" => self.encode_sse_store(ops, &[0x0F, 0x2B]),

            // SSE3 operations
            "haddpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x7C]),
            "hsubpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x7D]),
            "haddps" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x7C]),
            "hsubps" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x7D]),
            "addsubpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xD0]),
            "addsubps" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0xD0]),
            "lddqu" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0xF0]),
            "movddup" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0x12]),
            "movshdup" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x16]),
            "movsldup" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x12]),

            // SSE additional ops
            "comisd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x2F]),
            "comiss" => self.encode_sse_op(ops, &[0x0F, 0x2F]),
            "sqrtpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x51]),
            "sqrtps" => self.encode_sse_op(ops, &[0x0F, 0x51]),
            "rsqrtss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x52]),
            "rcpss" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x53]),
            "minpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x5D]),
            "maxpd" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x5F]),
            "minps" => self.encode_sse_op(ops, &[0x0F, 0x5D]),
            "maxps" => self.encode_sse_op(ops, &[0x0F, 0x5F]),
            "cvtps2pd" => self.encode_sse_op(ops, &[0x0F, 0x5A]),
            "cvtpd2ps" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x5A]),
            "cvtdq2ps" => self.encode_sse_op(ops, &[0x0F, 0x5B]),
            "cvtps2dq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0x5B]),
            "cvttps2dq" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0x5B]),
            "cvtdq2pd" => self.encode_sse_op(ops, &[0xF3, 0x0F, 0xE6]),
            "cvtpd2dq" => self.encode_sse_op(ops, &[0xF2, 0x0F, 0xE6]),
            "cvttpd2dq" => self.encode_sse_op(ops, &[0x66, 0x0F, 0xE6]),

            // Misc SSE moves
            "movmskpd" => self.encode_sse_cvt_xmm_to_gp(ops, &[0x66, 0x0F, 0x50], 4),
            "movmskps" => self.encode_sse_cvt_xmm_to_gp(ops, &[0x0F, 0x50], 4),

            // VEX-encoded AVX instructions
            "vmovdqa" => self.encode_vex_mov(ops, 0x6F, 0x7F, false),
            "vmovdqu" => self.encode_vex_mov(ops, 0x6F, 0x7F, true),
            "vmovaps" => self.encode_vex_mov_noprefix(ops, 0x28, 0x29),
            "vmovups" => self.encode_vex_mov_noprefix(ops, 0x10, 0x11),
            "vmovapd" => self.encode_vex_mov_66(ops, 0x28, 0x29),
            "vmovupd" => self.encode_vex_mov_66(ops, 0x10, 0x11),
            "vpxor" => self.encode_vex_binop(ops, 0x66, 0xEF),
            "vpand" => self.encode_vex_binop(ops, 0x66, 0xDB),
            "vpor" => self.encode_vex_binop(ops, 0x66, 0xEB),
            "vpandn" => self.encode_vex_binop(ops, 0x66, 0xDF),
            "vpaddb" => self.encode_vex_binop(ops, 0x66, 0xFC),
            "vpaddw" => self.encode_vex_binop(ops, 0x66, 0xFD),
            "vpaddd" => self.encode_vex_binop(ops, 0x66, 0xFE),
            "vpaddq" => self.encode_vex_binop(ops, 0x66, 0xD4),
            "vpsubb" => self.encode_vex_binop(ops, 0x66, 0xF8),
            "vpsubw" => self.encode_vex_binop(ops, 0x66, 0xF9),
            "vpsubd" => self.encode_vex_binop(ops, 0x66, 0xFA),
            "vpsubq" => self.encode_vex_binop(ops, 0x66, 0xFB),
            "vpmulld" => self.encode_vex_binop_38(ops, 0x66, 0x40),
            "vpmullw" => self.encode_vex_binop(ops, 0x66, 0xD5),
            "vpcmpeqb" => self.encode_vex_binop(ops, 0x66, 0x74),
            "vpcmpeqd" => self.encode_vex_binop(ops, 0x66, 0x76),
            "vpcmpgtb" => self.encode_vex_binop(ops, 0x66, 0x64),
            "vpcmpgtd" => self.encode_vex_binop(ops, 0x66, 0x66),
            "vpunpcklbw" => self.encode_vex_binop(ops, 0x66, 0x60),
            "vpunpckhbw" => self.encode_vex_binop(ops, 0x66, 0x68),
            "vpunpcklwd" => self.encode_vex_binop(ops, 0x66, 0x61),
            "vpunpckhwd" => self.encode_vex_binop(ops, 0x66, 0x69),
            "vpunpckldq" => self.encode_vex_binop(ops, 0x66, 0x62),
            "vpunpckhdq" => self.encode_vex_binop(ops, 0x66, 0x6A),
            "vpunpcklqdq" => self.encode_vex_binop(ops, 0x66, 0x6C),
            "vpunpckhqdq" => self.encode_vex_binop(ops, 0x66, 0x6D),
            "vpsllw" => self.encode_vex_shift(ops, 0x66, 0xF1, 6, 0x71),
            "vpslld" => self.encode_vex_shift(ops, 0x66, 0xF2, 6, 0x72),
            "vpsllq" => self.encode_vex_shift(ops, 0x66, 0xF3, 6, 0x73),
            "vpsrlw" => self.encode_vex_shift(ops, 0x66, 0xD1, 2, 0x71),
            "vpsrld" => self.encode_vex_shift(ops, 0x66, 0xD2, 2, 0x72),
            "vpsrlq" => self.encode_vex_shift(ops, 0x66, 0xD3, 2, 0x73),
            "vpsraw" => self.encode_vex_shift(ops, 0x66, 0xE1, 4, 0x71),
            "vpsrad" => self.encode_vex_shift(ops, 0x66, 0xE2, 4, 0x72),
            "vpshufb" => self.encode_vex_binop_38(ops, 0x66, 0x00),
            "vpshufd" => self.encode_vex_unary_imm8(ops, 0x66, 0x70),
            "vpshuflw" => self.encode_vex_unary_imm8(ops, 0xF2, 0x70),
            "vpshufhw" => self.encode_vex_unary_imm8(ops, 0xF3, 0x70),
            "vpslldq" => self.encode_vex_shift_imm_only(ops, 0x66, 7, 0x73),
            "vpsrldq" => self.encode_vex_shift_imm_only(ops, 0x66, 3, 0x73),
            "vaddpd" => self.encode_vex_binop(ops, 0x66, 0x58),
            "vsubpd" => self.encode_vex_binop(ops, 0x66, 0x5C),
            "vmulpd" => self.encode_vex_binop(ops, 0x66, 0x59),
            "vdivpd" => self.encode_vex_binop(ops, 0x66, 0x5E),
            "vaddps" => self.encode_vex_binop_np(ops, 0x58),
            "vsubps" => self.encode_vex_binop_np(ops, 0x5C),
            "vmulps" => self.encode_vex_binop_np(ops, 0x59),
            "vdivps" => self.encode_vex_binop_np(ops, 0x5E),
            "vaddsd" => self.encode_vex_binop(ops, 0xF2, 0x58),
            "vsubsd" => self.encode_vex_binop(ops, 0xF2, 0x5C),
            "vmulsd" => self.encode_vex_binop(ops, 0xF2, 0x59),
            "vdivsd" => self.encode_vex_binop(ops, 0xF2, 0x5E),
            "vaddss" => self.encode_vex_binop(ops, 0xF3, 0x58),
            "vsubss" => self.encode_vex_binop(ops, 0xF3, 0x5C),
            "vmulss" => self.encode_vex_binop(ops, 0xF3, 0x59),
            "vdivss" => self.encode_vex_binop(ops, 0xF3, 0x5E),
            "vxorpd" => self.encode_vex_binop(ops, 0x66, 0x57),
            "vxorps" => self.encode_vex_binop_np(ops, 0x57),
            "vandpd" => self.encode_vex_binop(ops, 0x66, 0x54),
            "vandps" => self.encode_vex_binop_np(ops, 0x54),
            "vandnpd" => self.encode_vex_binop(ops, 0x66, 0x55),
            "vandnps" => self.encode_vex_binop_np(ops, 0x55),
            "vorpd" => self.encode_vex_binop(ops, 0x66, 0x56),
            "vorps" => self.encode_vex_binop_np(ops, 0x56),
            "vbroadcastss" => self.encode_vex_unary_38(ops, 0x66, 0x18),
            "vbroadcastsd" => self.encode_vex_unary_38(ops, 0x66, 0x19),
            "vpbroadcastd" => self.encode_vex_unary_38(ops, 0x66, 0x58),
            "vpbroadcastb" => self.encode_vex_unary_38(ops, 0x66, 0x78),
            "vzeroupper" => { self.bytes.extend_from_slice(&[0xC5, 0xF8, 0x77]); Ok(()) }
            "vzeroall" => { self.bytes.extend_from_slice(&[0xC5, 0xFC, 0x77]); Ok(()) }
            "vinserti128" => self.encode_vex_insert128(ops, 0x66, 0x38),
            "vextracti128" => self.encode_vex_extract128(ops, 0x66, 0x39),
            "vperm2i128" => self.encode_vex_perm2(ops, 0x66, 0x46),
            "vperm2f128" => self.encode_vex_perm2(ops, 0x66, 0x06),
            "vinsertf128" => self.encode_vex_insert128(ops, 0x66, 0x18),
            "vextractf128" => self.encode_vex_extract128(ops, 0x66, 0x19),
            "vpermq" => self.encode_vex_unary_imm8_w1(ops, 0x66, 0x00),
            "vpermd" => self.encode_vex_binop_38(ops, 0x66, 0x36),
            "vpminub" => self.encode_vex_binop(ops, 0x66, 0xDA),
            "vpmaxub" => self.encode_vex_binop(ops, 0x66, 0xDE),
            "vpminsd" => self.encode_vex_binop_38(ops, 0x66, 0x39),
            "vpmaxsd" => self.encode_vex_binop_38(ops, 0x66, 0x3D),
            "vpmovmskb" => self.encode_vex_extract_gp(ops, 0x66, 0xD7),
            "vpackuswb" => self.encode_vex_binop(ops, 0x66, 0x67),
            "vpackssdw" => self.encode_vex_binop(ops, 0x66, 0x6B),
            "vpmaddwd" => self.encode_vex_binop(ops, 0x66, 0xF5),
            "vpmaddubsw" => self.encode_vex_binop_38(ops, 0x66, 0x04),
            "vpsadbw" => self.encode_vex_binop(ops, 0x66, 0xF6),
            "vpavgb" => self.encode_vex_binop(ops, 0x66, 0xE0),
            "vpavgw" => self.encode_vex_binop(ops, 0x66, 0xE3),
            "vpmuludq" => self.encode_vex_binop(ops, 0x66, 0xF4),
            "vpmulhw" => self.encode_vex_binop(ops, 0x66, 0xE5),
            "vpmulhuw" => self.encode_vex_binop(ops, 0x66, 0xE4),
            "vpalignr" => self.encode_vex_ternary_imm8(ops, 0x66, 0x0F),
            "vpblendw" => self.encode_vex_ternary_imm8(ops, 0x66, 0x0E),
            "vpblendvb" => self.encode_vex_blend4(ops, 0x66, 0x4C),
            "vblendvpd" => self.encode_vex_blend4(ops, 0x66, 0x4B),
            "vblendvps" => self.encode_vex_blend4(ops, 0x66, 0x4A),
            "vmovd" => self.encode_vex_movd(ops),
            "vmovq" => self.encode_vex_movq(ops),
            "vcvtsi2sdq" => self.encode_vex_cvt_gp(ops, 0xF2, 0x2A, true),
            "vcvtsi2sdl" => self.encode_vex_cvt_gp(ops, 0xF2, 0x2A, false),
            "vcvtsi2ssq" => self.encode_vex_cvt_gp(ops, 0xF3, 0x2A, true),
            "vcvtsi2ssl" => self.encode_vex_cvt_gp(ops, 0xF3, 0x2A, false),
            "vucomisd" => self.encode_vex_unary_op(ops, 0x66, 0x2E),
            "vucomiss" => self.encode_vex_unary_op(ops, 0x00, 0x2E),
            "vcvttsd2siq" => self.encode_vex_cvt_to_gp(ops, 0xF2, 0x2C, true),
            "vcvttss2siq" => self.encode_vex_cvt_to_gp(ops, 0xF3, 0x2C, true),
            "vcvttsd2sil" => self.encode_vex_cvt_to_gp(ops, 0xF2, 0x2C, false),
            "vcvttss2sil" => self.encode_vex_cvt_to_gp(ops, 0xF3, 0x2C, false),
            "vcvtsd2ss" => self.encode_vex_binop(ops, 0xF2, 0x5A),
            "vcvtss2sd" => self.encode_vex_binop(ops, 0xF3, 0x5A),
            "vminsd" => self.encode_vex_binop(ops, 0xF2, 0x5D),
            "vmaxsd" => self.encode_vex_binop(ops, 0xF2, 0x5F),
            "vminss" => self.encode_vex_binop(ops, 0xF3, 0x5D),
            "vmaxss" => self.encode_vex_binop(ops, 0xF3, 0x5F),
            "vsqrtsd" => self.encode_vex_binop(ops, 0xF2, 0x51),
            "vsqrtss" => self.encode_vex_binop(ops, 0xF3, 0x51),
            "vmovss" => self.encode_vex_mov_ss_sd(ops, 0xF3),
            "vmovsd" => self.encode_vex_mov_ss_sd(ops, 0xF2),
            "vpmovsxbw" => self.encode_vex_unary_38_op(ops, 0x66, 0x20),
            "vpmovsxwd" => self.encode_vex_unary_38_op(ops, 0x66, 0x23),
            "vpmovsxdq" => self.encode_vex_unary_38_op(ops, 0x66, 0x25),
            "vpmovzxbw" => self.encode_vex_unary_38_op(ops, 0x66, 0x30),
            "vpmovzxwd" => self.encode_vex_unary_38_op(ops, 0x66, 0x33),
            "vpmovzxdq" => self.encode_vex_unary_38_op(ops, 0x66, 0x35),
            "vpsubusb" => self.encode_vex_binop(ops, 0x66, 0xD8),
            "vpsubusw" => self.encode_vex_binop(ops, 0x66, 0xD9),
            "vpaddsb" => self.encode_vex_binop(ops, 0x66, 0xEC),
            "vpaddsw" => self.encode_vex_binop(ops, 0x66, 0xED),
            "vpaddusb" => self.encode_vex_binop(ops, 0x66, 0xDC),
            "vpaddusw" => self.encode_vex_binop(ops, 0x66, 0xDD),
            "vpsubsb" => self.encode_vex_binop(ops, 0x66, 0xE8),
            "vpsubsw" => self.encode_vex_binop(ops, 0x66, 0xE9),
            "vpackusdw" => self.encode_vex_binop_38(ops, 0x66, 0x2B),
            "vpacksswb" => self.encode_vex_binop(ops, 0x66, 0x63),
            "vpcmpeqw" => self.encode_vex_binop(ops, 0x66, 0x75),
            "vpcmpgtw" => self.encode_vex_binop(ops, 0x66, 0x65),
            "vpinsrw" => self.encode_vex_insert_gp(ops, 0x66, 0xC4),
            "vpextrw" => self.encode_vex_extract_gp(ops, 0x66, 0xC5),
            "vpinsrd" => self.encode_vex_insert_gp_3a(ops, 0x66, 0x22),
            "vpextrd" => self.encode_vex_extract_gp_3a(ops, 0x66, 0x16),
            "vpinsrb" => self.encode_vex_insert_gp_3a(ops, 0x66, 0x20),
            "vpextrb" => self.encode_vex_extract_gp_3a(ops, 0x66, 0x14),
            "vpinsrq" => self.encode_vex_insert_gp_3a_w1(ops, 0x66, 0x22),
            "vpextrq" => self.encode_vex_extract_gp_3a_w1(ops, 0x66, 0x16),
            "vshufpd" => self.encode_vex_ternary_imm8(ops, 0x66, 0xC6),
            "vshufps" => self.encode_vex_ternary_imm8_np(ops, 0xC6),
            "vunpcklpd" => self.encode_vex_binop(ops, 0x66, 0x14),
            "vunpckhpd" => self.encode_vex_binop(ops, 0x66, 0x15),
            "vunpcklps" => self.encode_vex_binop_np(ops, 0x14),
            "vunpckhps" => self.encode_vex_binop_np(ops, 0x15),
            "vpminsb" => self.encode_vex_binop_38(ops, 0x66, 0x38),
            "vpminuw" => self.encode_vex_binop_38(ops, 0x66, 0x3A),
            "vpminud" => self.encode_vex_binop_38(ops, 0x66, 0x3B),
            "vpmaxsb" => self.encode_vex_binop_38(ops, 0x66, 0x3C),
            "vpmaxuw" => self.encode_vex_binop_38(ops, 0x66, 0x3E),
            "vpmaxud" => self.encode_vex_binop_38(ops, 0x66, 0x3F),
            "vpminsw" => self.encode_vex_binop(ops, 0x66, 0xEA),
            "vpmaxsw" => self.encode_vex_binop(ops, 0x66, 0xEE),
            "vpmuldq" => self.encode_vex_binop_38(ops, 0x66, 0x28),

            // Additional VEX instructions
            "vpclmulqdq" => self.encode_vex_ternary_imm8(ops, 0x66, 0x44),
            "vaesenclast" => self.encode_vex_binop_38(ops, 0x66, 0xDD),
            "vaesenc" => self.encode_vex_binop_38(ops, 0x66, 0xDC),
            "vaesdec" => self.encode_vex_binop_38(ops, 0x66, 0xDE),
            "vaesdeclast" => self.encode_vex_binop_38(ops, 0x66, 0xDF),
            "vpcmpeqq" => self.encode_vex_binop_38(ops, 0x66, 0x29),
            "vpcmpgtq" => self.encode_vex_binop_38(ops, 0x66, 0x37),
            "vbroadcasti128" => self.encode_vex_unary_38(ops, 0x66, 0x5A),
            "vpblendd" => self.encode_vex_ternary_imm8(ops, 0x66, 0x02),
            "vpbroadcastq" => self.encode_vex_unary_38(ops, 0x66, 0x59),
            "vpbroadcastw" => self.encode_vex_unary_38(ops, 0x66, 0x79),
            "vpabsb" => self.encode_vex_unary_38_op(ops, 0x66, 0x1C),
            "vpabsw" => self.encode_vex_unary_38_op(ops, 0x66, 0x1D),
            "vpabsd" => self.encode_vex_unary_38_op(ops, 0x66, 0x1E),
            "vptest" => self.encode_vex_unary_38_op(ops, 0x66, 0x17),
            "vpxord" => self.encode_vex_binop(ops, 0x66, 0xEF), // Same as vpxor for VEX encoding

            // BMI instructions (VEX-encoded)
            "rorx" | "rorxl" | "rorxq" => self.encode_rorx(ops, mnemonic),
            "andn" | "andnl" | "andnq" => self.encode_andn(ops, mnemonic),

            // Additional missing instructions
            "movabs" | "movabsq" => self.encode_movabs(ops),
            "leave" | "leaveq" => { self.bytes.push(0xC9); Ok(()) }
            "fwait" | "wait" => { self.bytes.push(0x9B); Ok(()) }
            "movntil" | "movntiq" => self.encode_movnti(ops),
            "sldt" => self.encode_sldt(ops),
            "prefetcht0" => self.encode_prefetch(ops, 1),
            "prefetcht1" => self.encode_prefetch(ops, 2),
            "prefetcht2" => self.encode_prefetch(ops, 3),
            "prefetchnta" => self.encode_prefetch(ops, 0),
            "rdrand" | "rdrandl" | "rdrandq" => self.encode_rdrand(ops, mnemonic),
            "popcntw" => self.encode_popcnt(ops, 2),
            "loop" => self.encode_loop(ops, 0xE2),
            "loope" | "loopz" => self.encode_loop(ops, 0xE1),
            "loopne" | "loopnz" => self.encode_loop(ops, 0xE0),
            "jrcxz" => self.encode_loop(ops, 0xE3),
            "pblendvb" => self.encode_pblendvb(ops),

            // Suffix-less instructions (infer size from register operands)
            // These appear in inline assembly without AT&T suffixes.
            // Guard: only match when register operands are present so size can be inferred.
            "mov" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_mov(ops, s) }
            "add" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("add{}", size_suffix(s)); self.encode_alu(ops, &m, 0) }
            "sub" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("sub{}", size_suffix(s)); self.encode_alu(ops, &m, 5) }
            "cmp" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("cmp{}", size_suffix(s)); self.encode_alu(ops, &m, 7) }
            "and" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("and{}", size_suffix(s)); self.encode_alu(ops, &m, 4) }
            "or" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("or{}", size_suffix(s)); self.encode_alu(ops, &m, 1) }
            "xor" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("xor{}", size_suffix(s)); self.encode_alu(ops, &m, 6) }
            "test" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("test{}", size_suffix(s)); self.encode_test(ops, &m) }
            "adc" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("adc{}", size_suffix(s)); self.encode_alu(ops, &m, 2) }
            "sbb" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); let m = format!("sbb{}", size_suffix(s)); self.encode_alu(ops, &m, 3) }
            "neg" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 3, s) }
            "not" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 2, s) }
            "inc" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 0, s) }
            "dec" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 1, s) }
            "mul" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 4, s) }
            "div" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 6, s) }
            "idiv" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_unary_rm(ops, 7, s) }
            "imul" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_imul(ops, s) }
            "shl" | "sal" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 4, s) }
            "shr" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 5, s) }
            "sar" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 7, s) }
            "rol" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 0, s) }
            "ror" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 1, s) }
            "rcl" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 2, s) }
            "rcr" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_shift_infer(ops, 3, s) }
            "lea" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_lea(ops, s) }
            "push" => self.encode_push(ops),
            "pop" => self.encode_pop(ops),
            "xchg" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_xchg(ops, &format!("xchg{}", size_suffix(s))) }
            "bsf" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_bitscan(ops, &[0x0F, 0xBC], s) }
            "bsr" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_bitscan(ops, &[0x0F, 0xBD], s) }
            "bt" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_bt(ops, 0, s) }
            "bts" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_bt(ops, 5, s) }
            "btr" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_bt(ops, 6, s) }
            "btc" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_bt(ops, 7, s) }
            "shld" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_double_shift(ops, 0xA4, s) }
            "shrd" if infer_size_from_ops(ops).is_some() => { let s = infer_size_from_ops(ops).unwrap(); self.encode_double_shift(ops, 0xAC, s) }

            // Suffix-less conditional moves (from inline asm)
            "cmovz" | "cmove" | "cmovnz" | "cmovne"
            | "cmovl" | "cmovle" | "cmovg" | "cmovge"
            | "cmovb" | "cmovbe" | "cmova" | "cmovae"
            | "cmovc" | "cmovnc" | "cmovs" | "cmovns"
            | "cmovo" | "cmovno" | "cmovp" | "cmovnp"
            | "cmovnae" | "cmovna" if infer_size_from_ops(ops).is_some() => {
                let size = infer_size_from_ops(ops).unwrap();
                let suffix = size_suffix(size);
                let suffixed = format!("{}{}", mnemonic, suffix);
                self.encode_cmovcc(ops, &suffixed)
            }

            // Missing adcw/sbbw
            "adcw" => self.encode_alu(ops, mnemonic, 2),
            "sbbw" => self.encode_alu(ops, mnemonic, 3),
            "adcb" => self.encode_alu(ops, mnemonic, 2),
            "sbbb" => self.encode_alu(ops, mnemonic, 3),

            // CBW/CWDE/CWD
            "cwtl" | "cwde" => { self.bytes.push(0x98); Ok(()) }
            "cbtw" | "cbw" => { self.bytes.extend_from_slice(&[0x66, 0x98]); Ok(()) }
            "cwtd" | "cwd" => { self.bytes.extend_from_slice(&[0x66, 0x99]); Ok(()) }

            // xgetbv/xsetbv
            "xgetbv" => { self.bytes.extend_from_slice(&[0x0F, 0x01, 0xD0]); Ok(()) }

            // wrmsr/rdmsr
            "wrmsr" => { self.bytes.extend_from_slice(&[0x0F, 0x30]); Ok(()) }
            "rdmsr" => { self.bytes.extend_from_slice(&[0x0F, 0x32]); Ok(()) }

            // IN/OUT port instructions
            "inb" => self.encode_in_out(ops, 0xEC, 0xE4, 1),
            "inw" => self.encode_in_out(ops, 0xED, 0xE5, 2),
            "inl" => self.encode_in_out(ops, 0xED, 0xE5, 4),
            "outb" => self.encode_out(ops, 0xEE, 0xE6, 1),
            "outw" => self.encode_out(ops, 0xEF, 0xE7, 2),
            "outl" => self.encode_out(ops, 0xEF, 0xE7, 4),

            // Standalone prefix mnemonics (e.g. from "rep; nop" split on semicolon)
            "rep" | "repe" | "repz" if ops.is_empty() => { self.bytes.push(0xF3); Ok(()) }
            "repnz" | "repne" if ops.is_empty() => { self.bytes.push(0xF2); Ok(()) }
            "lock" if ops.is_empty() => { self.bytes.push(0xF0); Ok(()) }

            _ => {
                // TODO: many more instructions to handle
                Err(format!("unhandled instruction: {} {:?}", mnemonic, ops))
            }
        }
    }

    // ---- Encoding helpers ----

    /// Build a REX prefix byte.
    fn rex(&self, w: bool, r: bool, x: bool, b: bool) -> u8 {
        let mut rex = 0x40u8;
        if w { rex |= 0x08; }
        if r { rex |= 0x04; }
        if x { rex |= 0x02; }
        if b { rex |= 0x01; }
        rex
    }

    /// Encode ModR/M byte.
    fn modrm(&self, mod_: u8, reg: u8, rm: u8) -> u8 {
        (mod_ << 6) | ((reg & 7) << 3) | (rm & 7)
    }

    /// Encode SIB byte.
    fn sib(&self, scale: u8, index: u8, base: u8) -> u8 {
        let scale_bits = match scale {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        };
        (scale_bits << 6) | ((index & 7) << 3) | (base & 7)
    }

    /// Emit REX prefix if needed for reg-reg operation.
    fn emit_rex_rr(&mut self, size: u8, reg: &str, rm: &str) {
        let w = size == 8;
        let r = needs_rex_ext(reg);
        let b = needs_rex_ext(rm);
        let need_rex = w || r || b || is_rex_required_8bit(reg) || is_rex_required_8bit(rm);
        if need_rex {
            self.bytes.push(self.rex(w, r, false, b));
        }
    }

    /// Emit segment override prefix (0x64 for %fs, 0x65 for %gs) if present.
    /// Must be emitted before any operand-size override, REX prefix, or opcode.
    // TODO: emit_segment_prefix is currently only called in mov (reg-mem, mem-reg) and ALU ops.
    // Other instruction families that accept memory operands should also call this.
    fn emit_segment_prefix(&mut self, mem: &MemoryOperand) -> Result<(), String> {
        if let Some(ref seg) = mem.segment {
            match seg.as_str() {
                "fs" => self.bytes.push(0x64),
                "gs" => self.bytes.push(0x65),
                _ => return Err(format!("unsupported segment override: %{}", seg)),
            }
        }
        Ok(())
    }

    /// Emit REX prefix for a memory operand where 'reg' is the reg field.
    fn emit_rex_rm(&mut self, size: u8, reg: &str, mem: &MemoryOperand) {
        let w = size == 8;
        let r = needs_rex_ext(reg);
        let b = mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name));
        let x = mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name));
        let need_rex = w || r || b || x || is_rex_required_8bit(reg);
        if need_rex {
            self.bytes.push(self.rex(w, r, x, b));
        }
    }

    /// Emit REX prefix for unary operation on register.
    fn emit_rex_unary(&mut self, size: u8, rm: &str) {
        let w = size == 8;
        let b = needs_rex_ext(rm);
        let need_rex = w || b || is_rex_required_8bit(rm);
        if need_rex {
            self.bytes.push(self.rex(w, false, false, b));
        }
    }

    /// Encode ModR/M + SIB + displacement for a memory operand.
    /// Returns the bytes to append. `reg_field` is the /r value (3 bits).
    fn encode_modrm_mem(&mut self, reg_field: u8, mem: &MemoryOperand) -> Result<(), String> {
        let base = mem.base.as_ref();
        let index = mem.index.as_ref();

        // RIP-relative addressing
        if let Some(base_reg) = base {
            if base_reg.name == "rip" {
                // ModR/M: mod=00, rm=101 (RIP-relative)
                self.bytes.push(self.modrm(0, reg_field, 5));
                // 32-bit displacement (will be filled by relocation)
                match &mem.displacement {
                    Displacement::Symbol(sym) => {
                        self.add_relocation(sym, R_X86_64_PC32, -4);
                        self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                    }
                    Displacement::SymbolAddend(sym, addend) => {
                        self.add_relocation(sym, R_X86_64_PC32, *addend - 4);
                        self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                    }
                    Displacement::SymbolPlusOffset(sym, offset) => {
                        self.add_relocation(sym, R_X86_64_PC32, *offset - 4);
                        self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                    }
                    Displacement::SymbolMod(sym, modifier) => {
                        let reloc_type = match modifier.as_str() {
                            "GOTPCREL" => R_X86_64_GOTPCREL,
                            "GOTTPOFF" => R_X86_64_GOTTPOFF,
                            "PLT" => R_X86_64_PLT32,
                            _ => R_X86_64_PC32,
                        };
                        self.add_relocation(sym, reloc_type, -4);
                        self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                    }
                    Displacement::Integer(val) => {
                        self.bytes.extend_from_slice(&(*val as i32).to_le_bytes());
                    }
                    Displacement::None => {
                        self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                    }
                }
                return Ok(());
            }
        }

        // Handle symbol displacements that need relocations.
        // We defer emitting the relocation until after the ModR/M and SIB bytes
        // so the relocation offset correctly points to the displacement bytes.
        let (disp_val, has_symbol, deferred_reloc) = match &mem.displacement {
            Displacement::None => (0i64, false, None),
            Displacement::Integer(v) => (*v, false, None),
            Displacement::Symbol(sym) => {
                (0i64, true, Some((sym.clone(), R_X86_64_32S, 0i64)))
            }
            Displacement::SymbolAddend(sym, addend) => {
                (0i64, true, Some((sym.clone(), R_X86_64_32S, *addend)))
            }
            Displacement::SymbolPlusOffset(sym, offset) => {
                (0i64, true, Some((sym.clone(), R_X86_64_32S, *offset)))
            }
            Displacement::SymbolMod(sym, modifier) => {
                let reloc_type = match modifier.as_str() {
                    "TPOFF" => R_X86_64_TPOFF32,
                    "GOTPCREL" => R_X86_64_GOTPCREL,
                    _ => R_X86_64_32S,
                };
                (0i64, true, Some((sym.clone(), reloc_type, 0i64)))
            }
        };

        // No base register - need SIB with no-base encoding
        if base.is_none() && index.is_none() {
            // Direct memory reference - mod=00, rm=100 (SIB), SIB: base=101 (no base)
            self.bytes.push(self.modrm(0, reg_field, 4));
            self.bytes.push(self.sib(1, 4, 5)); // index=100 (none), base=101 (disp32)
            if let Some((sym, reloc_type, addend)) = deferred_reloc {
                self.add_relocation(&sym, reloc_type, addend);
            }
            self.bytes.extend_from_slice(&(disp_val as i32).to_le_bytes());
            return Ok(());
        }

        let base_reg = base.map(|r| &r.name as &str).unwrap_or("");
        let base_num = if !base_reg.is_empty() { reg_num(base_reg).unwrap_or(0) } else { 5 };

        // Determine if we need SIB
        let need_sib = index.is_some()
            || (base_num & 7) == 4  // RSP/R12 always need SIB
            || base.is_none();

        // Determine displacement size
        let (mod_bits, disp_size) = if has_symbol {
            (2, 4) // always use disp32 for symbols
        } else if disp_val == 0 && (base_num & 7) != 5 {
            // No displacement (RBP/R13 always need at least disp8)
            (0, 0)
        } else if disp_val >= -128 && disp_val <= 127 {
            (1, 1) // disp8
        } else {
            (2, 4) // disp32
        };

        if need_sib {
            let idx = index.as_ref();
            let idx_num = idx.map(|r| reg_num(&r.name).unwrap_or(4)).unwrap_or(4); // 4 = no index
            let scale = mem.scale.unwrap_or(1);

            if base.is_none() {
                // No base - disp32 with SIB
                self.bytes.push(self.modrm(0, reg_field, 4));
                self.bytes.push(self.sib(scale, idx_num, 5));
                if let Some((sym, reloc_type, addend)) = deferred_reloc {
                    self.add_relocation(&sym, reloc_type, addend);
                }
                self.bytes.extend_from_slice(&(disp_val as i32).to_le_bytes());
            } else {
                self.bytes.push(self.modrm(mod_bits, reg_field, 4));
                self.bytes.push(self.sib(scale, idx_num, base_num));
                if let Some((sym, reloc_type, addend)) = deferred_reloc {
                    self.add_relocation(&sym, reloc_type, addend);
                }
                match disp_size {
                    0 => {}
                    1 => self.bytes.push(disp_val as u8),
                    4 => self.bytes.extend_from_slice(&(disp_val as i32).to_le_bytes()),
                    _ => unreachable!(),
                }
            }
        } else {
            self.bytes.push(self.modrm(mod_bits, reg_field, base_num));
            if let Some((sym, reloc_type, addend)) = deferred_reloc {
                self.add_relocation(&sym, reloc_type, addend);
            }
            match disp_size {
                0 => {}
                1 => self.bytes.push(disp_val as u8),
                4 => self.bytes.extend_from_slice(&(disp_val as i32).to_le_bytes()),
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    /// Add a relocation relative to current position.
    fn add_relocation(&mut self, symbol: &str, reloc_type: u32, addend: i64) {
        self.relocations.push(Relocation {
            offset: self.offset + self.bytes.len() as u64 - (self.offset), // adjusted in caller
            symbol: symbol.to_string(),
            reloc_type,
            addend,
        });
    }

    // ---- Instruction-specific encoders ----

    fn encode_mov(&mut self, ops: &[Operand], size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err(format!("mov requires 2 operands, got {}", ops.len()));
        }

        match (&ops[0], &ops[1]) {
            // mov $imm, %reg
            (Operand::Immediate(imm), Operand::Register(dst)) => {
                self.encode_mov_imm_reg(imm, dst, size)
            }
            // mov %reg, %reg
            (Operand::Register(src), Operand::Register(dst)) => {
                self.encode_mov_rr(src, dst, size)
            }
            // mov mem, %reg
            (Operand::Memory(mem), Operand::Register(dst)) => {
                self.encode_mov_mem_reg(mem, dst, size)
            }
            // mov %reg, mem
            (Operand::Register(src), Operand::Memory(mem)) => {
                self.encode_mov_reg_mem(src, mem, size)
            }
            // mov $imm, mem
            (Operand::Immediate(imm), Operand::Memory(mem)) => {
                self.encode_mov_imm_mem(imm, mem, size)
            }
            _ => Err(format!("unsupported mov operand combination")),
        }
    }

    fn encode_mov_imm_reg(&mut self, imm: &ImmediateValue, dst: &Register, size: u8) -> Result<(), String> {
        let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;

        match imm {
            ImmediateValue::Integer(val) => {
                let val = *val;
                if size == 8 {
                    // For 64-bit: if value fits in signed 32-bit, use movq $imm32, %reg (sign-extended)
                    if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
                        self.emit_rex_unary(8, &dst.name);
                        self.bytes.push(0xC7);
                        self.bytes.push(self.modrm(3, 0, dst_num));
                        self.bytes.extend_from_slice(&(val as i32).to_le_bytes());
                    } else {
                        // Need movabsq for 64-bit immediate
                        let b = needs_rex_ext(&dst.name);
                        self.bytes.push(self.rex(true, false, false, b));
                        self.bytes.push(0xB8 + (dst_num & 7));
                        self.bytes.extend_from_slice(&val.to_le_bytes());
                    }
                } else if size == 4 {
                    if needs_rex_ext(&dst.name) {
                        self.bytes.push(self.rex(false, false, false, true));
                    }
                    self.bytes.push(0xC7);
                    self.bytes.push(self.modrm(3, 0, dst_num));
                    self.bytes.extend_from_slice(&(val as i32).to_le_bytes());
                } else if size == 2 {
                    self.bytes.push(0x66); // operand size prefix
                    if needs_rex_ext(&dst.name) {
                        self.bytes.push(self.rex(false, false, false, true));
                    }
                    self.bytes.push(0xC7);
                    self.bytes.push(self.modrm(3, 0, dst_num));
                    self.bytes.extend_from_slice(&(val as i16).to_le_bytes());
                } else {
                    // 8-bit
                    if needs_rex_ext(&dst.name) || is_rex_required_8bit(&dst.name) {
                        self.bytes.push(self.rex(false, false, false, needs_rex_ext(&dst.name)));
                    }
                    self.bytes.push(0xC6);
                    self.bytes.push(self.modrm(3, 0, dst_num));
                    self.bytes.push(val as u8);
                }
            }
            ImmediateValue::Symbol(sym) => {
                // movq $symbol, %reg - load address
                if size == 8 {
                    self.emit_rex_unary(8, &dst.name);
                    self.bytes.push(0xC7);
                    self.bytes.push(self.modrm(3, 0, dst_num));
                    self.add_relocation(sym, R_X86_64_32S, 0);
                    self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    self.emit_rex_unary(size, &dst.name);
                    self.bytes.push(0xC7);
                    self.bytes.push(self.modrm(3, 0, dst_num));
                    self.add_relocation(sym, R_X86_64_32, 0);
                    self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
            ImmediateValue::SymbolMod(_, _) => {
                Err(format!("unsupported immediate type for mov"))?
            }
        }
        Ok(())
    }

    fn encode_mov_rr(&mut self, src: &Register, dst: &Register, size: u8) -> Result<(), String> {
        let src_num = reg_num(&src.name).ok_or_else(|| format!("bad register: {}", src.name))?;
        let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;

        if size == 2 {
            self.bytes.push(0x66);
        }
        self.emit_rex_rr(size, &src.name, &dst.name);
        if size == 1 {
            self.bytes.push(0x88);
        } else {
            self.bytes.push(0x89);
        }
        self.bytes.push(self.modrm(3, src_num, dst_num));
        Ok(())
    }

    fn encode_mov_mem_reg(&mut self, mem: &MemoryOperand, dst: &Register, size: u8) -> Result<(), String> {
        let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;

        // Handle segment prefix
        self.emit_segment_prefix(mem)?;

        if size == 2 {
            self.bytes.push(0x66);
        }
        self.emit_rex_rm(size, &dst.name, mem);
        if size == 1 {
            self.bytes.push(0x8A);
        } else {
            self.bytes.push(0x8B);
        }
        self.encode_modrm_mem(dst_num, mem)
    }

    fn encode_mov_reg_mem(&mut self, src: &Register, mem: &MemoryOperand, size: u8) -> Result<(), String> {
        let src_num = reg_num(&src.name).ok_or_else(|| format!("bad register: {}", src.name))?;

        self.emit_segment_prefix(mem)?;

        if size == 2 {
            self.bytes.push(0x66);
        }
        self.emit_rex_rm(size, &src.name, mem);
        if size == 1 {
            self.bytes.push(0x88);
        } else {
            self.bytes.push(0x89);
        }
        self.encode_modrm_mem(src_num, mem)
    }

    fn encode_mov_imm_mem(&mut self, imm: &ImmediateValue, mem: &MemoryOperand, size: u8) -> Result<(), String> {
        if size == 2 {
            self.bytes.push(0x66);
        }
        // Use an empty string for REX calculation since the reg field is /0
        self.emit_rex_rm(if size == 8 { 8 } else { size }, "", mem);
        if size == 1 {
            self.bytes.push(0xC6);
        } else {
            self.bytes.push(0xC7);
        }
        self.encode_modrm_mem(0, mem)?;

        match imm {
            ImmediateValue::Integer(val) => {
                match size {
                    1 => self.bytes.push(*val as u8),
                    2 => self.bytes.extend_from_slice(&(*val as i16).to_le_bytes()),
                    4 | 8 => self.bytes.extend_from_slice(&(*val as i32).to_le_bytes()),
                    _ => unreachable!(),
                }
            }
            _ => return Err("unsupported immediate for mov to memory".to_string()),
        }
        Ok(())
    }

    fn encode_movabs(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movabsq requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(val)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let b = needs_rex_ext(&dst.name);
                self.bytes.push(self.rex(true, false, false, b));
                self.bytes.push(0xB8 + (dst_num & 7));
                self.bytes.extend_from_slice(&val.to_le_bytes());
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Symbol(sym)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let b = needs_rex_ext(&dst.name);
                self.bytes.push(self.rex(true, false, false, b));
                self.bytes.push(0xB8 + (dst_num & 7));
                self.add_relocation(sym, R_X86_64_64, 0);
                self.bytes.extend_from_slice(&[0u8; 8]);
                Ok(())
            }
            _ => Err("unsupported movabsq operands".to_string()),
        }
    }

    fn encode_movsx(&mut self, ops: &[Operand], src_size: u8, dst_size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movsx requires 2 operands".to_string());
        }

        let opcode = match (src_size, dst_size) {
            (1, _) => vec![0x0F, 0xBE],   // movsbq/movsbl/movsbw
            (2, _) => vec![0x0F, 0xBF],   // movswq/movswl/movsww
            (4, 8) => vec![0x63],          // movslq (movsxd)
            _ => return Err(format!("unsupported movsx combination: {} -> {}", src_size, dst_size)),
        };

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad src register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                if dst_size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rr(dst_size, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                if dst_size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(dst_size, &dst.name, mem);
                self.bytes.extend_from_slice(&opcode);
                self.encode_modrm_mem(dst_num, mem)?;
            }
            _ => return Err("unsupported movsx operands".to_string()),
        }
        Ok(())
    }

    fn encode_movzx(&mut self, ops: &[Operand], src_size: u8, dst_size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movzx requires 2 operands".to_string());
        }

        let opcode = match src_size {
            1 => vec![0x0F, 0xB6],  // movzbl/movzbq/movzbw
            2 => vec![0x0F, 0xB7],  // movzwl/movzwq
            _ => return Err(format!("unsupported movzx src size: {}", src_size)),
        };

        // Note: movzbl zero-extends to 64 bits implicitly (32-bit op clears upper 32)
        // So we use size=4 for REX calculation unless dst is an extended register needing REX.B
        let rex_size = if dst_size == 8 { 8 } else if dst_size == 2 { 2 } else { 4 };

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad src register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                if dst_size == 2 { self.bytes.push(0x66); }
                // movzbl uses 32-bit destination but we need REX if either operand is extended
                self.emit_rex_rr(rex_size, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                if dst_size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(rex_size, &dst.name, mem);
                self.bytes.extend_from_slice(&opcode);
                self.encode_modrm_mem(dst_num, mem)?;
            }
            _ => return Err("unsupported movzx operands".to_string()),
        }
        Ok(())
    }

    fn encode_lea(&mut self, ops: &[Operand], size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("lea requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                self.emit_rex_rm(size, &dst.name, mem);
                self.bytes.push(0x8D);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("lea requires memory source and register destination".to_string()),
        }
    }

    fn encode_push(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("push requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let num = reg_num(&reg.name).ok_or("bad register")?;
                if needs_rex_ext(&reg.name) {
                    self.bytes.push(self.rex(false, false, false, true));
                }
                self.bytes.push(0x50 + (num & 7));
                Ok(())
            }
            Operand::Immediate(ImmediateValue::Integer(val)) => {
                if *val >= -128 && *val <= 127 {
                    self.bytes.push(0x6A);
                    self.bytes.push(*val as u8);
                } else {
                    self.bytes.push(0x68);
                    self.bytes.extend_from_slice(&(*val as i32).to_le_bytes());
                }
                Ok(())
            }
            Operand::Memory(mem) => {
                self.emit_rex_rm(0, "", mem);
                self.bytes.push(0xFF);
                self.encode_modrm_mem(6, mem)
            }
            _ => Err("unsupported push operand".to_string()),
        }
    }

    fn encode_pop(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("pop requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let num = reg_num(&reg.name).ok_or("bad register")?;
                if needs_rex_ext(&reg.name) {
                    self.bytes.push(self.rex(false, false, false, true));
                }
                self.bytes.push(0x58 + (num & 7));
                Ok(())
            }
            _ => Err("unsupported pop operand".to_string()),
        }
    }

    /// Encode ALU operations (add/or/adc/sbb/and/sub/xor/cmp).
    /// `alu_op` is the operation number (0-7).
    fn encode_alu(&mut self, ops: &[Operand], mnemonic: &str, alu_op: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err(format!("{} requires 2 operands", mnemonic));
        }

        let size = mnemonic_size_suffix(mnemonic).unwrap_or(8);

        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(val)), Operand::Register(dst)) => {
                let val = *val;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;

                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);

                if size == 1 {
                    // 8-bit ALU with imm8
                    self.bytes.push(0x80);
                    self.bytes.push(self.modrm(3, alu_op, dst_num));
                    self.bytes.push(val as u8);
                } else if val >= -128 && val <= 127 {
                    // Sign-extended imm8
                    self.bytes.push(0x83);
                    self.bytes.push(self.modrm(3, alu_op, dst_num));
                    self.bytes.push(val as u8);
                } else {
                    // imm32
                    if dst_num == 0 && !needs_rex_ext(&dst.name) {
                        // Special short form: op eax/rax, imm32
                        self.bytes.push(if size == 1 { 0x04 } else { 0x05 } + alu_op * 8);
                    } else {
                        self.bytes.push(0x81);
                        self.bytes.push(self.modrm(3, alu_op, dst_num));
                    }
                    if size == 2 {
                        self.bytes.extend_from_slice(&(val as i16).to_le_bytes());
                    } else {
                        self.bytes.extend_from_slice(&(val as i32).to_le_bytes());
                    }
                }
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad src register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;

                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rr(size, &src.name, &dst.name);
                self.bytes.push(if size == 1 { 0x00 } else { 0x01 } + alu_op * 8);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                self.emit_segment_prefix(mem)?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &dst.name, mem);
                self.bytes.push(if size == 1 { 0x02 } else { 0x03 } + alu_op * 8);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or("bad src register")?;
                self.emit_segment_prefix(mem)?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &src.name, mem);
                self.bytes.push(if size == 1 { 0x00 } else { 0x01 } + alu_op * 8);
                self.encode_modrm_mem(src_num, mem)
            }
            (Operand::Immediate(ImmediateValue::Integer(val)), Operand::Memory(mem)) => {
                let val = *val;
                self.emit_segment_prefix(mem)?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, "", mem);

                if size == 1 {
                    self.bytes.push(0x80);
                    self.encode_modrm_mem(alu_op, mem)?;
                    self.bytes.push(val as u8);
                } else if val >= -128 && val <= 127 {
                    self.bytes.push(0x83);
                    self.encode_modrm_mem(alu_op, mem)?;
                    self.bytes.push(val as u8);
                } else {
                    self.bytes.push(0x81);
                    self.encode_modrm_mem(alu_op, mem)?;
                    if size == 2 {
                        self.bytes.extend_from_slice(&(val as i16).to_le_bytes());
                    } else {
                        self.bytes.extend_from_slice(&(val as i32).to_le_bytes());
                    }
                }
                Ok(())
            }
            _ => Err(format!("unsupported {} operands", mnemonic)),
        }
    }

    fn encode_test(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 2 {
            return Err(format!("{} requires 2 operands", mnemonic));
        }

        let size = mnemonic_size_suffix(mnemonic).unwrap_or(8);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad src register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rr(size, &src.name, &dst.name);
                self.bytes.push(if size == 1 { 0x84 } else { 0x85 });
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Integer(val)), Operand::Register(dst)) => {
                let val = *val;
                let dst_num = reg_num(&dst.name).ok_or("bad dst register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);

                if size == 1 {
                    if dst_num == 0 && !needs_rex_ext(&dst.name) {
                        self.bytes.push(0xA8);
                    } else {
                        self.bytes.push(0xF6);
                        self.bytes.push(self.modrm(3, 0, dst_num));
                    }
                    self.bytes.push(val as u8);
                } else {
                    if dst_num == 0 && !needs_rex_ext(&dst.name) {
                        self.bytes.push(0xA9);
                    } else {
                        self.bytes.push(0xF7);
                        self.bytes.push(self.modrm(3, 0, dst_num));
                    }
                    if size == 2 {
                        self.bytes.extend_from_slice(&(val as i16).to_le_bytes());
                    } else {
                        self.bytes.extend_from_slice(&(val as i32).to_le_bytes());
                    }
                }
                Ok(())
            }
            _ => Err(format!("unsupported test operands")),
        }
    }

    fn encode_imul(&mut self, ops: &[Operand], size: u8) -> Result<(), String> {
        match ops.len() {
            1 => {
                // One-operand form: imul r/m (result in rdx:rax)
                self.encode_unary_rm(ops, 5, size)
            }
            2 => {
                // Two-operand form: imul src, dst
                match (&ops[0], &ops[1]) {
                    (Operand::Register(src), Operand::Register(dst)) => {
                        let src_num = reg_num(&src.name).ok_or("bad register")?;
                        let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                        self.emit_rex_rr(size, &dst.name, &src.name);
                        self.bytes.extend_from_slice(&[0x0F, 0xAF]);
                        self.bytes.push(self.modrm(3, dst_num, src_num));
                        Ok(())
                    }
                    (Operand::Memory(mem), Operand::Register(dst)) => {
                        let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                        self.emit_rex_rm(size, &dst.name, mem);
                        self.bytes.extend_from_slice(&[0x0F, 0xAF]);
                        self.encode_modrm_mem(dst_num, mem)
                    }
                    _ => Err("unsupported imul operands".to_string()),
                }
            }
            3 => {
                // Three-operand form: imul $imm, src, dst
                match (&ops[0], &ops[1], &ops[2]) {
                    (Operand::Immediate(ImmediateValue::Integer(val)), Operand::Register(src), Operand::Register(dst)) => {
                        let src_num = reg_num(&src.name).ok_or("bad register")?;
                        let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                        self.emit_rex_rr(size, &dst.name, &src.name);
                        if *val >= -128 && *val <= 127 {
                            self.bytes.push(0x6B);
                            self.bytes.push(self.modrm(3, dst_num, src_num));
                            self.bytes.push(*val as u8);
                        } else {
                            self.bytes.push(0x69);
                            self.bytes.push(self.modrm(3, dst_num, src_num));
                            self.bytes.extend_from_slice(&(*val as i32).to_le_bytes());
                        }
                        Ok(())
                    }
                    _ => Err("unsupported imul operands".to_string()),
                }
            }
            _ => Err("imul requires 1-3 operands".to_string()),
        }
    }

    fn encode_unary_rm(&mut self, ops: &[Operand], op_ext: u8, size: u8) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("unary op requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let num = reg_num(&reg.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &reg.name);
                self.bytes.push(if size == 1 { 0xF6 } else { 0xF7 });
                self.bytes.push(self.modrm(3, op_ext, num));
                Ok(())
            }
            Operand::Memory(mem) => {
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, "", mem);
                self.bytes.push(if size == 1 { 0xF6 } else { 0xF7 });
                self.encode_modrm_mem(op_ext, mem)
            }
            _ => Err("unsupported unary operand".to_string()),
        }
    }

    fn encode_shift(&mut self, ops: &[Operand], mnemonic: &str, shift_op: u8) -> Result<(), String> {
        let size = mnemonic_size_suffix(mnemonic).unwrap_or(8);

        // Handle 1-operand form: shr %reg (implicit count of 1)
        if ops.len() == 1 {
            match &ops[0] {
                Operand::Register(dst) => {
                    let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                    if size == 2 { self.bytes.push(0x66); }
                    self.emit_rex_unary(size, &dst.name);
                    self.bytes.push(if size == 1 { 0xD0 } else { 0xD1 });
                    self.bytes.push(self.modrm(3, shift_op, dst_num));
                    return Ok(());
                }
                Operand::Memory(mem) => {
                    if size == 2 { self.bytes.push(0x66); }
                    self.emit_rex_rm(size, "", mem);
                    self.bytes.push(if size == 1 { 0xD0 } else { 0xD1 });
                    return self.encode_modrm_mem(shift_op, mem);
                }
                _ => return Err(format!("unsupported {} operand", mnemonic)),
            }
        }

        if ops.len() != 2 {
            return Err(format!("{} requires 1-2 operands", mnemonic));
        }

        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(count)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let count = *count as u8;

                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);

                if count == 1 {
                    self.bytes.push(if size == 1 { 0xD0 } else { 0xD1 });
                    self.bytes.push(self.modrm(3, shift_op, dst_num));
                } else {
                    self.bytes.push(if size == 1 { 0xC0 } else { 0xC1 });
                    self.bytes.push(self.modrm(3, shift_op, dst_num));
                    self.bytes.push(count);
                }
                Ok(())
            }
            (Operand::Register(cl), Operand::Register(dst)) if cl.name == "cl" => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);
                self.bytes.push(if size == 1 { 0xD2 } else { 0xD3 });
                self.bytes.push(self.modrm(3, shift_op, dst_num));
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Integer(count)), Operand::Memory(mem)) => {
                let count = *count as u8;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, "", mem);
                if count == 1 {
                    self.bytes.push(if size == 1 { 0xD0 } else { 0xD1 });
                    self.encode_modrm_mem(shift_op, mem)
                } else {
                    self.bytes.push(if size == 1 { 0xC0 } else { 0xC1 });
                    self.encode_modrm_mem(shift_op, mem)?;
                    self.bytes.push(count);
                    Ok(())
                }
            }
            (Operand::Register(cl), Operand::Memory(mem)) if cl.name == "cl" => {
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, "", mem);
                self.bytes.push(if size == 1 { 0xD2 } else { 0xD3 });
                self.encode_modrm_mem(shift_op, mem)
            }
            _ => Err(format!("unsupported {} operands", mnemonic)),
        }
    }

    fn encode_double_shift(&mut self, ops: &[Operand], opcode: u8, size: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("double shift requires 3 operands".to_string());
        }

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(count)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.emit_rex_rr(size, &src.name, &dst.name);
                self.bytes.extend_from_slice(&[0x0F, opcode]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                self.bytes.push(*count as u8);
                Ok(())
            }
            (Operand::Register(cl), Operand::Register(src), Operand::Register(dst)) if cl.name == "cl" => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.emit_rex_rr(size, &src.name, &dst.name);
                self.bytes.extend_from_slice(&[0x0F, opcode + 1]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            _ => Err("unsupported double shift operands".to_string()),
        }
    }

    fn encode_bswap(&mut self, ops: &[Operand], size: u8) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("bswap requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let num = reg_num(&reg.name).ok_or("bad register")?;
                self.emit_rex_unary(size, &reg.name);
                self.bytes.extend_from_slice(&[0x0F, 0xC8 + (num & 7)]);
                Ok(())
            }
            _ => Err("bswap requires register operand".to_string()),
        }
    }

    fn encode_bit_count(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 2 {
            return Err(format!("{} requires 2 operands", mnemonic));
        }

        let (prefix, opcode) = match mnemonic {
            "lzcntl" | "lzcntq" => (0xF3u8, [0x0F, 0xBD]),
            "tzcntl" | "tzcntq" => (0xF3, [0x0F, 0xBC]),
            "popcntl" | "popcntq" => (0xF3, [0x0F, 0xB8]),
            _ => return Err(format!("unknown bit count: {}", mnemonic)),
        };

        let size = mnemonic_size_suffix(mnemonic).unwrap_or(4);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(prefix);
                self.emit_rex_rr(size, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(prefix);
                self.emit_rex_rm(size, &dst.name, mem);
                self.bytes.extend_from_slice(&opcode);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err(format!("unsupported {} operands", mnemonic)),
        }
    }

    fn encode_setcc(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("setcc requires 1 operand".to_string());
        }

        // Strip "set" prefix and optional 'b' suffix
        let cc_part = &mnemonic[3..];
        let cc_part = if cc_part.ends_with('b') && cc_part.len() > 1 {
            &cc_part[..cc_part.len()-1]
        } else {
            cc_part
        };
        let cc = cc_from_mnemonic(cc_part)?;

        match &ops[0] {
            Operand::Register(reg) => {
                let num = reg_num(&reg.name).ok_or("bad register")?;
                if needs_rex_ext(&reg.name) || is_rex_required_8bit(&reg.name) {
                    self.bytes.push(self.rex(false, false, false, needs_rex_ext(&reg.name)));
                }
                self.bytes.extend_from_slice(&[0x0F, 0x90 + cc]);
                self.bytes.push(self.modrm(3, 0, num));
                Ok(())
            }
            Operand::Memory(mem) => {
                self.emit_rex_rm(0, "", mem);
                self.bytes.extend_from_slice(&[0x0F, 0x90 + cc]);
                self.encode_modrm_mem(0, mem)
            }
            _ => Err("setcc requires register or memory operand".to_string()),
        }
    }

    fn encode_cmovcc(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("cmovcc requires 2 operands".to_string());
        }

        // Extract condition code: strip "cmov" prefix and size suffix
        let without_prefix = &mnemonic[4..];
        let (cc_str, size) = if without_prefix.ends_with('q') {
            (&without_prefix[..without_prefix.len()-1], 8u8)
        } else if without_prefix.ends_with('l') {
            (&without_prefix[..without_prefix.len()-1], 4u8)
        } else {
            (without_prefix, 8u8) // default to 64-bit
        };
        let cc = cc_from_mnemonic(cc_str)?;

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.emit_rex_rr(size, &dst.name, &src.name);
                self.bytes.extend_from_slice(&[0x0F, 0x40 + cc]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported cmov operands".to_string()),
        }
    }

    fn encode_jmp(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("jmp requires 1 operand".to_string());
        }

        match &ops[0] {
            Operand::Label(label) => {
                // Near jump with 32-bit displacement (will be resolved by linker/relocator)
                self.bytes.push(0xE9);
                self.add_relocation(label, R_X86_64_PC32, -4);
                self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                Ok(())
            }
            Operand::Indirect(inner) => {
                match inner.as_ref() {
                    Operand::Register(reg) => {
                        let num = reg_num(&reg.name).ok_or("bad register")?;
                        if needs_rex_ext(&reg.name) {
                            self.bytes.push(self.rex(false, false, false, true));
                        }
                        self.bytes.push(0xFF);
                        self.bytes.push(self.modrm(3, 4, num));
                        Ok(())
                    }
                    Operand::Memory(mem) => {
                        self.emit_rex_rm(0, "", mem);
                        self.bytes.push(0xFF);
                        self.encode_modrm_mem(4, mem)
                    }
                    _ => Err("unsupported indirect jmp target".to_string()),
                }
            }
            _ => Err("unsupported jmp operand".to_string()),
        }
    }

    fn encode_jcc(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("jcc requires 1 operand".to_string());
        }

        let cc = cc_from_mnemonic(&mnemonic[1..])?;

        match &ops[0] {
            Operand::Label(label) => {
                // Near jcc with 32-bit displacement
                self.bytes.extend_from_slice(&[0x0F, 0x80 + cc]);
                self.add_relocation(label, R_X86_64_PC32, -4);
                self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                Ok(())
            }
            _ => Err("jcc requires label operand".to_string()),
        }
    }

    fn encode_call(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("call requires 1 operand".to_string());
        }

        match &ops[0] {
            Operand::Label(label) => {
                self.bytes.push(0xE8);
                // Use PLT32 for external function calls (linker will resolve)
                let reloc_type = if label.ends_with("@PLT") {
                    R_X86_64_PLT32
                } else {
                    R_X86_64_PLT32 // Default to PLT32 for calls
                };
                let sym = label.trim_end_matches("@PLT");
                self.add_relocation(sym, reloc_type, -4);
                self.bytes.extend_from_slice(&[0, 0, 0, 0]);
                Ok(())
            }
            Operand::Indirect(inner) => {
                match inner.as_ref() {
                    Operand::Register(reg) => {
                        let num = reg_num(&reg.name).ok_or("bad register")?;
                        if needs_rex_ext(&reg.name) {
                            self.bytes.push(self.rex(false, false, false, true));
                        }
                        self.bytes.push(0xFF);
                        self.bytes.push(self.modrm(3, 2, num));
                        Ok(())
                    }
                    Operand::Memory(mem) => {
                        self.emit_rex_rm(0, "", mem);
                        self.bytes.push(0xFF);
                        self.encode_modrm_mem(2, mem)
                    }
                    _ => Err("unsupported indirect call target".to_string()),
                }
            }
            _ => Err("unsupported call operand".to_string()),
        }
    }

    fn encode_xchg(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("xchg requires 2 operands".to_string());
        }
        let size = mnemonic_size_suffix(mnemonic).unwrap_or(8);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &src.name, mem);
                self.bytes.push(if size == 1 { 0x86 } else { 0x87 });
                self.encode_modrm_mem(src_num, mem)
            }
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rr(size, &src.name, &dst.name);
                self.bytes.push(if size == 1 { 0x86 } else { 0x87 });
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            _ => Err("unsupported xchg operands".to_string()),
        }
    }

    fn encode_cmpxchg(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("cmpxchg requires 2 operands".to_string());
        }
        let size = mnemonic_size_suffix(mnemonic).unwrap_or(8);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &src.name, mem);
                self.bytes.extend_from_slice(&[0x0F, if size == 1 { 0xB0 } else { 0xB1 }]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported cmpxchg operands".to_string()),
        }
    }

    fn encode_xadd(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("xadd requires 2 operands".to_string());
        }
        let size = mnemonic_size_suffix(mnemonic).unwrap_or(8);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &src.name, mem);
                self.bytes.extend_from_slice(&[0x0F, if size == 1 { 0xC0 } else { 0xC1 }]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported xadd operands".to_string()),
        }
    }

    fn encode_clflush(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("clflush requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Memory(mem) => {
                self.emit_rex_rm(0, "", mem);
                self.bytes.extend_from_slice(&[0x0F, 0xAE]);
                self.encode_modrm_mem(7, mem)
            }
            _ => Err("clflush requires memory operand".to_string()),
        }
    }

    // ---- SSE/SSE2 encoding helpers ----

    /// Encode SSE instruction: load (xmm<-xmm/mem) or store (xmm->mem).
    fn encode_sse_rr_rm(&mut self, ops: &[Operand], load_opcode: &[u8], store_opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("SSE mov requires 2 operands".to_string());
        }

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) && is_xmm(&dst.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                // Emit mandatory prefix bytes (e.g. 0x66, 0xF2, 0xF3) before REX.
                // The prefix ends where the 0x0F escape byte begins.
                let prefix_len = load_opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &load_opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rr(0, &dst.name, &src.name);
                // Emit remaining opcode bytes (0x0F + opcode byte)
                self.bytes.extend_from_slice(&load_opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) if is_xmm(&dst.name) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = load_opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &load_opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rm(0, &dst.name, mem);
                self.bytes.extend_from_slice(&load_opcode[prefix_len..]);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Memory(mem)) if is_xmm(&src.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let prefix_len = store_opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &store_opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rm(0, &src.name, mem);
                self.bytes.extend_from_slice(&store_opcode[prefix_len..]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported SSE mov operands".to_string()),
        }
    }

    /// Encode a standard SSE operation: op xmm/mem, xmm
    fn encode_sse_op(&mut self, ops: &[Operand], opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("SSE op requires 2 operands".to_string());
        }

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rr(0, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rm(0, &dst.name, mem);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported SSE op operands".to_string()),
        }
    }

    fn encode_sse_store(&mut self, ops: &[Operand], opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("SSE store requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Memory(mem)) if is_xmm(&src.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rm(0, &src.name, mem);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported SSE store operands".to_string()),
        }
    }

    fn encode_sse_op_imm8(&mut self, ops: &[Operand], opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("SSE op+imm8 requires 3 operands".to_string());
        }
        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rr(0, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported SSE op+imm8 operands".to_string()),
        }
    }

    fn encode_movd(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movd requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&dst.name) => {
                // GP -> XMM: 66 0F 6E /r
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                self.emit_rex_rr(0, &dst.name, &src.name);
                self.bytes.extend_from_slice(&[0x0F, 0x6E]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) => {
                // XMM -> GP: 66 0F 7E /r
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                self.emit_rex_rr(0, &src.name, &dst.name);
                self.bytes.extend_from_slice(&[0x0F, 0x7E]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) if is_xmm(&dst.name) => {
                // Mem -> XMM: 66 0F 6E /r
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                self.emit_rex_rm(0, &dst.name, mem);
                self.bytes.extend_from_slice(&[0x0F, 0x6E]);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Memory(mem)) if is_xmm(&src.name) => {
                // XMM -> Mem: 66 0F 7E /r
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                self.emit_rex_rm(0, &src.name, mem);
                self.bytes.extend_from_slice(&[0x0F, 0x7E]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported movd operands".to_string()),
        }
    }

    fn encode_movnti(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movnti requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let size = if is_reg64(&src.name) { 8 } else { 4 };
                self.emit_rex_rm(size, &src.name, mem);
                self.bytes.extend_from_slice(&[0x0F, 0xC3]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported movnti operands".to_string()),
        }
    }

    fn encode_sse_cvt_gp_to_xmm(&mut self, ops: &[Operand], opcode: &[u8], gp_size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("cvt requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&dst.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                // Need REX.W for 64-bit source
                self.emit_rex_rr(gp_size, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported cvt operands".to_string()),
        }
    }

    fn encode_sse_cvt_xmm_to_gp(&mut self, ops: &[Operand], opcode: &[u8], gp_size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("cvt requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rr(gp_size, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported cvt operands".to_string()),
        }
    }

    fn encode_sse_shift(&mut self, ops: &[Operand], reg_opcode: &[u8], imm_ext: u8, imm_opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("SSE shift requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = imm_opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &imm_opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                if needs_rex_ext(&dst.name) {
                    self.bytes.push(self.rex(false, false, false, true));
                }
                self.bytes.extend_from_slice(&imm_opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, imm_ext, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) => {
                self.encode_sse_op(&[ops[0].clone(), ops[1].clone()], reg_opcode)
            }
            _ => Err("unsupported SSE shift operands".to_string()),
        }
    }

    fn encode_sse_shift_imm_only(&mut self, ops: &[Operand], imm_ext: u8, opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("SSE shift requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                if needs_rex_ext(&dst.name) {
                    self.bytes.push(self.rex(false, false, false, true));
                }
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, imm_ext, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported SSE shift operands".to_string()),
        }
    }

    fn encode_sse_insert(&mut self, ops: &[Operand], opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("pinsrX requires 3 operands".to_string());
        }
        // pinsrX $imm, r/m, xmm
        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rr(0, &dst.name, &src.name);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported pinsrX operands".to_string()),
        }
    }

    fn encode_sse_extract(&mut self, ops: &[Operand], opcode: &[u8]) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("pextrX requires 3 operands".to_string());
        }
        // pextrX $imm, xmm, r/m
        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let prefix_len = opcode.iter().position(|&b| b == 0x0F).unwrap_or(0);
                for &b in &opcode[..prefix_len] {
                    self.bytes.push(b);
                }
                self.emit_rex_rr(0, &src.name, &dst.name);
                self.bytes.extend_from_slice(&opcode[prefix_len..]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported pextrX operands".to_string()),
        }
    }

    // ---- MOVQ with XMM registers ----

    /// Encode movq with XMM register operands.
    /// Forms:
    ///   movq %gp64, %xmm  -> 66 REX.W 0F 6E /r (GP -> XMM)
    ///   movq %xmm, %gp64  -> 66 REX.W 0F 7E /r (XMM -> GP)
    ///   movq mem, %xmm    -> F3 0F 7E /r (load 64-bit from memory)
    ///   movq %xmm, mem    -> 66 0F D6 /r (store 64-bit to memory)
    ///   movq %xmm, %xmm   -> F3 0F 7E /r (xmm-to-xmm)
    fn encode_movq_xmm(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movq requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            // movq %gp64, %xmm -> 66 REX.W 0F 6E /r
            (Operand::Register(src), Operand::Register(dst)) if !is_xmm(&src.name) && is_xmm(&dst.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                // REX.W with proper R (dst xmm ext) and B (src gp ext)
                let r = needs_rex_ext(&dst.name);
                let b = needs_rex_ext(&src.name);
                self.bytes.push(self.rex(true, r, false, b));
                self.bytes.extend_from_slice(&[0x0F, 0x6E]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            // movq %xmm, %gp64 -> 66 REX.W 0F 7E /r
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) && !is_xmm(&dst.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                // REX.W with proper R (src xmm ext) and B (dst gp ext)
                let r = needs_rex_ext(&src.name);
                let b = needs_rex_ext(&dst.name);
                self.bytes.push(self.rex(true, r, false, b));
                self.bytes.extend_from_slice(&[0x0F, 0x7E]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            // movq %xmm, %xmm -> F3 0F 7E /r
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) && is_xmm(&dst.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0xF3);
                self.emit_rex_rr(0, &dst.name, &src.name);
                self.bytes.extend_from_slice(&[0x0F, 0x7E]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            // movq mem, %xmm -> F3 0F 7E /r (load)
            (Operand::Memory(mem), Operand::Register(dst)) if is_xmm(&dst.name) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                self.bytes.push(0xF3);
                self.emit_rex_rm(0, &dst.name, mem);
                self.bytes.extend_from_slice(&[0x0F, 0x7E]);
                self.encode_modrm_mem(dst_num, mem)
            }
            // movq %xmm, mem -> 66 0F D6 /r (store)
            (Operand::Register(src), Operand::Memory(mem)) if is_xmm(&src.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                self.emit_rex_rm(0, &src.name, mem);
                self.bytes.extend_from_slice(&[0x0F, 0xD6]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported movq operand combination".to_string()),
        }
    }

    // ---- CRC32 encoding ----

    /// Encode CRC32 instruction.
    /// crc32b: F2 0F 38 F0 /r (8-bit source)
    /// crc32w: 66 F2 0F 38 F1 /r (16-bit source)
    /// crc32l: F2 0F 38 F1 /r (32-bit source)
    /// crc32q: F2 REX.W 0F 38 F1 /r (64-bit source)
    fn encode_crc32(&mut self, ops: &[Operand], src_size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("crc32 requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;

                if src_size == 2 {
                    self.bytes.push(0x66); // operand size prefix for 16-bit
                }
                self.bytes.push(0xF2); // mandatory prefix

                // REX prefix
                let w = src_size == 8;
                let r = needs_rex_ext(&dst.name);
                let b = needs_rex_ext(&src.name);
                let need_rex = w || r || b || (src_size == 1 && is_rex_required_8bit(&src.name));
                if need_rex {
                    self.bytes.push(self.rex(w, r, false, b));
                }

                self.bytes.extend_from_slice(&[0x0F, 0x38]);
                if src_size == 1 {
                    self.bytes.push(0xF0);
                } else {
                    self.bytes.push(0xF1);
                }
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;

                if src_size == 2 {
                    self.bytes.push(0x66);
                }
                self.bytes.push(0xF2);

                let w = src_size == 8;
                let r = needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name));
                let need_rex = w || r || b_ext || x;
                if need_rex {
                    self.bytes.push(self.rex(w, r, x, b_ext));
                }

                self.bytes.extend_from_slice(&[0x0F, 0x38]);
                if src_size == 1 {
                    self.bytes.push(0xF0);
                } else {
                    self.bytes.push(0xF1);
                }
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported crc32 operands".to_string()),
        }
    }

    // ---- x87 FPU encoding ----

    fn encode_x87_mem(&mut self, ops: &[Operand], opcode: &[u8], ext: u8) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("x87 mem op requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Memory(mem) => {
                // x87 instructions don't use REX.W
                self.emit_rex_rm(0, "", mem);
                self.bytes.extend_from_slice(opcode);
                self.encode_modrm_mem(ext, mem)
            }
            _ => Err("x87 mem op requires memory operand".to_string()),
        }
    }

    fn encode_fcomip(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() == 2 {
            // fcomip %st(N), %st
            match &ops[0] {
                Operand::Register(reg) => {
                    let n = parse_st_num(&reg.name)?;
                    self.bytes.extend_from_slice(&[0xDF, 0xF0 + n]);
                    Ok(())
                }
                _ => Err("fcomip requires st register".to_string()),
            }
        } else if ops.is_empty() {
            self.bytes.extend_from_slice(&[0xDF, 0xF1]);
            Ok(())
        } else {
            Err("fcomip requires 0 or 2 operands".to_string())
        }
    }

    fn encode_fucomip(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() == 2 {
            match &ops[0] {
                Operand::Register(reg) => {
                    let n = parse_st_num(&reg.name)?;
                    self.bytes.extend_from_slice(&[0xDF, 0xE8 + n]);
                    Ok(())
                }
                _ => Err("fucomip requires st register".to_string()),
            }
        } else if ops.is_empty() {
            self.bytes.extend_from_slice(&[0xDF, 0xE9]);
            Ok(())
        } else {
            Err("fucomip requires 0 or 2 operands".to_string())
        }
    }

    fn encode_fld_st(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("fld requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let n = parse_st_num(&reg.name)?;
                self.bytes.extend_from_slice(&[0xD9, 0xC0 + n]);
                Ok(())
            }
            _ => Err("fld requires st register".to_string()),
        }
    }

    fn encode_fstp_st(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("fstp requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let n = parse_st_num(&reg.name)?;
                self.bytes.extend_from_slice(&[0xDD, 0xD8 + n]);
                Ok(())
            }
            _ => Err("fstp requires st register".to_string()),
        }
    }

    fn encode_fxch(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.is_empty() {
            self.bytes.extend_from_slice(&[0xD9, 0xC9]); // fxch %st(1)
            return Ok(());
        }
        if ops.len() != 1 {
            return Err("fxch requires 0 or 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let n = parse_st_num(&reg.name)?;
                self.bytes.extend_from_slice(&[0xD9, 0xC8 + n]);
                Ok(())
            }
            _ => Err("fxch requires st register".to_string()),
        }
    }

    // ---- Bit scan instructions ----

    fn encode_bitscan(&mut self, ops: &[Operand], opcode: &[u8], size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("bsf/bsr requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or_else(|| format!("bad register: {}", src.name))?;
                let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rr(size, &dst.name, &src.name);
                self.bytes.extend_from_slice(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &dst.name, mem);
                self.bytes.extend_from_slice(opcode);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported bsf/bsr operands".to_string()),
        }
    }

    // ---- Bit test instructions ----

    fn encode_bt(&mut self, ops: &[Operand], op_ext: u8, size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("bt/bts/btr/btc requires 2 operands".to_string());
        }

        // op_ext: 0=bt, 5=bts, 6=btr, 7=btc
        // Two forms:
        //   bt $imm8, r/m  -> 0F BA /ext ib
        //   bt %reg, r/m   -> 0F A3/AB/B3/BB /r (opcode depends on ext)
        let reg_opcode = match op_ext {
            0 => 0xA3u8, // bt
            5 => 0xAB,   // bts
            6 => 0xB3,   // btr
            7 => 0xBB,   // btc
            _ => unreachable!(),
        };

        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);
                self.bytes.extend_from_slice(&[0x0F, 0xBA]);
                self.bytes.push(self.modrm(3, op_ext, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Memory(mem)) => {
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, "", mem);
                self.bytes.extend_from_slice(&[0x0F, 0xBA]);
                self.encode_modrm_mem(op_ext, mem)?;
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or_else(|| format!("bad register: {}", src.name))?;
                let dst_num = reg_num(&dst.name).ok_or_else(|| format!("bad register: {}", dst.name))?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rr(size, &src.name, &dst.name);
                self.bytes.extend_from_slice(&[0x0F, reg_opcode]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            (Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or_else(|| format!("bad register: {}", src.name))?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_rm(size, &src.name, mem);
                self.bytes.extend_from_slice(&[0x0F, reg_opcode]);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported bt operands".to_string()),
        }
    }

    // ---- INT instruction ----

    fn encode_int(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("int requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Immediate(ImmediateValue::Integer(n)) => {
                if *n == 3 {
                    self.bytes.push(0xCC);
                } else {
                    self.bytes.push(0xCD);
                    self.bytes.push(*n as u8);
                }
                Ok(())
            }
            _ => Err("int requires immediate operand".to_string()),
        }
    }

    // ---- Bswap without suffix ----

    fn encode_bswap_infer(&mut self, ops: &[Operand]) -> Result<(), String> {
        let size = infer_size_from_ops(ops).unwrap_or(8);
        self.encode_bswap(ops, size)
    }

    // ---- Shift without suffix ----

    fn encode_shift_infer(&mut self, ops: &[Operand], shift_op: u8, size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("shift requires 2 operands".to_string());
        }

        match (&ops[0], &ops[1]) {
            (Operand::Immediate(ImmediateValue::Integer(count)), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let count = *count as u8;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);
                if count == 1 {
                    self.bytes.push(if size == 1 { 0xD0 } else { 0xD1 });
                    self.bytes.push(self.modrm(3, shift_op, dst_num));
                } else {
                    self.bytes.push(if size == 1 { 0xC0 } else { 0xC1 });
                    self.bytes.push(self.modrm(3, shift_op, dst_num));
                    self.bytes.push(count);
                }
                Ok(())
            }
            (Operand::Register(cl), Operand::Register(dst)) if cl.name == "cl" => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.emit_rex_unary(size, &dst.name);
                self.bytes.push(if size == 1 { 0xD2 } else { 0xD3 });
                self.bytes.push(self.modrm(3, shift_op, dst_num));
                Ok(())
            }
            _ => Err("unsupported shift operands".to_string()),
        }
    }

    // ---- MMX movq ----

    fn encode_mmx_movq(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("movq requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            // movq %mm, %mm
            (Operand::Register(src), Operand::Register(dst)) if is_mmx_reg(&src.name) && is_mmx_reg(&dst.name) => {
                let src_num = mmx_num(&src.name).ok_or("bad mmx register")?;
                let dst_num = mmx_num(&dst.name).ok_or("bad mmx register")?;
                self.bytes.extend_from_slice(&[0x0F, 0x6F]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            // movq mem, %mm
            (Operand::Memory(mem), Operand::Register(dst)) if is_mmx_reg(&dst.name) => {
                let dst_num = mmx_num(&dst.name).ok_or("bad mmx register")?;
                self.emit_rex_rm(0, "", mem);
                self.bytes.extend_from_slice(&[0x0F, 0x6F]);
                self.encode_modrm_mem(dst_num, mem)
            }
            // movq %mm, mem
            (Operand::Register(src), Operand::Memory(mem)) if is_mmx_reg(&src.name) => {
                let src_num = mmx_num(&src.name).ok_or("bad mmx register")?;
                self.emit_rex_rm(0, "", mem);
                self.bytes.extend_from_slice(&[0x0F, 0x7F]);
                self.encode_modrm_mem(src_num, mem)
            }
            // movq %gp64, %mm or movq %mm, %gp64
            (Operand::Register(src), Operand::Register(dst)) if is_mmx_reg(&dst.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = mmx_num(&dst.name).ok_or("bad mmx register")?;
                // movq %gp64, %mm -> REX.W 0F 6E /r
                let b = needs_rex_ext(&src.name);
                self.bytes.push(self.rex(true, false, false, b));
                self.bytes.extend_from_slice(&[0x0F, 0x6E]);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) if is_mmx_reg(&src.name) => {
                let src_num = mmx_num(&src.name).ok_or("bad mmx register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                // movq %mm, %gp64 -> REX.W 0F 7E /r
                let b = needs_rex_ext(&dst.name);
                self.bytes.push(self.rex(true, false, false, b));
                self.bytes.extend_from_slice(&[0x0F, 0x7E]);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            _ => Err("unsupported mmx movq operands".to_string()),
        }
    }

    // ---- IN/OUT instructions ----

    fn encode_in_out(&mut self, ops: &[Operand], dx_opcode: u8, imm_opcode: u8, size: u8) -> Result<(), String> {
        if size == 2 { self.bytes.push(0x66); }
        if ops.is_empty() {
            // inb (%dx), %al form without explicit operands
            self.bytes.push(dx_opcode);
            return Ok(());
        }
        match ops.len() {
            1 => {
                // in $imm, %al/%ax/%eax
                match &ops[0] {
                    Operand::Immediate(ImmediateValue::Integer(port)) => {
                        self.bytes.push(imm_opcode);
                        self.bytes.push(*port as u8);
                    }
                    _ => {
                        self.bytes.push(dx_opcode);
                    }
                }
                Ok(())
            }
            2 => {
                match &ops[0] {
                    Operand::Immediate(ImmediateValue::Integer(port)) => {
                        self.bytes.push(imm_opcode);
                        self.bytes.push(*port as u8);
                    }
                    _ => {
                        self.bytes.push(dx_opcode);
                    }
                }
                Ok(())
            }
            _ => Err("in requires 0-2 operands".to_string()),
        }
    }

    fn encode_out(&mut self, ops: &[Operand], dx_opcode: u8, imm_opcode: u8, size: u8) -> Result<(), String> {
        if size == 2 { self.bytes.push(0x66); }
        if ops.is_empty() {
            self.bytes.push(dx_opcode);
            return Ok(());
        }
        match ops.len() {
            1 | 2 => {
                // Check if we have an immediate port
                for op in ops {
                    if let Operand::Immediate(ImmediateValue::Integer(port)) = op {
                        self.bytes.push(imm_opcode);
                        self.bytes.push(*port as u8);
                        return Ok(());
                    }
                }
                self.bytes.push(dx_opcode);
                Ok(())
            }
            _ => Err("out requires 0-2 operands".to_string()),
        }
    }

    // ---- VEX-encoded AVX instructions ----

    /// Emit a 2-byte VEX prefix.
    /// pp: 0=none, 1=0x66, 2=0xF3, 3=0xF2
    /// r,b: inverted REX.R, REX.B
    /// vvvv: inverted register (for src2/NDS), 0xF = unused
    /// l: 0=128-bit, 1=256-bit
    fn emit_vex2(&mut self, r: bool, vvvv: u8, l: bool, pp: u8) {
        // 2-byte VEX: C5 [R vvvv L pp]
        let mut byte1 = 0u8;
        if !r { byte1 |= 0x80; } // R is inverted
        byte1 |= ((!vvvv) & 0xF) << 3;
        if l { byte1 |= 0x04; }
        byte1 |= pp & 3;
        self.bytes.push(0xC5);
        self.bytes.push(byte1);
    }

    /// Emit a 3-byte VEX prefix.
    fn emit_vex3(&mut self, r: bool, x: bool, b: bool, mmmmm: u8, w: bool, vvvv: u8, l: bool, pp: u8) {
        // 3-byte VEX: C4 [R X B mmmmm] [W vvvv L pp]
        let mut byte1 = 0u8;
        if !r { byte1 |= 0x80; }
        if !x { byte1 |= 0x40; }
        if !b { byte1 |= 0x20; }
        byte1 |= mmmmm & 0x1F;
        let mut byte2 = 0u8;
        if w { byte2 |= 0x80; }
        byte2 |= ((!vvvv) & 0xF) << 3;
        if l { byte2 |= 0x04; }
        byte2 |= pp & 3;
        self.bytes.push(0xC4);
        self.bytes.push(byte1);
        self.bytes.push(byte2);
    }

    /// Get VEX pp encoding from prefix byte.
    fn vex_pp(prefix: u8) -> u8 {
        match prefix {
            0x66 => 1,
            0xF3 => 2,
            0xF2 => 3,
            _ => 0,
        }
    }

    /// Determine if operands use 256-bit (YMM) registers.
    fn is_256bit(ops: &[Operand]) -> bool {
        ops.iter().any(|op| matches!(op, Operand::Register(r) if is_ymm(&r.name)))
    }

    /// Encode VEX MOV (vmovdqa/vmovdqu): load or store.
    fn encode_vex_mov(&mut self, ops: &[Operand], load_op: u8, store_op: u8, is_f3: bool) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("vmov requires 2 operands".to_string());
        }
        let l = Self::is_256bit(ops);
        let pp = if is_f3 { 2u8 } else { 1u8 }; // F3 or 66

        match (&ops[0], &ops[1]) {
            // xmm/ymm -> xmm/ymm (use load form)
            (Operand::Register(src), Operand::Register(dst))
                if (is_xmm(&src.name) || is_ymm(&src.name)) && (is_xmm(&dst.name) || is_ymm(&dst.name)) =>
            {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, l, pp);
                }
                self.bytes.push(load_op);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            // mem -> xmm/ymm (load)
            (Operand::Memory(mem), Operand::Register(dst))
                if is_xmm(&dst.name) || is_ymm(&dst.name) =>
            {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, l, pp);
                }
                self.bytes.push(load_op);
                self.encode_modrm_mem(dst_num, mem)
            }
            // xmm/ymm -> mem (store)
            (Operand::Register(src), Operand::Memory(mem))
                if is_xmm(&src.name) || is_ymm(&src.name) =>
            {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, l, pp);
                }
                self.bytes.push(store_op);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported vmov operands".to_string()),
        }
    }

    /// Encode VEX MOV with no mandatory prefix (vmovaps/vmovups).
    fn encode_vex_mov_noprefix(&mut self, ops: &[Operand], load_op: u8, store_op: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("vmov requires 2 operands".to_string());
        }
        let l = Self::is_256bit(ops);
        let pp = 0u8; // no prefix

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst))
                if (is_xmm(&src.name) || is_ymm(&src.name)) && (is_xmm(&dst.name) || is_ymm(&dst.name)) =>
            {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, l, pp);
                }
                self.bytes.push(load_op);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst))
                if is_xmm(&dst.name) || is_ymm(&dst.name) =>
            {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, l, pp);
                }
                self.bytes.push(load_op);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Memory(mem))
                if is_xmm(&src.name) || is_ymm(&src.name) =>
            {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, l, pp);
                }
                self.bytes.push(store_op);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported vmov operands".to_string()),
        }
    }

    /// Encode VEX MOV with 0x66 prefix (vmovapd/vmovupd).
    fn encode_vex_mov_66(&mut self, ops: &[Operand], load_op: u8, store_op: u8) -> Result<(), String> {
        self.encode_vex_mov(ops, load_op, store_op, false) // pp=1 is 0x66
    }

    /// Encode VEX binary operation: op src, vvvv, dst (3 operands in src, src2, dst form).
    /// For 2-operand forms, vvvv = dst (NDS encoding).
    fn encode_vex_binop(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX binop requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                if b {
                    self.emit_vex2(r, src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 }, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 }, l, pp);
                }
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(src2), Operand::Register(dst)) => {
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                if b_ext && x {
                    self.emit_vex2(r, vvvv, l, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, vvvv, l, pp);
                }
                self.bytes.push(opcode);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported VEX binop operands".to_string()),
        }
    }

    /// VEX binary op with no mandatory prefix.
    fn encode_vex_binop_np(&mut self, ops: &[Operand], opcode: u8) -> Result<(), String> {
        self.encode_vex_binop(ops, 0, opcode)
    }

    /// VEX binary op with 0F 38 escape (3-byte VEX required).
    fn encode_vex_binop_38(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX binop requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 2, false, vvvv, l, pp); // mmmmm=2 for 0F 38
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(src2), Operand::Register(dst)) => {
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, x, b_ext, 2, false, vvvv, l, pp);
                self.bytes.push(opcode);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported VEX binop operands".to_string()),
        }
    }

    /// VEX unary with imm8: e.g. vpshufd $imm, src, dst.
    fn encode_vex_unary_imm8(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX unary+imm8 requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, l, pp);
                }
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, l, pp);
                }
                self.bytes.push(opcode);
                self.encode_modrm_mem(dst_num, mem)?;
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX unary+imm8 operands".to_string()),
        }
    }

    /// VEX unary with imm8 and W=1 (e.g. vpermq).
    fn encode_vex_unary_imm8_w1(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX unary+imm8+W1 requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                self.emit_vex3(r, true, b, 3, true, 0xF, l, pp); // mmmmm=3 for 0F 3A, W=1
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX unary+imm8+W1 operands".to_string()),
        }
    }

    /// VEX shift: $imm, xmm/ymm, xmm/ymm (immediate form with NDS).
    fn encode_vex_shift(&mut self, ops: &[Operand], prefix: u8, _reg_op: u8, imm_ext: u8, imm_op: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX shift requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                // VEX shift imm: vpsllX $imm, %src, %dst
                // Encoding: VEX.NDS.128/256.66.0F imm_op /imm_ext ib
                // vvvv = dst (output register)
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = true; // no REX.R for /imm_ext form
                let b = !needs_rex_ext(&src.name);
                let vvvv = dst_num | if needs_rex_ext(&dst.name) { 8 } else { 0 };
                if b {
                    self.emit_vex2(r, vvvv, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, vvvv, l, pp);
                }
                self.bytes.push(imm_op);
                self.bytes.push(self.modrm(3, imm_ext, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Register(count), Operand::Register(src2), Operand::Register(dst)) => {
                // VEX shift reg: vpsllX %xmm_count, %src, %dst
                let count_num = reg_num(&count.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&count.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                if b {
                    self.emit_vex2(r, vvvv, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, vvvv, l, pp);
                }
                self.bytes.push(_reg_op);
                self.bytes.push(self.modrm(3, dst_num, count_num));
                Ok(())
            }
            _ => Err("unsupported VEX shift operands".to_string()),
        }
    }

    /// VEX shift imm-only (vpslldq/vpsrldq).
    fn encode_vex_shift_imm_only(&mut self, ops: &[Operand], prefix: u8, imm_ext: u8, imm_op: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX shift imm requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let b = !needs_rex_ext(&src.name);
                let vvvv = dst_num | if needs_rex_ext(&dst.name) { 8 } else { 0 };
                if b {
                    self.emit_vex2(true, vvvv, l, pp);
                } else {
                    self.emit_vex3(true, true, b, 1, false, vvvv, l, pp);
                }
                self.bytes.push(imm_op);
                self.bytes.push(self.modrm(3, imm_ext, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX shift imm operands".to_string()),
        }
    }

    /// VEX unary from 0F 38 map (e.g., vbroadcastss).
    fn encode_vex_unary_38(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("VEX unary requires 2 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1]) {
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                self.emit_vex3(r, x, b_ext, 2, false, 0xF, l, pp); // mmmmm=2 for 0F 38
                self.bytes.push(opcode);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                self.emit_vex3(r, true, b, 2, false, 0xF, l, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported VEX unary operands".to_string()),
        }
    }

    /// VEX unary op from 0F 38 (e.g., vpmovsxbw) - like encode_vex_unary_38 but for SSE4.
    fn encode_vex_unary_38_op(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        self.encode_vex_unary_38(ops, prefix, opcode)
    }

    /// VEX unary op from 0F map (e.g., vucomisd).
    fn encode_vex_unary_op(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("VEX unary op requires 2 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, false, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, false, pp);
                }
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, false, pp);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, pp);
                }
                self.bytes.push(opcode);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported VEX unary op operands".to_string()),
        }
    }

    /// VEX insert128 (vinserti128/vinsertf128).
    fn encode_vex_insert128(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("vinsert128 requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 3, false, vvvv, true, pp); // L=1 for 256-bit, mmmmm=3 for 0F 3A
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Memory(mem), Operand::Register(src2), Operand::Register(dst)) => {
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, x, b_ext, 3, false, vvvv, true, pp);
                self.bytes.push(opcode);
                self.encode_modrm_mem(dst_num, mem)?;
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported vinsert128 operands".to_string()),
        }
    }

    /// VEX extract128 (vextracti128/vextractf128).
    fn encode_vex_extract128(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("vextract128 requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b = !needs_rex_ext(&dst.name);
                self.emit_vex3(r, true, b, 3, false, 0xF, true, pp); // L=1 for 256-bit
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Memory(mem)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                self.emit_vex3(r, x, b_ext, 3, false, 0xF, true, pp);
                self.bytes.push(opcode);
                self.encode_modrm_mem(src_num, mem)?;
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported vextract128 operands".to_string()),
        }
    }

    /// VEX perm2 (vperm2i128/vperm2f128).
    fn encode_vex_perm2(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("vperm2 requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 3, false, vvvv, true, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported vperm2 operands".to_string()),
        }
    }

    /// VEX extract to GP register.
    fn encode_vex_extract_gp(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("VEX extract to GP requires 2 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, l, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, l, pp);
                }
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported VEX extract operands".to_string()),
        }
    }

    /// VEX ternary with imm8: e.g. vpalignr $imm, src, src2, dst.
    fn encode_vex_ternary_imm8(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("VEX ternary+imm8 requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                // Use 0F 3A map (mmmmm=3)
                self.emit_vex3(r, true, b, 3, false, vvvv, l, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX ternary+imm8 operands".to_string()),
        }
    }

    /// VEX ternary with imm8, no prefix.
    fn encode_vex_ternary_imm8_np(&mut self, ops: &[Operand], opcode: u8) -> Result<(), String> {
        self.encode_vex_ternary_imm8(ops, 0, opcode)
    }

    /// VEX blend4 (4-operand form with is4/imm register).
    fn encode_vex_blend4(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("VEX blend4 requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);
        let l = Self::is_256bit(ops);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Register(mask), Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let mask_num = reg_num(&mask.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 3, false, vvvv, l, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                // IS4 byte: mask register << 4
                let is4 = (mask_num | if needs_rex_ext(&mask.name) { 8 } else { 0 }) << 4;
                self.bytes.push(is4);
                Ok(())
            }
            _ => Err("unsupported VEX blend4 operands".to_string()),
        }
    }

    /// VEX movd (GP <-> XMM).
    fn encode_vex_movd(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("vmovd requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) if !is_xmm(&src.name) && is_xmm(&dst.name) => {
                // GP -> XMM
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, false, 1); // pp=1 for 66
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, false, 1);
                }
                self.bytes.push(0x6E);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) && !is_xmm(&dst.name) => {
                // XMM -> GP
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b = !needs_rex_ext(&dst.name);
                if b {
                    self.emit_vex2(r, 0xF, false, 1);
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, false, 1);
                }
                self.bytes.push(0x7E);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) if is_xmm(&dst.name) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, false, 1);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, 1);
                }
                self.bytes.push(0x6E);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Memory(mem)) if is_xmm(&src.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, false, 1);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, 1);
                }
                self.bytes.push(0x7E);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported vmovd operands".to_string()),
        }
    }

    /// VEX movq (64-bit GP <-> XMM or XMM <-> XMM).
    fn encode_vex_movq(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("vmovq requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) if !is_xmm(&src.name) && is_xmm(&dst.name) => {
                // GP64 -> XMM: VEX.128.66.0F.W1 6E /r
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                self.emit_vex3(r, true, b, 1, true, 0xF, false, 1); // W=1
                self.bytes.push(0x6E);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) && !is_xmm(&dst.name) => {
                // XMM -> GP64: VEX.128.66.0F.W1 7E /r
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b = !needs_rex_ext(&dst.name);
                self.emit_vex3(r, true, b, 1, true, 0xF, false, 1);
                self.bytes.push(0x7E);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                Ok(())
            }
            (Operand::Register(src), Operand::Register(dst)) if is_xmm(&src.name) && is_xmm(&dst.name) => {
                // XMM -> XMM: VEX.128.F3.0F 7E /r
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                if b {
                    self.emit_vex2(r, 0xF, false, 2); // pp=2 for F3
                } else {
                    self.emit_vex3(r, true, b, 1, false, 0xF, false, 2);
                }
                self.bytes.push(0x7E);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) if is_xmm(&dst.name) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, false, 2);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, 2);
                }
                self.bytes.push(0x7E);
                self.encode_modrm_mem(dst_num, mem)
            }
            (Operand::Register(src), Operand::Memory(mem)) if is_xmm(&src.name) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                if b_ext && x {
                    self.emit_vex2(r, 0xF, false, 1);
                } else {
                    self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, 1);
                }
                self.bytes.push(0xD6);
                self.encode_modrm_mem(src_num, mem)
            }
            _ => Err("unsupported vmovq operands".to_string()),
        }
    }

    /// VEX cvt GP -> XMM (e.g., vcvtsi2sd).
    fn encode_vex_cvt_gp(&mut self, ops: &[Operand], prefix: u8, opcode: u8, is_64: bool) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX cvt requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Register(src), Operand::Register(src2), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 1, is_64, vvvv, false, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported VEX cvt operands".to_string()),
        }
    }

    /// VEX cvt XMM -> GP (e.g., vcvttsd2si).
    fn encode_vex_cvt_to_gp(&mut self, ops: &[Operand], prefix: u8, opcode: u8, is_64: bool) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("VEX cvt to GP requires 2 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                self.emit_vex3(r, true, b, 1, is_64, 0xF, false, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported VEX cvt to GP operands".to_string()),
        }
    }

    /// VEX movss/movsd.
    fn encode_vex_mov_ss_sd(&mut self, ops: &[Operand], prefix: u8) -> Result<(), String> {
        if ops.len() != 2 && ops.len() != 3 {
            return Err("vmovss/sd requires 2-3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        if ops.len() == 2 {
            match (&ops[0], &ops[1]) {
                (Operand::Memory(mem), Operand::Register(dst)) => {
                    let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                    let r = !needs_rex_ext(&dst.name);
                    let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                    let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                    if b_ext && x {
                        self.emit_vex2(r, 0xF, false, pp);
                    } else {
                        self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, pp);
                    }
                    self.bytes.push(0x10);
                    self.encode_modrm_mem(dst_num, mem)
                }
                (Operand::Register(src), Operand::Memory(mem)) => {
                    let src_num = reg_num(&src.name).ok_or("bad register")?;
                    let r = !needs_rex_ext(&src.name);
                    let b_ext = mem.base.as_ref().map_or(true, |b| !needs_rex_ext(&b.name));
                    let x = mem.index.as_ref().map_or(true, |i| !needs_rex_ext(&i.name));
                    if b_ext && x {
                        self.emit_vex2(r, 0xF, false, pp);
                    } else {
                        self.emit_vex3(r, x, b_ext, 1, false, 0xF, false, pp);
                    }
                    self.bytes.push(0x11);
                    self.encode_modrm_mem(src_num, mem)
                }
                (Operand::Register(src), Operand::Register(dst)) => {
                    // 2-op reg-reg: treated like 3-op with dst=src2
                    let src_num = reg_num(&src.name).ok_or("bad register")?;
                    let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                    let r = !needs_rex_ext(&dst.name);
                    let b = !needs_rex_ext(&src.name);
                    let vvvv = dst_num | if needs_rex_ext(&dst.name) { 8 } else { 0 };
                    if b {
                        self.emit_vex2(r, vvvv, false, pp);
                    } else {
                        self.emit_vex3(r, true, b, 1, false, vvvv, false, pp);
                    }
                    self.bytes.push(0x10);
                    self.bytes.push(self.modrm(3, dst_num, src_num));
                    Ok(())
                }
                _ => Err("unsupported vmovss/sd operands".to_string()),
            }
        } else {
            // 3-operand form
            match (&ops[0], &ops[1], &ops[2]) {
                (Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                    let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                    let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                    let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                    let r = !needs_rex_ext(&dst.name);
                    let b = !needs_rex_ext(&src1.name);
                    let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                    if b {
                        self.emit_vex2(r, vvvv, false, pp);
                    } else {
                        self.emit_vex3(r, true, b, 1, false, vvvv, false, pp);
                    }
                    self.bytes.push(0x10);
                    self.bytes.push(self.modrm(3, dst_num, src1_num));
                    Ok(())
                }
                _ => Err("unsupported vmovss/sd 3-op operands".to_string()),
            }
        }
    }

    /// VEX insert from GP register (e.g., vpinsrw).
    fn encode_vex_insert_gp(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("VEX insert requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(src2), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                if b {
                    self.emit_vex2(r, vvvv, false, pp);
                } else {
                    self.emit_vex3(r, true, b, 1, false, vvvv, false, pp);
                }
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX insert operands".to_string()),
        }
    }

    /// VEX insert from 0F 3A map.
    fn encode_vex_insert_gp_3a(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("VEX insert requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(src2), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 3, false, vvvv, false, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX insert 3A operands".to_string()),
        }
    }

    /// VEX extract from 0F 3A map.
    fn encode_vex_extract_gp_3a(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX extract requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b = !needs_rex_ext(&dst.name);
                self.emit_vex3(r, true, b, 3, false, 0xF, false, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX extract 3A operands".to_string()),
        }
    }

    /// VEX insert from 0F 3A map with VEX.W=1 (for 64-bit operands like vpinsrq).
    fn encode_vex_insert_gp_3a_w1(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 4 {
            return Err("VEX insert requires 4 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2], &ops[3]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(src2), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                self.emit_vex3(r, true, b, 3, true, vvvv, false, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX insert 3A W1 operands".to_string()),
        }
    }

    /// VEX extract from 0F 3A map with VEX.W=1 (for 64-bit operands like vpextrq).
    fn encode_vex_extract_gp_3a_w1(&mut self, ops: &[Operand], prefix: u8, opcode: u8) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("VEX extract requires 3 operands".to_string());
        }
        let pp = Self::vex_pp(prefix);

        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let r = !needs_rex_ext(&src.name);
                let b = !needs_rex_ext(&dst.name);
                self.emit_vex3(r, true, b, 3, true, 0xF, false, pp);
                self.bytes.push(opcode);
                self.bytes.push(self.modrm(3, src_num, dst_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported VEX extract 3A W1 operands".to_string()),
        }
    }

    /// CMPXCHG8B m64 - 0F C7 /1
    fn encode_cmpxchg8b(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("cmpxchg8b requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Memory(mem) => {
                let rex_needed = mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name))
                    || mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name));
                if rex_needed {
                    let b_ext = mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name));
                    let x_ext = mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name));
                    let rex = 0x40 | if b_ext { 1 } else { 0 } | if x_ext { 2 } else { 0 };
                    self.bytes.push(rex);
                }
                self.bytes.push(0x0F);
                self.bytes.push(0xC7);
                self.encode_modrm_mem(1, mem)
            }
            _ => Err("cmpxchg8b requires memory operand".to_string()),
        }
    }

    /// CMPXCHG16B m128 - REX.W + 0F C7 /1
    fn encode_cmpxchg16b(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("cmpxchg16b requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Memory(mem) => {
                let b_ext = mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name));
                let x_ext = mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name));
                let rex = 0x48 | if b_ext { 1 } else { 0 } | if x_ext { 2 } else { 0 };
                self.bytes.push(rex);
                self.bytes.push(0x0F);
                self.bytes.push(0xC7);
                self.encode_modrm_mem(1, mem)
            }
            _ => Err("cmpxchg16b requires memory operand".to_string()),
        }
    }

    /// FSUBP - DE E9 (no operands = fsubp %st, %st(1))
    /// With operands: fsubp %st, %st(i) = DE E8+i
    fn encode_fsubp(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.is_empty() {
            self.bytes.extend_from_slice(&[0xDE, 0xE9]);
            return Ok(());
        }
        if ops.len() == 2 {
            // fsubp %st, %st(i)
            match &ops[1] {
                Operand::Register(reg) => {
                    let n = parse_st_num(&reg.name)?;
                    self.bytes.extend_from_slice(&[0xDE, 0xE8 + n]);
                    Ok(())
                }
                _ => Err("fsubp requires st register".to_string()),
            }
        } else if ops.len() == 1 {
            match &ops[0] {
                Operand::Register(reg) => {
                    let n = parse_st_num(&reg.name)?;
                    self.bytes.extend_from_slice(&[0xDE, 0xE8 + n]);
                    Ok(())
                }
                _ => Err("fsubp requires st register".to_string()),
            }
        } else {
            Err("fsubp requires 0-2 operands".to_string())
        }
    }

    /// FDIVP - DE F9 (no operands = fdivp %st, %st(1))
    /// With operands: fdivp %st, %st(i) = DE F8+i
    fn encode_fdivp(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.is_empty() {
            self.bytes.extend_from_slice(&[0xDE, 0xF9]);
            return Ok(());
        }
        if ops.len() == 2 {
            match &ops[1] {
                Operand::Register(reg) => {
                    let n = parse_st_num(&reg.name)?;
                    self.bytes.extend_from_slice(&[0xDE, 0xF8 + n]);
                    Ok(())
                }
                _ => Err("fdivp requires st register".to_string()),
            }
        } else if ops.len() == 1 {
            match &ops[0] {
                Operand::Register(reg) => {
                    let n = parse_st_num(&reg.name)?;
                    self.bytes.extend_from_slice(&[0xDE, 0xF8 + n]);
                    Ok(())
                }
                _ => Err("fdivp requires st register".to_string()),
            }
        } else {
            Err("fdivp requires 0-2 operands".to_string())
        }
    }

    /// SLDT - store local descriptor table register (0F 00 /0)
    fn encode_sldt(&mut self, ops: &[Operand]) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("sldt requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let reg_n = reg_num(&reg.name).ok_or("bad register")?;
                if is_reg16(&reg.name) {
                    self.bytes.push(0x66);
                }
                if needs_rex_ext(&reg.name) {
                    self.bytes.push(self.rex(false, false, false, true));
                }
                self.bytes.push(0x0F);
                self.bytes.push(0x00);
                self.bytes.push(self.modrm(3, 0, reg_n));
                Ok(())
            }
            Operand::Memory(mem) => {
                self.bytes.push(0x0F);
                self.bytes.push(0x00);
                self.encode_modrm_mem(0, mem)
            }
            _ => Err("unsupported sldt operand".to_string()),
        }
    }

    /// PREFETCH instructions (0F 18 /hint)
    fn encode_prefetch(&mut self, ops: &[Operand], hint: u8) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("prefetch requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Memory(mem) => {
                if mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name)) || mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name)) {
                    let rex = 0x40
                        | if mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name)) { 1 } else { 0 }
                        | if mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name)) { 2 } else { 0 };
                    self.bytes.push(rex);
                }
                self.bytes.push(0x0F);
                self.bytes.push(0x18);
                self.encode_modrm_mem(hint, mem)
            }
            _ => Err("prefetch requires memory operand".to_string()),
        }
    }

    /// RDRAND (0F C7 /6)
    fn encode_rdrand(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("rdrand requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Register(reg) => {
                let reg_n = reg_num(&reg.name).ok_or("bad register")?;
                let size = if mnemonic == "rdrandq" { 8 } else if mnemonic == "rdrandl" { 4 } else { infer_size_from_ops(ops).unwrap_or(8) };
                if size == 2 { self.bytes.push(0x66); }
                if size == 8 {
                    self.bytes.push(0x48 | if needs_rex_ext(&reg.name) { 1 } else { 0 });
                } else if needs_rex_ext(&reg.name) {
                    self.bytes.push(0x41);
                }
                self.bytes.push(0x0F);
                self.bytes.push(0xC7);
                self.bytes.push(self.modrm(3, 6, reg_n));
                Ok(())
            }
            _ => Err("rdrand requires register operand".to_string()),
        }
    }

    /// POPCNT (F3 0F B8 /r)
    fn encode_popcnt(&mut self, ops: &[Operand], size: u8) -> Result<(), String> {
        if ops.len() != 2 {
            return Err("popcnt requires 2 operands".to_string());
        }
        match (&ops[0], &ops[1]) {
            (Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.bytes.push(0xF3);
                if size == 8 {
                    let rex = 0x48 | if needs_rex_ext(&dst.name) { 4 } else { 0 } | if needs_rex_ext(&src.name) { 1 } else { 0 };
                    self.bytes.push(rex);
                } else if needs_rex_ext(&src.name) || needs_rex_ext(&dst.name) {
                    let rex = 0x40 | if needs_rex_ext(&dst.name) { 4 } else { 0 } | if needs_rex_ext(&src.name) { 1 } else { 0 };
                    self.bytes.push(rex);
                }
                self.bytes.push(0x0F);
                self.bytes.push(0xB8);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            (Operand::Memory(mem), Operand::Register(dst)) => {
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                if size == 2 { self.bytes.push(0x66); }
                self.bytes.push(0xF3);
                if size == 8 {
                    let rex = 0x48 | if needs_rex_ext(&dst.name) { 4 } else { 0 }
                        | if mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name)) { 1 } else { 0 }
                        | if mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name)) { 2 } else { 0 };
                    self.bytes.push(rex);
                } else if needs_rex_ext(&dst.name) || mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name)) || mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name)) {
                    let rex = 0x40 | if needs_rex_ext(&dst.name) { 4 } else { 0 }
                        | if mem.base.as_ref().map_or(false, |b| needs_rex_ext(&b.name)) { 1 } else { 0 }
                        | if mem.index.as_ref().map_or(false, |i| needs_rex_ext(&i.name)) { 2 } else { 0 };
                    self.bytes.push(rex);
                }
                self.bytes.push(0x0F);
                self.bytes.push(0xB8);
                self.encode_modrm_mem(dst_num, mem)
            }
            _ => Err("unsupported popcnt operands".to_string()),
        }
    }

    /// LOOP/LOOPE/LOOPNE/JRCXZ - short branch with 8-bit relative offset
    fn encode_loop(&mut self, ops: &[Operand], opcode: u8) -> Result<(), String> {
        if ops.len() != 1 {
            return Err("loop/jrcxz requires 1 operand".to_string());
        }
        match &ops[0] {
            Operand::Label(label) => {
                self.bytes.push(opcode);
                // Use internal PC8 relocation that gets resolved during assembly
                self.add_relocation(label, R_X86_64_PC8_INTERNAL, -1);
                self.bytes.push(0xFE); // placeholder
                Ok(())
            }
            _ => Err("loop/jrcxz requires label operand".to_string()),
        }
    }

    /// PBLENDVB - SSE4.1 variable blend (66 0F 38 10 /r, implicit xmm0)
    fn encode_pblendvb(&mut self, ops: &[Operand]) -> Result<(), String> {
        // Can be 2 operands (implicit xmm0) or 3 operands (explicit xmm0)
        if ops.len() != 2 && ops.len() != 3 {
            return Err("pblendvb requires 2-3 operands".to_string());
        }
        let (src, dst) = if ops.len() == 3 {
            (&ops[0], &ops[2])  // AT&T: src, xmm0_implicit, dst -> we want src and dst
        } else {
            (&ops[0], &ops[1])
        };
        match (src, dst) {
            (Operand::Register(src_reg), Operand::Register(dst_reg)) => {
                let src_num = reg_num(&src_reg.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst_reg.name).ok_or("bad register")?;
                self.bytes.push(0x66);
                if needs_rex_ext(&src_reg.name) || needs_rex_ext(&dst_reg.name) {
                    let rex = 0x40 | if needs_rex_ext(&dst_reg.name) { 4 } else { 0 } | if needs_rex_ext(&src_reg.name) { 1 } else { 0 };
                    self.bytes.push(rex);
                }
                self.bytes.push(0x0F);
                self.bytes.push(0x38);
                self.bytes.push(0x10);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                Ok(())
            }
            _ => Err("unsupported pblendvb operands".to_string()),
        }
    }

    /// RORX - VEX-encoded rotate right without affecting flags
    /// VEX.LZ.F2.0F3A.W0 F0 /r ib (32-bit)
    /// VEX.LZ.F2.0F3A.W1 F0 /r ib (64-bit)
    fn encode_rorx(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("rorx requires 3 operands".to_string());
        }
        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Immediate(ImmediateValue::Integer(imm)), Operand::Register(src), Operand::Register(dst)) => {
                let src_num = reg_num(&src.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let is_64 = mnemonic == "rorxq" || (mnemonic == "rorx" && infer_size_from_ops(&ops[1..]) == Some(8));
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src.name);
                // VEX.LZ.F2.0F3A.W{0,1} F0 /r ib
                // pp=3 for F2, mmmmm=3 for 0F3A
                self.emit_vex3(r, true, b, 3, is_64, 0xF, false, 3);
                self.bytes.push(0xF0);
                self.bytes.push(self.modrm(3, dst_num, src_num));
                self.bytes.push(*imm as u8);
                Ok(())
            }
            _ => Err("unsupported rorx operands".to_string()),
        }
    }

    /// ANDN - VEX-encoded AND NOT
    /// VEX.LZ.0F38.W0 F2 /r (32-bit)
    /// VEX.LZ.0F38.W1 F2 /r (64-bit)
    fn encode_andn(&mut self, ops: &[Operand], mnemonic: &str) -> Result<(), String> {
        if ops.len() != 3 {
            return Err("andn requires 3 operands".to_string());
        }
        match (&ops[0], &ops[1], &ops[2]) {
            (Operand::Register(src1), Operand::Register(src2), Operand::Register(dst)) => {
                let src1_num = reg_num(&src1.name).ok_or("bad register")?;
                let src2_num = reg_num(&src2.name).ok_or("bad register")?;
                let dst_num = reg_num(&dst.name).ok_or("bad register")?;
                let is_64 = mnemonic == "andnq" || (mnemonic == "andn" && infer_size_from_ops(ops) == Some(8));
                let r = !needs_rex_ext(&dst.name);
                let b = !needs_rex_ext(&src1.name);
                let vvvv = src2_num | if needs_rex_ext(&src2.name) { 8 } else { 0 };
                // VEX.LZ.0F38.W{0,1} F2 /r - pp=0 (no prefix), mmmmm=2 (0F38)
                self.emit_vex3(r, true, b, 2, is_64, vvvv, false, 0);
                self.bytes.push(0xF2);
                self.bytes.push(self.modrm(3, dst_num, src1_num));
                Ok(())
            }
            _ => Err("unsupported andn operands".to_string()),
        }
    }

    /// VPROLD - VEX-encoded packed rotate left (EVEX needed for AVX-512, but for AVX2 we use vpsll + vpsr combo)
    /// Actually vprold is only AVX-512, so this is a placeholder that returns an error for now
    fn encode_vprold(&mut self, _ops: &[Operand]) -> Result<(), String> {
        Err("vprold requires AVX-512 (EVEX encoding not supported)".to_string())
    }
}

/// Parse x87 register number: "st(0)" -> 0, "st" -> 0, "st(1)" -> 1, etc.
fn parse_st_num(name: &str) -> Result<u8, String> {
    if name == "st" || name == "st(0)" {
        return Ok(0);
    }
    if name.starts_with("st(") && name.ends_with(')') {
        let n: u8 = name[3..name.len()-1].parse()
            .map_err(|_| format!("bad st register: {}", name))?;
        if n > 7 {
            return Err(format!("st register out of range: {}", name));
        }
        return Ok(n);
    }
    Err(format!("not an st register: {}", name))
}

/// Get AT&T size suffix character for a given operand size.
fn size_suffix(size: u8) -> char {
    match size {
        1 => 'b',
        2 => 'w',
        4 => 'l',
        8 => 'q',
        _ => 'q',
    }
}

/// Map condition code suffix to encoding.
fn cc_from_mnemonic(cc_str: &str) -> Result<u8, String> {
    match cc_str {
        "o" => Ok(0),
        "no" => Ok(1),
        "b" | "c" | "nae" => Ok(2),
        "nb" | "nc" | "ae" => Ok(3),
        "e" | "z" => Ok(4),
        "ne" | "nz" => Ok(5),
        "be" | "na" => Ok(6),
        "nbe" | "a" => Ok(7),
        "s" => Ok(8),
        "ns" => Ok(9),
        "p" | "pe" => Ok(10),
        "np" | "po" => Ok(11),
        "l" | "nge" => Ok(12),
        "nl" | "ge" => Ok(13),
        "le" | "ng" => Ok(14),
        "nle" | "g" => Ok(15),
        _ => Err(format!("unknown condition code: {}", cc_str)),
    }
}
