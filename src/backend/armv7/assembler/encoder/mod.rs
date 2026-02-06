//! ARMv7 instruction encoder.
//!
//! Encodes ARM instructions into 32-bit machine words.
//! ARM mode only (no Thumb-2 for now).

use super::parser::{Operand, MemOffset};

/// ARM condition codes
pub fn encode_condition(cond: &str) -> u32 {
    match cond {
        "eq" => 0b0000,
        "ne" => 0b0001,
        "cs" | "hs" => 0b0010,
        "cc" | "lo" => 0b0011,
        "mi" => 0b0100,
        "pl" => 0b0101,
        "vs" => 0b0110,
        "vc" => 0b0111,
        "hi" => 0b1000,
        "ls" => 0b1001,
        "ge" => 0b1010,
        "lt" => 0b1011,
        "gt" => 0b1100,
        "le" => 0b1101,
        "al" | "" => 0b1110,
        _ => 0b1110, // Default: always
    }
}

/// Parse a mnemonic into (base_mnemonic, condition, set_flags)
pub fn parse_mnemonic(mnemonic: &str) -> (&str, &str, bool) {
    // Check for condition code suffix
    let conditions = [
        "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl",
        "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
    ];

    let m = mnemonic;

    // Check for 's' suffix (set flags) with condition
    for cc in &conditions {
        if m.len() > cc.len() + 1 {
            let suffix_start = m.len() - cc.len() - 1;
            if &m[suffix_start + 1..] == *cc && &m[suffix_start..suffix_start + 1] == "s" {
                return (&m[..suffix_start], cc, true);
            }
        }
    }

    // Check for condition code suffix only
    for cc in &conditions {
        if m.len() > cc.len() && m.ends_with(cc) {
            let base = &m[..m.len() - cc.len()];
            // Make sure the base is a known instruction
            if is_known_base_mnemonic(base) {
                return (base, cc, false);
            }
        }
    }

    // Check for 's' suffix only (set flags)
    if m.ends_with('s') && m.len() > 1 {
        let base = &m[..m.len() - 1];
        if is_known_base_mnemonic(base) {
            return (base, "", true);
        }
    }

    (m, "", false)
}

fn is_known_base_mnemonic(m: &str) -> bool {
    matches!(m,
        "add" | "sub" | "rsb" | "adc" | "sbc" | "rsc" |
        "and" | "orr" | "eor" | "bic" | "orn" |
        "mov" | "mvn" | "lsl" | "lsr" | "asr" | "ror" | "rrx" |
        "mul" | "mla" | "mls" | "umull" | "umlal" | "smull" | "smlal" |
        "cmp" | "cmn" | "tst" | "teq" |
        "b" | "bl" | "bx" | "blx" |
        "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" |
        "ldrsb" | "ldrsh" | "ldrd" | "strd" |
        "ldm" | "stm" | "push" | "pop" |
        "nop" | "svc" | "bkpt" | "dmb" | "dsb" | "isb" |
        "clz" | "rbit" | "rev" | "rev16" | "revsh" |
        "sxtb" | "sxth" | "uxtb" | "uxth" |
        "ldrex" | "ldrexb" | "ldrexh" | "strex" | "strexb" | "strexh" | "clrex" |
        "movw" | "movt" |
        "vadd" | "vsub" | "vmul" | "vdiv" | "vmov" | "vcvt" | "vcmp" | "vneg" | "vabs" |
        "vmrs" | "vldr" | "vstr" | "vpush" | "vpop" |
        "pld" | "udiv" | "sdiv"
    )
}

/// Encode a register name to its number (0-15).
pub fn encode_reg(name: &str) -> u32 {
    match name {
        "r0" => 0, "r1" => 1, "r2" => 2, "r3" => 3,
        "r4" => 4, "r5" => 5, "r6" => 6, "r7" => 7,
        "r8" => 8, "r9" => 9, "r10" => 10,
        "r11" | "fp" => 11,
        "r12" | "ip" => 12,
        "r13" | "sp" => 13,
        "r14" | "lr" => 14,
        "r15" | "pc" => 15,
        _ => panic!("unknown register: {}", name),
    }
}

/// Encode a VFP single-precision register (s0-s31).
pub fn encode_sreg(name: &str) -> u32 {
    let n: u32 = name[1..].parse().expect("invalid sreg");
    n
}

/// Encode a VFP double-precision register (d0-d15).
pub fn encode_dreg(name: &str) -> u32 {
    let n: u32 = name[1..].parse().expect("invalid dreg");
    n
}

/// Encode a register list to a 16-bit bitmask.
pub fn encode_reg_list(regs: &[String]) -> u32 {
    let mut mask = 0u32;
    for reg in regs {
        // Handle register ranges like "r4-r10"
        if reg.contains('-') {
            let parts: Vec<&str> = reg.split('-').collect();
            let start = encode_reg(parts[0].trim());
            let end = encode_reg(parts[1].trim());
            for r in start..=end {
                mask |= 1 << r;
            }
        } else {
            mask |= 1 << encode_reg(reg);
        }
    }
    mask
}

/// Try to encode a 32-bit immediate as an ARM "modified immediate"
/// (8-bit value rotated right by an even number of bits).
/// Returns Some((imm8, rotate)) or None if not encodable.
pub fn encode_arm_imm(value: u32) -> Option<(u32, u32)> {
    for rot in 0..16u32 {
        let rotation = rot * 2;
        let shifted = value.rotate_left(rotation);
        if shifted <= 0xFF {
            return Some((shifted, rot));
        }
    }
    None
}

/// Encode a data processing instruction.
/// Format: cond 00 I opcode S Rn Rd operand2
pub fn encode_data_proc(cond: u32, opcode: u32, set_flags: bool, rn: u32, rd: u32, operand2: u32, is_imm: bool) -> u32 {
    let i = if is_imm { 1 } else { 0 };
    let s = if set_flags { 1 } else { 0 };
    (cond << 28) | (i << 25) | (opcode << 21) | (s << 20) | (rn << 16) | (rd << 12) | operand2
}

/// Data processing opcodes
pub fn dp_opcode(mnemonic: &str) -> u32 {
    match mnemonic {
        "and" => 0b0000,
        "eor" => 0b0001,
        "sub" => 0b0010,
        "rsb" => 0b0011,
        "add" => 0b0100,
        "adc" => 0b0101,
        "sbc" => 0b0110,
        "rsc" => 0b0111,
        "tst" => 0b1000,
        "teq" => 0b1001,
        "cmp" => 0b1010,
        "cmn" => 0b1011,
        "orr" => 0b1100,
        "mov" => 0b1101,
        "bic" => 0b1110,
        "mvn" => 0b1111,
        _ => panic!("unknown dp opcode: {}", mnemonic),
    }
}

/// Encode a branch instruction.
/// Format: cond 101 L offset24
pub fn encode_branch(cond: u32, link: bool, offset: i32) -> u32 {
    let l = if link { 1 } else { 0 };
    let offset24 = ((offset >> 2) & 0x00FFFFFF) as u32;
    (cond << 28) | (0b101 << 25) | (l << 24) | offset24
}

/// Encode a load/store instruction.
/// Format: cond 01 I P U B W L Rn Rd offset
pub fn encode_load_store(
    cond: u32, is_load: bool, is_byte: bool,
    pre_index: bool, add: bool, writeback: bool,
    rn: u32, rd: u32, offset: u32, is_imm: bool,
) -> u32 {
    let i = if !is_imm { 1 } else { 0 }; // Note: inverted for load/store!
    let p = if pre_index { 1 } else { 0 };
    let u = if add { 1 } else { 0 };
    let b = if is_byte { 1 } else { 0 };
    let w = if writeback { 1 } else { 0 };
    let l = if is_load { 1 } else { 0 };
    (cond << 28) | (0b01 << 26) | (i << 25) | (p << 24) | (u << 23) | (b << 22) |
    (w << 21) | (l << 20) | (rn << 16) | (rd << 12) | offset
}

/// Encode a halfword/signed load/store (LDRH, LDRSH, LDRSB, STRH).
pub fn encode_halfword_load_store(
    cond: u32, is_load: bool, sign: bool, half: bool,
    pre_index: bool, add: bool, writeback: bool, is_imm: bool,
    rn: u32, rd: u32, offset: u32,
) -> u32 {
    let p = if pre_index { 1 } else { 0 };
    let u = if add { 1 } else { 0 };
    let i = if is_imm { 1 } else { 0 };
    let w = if writeback { 1 } else { 0 };
    let l = if is_load { 1 } else { 0 };
    let s = if sign { 1 } else { 0 };
    let h = if half { 1 } else { 0 };
    let imm_hi = (offset >> 4) & 0xF;
    let imm_lo = offset & 0xF;
    (cond << 28) | (p << 24) | (u << 23) | (i << 22) | (w << 21) | (l << 20) |
    (rn << 16) | (rd << 12) | (imm_hi << 8) | (1 << 7) | (s << 6) | (h << 5) | (1 << 4) | imm_lo
}

/// Encode a multiply instruction.
pub fn encode_multiply(cond: u32, rd: u32, rn: u32, rs: u32, rm: u32, accumulate: bool, set_flags: bool) -> u32 {
    let a = if accumulate { 1 } else { 0 };
    let s = if set_flags { 1 } else { 0 };
    (cond << 28) | (a << 21) | (s << 20) | (rd << 16) | (rn << 12) | (rs << 8) | (0b1001 << 4) | rm
}

/// Encode a block data transfer (LDM/STM, PUSH/POP).
pub fn encode_block_transfer(cond: u32, is_load: bool, pre_index: bool, add: bool, writeback: bool, rn: u32, reg_list: u32) -> u32 {
    let p = if pre_index { 1 } else { 0 };
    let u = if add { 1 } else { 0 };
    let w = if writeback { 1 } else { 0 };
    let l = if is_load { 1 } else { 0 };
    (cond << 28) | (0b100 << 25) | (p << 24) | (u << 23) | (w << 21) | (l << 20) | (rn << 16) | reg_list
}

/// Encode a BX instruction (Branch and Exchange).
pub fn encode_bx(cond: u32, rm: u32) -> u32 {
    (cond << 28) | 0x012FFF10 | rm
}

/// Encode a BLX register instruction.
pub fn encode_blx_reg(cond: u32, rm: u32) -> u32 {
    (cond << 28) | 0x012FFF30 | rm
}

/// Encode NOP.
pub fn encode_nop(cond: u32) -> u32 {
    (cond << 28) | 0x0320F000
}

/// Encode DMB/DSB/ISB.
pub fn encode_barrier(barrier_type: &str, option: &str) -> u32 {
    let op = match barrier_type {
        "dmb" => 0xF57FF050,
        "dsb" => 0xF57FF040,
        "isb" => 0xF57FF060,
        _ => 0xF57FF050,
    };
    let opt = match option {
        "sy" | "" => 0xF,
        "st" => 0xE,
        "ld" => 0xD,
        "ish" => 0xB,
        "ishst" => 0xA,
        "ishld" => 0x9,
        "nsh" => 0x7,
        "nshst" => 0x6,
        "osh" => 0x3,
        "oshst" => 0x2,
        _ => 0xF,
    };
    op | opt
}

/// Encode MOVW (move wide - lower 16 bits).
pub fn encode_movw(cond: u32, rd: u32, imm16: u32) -> u32 {
    let imm12 = imm16 & 0xFFF;
    let imm4 = (imm16 >> 12) & 0xF;
    (cond << 28) | (0b00110000 << 20) | (imm4 << 16) | (rd << 12) | imm12
}

/// Encode MOVT (move top - upper 16 bits).
pub fn encode_movt(cond: u32, rd: u32, imm16: u32) -> u32 {
    let imm12 = imm16 & 0xFFF;
    let imm4 = (imm16 >> 12) & 0xF;
    (cond << 28) | (0b00110100 << 20) | (imm4 << 16) | (rd << 12) | imm12
}

/// Encode CLZ instruction.
pub fn encode_clz(cond: u32, rd: u32, rm: u32) -> u32 {
    (cond << 28) | 0x016F0F10 | (rd << 12) | rm
}

/// Encode RBIT instruction.
pub fn encode_rbit(cond: u32, rd: u32, rm: u32) -> u32 {
    (cond << 28) | 0x06FF0F30 | (rd << 12) | rm
}

/// Encode REV instruction.
pub fn encode_rev(cond: u32, rd: u32, rm: u32) -> u32 {
    (cond << 28) | 0x06BF0F30 | (rd << 12) | rm
}

/// Encode SXTB/SXTH/UXTB/UXTH.
pub fn encode_extend(cond: u32, mnemonic: &str, rd: u32, rm: u32) -> u32 {
    let opcode = match mnemonic {
        "sxtb" => 0x06AF0070,
        "sxth" => 0x06BF0070,
        "uxtb" => 0x06EF0070,
        "uxth" => 0x06FF0070,
        _ => panic!("unknown extend: {}", mnemonic),
    };
    (cond << 28) | opcode | (rd << 12) | rm
}

/// Encode LDREX.
pub fn encode_ldrex(cond: u32, rt: u32, rn: u32) -> u32 {
    (cond << 28) | 0x01900F9F | (rn << 16) | (rt << 12)
}

/// Encode LDREXB (byte).
pub fn encode_ldrexb(cond: u32, rt: u32, rn: u32) -> u32 {
    (cond << 28) | 0x01D00F9F | (rn << 16) | (rt << 12)
}

/// Encode LDREXH (halfword).
pub fn encode_ldrexh(cond: u32, rt: u32, rn: u32) -> u32 {
    (cond << 28) | 0x01F00F9F | (rn << 16) | (rt << 12)
}

/// Encode STREX.
pub fn encode_strex(cond: u32, rd: u32, rt: u32, rn: u32) -> u32 {
    (cond << 28) | 0x01800F90 | (rn << 16) | (rd << 12) | rt
}

/// Encode STREXB (byte).
pub fn encode_strexb(cond: u32, rd: u32, rt: u32, rn: u32) -> u32 {
    (cond << 28) | 0x01C00F90 | (rn << 16) | (rd << 12) | rt
}

/// Encode STREXH (halfword).
pub fn encode_strexh(cond: u32, rd: u32, rt: u32, rn: u32) -> u32 {
    (cond << 28) | 0x01E00F90 | (rn << 16) | (rd << 12) | rt
}

/// Encode CLREX.
pub fn encode_clrex() -> u32 {
    0xF57FF01F
}

/// Encode UMULL instruction.
pub fn encode_umull(cond: u32, rdlo: u32, rdhi: u32, rn: u32, rm: u32, set_flags: bool) -> u32 {
    let s = if set_flags { 1 } else { 0 };
    (cond << 28) | (0b0000100 << 21) | (s << 20) | (rdhi << 16) | (rdlo << 12) | (rm << 8) | (0b1001 << 4) | rn
}

/// Encode MLA instruction.
pub fn encode_mla(cond: u32, rd: u32, rn: u32, rm: u32, ra: u32, set_flags: bool) -> u32 {
    let s = if set_flags { 1 } else { 0 };
    (cond << 28) | (0b0000001 << 21) | (s << 20) | (rd << 16) | (ra << 12) | (rm << 8) | (0b1001 << 4) | rn
}

// ============================================================
// VFP (Vector Floating-Point) instruction encoding
// ============================================================
// Reference: ARM Architecture Reference Manual ARMv7-A/R, sections A8.8.xxx

/// Split a single-precision register number (s0-s31) into (Vx, X) fields.
/// The 5-bit register number is encoded as: Vx = reg >> 1, X = reg & 1.
/// X goes into D (bit 22 for dest), N (bit 7 for Vn), or M (bit 5 for Vm).
fn split_sreg(reg: u32) -> (u32, u32) {
    (reg >> 1, reg & 1)
}

/// Split a double-precision register number (d0-d15) into (Vx, X) fields.
/// The register number is encoded as: X = (reg >> 4) & 1, Vx = reg & 0xF.
/// For VFPv3-D16 (d0-d15), X is always 0.
fn split_dreg(reg: u32) -> (u32, u32) {
    (reg & 0xF, (reg >> 4) & 1)
}

/// VFP arithmetic operation type.
pub enum VfpArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Encode a VFP 3-operand arithmetic instruction (vadd, vsub, vmul, vdiv).
/// `is_double`: true for .f64, false for .f32
/// `fd`, `fn_`, `fm`: register numbers (s0-s31 for single, d0-d15 for double)
pub fn encode_vfp_arith(cond: u32, op: VfpArithOp, is_double: bool, fd: u32, fn_: u32, fm: u32) -> u32 {
    let (vd, d_bit, vn, n_bit, vm, m_bit);
    if is_double {
        let (vd_, d_) = split_dreg(fd);
        let (vn_, n_) = split_dreg(fn_);
        let (vm_, m_) = split_dreg(fm);
        vd = vd_; d_bit = d_;
        vn = vn_; n_bit = n_;
        vm = vm_; m_bit = m_;
    } else {
        let (vd_, d_) = split_sreg(fd);
        let (vn_, n_) = split_sreg(fn_);
        let (vm_, m_) = split_sreg(fm);
        vd = vd_; d_bit = d_;
        vn = vn_; n_bit = n_;
        vm = vm_; m_bit = m_;
    }
    let sz = if is_double { 1u32 } else { 0 };
    // Encoding bits that vary by operation:
    // VADD: bit23=0, bits21:20=11, bit6=0
    // VSUB: bit23=0, bits21:20=11, bit6=1
    // VMUL: bit23=0, bits21:20=10, bit6=0
    // VDIV: bit23=1, bits21:20=00, bit6=0
    let (bit23, bits21_20, bit6) = match op {
        VfpArithOp::Add => (0u32, 0b11u32, 0u32),
        VfpArithOp::Sub => (0, 0b11, 1),
        VfpArithOp::Mul => (0, 0b10, 0),
        VfpArithOp::Div => (1, 0b00, 0),
    };
    (cond << 28) | (0b1110 << 24) | (bit23 << 23) | (d_bit << 22) | (bits21_20 << 20)
        | (vn << 16) | (vd << 12) | (0b101 << 9) | (sz << 8)
        | (n_bit << 7) | (bit6 << 6) | (m_bit << 5) | vm
}

/// Encode VMOV between ARM core register and single-precision register.
/// `to_fp`: true for VMOV Sn, Rt (core→FP), false for VMOV Rt, Sn (FP→core)
pub fn encode_vmov_core_single(cond: u32, to_fp: bool, sn: u32, rt: u32) -> u32 {
    let (vn, n_bit) = split_sreg(sn);
    let op = if to_fp { 0u32 } else { 1 };
    // cond 1110 000L Vn Rt 1010 N001 0000
    (cond << 28) | (0b1110 << 24) | (0b000 << 21) | (op << 20)
        | (vn << 16) | (rt << 12) | (0b1010 << 8) | (n_bit << 7) | (0b0001 << 4)
}

/// Encode VMOV between two ARM core registers and a double-precision register.
/// `to_fp`: true for VMOV Dm, Rt, Rt2 (core→FP), false for VMOV Rt, Rt2, Dm (FP→core)
pub fn encode_vmov_core_double(cond: u32, to_fp: bool, dm: u32, rt: u32, rt2: u32) -> u32 {
    let (vm, m_bit) = split_dreg(dm);
    let op = if to_fp { 0u32 } else { 1 };
    // cond 1100 010 op Rt2 Rt 1011 00M1 Vm
    (cond << 28) | (0b1100 << 24) | (0b010 << 21) | (op << 20)
        | (rt2 << 16) | (rt << 12) | (0b1011 << 8) | (m_bit << 5) | (1 << 4) | vm
}

/// Encode VCVT between single and double precision.
/// `to_double`: true for VCVT.F64.F32 Dd, Sm, false for VCVT.F32.F64 Sd, Dm
pub fn encode_vcvt_f2f(cond: u32, to_double: bool, fd: u32, fm: u32) -> u32 {
    let (vd, d_bit, vm, m_bit, sz);
    if to_double {
        // VCVT.F64.F32 Dd, Sm - dest is double, src is single
        // ARM ARM: sz=0 means double_to_single=FALSE (i.e. single→double)
        let (vd_, d_) = split_dreg(fd);
        let (vm_, m_) = split_sreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
        sz = 0u32;
    } else {
        // VCVT.F32.F64 Sd, Dm - dest is single, src is double
        // ARM ARM: sz=1 means double_to_single=TRUE (i.e. double→single)
        let (vd_, d_) = split_sreg(fd);
        let (vm_, m_) = split_dreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
        sz = 1u32;
    }
    // cond 11101 D 11 0111 Vd 101 sz 11 M 0 Vm
    (cond << 28) | (0b11101 << 23) | (d_bit << 22) | (0b11 << 20) | (0b0111 << 16)
        | (vd << 12) | (0b101 << 9) | (sz << 8) | (0b11 << 6) | (m_bit << 5) | vm
}

/// Encode VCVT from integer (in FP register) to floating-point.
/// `to_double`: result is F64 (true) or F32 (false)
/// `from_signed`: source is signed int (true) or unsigned (false)
/// `fd`: destination FP register (s-reg if F32, d-reg if F64)
/// `sm`: source single-precision register containing the integer
pub fn encode_vcvt_int_to_fp(cond: u32, to_double: bool, from_signed: bool, fd: u32, sm: u32) -> u32 {
    let (vd, d_bit);
    if to_double {
        let (vd_, d_) = split_dreg(fd);
        vd = vd_; d_bit = d_;
    } else {
        let (vd_, d_) = split_sreg(fd);
        vd = vd_; d_bit = d_;
    }
    let (vm, m_bit) = split_sreg(sm);
    let sz = if to_double { 1u32 } else { 0 };
    let op = if from_signed { 1u32 } else { 0 };
    // cond 11101 D 11 1000 Vd 101 sz op 1 M 0 Vm
    (cond << 28) | (0b11101 << 23) | (d_bit << 22) | (0b11 << 20) | (0b1000 << 16)
        | (vd << 12) | (0b101 << 9) | (sz << 8) | (op << 7) | (1 << 6) | (m_bit << 5) | vm
}

/// Encode VCVT from floating-point to integer (in FP register).
/// Round toward zero (VCVTR variant).
/// `from_double`: source is F64 (true) or F32 (false)
/// `to_signed`: result is signed int (true) or unsigned (false)
/// `sd`: destination single-precision register to hold integer result
/// `fm`: source FP register (s-reg if F32, d-reg if F64)
pub fn encode_vcvt_fp_to_int(cond: u32, from_double: bool, to_signed: bool, sd: u32, fm: u32) -> u32 {
    let (vd, d_bit) = split_sreg(sd);
    let (vm, m_bit);
    if from_double {
        let (vm_, m_) = split_dreg(fm);
        vm = vm_; m_bit = m_;
    } else {
        let (vm_, m_) = split_sreg(fm);
        vm = vm_; m_bit = m_;
    }
    let sz = if from_double { 1u32 } else { 0 };
    let opc2 = if to_signed { 0b101u32 } else { 0b100 };
    // cond 11101 D 11 1 opc2 Vd 101 sz 1 1 M 0 Vm
    // Round toward zero: bit 7 = 1
    (cond << 28) | (0b11101 << 23) | (d_bit << 22) | (0b11 << 20) | (1 << 19)
        | (opc2 << 16) | (vd << 12) | (0b101 << 9) | (sz << 8)
        | (1 << 7) | (1 << 6) | (m_bit << 5) | vm
}

/// Encode VCMP (compare two FP registers).
/// `is_double`: true for .f64, false for .f32
pub fn encode_vcmp(cond: u32, is_double: bool, fd: u32, fm: u32) -> u32 {
    let (vd, d_bit, vm, m_bit);
    if is_double {
        let (vd_, d_) = split_dreg(fd);
        let (vm_, m_) = split_dreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
    } else {
        let (vd_, d_) = split_sreg(fd);
        let (vm_, m_) = split_sreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
    }
    let sz = if is_double { 1u32 } else { 0 };
    // cond 11101 D 11 0100 Vd 101 sz 01 M 0 Vm
    (cond << 28) | (0b11101 << 23) | (d_bit << 22) | (0b11 << 20) | (0b0100 << 16)
        | (vd << 12) | (0b101 << 9) | (sz << 8) | (0b01 << 6) | (m_bit << 5) | vm
}

/// Encode VMRS APSR_nzcv, FPSCR (transfer FPSCR flags to APSR condition flags).
pub fn encode_vmrs_apsr(cond: u32) -> u32 {
    // cond 1110 1111 0001 1111 1010 0001 0000
    (cond << 28) | 0x0EF1FA10
}

/// Encode VNEG (FP negate).
pub fn encode_vneg(cond: u32, is_double: bool, fd: u32, fm: u32) -> u32 {
    let (vd, d_bit, vm, m_bit);
    if is_double {
        let (vd_, d_) = split_dreg(fd);
        let (vm_, m_) = split_dreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
    } else {
        let (vd_, d_) = split_sreg(fd);
        let (vm_, m_) = split_sreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
    }
    let sz = if is_double { 1u32 } else { 0 };
    // cond 11101 D 11 0001 Vd 101 sz 01 M 0 Vm
    (cond << 28) | (0b11101 << 23) | (d_bit << 22) | (0b11 << 20) | (0b0001 << 16)
        | (vd << 12) | (0b101 << 9) | (sz << 8) | (0b01 << 6) | (m_bit << 5) | vm
}

/// Encode VABS (FP absolute value).
pub fn encode_vabs(cond: u32, is_double: bool, fd: u32, fm: u32) -> u32 {
    let (vd, d_bit, vm, m_bit);
    if is_double {
        let (vd_, d_) = split_dreg(fd);
        let (vm_, m_) = split_dreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
    } else {
        let (vd_, d_) = split_sreg(fd);
        let (vm_, m_) = split_sreg(fm);
        vd = vd_; d_bit = d_;
        vm = vm_; m_bit = m_;
    }
    let sz = if is_double { 1u32 } else { 0 };
    // cond 11101 D 11 0000 Vd 101 sz 11 M 0 Vm
    (cond << 28) | (0b11101 << 23) | (d_bit << 22) | (0b11 << 20) | (0b0000 << 16)
        | (vd << 12) | (0b101 << 9) | (sz << 8) | (0b11 << 6) | (m_bit << 5) | vm
}

/// Parse an FP register name and return (is_double, reg_number).
pub fn parse_fp_reg(name: &str) -> Option<(bool, u32)> {
    if let Some(rest) = name.strip_prefix('d') {
        if let Ok(n) = rest.parse::<u32>() {
            if n <= 15 { return Some((true, n)); }
        }
    }
    if let Some(rest) = name.strip_prefix('s') {
        if let Ok(n) = rest.parse::<u32>() {
            if n <= 31 { return Some((false, n)); }
        }
    }
    None
}

/// Check if a register name is a GP register (r0-r15, sp, lr, pc, etc.).
pub fn is_gp_reg(name: &str) -> bool {
    matches!(name,
        "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" |
        "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15" |
        "fp" | "ip" | "sp" | "lr" | "pc"
    )
}

/// Check if a register name is a VFP register (s0-s31, d0-d15).
pub fn is_fp_reg(name: &str) -> bool {
    parse_fp_reg(name).is_some()
}
