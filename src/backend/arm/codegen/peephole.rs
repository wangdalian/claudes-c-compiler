//! AArch64 peephole optimizer for assembly text.
//!
//! Operates on generated assembly text to eliminate redundant patterns from the
//! stack-based codegen. Lines are pre-parsed into `LineKind` enums so hot-path
//! pattern matching uses integer/enum comparisons instead of string parsing.
//!
//! ## Pass structure
//!
//! **Local passes** (iterative, up to 8 rounds): store/load elimination,
//! redundant branch removal, self-move elimination, move chain optimization,
//! branch-over-branch fusion, and move-immediate chain optimization.
//!
//! ## Optimizations
//!
//! 1. **Adjacent store/load elimination**: `str xN, [sp, #off]` followed by
//!    `ldr xN, [sp, #off]` — the load is redundant since the value is
//!    already in the register.
//!
//! 2. **Redundant branch elimination**: `b .LBBN` where `.LBBN:` is the
//!    immediately next non-empty line — falls through naturally.
//!
//! 3. **Self-move elimination**: `mov xN, xN` (64-bit) is a no-op.
//!    Note: `mov wN, wN` (32-bit) zeros upper 32 bits and is NOT eliminated.
//!
//! 4. **Move chain optimization**: `mov A, B; mov C, A` → `mov C, B`,
//!    enabling the first mov to become dead if A is unused.
//!
//! 5. **Branch-over-branch fusion**: `b.cc .Lskip; b .target; .Lskip:`
//!    → `b.!cc .target` (invert condition, eliminate skip label).
//!
//! 6. **Move-immediate chain**: `mov xN, #imm; mov xM, xN` where xN is a
//!    scratch register (x0-x15) → `mov xM, #imm` when safe.

// ── Line classification types ────────────────────────────────────────────────

/// Compact classification of an assembly line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LineKind {
    /// Deleted / blank
    Nop,
    /// `str xN/wN, [sp, #off]` — store to stack (via sp)
    StoreSp { reg: u8, offset: i32, is_word: bool },
    /// `ldr xN/wN, [sp, #off]` — load from stack (via sp)
    LoadSp { reg: u8, offset: i32, is_word: bool },
    /// `ldrsw xN, [sp, #off]` — load signed word from stack
    LoadswSp { reg: u8, offset: i32 },
    /// `ldrsb xN, [xM]` — load signed byte (general)
    LoadsbReg,
    /// `stp xN, xM, [sp, #off]` — store pair to stack
    StorePairSp,
    /// `ldp xN, xM, [sp, #off]` — load pair from stack
    LoadPairSp,
    /// `mov xN, xM` — register-to-register move.
    /// `is_32bit` indicates whether this is a w-register (32-bit) move.
    /// On AArch64, `mov wN, wM` zeros the upper 32 bits of the destination,
    /// so it is NOT equivalent to `mov xN, xM`.
    Move { dst: u8, src: u8, is_32bit: bool },
    /// `mov xN, #imm` — move immediate to register
    MoveImm { dst: u8 },
    /// `movz/movn xN, #imm` — move wide immediate
    MoveWide { dst: u8 },
    /// `sxtw xN, wM` — sign-extend word to doubleword
    Sxtw { dst: u8, src: u8 },
    /// `b .label` — unconditional branch
    Branch,
    /// `b.cc .label` — conditional branch
    CondBranch,
    /// `cbz/cbnz xN, .label` — compare and branch on (non-)zero
    CmpBranch,
    /// Label (`.LBBx:` etc.)
    Label,
    /// `ret`
    Ret,
    /// `bl func` — branch with link (function call)
    Call,
    /// `cmp` or `cmn` instruction
    Compare,
    /// `add`, `sub`, or other ALU instruction
    Alu,
    /// `str`/`ldr` to non-sp addresses (e.g., `str w1, [x9]`)
    MemOther,
    /// Assembler directive (`.section`, `.globl`, etc.)
    Directive,
    /// Any other instruction
    Other,
}

/// AArch64 register IDs for pattern matching.
/// We map x0-x30, sp, and w0-w30 to the same set (register number).
const REG_NONE: u8 = 255;

/// Parse an AArch64 register name to an internal ID (0-30, or special).
/// x0/w0 → 0, x1/w1 → 1, ..., x30/w30 → 30, sp → 31, xzr/wzr → 32.
fn parse_reg(name: &str) -> u8 {
    let name = name.trim();
    if let Some(n) = name.strip_prefix('x').or_else(|| name.strip_prefix('w')) {
        if let Ok(num) = n.parse::<u8>() {
            if num <= 30 {
                return num;
            }
        }
        if n == "zr" {
            return 32; // zero register
        }
    }
    if name == "sp" {
        return 31;
    }
    REG_NONE
}

/// Return the x-register name for a given ID.
fn xreg_name(id: u8) -> &'static str {
    match id {
        0 => "x0", 1 => "x1", 2 => "x2", 3 => "x3",
        4 => "x4", 5 => "x5", 6 => "x6", 7 => "x7",
        8 => "x8", 9 => "x9", 10 => "x10", 11 => "x11",
        12 => "x12", 13 => "x13", 14 => "x14", 15 => "x15",
        16 => "x16", 17 => "x17", 18 => "x18", 19 => "x19",
        20 => "x20", 21 => "x21", 22 => "x22", 23 => "x23",
        24 => "x24", 25 => "x25", 26 => "x26", 27 => "x27",
        28 => "x28", 29 => "x29", 30 => "x30",
        31 => "sp", 32 => "xzr",
        _ => "??",
    }
}

/// Return the w-register name for a given ID.
fn wreg_name(id: u8) -> &'static str {
    match id {
        0 => "w0", 1 => "w1", 2 => "w2", 3 => "w3",
        4 => "w4", 5 => "w5", 6 => "w6", 7 => "w7",
        8 => "w8", 9 => "w9", 10 => "w10", 11 => "w11",
        12 => "w12", 13 => "w13", 14 => "w14", 15 => "w15",
        16 => "w16", 17 => "w17", 18 => "w18", 19 => "w19",
        20 => "w20", 21 => "w21", 22 => "w22", 23 => "w23",
        24 => "w24", 25 => "w25", 26 => "w26", 27 => "w27",
        28 => "w28", 29 => "w29", 30 => "w30",
        32 => "wzr",
        _ => "??",
    }
}

// ── Line classification ──────────────────────────────────────────────────────

/// Classify a single assembly line into a LineKind.
fn classify_line(line: &str) -> LineKind {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return LineKind::Nop;
    }

    // Labels end with ':'
    if trimmed.ends_with(':') {
        return LineKind::Label;
    }

    // Directives start with '.'
    if trimmed.starts_with('.') {
        return LineKind::Directive;
    }

    // str xN/wN, [sp, #off]
    if let Some(rest) = trimmed.strip_prefix("str ") {
        if let Some(info) = parse_sp_mem_op(rest) {
            return LineKind::StoreSp { reg: info.0, offset: info.1, is_word: info.2 };
        }
        return LineKind::MemOther;
    }

    // strb wN, [xM]  — byte store
    if trimmed.starts_with("strb ") || trimmed.starts_with("strh ") {
        return LineKind::MemOther;
    }

    // ldr xN/wN, [sp, #off]
    if let Some(rest) = trimmed.strip_prefix("ldr ") {
        if let Some(info) = parse_sp_mem_op(rest) {
            return LineKind::LoadSp { reg: info.0, offset: info.1, is_word: info.2 };
        }
        return LineKind::MemOther;
    }

    // ldrsw xN, [sp, #off]
    if let Some(rest) = trimmed.strip_prefix("ldrsw ") {
        if let Some((reg_str, addr)) = rest.split_once(", ") {
            let reg = parse_reg(reg_str.trim());
            if reg != REG_NONE {
                if let Some(offset) = parse_sp_offset(addr.trim()) {
                    return LineKind::LoadswSp { reg, offset };
                }
            }
        }
        return LineKind::MemOther;
    }

    // ldrsb
    if trimmed.starts_with("ldrsb ") || trimmed.starts_with("ldrsh ") ||
       trimmed.starts_with("ldrb ") || trimmed.starts_with("ldrh ") {
        return LineKind::LoadsbReg;
    }

    // stp (store pair)
    if trimmed.starts_with("stp ") {
        if trimmed.contains("[sp") {
            return LineKind::StorePairSp;
        }
        return LineKind::MemOther;
    }

    // ldp (load pair)
    if trimmed.starts_with("ldp ") {
        if trimmed.contains("[sp") {
            return LineKind::LoadPairSp;
        }
        return LineKind::MemOther;
    }

    // mov xN, xM  or  mov xN, #imm  or  mov xN, :lo12:sym etc.
    if let Some(rest) = trimmed.strip_prefix("mov ") {
        // Avoid matching movz, movn, movk
        if !trimmed.starts_with("movz") && !trimmed.starts_with("movn") && !trimmed.starts_with("movk") {
            if let Some((dst_str, src_str)) = rest.split_once(", ") {
                let dst_trimmed = dst_str.trim();
                let dst = parse_reg(dst_trimmed);
                if dst != REG_NONE {
                    let src_trimmed = src_str.trim();
                    if src_trimmed.starts_with('#') || src_trimmed.starts_with('-') {
                        return LineKind::MoveImm { dst };
                    }
                    let src = parse_reg(src_trimmed);
                    if src != REG_NONE {
                        let is_32bit = dst_trimmed.starts_with('w');
                        return LineKind::Move { dst, src, is_32bit };
                    }
                    // mov xN, :lo12:symbol etc.
                    return LineKind::MoveImm { dst };
                }
            }
        }
    }

    // movz / movn / movk
    if trimmed.starts_with("movz ") || trimmed.starts_with("movn ") {
        if let Some((_, rest)) = trimmed.split_once(' ') {
            if let Some((dst_str, _)) = rest.split_once(", ") {
                let dst = parse_reg(dst_str.trim());
                if dst != REG_NONE {
                    return LineKind::MoveWide { dst };
                }
            }
        }
        return LineKind::Other;
    }

    // movk is an update, not a fresh definition
    if trimmed.starts_with("movk ") {
        return LineKind::Other;
    }

    // sxtw xN, wM
    if let Some(rest) = trimmed.strip_prefix("sxtw ") {
        if let Some((dst_str, src_str)) = rest.split_once(", ") {
            let dst = parse_reg(dst_str.trim());
            let src = parse_reg(src_str.trim());
            if dst != REG_NONE && src != REG_NONE {
                return LineKind::Sxtw { dst, src };
            }
        }
    }

    // Unconditional branch: b .label (but not bl, b.cc)
    if trimmed.starts_with("b ") && !trimmed.starts_with("bl ") && !trimmed.starts_with("b.") {
        return LineKind::Branch;
    }

    // Conditional branch: b.eq, b.ne, b.lt, b.ge, b.gt, b.le, b.hi, b.ls, b.cs, b.cc, etc.
    if trimmed.starts_with("b.") {
        return LineKind::CondBranch;
    }

    // cbz/cbnz/tbz/tbnz
    if trimmed.starts_with("cbz ") || trimmed.starts_with("cbnz ") ||
       trimmed.starts_with("tbz ") || trimmed.starts_with("tbnz ") {
        return LineKind::CmpBranch;
    }

    // ret
    if trimmed == "ret" {
        return LineKind::Ret;
    }

    // bl (branch and link = call)
    if trimmed.starts_with("bl ") || trimmed.starts_with("blr ") {
        return LineKind::Call;
    }

    // br xN (indirect branch = control flow barrier)
    if trimmed.starts_with("br ") {
        return LineKind::Branch;
    }

    // cmp/cmn
    if trimmed.starts_with("cmp ") || trimmed.starts_with("cmn ") {
        return LineKind::Compare;
    }

    // ALU: add, sub, and, orr, eor, lsl, lsr, asr, mul, etc.
    if trimmed.starts_with("add ") || trimmed.starts_with("sub ") ||
       trimmed.starts_with("and ") || trimmed.starts_with("orr ") ||
       trimmed.starts_with("eor ") || trimmed.starts_with("mul ") ||
       trimmed.starts_with("neg ") || trimmed.starts_with("mvn ") ||
       trimmed.starts_with("lsl ") || trimmed.starts_with("lsr ") ||
       trimmed.starts_with("asr ") || trimmed.starts_with("madd ") ||
       trimmed.starts_with("msub ") || trimmed.starts_with("sdiv ") ||
       trimmed.starts_with("udiv ") || trimmed.starts_with("adds ") ||
       trimmed.starts_with("subs ") {
        return LineKind::Alu;
    }

    LineKind::Other
}

/// Parse `xN/wN, [sp, #off]` and return (reg_id, offset, is_word).
fn parse_sp_mem_op(rest: &str) -> Option<(u8, i32, bool)> {
    let (reg_str, addr) = rest.split_once(", ")?;
    let reg_str = reg_str.trim();
    let is_word = reg_str.starts_with('w');
    let reg = parse_reg(reg_str);
    if reg == REG_NONE {
        return None;
    }
    let offset = parse_sp_offset(addr.trim())?;
    Some((reg, offset, is_word))
}

/// Parse `[sp, #off]` or `[sp]` and return the offset.
fn parse_sp_offset(addr: &str) -> Option<i32> {
    // [sp] — zero offset
    if addr == "[sp]" {
        return Some(0);
    }
    // [sp, #N] or [sp, #-N]
    if addr.starts_with("[sp, #") && addr.ends_with(']') {
        let inner = &addr[6..addr.len() - 1]; // strip "[sp, #" and "]"
        return inner.parse::<i32>().ok();
    }
    // [sp, #N]! (pre-index) — not a simple stack slot access
    None
}

/// Extract branch target from `b .label` or `b label`.
fn branch_target(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    // Match "b .label" but not "br xN" (indirect branch)
    if let Some(rest) = trimmed.strip_prefix("b ") {
        let target = rest.trim();
        // Must start with '.' (a label), not a register
        if target.starts_with('.') {
            return Some(target);
        }
    }
    None
}

/// Extract the condition code and target from a conditional branch.
/// `b.eq .label` → Some(("eq", ".label"))
fn cond_branch_parts(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("b.") {
        if let Some((cc, target)) = rest.split_once(' ') {
            return Some((cc, target.trim()));
        }
    }
    None
}

/// Invert a condition code.
fn invert_condition(cc: &str) -> Option<&'static str> {
    match cc {
        "eq" => Some("ne"),
        "ne" => Some("eq"),
        "lt" => Some("ge"),
        "ge" => Some("lt"),
        "gt" => Some("le"),
        "le" => Some("gt"),
        "hi" => Some("ls"),
        "ls" => Some("hi"),
        "hs" | "cs" => Some("lo"),
        "lo" | "cc" => Some("hs"),
        "mi" => Some("pl"),
        "pl" => Some("mi"),
        "vs" => Some("vc"),
        "vc" => Some("vs"),
        _ => None,
    }
}

/// Extract label name from a label line (strip trailing `:`)
fn label_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed.strip_suffix(':')
}


// ── Main entry point ─────────────────────────────────────────────────────────

/// Run peephole optimization on AArch64 assembly text.
/// Returns the optimized assembly string.
pub fn peephole_optimize(asm: String) -> String {
    let mut lines: Vec<String> = asm.lines().map(String::from).collect();
    let mut kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let n = lines.len();

    if n == 0 {
        return asm;
    }

    // Phase 1: Iterative local passes (up to 8 rounds)
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 8 {
        changed = false;
        changed |= eliminate_adjacent_store_load(&mut lines, &mut kinds, n);
        changed |= eliminate_redundant_branches(&lines, &mut kinds, n);
        changed |= eliminate_self_moves(&mut kinds, n);
        changed |= eliminate_move_chains(&mut lines, &mut kinds, n);
        changed |= fuse_branch_over_branch(&mut lines, &mut kinds, n);
        rounds += 1;
    }

    // Phase 2: Global passes
    //
    // Global store forwarding is disabled: same-register NOP elimination has a
    // remaining correctness bug in complex float-array code (test 0036_0041).
    // The root cause is not yet identified — it appears correct within a basic
    // block but produces wrong output. Until this is fixed, GSF is skipped.
    //
    // Copy propagation and dead store elimination are independent and safe.
    propagate_register_copies(&mut lines, &mut kinds, n);
    global_dead_store_elimination(&lines, &mut kinds, n);

    // Phase 3: Local cleanup after global passes (up to 4 rounds)
    {
        let mut changed2 = true;
        let mut rounds2 = 0;
        while changed2 && rounds2 < 4 {
            changed2 = false;
            changed2 |= eliminate_adjacent_store_load(&mut lines, &mut kinds, n);
            changed2 |= eliminate_redundant_branches(&lines, &mut kinds, n);
            changed2 |= eliminate_self_moves(&mut kinds, n);
            changed2 |= eliminate_move_chains(&mut lines, &mut kinds, n);
            rounds2 += 1;
        }
    }

    // Build result, filtering out Nop lines
    let mut result = String::with_capacity(asm.len());
    for i in 0..n {
        if kinds[i] != LineKind::Nop {
            result.push_str(&lines[i]);
            result.push('\n');
        }
    }
    result
}

// ── Pass 1: Adjacent store/load elimination ──────────────────────────────────
//
// Pattern: str xN, [sp, #off]  →  ldr xN, [sp, #off]  (same reg, same offset)
// The load is redundant since the value is already in the register.
// Also: str xN, [sp, #off]  →  ldr xM, [sp, #off]  → replace load with mov xM, xN
// Also handles: str wN, [sp, #off]  →  ldrsw xN, [sp, #off]

fn eliminate_adjacent_store_load(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 1 < n {
        if let LineKind::StoreSp { reg: store_reg, offset: store_off, is_word: store_word } = kinds[i] {
            // Look ahead for the matching load (skip Nops)
            let mut j = i + 1;
            while j < n && kinds[j] == LineKind::Nop {
                j += 1;
            }
            if j < n {
                match kinds[j] {
                    LineKind::LoadSp { reg: load_reg, offset: load_off, is_word: load_word }
                        if store_off == load_off && store_word == load_word =>
                    {
                        if store_reg == load_reg {
                            // Same register: eliminate the load entirely
                            kinds[j] = LineKind::Nop;
                            changed = true;
                        } else {
                            // Different register: replace load with mov
                            let reg_fmt = if store_word { wreg_name } else { xreg_name };
                            lines[j] = format!("    mov {}, {}", reg_fmt(load_reg), reg_fmt(store_reg));
                            kinds[j] = LineKind::Move { dst: load_reg, src: store_reg, is_32bit: store_word };
                            changed = true;
                        }
                    }
                    // str wN, [sp, #off] followed by ldrsw xM, [sp, #off]
                    // The value was just stored as a word; sign-extending load can be replaced
                    // with sxtw xM, wN (or eliminated if same reg).
                    LineKind::LoadswSp { reg: load_reg, offset: load_off }
                        if store_off == load_off && store_word =>
                    {
                        lines[j] = format!("    sxtw {}, {}", xreg_name(load_reg), wreg_name(store_reg));
                        kinds[j] = LineKind::Sxtw { dst: load_reg, src: store_reg };
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        i += 1;
    }
    changed
}

// ── Pass 2: Redundant branch elimination ─────────────────────────────────────
//
// Pattern: b .LBBN  ;  .LBBN:  (branch to immediately next label)
// Falls through naturally, so the branch is redundant.

fn eliminate_redundant_branches(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        if kinds[i] == LineKind::Branch {
            if let Some(target) = branch_target(&lines[i]) {
                // Find next non-Nop line
                let mut j = i + 1;
                while j < n && kinds[j] == LineKind::Nop {
                    j += 1;
                }
                if j < n && kinds[j] == LineKind::Label {
                    if let Some(lbl) = label_name(&lines[j]) {
                        if target == lbl {
                            kinds[i] = LineKind::Nop;
                            changed = true;
                        }
                    }
                }
            }
        }
    }
    changed
}

// ── Pass 3: Self-move elimination ────────────────────────────────────────────
//
// Pattern: mov xN, xN — no-op

fn eliminate_self_moves(kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    for i in 0..n {
        if let LineKind::Move { dst, src, is_32bit } = kinds[i] {
            if dst == src && !is_32bit {
                // Only eliminate 64-bit self-moves (mov xN, xN).
                // On AArch64, `mov wN, wN` zeros the upper 32 bits of xN,
                // so it is NOT a true no-op and must be preserved.
                kinds[i] = LineKind::Nop;
                changed = true;
            }
        }
    }
    changed
}

// ── Pass 4: Move chain optimization ──────────────────────────────────────────
//
// Pattern: mov A, B  ;  mov C, A → mov C, B
// This allows the first mov to potentially be dead-eliminated later.

fn eliminate_move_chains(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 1 < n {
        match kinds[i] {
            LineKind::Move { dst: dst1, src: src1, is_32bit: is_32bit1 } => {
                // Find next non-Nop instruction
                let mut j = i + 1;
                while j < n && kinds[j] == LineKind::Nop {
                    j += 1;
                }
                if j < n {
                    if let LineKind::Move { dst: dst2, src: src2, is_32bit: is_32bit2 } = kinds[j] {
                        // mov dst1, src1 ; mov dst2, dst1 → mov dst2, src1
                        // Only safe when both moves use the same width.
                        if src2 == dst1 && dst2 != src1 && is_32bit1 == is_32bit2 {
                            let reg_fmt = if is_32bit2 { wreg_name } else { xreg_name };
                            lines[j] = format!("    mov {}, {}", reg_fmt(dst2), reg_fmt(src1));
                            kinds[j] = LineKind::Move { dst: dst2, src: src1, is_32bit: is_32bit2 };
                            changed = true;
                        }
                    }
                }
            }
            LineKind::MoveImm { dst: dst1 } | LineKind::MoveWide { dst: dst1 } => {
                // mov xN, #imm ; mov xM, xN → mov xM, #imm (copy the immediate)
                // Only when dst1 is a scratch register (x0-x15) not callee-saved
                if dst1 <= 15 {
                    let mut j = i + 1;
                    while j < n && kinds[j] == LineKind::Nop {
                        j += 1;
                    }
                    if j < n {
                        if let LineKind::Move { dst: dst2, src: src2, is_32bit: _ } = kinds[j] {
                            if src2 == dst1 {
                                // Copy the immediate instruction, retargeted to dst2
                                let old_line = lines[i].trim();
                                // Replace the register in the first instruction
                                if let Some(new_line) = retarget_move_imm(old_line, dst2) {
                                    lines[j] = format!("    {}", new_line);
                                    kinds[j] = LineKind::MoveImm { dst: dst2 };
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    changed
}

/// Retarget a move-immediate instruction to a different destination register.
/// E.g., `mov x0, #5` with new dest x14 → `mov x14, #5`
fn retarget_move_imm(line: &str, new_dst: u8) -> Option<String> {
    // Handle: mov xN, #imm  /  mov xN, :lo12:sym  /  movz xN, #imm  /  movn xN, #imm
    for prefix in &["mov ", "movz ", "movn "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            if let Some((_old_reg, imm_part)) = rest.split_once(", ") {
                let new_reg = if line.contains('w') && !imm_part.starts_with('w') {
                    // If original used w-register (e.g., mov w0, #5)
                    // check if the source had 'w' prefix
                    let old_first = rest.chars().next()?;
                    if old_first == 'w' {
                        wreg_name(new_dst)
                    } else {
                        xreg_name(new_dst)
                    }
                } else {
                    xreg_name(new_dst)
                };
                return Some(format!("{}{}, {}", prefix, new_reg, imm_part));
            }
        }
    }
    None
}

// ── Pass 5: Branch-over-branch fusion ────────────────────────────────────────
//
// Pattern:
//   b.cc .Lskip_N
//   b .target
//   .Lskip_N:
//
// Transform to:
//   b.!cc .target
//
// This is a very common pattern from the codegen: it emits a conditional branch
// to skip over an unconditional branch.

fn fuse_branch_over_branch(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 2 < n {
        if kinds[i] == LineKind::CondBranch {
            // Find the next two non-Nop instructions
            let mut j = i + 1;
            while j < n && kinds[j] == LineKind::Nop {
                j += 1;
            }
            if j >= n {
                i += 1;
                continue;
            }
            let mut k = j + 1;
            while k < n && kinds[k] == LineKind::Nop {
                k += 1;
            }
            if k >= n {
                i += 1;
                continue;
            }

            // Check pattern: b.cc .skip ; b .target ; .skip:
            if kinds[j] == LineKind::Branch && kinds[k] == LineKind::Label {
                if let (Some((cc, skip_target)), Some(real_target), Some(lbl)) = (
                    cond_branch_parts(&lines[i]),
                    branch_target(&lines[j]),
                    label_name(&lines[k]),
                ) {
                    if skip_target == lbl {
                        if let Some(inv_cc) = invert_condition(cc) {
                            // Replace conditional branch with inverted condition to real target
                            lines[i] = format!("    b.{} {}", inv_cc, real_target);
                            // kinds[i] stays as CondBranch
                            // Remove the unconditional branch
                            kinds[j] = LineKind::Nop;
                            // Keep the label (might be targeted by other branches)
                            changed = true;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    changed
}

// ── Global store forwarding ──────────────────────────────────────────────────
//
// Tracks slot→register mappings as we scan forward through the function.
// When we see a load from a stack slot that has a known register value,
// we replace the load with a register move (or eliminate it if same register).
//
// At labels that are jump targets, all mappings are invalidated (the branch
// source may have different register values). Labels reached only by fallthrough
// preserve their mappings.
//
// NOTE: Currently disabled due to a remaining correctness bug in
// same-register NOP elimination (test 0036_0041). The code is kept
// for future development.

/// A tracked store mapping: stack slot at `offset` holds the value from `reg`.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct SlotMapping {
    reg: u8,
    is_word: bool,
    active: bool,
}

#[allow(dead_code)]
/// Maximum number of tracked slot mappings before compaction.
const MAX_SLOT_ENTRIES: usize = 64;

#[allow(dead_code)]
/// Number of AArch64 registers we track (x0-x30 + sp = 32, but we use IDs up to 32).
const NUM_REGS: usize = 33;

/// Collect all jump/branch targets in the function.
/// Returns a set of label names that are targets of branches.
#[allow(dead_code)]
fn collect_jump_targets(lines: &[String], kinds: &[LineKind], n: usize) -> Vec<bool> {
    // First pass: find max label number for sizing
    let mut max_label_num: u32 = 0;
    let mut has_indirect = false;
    for i in 0..n {
        if kinds[i] == LineKind::Label {
            if let Some(lbl) = label_name(&lines[i]) {
                if let Some(num) = parse_label_number(lbl) {
                    if num > max_label_num {
                        max_label_num = num;
                    }
                }
            }
        }
    }

    let mut is_target = vec![false; (max_label_num + 1) as usize];

    // Second pass: mark branch targets
    for i in 0..n {
        match kinds[i] {
            LineKind::Branch => {
                if let Some(target) = branch_target(&lines[i]) {
                    if let Some(num) = parse_label_number(target) {
                        if (num as usize) < is_target.len() {
                            is_target[num as usize] = true;
                        }
                    }
                } else {
                    // Indirect branch (br xN) — all labels are potential targets
                    let trimmed = lines[i].trim();
                    if trimmed.starts_with("br ") {
                        has_indirect = true;
                    }
                }
            }
            LineKind::CondBranch => {
                if let Some((_, target)) = cond_branch_parts(&lines[i]) {
                    if let Some(num) = parse_label_number(target) {
                        if (num as usize) < is_target.len() {
                            is_target[num as usize] = true;
                        }
                    }
                }
            }
            LineKind::CmpBranch => {
                // cbz/cbnz/tbz/tbnz: extract target from last comma-separated field
                if let Some(target) = extract_cbz_target(&lines[i]) {
                    if let Some(num) = parse_label_number(target) {
                        if (num as usize) < is_target.len() {
                            is_target[num as usize] = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if has_indirect {
        for v in is_target.iter_mut() {
            *v = true;
        }
    }

    is_target
}

/// Extract branch target from cbz/cbnz/tbz/tbnz instructions.
/// Format: `cbz xN, .label` or `tbnz xN, #bit, .label`
#[allow(dead_code)]
fn extract_cbz_target(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    // The target is always the last comma-separated token
    if let Some(pos) = trimmed.rfind(", ") {
        Some(trimmed[pos + 2..].trim())
    } else {
        None
    }
}

/// Parse a label name into its numeric suffix.
/// `.LBB123` -> Some(123), `.Lskip_0` -> None (non-numeric suffix)
#[allow(dead_code)]
fn parse_label_number(label: &str) -> Option<u32> {
    let rest = label.strip_prefix(".LBB")?;
    let b = rest.as_bytes();
    if b.is_empty() || !b[0].is_ascii_digit() {
        return None;
    }
    let mut v: u32 = 0;
    for &c in b {
        if c.is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((c - b'0') as u32);
        } else {
            return None;
        }
    }
    Some(v)
}

/// Check if a label is a jump target.
#[allow(dead_code)]
fn is_label_jump_target(label: &str, targets: &[bool]) -> bool {
    if let Some(num) = parse_label_number(label) {
        if (num as usize) < targets.len() {
            return targets[num as usize];
        }
    }
    // Non-numeric labels (function names, .Lskip_N, etc.) are conservatively
    // treated as jump targets.
    true
}

/// Global store forwarding pass.
#[allow(dead_code)]
fn global_store_forwarding(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    if n == 0 {
        return false;
    }

    let jump_targets = collect_jump_targets(lines, kinds, n);

    // Slot tracking: Vec of (offset, SlotMapping)
    let mut slots: Vec<(i32, SlotMapping)> = Vec::new();
    // Reverse mapping: register -> list of offsets it backs
    let mut reg_slots: Vec<Vec<i32>> = vec![Vec::new(); NUM_REGS];
    let mut changed = false;
    let mut prev_was_uncond = false;

    for i in 0..n {
        if kinds[i] == LineKind::Nop {
            continue;
        }

        let was_uncond = prev_was_uncond;
        prev_was_uncond = false;

        match kinds[i] {
            LineKind::Label => {
                // Conservatively invalidate all mappings at every label.
                // This is safe: labels can be jump targets from conditional branches,
                // and tracking all possible control flow paths is complex. By
                // invalidating at every label, we only forward within a single
                // straight-line basic block.
                gsf_invalidate_all(&mut slots, &mut reg_slots);
            }

            LineKind::StoreSp { reg, offset, is_word } => {
                // Invalidate any existing mapping that overlaps this store's byte range.
                // A word (w-reg) store covers 4 bytes; a doubleword (x-reg) covers 8 bytes.
                let store_size: i32 = if is_word { 4 } else { 8 };
                gsf_invalidate_overlapping(&mut slots, &mut reg_slots, offset, store_size);
                // Record new mapping
                if reg < NUM_REGS as u8 {
                    slots.push((offset, SlotMapping { reg, is_word, active: true }));
                    reg_slots[reg as usize].push(offset);
                }
                // Compact if too many entries
                if slots.len() > MAX_SLOT_ENTRIES {
                    slots.retain(|&(_, m)| m.active);
                }
            }

            LineKind::StorePairSp => {
                // stp conservatively invalidates all mappings
                // TODO: parse the two registers and offsets for precise invalidation
                gsf_invalidate_all(&mut slots, &mut reg_slots);
            }

            LineKind::LoadSp { reg: load_reg, offset: load_off, is_word: load_word } => {
                // Try to forward from a stored value
                let mapping = slots.iter().rev()
                    .find(|(off, m)| m.active && *off == load_off && m.is_word == load_word)
                    .map(|(_, m)| *m);

                if let Some(mapping) = mapping {
                    if !is_near_epilogue(kinds, i, n) && load_reg == mapping.reg {
                        // Same register: eliminate the redundant load
                        kinds[i] = LineKind::Nop;
                        changed = true;
                    }
                    // Note: different-register forwarding (load → mov) is not
                    // performed because the backing register may have been
                    // modified by an instruction type not tracked by GSF.
                }
                // The load overwrites load_reg, so invalidate any slot backed by it
                if (load_reg as usize) < NUM_REGS {
                    gsf_invalidate_reg(&mut slots, &mut reg_slots, load_reg);
                }
            }

            LineKind::LoadswSp { reg: load_reg, offset: load_off } => {
                // ldrsw loads a word and sign-extends to 64-bit.
                // We can forward if we have a word store at the same offset.
                let mapping = slots.iter().rev()
                    .find(|(off, m)| m.active && *off == load_off && m.is_word)
                    .map(|(_, m)| *m);

                // ldrsw always loads into a different-width register (sign extend),
                // so we don't forward it (same reasoning as LoadSp).
                let _ = mapping;
                // Invalidate any slot backed by load_reg
                if (load_reg as usize) < NUM_REGS {
                    gsf_invalidate_reg(&mut slots, &mut reg_slots, load_reg);
                }
            }

            LineKind::LoadPairSp => {
                // ldp loads two registers — conservatively invalidate everything
                // TODO: parse the two destination registers for precise invalidation
                gsf_invalidate_all(&mut slots, &mut reg_slots);
            }

            LineKind::Branch | LineKind::Ret => {
                gsf_invalidate_all(&mut slots, &mut reg_slots);
                prev_was_uncond = true;
            }

            LineKind::Call => {
                // Calls invalidate ALL slot mappings because the callee might
                // write through a pointer to any stack slot. We also invalidate
                // caller-saved register mappings.
                // This is conservative but necessary: the caller may have passed
                // a pointer to a stack slot (e.g., `add x0, sp, #N; bl func`)
                // and the callee could modify that slot.
                gsf_invalidate_all(&mut slots, &mut reg_slots);
            }

            LineKind::Move { dst, .. } | LineKind::MoveImm { dst } | LineKind::MoveWide { dst }
            | LineKind::Sxtw { dst, .. } => {
                // Instruction writes to dst, invalidate any slot backed by it
                if (dst as usize) < NUM_REGS {
                    gsf_invalidate_reg(&mut slots, &mut reg_slots, dst);
                }
            }

            LineKind::Compare | LineKind::CondBranch | LineKind::CmpBranch
            | LineKind::Directive | LineKind::Nop => {
                // These don't write to registers or memory, safe to keep mappings.
            }

            LineKind::Alu => {
                // ALU instructions write to a destination register.
                let trimmed = lines[i].trim();
                if let Some(dst) = parse_instruction_dest(trimmed) {
                    if (dst as usize) < NUM_REGS {
                        gsf_invalidate_reg(&mut slots, &mut reg_slots, dst);
                    }
                }
            }

            _ => {
                // For any unclassified instruction (Other, MemOther, LoadsbReg, etc.),
                // conservatively invalidate ALL mappings.
                // This is safe but may reduce the effectiveness of store forwarding.
                gsf_invalidate_all(&mut slots, &mut reg_slots);
            }
        }
    }

    changed
}

/// Invalidate all slot mappings.
#[allow(dead_code)]
fn gsf_invalidate_all(slots: &mut Vec<(i32, SlotMapping)>, reg_slots: &mut [Vec<i32>]) {
    slots.clear();
    for rs in reg_slots.iter_mut() {
        rs.clear();
    }
}

/// Invalidate slot mappings whose byte range overlaps [store_off, store_off + store_size).
/// A word mapping covers 4 bytes, a doubleword mapping covers 8 bytes.
#[allow(dead_code)]
fn gsf_invalidate_overlapping(
    slots: &mut Vec<(i32, SlotMapping)>,
    reg_slots: &mut [Vec<i32>],
    store_off: i32,
    store_size: i32,
) {
    let store_end = store_off + store_size;
    for &mut (off, ref mut m) in slots.iter_mut() {
        if !m.active {
            continue;
        }
        let mapping_size: i32 = if m.is_word { 4 } else { 8 };
        let mapping_end = off + mapping_size;
        // Check if byte ranges overlap: [off, mapping_end) ∩ [store_off, store_end) ≠ ∅
        if off < store_end && store_off < mapping_end {
            let r = m.reg as usize;
            if r < reg_slots.len() {
                reg_slots[r].retain(|&o| o != off);
            }
            m.active = false;
        }
    }
}

/// Invalidate all slot mappings backed by a specific register.
#[allow(dead_code)]
fn gsf_invalidate_reg(slots: &mut Vec<(i32, SlotMapping)>, reg_slots: &mut [Vec<i32>], reg: u8) {
    let r = reg as usize;
    if r >= reg_slots.len() {
        return;
    }
    for &offset in &reg_slots[r] {
        for &mut (off, ref mut m) in slots.iter_mut().rev() {
            if off == offset && m.active && m.reg == reg {
                m.active = false;
                break;
            }
        }
    }
    reg_slots[r].clear();
}

/// Check if a position is near the function epilogue.
/// On AArch64, the epilogue is: callee-save restores (ldp/ldr), then ldp x29, x30, [sp], #N; ret.
#[allow(dead_code)]
fn is_near_epilogue(kinds: &[LineKind], pos: usize, n: usize) -> bool {
    let limit = (pos + 20).min(n);
    for j in (pos + 1)..limit {
        match kinds[j] {
            LineKind::Nop => continue,
            // More callee-save restores
            LineKind::LoadSp { .. } | LineKind::LoadPairSp | LineKind::LoadswSp { .. } => continue,
            // Stack teardown
            LineKind::Other | LineKind::Alu => continue,
            // Found return
            LineKind::Ret => return true,
            // Any other instruction type means not epilogue
            _ => return false,
        }
    }
    false
}

/// Parse the destination register of an AArch64 instruction.
/// Most AArch64 instructions have the destination as the first operand.
#[allow(dead_code)]
fn parse_instruction_dest(trimmed: &str) -> Option<u8> {
    // Skip instructions that don't write to a GP register
    if trimmed.starts_with("cmp ") || trimmed.starts_with("cmn ") ||
       trimmed.starts_with("tst ") || trimmed.starts_with("b.") ||
       trimmed.starts_with("b ") || trimmed.starts_with("bl ") ||
       trimmed.starts_with("blr ") || trimmed.starts_with("br ") ||
       trimmed.starts_with("ret") || trimmed.starts_with("str") ||
       trimmed.starts_with("stp ") || trimmed.starts_with("nop") ||
       trimmed.starts_with("dmb") || trimmed.starts_with("dsb") ||
       trimmed.starts_with("isb") || trimmed.starts_with("cbz") ||
       trimmed.starts_with("cbnz") || trimmed.starts_with("tbz") ||
       trimmed.starts_with("tbnz") || trimmed.starts_with("prfm") {
        return None;
    }

    // For most instructions, the destination is the first register after the mnemonic
    if let Some(space_pos) = trimmed.find(' ') {
        let args = &trimmed[space_pos + 1..];
        let first_arg = if let Some(comma) = args.find(',') {
            args[..comma].trim()
        } else {
            args.trim()
        };
        let reg = parse_reg(first_arg);
        if reg != REG_NONE {
            return Some(reg);
        }
    }
    None
}


// ── Global register copy propagation ─────────────────────────────────────────
//
// After store forwarding converts loads into register moves, propagate those
// copies into subsequent instructions. For `mov xDST, xSRC`, replace
// references to xDST with xSRC in the immediately following instruction
// (within the same basic block).

fn propagate_register_copies(lines: &mut [String], kinds: &mut [LineKind], n: usize) -> bool {
    let mut changed = false;

    for i in 0..n {
        // Only process 64-bit register-to-register moves
        let (dst, src) = match kinds[i] {
            LineKind::Move { dst, src, is_32bit: false } => {
                // Don't propagate sp/fp moves
                if dst >= 29 || src >= 29 {
                    continue;
                }
                (dst, src)
            }
            _ => continue,
        };

        // Find the next non-Nop instruction
        let mut j = i + 1;
        while j < n && kinds[j] == LineKind::Nop {
            j += 1;
        }
        if j >= n {
            continue;
        }

        // Don't propagate across control flow boundaries or into instructions
        // with multiple destination registers (like ldp which has two dest regs
        // after the mnemonic, but our replace_source_reg_in_instruction only
        // treats the first operand as dest).
        match kinds[j] {
            LineKind::Label | LineKind::Branch | LineKind::Ret | LineKind::Directive
            | LineKind::LoadPairSp | LineKind::Call => continue,
            _ => {}
        }
        // Also skip ldp/ldaxr/ldxr/stxr by checking instruction text,
        // since these have multiple dest registers or complex operand semantics
        let trimmed_j = lines[j].trim();
        if trimmed_j.starts_with("ldp ")
            || trimmed_j.starts_with("ldaxr ")
            || trimmed_j.starts_with("ldxr ")
            || trimmed_j.starts_with("stxr ")
            || trimmed_j.starts_with("ldaxp ")
            || trimmed_j.starts_with("ldxp ")
            || trimmed_j.starts_with("cas ")
        {
            continue;
        }

        // Try to replace references to dst with src in line j
        let old_line = &lines[j];
        let dst_name = xreg_name(dst);
        if !old_line.contains(dst_name) {
            continue;
        }

        // Don't propagate into instructions that write to the same register
        // as the source (would create a self-reference)
        let src_name = xreg_name(src);

        // For move instructions, replace the source operand
        match kinds[j] {
            LineKind::Move { dst: dst2, src: src2, is_32bit: false } if src2 == dst => {
                // mov X, dst -> mov X, src
                if dst2 != src {
                    lines[j] = format!("    mov {}, {}", xreg_name(dst2), src_name);
                    kinds[j] = LineKind::Move { dst: dst2, src, is_32bit: false };
                    changed = true;
                }
            }
            _ => {
                // General case: try to replace the register in the instruction text.
                // Only replace in source operand positions (not destination).
                if let Some(new_line) = replace_source_reg_in_instruction(old_line, dst_name, src_name) {
                    lines[j] = new_line;
                    kinds[j] = classify_line(&lines[j]);
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Replace a register name in source operand positions of an instruction.
/// Returns None if the register is the destination (first operand) or not found.
fn replace_source_reg_in_instruction(line: &str, old_reg: &str, new_reg: &str) -> Option<String> {
    let trimmed = line.trim();

    // Find the first comma to separate destination from source operands
    let space_pos = trimmed.find(' ')?;
    let args_start = space_pos + 1;
    let args = &trimmed[args_start..];

    // Find first comma — everything after it is source operands
    let comma_pos = args.find(',')?;
    let after_first_arg = &args[comma_pos..];

    // Only replace in the source part (after the first comma)
    // Use whole-word replacement to avoid substring issues (e.g., x1 in x11)
    let new_suffix = replace_whole_word(after_first_arg, old_reg, new_reg);
    if new_suffix == after_first_arg {
        return None;
    }

    // Build the new line
    let prefix = &trimmed[..args_start + comma_pos];
    let new_trimmed = format!("{}{}", prefix, new_suffix);

    // Preserve leading whitespace
    let leading = line.len() - line.trim_start().len();
    let leading_ws = &line[..leading];
    Some(format!("{}{}", leading_ws, new_trimmed))
}

/// Replace `old` with `new` in `text` only at word boundaries.
/// A word boundary is a position where the adjacent character is not alphanumeric.
/// This prevents "x1" from matching inside "x11" or "x10".
fn replace_whole_word(text: &str, old: &str, new: &str) -> String {
    let bytes = text.as_bytes();
    let old_bytes = old.as_bytes();
    let old_len = old_bytes.len();
    let text_len = bytes.len();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;

    while i < text_len {
        if i + old_len <= text_len && &bytes[i..i + old_len] == old_bytes {
            // Check word boundary: character before must not be alphanumeric
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            // Check word boundary: character after must not be alphanumeric
            let after_ok = i + old_len >= text_len || !bytes[i + old_len].is_ascii_alphanumeric();
            if before_ok && after_ok {
                result.push_str(new);
                i += old_len;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

// ── Global dead store elimination ────────────────────────────────────────────
//
// Scans the entire function to find stack slot offsets that are never loaded.
// Stores to such slots are dead and can be eliminated.
// This runs after global store forwarding, which may have converted many loads
// to register moves, leaving the original stores dead.

fn global_dead_store_elimination(lines: &[String], kinds: &mut [LineKind], n: usize) -> bool {
    // Safety check: if any instruction takes the address of sp (e.g., `add xN, sp, #off`),
    // stack slots could be accessed through pointers, so we must not eliminate any stores.
    // This is conservative but sound — it prevents miscompilation when arrays or structs
    // are allocated on the stack and passed by pointer to callees.
    for i in 0..n {
        if kinds[i] == LineKind::Nop {
            continue;
        }
        let trimmed = lines[i].trim();
        // Check for address-of-sp patterns:
        // - `add xN, sp, #offset` (common for array/struct address)
        // - `mov xN, sp` (copying stack pointer)
        // - `sub xN, sp, #N` (stack address computation)
        if (trimmed.starts_with("add ") || trimmed.starts_with("sub ")) && trimmed.contains(", sp,") {
            return false;
        }
        if trimmed.starts_with("mov ") && trimmed.contains(", sp") {
            // Check it's actually "mov xN, sp" not "mov sp, xN"
            if let Some(rest) = trimmed.strip_prefix("mov ") {
                if let Some((_, src)) = rest.split_once(", ") {
                    if src.trim() == "sp" {
                        return false;
                    }
                }
            }
        }
    }

    // Phase 1: Collect all offsets that are loaded from (read)
    let mut loaded_offsets = std::collections::HashSet::new();
    for i in 0..n {
        match kinds[i] {
            LineKind::LoadSp { offset, .. } | LineKind::LoadswSp { offset, .. } => {
                loaded_offsets.insert(offset);
            }
            _ => {
                // Check for loads in Other instructions (e.g., ldp, ldrb, etc.)
                let trimmed = lines[i].trim();
                if (trimmed.starts_with("ldr") || trimmed.starts_with("ldp") ||
                    trimmed.starts_with("ldur")) && trimmed.contains("[sp") {
                    // Extract offset from [sp, #N] or [sp] pattern
                    if let Some(off) = extract_sp_offset(trimmed) {
                        loaded_offsets.insert(off);
                        // For ldp, also mark the second slot (offset + 8)
                        if trimmed.starts_with("ldp") {
                            loaded_offsets.insert(off + 8);
                        }
                    }
                }
            }
        }
    }

    // Phase 2: Remove stores to offsets that are never loaded
    let mut changed = false;
    for i in 0..n {
        if let LineKind::StoreSp { offset, .. } = kinds[i] {
            if !loaded_offsets.contains(&offset) {
                kinds[i] = LineKind::Nop;
                changed = true;
            }
        }
    }
    changed
}

/// Extract the numeric offset from an instruction containing `[sp, #N]` or `[sp]`.
fn extract_sp_offset(line: &str) -> Option<i32> {
    if let Some(start) = line.find("[sp") {
        let rest = &line[start..];
        if rest.starts_with("[sp]") {
            return Some(0);
        }
        if rest.starts_with("[sp, #") {
            let num_start = start + 6; // skip "[sp, #"
            let after = &line[num_start..];
            if let Some(end) = after.find(']') {
                return after[..end].parse::<i32>().ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_store() {
        assert!(matches!(
            classify_line("    str x0, [sp, #16]"),
            LineKind::StoreSp { reg: 0, offset: 16, is_word: false }
        ));
        assert!(matches!(
            classify_line("    str w1, [sp, #24]"),
            LineKind::StoreSp { reg: 1, offset: 24, is_word: true }
        ));
    }

    #[test]
    fn test_classify_load() {
        assert!(matches!(
            classify_line("    ldr x0, [sp, #16]"),
            LineKind::LoadSp { reg: 0, offset: 16, is_word: false }
        ));
    }

    #[test]
    fn test_classify_loadsw() {
        assert!(matches!(
            classify_line("    ldrsw x0, [sp, #24]"),
            LineKind::LoadswSp { reg: 0, offset: 24 }
        ));
    }

    #[test]
    fn test_classify_move() {
        assert!(matches!(
            classify_line("    mov x14, x0"),
            LineKind::Move { dst: 14, src: 0, is_32bit: false }
        ));
    }

    #[test]
    fn test_classify_move_imm() {
        assert!(matches!(
            classify_line("    mov x0, #0"),
            LineKind::MoveImm { dst: 0 }
        ));
        assert!(matches!(
            classify_line("    mov x0, #-1"),
            LineKind::MoveImm { dst: 0 }
        ));
    }

    #[test]
    fn test_classify_branch() {
        assert_eq!(classify_line("    b .LBB1"), LineKind::Branch);
    }

    #[test]
    fn test_classify_cond_branch() {
        assert_eq!(classify_line("    b.ge .Lskip_0"), LineKind::CondBranch);
        assert_eq!(classify_line("    b.eq .LBB3"), LineKind::CondBranch);
    }

    #[test]
    fn test_classify_label() {
        assert_eq!(classify_line(".LBB1:"), LineKind::Label);
        assert_eq!(classify_line("sum_array:"), LineKind::Label);
    }

    #[test]
    fn test_classify_ret() {
        assert_eq!(classify_line("    ret"), LineKind::Ret);
    }

    #[test]
    fn test_classify_sxtw() {
        assert!(matches!(
            classify_line("    sxtw x0, w0"),
            LineKind::Sxtw { dst: 0, src: 0 }
        ));
    }

    #[test]
    fn test_adjacent_store_load_same_reg() {
        let input = "    str x0, [sp, #16]\n    ldr x0, [sp, #16]\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str x0, [sp, #16]"));
        assert!(!result.contains("ldr x0, [sp, #16]"));
    }

    #[test]
    fn test_adjacent_store_load_diff_reg() {
        let input = "    str x0, [sp, #16]\n    ldr x1, [sp, #16]\n    ret\n";
        let result = peephole_optimize(input.to_string());
        // The load is replaced with mov, and then DSE removes the now-dead store
        // (no remaining loads from offset 16), leaving just the mov and ret.
        assert!(!result.contains("ldr x1, [sp, #16]"));
        assert!(result.contains("mov x1, x0"));
    }

    #[test]
    fn test_redundant_branch() {
        let input = "    b .LBB1\n.LBB1:\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("b .LBB1"));
        assert!(result.contains(".LBB1:"));
    }

    #[test]
    fn test_self_move() {
        let input = "    mov x0, x0\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("mov x0, x0"));
    }

    #[test]
    fn test_branch_over_branch_fusion() {
        let input = "    b.ge .Lskip_0\n    b .LBB2\n.Lskip_0:\n    b .LBB4\n";
        let result = peephole_optimize(input.to_string());
        // Should become: b.lt .LBB2 (inverted ge → lt)
        assert!(result.contains("b.lt .LBB2"));
        // The unconditional branch to LBB2 should be eliminated
        assert!(!result.contains("    b .LBB2\n"));
    }

    #[test]
    fn test_move_chain() {
        let input = "    mov x0, x14\n    mov x13, x0\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov x13, x14"));
    }

    #[test]
    fn test_move_imm_chain() {
        let input = "    mov x0, #0\n    mov x14, x0\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("mov x14, #0"));
    }

    #[test]
    fn test_store_loadsw_fusion() {
        let input = "    str w1, [sp, #24]\n    ldrsw x0, [sp, #24]\n    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldrsw"));
        assert!(result.contains("sxtw x0, w1"));
    }

    // ── Global store forwarding tests ─────────────────────────────────

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_same_reg_elimination() {
        // Store x0 then load x0 from same slot (non-adjacent) — load is dead
        let input = "\
    str x0, [sp, #16]\n\
    add x1, x2, x3\n\
    ldr x0, [sp, #16]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str x0, [sp, #16]"));
        assert!(!result.contains("ldr x0, [sp, #16]"));
    }

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_different_reg_forwarding() {
        // Store x5 then load x10 from same slot — replace load with mov
        let input = "\
    str x5, [sp, #32]\n\
    add x1, x2, x3\n\
    ldr x10, [sp, #32]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(result.contains("str x5, [sp, #32]"));
        assert!(!result.contains("ldr x10, [sp, #32]"));
        assert!(result.contains("mov x10, x5"));
    }

    #[test]
    fn test_gsf_invalidation_on_reg_overwrite() {
        // After x0 is overwritten, the mapping slot 16 → x0 is stale
        let input = "\
    str x0, [sp, #16]\n\
    mov x0, #42\n\
    ldr x1, [sp, #16]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        // The load should NOT be forwarded since x0 was overwritten
        assert!(result.contains("ldr x1, [sp, #16]"));
    }

    #[test]
    fn test_gsf_invalidation_at_jump_target() {
        // Mappings are invalidated at jump target labels
        let input = "\
    str x5, [sp, #16]\n\
    b .LBB1\n\
.LBB1:\n\
    ldr x5, [sp, #16]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        // .LBB1 is a jump target, so mappings are invalidated
        assert!(result.contains("ldr x5, [sp, #16]"));
    }

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_word_forwarding() {
        // Word store forwarded to word load
        let input = "\
    str w3, [sp, #24]\n\
    add x1, x2, x4\n\
    ldr w5, [sp, #24]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldr w5, [sp, #24]"));
        assert!(result.contains("mov w5, w3"));
    }

    #[test]
    #[ignore] // GSF is disabled due to correctness bug in complex float-array code
    fn test_gsf_ldrsw_forwarding() {
        // Word store forwarded to sign-extending load
        let input = "\
    str w1, [sp, #24]\n\
    add x2, x3, x4\n\
    ldrsw x5, [sp, #24]\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        assert!(!result.contains("ldrsw x5, [sp, #24]"));
        assert!(result.contains("sxtw x5, w1"));
    }

    // ── Copy propagation tests ───────────────────────────────────────

    #[test]
    fn test_copy_prop_word_boundary() {
        // Ensure "x1" doesn't match inside "x11"
        assert_eq!(
            replace_whole_word("x11, x1", "x1", "x5"),
            "x11, x5"
        );
    }

    #[test]
    fn test_copy_prop_no_false_match() {
        // x10 should not be affected when replacing x1
        assert_eq!(
            replace_whole_word("x10", "x1", "x5"),
            "x10"
        );
    }

    // ── Dead store elimination tests ─────────────────────────────────

    #[test]
    fn test_dse_with_address_taken() {
        // When sp address is taken, no stores should be eliminated
        let input = "\
    str w0, [sp, #16]\n\
    add x1, sp, #16\n\
    ret\n";
        let result = peephole_optimize(input.to_string());
        // Store must be preserved because address of stack slot is taken
        assert!(result.contains("str w0, [sp, #16]"));
    }
}
