//! ARMv7 assembly parser.
//!
//! Parses GNU ARM assembly syntax into structured statements.

use crate::backend::asm_preprocess;

/// A parsed assembly operand.
#[derive(Debug, Clone)]
pub enum Operand {
    Reg(String),
    Imm(i64),
    Symbol(String),
    Label(String),
    SymbolOffset(String, i64),
    /// Memory reference: [base] or [base, #offset] or [base, reg]
    /// writeback: true for pre-indexed with writeback [base, #offset]!
    Mem { base: String, offset: MemOffset, writeback: bool },
    /// Register list: {r0, r1, r2}
    RegList(Vec<String>),
    /// Shift: lsl #n, etc.
    Shift(String, i64),
    /// Raw expression string
    Expr(String),
    /// :lower16:sym or :upper16:sym
    Reloc(String, String), // (reloc_type, symbol)
}

#[derive(Debug, Clone)]
pub enum MemOffset {
    None,
    Imm(i64),
    Reg(String),
    RegShift(String, String, i64), // reg, shift_type, amount
}

/// Assembly data value.
#[derive(Debug, Clone)]
pub enum DataValue {
    Integer(i64),
    Symbol(String),
    SymbolOffset(String, i64),
    SymbolDiff(String, String),
    SymbolDiffAddend(String, String, i64),
    Expr(String),
}

/// Assembly directive.
#[derive(Debug, Clone)]
pub enum AsmDirective {
    Section(String),
    PushSection(String),
    PopSection,
    Previous,
    Text,
    Data,
    Bss,
    Global(String),
    Local(String),
    Hidden(String),
    Protected(String),
    Weak(String),
    Type(String, String),
    Size(String, String),
    Byte(Vec<DataValue>),
    Short(Vec<DataValue>),
    Word(Vec<DataValue>),
    Long(Vec<DataValue>),
    Quad(Vec<DataValue>),
    Ascii(Vec<Vec<u8>>),
    Asciz(Vec<Vec<u8>>),
    String(Vec<Vec<u8>>),
    Zero(usize),
    Space(usize),
    Fill(usize, usize, u8),
    Align(usize),
    Balign(usize),
    P2Align(usize),
    Comm(String, usize, usize),
    Set(String, String),
    Equiv(String, String),
    File(String),
    Ident(String),
    Loc(Vec<String>),
    CfiStartproc,
    CfiEndproc,
    CfiDefCfa(String, i64),
    CfiDefCfaOffset(i64),
    CfiDefCfaRegister(String),
    CfiOffset(String, i64),
    CfiRestore(String),
    CfiRememberState,
    CfiRestoreState,
    CfiSections(String),
    Ltorg,
    Pool,
    Syntax(String),
    Arch(String),
    Fpu(String),
    EabiAttribute(Vec<String>),
    Code(u32),
    Thumb,
    ThumbFunc,
    Fnstart,
    Fnend,
    Cantunwind,
    Handlerdata,
    Personality(String),
    PersonalityIndex(u32),
    Pad(u32),
    Save(Vec<String>),
    Vsave(Vec<String>),
    Setfp(String, String, Option<i64>),
    /// ARM-specific mapping symbols
    ArmMapping,
    ThumbMapping,
    DataMapping,
    /// .inst directive (raw instruction word)
    Inst(u32),
    /// Patchable function entry
    PatchableFunctionEntry(u32),
}

/// A parsed assembly statement.
#[derive(Debug, Clone)]
pub enum AsmStatement {
    Label(String),
    Instruction {
        mnemonic: String,
        operands: Vec<Operand>,
        raw_operands: String,
    },
    Directive(AsmDirective),
    /// Empty line or comment
    Empty,
}

/// Parse ARM assembly text into statements.
pub fn parse_asm(text: &str) -> Result<Vec<AsmStatement>, String> {
    let preprocessed = asm_preprocess::strip_c_comments(text);
    let mut statements = Vec::new();

    for line in preprocessed.lines() {
        let line = strip_comment(line);
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        // Check for label
        if let Some(pos) = trimmed.find(':') {
            let potential_label = &trimmed[..pos];
            if is_valid_label(potential_label) {
                statements.push(AsmStatement::Label(potential_label.to_string()));
                let rest = trimmed[pos + 1..].trim();
                if !rest.is_empty() {
                    if let Some(stmt) = parse_line(rest)? {
                        statements.push(stmt);
                    }
                }
                continue;
            }
        }

        if let Some(stmt) = parse_line(trimmed)? {
            statements.push(stmt);
        }
    }

    Ok(statements)
}

fn strip_comment(line: &str) -> &str {
    // ARM assembly uses @ for line comments, // and /* */ also supported
    if let Some(pos) = line.find('@') {
        &line[..pos]
    } else if let Some(pos) = line.find("//") {
        &line[..pos]
    } else {
        line
    }
}

fn is_valid_label(s: &str) -> bool {
    if s.is_empty() { return false; }
    let first = s.as_bytes()[0];
    if first.is_ascii_digit() {
        // Numeric labels
        return s.bytes().all(|b| b.is_ascii_digit());
    }
    (first == b'.' || first == b'_' || first.is_ascii_alphabetic())
        && s.bytes().all(|b| b == b'.' || b == b'_' || b == b'$' || b.is_ascii_alphanumeric())
}

fn parse_line(line: &str) -> Result<Option<AsmStatement>, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() { return Ok(None); }

    if trimmed.starts_with('.') {
        // Directive
        return parse_directive(trimmed).map(|d| Some(AsmStatement::Directive(d)));
    }

    // Instruction
    parse_instruction(trimmed).map(Some)
}

fn parse_directive(line: &str) -> Result<AsmDirective, String> {
    let (name, args) = split_directive(line);

    match name {
        ".section" => Ok(AsmDirective::Section(args.to_string())),
        ".pushsection" => Ok(AsmDirective::PushSection(args.to_string())),
        ".popsection" => Ok(AsmDirective::PopSection),
        ".previous" => Ok(AsmDirective::Previous),
        ".text" => Ok(AsmDirective::Text),
        ".data" => Ok(AsmDirective::Data),
        ".bss" => Ok(AsmDirective::Bss),
        ".globl" | ".global" => Ok(AsmDirective::Global(args.trim().to_string())),
        ".local" => Ok(AsmDirective::Local(args.trim().to_string())),
        ".hidden" => Ok(AsmDirective::Hidden(args.trim().to_string())),
        ".protected" => Ok(AsmDirective::Protected(args.trim().to_string())),
        ".weak" => Ok(AsmDirective::Weak(args.trim().to_string())),
        ".type" => {
            let parts: Vec<&str> = args.splitn(2, ',').collect();
            if parts.len() == 2 {
                Ok(AsmDirective::Type(parts[0].trim().to_string(), parts[1].trim().to_string()))
            } else {
                Ok(AsmDirective::Type(args.trim().to_string(), String::new()))
            }
        }
        ".size" => {
            let parts: Vec<&str> = args.splitn(2, ',').collect();
            if parts.len() == 2 {
                Ok(AsmDirective::Size(parts[0].trim().to_string(), parts[1].trim().to_string()))
            } else {
                Ok(AsmDirective::Size(args.trim().to_string(), String::new()))
            }
        }
        ".byte" => Ok(AsmDirective::Byte(parse_data_values(args))),
        ".short" | ".hword" | ".2byte" => Ok(AsmDirective::Short(parse_data_values(args))),
        ".word" | ".4byte" => Ok(AsmDirective::Word(parse_data_values(args))),
        ".long" => Ok(AsmDirective::Long(parse_data_values(args))),
        ".quad" | ".8byte" | ".xword" => Ok(AsmDirective::Quad(parse_data_values(args))),
        ".ascii" => Ok(AsmDirective::Ascii(parse_strings(args))),
        ".asciz" | ".string" => Ok(AsmDirective::Asciz(parse_strings(args))),
        ".zero" => Ok(AsmDirective::Zero(parse_usize(args))),
        ".space" | ".skip" => Ok(AsmDirective::Space(parse_usize(args))),
        ".fill" => {
            let parts: Vec<&str> = args.split(',').collect();
            let count = parse_usize(parts[0]);
            let size = if parts.len() > 1 { parse_usize(parts[1]) } else { 1 };
            let val = if parts.len() > 2 { parse_usize(parts[2]) as u8 } else { 0 };
            Ok(AsmDirective::Fill(count, size, val))
        }
        ".align" => Ok(AsmDirective::Align(parse_usize(args))),
        ".balign" => Ok(AsmDirective::Balign(parse_usize(args))),
        ".p2align" => Ok(AsmDirective::P2Align(parse_usize(args))),
        ".comm" | ".common" => {
            let parts: Vec<&str> = args.split(',').collect();
            let name = parts[0].trim().to_string();
            let size = if parts.len() > 1 { parse_usize(parts[1]) } else { 0 };
            let align = if parts.len() > 2 { parse_usize(parts[2]) } else { 4 };
            Ok(AsmDirective::Comm(name, size, align))
        }
        ".set" | ".equ" => {
            let parts: Vec<&str> = args.splitn(2, ',').collect();
            if parts.len() == 2 {
                Ok(AsmDirective::Set(parts[0].trim().to_string(), parts[1].trim().to_string()))
            } else {
                Ok(AsmDirective::Set(args.trim().to_string(), "0".to_string()))
            }
        }
        ".file" => Ok(AsmDirective::File(args.trim().to_string())),
        ".ident" => Ok(AsmDirective::Ident(args.trim().to_string())),
        ".loc" => Ok(AsmDirective::Loc(args.split_whitespace().map(|s| s.to_string()).collect())),
        ".cfi_startproc" => Ok(AsmDirective::CfiStartproc),
        ".cfi_endproc" => Ok(AsmDirective::CfiEndproc),
        ".cfi_def_cfa" => {
            let parts: Vec<&str> = args.split(',').collect();
            let reg = parts[0].trim().to_string();
            let offset = if parts.len() > 1 { parse_i64(parts[1]) } else { 0 };
            Ok(AsmDirective::CfiDefCfa(reg, offset))
        }
        ".cfi_def_cfa_offset" => Ok(AsmDirective::CfiDefCfaOffset(parse_i64(args))),
        ".cfi_def_cfa_register" => Ok(AsmDirective::CfiDefCfaRegister(args.trim().to_string())),
        ".cfi_offset" => {
            let parts: Vec<&str> = args.split(',').collect();
            let reg = parts[0].trim().to_string();
            let offset = if parts.len() > 1 { parse_i64(parts[1]) } else { 0 };
            Ok(AsmDirective::CfiOffset(reg, offset))
        }
        ".cfi_restore" => Ok(AsmDirective::CfiRestore(args.trim().to_string())),
        ".cfi_remember_state" => Ok(AsmDirective::CfiRememberState),
        ".cfi_restore_state" => Ok(AsmDirective::CfiRestoreState),
        ".cfi_sections" => Ok(AsmDirective::CfiSections(args.trim().to_string())),
        ".ltorg" | ".pool" => Ok(AsmDirective::Ltorg),
        ".syntax" => Ok(AsmDirective::Syntax(args.trim().to_string())),
        ".arch" => Ok(AsmDirective::Arch(args.trim().to_string())),
        ".fpu" => Ok(AsmDirective::Fpu(args.trim().to_string())),
        ".eabi_attribute" => Ok(AsmDirective::EabiAttribute(args.split(',').map(|s| s.trim().to_string()).collect())),
        ".code" => Ok(AsmDirective::Code(parse_usize(args) as u32)),
        ".thumb" => Ok(AsmDirective::Thumb),
        ".thumb_func" => Ok(AsmDirective::ThumbFunc),
        ".fnstart" => Ok(AsmDirective::Fnstart),
        ".fnend" => Ok(AsmDirective::Fnend),
        ".cantunwind" => Ok(AsmDirective::Cantunwind),
        ".handlerdata" => Ok(AsmDirective::Handlerdata),
        ".inst" | ".inst.w" => {
            let val = parse_u32(args);
            Ok(AsmDirective::Inst(val))
        }
        ".patchable_function_entry" => Ok(AsmDirective::PatchableFunctionEntry(parse_usize(args) as u32)),
        _ => {
            // Unknown directive, treat as comment
            Ok(AsmDirective::Ident(format!("unknown: {} {}", name, args)))
        }
    }
}

fn split_directive(line: &str) -> (&str, &str) {
    let trimmed = line.trim();
    if let Some(pos) = trimmed.find(|c: char| c.is_whitespace() || c == '\t') {
        (&trimmed[..pos], trimmed[pos..].trim())
    } else {
        (trimmed, "")
    }
}

fn parse_instruction(line: &str) -> Result<AsmStatement, String> {
    let trimmed = line.trim();
    let (mnemonic, rest) = if let Some(pos) = trimmed.find(|c: char| c.is_whitespace()) {
        (&trimmed[..pos], trimmed[pos..].trim())
    } else {
        (trimmed, "")
    };

    let operands = if rest.is_empty() {
        Vec::new()
    } else {
        parse_operands(rest)
    };

    Ok(AsmStatement::Instruction {
        mnemonic: mnemonic.to_lowercase(),
        operands,
        raw_operands: rest.to_string(),
    })
}

fn parse_operands(text: &str) -> Vec<Operand> {
    let mut operands = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, ch) in text.char_indices() {
        match ch {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                let part = text[start..i].trim();
                if !part.is_empty() {
                    operands.push(parse_single_operand(part));
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() {
        operands.push(parse_single_operand(last));
    }
    operands
}

fn parse_single_operand(text: &str) -> Operand {
    let trimmed = text.trim();

    // Register list: {r0, r1, ...}
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        let inner = &trimmed[1..trimmed.len()-1];
        let regs: Vec<String> = inner.split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        return Operand::RegList(regs);
    }

    // Memory reference: [base, ...]
    if trimmed.starts_with('[') {
        return parse_mem_operand(trimmed);
    }

    // Immediate: #value
    if let Some(rest) = trimmed.strip_prefix('#') {
        // Check for :lower16: or :upper16: relocations
        if let Some(sym) = rest.strip_prefix(":lower16:") {
            return Operand::Reloc("lower16".to_string(), sym.to_string());
        }
        if let Some(sym) = rest.strip_prefix(":upper16:") {
            return Operand::Reloc("upper16".to_string(), sym.to_string());
        }
        if let Ok(v) = parse_imm_value(rest) {
            return Operand::Imm(v);
        }
        return Operand::Expr(rest.to_string());
    }

    // Register
    let lower = trimmed.to_lowercase();
    if is_arm_register(&lower) {
        return Operand::Reg(lower);
    }

    // Symbol or label
    Operand::Symbol(trimmed.to_string())
}

fn parse_mem_operand(text: &str) -> Operand {
    let inner = text.trim_start_matches('[');
    let (inner, writeback) = if inner.ends_with("]!") {
        (&inner[..inner.len()-2], true)
    } else if inner.ends_with(']') {
        (&inner[..inner.len()-1], false)
    } else {
        (inner, false)
    };

    let parts: Vec<&str> = inner.splitn(2, ',').collect();
    let base = parts[0].trim().to_lowercase();

    let offset = if parts.len() > 1 {
        let off_str = parts[1].trim();
        if let Some(rest) = off_str.strip_prefix('#') {
            if let Ok(v) = parse_imm_value(rest) {
                MemOffset::Imm(v)
            } else {
                MemOffset::Imm(0)
            }
        } else if is_arm_register(&off_str.to_lowercase()) {
            MemOffset::Reg(off_str.to_lowercase())
        } else {
            MemOffset::Imm(0)
        }
    } else {
        MemOffset::None
    };

    Operand::Mem { base, offset, writeback }
}

fn is_arm_register(name: &str) -> bool {
    matches!(name,
        "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" |
        "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15" |
        "sp" | "lr" | "pc" | "ip" | "fp" |
        "s0" | "s1" | "s2" | "s3" | "s4" | "s5" | "s6" | "s7" |
        "s8" | "s9" | "s10" | "s11" | "s12" | "s13" | "s14" | "s15" |
        "s16" | "s17" | "s18" | "s19" | "s20" | "s21" | "s22" | "s23" |
        "s24" | "s25" | "s26" | "s27" | "s28" | "s29" | "s30" | "s31" |
        "d0" | "d1" | "d2" | "d3" | "d4" | "d5" | "d6" | "d7" |
        "d8" | "d9" | "d10" | "d11" | "d12" | "d13" | "d14" | "d15" |
        "apsr_nzcv" | "fpscr"
    )
}

fn parse_imm_value(s: &str) -> Result<i64, ()> {
    let trimmed = s.trim();
    if trimmed.is_empty() { return Err(()); }
    let (neg, s) = if trimmed.starts_with('-') { (true, &trimmed[1..]) } else { (false, trimmed) };
    let val = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).map_err(|_| ())?
    } else if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        i64::from_str_radix(bin, 2).map_err(|_| ())?
    } else {
        s.parse::<i64>().map_err(|_| ())?
    };
    Ok(if neg { -val } else { val })
}

fn parse_data_values(text: &str) -> Vec<DataValue> {
    text.split(',')
        .map(|s| {
            let t = s.trim();
            if let Ok(v) = parse_imm_value(t) {
                DataValue::Integer(v)
            } else if t.contains('-') && !t.starts_with('-') {
                let parts: Vec<&str> = t.splitn(2, '-').collect();
                DataValue::SymbolDiff(parts[0].trim().to_string(), parts[1].trim().to_string())
            } else if t.contains('+') {
                let parts: Vec<&str> = t.splitn(2, '+').collect();
                let sym = parts[0].trim().to_string();
                let off = parse_imm_value(parts[1]).unwrap_or(0);
                DataValue::SymbolOffset(sym, off)
            } else {
                DataValue::Symbol(t.to_string())
            }
        })
        .collect()
}

fn parse_strings(text: &str) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let trimmed = text.trim();
    if trimmed.starts_with('"') {
        let mut bytes = Vec::new();
        let mut chars = trimmed[1..].chars();
        while let Some(ch) = chars.next() {
            if ch == '"' { break; }
            if ch == '\\' {
                match chars.next() {
                    Some('n') => bytes.push(b'\n'),
                    Some('t') => bytes.push(b'\t'),
                    Some('r') => bytes.push(b'\r'),
                    Some('0') => bytes.push(0),
                    Some('\\') => bytes.push(b'\\'),
                    Some('"') => bytes.push(b'"'),
                    Some(c) => {
                        bytes.push(b'\\');
                        bytes.push(c as u8);
                    }
                    None => break,
                }
            } else {
                bytes.push(ch as u8);
            }
        }
        result.push(bytes);
    }
    result
}

fn parse_usize(text: &str) -> usize {
    parse_imm_value(text.trim()).unwrap_or(0) as usize
}

fn parse_i64(text: &str) -> i64 {
    parse_imm_value(text.trim()).unwrap_or(0)
}

fn parse_u32(text: &str) -> u32 {
    parse_imm_value(text.trim()).unwrap_or(0) as u32
}
