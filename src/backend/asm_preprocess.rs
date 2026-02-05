//! Shared GAS assembly preprocessing utilities.
//!
//! Functions used by multiple assembler parsers (ARM, RISC-V, x86) for
//! text-level preprocessing before architecture-specific parsing:
//!
//! - C-style comment stripping (`/* ... */`)
//! - Line comment stripping (`#`, `//`, `@`)
//! - Semicolon splitting (GAS statement separator)
//! - `.rept`/`.irp`/`.endr` block expansion
//! - `.macro`/`.endm` definition and expansion
//! - `.if`/`.elseif`/`.else`/`.endif` conditional assembly evaluation

use crate::backend::asm_expr;

// ── Comment handling ───────────────────────────────────────────────────

/// Characters that start a line comment for each architecture.
///
/// Used by `strip_comment` to determine where comments begin.
/// Each architecture may have multiple comment styles.
pub enum CommentStyle {
    /// `#` only (x86/x86-64 AT&T syntax)
    Hash,
    /// `#` and `//` (RISC-V GAS)
    HashAndSlashSlash,
    /// `//` and `@` (ARM GAS — but `@` is not a comment before type specifiers
    /// like `@function`, `@object`, `@progbits`, `@nobits`, `@tls_object`, `@note`)
    /// Currently the ARM assembler uses its own strip_comment; this variant will
    /// be used when ARM migrates to the shared preprocessor.
    #[allow(dead_code)]
    SlashSlashAndAt,
}

/// Strip C-style `/* ... */` comments from assembly text, handling multi-line spans.
/// Preserves newlines inside comments so line numbers remain correct for error messages.
/// String-aware: does not strip `/* */` inside quoted string literals (e.g. `.asciz` data).
pub fn strip_c_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    let mut escape = false;
    while i < bytes.len() {
        if in_string {
            if escape {
                escape = false;
                result.push(bytes[i] as char);
                i += 1;
                continue;
            }
            if bytes[i] == b'\\' {
                escape = true;
                result.push(bytes[i] as char);
                i += 1;
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
            // Newlines end the string context (unterminated string on this line)
            if bytes[i] == b'\n' {
                in_string = false;
            }
            result.push(bytes[i] as char);
            i += 1;
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                if bytes[i] == b'\n' {
                    result.push('\n');
                }
                i += 1;
            }
        } else {
            if bytes[i] == b'"' {
                in_string = true;
            }
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

/// Strip trailing line comment from a single line, respecting string literals.
///
/// Scans character by character tracking quote state so that comment characters
/// inside `"..."` strings are not treated as comment starts. Handles escaped
/// quotes (`\"`) correctly.
pub fn strip_comment<'a>(line: &'a str, style: &CommentStyle) -> &'a str {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                i += 2; // skip escaped character
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        // Not in string
        if bytes[i] == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        match style {
            CommentStyle::Hash => {
                if bytes[i] == b'#' {
                    return &line[..i];
                }
            }
            CommentStyle::HashAndSlashSlash => {
                if bytes[i] == b'#' {
                    return &line[..i];
                }
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    return &line[..i];
                }
            }
            CommentStyle::SlashSlashAndAt => {
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    return &line[..i];
                }
                if bytes[i] == b'@' {
                    // `@` is NOT a comment before type specifiers used in GAS directives
                    let after = &line[i + 1..];
                    if !after.starts_with("object")
                        && !after.starts_with("function")
                        && !after.starts_with("progbits")
                        && !after.starts_with("nobits")
                        && !after.starts_with("tls_object")
                        && !after.starts_with("note")
                    {
                        return &line[..i];
                    }
                }
            }
        }
        i += 1;
    }
    line
}

/// Split a line on `;` characters, respecting string literals and comments.
/// In GAS syntax, `;` separates multiple statements on the same line.
/// Stops splitting once a line comment (`#` or `//`) is encountered outside strings,
/// so semicolons inside comments are not treated as statement separators.
pub fn split_on_semicolons(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    let mut start = 0;
    let bytes = line.as_bytes();
    for (i, c) in line.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            // Stop splitting at # comment start
            if c == '#' {
                break;
            }
            // Stop splitting at // comment start
            if c == '/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                break;
            }
            if c == ';' {
                parts.push(&line[start..i]);
                start = i + 1;
            }
        }
    }
    parts.push(&line[start..]);
    parts
}

// ── .rept / .irp block expansion ───────────────────────────────────────

fn is_rept_start(trimmed: &str) -> bool {
    trimmed.starts_with(".rept ") || trimmed.starts_with(".rept\t")
}

fn is_irp_start(trimmed: &str) -> bool {
    trimmed.starts_with(".irp ") || trimmed.starts_with(".irp\t")
}

fn is_block_start(trimmed: &str) -> bool {
    is_rept_start(trimmed) || is_irp_start(trimmed)
}

/// Collect the body lines of a `.rept`/`.irp` block, handling nesting.
/// Returns the body lines and advances `i` past the closing `.endr`.
fn collect_block_body<'a>(
    lines: &[&'a str],
    i: &mut usize,
    comment_style: &CommentStyle,
) -> Result<Vec<&'a str>, String> {
    let mut depth = 1;
    let mut body = Vec::new();
    *i += 1;
    while *i < lines.len() {
        let inner = strip_comment(lines[*i], comment_style).trim().to_string();
        if is_block_start(&inner) {
            depth += 1;
        } else if inner == ".endr" {
            depth -= 1;
            if depth == 0 {
                break;
            }
        }
        body.push(lines[*i]);
        *i += 1;
    }
    if depth != 0 {
        return Err(".rept/.irp without matching .endr".to_string());
    }
    Ok(body)
}

/// Expand `.rept`/`.endr` and `.irp`/`.endr` blocks by repeating or
/// substituting contained lines.
///
/// Handles nested blocks and recursive expansion. Uses `parse_int` to
/// evaluate the `.rept` count expression.
pub fn expand_rept_blocks(
    lines: &[&str],
    comment_style: &CommentStyle,
    parse_int: fn(&str) -> Result<i64, String>,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = strip_comment(lines[i], comment_style).trim().to_string();
        if is_rept_start(&trimmed) {
            let count_str = trimmed[".rept".len()..].trim();
            let count_val = parse_int(count_str)
                .map_err(|e| format!(".rept: bad count '{}': {}", count_str, e))?;
            // Treat negative counts as 0 (matches GNU as behavior)
            let count = if count_val < 0 { 0usize } else { count_val as usize };
            let body = collect_block_body(lines, &mut i, comment_style)?;
            let expanded_body = expand_rept_blocks(&body, comment_style, parse_int)?;
            for _ in 0..count {
                result.extend(expanded_body.iter().cloned());
            }
        } else if is_irp_start(&trimmed) {
            // .irp var, val1, val2, ...
            let args_str = trimmed[".irp".len()..].trim();
            let (var, values_str) = match args_str.find(',') {
                Some(pos) => (args_str[..pos].trim(), args_str[pos + 1..].trim()),
                None => (args_str, ""),
            };
            let values: Vec<&str> = values_str.split(',').map(|s| s.trim()).collect();
            let body = collect_block_body(lines, &mut i, comment_style)?;
            for val in &values {
                let subst_body: Vec<String> = body.iter().map(|line| {
                    let pattern = format!("\\{}", var);
                    line.replace(&pattern, val)
                }).collect();
                let subst_refs: Vec<&str> = subst_body.iter().map(|s| s.as_str()).collect();
                let expanded = expand_rept_blocks(&subst_refs, comment_style, parse_int)?;
                result.extend(expanded);
            }
        } else if trimmed == ".endr" {
            // stray .endr without .rept — skip
        } else {
            result.push(lines[i].to_string());
        }
        i += 1;
    }
    Ok(result)
}

// ── .macro / .endm expansion ───────────────────────────────────────────

/// Macro definition: name, parameter list (with optional defaults), and body lines.
pub(crate) struct MacroDef {
    params: Vec<String>,
    defaults: Vec<Option<String>>,
    body: Vec<String>,
}

/// Parse macro parameter list, handling `param = default_value` syntax.
///
/// GAS allows parameters like `enable = 1` where `1` is the default value
/// used when the caller omits that argument. Parameters are separated by
/// commas or whitespace.
fn parse_macro_params(params_str: &str) -> (Vec<String>, Vec<Option<String>>) {
    if params_str.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut params = Vec::new();
    let mut defaults = Vec::new();

    // Split on commas first (primary separator), then handle whitespace within each part
    for part in params_str.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // Check for `param = default_value` syntax
        if let Some(eq_pos) = part.find('=') {
            let param_name = part[..eq_pos].trim();
            let default_val = part[eq_pos + 1..].trim();
            // The param name might contain spaces (e.g., "enable = 1")
            // Take the last whitespace-delimited token as the param name
            // in case there are multiple tokens before the =
            let tokens: Vec<&str> = param_name.split_whitespace().collect();
            if tokens.len() > 1 {
                // Everything before the last token are separate params with no default
                for t in &tokens[..tokens.len() - 1] {
                    params.push(t.to_string());
                    defaults.push(None);
                }
            }
            if let Some(last) = tokens.last() {
                if !last.is_empty() {
                    params.push(last.to_string());
                    defaults.push(Some(default_val.to_string()));
                }
            }
        } else {
            // No default value - may contain space-separated params
            for token in part.split_whitespace() {
                if !token.is_empty() {
                    params.push(token.to_string());
                    defaults.push(None);
                }
            }
        }
    }
    (params, defaults)
}

/// Split macro invocation arguments, separating on commas and whitespace.
///
/// GAS allows both commas and spaces as macro argument separators.
/// Quoted strings are kept as a single argument with quotes stripped.
/// Parenthesized groups like `0(a1)` are kept together.
pub fn split_macro_args(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut args = Vec::new();
    let mut current = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut paren_depth = 0i32;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => {
                paren_depth += 1;
                current.push('(');
            }
            b')' => {
                paren_depth -= 1;
                current.push(')');
            }
            b',' if paren_depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    args.push(trimmed);
                }
                current.clear();
            }
            b' ' | b'\t' if paren_depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    args.push(trimmed);
                    current.clear();
                }
                // Skip remaining whitespace
                while i + 1 < bytes.len() && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\t') {
                    i += 1;
                }
                // If next char is comma, let the comma handle the split
                if i + 1 < bytes.len() && bytes[i + 1] == b',' {
                    // skip
                }
            }
            b'"' => {
                // Consume quoted string, stripping the outer quotes
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        current.push(bytes[i + 1] as char);
                        i += 2;
                        continue;
                    }
                    current.push(bytes[i] as char);
                    i += 1;
                }
                // Skip closing quote
            }
            _ => {
                current.push(bytes[i] as char);
            }
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        args.push(trimmed);
    }
    args
}

/// Expand `.macro`/`.endm` definitions and macro invocations.
///
/// Two-pass approach:
/// 1. Collect macro definitions (`.macro name [params]` ... `.endm`)
/// 2. Expand macro invocations: lines where the first word matches a defined macro
///
/// Handles nested macro definitions and recursive expansion.
pub fn expand_macros(
    lines: &[&str],
    comment_style: &CommentStyle,
) -> Result<Vec<String>, String> {
    use std::collections::HashMap;
    let mut macros: HashMap<String, MacroDef> = HashMap::new();
    let mut result = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = strip_comment(lines[i], comment_style).trim().to_string();
        if trimmed.starts_with(".macro ") || trimmed.starts_with(".macro\t") {
            // Parse: .macro name [param1[, param2, ...]]
            let rest = trimmed[".macro".len()..].trim();
            let (name, params_str) = match rest.find([' ', '\t', ',']) {
                Some(pos) => (rest[..pos].trim(), rest[pos..].trim().trim_start_matches(',')),
                None => (rest, ""),
            };
            let (params, defaults) = parse_macro_params(params_str);
            let mut body = Vec::new();
            let mut depth = 1;
            i += 1;
            while i < lines.len() {
                let inner = strip_comment(lines[i], comment_style).trim().to_string();
                if inner.starts_with(".macro ") || inner.starts_with(".macro\t") {
                    depth += 1;
                } else if inner == ".endm" || inner.starts_with(".endm ") || inner.starts_with(".endm\t") {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                body.push(lines[i].to_string());
                i += 1;
            }
            macros.insert(name.to_string(), MacroDef { params, defaults, body });
        } else if trimmed == ".endm" || trimmed.starts_with(".endm ") || trimmed.starts_with(".endm\t") {
            // stray .endm — skip
        } else if !trimmed.is_empty() && !trimmed.starts_with('.') && !trimmed.starts_with('#') {
            let first_word = trimmed.split([' ', '\t']).next().unwrap_or("");
            let potential_name = first_word.trim_end_matches(':');
            if potential_name != first_word {
                // It's a label, not a macro invocation
                result.push(lines[i].to_string());
            } else if let Some(mac) = macros.get(potential_name) {
                let args_str = trimmed[first_word.len()..].trim();
                let args = split_macro_args(args_str);
                let mut expanded_lines = Vec::new();
                for body_line in &mac.body {
                    let mut expanded = body_line.clone();
                    for (pi, param) in mac.params.iter().enumerate() {
                        let pattern = format!("\\{}", param);
                        let replacement = args.get(pi).map(|s| s.as_str()).unwrap_or_else(|| {
                            mac.defaults.get(pi)
                                .and_then(|d| d.as_deref())
                                .unwrap_or("0")
                        });
                        expanded = expanded.replace(&pattern, replacement);
                    }
                    expanded_lines.push(expanded);
                }
                let refs: Vec<&str> = expanded_lines.iter().map(|s| s.as_str()).collect();
                let re_expanded = expand_macros_with(&refs, &macros, comment_style)?;
                result.extend(re_expanded);
            } else {
                result.push(lines[i].to_string());
            }
        } else {
            result.push(lines[i].to_string());
        }
        i += 1;
    }
    Ok(result)
}

/// Re-expand macro invocations using already-collected macro definitions.
fn expand_macros_with(
    lines: &[&str],
    macros: &std::collections::HashMap<String, MacroDef>,
    comment_style: &CommentStyle,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    for line in lines {
        let trimmed = strip_comment(line, comment_style).trim().to_string();
        if trimmed.is_empty() || trimmed.starts_with('.') || trimmed.starts_with('#') {
            result.push(line.to_string());
            continue;
        }
        let first_word = trimmed.split([' ', '\t']).next().unwrap_or("");
        let potential_name = first_word.trim_end_matches(':');
        if potential_name != first_word {
            result.push(line.to_string());
        } else if let Some(mac) = macros.get(potential_name) {
            let args_str = trimmed[first_word.len()..].trim();
            let args = split_macro_args(args_str);
            let mut expanded_lines = Vec::new();
            for body_line in &mac.body {
                let mut expanded = body_line.clone();
                for (pi, param) in mac.params.iter().enumerate() {
                    let pattern = format!("\\{}", param);
                    let replacement = args.get(pi).map(|s| s.as_str()).unwrap_or_else(|| {
                        mac.defaults.get(pi)
                            .and_then(|d| d.as_deref())
                            .unwrap_or("0")
                    });
                    expanded = expanded.replace(&pattern, replacement);
                }
                expanded_lines.push(expanded);
            }
            let refs: Vec<&str> = expanded_lines.iter().map(|s| s.as_str()).collect();
            let re_expanded = expand_macros_with(&refs, macros, comment_style)?;
            result.extend(re_expanded);
        } else {
            result.push(line.to_string());
        }
    }
    Ok(result)
}

// ── .if / .else / .endif conditional assembly ──────────────────────────

/// Map x86-64 register names to unique integer values for use in `.if` expressions.
///
/// GAS assigns internal register encoding numbers to register names, allowing
/// comparisons like `.if %rsp == %rbp` in conditional assembly. The exact values
/// match GAS's internal encoding (AT&T register numbers starting at 104 for %rax).
/// Only the 64-bit GPRs are mapped since those are the only ones used in kernel
/// `.if` comparisons (UNWIND_HINT_REGS).
pub fn resolve_x86_registers(expr: &str) -> String {
    // Replace register names with numeric values, longest first to avoid partial matches.
    // Values match GAS internal encoding for x86-64 registers.
    const REGS: &[(&str, &str)] = &[
        ("%r10", "114"), ("%r11", "115"), ("%r12", "116"), ("%r13", "117"),
        ("%r14", "118"), ("%r15", "119"), ("%r8", "112"), ("%r9", "113"),
        ("%rax", "104"), ("%rcx", "105"), ("%rdx", "106"), ("%rbx", "107"),
        ("%rsp", "108"), ("%rbp", "109"), ("%rsi", "110"), ("%rdi", "111"),
        // 32-bit registers (used in some kernel macros)
        ("%eax", "40"), ("%ecx", "41"), ("%edx", "42"), ("%ebx", "43"),
        ("%esp", "44"), ("%ebp", "45"), ("%esi", "46"), ("%edi", "47"),
    ];
    let mut result = expr.to_string();
    for &(name, val) in REGS {
        result = result.replace(name, val);
    }
    result
}

/// Evaluate a simple `.if` condition expression.
///
/// Supports: integer literals, `==`, `!=`, `>=`, `<=`, `>`, `<`, and simple
/// arithmetic via `asm_expr::parse_integer_expr`. Non-zero result is true.
pub fn eval_if_condition(cond: &str) -> bool {
    eval_if_condition_inner(cond, |s| s.to_string())
}

/// Evaluate a `.if` condition with a pre-processing step for resolving names.
///
/// The `resolve` function is called on each side of a comparison operator
/// before integer expression parsing. Use this to resolve register names
/// or symbol values.
pub fn eval_if_condition_with_resolver<F: Fn(&str) -> String>(cond: &str, resolve: F) -> bool {
    eval_if_condition_inner(cond, resolve)
}

/// Find position of an isolated `>` or `<` that is not part of `>>`, `<<`, `>=`, or `<=`,
/// and not inside parentheses.
fn find_isolated_cmp(cond: &str, ch: char) -> Option<usize> {
    let bytes = cond.as_bytes();
    let target = ch as u8;
    let mut depth = 0i32;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        if depth == 0 && bytes[i] == target {
            let next = bytes.get(i + 1).copied();
            let prev = if i > 0 { Some(bytes[i - 1]) } else { None };
            if next != Some(b'>') && next != Some(b'<') && next != Some(b'=')
                && prev != Some(b'>') && prev != Some(b'<')
            {
                return Some(i);
            }
        }
    }
    None
}

fn eval_if_condition_inner<F: Fn(&str) -> String>(cond: &str, resolve: F) -> bool {
    eval_if_condition_dyn(cond, &resolve)
}

fn eval_if_condition_dyn(cond: &str, resolve: &dyn Fn(&str) -> String) -> bool {
    let cond = cond.trim();
    // Strip outer parentheses: (.Lfound != 1) -> .Lfound != 1
    let cond = strip_outer_parens(cond);

    // Handle || (logical OR) at top level — lowest precedence
    if let Some(pos) = find_top_level_op(cond, "||") {
        let lhs = &cond[..pos];
        let rhs = &cond[pos + 2..];
        return eval_if_condition_dyn(lhs, resolve) || eval_if_condition_dyn(rhs, resolve);
    }
    // Handle && (logical AND) at top level
    if let Some(pos) = find_top_level_op(cond, "&&") {
        let lhs = &cond[..pos];
        let rhs = &cond[pos + 2..];
        return eval_if_condition_dyn(lhs, resolve) && eval_if_condition_dyn(rhs, resolve);
    }

    // Find comparison operators at the top level (not inside parentheses).
    // Check "!=" before "==" and ">="/"<=" before ">"/"<".
    if let Some(pos) = find_top_level_op(cond, "!=") {
        let lhs = resolve(cond[..pos].trim());
        let rhs = resolve(cond[pos + 2..].trim());
        let l = asm_expr::parse_integer_expr(&lhs).unwrap_or(i64::MIN);
        let r = asm_expr::parse_integer_expr(&rhs).unwrap_or(i64::MAX);
        return l != r;
    }
    if let Some(pos) = find_top_level_op(cond, "==") {
        let lhs = resolve(cond[..pos].trim());
        let rhs = resolve(cond[pos + 2..].trim());
        let l = asm_expr::parse_integer_expr(&lhs).unwrap_or(i64::MIN);
        let r = asm_expr::parse_integer_expr(&rhs).unwrap_or(i64::MAX);
        return l == r;
    }
    if let Some(pos) = find_top_level_op(cond, ">=") {
        let lhs = resolve(cond[..pos].trim());
        let rhs = resolve(cond[pos + 2..].trim());
        let l = asm_expr::parse_integer_expr(&lhs).unwrap_or(i64::MIN);
        let r = asm_expr::parse_integer_expr(&rhs).unwrap_or(i64::MAX);
        return l >= r;
    }
    if let Some(pos) = find_top_level_op(cond, "<=") {
        let lhs = resolve(cond[..pos].trim());
        let rhs = resolve(cond[pos + 2..].trim());
        let l = asm_expr::parse_integer_expr(&lhs).unwrap_or(i64::MIN);
        let r = asm_expr::parse_integer_expr(&rhs).unwrap_or(i64::MAX);
        return l <= r;
    }
    // Try isolated ">" not part of ">>" or ">="
    if let Some(pos) = find_isolated_cmp(cond, '>') {
        let lhs = resolve(cond[..pos].trim());
        let rhs = resolve(cond[pos + 1..].trim());
        let l = asm_expr::parse_integer_expr(&lhs).unwrap_or(i64::MIN);
        let r = asm_expr::parse_integer_expr(&rhs).unwrap_or(i64::MAX);
        return l > r;
    }
    // Try isolated "<" not part of "<<" or "<="
    if let Some(pos) = find_isolated_cmp(cond, '<') {
        let lhs = resolve(cond[..pos].trim());
        let rhs = resolve(cond[pos + 1..].trim());
        let l = asm_expr::parse_integer_expr(&lhs).unwrap_or(i64::MIN);
        let r = asm_expr::parse_integer_expr(&rhs).unwrap_or(i64::MAX);
        return l < r;
    }
    // Simple integer expression: non-zero is true
    let resolved = resolve(cond);
    asm_expr::parse_integer_expr(&resolved).unwrap_or(0) != 0
}

/// Strip balanced outer parentheses from an expression.
/// `(expr)` -> `expr`, `((expr))` -> `expr`, `(a) + (b)` -> unchanged.
fn strip_outer_parens(s: &str) -> &str {
    let s = s.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return s;
    }
    // Check if the outer parens are truly balanced as a pair
    let inner = &s[1..s.len() - 1];
    let mut depth = 0i32;
    for ch in inner.bytes() {
        match ch {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth < 0 {
                    // Unbalanced: the closing paren in the middle means
                    // the outer parens are not a matching pair
                    return s;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        inner.trim()
    } else {
        s
    }
}

/// Find a comparison operator at the top level (not inside parentheses).
fn find_top_level_op(s: &str, op: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let op_bytes = op.as_bytes();
    let op_len = op_bytes.len();
    if bytes.len() < op_len {
        return None;
    }
    let mut depth = 0i32;
    for i in 0..=bytes.len() - op_len {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        if depth == 0 && &bytes[i..i + op_len] == op_bytes {
            return Some(i);
        }
    }
    None
}

// ── Shared data-value helpers ──────────────────────────────────────────

/// Check if a string looks like a GNU numeric label reference (e.g. "2f", "1b", "42f").
pub fn is_numeric_label_ref(s: &str) -> bool {
    if s.len() < 2 {
        return false;
    }
    let last = s.as_bytes()[s.len() - 1];
    if last != b'f' && last != b'F' && last != b'b' && last != b'B' {
        return false;
    }
    s[..s.len() - 1].bytes().all(|b| b.is_ascii_digit())
}

/// Find the position of the `-` operator in a symbol difference expression
/// like `sym_a - sym_b`. Skips position 0 to avoid matching a leading negation.
///
/// Returns `None` if no valid symbol difference operator is found.
pub fn find_symbol_diff_minus(expr: &str) -> Option<usize> {
    let bytes = expr.as_bytes();
    let len = bytes.len();
    let mut i = 1;
    while i < len {
        if bytes[i] == b'-' {
            let left_char = bytes[i - 1];
            let left_ok = left_char.is_ascii_alphanumeric()
                || left_char == b'_'
                || left_char == b'.'
                || left_char == b' '
                || left_char == b')';
            let right_start = expr[i + 1..].trim_start();
            if !right_start.is_empty() {
                let right_char = right_start.as_bytes()[0];
                let right_ok = right_char.is_ascii_alphabetic()
                    || right_char == b'_'
                    || right_char == b'.'
                    || right_char.is_ascii_digit()
                    || right_char == b'(';
                if left_ok && right_ok {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_c_comments() {
        assert_eq!(strip_c_comments("a /* b */ c"), "a  c");
        assert_eq!(strip_c_comments("a /* b\nc */ d"), "a \n d");
    }

    #[test]
    fn test_strip_c_comments_preserves_strings() {
        // /* inside a quoted string must not be treated as a comment
        assert_eq!(
            strip_c_comments(r#".asciz "hello/*world*/end""#),
            r#".asciz "hello/*world*/end""#
        );
        // Escaped quotes inside strings should not end the string
        assert_eq!(
            strip_c_comments(r#".asciz "a\"/*b*/c""#),
            r#".asciz "a\"/*b*/c""#
        );
        // /* outside strings should still be stripped
        assert_eq!(
            strip_c_comments(r#".asciz "hello" /* comment */ .byte 1"#),
            r#".asciz "hello"  .byte 1"#
        );
    }

    #[test]
    fn test_strip_comment_hash() {
        let style = CommentStyle::Hash;
        assert_eq!(strip_comment("movq %rax, %rbx # comment", &style), "movq %rax, %rbx ");
        assert_eq!(strip_comment(".asciz \"a#b\"", &style), ".asciz \"a#b\"");
    }

    #[test]
    fn test_strip_comment_slash_slash_and_at() {
        let style = CommentStyle::SlashSlashAndAt;
        assert_eq!(strip_comment("mov x0, x1 // comment", &style), "mov x0, x1 ");
        assert_eq!(strip_comment("mov x0, x1 @ comment", &style), "mov x0, x1 ");
        assert_eq!(strip_comment(".type foo, @function", &style), ".type foo, @function");
    }

    #[test]
    fn test_split_on_semicolons() {
        let parts = split_on_semicolons("a; b; c");
        assert_eq!(parts, vec!["a", " b", " c"]);
        let parts = split_on_semicolons(".asciz \"a;b\"; nop");
        assert_eq!(parts, vec![".asciz \"a;b\"", " nop"]);
        // Semicolons inside # comments should not cause splitting
        let parts = split_on_semicolons("nop # comment; with semicolons");
        assert_eq!(parts, vec!["nop # comment; with semicolons"]);
        // Semicolons inside // comments should not cause splitting
        let parts = split_on_semicolons("nop // comment; with semicolons");
        assert_eq!(parts, vec!["nop // comment; with semicolons"]);
        // Full-line // comments with semicolons
        let parts = split_on_semicolons("// struct {size_t a,b;} *p = (void*)x;");
        assert_eq!(parts, vec!["// struct {size_t a,b;} *p = (void*)x;"]);
        // Semicolons before comment still split normally
        let parts = split_on_semicolons("a; b # comment; c");
        assert_eq!(parts, vec!["a", " b # comment; c"]);
    }

    #[test]
    fn test_eval_if_condition() {
        assert!(eval_if_condition("1"));
        assert!(!eval_if_condition("0"));
        assert!(eval_if_condition("1 == 1"));
        assert!(!eval_if_condition("1 == 2"));
        assert!(eval_if_condition("1 != 2"));
        assert!(eval_if_condition("3 >= 2"));
        assert!(eval_if_condition("2 >= 2"));
        assert!(!eval_if_condition("1 >= 2"));
        assert!(eval_if_condition("1 <= 2"));
        assert!(eval_if_condition("3 > 2"));
        assert!(!eval_if_condition("2 > 2"));
        assert!(eval_if_condition("1 < 2"));
    }

    #[test]
    fn test_eval_if_condition_with_x86_registers() {
        // Test register equality (like kernel UNWIND_HINT_REGS)
        assert!(eval_if_condition_with_resolver("%rsp == %rsp", resolve_x86_registers));
        assert!(!eval_if_condition_with_resolver("%rsp == %rbp", resolve_x86_registers));
        assert!(eval_if_condition_with_resolver("%rbp == %rbp", resolve_x86_registers));
        assert!(eval_if_condition_with_resolver("%rdi == %rdi", resolve_x86_registers));
        assert!(eval_if_condition_with_resolver("%rdx == %rdx", resolve_x86_registers));
        assert!(eval_if_condition_with_resolver("%r10 == %r10", resolve_x86_registers));
        assert!(eval_if_condition_with_resolver("%rsp != %rbp", resolve_x86_registers));
    }

    #[test]
    fn test_resolve_x86_registers() {
        assert_eq!(resolve_x86_registers("%rsp"), "108");
        assert_eq!(resolve_x86_registers("%rbp"), "109");
        assert_eq!(resolve_x86_registers("%rdi"), "111");
        assert_eq!(resolve_x86_registers("%r10"), "114");
        // Ordering matters: %r10 must be replaced before %r8/%r9 to avoid partial matches
        assert_eq!(resolve_x86_registers("%r13"), "117");
    }

    #[test]
    fn test_is_numeric_label_ref() {
        assert!(is_numeric_label_ref("1f"));
        assert!(is_numeric_label_ref("42b"));
        assert!(!is_numeric_label_ref("f"));
        assert!(!is_numeric_label_ref("abc"));
    }

    #[test]
    fn test_find_symbol_diff_minus() {
        assert_eq!(find_symbol_diff_minus("a - b"), Some(2));
        assert_eq!(find_symbol_diff_minus("-5"), None);
        assert_eq!(find_symbol_diff_minus(".Lfoo-.Lbar"), Some(5));
    }

    #[test]
    fn test_split_macro_args() {
        assert_eq!(split_macro_args("a, b, c"), vec!["a", "b", "c"]);
        assert_eq!(split_macro_args("0(a1), x, y"), vec!["0(a1)", "x", "y"]);
        assert_eq!(split_macro_args(""), Vec::<String>::new());
    }

    #[test]
    fn test_parse_macro_params_simple() {
        let (params, defaults) = parse_macro_params("a, b, c");
        assert_eq!(params, vec!["a", "b", "c"]);
        assert_eq!(defaults, vec![None, None, None]);
    }

    #[test]
    fn test_parse_macro_params_with_defaults() {
        let (params, defaults) = parse_macro_params("a, b = 5, c");
        assert_eq!(params, vec!["a", "b", "c"]);
        assert_eq!(defaults, vec![None, Some("5".to_string()), None]);
    }

    #[test]
    fn test_parse_macro_params_space_separated() {
        let (params, defaults) = parse_macro_params("a b c");
        assert_eq!(params, vec!["a", "b", "c"]);
        assert_eq!(defaults, vec![None, None, None]);
    }

    #[test]
    fn test_parse_macro_params_mixed() {
        // GAS-style: `.macro ALT_NEW_CONTENT vendor_id, patch_id, enable = 1, new_c`
        let (params, defaults) = parse_macro_params("vendor_id, patch_id, enable = 1, new_c");
        assert_eq!(params, vec!["vendor_id", "patch_id", "enable", "new_c"]);
        assert_eq!(defaults, vec![None, None, Some("1".to_string()), None]);
    }

    #[test]
    fn test_parse_macro_params_empty() {
        let (params, defaults) = parse_macro_params("");
        assert!(params.is_empty());
        assert!(defaults.is_empty());
    }
}
