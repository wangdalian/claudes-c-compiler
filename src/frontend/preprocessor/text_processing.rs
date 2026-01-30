//! Text processing utilities for the preprocessor.
//!
//! Low-level text transformations: block/line comment stripping,
//! line continuation (backslash-newline) joining, parenthesis
//! balancing checks, and directive line splitting.

use super::utils::{skip_literal_bytes, copy_literal_bytes_raw};
use super::preprocessor::Preprocessor;

impl Preprocessor {
    /// Check if a line has unbalanced parentheses, indicating a multi-line
    /// macro invocation that needs to be joined with subsequent lines.
    /// Skips string/char literals and line comments.
    pub(super) fn has_unbalanced_parens(line: &str) -> bool {
        let mut depth: i32 = 0;
        let bytes = line.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            match bytes[i] {
                b'"' | b'\'' => {
                    i = skip_literal_bytes(bytes, i, bytes[i]);
                }
                b'(' => { depth += 1; i += 1; }
                b')' => { depth -= 1; i += 1; }
                b'/' if i + 1 < len && bytes[i + 1] == b'/' => break,
                _ => { i += 1; }
            }
        }

        depth > 0
    }

    /// Strip C-style block comments (/* ... */) and C++ line comments (// ...).
    /// Returns the stripped source and a mapping from output line numbers to
    /// original source line numbers, for correct __LINE__ tracking.
    ///
    /// Block comments are replaced with a single space (per C11 5.1.1.2 phase 3),
    /// which avoids breaking preprocessor directives that have block comments
    /// between `#` and the directive keyword (e.g., `#/*...\n...*/if 1`).
    /// Uses raw byte operations to preserve UTF-8 sequences in string literals.
    pub(super) fn strip_block_comments(source: &str) -> (String, Vec<usize>) {
        let mut result: Vec<u8> = Vec::with_capacity(source.len());
        let bytes = source.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        // Track source line number (0-based) as we scan through input
        let mut src_line: usize = 0;
        // For each output line, record which source line it corresponds to
        let mut line_map: Vec<usize> = Vec::new();
        // Track the source line at the start of the current output line
        let mut current_output_line_src = 0usize;

        while i < len {
            match bytes[i] {
                b'"' | b'\'' => {
                    // Copy string/char literals verbatim (don't strip comments inside)
                    let old_i = i;
                    i = copy_literal_bytes_raw(bytes, i, bytes[i], &mut result);
                    // Count newlines in the literal for source line tracking
                    for &b in &bytes[old_i..i] {
                        if b == b'\n' {
                            src_line += 1;
                            // Record mapping for each output line
                            line_map.push(current_output_line_src);
                            current_output_line_src = src_line;
                        }
                    }
                }
                b'/' if i + 1 < len && bytes[i + 1] == b'*' => {
                    // Block comment - replace entire comment with a single space
                    i += 2;
                    result.push(b' ');
                    while i < len {
                        if i + 1 < len && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                            i += 2;
                            break;
                        }
                        if bytes[i] == b'\n' {
                            src_line += 1;
                        }
                        i += 1;
                    }
                    // Don't emit newlines - the comment is fully replaced by one space
                }
                b'/' if i + 1 < len && bytes[i + 1] == b'/' => {
                    // Line comment - skip to end of line
                    i += 2;
                    while i < len && bytes[i] != b'\n' {
                        i += 1;
                    }
                }
                b'\n' => {
                    result.push(b'\n');
                    line_map.push(current_output_line_src);
                    src_line += 1;
                    current_output_line_src = src_line;
                    i += 1;
                }
                _ => {
                    result.push(bytes[i]);
                    i += 1;
                }
            }
        }
        // Record the last line
        line_map.push(current_output_line_src);

        // Input was valid UTF-8, and we only removed/replaced ASCII characters
        // (comments with spaces/newlines), so result is still valid UTF-8.
        let text = String::from_utf8(result)
            .expect("comment stripping produced non-UTF8 (input was valid UTF-8)");
        (text, line_map)
    }

    /// Join lines that end with backslash (line continuation).
    /// Also handles backslash followed by trailing whitespace before newline,
    /// matching GCC/Clang behavior (GCC warns: "backslash and newline separated by space").
    pub(super) fn join_continued_lines(&self, source: &str) -> String {
        let mut result = String::with_capacity(source.len());
        let mut continuation = false;

        for line in source.lines() {
            let bs_pos = Self::find_continuation_backslash(line);
            if continuation {
                // This line continues from the previous
                if let Some(pos) = bs_pos {
                    result.push_str(&line[..pos]);
                    // Still continuing
                } else {
                    result.push_str(line);
                    result.push('\n');
                    continuation = false;
                }
            } else if let Some(pos) = bs_pos {
                result.push_str(&line[..pos]);
                continuation = true;
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        result
    }

    /// Check if a line ends with a continuation backslash (optionally followed by whitespace).
    /// Returns the byte position of the backslash if found, None otherwise.
    pub(super) fn find_continuation_backslash(line: &str) -> Option<usize> {
        let trimmed = line.trim_end();
        if trimmed.ends_with('\\') {
            Some(trimmed.len() - 1)
        } else {
            None
        }
    }
}

/// Strip a // comment from a directive line, but not inside string literals.
/// Returns a `Cow<str>` to avoid allocation when no comment is found (the
/// common case). Only allocates a new String when a `//` comment is present
/// and needs to be stripped.
pub(super) fn strip_line_comment(line: &str) -> std::borrow::Cow<'_, str> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        match bytes[i] {
            b'"' | b'\'' => {
                i = skip_literal_bytes(bytes, i, bytes[i]);
            }
            b'/' if i + 1 < len && bytes[i + 1] == b'/' => {
                return std::borrow::Cow::Owned(line[..i].trim_end().to_string());
            }
            _ => i += 1,
        }
    }

    std::borrow::Cow::Borrowed(line)
}

/// Split a string into the first word and the rest.
/// For preprocessor directives, '(' is also a word boundary so that
/// `#if(expr)` is correctly parsed as keyword="if", rest="(expr)".
pub(super) fn split_first_word(s: &str) -> (&str, &str) {
    let s = s.trim();
    if let Some(pos) = s.find(|c: char| c.is_whitespace() || c == '(') {
        if s.as_bytes()[pos] == b'(' {
            // Don't trim the '(' - it's part of the rest
            (&s[..pos], &s[pos..])
        } else {
            (&s[..pos], s[pos..].trim())
        }
    } else {
        (s, "")
    }
}
