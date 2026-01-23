/// Shared utility functions for the preprocessor module.

/// Check if a character can start a C identifier.
pub fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Check if a character can continue a C identifier.
pub fn is_ident_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Skip past a string or character literal in a char slice, starting at position `i`.
/// Returns the position after the closing quote. Handles backslash escapes.
/// `quote` should be either `'"'` or `'\''`.
pub fn skip_literal(chars: &[char], start: usize, quote: char) -> usize {
    let len = chars.len();
    let mut i = start + 1; // skip opening quote
    while i < len {
        if chars[i] == '\\' && i + 1 < len {
            i += 2; // skip escape sequence
        } else if chars[i] == quote {
            return i + 1; // past closing quote
        } else {
            i += 1;
        }
    }
    i // unterminated literal - return end
}

/// Skip past a string or character literal in a byte slice, starting at position `i`.
/// Returns the position after the closing quote. Handles backslash escapes.
/// Byte-oriented version for code that processes `&[u8]` directly.
pub fn skip_literal_bytes(bytes: &[u8], start: usize, quote: u8) -> usize {
    let len = bytes.len();
    let mut i = start + 1; // skip opening quote
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2;
        } else if bytes[i] == quote {
            return i + 1;
        } else {
            i += 1;
        }
    }
    i
}

/// Copy a string or character literal from a byte slice into a String result.
/// Returns the position after the closing quote. Handles backslash escapes.
/// Byte-oriented version for code that processes `&[u8]` directly.
pub fn copy_literal_bytes(bytes: &[u8], start: usize, quote: u8, result: &mut String) -> usize {
    let len = bytes.len();
    result.push(bytes[start] as char); // opening quote
    let mut i = start + 1;
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            result.push(bytes[i] as char);
            result.push(bytes[i + 1] as char);
            i += 2;
        } else if bytes[i] == quote {
            result.push(bytes[i] as char);
            return i + 1;
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    i
}

/// Copy a string or character literal from chars into result, starting at position `i`.
/// Returns the position after the closing quote. Handles backslash escapes.
pub fn copy_literal(chars: &[char], start: usize, quote: char, result: &mut String) -> usize {
    let len = chars.len();
    result.push(chars[start]); // push opening quote
    let mut i = start + 1;
    while i < len {
        if chars[i] == '\\' && i + 1 < len {
            result.push(chars[i]);
            result.push(chars[i + 1]);
            i += 2;
        } else if chars[i] == quote {
            result.push(chars[i]);
            return i + 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    i
}
