//! Structured diagnostic infrastructure for the compiler.
//!
//! Provides a `DiagnosticEngine` that collects errors, warnings, and notes
//! with source locations and renders them in GCC-compatible format with
//! optional source snippet display.
//!
//! # Output format
//! ```text
//! file.c:10:5: error: expected ';', got '}'
//!     int x = 42
//!             ^
//! ```

use crate::common::source::{SourceManager, Span};

/// Severity level for a diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A fatal error that prevents compilation from continuing.
    Error,
    /// A warning that does not prevent compilation.
    Warning,
    /// A supplementary note attached to a previous error or warning.
    Note,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
        }
    }
}

/// A single diagnostic message with severity, location, and message text.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    /// Optional follow-up notes providing additional context.
    pub notes: Vec<Diagnostic>,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    /// Create a new note diagnostic.
    pub fn note(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Note,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    /// Attach a source span to this diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a follow-up note to this diagnostic.
    /// TODO: Use for "note: previous declaration was here" style follow-ups.
    #[allow(dead_code)]
    pub fn with_note(mut self, note: Diagnostic) -> Self {
        self.notes.push(note);
        self
    }
}

/// Collects and renders compiler diagnostics with source context.
///
/// The engine is designed to be threaded through compilation phases. Each
/// phase calls `emit()` to report errors and warnings. After each phase,
/// the driver checks `has_errors()` to decide whether to continue.
///
/// Diagnostics are printed to stderr immediately on emit, matching GCC
/// behavior where errors appear as soon as they are discovered.
pub struct DiagnosticEngine {
    error_count: usize,
    warning_count: usize,
    /// Reference to the source manager for span resolution and snippet display.
    /// Set after the preprocessing/lexing phase creates the SourceManager.
    source_manager: Option<SourceManager>,
}

impl DiagnosticEngine {
    /// Create a new diagnostic engine with no source manager.
    pub fn new() -> Self {
        Self {
            error_count: 0,
            warning_count: 0,
            source_manager: None,
        }
    }

    /// Set the source manager for span resolution and snippet display.
    pub fn set_source_manager(&mut self, sm: SourceManager) {
        self.source_manager = Some(sm);
    }

    /// Take ownership of the source manager (for passing to codegen for debug info).
    pub fn take_source_manager(&mut self) -> Option<SourceManager> {
        self.source_manager.take()
    }

    /// Get a reference to the source manager (if set).
    pub fn source_manager(&self) -> Option<&SourceManager> {
        self.source_manager.as_ref()
    }

    /// Emit a diagnostic: print it to stderr and update counts.
    pub fn emit(&mut self, diag: &Diagnostic) {
        self.render_diagnostic(diag);
        match diag.severity {
            Severity::Error => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            Severity::Note => {} // notes don't count separately
        }
    }

    /// Emit an error with a message and span.
    pub fn error(&mut self, message: impl Into<String>, span: Span) {
        let diag = Diagnostic::error(message).with_span(span);
        self.emit(&diag);
    }

    /// Emit an error with just a message (no span).
    pub fn error_no_span(&mut self, message: impl Into<String>) {
        let diag = Diagnostic::error(message);
        self.emit(&diag);
    }

    /// Emit a warning with a message and span.
    pub fn warning(&mut self, message: impl Into<String>, span: Span) {
        let diag = Diagnostic::warning(message).with_span(span);
        self.emit(&diag);
    }

    /// Emit a warning with just a message (no span).
    pub fn warning_no_span(&mut self, message: impl Into<String>) {
        let diag = Diagnostic::warning(message);
        self.emit(&diag);
    }

    /// Returns true if any errors have been emitted.
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Number of errors emitted so far.
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Number of warnings emitted so far.
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Render a single diagnostic to stderr, including location, severity,
    /// message, source snippet with caret, and any follow-up notes.
    fn render_diagnostic(&self, diag: &Diagnostic) {
        let mut msg = String::new();

        // Format: "file:line:col: severity: message"
        if let Some(span) = diag.span {
            if let Some(ref sm) = self.source_manager {
                let loc = sm.resolve_span(span);
                msg.push_str(&format!("{}:{}:{}: ", loc.file, loc.line, loc.column));
            }
        }
        msg.push_str(&format!("{}: {}", diag.severity, diag.message));
        eprintln!("{}", msg);

        // Source snippet with caret underline
        if let Some(span) = diag.span {
            self.render_snippet(span);
        }

        // Render any follow-up notes
        for note in &diag.notes {
            self.render_diagnostic(note);
        }
    }

    /// Render a source code snippet with a caret pointing to the error location.
    ///
    /// Output format:
    /// ```text
    ///     int x = 42
    ///             ^~
    /// ```
    fn render_snippet(&self, span: Span) {
        let sm = match &self.source_manager {
            Some(sm) => sm,
            None => return,
        };

        let source_line = match sm.get_source_line(span) {
            Some(line) => line,
            None => return,
        };

        // Don't render snippets for empty or whitespace-only lines
        if source_line.trim().is_empty() {
            return;
        }

        // Resolve the column for caret positioning
        let loc = sm.resolve_span(span);
        let col = loc.column as usize;

        // Print the source line with indentation
        eprintln!(" {}", source_line);

        // Build the caret line: spaces up to the column, then ^ with tildes
        if col > 0 {
            let padding = " ".repeat(col);
            let span_len = (span.end.saturating_sub(span.start)) as usize;
            let underline = if span_len > 1 {
                format!("^{}", "~".repeat(span_len - 1))
            } else {
                "^".to_string()
            };
            eprintln!("{}{}", padding, underline);
        }
    }
}

impl Default for DiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}
