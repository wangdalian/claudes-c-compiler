Fix parser README: add missing `[` token to is_paren_declarator() table.
The table omitted `[` (LBracket) from the "Declarator grouping" row,
even though the code at declarators.rs:135 includes it.
