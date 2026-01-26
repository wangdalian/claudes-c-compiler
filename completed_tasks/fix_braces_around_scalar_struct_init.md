Task: Fix flat struct initialization with braces-around-scalar and array-of-struct fields
Status: Complete
Branch: master

Problem:
Initializing arrays of structs with flat (brace-less) initializer lists caused
segfaults when the initializer contained braces-around-scalar values (e.g., {&b})
or when struct fields were themselves arrays of structs.

Example of code that segfaulted:
  struct tag1 { int *i01[2]; char *c01[2]; };
  struct tag2 { int *i02[2]; struct tag1 st02[2]; };
  struct tag2 st[2] = {
      {{&a,&a},{{{&a,&a},{&b,&b}},{{&a,&a},{&b,&b}}}},
      {&a,&a,&a,&a,&b,&b,&a,&a,&b,&b}    // Second element: flat init
  };

Root cause:
Two distinct bugs in the struct initializer lowering code:

1. Braces-around-scalar in flat array init (struct_init.rs):
   When consuming flat initializer items for array fields, both
   emit_array_field_expr_init and emit_field_array_designated only handled
   Initializer::Expr, silently skipping Initializer::List items. Per C11
   6.7.9p13, "The initializer for a scalar may optionally be enclosed in
   braces." Values like {&b} were skipped, leaving array elements as NULL,
   which caused segfaults when dereferenced.

2. Array-of-struct fields in lower_local_struct_init (stmt.rs):
   When a struct field is an array of structs (e.g., struct tag1 st02[2])
   and the initializer uses flat expressions without braces, the code used
   IrType::from_ctype on the struct element type, which returns IrType::Ptr
   (8 bytes). This caused it to:
   - Treat each 32-byte struct as an 8-byte pointer
   - Store only one flat init item per struct element instead of recursively
     filling all fields
   - Leave most struct fields uninitialized (NULL pointers -> segfault)

Fix:
1. In struct_init.rs (two locations): Changed flat array element consumption
   loops to use unwrap_nested_init_expr to handle Initializer::List items
   that wrap scalar values in braces.

2. In stmt.rs: Added detection for struct/union element types in the
   array field flat init path. When the array element type is a struct,
   delegate to emit_struct_init which correctly consumes multiple flat items
   per struct element, instead of treating each item as a simple scalar store.

Files changed:
- src/ir/lowering/struct_init.rs: Two locations updated to unwrap braces
- src/ir/lowering/stmt.rs: Added struct array field handling + braces unwrap

Tests fixed:
- compiler_suite_0054_0007 (x86, ARM, RISC-V)
- compiler_suite_0150_0004 (x86, ARM, RISC-V)

Verification:
- x86:    2946/2990 passed (98.5%) - no regressions
- ARM:    2809/2868 passed (97.9%) - no regressions
- RISC-V: 2845/2859 passed (99.5%) - no regressions
- Internal repo tests: 341/343 passed (99.4%) - no regressions
