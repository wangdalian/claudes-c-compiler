# Mem2Reg (SSA Promotion)

Promotes stack allocations to SSA registers using the standard dominance frontier algorithm.

## Files

- `mem2reg.rs` - Core mem2reg pass. Builds dominator tree and dominance frontiers, identifies promotable allocas (non-address-taken, single-type), inserts phi nodes, and renames values.
- `phi_eliminate.rs` - Phi node elimination pass. Converts SSA phi nodes back to register moves for backend code generation. Uses smart conflict detection: when a block has multiple phis, only those involved in cyclic dependencies (e.g., swap patterns like `a=b, b=a`) use temporary values. Non-conflicting phis emit direct copies, avoiding unnecessary stack traffic.
