Task: Rewrite the optimization passes README (src/passes/README.md)

The current README has several inaccuracies and missing details:
- if_convert description says "0-1 simple instructions" but actual limit is 8 (MAX_ARM_INSTS)
- if_convert only describes diamond pattern, misses triangle pattern
- simplify is missing many transformations (GEP, math lib calls, select, cast chains, cmp simplifications)
- gvn is missing store-to-load forwarding, incorrectly implies Cmp commutative canonicalization
- narrow only describes one of three transformation phases
- inline description has minimal detail about heuristics/thresholds
- dependency graph has missing edges (copy_prop -> licm, etc.)
- Various other minor inaccuracies

Will carefully verify every claim against the code and rewrite the full README as
a comprehensive design document.
