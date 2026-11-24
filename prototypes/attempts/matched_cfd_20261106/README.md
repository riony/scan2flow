# Matched CFD evidence policy, 5-6 November 2026

These evidence-only prototypes never launch OpenFOAM or change a case. `evaluate.py` retains the strict log/force parser for the historical v2412, 24-rank, full-domain matched control. Its patch identities, domain bounds, reference values and thresholds are protocol-specific. `gate.py` checks the recorded invocation against the log, requires one constant mesh and all configured quality rows, and keeps full-geometry diagnostics separate from primary acceptance. `cellset.py` preserves both the compact-empty-set failure and its later parser correction. NumPy is the only nonstandard Python dependency.

Historical sequence:

1. On 5 November, preserving the baseline's original facets conflicted with the uniform input-triangle-quality gate. Preflight stopped pending a decision.
2. The owner approved a baseline-only input-quality exception on **6 November**. Its directory and UTC evening timestamps say 5 November; the owner date and Singapore local date are 6 November. Both surfaces then qualified with every other gate unchanged.
3. The first baseline volume mesh failed the original complete `allGeometry` gate: 764,724 cells, 28,449 concave cells and one failed check. Return code zero did not override that failure. The H1 mesh and both solves were unstarted.
4. Read-only diagnosis located most flagged cells at refinement transitions. An internal-smoothing setting was proposed but never run. A subsequent review corrected the retained historical calibration-mesh identity and kept the frozen matched dictionaries.
5. An explicitly accepted and calibrated gate amendment made `-constant -allTopology -meshQuality` primary; complete `allGeometry` stayed diagnostic. The serial calibration passed primary checks while retaining 15,197 concave-cell and 34 low-connectivity-determinant diagnostics. This did not retroactively pass the original failed gate or prove the diagnostic flags harmless.
6. The saved matched pair then passed the aerodynamic retention check: relative Cd change -0.0070%, absolute Cl change 0.00017307. One H1 mesh and two solves ran, with no baseline remesh or internal-smoothing variant. This is the saved pair at that discretization, not mesh independence, experimental validation or qualification of footage-derived construction.
7. A diagnostic helper stopped on valid compact `0()` cellSet syntax. The fixed parser completed existing-mesh diagnostics; no mesh, check or solve was repeated. The old `cell_set_before_compact_zero_fix` remains available to reproduce that parser failure with a synthetic payload.

Force means use iterations 2500-3000 inclusive. Steadiness compares 2501-2750 with 2751-3000, with limits 0.002 Cd and 0.005 Cl. Pair thresholds were 2% relative Cd and 0.02 absolute Cl. Missing/duplicate/nonfinite evidence, incomplete schedules and mismatched invocation remain failures. A diagnostic `completed` field is never a primary `passed` result.

All test logs and cellSet payloads are fabricated. No historical mesh, fields, host paths or case inputs are included. `provenance.json` records source hashes, adaptations and separate historical outcomes.
