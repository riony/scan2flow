# wrap_controls_20261031

Historical date: 2026-10-31. Retrospective code extraction, not a new run.

Original dense controls were invalidated by an in-place cached-mesh transform and
an omitted named sharp feature. `dense_cache_fix_unrun.py` was saved but not rerun
within that original allowance. Amendment 1 used a copied extractor and explicit
feature enumeration; all five dense/OpenVDB recipes still failed feature gates.
Exact reader-geometry diagnostics cleared one saved sparse trial's reported
contacts, not its feature error or the other trials.

Three named Python variants keep the legacy defect, the originally-unrun fix and
the amended implementation distinct. `run(mesh, segments, candidate, prefix)`
requires explicit feature segments and config keys `pitch`, `erode_voxels`,
`protected_filter`, `smooth_iterations`, `id`; emits OFF checkpoints/JSON.
NumPy/SciPy/trimesh and optional scikit-image support voxel extraction. Feature
strips retain .013/.026 input-unit widths. Native `vdb_adapter` accepts OFF,
segment text (count then six coordinates per row), pitch, closing voxels, mean
iterations, output prefix; it needs OpenVDB/TBB (historically OpenVDB 10.0.1).
`classify_hits` takes reader OFF, reported endpoint/triangle index triples, JSON;
CGAL exact constructions is required. Its separation field assumes metre inputs.

No source meshes, feature coordinate sets, frozen recipe tables, orchestration,
complete evaluator or CFD is included. Features must be explicitly complete:
thresholding dihedral angles alone omitted the historically required edge.
The legacy extractor intentionally mutates its cache and is not a safe default.
Native classifier code is authored diagnostic code, not copied vendor checker
source; raw consumer logs are not automatically waived.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
