# return_mesh_20261028

Historical date: 2026-10-28. Retrospective code extraction, not a new run.

Exact splitting, positive-winding classification, fan separation and numerical
crack caps were parts of a retained diagnostic. The rounded diagnostic passed an
exact intersection test but failed the consumer intersection/triangle-quality
gates; no matching smooth CAD was produced.

Native `split_intersections` takes triangle OFF, per-face integer IDs, and output
prefix; it emits split OFF, IDs, cut edges and diagnostics. Exit 5 can retain a
useful split intermediate while reporting an unsuccessful final repair.
`check_exact` takes OFF and JSON output (plus pair sidecar). CGAL/GMP/MPFR and the
experimental autorefine API are optional native requirements. Python NumPy
helpers accept caller arrays; winding input triangles and patch arrays are explicit.

No exact-split graph orchestration, full winding exterior selection, spherical
rounding, sample-retention measurement, source mesh or consumer-check invocation
is included. The cap helper retains historical tag 2000 for prescribed caps;
caller must use compatible provenance IDs. Source positions are not bundled.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
