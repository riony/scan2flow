# Damaged-support prior, 5 November 2026

This directory preserves explicit closure-domain and verifier prototypes. It does not infer missing geometry. Callers supply their own mesh, chart prescription and provenance; no saved donor, chart manifest or source-specific vertex selections are included.

`prior_cdt.cpp` is intentionally distinct from the patch adapter: it retains the complete convex chart domain, including triangles inside constrained donor islands. Applying the patch adapter's odd/even hole rule here would remove those islands. The caller must establish a convex, non-overlapping chart and a valid lift. `charts.py` retains those checks for explicit list-valued chart definitions, plus rejection of a cap that duplicates an owned donor face. Its chart inspection needs Shapely; the ownership check needs only NumPy.

`stl_encoding.py` retains both the original exact-canonical comparison and the corrected explicit float32 encoding check. The corrected check proves a bijective vertex mapping and preserves oriented donor faces without applying a geometric tolerance. It returns a report and does not replace older evidence.

Historical stages:

1. The damaged-input attempt stopped at its open-prior prerequisite. Oriented face fans resolved the branching boundary, but five walks remained nonplanar. An eighth planar cap duplicated an isolated surviving donor triangle; the ownership correction rejected that saved cap. The partial result had 203 faces and 61 open edges, so neither a complete H1 nor a VDB/CFD result existed.
2. A separately prescribed seven-chart prior retained all 124 donor vertices and 196 donor faces, adding 48 explicitly prescribed faces. The sign/topology/intersection checks passed. Its minimum quality was 0.016285745, with 226 faces at or below 0.05; final H1 quality was still outstanding.
3. The first reader verifier failed because canonical doubles do not always equal binary STL's float32 coordinates. The later exact-encoding verifier passed with the same saved geometry and intersection evidence. The original failed check remains represented by `donor_matches_canonical_reader_before_encoding_fix` and the recorded outcome.

The prior pass was a closed-sign-input prerequisite, not a reconstruction or aerodynamic validation. Missing support legs remained absent. The unchanged native adapter requires the same optional CGAL C++17 toolchain as the patch prototype. No native run or case orchestration is provided.
