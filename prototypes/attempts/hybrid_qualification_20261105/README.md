# Retained prior-remeshed hybrid, 5 November 2026

`remesh_carrier.cpp` is the retained authored CGAL closed-prior remesher, including its historical constrained remeshing, quality edge flips and tangent optimization. The source is preserved without numerical changes. It requires a compatible CGAL C++17 toolchain and is not a production backend. No binary, saved geometry or larger native-remeshing run is included in the import checks.

`transfer.py` extracts the existing numerical transfer into an explicit-input function. The caller supplies a closed carrier, declared feature segments, both donor projection targets and surviving-donor distances. The original 13 mm fixed strip, 13-26 mm transition, 1.3-3.9 mm donor transition and finite quality/crease damping are retained. The function returns the result and processing labels; it does not query source archives, run OpenVDB or write geometry.

This construction remeshed the closed prior directly with CGAL. OpenVDB supplied a transfer guide, and surviving panels supplied nearby donor targets. It did not extract carrier topology from VDB or establish an unsigned open-sheet union. Donor selection is distance-based, without a normal-qualified support test. Full-weight equality is to the composite target after damping; approximate nearest-prior region provenance is not independent physical ownership evidence. The finite construction guard does not replace a qualification check.

The saved carrier/hybrid passed the **retained, retrospectively amended evaluator**, with 323,748 faces and minimum qualities 0.120056/0.082221. The original first-pass assessment and evaluator source were overwritten and were not recovered. Their verdict is unknown. Current source hashes must not be presented as recovered intermediate versions.

The original guarded runner exited 1 after geometry creation on a metadata filename mismatch. A side script recovered evaluation of the saved geometry. Evaluator mask changes after scoring are distinct from that filename fix. The original 900 s/16 GB protocol and 1200 s/32 GiB runner limits differed, although measured use was below both. Record reconciliation also documented a deleted intermediate archive directory outside the protected-file set; its exact former contents were not recovered.

These caveats remain part of the result. No new evaluator, clean-reference masking scheme, geometry or CFD run was created during extraction. NumPy and trimesh are sufficient for the extracted transfer; its shared metrics are in the constrained-triangulation prototype.
