# Constrained patch proof and full H0 qualification, 5 November 2026

These prototypes retain the authored CGAL adapter and deterministic checks used in two separate historical control stages. They do not run on import and are outside the installed CLI. No saved control surface or capture-derived geometry is included.

`cdt_patch.cpp` uses exact CGAL predicates/constructions, explicit atomic constraints and odd/even domain nesting. It rejects duplicate projected points, crossed or overlapping constraints, omitted input IDs and added vertices. It preserves holes. `synthetic_fixtures.json` contains only the original hand-authored collinear, concave-with-hole, duplicate-point and crossing fixtures.

`metrics.py` retains triangle quality, atomic boundary extraction, point-to-segment distance, deterministic surface sampling and feature metrics. VTK is imported only for explicit closest-surface queries; SciPy is needed for component counting. `checks.py` retains topology, opposite seam incidence and full-body structural/exact-contact reconciliation. The latter consumes already-produced evidence and never runs external tools. An open pair cannot satisfy its closed-body gate. Sampled distance is not a continuous certificate.

Historical outcomes are deliberately separate:

1. The first proof verifier stopped after the first patch was triangulated: `Trimesh.split` attempted repair and needed unavailable `networkx`. The corrected verifier used direct SciPy face adjacency, without installing a dependency or regenerating that patch.
2. The two-patch proof passed: all 512 shared segments and 513 seam vertices were retained without coordinate changes, with minimum quality 0.087988657. This was an open synthetic control pair, not a closed CFD body.
3. Full saved H0 qualification subsequently passed on all seven patches and all 2,688 locked segments. The saved carrier and transferred positions shared 338,118 faces. It did not test a carrier built from damaged input or qualify reconstruction/aerodynamics.

Python helpers require NumPy/trimesh, plus the optional libraries above. The unchanged C++ source was historically built with CGAL 6.1.1, a C++17 compiler and its GMP/MPFR dependencies. No native binary is included. Tiny synthetic adapter tests are import verification, not reruns of either historical qualification. `provenance.json` identifies retained source hashes, extracted sections and outcomes.
