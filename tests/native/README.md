# Optional native adapter import verification

These six tiny fixtures check the retained CGAL adapters using hand-authored points only. They do not replay the historical H0/H1 controls, create capture-derived geometry, or run the larger remesher, CAD, meshing or CFD tools. No binary or output is checked in.

With a C++17 compiler, compatible CGAL headers and GMP/MPFR already available, run from the repository root. CGAL 6.1.1 was used for the original adapters. The Python environment needs NumPy.

```bash
control_build_dir=$(mktemp -d)
g++ -std=c++17 -O0 prototypes/attempts/constrained_triangulation_20261105/cdt_patch.cpp -o "$control_build_dir/cdt_patch" -lgmp -lmpfr
g++ -std=c++17 -O0 prototypes/attempts/damaged_prior_20261105/prior_cdt.cpp -o "$control_build_dir/prior_cdt" -lgmp -lmpfr
python tests/native/verify_cdt.py --patch-binary "$control_build_dir/cdt_patch" --prior-binary "$control_build_dir/prior_cdt" --fixtures prototypes/attempts/constrained_triangulation_20261105/synthetic_fixtures.json --output "$control_build_dir/results"
```

The patch tests check collinear atomic boundary retention, a concave domain with a hole, duplicate-point rejection and crossing rejection. The separate prior tests check preservation of a donor island inside a complete convex domain and rejection of crossing constraints. Each native fixture has a 40-second subprocess timeout; logs and the scalar assessment go only to the explicitly supplied new output directory.
