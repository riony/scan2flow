# Nonuniform spline refit attempt

Historical date: 17 October 2026. Imported as a prototype on 25 November 2026.

The original refit concentrated knots around an identified feature using a
65% uniform and 35% Gaussian CDF mixture. Its reference-fidelity selection
carried forward a 40 by 28 bicubic interior candidate with lambda 1e-5.
Unsupported physical curvature remained unvalidated. The selected knot family,
feature location and selection tolerance were modeling choices.

`fit.py` retains the CDF inversion, physical derivative scaling, spatial folds,
45 by 35 sampled Hessian objective and four-step robust solve. Callers supply
the feature center, chart range and generic input coordinates. The source
reference data, alternate fusion input, search driver, caches, boundaries and
exports are absent.

The 17 November strip-withholding audit subsequently found severe
under-integration of this sampled bending objective for an oscillatory withheld
strip. The objective here is deliberately preserved as historical code.
No exact-quadrature replacement or corrected fit is claimed. Synthetic checks
verify affine reproduction and consistency with the later retained fitting
kernel; they do not validate gap geometry.

`provenance.json` records original source hashes, symbol ranges and adaptations.
