# Uniform spatial-median spline attempt

Historical date: 16 October 2026. Imported as a prototype on 25 November 2026.

The original attempt reduced dense reference samples to equal-weight spatial
median cells, then fit a robust bicubic height field. Four spatial folds selected
the smallest control net within 10% of the best reference p90 residual, followed
by the strongest qualifying bending penalty. This produced a useful smooth
open-patch candidate, but did not earn expansion or CFD acceptance. Reference
fidelity did not establish physical accuracy in unsupported regions.

`fit.py` retains spatial medians, uniform knots, tensor basis derivatives,
25 by 20 sampled Hessian rows, four Huber iterations and candidate selection.
Callers supply selected chart points, chart scale and smoothing parameters.
No capture segmentation, chart discovery, candidate-search driver, source
geometry, boundary construction or export is included. A historical data-extent
boundary check was later retired as an interior-curvature acceptance gate.

The numerical constants retain their historical chart-unit meaning. They are
not calibrated tolerances or a validated general recipe. This module remains
outside the production package. Synthetic checks cover median balancing,
robust fitting and the selection rule; no historical fit was rerun.

`provenance.json` records original source hashes, symbol ranges and adaptations.
