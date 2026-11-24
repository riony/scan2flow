# Strip withholding and saved-coefficient penalty audit

Historical date: 17 November 2026. Imported as a prototype on 25 November 2026.

Neither interpolant in the original comparison earned promotion. A withheld
strip produced a large oscillation in the historical thin-plate fit; the
normal-assisted alternative did not qualify on that fold. A read-only audit of
the saved coefficients then identified severe under-integration of the 45 by 35
uniformly sampled bending penalty. This diagnosed the saved shape, not physical
error or the success of a future correction.

`fit.py` preserves the sampled penalty and four-step robust fitting equations,
plus XY-only strip masks, bracketing rules and a convex-hull scaffold grid.
Optional normal/constraint rows must already be supplied by the caller. No
observation lookup, saved normal predictions, camera data, image arrays,
raycasts, scoring or experiment orchestration is included. Strip widths, guards
and eligibility counts retain their historical chart-unit meaning; they are
not calibrated universal defaults.

`penalty_audit.py` evaluates supplied coefficient vectors only. Four
Gauss-Legendre points in each distinct knot span integrate the squared bicubic
Hessian polynomials up to roundoff. Both reported energies use physical
derivatives and normalized chart area, with division by three matching the
historical stacked Hessian rows. The module returns energies and empty-span
diagnostics. It does not produce a corrected fitting matrix, optimize, refit,
choose a candidate or promote a surface.

The historical proposal to refit with exact integration was not run. A synthetic
narrow-span regression demonstrates that the retained sampling can report zero
energy for a bent spline; an independent quadrature calculation checks the
audit. Analytic quadratic and mixed-derivative fixtures check normalization.
These are new synthetic tests, not new evidence about captured geometry.

`provenance.json` records original source hashes, symbol ranges and adaptations.
