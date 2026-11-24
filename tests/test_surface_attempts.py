"""Synthetic regression checks for retained historical numerical methods."""

import numpy as np
import pytest
import trimesh
from scipy.integrate import quad
from scipy.interpolate import BSpline

from prototypes.attempts.exact_sections_20261107.sections import cut_mesh, fixtures
from prototypes.attempts.nonuniform_spline_20261017 import fit as nonuniform
from prototypes.attempts.spline_fitting_20261016 import fit as uniform
from prototypes.attempts.strip_withholding_20261117 import fit as strip
from prototypes.attempts.strip_withholding_20261117.penalty_audit import audit_energies


def chart(ku, kv, span=(1.0, 1.0)):
    return {"lo": [0.0, 0.0], "span": list(span), "u_knots_full": ku, "v_knots_full": kv}


def test_spatial_medians_balance_unequal_sampling_and_omit_sparse_cell():
    points = np.array(
        [[0.01, 0.01, 2.0]] * 20
        + [
            [0.06, 0.01, 3.0],
            [0.07, 0.01, 4.0],
            [0.08, 0.01, 100.0],
            [0.12, 0.01, -10.0],
        ]
    )
    anchors, bins = uniform.spatial_medians(points, np.zeros(2))
    np.testing.assert_array_equal(bins, [[0, 0], [1, 0]])
    np.testing.assert_allclose(anchors, [[0.01, 0.01, 2.0], [0.07, 0.01, 4.0]])


def test_historical_uniform_fit_retains_plane_despite_one_gross_outlier():
    uv = np.array(np.meshgrid(np.linspace(0, 1, 17), np.linspace(0, 1, 15))).reshape(2, -1).T
    A = uniform.design(uv, 4, 4, (2.0, 3.0))
    expected = 0.1 + 0.2 * uv[:, 0] - 0.3 * uv[:, 1]
    z = expected.copy()
    z[len(z) // 2] += 2.0
    coefficients = uniform.solve(A, z, uniform.penalty_rows(4, 4, (2.0, 3.0)), 1e-3)
    assert np.max(abs(A @ coefficients - expected)) < 0.002


def test_nonuniform_basis_preserves_physical_affine_derivatives():
    lo = np.array([0.0, 0.0])
    span = np.array([4.0, 2.0])
    center = span / 2
    ku = nonuniform.knots(10, 0, center, lo, span)
    kv = nonuniform.knots(8, 1, center, lo, span)
    assert np.all(np.diff(ku[3:-3]) > 0)
    u = np.array([ku[i + 1 : i + 4].mean() for i in range(len(ku) - 4)]) * span[0]
    v = np.array([kv[i + 1 : i + 4].mean() for i in range(len(kv) - 4)]) * span[1]
    coeff = (1.0 + 2.0 * u[:, None] - 3.0 * v[None, :]).ravel()
    uv = np.array([[0.0, 0.0], [0.27, 0.63], [1.0, 1.0]])
    np.testing.assert_allclose(
        nonuniform.design(uv, ku, kv, span) @ coeff,
        1 + 2 * uv[:, 0] * span[0] - 3 * uv[:, 1] * span[1],
    )
    np.testing.assert_allclose(nonuniform.design(uv, ku, kv, span, 1, 0) @ coeff, 2.0)
    np.testing.assert_allclose(nonuniform.design(uv, ku, kv, span, 0, 1) @ coeff, -3.0)


def test_strip_kernel_preserves_nonuniform_historical_objective():
    span = np.array([2.0, 3.0])
    ku = nonuniform.knots(5, 0, span / 2, np.zeros(2), span)
    kv = nonuniform.knots(5, 1, span / 2, np.zeros(2), span)
    c = chart(ku, kv, span)
    uv = np.array(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 9))).reshape(2, -1).T
    A = nonuniform.design(uv, ku, kv, span)
    z = np.sin(uv[:, 0]) * np.cos(uv[:, 1])
    pen = nonuniform.sampled_penalty(ku, kv, span)
    np.testing.assert_allclose(strip.penalty(c), pen, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        strip.fit(uv * span, z, c, pen), nonuniform.solve(A, z, pen, 1e-5), atol=1e-12
    )


def test_candidate_selection_prefers_smallest_net_then_strongest_allowed_penalty():
    records = [
        {"nu": 8, "nv": 6, "lambda": 0.001, "cv_p90": 1.08},
        {"nu": 10, "nv": 7, "lambda": 0.001, "cv_p90": 1.0},
        {"nu": 8, "nv": 6, "lambda": 0.01, "cv_p90": 1.09},
        {"nu": 8, "nv": 6, "lambda": 0.1, "cv_p90": 1.11},
    ]
    assert uniform.select_candidate(records) == records[2]
    assert nonuniform.select_candidate(records) == records[2]


def test_strip_masks_preserve_guard_and_include_every_planned_fold():
    xy = np.array(np.meshgrid(np.linspace(0, 6, 100), np.linspace(0, 4, 60))).reshape(2, -1).T
    folds = strip.folds_from_xy(xy)
    assert [f["name"] for f in folds] == ["u20", "u40", "u60", "u80", "v35", "v65"]
    for f in folds:
        core = set(f["core_ids"])
        excluded = set(f["excluded_ids"])
        training = set(f["training_ids"])
        assert core < excluded
        assert excluded.isdisjoint(training)
        assert len(excluded | training) == len(xy)
        d = abs(xy[f["training_ids"], f["axis"]] - f["center"])
        assert np.all(d > f["width"] / 2 + f["guard"])


def test_penalty_audit_matches_analytic_quadratic_and_mixed_hessian_energies():
    knots = np.r_[np.zeros(4), np.ones(4)]
    c = chart(knots, knots, (2.0, 3.0))
    squared = np.array([0.0, 0.0, 1.0 / 3.0, 1.0])
    linear = np.linspace(0, 1, 4)
    quadratic = (4 * squared[:, None] + 9 * squared[None, :]).ravel()
    mixed = (6 * linear[:, None] * linear[None, :]).ravel()
    coefficients = np.column_stack([quadratic, mixed])
    before = coefficients.copy()
    result = audit_energies(c, coefficients)
    np.testing.assert_array_equal(coefficients, before)
    np.testing.assert_allclose(result["per_span_integrated_energy"], [8 / 3, 2 / 3])
    np.testing.assert_allclose(result["historic_uniform_sampled_energy"], [8 / 3, 2 / 3])


def test_narrow_span_bending_is_missed_by_historical_samples():
    # This analytic synthetic spline has no geometry or coefficients from a capture.
    ku = np.r_[np.zeros(4), [0.205, 0.206, 0.207, 0.208, 0.209], np.ones(4)]
    kv = np.r_[np.zeros(4), np.ones(4)]
    coefficients = np.zeros((len(ku) - 4, len(kv) - 4))
    coefficients[4, :] = 1.0  # Constant in v, supported only inside the narrow u interval.
    result = audit_energies(chart(ku, kv), coefficients.ravel())
    assert result["historic_uniform_sampled_energy"][0] == 0.0
    assert result["per_span_integrated_energy"][0] > 1e7
    assert len(result["basis_support"]["u"]["intervals_without_interior_uniform_penalty_site"]) == 4
    basis = BSpline(ku, np.eye(len(ku) - 4)[4], 3).derivative(2)
    independent = (
        sum(
            quad(lambda u: basis(u) ** 2, lo, hi)[0]
            for lo, hi in zip(np.unique(ku)[:-1], np.unique(ku)[1:])
        )
        / 3
    )
    assert result["per_span_integrated_energy"][0] == pytest.approx(independent, rel=1e-12)


def test_original_analytic_sections_preserve_branches_and_exceptional_cases():
    result = fixtures()
    assert result["status"] == "passed"
    assert result["coincident_disconnected_boxes"]["closed_branches"] == 2
    assert result["junction"]["branches"] == 3
    assert result["on_plane_degeneracies"]["zero_length_face_intersections"] == 1


def test_section_owns_each_segment_once_without_welding_coincident_boxes():
    box = trimesh.creation.box(extents=[2, 2, 2])
    vertices = np.vstack([box.vertices, box.vertices])
    faces = np.vstack([box.faces, box.faces + len(box.vertices)])
    cut = cut_mesh(vertices, faces, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert cut["metadata"]["closed_branches"] == 2
    assert len(cut["xyz"]) == 2 * len(np.unique(cut["xyz"], axis=0))
    assigned = [e for branch in cut["branches"] for e in branch["edge_ids"]]
    assert sorted(assigned) == list(range(len(cut["edges"])))
    branch_owners = [
        {fid for e in b["edge_ids"] for fid in cut["owners"][e]} for b in cut["branches"]
    ]
    assert branch_owners[0].isdisjoint(branch_owners[1])
    assert cut["metadata"]["maximum_independent_endpoint_difference"] < 1e-12


def test_section_outside_mesh_reports_empty_intersection():
    box = trimesh.creation.box()
    cut = cut_mesh(box.vertices, box.faces, np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]))
    assert cut["xyz"].shape == (0, 3)
    assert cut["edges"].shape == (0, 2)
    assert cut["metadata"]["all_segments_assigned_once"]
    assert cut["metadata"]["branches"] == 0
