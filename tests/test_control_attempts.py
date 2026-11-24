"""Synthetic log mutations only; these tests are not meshing/CFD evidence."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import trimesh

from prototypes.attempts.constrained_triangulation_20261105.checks import (
    audit_seam,
    reconcile_surface_checks,
    topology,
)
from prototypes.attempts.constrained_triangulation_20261105.metrics import boundary, quality
from prototypes.attempts.damaged_prior_20261105.charts import cap_ownership_conflicts, inspect_chart
from prototypes.attempts.damaged_prior_20261105.stl_encoding import (
    donor_matches_canonical_reader_before_encoding_fix,
    verify_reader_encoding,
)
from prototypes.attempts.hybrid_qualification_20261105.transfer import transfer
from prototypes.attempts.matched_cfd_20261106.cellset import (
    cell_set,
    cell_set_before_compact_zero_fix,
)
from prototypes.attempts.matched_cfd_20261106.evaluate import (
    complete_log,
    force_metrics,
    parse_mesh,
)
from prototypes.attempts.matched_cfd_20261106.gate import evaluate_check, evaluate_diagnostic

QUALITY = """Checking faces in error :
    non-orthogonality > 65 degrees : 0
    faces with face pyramid volume < 1e-13 : 0
    faces with face-decomposition tet quality < 1e-15 : 0
    faces with concavity > 80 degrees : 0
    faces with skewness > 4 (internal) or 4 (boundary) : 0
    faces with interpolation weights (0..1) < 0.02 : 0
    faces with volume ratio of neighbour cells < 0.01 : 0
    faces with face twist < 0.02 : 0
    faces on cells with determinant < 0.001 : 0
"""


def fixture(calibration=False):
    argv = ["checkMesh", "-case", "/tmp/example", "-constant", "-allTopology", "-meshQuality"]
    if not calibration:
        argv.append("-parallel")
    text = f"""Version: 2412
Build: OPENFOAM=2412
Exec : {" ".join(argv)}
Case : /tmp/example
nProcs : {1 if calibration else 24}
Enabling all (cell, face, edge, point) topology checks.
Enabling user-defined geometry checks.
Time = constant
points: 1000
faces: 2400
cells: 700
Number of regions: 1 (OK).
Checking patch topology for multiply connected surfaces...
Patch Faces Points Surface topology Bounding box
inlet 10 20 ok (non-closed singly connected)
outlet 10 20 ok (non-closed singly connected)
ground 10 20 ok (non-closed singly connected)
top 10 20 ok (non-closed singly connected)
sides 20 30 ok (non-closed singly connected)
car 40 60 ok (closed singly connected)
Checking geometry...
Overall domain bounding box (-3.1 -3.5 0) (9.4 3.5 3.5)
Min volume = 1e-9. Max volume = .01. Total volume = 306.15. Cell volumes OK.
Mesh non-orthogonality Max: 50 average: 8
Max skewness = 2.5 OK.
{QUALITY}Mesh OK.
End
"""
    return text, argv


class GateTests(unittest.TestCase):
    def call(self, text, argv, *, calibration=False, diagnostic=False, rc=0):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "check.log"
            log.write_text(text)
            record = {"name": "checkMesh", "argv": argv, "log": str(log), "returncode": rc}
            return (evaluate_diagnostic if diagnostic else evaluate_check)(log, record, calibration)

    def test_complete_matched_and_serial_calibration(self):
        t, a = fixture()
        self.assertTrue(self.call(t, a)["passed"])
        t, a = fixture(True)
        t = t.replace("(-3.1 -3.5 0)", "(-3.1 0 0)").replace("sides 20", "symmetry 20")
        self.assertTrue(self.call(t, a, calibration=True)["passed"])
        self.assertFalse(self.call(t, a)["passed"])

    def test_missing_invocation_output_or_end_fails(self):
        t, a = fixture()
        for remove in (
            "-meshQuality",
            "Enabling user-defined geometry checks.",
            "faces with face twist < 0.02 : 0",
            "End",
            "Mesh OK.",
            "Time = constant",
            "Max skewness = 2.5 OK.",
        ):
            with self.subTest(remove=remove):
                self.assertFalse(self.call(t.replace(remove, ""), a)["passed"])
        self.assertFalse(self.call(t, [x for x in a if x != "-meshQuality"])["passed"])
        self.assertFalse(self.call(t, a, rc=1)["passed"])

    def test_threshold_count_finite_and_duplicate_failures(self):
        t, a = fixture()
        for old, new in [
            ("face twist < 0.02 : 0", "face twist < 0.02 : 1"),
            ("weights (0..1) < 0.02", "weights (0..1) < 0.05"),
            ("Max: 50", "Max: 66"),
            ("= 2.5 OK.", "= 4.01 OK."),
            ("regions: 1", "regions: 2"),
            ("Total volume = 306.15", "Total volume = 0"),
            ("Min volume = 1e-9", "Min volume = nan"),
        ]:
            with self.subTest(new=new):
                self.assertFalse(self.call(t.replace(old, new), a)["passed"])
        self.assertFalse(self.call(t.replace("Mesh OK.", QUALITY + "Mesh OK."), a)["passed"])
        self.assertFalse(self.call(t + "\nTime = 500\nMesh OK.\n", a)["passed"])

    def test_diagnostic_failures_do_not_become_primary_acceptance(self):
        t, a = fixture(True)
        a = [x.replace("-meshQuality", "-allGeometry") for x in a]
        t = (
            t.replace("-meshQuality", "-allGeometry")
            .replace("Enabling user-defined geometry checks.", "Enabling all geometry checks.")
            .replace(QUALITY, "")
            .replace(
                "Mesh OK.",
                "***Concave cells (using face planes) found, number of cells: 28449\nFailed 1 mesh checks.",
            )
        )
        result = self.call(t, a, calibration=True, diagnostic=True)
        self.assertTrue(result["completed"])
        self.assertEqual(result["concave_cells"], 28449)
        self.assertEqual(result["reported_failed_checks"], 1)
        self.assertNotIn("passed", result)
        self.assertFalse(self.call(t, a, calibration=True)["passed"])
        for remove in ("End", "Failed 1 mesh checks.", "number of cells: 28449"):
            self.assertFalse(
                self.call(t.replace(remove, ""), a, calibration=True, diagnostic=True)["completed"]
            )


def mesh_fixture():
    return """Version: 2412
Build: OPENFOAM=2412
nProcs : 24
points: 1000
faces: 2400
cells: 700
Number of regions: 1 (OK).
inlet 10 20 ok (non-closed singly connected)
outlet 10 20 ok (non-closed singly connected)
ground 10 20 ok (non-closed singly connected)
top 10 20 ok (non-closed singly connected)
sides 20 30 ok (non-closed singly connected)
car 40 60 ok (closed singly connected)
Overall domain bounding box (-3.1 -3.5 0) (9.4 3.5 3.5)
Min volume = 1e-9. Max volume = .01. Total volume = 306.15. Cell volumes OK.
Mesh non-orthogonality Max: 50 average: 8
Max skewness = 2.5 OK.
Mesh OK.
End
Finalising parallel run
"""


def force_fixture():
    header = """# dragDir : (1 0 0)
# liftDir : (0 0 1)
# CofR : (.5 0 .15)
# magUInf : 40
# lRef : .288
# Aref : .11510
# Time Cd Cd(f) Cd(r) Cl
"""
    return header + "".join(f"{i} .3 .15 .15 .27\n" for i in range(1, 3001))


class EvidenceParserTests(unittest.TestCase):
    def test_complete_mesh_and_missing_metrics(self):
        text = mesh_fixture()
        self.assertTrue(parse_mesh(text)["passed"])
        for removed in ["Max skewness = 2.5 OK.", "Min volume = 1e-9.", "End", "nProcs : 24"]:
            with self.subTest(removed=removed):
                self.assertFalse(parse_mesh(text.replace(removed, ""))["passed"])
        self.assertFalse(parse_mesh(text, returncode=1)["passed"])

    def test_wrong_version_limits_topology_and_domain(self):
        text = mesh_fixture()
        for old, new in [
            ("2412", "2512"),
            ("Max: 50", "Max: 66"),
            ("= 2.5 OK.", "= 4.01 OK."),
            ("regions: 1", "regions: 2"),
            ("car 40", "car 0"),
            ("sides 20", "symmetry 20"),
            ("(-3.1 -3.5 0)", "(-3.1 0 0)"),
            ("cells: 700", "cells: 8000001"),
        ]:
            with self.subTest(new=new):
                self.assertFalse(parse_mesh(text.replace(old, new))["passed"])
        self.assertFalse(parse_mesh(text + "\n*** Failed 1 mesh checks.\n")["passed"])

    def test_fatal_and_nonfinite_without_banner_false_positive(self):
        text = (
            mesh_fixture() + "Floating point exception trapping enabled (FOAM_SIGFPE).\nUinf 40\n"
        )
        self.assertTrue(all(complete_log(text, 0, 24).values()))
        for bad in [
            "FOAM FATAL ERROR",
            "Floating point exception",
            "residual = nan",
            "force = -inf",
        ]:
            self.assertFalse(all(complete_log(text + bad, 0, 24).values()))

    def test_force_columns_windows_and_area(self):
        result = force_metrics(force_fixture())
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["means_2500_3000"]["Cd"], 0.3)
        self.assertAlmostEqual(result["means_2500_3000"]["Cl"], 0.27)
        self.assertFalse(
            force_metrics(force_fixture().replace("# Aref : .11510", "# Aref : .05755"))["passed"]
        )
        self.assertFalse(
            force_metrics(force_fixture().replace("2500 .3 .15 .15 .27\n", ""))["passed"]
        )

    def test_force_duplicates_nonfinite_nonsteady(self):
        text = force_fixture()
        with self.assertRaises(AssertionError):
            force_metrics(text + "3000 .3 .15 .15 .27\n")
        with self.assertRaises(AssertionError):
            force_metrics(text.replace("3000 .3", "3000 nan"))
        for i in range(2751, 3001):
            text = text.replace(f"{i} .3 .15 .15 .27", f"{i} .31 .15 .16 .28")
        self.assertFalse(force_metrics(text)["passed"])


def square_mesh():
    return trimesh.Trimesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )


def test_patch_topology_and_atomic_seam():
    mesh = square_mesh()
    result = topology(mesh.vertices, mesh.faces)
    assert result["components"] == 1 and result["boundary_components"] == 1
    assert result["euler"] == 1 and result["boundary_degree_two"]
    assert boundary(mesh.faces) == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert audit_seam(mesh.faces, np.array([[0, 2]]))["all_two_opposite"]
    changed = mesh.faces.copy()
    changed[1] = changed[1, ::-1]
    assert not audit_seam(changed, np.array([[0, 2]]))["all_two_opposite"]
    equilateral = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    assert quality(equilateral, np.array([[0, 1, 2]])).item() == pytest.approx(1.0)


def test_contact_reconciliation_cannot_waive_structure_or_missing_evidence():
    sf = {
        "process_success": True,
        "complete": True,
        "fatal_error": False,
        "illegal_triangles": 0,
        "closed": True,
        "components": 1,
        "normal_zones": 1,
        "min_triangle_quality": 0.1,
        "intersection_check_present": True,
    }
    exact = {"canonical": {"intersecting_pairs": 0}, "reader": {"intersecting_pairs": 0}}
    hits = {
        "reported": 2,
        "exact_disjoint": 2,
        "exact_segment_triangle_intersections": 0,
        "degenerate_pairs": 0,
    }
    assert all(reconcile_surface_checks(sf, exact, 2, hits).values())
    assert not all(reconcile_surface_checks(sf | {"closed": False}, exact, 2, hits).values())
    assert not all(reconcile_surface_checks(sf, exact, 2, {}).values())
    assert not all(reconcile_surface_checks(sf, exact, 2, hits | {"exact_disjoint": 1}).values())


def test_owned_cap_is_rejected():
    mesh = square_mesh()
    assert cap_ownership_conflicts(mesh.faces, np.array([[2, 1, 0]])) == [0]
    assert cap_ownership_conflicts(mesh.faces, np.array([[0, 1, 3]])) == []


def test_explicit_chart():
    mesh = square_mesh()
    pytest.importorskip("shapely")
    result = inspect_chart(
        mesh,
        name="plane",
        donors=[0, 1],
        loop=[0, 1, 2, 3],
        axes=[0, 1],
        normal=[0, 0, 1],
        planar=True,
    )
    assert result["record"]["domain_area_m2"] == 1
    assert result["record"]["missing_domains"] == 0
    assert (0, 2) in result["constraints"]
    with pytest.raises(AssertionError):
        inspect_chart(
            mesh,
            name="crossed",
            donors=[0, 1],
            loop=[0, 2, 1, 3],
            axes=[0, 1],
            normal=[0, 0, 1],
            planar=True,
        )


def test_stl_encoding_fix_preserves_oriented_faces():
    source = trimesh.Trimesh([[0.1, 0, 0], [1.1, 0, 0], [0.1, 1.1, 0]], [[0, 1, 2]], process=False)
    reader = trimesh.Trimesh(
        source.vertices.astype(np.float32).astype(np.float64), source.faces, process=False
    )
    assert not donor_matches_canonical_reader_before_encoding_fix(source, reader)
    audit = verify_reader_encoding(source, reader)
    assert audit["all_reader_vertices_equal_encoded_source_exactly"]
    assert not audit["reader_equals_canonical_exactly"]
    flipped = trimesh.Trimesh(reader.vertices, reader.faces[:, ::-1], process=False)
    with pytest.raises(AssertionError):
        verify_reader_encoding(source, flipped)


def test_retained_feature_vertices_do_not_move_in_transfer():
    mesh = trimesh.Trimesh(
        [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        process=False,
    )
    segments = mesh.vertices[mesh.edges_unique]
    targets = mesh.vertices + 0.001
    result = transfer(mesh, segments, targets, targets, np.zeros(len(mesh.vertices)))
    assert np.array_equal(result["hybrid"].vertices, mesh.vertices)
    assert np.array_equal(result["hybrid"].faces, mesh.faces)
    assert np.all(result["vertex_weight"] == 0)
    assert np.all(result["face_transfer_class"] == 0)


CELLSET_HEADER = (
    b'FoamFile\n{\n format binary;\n class cellSet;\n arch "LSB;label=32;scalar=64";\n}\n'
)


def test_compact_cellset_fix_and_historical_failure(tmp_path):
    path = tmp_path / "exampleSet"
    path.write_bytes(CELLSET_HEADER + b"0()\n")
    with pytest.raises(AssertionError):
        cell_set_before_compact_zero_fix(path)
    assert cell_set(path).tolist() == []
    for payload in [b"0\n()\n", b"0\n(\n)\n", b"0 \n ( \n )\n"]:
        path.write_bytes(CELLSET_HEADER + payload)
        assert cell_set(path).tolist() == []
    path.write_bytes(CELLSET_HEADER + b"2\n(7 3)\n")
    assert cell_set(path).tolist() == [3, 7]
    for payload in [b"1()\n", b"2\n(7)\n", b"0(\n", b"2(7 7)\n"]:
        path.write_bytes(CELLSET_HEADER + payload)
        with pytest.raises(AssertionError):
            cell_set(path)


def test_full_locked_graph_requires_distinct_patches():
    from prototypes.attempts.constrained_triangulation_20261105.checks import audit_locked_graph

    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    edges = np.unique(
        np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0,
    )
    report = audit_locked_graph(faces, np.arange(4), edges, faces)
    assert report["all_two_opposite_distinct_patch"]
    assert report["all_mesh_edges_two_faces"]
    assert report["all_global_ids_retained"]
    assert not audit_locked_graph(faces, np.zeros(4), edges, faces)[
        "all_two_opposite_distinct_patch"
    ]
