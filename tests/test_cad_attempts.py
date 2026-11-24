"""Tiny synthetic checks for extracted historical behavior; no archive inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from prototypes.attempts.retriangulation_20261031.retriangulate import boundary, patch
from prototypes.attempts.return_mesh_20261028.topology import (
    close_numeric_cracks,
    split_vertex_fans,
)
from prototypes.attempts.return_mesh_20261028.winding import winding
from prototypes.attempts.step_roundtrip_20261028.compact_step import real, write_step
from prototypes.attempts.wrap_controls_20261031 import (
    dense_cache_fix_unrun,
    dense_legacy,
    mesh_extract,
)
from prototypes.attempts.wrap_controls_20261031.control_common import distance_segments


def test_fan_split_keeps_coordinates_and_faces():
    # Two triangle fans touch only at vertex zero.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    out, triangles, changes = split_vertex_fans(vertices, faces)
    assert len(out) == 6
    assert changes[0]["fans"] == 2
    np.testing.assert_array_equal(out[triangles], vertices[faces])
    assert not set(triangles[0]).intersection(triangles[1])


def test_crack_cap_refuses_large_open_triangle():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(AssertionError):
        close_numeric_cracks(vertices, np.array([[0, 1, 2]]), np.array([7]))


def test_tiny_triangular_cap_is_prescribed_and_reversed():
    vertices = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]])
    faces, ids, records = close_numeric_cracks(vertices, np.array([[0, 1, 2]]), np.array([7]))
    np.testing.assert_array_equal(faces, [[0, 1, 2], [2, 1, 0]])
    np.testing.assert_array_equal(ids, [7, 2000])
    assert records[0]["diameter"] < 2e-4


def test_winding_distinguishes_cube_inside_and_outside():
    cube = trimesh.creation.box()
    assert winding(cube.triangles, np.zeros(3)) == pytest.approx(1.0)
    assert winding(cube.triangles, np.array([2.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-14)


def test_segment_distance_includes_endpoints_and_degenerate_segment():
    segments = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    points = np.array([[0.5, 2.0, 0.0], [-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.testing.assert_allclose(distance_segments(points, segments), [2.0, 1.0, 0.0])


def voxel_stub():
    transform = np.diag([0.25, 0.25, 0.25, 1.0])
    return SimpleNamespace(marching_cubes=trimesh.creation.box(), transform=transform)


def test_legacy_cache_mutation_is_retained(tmp_path):
    voxel = voxel_stub()
    first = dense_legacy.extract(voxel, tmp_path / "one.off").vertices.copy()
    second = dense_legacy.extract(voxel, tmp_path / "two.off").vertices.copy()
    np.testing.assert_allclose(second, first * 0.25)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("extractor", [dense_cache_fix_unrun.extract, mesh_extract.extract])
def test_corrected_variants_do_not_transform_cache_twice(tmp_path, extractor):
    voxel = voxel_stub()
    original = voxel.marching_cubes.vertices.copy()
    first = extractor(voxel, tmp_path / "one.off")
    second = extractor(voxel, tmp_path / "two.off")
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(voxel.marching_cubes.vertices, original)


def test_compact_step_preserves_coordinate_precision_and_refuses_overwrite(tmp_path):
    mesh = trimesh.creation.box()
    mesh.vertices[0, 0] = -0.5000000000000001
    path = tmp_path / "mesh.step"
    write_step(path, mesh.vertices, mesh.faces)
    text = path.read_text()
    assert real(mesh.vertices[0, 0]) in text
    assert text.count("=FACE_SURFACE(") == len(mesh.faces)
    assert text.count("=CARTESIAN_POINT(") == len(mesh.vertices)
    assert "=CLOSED_SHELL(" in text
    with pytest.raises(FileExistsError):
        write_step(path, mesh.vertices, mesh.faces)


def test_planar_patch_can_pass_small_fixture_but_nonplanar_is_refused():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    triangles, _ = patch(vertices, faces)
    assert boundary(triangles) == boundary(faces)
    bent = vertices.copy()
    bent[0, 2] = 0.1
    with pytest.raises(ValueError, match="Nonplanar"):
        patch(bent, faces)


def test_delaunay_boundary_guard_keeps_a_missing_segment_failure():
    # Duplicate-position boundary IDs cannot be represented by the unconstrained
    # triangulation. The extraction must refuse that boundary change.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    faces = np.array([[0, 4, 3], [4, 5, 3], [5, 1, 2], [5, 2, 3]])
    with pytest.raises(ValueError, match="Delaunay did not preserve"):
        patch(vertices, faces)


@pytest.mark.parametrize("shell_count", [0, 2])
def test_initial_bridge_retains_one_shell_assertion(monkeypatch, shell_count):
    from prototypes.attempts.cad_closure_20261028 import closures

    calls = []
    _mock_bridge_ocp(monkeypatch, closures, shell_count, calls)
    with pytest.raises(AssertionError):
        closures.reflected_bridge(object())
    assert "solid" not in calls


def test_initial_bridge_makes_solid_even_when_free_edges_remain(monkeypatch):
    from prototypes.attempts.cad_closure_20261028 import closures

    calls = []
    _mock_bridge_ocp(monkeypatch, closures, 1, calls)
    shape, audit = closures.reflected_bridge(object())
    assert shape == "solid"
    assert audit["free_edges"] == 2
    assert calls == ["perform", "solid", "orient"]
    # The clipped branch retains its distinct conditional completion.
    monkeypatch.setattr(closures, "finish", lambda sewing: "conditional")
    assert closures.reflected_bridge(object(), skip_centreline=True) == "conditional"


def _mock_bridge_ocp(monkeypatch, closures, shell_count, calls):
    import sys

    sewing = SimpleNamespace(
        Add=lambda shape: None,
        Perform=lambda: calls.append("perform"),
        SewedShape=lambda: "sewn",
        NbFreeEdges=lambda: 2,
        NbMultipleEdges=lambda: 0,
    )

    def make_solid(shell):
        calls.append("solid")
        return SimpleNamespace(Solid=lambda: "solid")

    modules = {
        "OCP": SimpleNamespace(__path__=[]),
        "OCP.BRepBuilderAPI": SimpleNamespace(
            BRepBuilderAPI_Sewing=lambda tolerance: sewing, BRepBuilderAPI_MakeSolid=make_solid
        ),
        "OCP.BRepFill": SimpleNamespace(BRepFill=SimpleNamespace()),
        "OCP.BRepLib": SimpleNamespace(
            BRepLib=SimpleNamespace(OrientClosedSolid_s=lambda shape: calls.append("orient"))
        ),
        "OCP.TopAbs": SimpleNamespace(TopAbs_SHELL="shell"),
        "OCP.TopoDS": SimpleNamespace(TopoDS=SimpleNamespace(Shell=lambda shape: shape)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(closures, "free_edges", lambda shape: [])
    monkeypatch.setattr(closures, "mirror", lambda shape: shape)
    monkeypatch.setattr(closures, "items", lambda *args: ["shell"] * shell_count)


def _mock_orientation_control(monkeypatch, inverted):
    from prototypes.attempts.step_roundtrip_20261028 import orientation_control

    monkeypatch.setattr(
        orientation_control.trimesh.creation, "icosphere", lambda **kwargs: trimesh.creation.box()
    )
    monkeypatch.setattr(
        orientation_control, "read_preserving_orientation", lambda path, fixture: path.name
    )
    monkeypatch.setattr(
        orientation_control,
        "check_orientation",
        lambda shape, mesh: (
            inverted
            if shape == "inverted.step"
            else {"BRep_valid": True, "normal_mismatches": 0, "passes": True}
        ),
    )
    return orientation_control


def test_orientation_control_requires_a_measured_normal_mismatch(monkeypatch, tmp_path):
    control = _mock_orientation_control(
        monkeypatch, {"BRep_valid": False, "normal_mismatches": 0, "passes": False}
    )
    with pytest.raises(AssertionError):
        control.run(tmp_path / "invalid_without_normal_mismatch")
    control = _mock_orientation_control(
        monkeypatch, {"BRep_valid": False, "normal_mismatches": 1, "passes": False}
    )
    assert control.run(tmp_path / "inverted")["inverted_face"]["normal_mismatches"] == 1


@pytest.mark.parametrize("failure", ["ReadFile", "TransferRoots", "default parameters", "topology"])
def test_orientation_control_does_not_swallow_other_failures(monkeypatch, tmp_path, failure):
    control = _mock_orientation_control(
        monkeypatch, {"BRep_valid": False, "normal_mismatches": 1, "passes": False}
    )
    original = control.read_preserving_orientation

    def failed_reader(path, fixture):
        if path.name == "inverted.step":
            raise AssertionError(failure)
        return original(path, fixture)

    if failure == "topology":
        original_check = control.check_orientation

        def failed_check(shape, mesh):
            if shape == "inverted.step":
                raise AssertionError(failure)
            return original_check(shape, mesh)

        monkeypatch.setattr(control, "check_orientation", failed_check)
    else:
        monkeypatch.setattr(control, "read_preserving_orientation", failed_reader)
    with pytest.raises(AssertionError, match=failure):
        control.run(tmp_path / "failed")
