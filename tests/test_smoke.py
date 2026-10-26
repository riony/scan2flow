import json
from pathlib import Path
import numpy as np
import trimesh
from scan2foam.cli import main, inspect_mesh

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_cli_fixtures_are_open_and_unchanged(capsys):
    for name in ("open_shell.ply", "rough_patch.ply"):
        path = FIXTURES / name
        before = path.read_bytes()
        assert main(["inspect", str(path)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert 5000 <= result["faces"] <= 50000
        assert result["boundary_edges"] > 0
        assert result["watertight"] is False
        assert path.read_bytes() == before


def test_topology_and_missing_input(tmp_path, capsys):
    box = trimesh.creation.box()
    path = tmp_path / "box.stl"
    box.export(path)
    result = inspect_mesh(path)
    assert result["watertight"] and result["boundary_edges"] == 0
    assert result["euler_number"] == 2 and result["boundary_loops"] == 0
    # Three faces share one edge: boundary components must not be called loops.
    fan = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],
                          faces=[[0,1,2],[1,0,3],[0,1,4]], process=False)
    fan.export(path.with_suffix('.ply'))
    result = inspect_mesh(path.with_suffix('.ply'))
    assert result['nonmanifold_edges'] == 1
    assert result['boundary_branched_components'] == 1
    assert result['boundary_loops'] == 0
    assert main(['inspect', str(tmp_path / 'missing.ply')]) == 2
    assert 'scan2foam:' in capsys.readouterr().err
