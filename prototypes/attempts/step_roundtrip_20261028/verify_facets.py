"""Orientation-preserving STEP import and independent planar-facet checks.

Historical reader contract: OCP 8.0.1.1 with 54 default shape-fix parameters.
No geometry is loaded and no CAD library imported until a function is called.
"""

import numpy as np


def read_preserving_orientation(path, default_fixture_path):
    """Copy defaults from a successfully transferred small fixture, override one flag."""
    from OCP.collections import (
        DataMap_TCollection_AsciiString_TCollection_AsciiString as ParameterMap,
    )
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TCollection import TCollection_AsciiString as String

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    defaults = STEPControl_Reader()
    assert defaults.ReadFile(str(default_fixture_path)) == IFSelect_RetDone
    assert defaults.TransferRoots() == 1
    parameters = ParameterMap()
    parameters.Assign(defaults.GetShapeFixParameters())
    assert parameters.Size() == 54
    parameters.Bind(String("FixShape.FixFaceOrientationMode"), String("0"))
    reader.SetShapeFixParameters(parameters)
    assert (
        reader.GetShapeFixParameters().Find(String("FixShape.FixFaceOrientationMode")).ToCString()
        == "0"
    )
    assert reader.TransferRoots() == 1
    return reader.OneShape()


def verify(shape, vertices, faces):
    """Verify all corners/areas/normals and bijection; failures raise AssertionError."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as Map,
    )
    from OCP.collections import (
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeMap,
    )
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_REVERSED,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
    )
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from scipy.spatial import cKDTree

    vertices, faces = np.asarray(vertices), np.asarray(faces)
    expected_volume = float(trimesh.Trimesh(vertices, faces, process=False).volume)
    assert BRepCheck_Analyzer(shape, True, True).IsValid()
    counts = {}
    for label, kind in [("faces", TopAbs_FACE), ("shells", TopAbs_SHELL), ("solids", TopAbs_SOLID)]:
        explorer = TopExp_Explorer(shape, kind)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        counts[label] = count
    assert counts == {"faces": len(faces), "shells": 1, "solids": 1}, counts
    edge_map = Map()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)
    free = nonmanifold = 0
    for index in range(1, edge_map.Extent() + 1):
        count = edge_map.FindFromIndex(index).Size()
        free += count == 1
        nonmanifold += count > 2
    assert free == 0 and nonmanifold == 0
    vertex_map = ShapeMap()
    TopExp.MapShapes_s(shape, TopAbs_VERTEX, vertex_map)
    assert vertex_map.Extent() == len(vertices)
    shell = TopExp_Explorer(shape, TopAbs_SHELL)
    assert shell.Current().Closed()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = props.Mass()
    assert abs(volume - expected_volume) < 1e-7
    tree = cKDTree(vertices)
    centroids = vertices[faces].mean(1)
    centroid_tree = cKDTree(centroids)
    mapping = np.empty(len(faces), np.int64)
    corner_max, centroid_max, normal_min = 0.0, 0.0, 1.0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    index = 0
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        points, seen = [], set()
        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = TopoDS.Vertex(vertex_explorer.Current())
            vertex_id = vertex_map.FindIndex(vertex)
            if vertex_id not in seen:
                seen.add(vertex_id)
                point = BRep_Tool.Pnt_s(vertex)
                points.append([point.X(), point.Y(), point.Z()])
            vertex_explorer.Next()
        assert len(points) == 3, (index, len(points))
        points = np.array(points)
        centroid_distance, reference_id = centroid_tree.query(points.mean(0))
        distances, ids = tree.query(points)
        assert (
            centroid_distance < 1e-8
            and distances.max() < 1e-8
            and set(ids) == set(faces[reference_id])
        )
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        centre, area = props.CentreOfMass(), props.Mass()
        triangle = vertices[faces[reference_id]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        expected_area = np.linalg.norm(normal) / 2
        direction = BRepAdaptor_Surface(face).Plane().Axis().Direction()
        got = np.array([direction.X(), direction.Y(), direction.Z()])
        got *= -1 if face.Orientation() == TopAbs_REVERSED else 1
        dot = float(np.dot(got, normal / np.linalg.norm(normal)))
        assert dot > 1 - 1e-10, (index, dot)
        normal_min = min(normal_min, dot)
        assert abs(area - expected_area) < max(1e-12, expected_area * 1e-7)
        assert (
            np.linalg.norm(np.array([centre.X(), centre.Y(), centre.Z()]) - centroids[reference_id])
            < 1e-8
        )
        mapping[index] = reference_id
        corner_max = max(corner_max, float(distances.max()))
        centroid_max = max(centroid_max, float(centroid_distance))
        index += 1
        explorer.Next()
    assert len(np.unique(mapping)) == len(faces)
    return {
        "face_mapping": mapping,
        "max_corner_distance": corner_max,
        "max_centroid_distance": centroid_max,
        "minimum_normal_dot": normal_min,
        "volume_absolute_error": abs(volume - expected_volume),
        **counts,
    }
