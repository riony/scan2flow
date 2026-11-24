"""Exact binary-STL encoding check; canonical doubles may differ from float32."""

import numpy as np


def oriented_key(face):
    a, b, c = map(int, face)
    return min((a, b, c), (b, c, a), (c, a, b))


def verify_reader_encoding(source, reader):
    """Compare supplied meshes without writing or replacing historical evidence."""
    encoded = source.vertices.astype(np.float32).astype(np.float64)
    lookup = {tuple(p): i for i, p in enumerate(reader.vertices)}
    assert len(lookup) == len(reader.vertices) == len(source.vertices)
    assert all(tuple(p) in lookup for p in encoded)
    mapping = np.array([lookup[tuple(p)] for p in encoded])
    assert len(np.unique(mapping)) == len(source.vertices)
    reader_faces = {oriented_key(f) for f in reader.faces}
    assert all(oriented_key(mapping[f]) in reader_faces for f in source.faces)
    delta = np.linalg.norm(reader.vertices[mapping] - source.vertices, axis=1)
    assert np.array_equal(reader.vertices[mapping], encoded)
    audit = {
        "STL_encoding": "IEEE754 float32 per stored vertex component",
        "all_reader_vertices_equal_encoded_source_exactly": True,
        "bijective_vertex_mapping": True,
        "all_oriented_donor_faces_preserved": True,
        "canonical_to_reader_max_m": float(delta.max()),
        "reader_equals_canonical_exactly": bool(
            np.array_equal(reader.vertices[mapping], source.vertices)
        ),
    }
    return audit


def donor_matches_canonical_reader_before_encoding_fix(source, reader):
    """Retained failed verifier: compares canonical doubles to reader values."""

    def geometric_key(points):
        return tuple(sorted(tuple(p) for p in points))

    reader_keys = {geometric_key(t) for t in reader.triangles}
    return all(geometric_key(t) in reader_keys for t in source.triangles)
