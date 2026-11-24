"""Nonmoving topology operations from a failed closure-quality attempt."""

from collections import defaultdict

import numpy as np


def split_vertex_fans(vertices, faces):
    """Duplicate multi-fan vertices without moving positions or changing face order."""
    vertices = np.asarray(vertices).tolist()
    faces = np.asarray(faces).copy()
    incident = defaultdict(list)
    for index, triangle in enumerate(faces):
        for vertex in triangle:
            incident[int(vertex)].append(index)
    changes = []
    for vertex, face_ids in incident.items():
        neighbors = defaultdict(list)
        for index in face_ids:
            for other in faces[index]:
                if other != vertex:
                    neighbors[int(other)].append(index)
        adjacency = {index: set() for index in face_ids}
        for row in neighbors.values():
            for index in row:
                adjacency[index].update(row)
        pending = set(face_ids)
        groups = []
        while pending:
            todo = [pending.pop()]
            group = []
            while todo:
                index = todo.pop()
                group.append(index)
                new = adjacency[index] & pending
                pending -= new
                todo.extend(new)
            groups.append(group)
        if len(groups) > 1:
            changes.append(
                {
                    "vertex": vertex,
                    "fans": len(groups),
                    "fan_faces": [len(group) for group in groups],
                }
            )
            for group in groups[1:]:
                new = len(vertices)
                vertices.append(vertices[vertex])
                block = faces[group].copy()
                block[block == vertex] = new
                faces[group] = block
    return np.array(vertices), faces, changes


def close_numeric_cracks(vertices, faces, face_ids):
    """Cap only triangular loops meeting the original diameter/area guards."""
    vertices, faces, face_ids = map(np.asarray, (vertices, faces, face_ids))
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, count = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    boundary = edges[count[inverse] == 1]
    forward = {int(edge[0]): int(edge[1]) for edge in boundary}
    assert len(forward) == len(boundary)
    caps, records = [], []
    while forward:
        start = next(iter(forward))
        loop = [start]
        point = forward.pop(start)
        while point != start:
            loop.append(point)
            point = forward.pop(point)
        assert len(loop) == 3, loop
        xyz = vertices[loop]
        diameter = float(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2).max())
        area = float(np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) / 2)
        assert diameter < 2e-4 and area < 1e-8, (diameter, area)
        records.append({"vertices": loop, "diameter": diameter, "area": area})
        caps.append(loop[::-1])
    if not caps:
        return faces.copy(), face_ids.copy(), records
    return np.vstack((faces, np.array(caps))), np.r_[face_ids, np.full(len(caps), 2000)], records
