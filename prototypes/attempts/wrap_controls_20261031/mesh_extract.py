"""Amendment 1 cache-safe extractor; geometry recipe otherwise unchanged."""

from .control_common import write_off


def extract(vox, path):
    mesh = vox.marching_cubes.copy()
    mesh.apply_transform(vox.transform)
    mesh.merge_vertices()
    mesh.fix_normals()
    if len(mesh.faces) > 10000000:
        raise RuntimeError("triangle cap")
    write_off(mesh, path)
    return mesh
