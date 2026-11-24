"""Lossless planar FACETED_BREP serialization of an oriented triangle mesh.

Coordinates remain arbitrary SfM units. STEP's millimetre unit declaration is
an exchange convention, not a physical scale assignment. No vertex merging,
decimation, smoothing, healing or reorientation is performed here.
"""

from pathlib import Path

import numpy as np


def real(x):
    s = format(float(x), ".17g").upper()
    if "." not in s and "E" not in s:
        s += "."
    return s


def write_step(path, vertices, faces):
    path = Path(path)
    with path.open("x") as out:
        out.write(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('UNSCALED planar faceted mesh'),'2;1');\n"
        )
        out.write(
            "FILE_NAME('UNSCALED','2026-10-28T00:00:00',(''),(''),'prototype faceted serializer','', '');\n"
        )
        out.write("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")
        out.write(
            "#1=APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#2);\n"
        )
        out.write(
            "#2=APPLICATION_CONTEXT('core data for automotive mechanical design processes');\n"
        )
        out.write(
            "#3=SHAPE_DEFINITION_REPRESENTATION(#4,#10);\n#4=PRODUCT_DEFINITION_SHAPE('','',#5);\n"
        )
        out.write(
            "#5=PRODUCT_DEFINITION('design','',#6,#9);\n#6=PRODUCT_DEFINITION_FORMATION('','',#7);\n"
        )
        out.write("#7=PRODUCT('UNSCALED','UNSCALED','Arbitrary SfM coordinates',(#8));\n")
        out.write(
            "#8=PRODUCT_CONTEXT('',#2,'mechanical');\n#9=PRODUCT_DEFINITION_CONTEXT('part definition',#2,'design');\n"
        )
        out.write(
            "#10=FACETED_BREP_SHAPE_REPRESENTATION('',(#11),#13);\n#11=FACETED_BREP('',#12);\n"
        )
        out.write(
            "#13=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#17)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#14,#15,#16)) REPRESENTATION_CONTEXT('UNSCALED','Exchange units only'));\n"
        )
        out.write(
            "#14=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));\n#15=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));\n#16=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());\n"
        )
        out.write(
            "#17=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#14,'distance_accuracy_value','exchange tolerance');\n"
        )
        out.write("#18=PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#7));\n")
        base = 20
        for i, p in enumerate(vertices):
            out.write(f"#{base + i}=CARTESIAN_POINT('',({','.join(map(real, p))}));\n")
        ids = []
        start = base + len(vertices)
        for i, tri in enumerate(faces):
            p = vertices[tri]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            normal /= np.linalg.norm(normal)
            tangent = p[1] - p[0]
            tangent /= np.linalg.norm(tangent)
            k = start + 7 * i
            out.write(f"#{k}=POLY_LOOP('',({','.join('#' + str(base + int(j)) for j in tri)}));\n")
            out.write(f"#{k + 1}=FACE_OUTER_BOUND('',#{k},.T.);\n")
            out.write(f"#{k + 2}=DIRECTION('',({','.join(map(real, normal))}));\n")
            out.write(f"#{k + 3}=DIRECTION('',({','.join(map(real, tangent))}));\n")
            out.write(f"#{k + 4}=AXIS2_PLACEMENT_3D('',#{base + int(tri[0])},#{k + 2},#{k + 3});\n")
            out.write(f"#{k + 5}=PLANE('',#{k + 4});\n")
            out.write(f"#{k + 6}=FACE_SURFACE('',(#{k + 1}),#{k + 5},.T.);\n")
            ids.append(k + 6)
            if i and i % 100000 == 0:
                print("serialized faces", i, flush=True)
        out.write("#12=CLOSED_SHELL('',(")
        for i, k in enumerate(ids):
            out.write(("," if i else "") + f"#{k}")
            if i % 100 == 99:
                out.write("\n")
        out.write("));\nENDSEC;\nEND-ISO-10303-21;\n")
