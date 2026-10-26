"""v0.1 exploratory dictionary writers. Fixed wrapping assumptions and no layer generation; migrate with domain and parser tests. No solver invocation."""


def _block_dict(bmin, bmax, n) -> str:
    v = lambda a: f"({a[0]:.4f} {a[1]:.4f} {a[2]:.4f})"
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
 ({bmin[0]:.4f} {bmin[1]:.4f} {bmin[2]:.4f}) ({bmax[0]:.4f} {bmin[1]:.4f} {bmin[2]:.4f})
 ({bmax[0]:.4f} {bmax[1]:.4f} {bmin[2]:.4f}) ({bmin[0]:.4f} {bmax[1]:.4f} {bmin[2]:.4f})
 ({bmin[0]:.4f} {bmin[1]:.4f} {bmax[2]:.4f}) ({bmax[0]:.4f} {bmin[1]:.4f} {bmax[2]:.4f})
 ({bmax[0]:.4f} {bmax[1]:.4f} {bmax[2]:.4f}) ({bmin[0]:.4f} {bmax[1]:.4f} {bmax[2]:.4f})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( domain {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(0 4 7 3)(1 2 6 5)); }} );
mergePatchPairs ();
"""

def _snappy_wrap_dict(level, seed, feature_snap) -> str:
    fs = "true" if feature_snap else "false"
    feat = '{ file "car.eMesh"; level %d; }' % level if feature_snap else ""
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry {{ car {{ type triSurfaceMesh; file "car.stl"; }} }}
castellatedMeshControls
{{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 0;
    nCellsBetweenLevels 3; resolveFeatureAngle 30;
    features ( {feat} );
    refinementSurfaces {{ car {{ level ({level} {level}); patchInfo {{ type wall; }} }} }}
    refinementRegions {{ }}
    locationInMesh ({seed[0]:.4f} {seed[1]:.4f} {seed[2]:.4f});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8;
    nFeatureSnapIter 12; implicitFeatureSnap false; explicitFeatureSnap {fs};
    multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2;
    finalLayerThickness 0.3; minThickness 0.1; nGrow 0; }}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01;
    minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6;
"""

def _sfe_dict() -> str:
    return """FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
car.stl { extractionMethod extractFromSurface;
  extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
"""

def _decompose_dict(n) -> str:
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains {n}; method scotch;
"""
