"""Report worst-case geometry of produced meshes: min edge length and min
tet quality (gamma). Read-only; opens each .msh in its own gmsh session.
"""
import sys
import numpy as np
import gmsh


def report(path):
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.open(path)
        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        if not etypes:
            print(f'{path}: no 3D elements')
            return
        tags = np.asarray(etags[0], dtype=np.int64)
        q_gamma = np.asarray(gmsh.model.mesh.getElementQualities(tags, 'gamma'))
        q_vol = np.asarray(gmsh.model.mesh.getElementQualities(tags, 'volume'))

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords).reshape(-1, 3)
        idx = {t: i for i, t in enumerate(np.asarray(ntags, dtype=np.int64))}
        conn = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 4)
        pts = xyz[np.vectorize(idx.get)(conn)]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        edges = np.stack([np.linalg.norm(pts[:, a] - pts[:, b], axis=1)
                          for a, b in pairs], axis=1)

        print(f'--- {path} ---')
        print(f'  tets            : {len(tags)}')
        print(f'  gamma  min/med  : {q_gamma.min():.3e} / {np.median(q_gamma):.3f}')
        print(f'  tets gamma<1e-3 : {(q_gamma < 1e-3).sum()}')
        print(f'  volume min      : {q_vol.min():.3e}')
        print(f'  edge   min      : {edges.min():.3f} m')
        print(f'  edge   max      : {edges.max():.1f} m')
        print(f'  worst aspect    : {edges.max(axis=1).max() / max(edges.min(), 1e-12):.3e}')
    finally:
        gmsh.finalize()


for p in sys.argv[1:]:
    report(p)
