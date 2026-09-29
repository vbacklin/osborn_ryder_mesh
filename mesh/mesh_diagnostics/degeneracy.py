"""Characterise the degenerate elements: are negative volumes just orientation
convention, and how many edges are truly zero-length?
"""
import sys
import numpy as np
import gmsh

path = sys.argv[1]
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.open(path)

etypes, etags, enodes = gmsh.model.mesh.getElements(3)
tags = np.asarray(etags[0], dtype=np.int64)
vol = np.asarray(gmsh.model.mesh.getElementQualities(tags, 'volume'))

ntags, coords, _ = gmsh.model.mesh.getNodes()
xyz = np.asarray(coords).reshape(-1, 3)
idx = {t: i for i, t in enumerate(np.asarray(ntags, dtype=np.int64))}
conn = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 4)
pts = xyz[np.vectorize(idx.get)(conn)]

# Signed volume from the node ordering itself.
signed = np.einsum('ij,ij->i',
                   np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]),
                   pts[:, 3] - pts[:, 0]) / 6.0

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
edges = np.stack([np.linalg.norm(pts[:, a] - pts[:, b], axis=1)
                  for a, b in pairs], axis=1)
emin = edges.min(axis=1)

print(f'=== {path} ===')
print(f'tets                       : {len(tags)}')
print(f'signed vol  <0             : {(signed < 0).sum()}')
print(f'signed vol  >0             : {(signed > 0).sum()}')
print(f'signed vol ==0 (exactly)   : {(signed == 0).sum()}')
print(f'|vol| < 1 m^3              : {(np.abs(signed) < 1).sum()}')
print(f'gmsh vol   <0              : {(vol < 0).sum()}   (all-negative => orientation convention)')
print()
print(f'edges exactly 0.0 m        : {(edges == 0).sum()}')
print(f'tets with a 0.0 m edge     : {(emin == 0).sum()}')
for t in (1e-6, 1e-3, 0.1, 1.0, 10.0):
    print(f'tets with min edge < {t:<8g} : {(emin < t).sum()}')
print()
print('smallest 10 min-edges (m)  :', np.sort(emin)[:10])

# Where are the degenerate ones? Report z-extent of the worst offenders.
worst = np.argsort(emin)[:200]
zs = pts[worst][:, :, 2]
print(f'z range of 200 worst tets  : {zs.min():.1f} .. {zs.max():.1f}')
allz = pts[:, :, 2]
print(f'z range of whole mesh      : {allz.min():.1f} .. {allz.max():.1f}')
gmsh.finalize()
