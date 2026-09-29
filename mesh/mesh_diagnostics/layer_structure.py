"""How much vertical structure does a mesh actually have?
Group nodes into (x,y) columns and count distinct z per column."""
import sys, numpy as np, gmsh
for path in sys.argv[1:]:
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
    gmsh.open(path)
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords).reshape(-1, 3)
    et, etags, _ = gmsh.model.mesh.getElements(3)
    ntet = len(etags[0]) if etags else 0
    # columns: round x,y to 1 m
    key = np.round(xyz[:, :2], 0)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    ncol = inv.max() + 1
    per = np.bincount(inv)
    print(f'=== {path} ===')
    print(f'  nodes {len(xyz)}   tets {ntet}   distinct (x,y) columns {ncol}')
    print(f'  nodes per column: min {per.min()}  median {int(np.median(per))}  '
          f'max {per.max()}  mean {per.mean():.1f}')
    print(f'  columns with >=30 nodes: {(per>=30).sum()}   >=15: {(per>=15).sum()}   ==1: {(per==1).sum()}')
    gmsh.finalize()
