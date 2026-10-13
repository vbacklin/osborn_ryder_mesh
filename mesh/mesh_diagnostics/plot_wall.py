"""Unroll the side walls (shoreline + grounding line + inflow) of a stacked
mesh into a flat strip: x = distance along the boundary, y = z (unscaled).
Shows how many vertical levels each wall column carries.

usage: python mesh_diagnostics/plot_wall.py OUT.png MESH.msh [MESH2.msh ...]
       [--scale 6] [--range S0 S1]   (S in km along the boundary)
"""
import sys
import numpy as np
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

args = sys.argv[1:]
scale, srange = 6.0, None
if "--scale" in args:
    i = args.index("--scale"); scale = float(args[i + 1]); del args[i:i + 2]
if "--range" in args:
    i = args.index("--range"); srange = (float(args[i + 1]), float(args[i + 2])); del args[i:i + 3]
out, paths = args[0], args[1:]
COLORS = {2: "#4a7fb5", 4: "#d08c2c"}
NAMES = {2: "Bathymetry wall", 4: "Inflow"}

fig, axes = plt.subplots(len(paths), 1, figsize=(15, 3.6 * len(paths)), squeeze=False)
for ax, path in zip(axes[:, 0], paths):
    m = meshio.read(path)
    P = m.points
    _, col = np.unique(np.round(P[:, :2], 3), axis=0, return_inverse=True)
    col = col.ravel()
    tri = np.concatenate([c.data for c in m.cells if c.type == "triangle"])
    tag = np.concatenate([d for c, d in zip(m.cells, m.cell_data["gmsh:physical"])
                          if c.type == "triangle"])
    c3 = col[tri]
    side = (c3[:, 0] == c3[:, 1]) | (c3[:, 1] == c3[:, 2]) | (c3[:, 0] == c3[:, 2])
    tri, tag, c3 = tri[side], tag[side], c3[side]

    # walk the boundary loop of columns to get arc length
    pairs = {tuple(sorted(set(r))) for r in c3.tolist()}
    nb = {}
    for a, b in pairs:
        nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    xy = np.zeros((col.max() + 1, 2)); xy[col] = P[:, :2]
    start = min(nb); order = [start]; prev = None; cur = start
    while True:
        nxt = [c for c in nb[cur] if c != prev]
        if not nxt or nxt[0] == start:
            break
        prev, cur = cur, nxt[0]; order.append(cur)
    s = {}
    acc = 0.0
    for i, c in enumerate(order):
        if i:
            acc += np.linalg.norm(xy[c] - xy[order[i - 1]])
        s[c] = acc / 1000.0
    total = acc / 1000.0

    polys, cols = [], []
    for t, tg, cc in zip(tri, tag, c3):
        xs = np.array([s.get(c, np.nan) for c in cc])
        if np.isnan(xs).any() or xs.max() - xs.min() > total / 2:
            continue
        if srange and (xs.max() < srange[0] or xs.min() > srange[1]):
            continue
        polys.append(np.column_stack([xs, P[t, 2] / scale]))
        cols.append(COLORS.get(int(tg), "#999999"))
    pc = PolyCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.15, alpha=0.9)
    ax.add_collection(pc)
    ax.autoscale_view()
    if srange:
        ax.set_xlim(*srange)
    nlev = np.bincount(col)[order]
    ax.set_title(f"{path}   —   wall columns: {len(order)}, vertical nodes per wall "
                 f"column min {nlev.min()} / median {int(np.median(nlev))} / max {nlev.max()}",
                 fontsize=9)
    ax.set_ylabel("z (m, unscaled)")
axes[-1, 0].set_xlabel("distance along the domain boundary (km)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
axes[0, 0].legend(handles, NAMES.values(), loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
