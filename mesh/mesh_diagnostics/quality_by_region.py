"""Read-only: tet quality (gmsh gamma) binned by plan-view distance to each
boundary feature of a stacked sigma mesh.

Features (derived from the physical groups, there is no grounding-line tag):
  grounding  - side-wall columns under ice   (Bathymetry wall, ice on top)
  shore      - side-wall columns in open water
  inflow     - columns on the Inflow face
  icefront   - columns touching both Ice and Water Surface top triangles

Alongside gamma it reports the two things that drive it in a sigma mesh:
column thickness and layer slope (|grad z| of the prism's top face).
z in the file is scaled by SCALE; thickness is reported unscaled.

usage: python mesh_diagnostics/quality_by_region.py MESH.msh [h] [SCALE]
"""
import sys
import numpy as np
import gmsh

path = sys.argv[1]
H = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
ICE, BATH, INFLOW, WATER = 1, 2, 4, 5

gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.open(path)
ntags, coords, _ = gmsh.model.mesh.getNodes()
xyz = np.asarray(coords).reshape(-1, 3)
node_idx = np.zeros(int(ntags.max()) + 1, dtype=np.int64)
node_idx[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))

et, etags, enodes = gmsh.model.mesh.getElements(3)
ttags = np.asarray(etags[0], dtype=np.int64)
tets = node_idx[np.asarray(enodes[0], dtype=np.int64).reshape(-1, 4)]
gamma = np.asarray(gmsh.model.mesh.getElementQualities(ttags, 'gamma'))

tris_by_tag = {}
for dim, ptag in gmsh.model.getPhysicalGroups(2):
    tri = []
    for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
        _, _, en = gmsh.model.mesh.getElements(2, ent)
        if en:
            tri.append(np.asarray(en[0], dtype=np.int64).reshape(-1, 3))
    tris_by_tag[ptag] = node_idx[np.concatenate(tri)]
gmsh.finalize()

# --- columns -----------------------------------------------------------------
_, col, = np.unique(np.round(xyz[:, :2], 3), axis=0, return_inverse=True)[:2]
col = col.ravel()
ncol = col.max() + 1
cxy = np.zeros((ncol, 2)); cxy[col] = xyz[:, :2]
ztop = np.full(ncol, -np.inf); zbot = np.full(ncol, np.inf)
np.maximum.at(ztop, col, xyz[:, 2]); np.minimum.at(zbot, col, xyz[:, 2])
thick = (ztop - zbot) / SCALE

def side_cols(tris):
    c = col[tris]
    side = (c[:, 0] == c[:, 1]) | (c[:, 0] == c[:, 2]) | (c[:, 1] == c[:, 2])
    return np.unique(c[side])

def top_cols(tris):
    c = col[tris]
    flat = (c[:, 0] != c[:, 1]) & (c[:, 0] != c[:, 2]) & (c[:, 1] != c[:, 2])
    return np.unique(c[flat])

ice_top = np.zeros(ncol, bool); ice_top[top_cols(tris_by_tag[ICE])] = True
water_top = np.zeros(ncol, bool); water_top[top_cols(tris_by_tag[WATER])] = True
wall = side_cols(tris_by_tag[BATH])
features = {
    'grounding': wall[ice_top[wall] & ~water_top[wall]],
    'shore': wall[~ice_top[wall]],
    'inflow': side_cols(tris_by_tag[INFLOW]),
    'icefront': np.flatnonzero(ice_top & water_top),
}

# --- per-tet plan-view descriptors -------------------------------------------
tc = col[tets]                                  # 4 column ids per tet
cen = cxy[tc].mean(axis=1)
tet_thick = thick[tc].min(axis=1)

# layer slope: fit a plane through the top of the tet's (up to 3) columns
def slope_of(cols3, z):
    p = cxy[cols3]; zz = z[cols3] / SCALE
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dz1 = zz[:, 1] - zz[:, 0]; dz2 = zz[:, 2] - zz[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        gx = (dz1 * b[:, 1] - dz2 * a[:, 1]) / det
        gy = (a[:, 0] * dz2 - b[:, 0] * dz1) / det
    return np.hypot(gx, gy)

# the 3 distinct columns of each tet's prism
srt = np.sort(tc, axis=1)
prism = np.stack([srt[:, 0], np.where(srt[:, 1] != srt[:, 0], srt[:, 1], srt[:, 2]),
                  srt[:, 3]], axis=1)
slope = np.maximum(slope_of(prism, ztop), slope_of(prism, zbot))

def min_dist(P, Q, chunk=4096):
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        d = ((P[i:i + chunk, None, :] - Q[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out

print(f'=== {path}   tets {len(tets)}   columns {ncol}   h={H:g}  scale={SCALE:g}')
print(f'feature columns: ' + ', '.join(f'{k} {len(v)}' for k, v in features.items()))

def row(label, m):
    if not m.any():
        return
    g = gamma[m]
    print(f'  {label:<14} n={m.sum():7d}  gamma med {np.median(g):.3f}  p5 {np.percentile(g, 5):.3f}'
          f'  min {g.min():.1e}  <0.01 {100 * (g < 0.01).mean():5.1f}%'
          f'  | thick med {np.median(tet_thick[m]):6.1f} m  slope med {np.median(slope[m]):.3f}')

row('ALL', np.ones(len(tets), bool))
dist = {k: min_dist(cen, cxy[v]) for k, v in features.items() if len(v)}
nearest = np.min(np.stack(list(dist.values())), axis=0)
row('interior >4h', nearest >= 4 * H)
bands = [(0, 1), (1, 2), (2, 4)]
for k, d in dist.items():
    print(f'-- distance to {k}')
    for lo, hi in bands:
        row(f'[{lo}h,{hi}h)', (d >= lo * H) & (d < hi * H))

print('-- worst 1% of tets: which feature is nearest, and how far')
worst = gamma <= np.percentile(gamma, 1)
names = list(dist)
which = np.argmin(np.stack([dist[k] for k in names]), axis=0)
for i, k in enumerate(names):
    m = worst & (which == i)
    if m.any():
        print(f'  {k:<10} {m.sum():6d} tets ({100 * m.sum() / worst.sum():4.1f}%)  '
              f'median dist {np.median(dist[k][m]) / H:.2f}h  thick med {np.median(tet_thick[m]):.1f} m')
m = worst & (nearest >= 4 * H)
print(f'  interior   {m.sum():6d} tets ({100 * m.sum() / worst.sum():4.1f}%)')

print('-- gamma vs column thickness (all tets)')
for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 150), (150, 400), (400, 1e9)]:
    row(f'thick {lo}-{hi if hi < 1e9 else "inf"}', (tet_thick >= lo) & (tet_thick < hi))
