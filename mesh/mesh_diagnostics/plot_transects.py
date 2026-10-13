"""Cut a stacked mesh with planes y = const and draw the actual element cuts
(as a ParaView slice would), over the raster ice and bed.

The mesh must be in GIS coordinates (the stacked path is not rotated).
z is divided by SCALE so it is comparable with the rasters.

usage: python mesh_diagnostics/plot_transects.py MESH.msh OUT.png Y0 Y1 STEP
       [--scale 1] [--xlim X0 X1]       (all coordinates in km)
"""
import sys
import numpy as np
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from osgeo import gdal

gdal.UseExceptions()
args = sys.argv[1:]
scale, xlim = 1.0, None
if "--scale" in args:
    i = args.index("--scale"); scale = float(args[i + 1]); del args[i:i + 2]
if "--xlim" in args:
    i = args.index("--xlim"); xlim = (float(args[i + 1]), float(args[i + 2])); del args[i:i + 3]
mesh_path, out = args[0], args[1]
y0, y1, step = (float(a) * 1000 for a in args[2:5])
ys = np.arange(y0, y1 + 0.5 * np.sign(step) * abs(step), step) if step else [y0]


def raster(path):
    ds = gdal.Open(path)
    return ds.ReadAsArray().astype(float), ds.GetGeoTransform()


S, gt = raster("../data/surface_pos.tif")
T, _ = raster("../data/thickness.tif")
B, _ = raster("../data/elevation.tif")
Cr, cgt = raster("../data/gis_data/categories.tif")


def samp(A, g, x, y):
    c = np.floor((x - g[0]) / g[1]).astype(int)
    r = np.floor((y - g[3]) / g[5]).astype(int)
    ok = (c >= 0) & (c < A.shape[1]) & (r >= 0) & (r < A.shape[0])
    out = np.full(len(x), np.nan)
    out[ok] = A[r[ok], c[ok]]
    return out


m = meshio.read(mesh_path)
P = m.points.copy()
P[:, 2] /= scale
tets = [c.data for c in m.cells if c.type == "tetra"][0]
ty = P[tets][:, :, 1]
tmin, tmax = ty.min(1), ty.max(1)
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def slice_polys(yc):
    sel = tets[(tmin < yc) & (tmax > yc)]
    p = P[sel]
    polys = []
    for k in range(len(sel)):
        q = p[k]
        pts = []
        for a, b in EDGES:
            da, db = q[a, 1] - yc, q[b, 1] - yc
            if da * db < 0:
                t = da / (da - db)
                pts.append(q[a] + t * (q[b] - q[a]))
        if len(pts) >= 3:
            pts = np.array(pts)[:, [0, 2]]
            c = pts.mean(0)
            pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
            polys.append(pts / [1000, 1])
    return polys


n = len(ys)
ncol = 2 if n > 4 else 1
nrow = int(np.ceil(n / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(9 * ncol, 2.9 * nrow), squeeze=False)
for ax, yc in zip(axs.T.ravel(), ys):
    polys = slice_polys(yc)
    x = np.linspace(-96000, -77000, 1500)
    yy = np.full_like(x, yc)
    s, t, b = samp(S, gt, x, yy), samp(T, gt, x, yy), samp(B, gt, x, yy)
    c = samp(Cr, cgt, x, yy)
    b = np.where(b == -9999, np.nan, b)
    ib = s - t
    ice = (c == 2) | (c == 3)
    ax.fill_between(x / 1000, ib, s, where=ice, color="#cfe8f5", step="mid", label="ice (raster)")
    ax.plot(x / 1000, b, "k-", lw=0.9, drawstyle="steps-mid", label="bed (raster)")
    ax.plot(x / 1000, np.minimum(ib, 0), "b-", lw=0.8, drawstyle="steps-mid",
            label="ice bottom (raster)")
    ax.add_collection(PolyCollection(polys, facecolors="#f2a7b8", edgecolors="#8a1c3a",
                                     linewidths=0.25, alpha=0.75))
    ax.axhline(0, color="grey", lw=0.4)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        xs = np.concatenate([q[:, 0] for q in polys]) if polys else np.array([-92, -80])
        ax.set_xlim(xs.min() - 1.0, xs.max() + 1.0)
    ax.set_ylim(-950, 150)
    ax.set_title(f"y = {yc / 1000:.1f} km   ({len(polys)} cut tets)", fontsize=9)
    ax.tick_params(labelsize=7)
for ax in axs.T.ravel()[n:]:
    ax.axis("off")
axs[0, 0].legend(fontsize=7, loc="lower left")
fig.suptitle(f"{mesh_path}: element cuts (pink) vs rasters, z unscaled", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=95)
print("wrote", out)
