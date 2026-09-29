"""Read-only: distribution of water-column thickness inside the domain polygon.

Uses only the pristine source rasters in ../data (never new_data/), so it is
safe to run while a meshing job is using new_data/ as scratch.
"""
import json
import numpy as np
from osgeo import gdal
from matplotlib.path import Path as MplPath

gdal.UseExceptions()

surf = gdal.Open('../data/surface_pos.tif')
thick = gdal.Open('../data/thickness.tif')
bath = gdal.Open('../data/elevation.tif')

S = surf.GetRasterBand(1).ReadAsArray().astype(float)
T = thick.GetRasterBand(1).ReadAsArray().astype(float)
B = bath.GetRasterBand(1).ReadAsArray().astype(float)

icebottom = S - T
seasurface = np.minimum(icebottom, 0.0)
seathk = seasurface - B

gt = surf.GetGeoTransform()
ny, nx = seathk.shape
cols = np.arange(nx)
rows = np.arange(ny)
X = gt[0] + (cols + 0.5) * gt[1]
Y = gt[3] + (rows + 0.5) * gt[5]
XX, YY = np.meshgrid(X, Y)

for tag in ('10', '30'):
    poly = json.loads(open(f'new_data/saved_shoreline_t{tag}.json').read())
    path = MplPath(np.asarray(poly))
    inside = path.contains_points(
        np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    valid = inside & (B != -9999) & (B <= 0)
    v = seathk[valid]
    v = v[np.isfinite(v)]
    print(f'--- shoreline t={tag}  ({len(poly)} polygon points) ---')
    print(f'  cells inside domain : {valid.sum()}')
    print(f'  seathickness min    : {v.min():.2f} m')
    for q in (0.1, 0.5, 1, 5, 25, 50):
        print(f'  {q:>5}th pct         : {np.percentile(v, q):8.2f} m')
    print(f'  max                 : {v.max():.2f} m')
    for thr in (10, 20, 30, 50):
        print(f'  cells thinner than {thr:3d} m : {(v < thr).sum()} '
              f'({100.0 * (v < thr).mean():.2f}%)')
