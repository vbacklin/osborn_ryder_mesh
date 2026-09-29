"""Isolated sweep of (shoreline_res, min_thickness) -> does the polygon actually
exclude the thin columns?

Writes ONLY into the scratch dir, so it is safe to run while a meshing job is
using new_data/. Reuses the real get_boundary_string/get_main_boundary so the
polygon extraction matches the pipeline exactly.
"""
import os
import sys
import numpy as np
from osgeo import gdal
from osgeo_utils import gdal_calc
from matplotlib.path import Path as MplPath

sys.path.insert(0, '/home/josefin/Ryder/osborn_ryder_mesh/mesh')
from ryder_3D_meshing_manylayers import get_boundary_string, get_main_boundary

gdal.UseExceptions()

SCRATCH = os.path.dirname(os.path.abspath(__file__))
ISO = os.path.join(SCRATCH, 'iso')
DATA = '/home/josefin/Ryder/osborn_ryder_mesh/data'

surf = gdal.Open(f'{DATA}/surface_pos.tif')
thick = gdal.Open(f'{DATA}/thickness.tif')
bath = gdal.Open(f'{DATA}/elevation.tif')

S = surf.GetRasterBand(1).ReadAsArray().astype(float)
T = thick.GetRasterBand(1).ReadAsArray().astype(float)
B = bath.GetRasterBand(1).ReadAsArray().astype(float)
seathk = np.minimum(S - T, 0.0) - B

# Write the seathickness raster into the isolated dir (mirrors what main() puts
# in new_data/seathickness.tif, but private to this experiment).
drv = gdal.GetDriverByName('GTiff')
st_path = os.path.join(ISO, 'seathickness.tif')
ds = drv.Create(st_path, surf.RasterXSize, surf.RasterYSize, 1, gdal.GDT_Float32)
ds.SetGeoTransform(surf.GetGeoTransform())
ds.SetProjection(surf.GetProjection())
ds.GetRasterBand(1).WriteArray(seathk)
ds.FlushCache()
ds = None

gt = surf.GetGeoTransform()
cols = np.arange(surf.RasterXSize)
rows = np.arange(surf.RasterYSize)
XX, YY = np.meshgrid(gt[0] + (cols + 0.5) * gt[1], gt[3] + (rows + 0.5) * gt[5])
flat_pts = np.column_stack([XX.ravel(), YY.ravel()])
valid_base = (B != -9999) & (B <= 0)

print(f'{"thr":>5} {"res":>6} {"pts":>5} {"maxseg_m":>9} {"min_thk":>8} '
      f'{"p0.5":>7} {"n<10m":>6} {"n<thr":>6}')
print('-' * 62)

for min_thickness in (10, 30):
    for shoreline_res in (30, 60, 120, 250, 500, 'unlimited'):
        tagr = shoreline_res if shoreline_res != 'unlimited' else 'unlim'
        mask_path = os.path.join(ISO, f'mask_t{min_thickness}_r{tagr}.tif')
        geo_path = os.path.join(ISO, f'outline_t{min_thickness}_r{tagr}.geojson')
        for p in (mask_path, geo_path):
            if os.path.exists(p):
                os.remove(p)

        mask = gdal_calc.Calc(
            calc=[f'(A>{min_thickness}) * (B!=-9999) * (B<=0) * (C==1)'],
            A=st_path, B=f'{DATA}/elevation.tif',
            C=os.path.join(ISO, 'domainmask.tif'),
            outfile=mask_path, NoDataValue=0, overwrite=True, quiet=True,
        )
        gdal.Footprint(geo_path, mask, format='GeoJSON',
                       maxPoints=shoreline_res, srcNodata=0)
        poly = np.asarray(get_main_boundary(get_boundary_string(geo_path)))

        seg = np.hypot(*(np.diff(np.vstack([poly, poly[:1]]), axis=0).T))
        inside = MplPath(poly).contains_points(flat_pts).reshape(XX.shape)
        v = seathk[inside & valid_base]
        v = v[np.isfinite(v)]
        print(f'{min_thickness:>5} {str(shoreline_res):>6} {len(poly):>5} '
              f'{seg.max():>9.0f} {v.min():>8.2f} '
              f'{np.percentile(v, 0.5):>7.2f} {(v < 10).sum():>6} '
              f'{(v < min_thickness).sum():>6}')
