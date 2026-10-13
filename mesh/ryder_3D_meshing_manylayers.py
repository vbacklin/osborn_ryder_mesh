import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import math
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo_utils import gdal_calc
import shutil

import os
import locale
os.environ["LC_NUMERIC"] = "C"
try:
    locale.setlocale(locale.LC_NUMERIC, "C")
except locale.Error as e:
    print("Failed to set LC_NUMERIC:", e)

import gmsh
import meshio
from stacked_volume import (build_stacked_volume, layers_from_min_thickness,
                            TOP, BOTTOM, SIDE, CALVING)
from typing import Dict, NamedTuple, Sequence, List, Optional, Union
from pathlib import Path
import subprocess, sys


# plt.style.use('seaborn-v0_8')
gdal.UseExceptions()

MIN_ICE_THICKNESS = 0.05
MIN_WATER_DEPTH = 0.1
EPS_Z_CLAMP = 10.0

# Minimum water-column thickness (meters) enforced when positioning mesh nodes.
# Unlike SHORELINE_MIN_THICKNESS below -- which only moves the domain *boundary*
# and so cannot reach nodes in shallow pockets enclosed by that boundary -- this
# pushes the sampled sea floor down until the column between the ice draft (or
# the sea surface, in open water) and the floor is at least this thick. Every
# column is then non-degenerate by construction, so sigma layers stay well-formed
# everywhere no matter what the shoreline polygon encloses.
#
# The trade-off is physical: shallow fjord edges get artificially deepened to
# this value. 0.0 disables the clamp entirely and reproduces the previous
# behaviour exactly. Overridable per-run via the MIN_WATER_COLUMN env var.
MIN_WATER_COLUMN = float(os.environ.get("MIN_WATER_COLUMN", 0.0))

# Diagnostic switch: whether to embed the per-column sigma-level seed points as
# 0D constraints in the volume before generate(3). The seed count scales with
# num_of_layers (2492 at 15 layers, 6586 at 38), and TetGen has to insert every
# one during boundary recovery -- so this isolates "too many embedded points"
# from "degenerate columns" as the cause of a generate(3) failure. Set to 0 to
# skip the embed. Default 1 = previous behaviour.
EMBED_SIGMA_SEEDS = int(os.environ.get("EMBED_SIGMA_SEEDS", 1))

# Minimum sea-water column thickness (meters) for a location to be included
# in the domain at all -- the shoreline boundary is drawn where thickness
# drops to this value, rather than at the literal (zero-thickness) coastline.
# Keeps every column in the mesh non-degenerate, close to MIN_WATER_DEPTH but
# with a small safety margin (not exaggerated) so real sigma-layer structure
# stays reasonable right up to the boundary. The domain therefore stops
# slightly short of the true coastline -- acceptable for now.
# Overridable per-run via the SHORELINE_MIN_THICKNESS environment variable so
# the threshold can be swept without editing this file; the shoreline cache is
# keyed on this value (see main()), so changing it never reuses a stale boundary.
SHORELINE_MIN_THICKNESS = float(os.environ.get("SHORELINE_MIN_THICKNESS", 30.0))

# Same clip, but for water *under ice* (thickness.tif > 0).  Near the fjord walls
# the shelf data is not measured: the ice draft is exactly hydrostatic and ~13% of
# the shelf has a filler bed of "ice base - 10 m", so the cavity ramps up through
# thin sloping wedges where the shelf in reality rests on the walls.  A higher
# threshold here starts the domain where the cavity is well defined.  75 m is the
# lowest value that keeps every 10 m filler pixel out of the domain (h=500: 0
# inverted tets, quality unchanged); set it equal to SHORELINE_MIN_THICKNESS to
# turn the separate treatment off.
SHELF_MIN_THICKNESS = float(os.environ.get("SHELF_MIN_THICKNESS", 75.0))

# Pull the boundary this far (m) inward from the threshold contour before it is
# traced.  The fjord walls are cliffs: water goes from 0 to >100 m within one or
# two 150 m pixels, so the 30 m and 100 m contours are nearly the same line and
# raising a threshold barely moves the boundary.  Walking a few pixels in
# lands it on the deep side instead.  At 300 m (75 m clip), boundary samples
# with <10 m of water drop from 19% to 0.3% for ~6% of the area.  The margin is
# not applied where the domain ends at the data edge (the ocean mouth).
SHORELINE_MARGIN = float(os.environ.get("SHORELINE_MARGIN", 300.0))

# Shoreline vertex density.  The boundary is traced at full pixel resolution
# and then simplified (Douglas-Peucker) with a per-vertex tolerance: tight
# where ice is within ~450 m (the grounding line), loose elsewhere.  Below
# ~100 m the polygon follows the 150 m pixel staircase, so do not go lower.
# SHORELINE_GL_TOL=0 falls back to gdal.Footprint with SHORELINE_RES vertices.
SHORELINE_GL_TOL = float(os.environ.get("SHORELINE_GL_TOL", 150.0))
SHORELINE_TOL = float(os.environ.get("SHORELINE_TOL", 600.0))

# 1 = open an interactive editor on the shoreline polygon before meshing (drag,
# insert, delete vertices; close the window to save).  It starts from the saved
# polygon for the current thresholds if there is one, else from the automatic
# contour, and writes the result back to that cache file.
SHORELINE_EDIT = int(os.environ.get("SHORELINE_EDIT", 0))

# Minimum column thickness the direct prism-stacking path will accept.  Unlike
# the TetGen path, build_stacked_volume refuses to mesh a collapsed column
# rather than quietly producing a sliver, so a floor is mandatory there.
# MIN_WATER_COLUMN takes precedence when it is set; this is the fallback.
MIN_STACK_THICKNESS = float(os.environ.get("MIN_STACK_THICKNESS", 10.0))

# Select the meshing path: 1 = direct prism stacking (no TetGen), 0 = the
# original skin + TetGen route.
USE_STACKED = int(os.environ.get("USE_STACKED", 1))
SHORELINE_RES = int(os.environ.get("SHORELINE_RES", 30))

# Stacked path only: thinnest layer allowed, in true (unscaled) metres.  A
# column of thickness H gets clip(floor(H / MIN_LAYER_THICKNESS), 1, NUM_LAYERS)
# layers, so the thin columns at the shoreline / grounding line carry fewer,
# thicker layers instead of 38 slivers.  0 = every column gets NUM_LAYERS.
# Default 10 m: grounding-line tets with gamma<0.01 drop from 30% to 0.1% at
# h=500 (mesh_diagnostics/quality_by_region.py); columns under ~380 m get
# fewer than 38 layers.
MIN_LAYER_THICKNESS = float(os.environ.get("MIN_LAYER_THICKNESS", 10.0))

# Vertical stretching under the ice (stacked path): in ice-covered columns the
# first layer below the ice base is ICE_TOP_LAYER m thick and each layer below
# is a constant factor thicker (at most LAYER_GROWTH), so the plume boundary
# layer is resolved and the deep cavity is coarse.  0 = evenly spaced layers.
ICE_TOP_LAYER = float(os.environ.get("ICE_TOP_LAYER", 5.0))
LAYER_GROWTH = float(os.environ.get("LAYER_GROWTH", 1.2))

# Horizontal refinement of the plan-view mesh (stacked path).  Under the ice
# the triangles are ICE_ELEMENT_SIZE, along the grounding line (outline walls
# around the ice) GL_ELEMENT_SIZE, and both grow back to ELEMENT_SIZE over
# SIZE_GRADING m -- from the grounding line, and from the ice front into open
# water.  0 for the two sizes means h/2 and h/4.  HORIZONTAL_REFINE=0: uniform.
HORIZONTAL_REFINE = int(os.environ.get("HORIZONTAL_REFINE", 1))
ICE_ELEMENT_SIZE = float(os.environ.get("ICE_ELEMENT_SIZE", 0))
GL_ELEMENT_SIZE = float(os.environ.get("GL_ELEMENT_SIZE", 0))
SIZE_GRADING = float(os.environ.get("SIZE_GRADING", 3000.0))

WATER_SURFACE_TAG = 5
ICE_TAG = 1
BATHYMETRY_TAG = 2
INFLOW_TAG = 4
# Boundary tag of the vertical calving face.  Defaults to the ice tag, so the
# solver treats the front like the ice base; set 7 to keep it separate.
CALVING_FRONT_TAG = int(os.environ.get("CALVING_FRONT_TAG", ICE_TAG))

# Stacked path: build the calving front as a vertical face (sea surface down to
# the ice draft) instead of a ramp across one element.  0 = old ramp.
CALVING_FRONT = int(os.environ.get("CALVING_FRONT", 1))
# Front columns whose draft is shallower than this (m) get no vertical face.
MIN_FRONT_DRAFT = 1.0
INFLOW_LINE_TAG = 3
SURFACE_PHYSICAL_TAGS = (ICE_TAG, BATHYMETRY_TAG, INFLOW_TAG, WATER_SURFACE_TAG)
GMESH_NUM_THREADS = 10
ROTATE_X_DEGREES = 270 # rotate so that y is up
ROTATE_Y_DEGREES = (-246+135-4.15) # rotate so that x_max is at open ocean ("inflow")
ROTATE_X_RAD = math.radians(ROTATE_X_DEGREES)
ROTATE_Y_RAD = math.radians(ROTATE_Y_DEGREES)
ROTATE_X_COS = math.cos(ROTATE_X_RAD)
ROTATE_X_SIN = math.sin(ROTATE_X_RAD)
ROTATE_Y_COS = math.cos(ROTATE_Y_RAD)
ROTATE_Y_SIN = math.sin(ROTATE_Y_RAD)

# Physical groups to target (optional).
phys_line_targets: Sequence[int] = []     # Physical Line tags to follow (1D)
phys_surface_targets: Sequence[int] = []  # Physical Surface tags to follow (2D)

BandBoundarySpec = Union[str, int]
# Boundaries within the band that should retain LC_MAX instead of LC_MIN. Entries
# may be category names such as "inflow" or explicit curve tags.
BAND_BOUNDARIES_USE_LC_MAX: Sequence[BandBoundarySpec] = ("inflow", "intersection", INFLOW_LINE_TAG)


        # "inflow": inflow_curve_tags,
        # "inflow_lines": inflow_curve_tags,
        # "shoreline": shoreline_curve_tags,
        # "bathymetry": shoreline_curve_tags,
        # "grounding": grounding_curve_tags,
        # "grounding_line": grounding_curve_tags,
        # "intersection": intersection_curve_tags,
# --- General utilities ------------------------------------------------------

def ensure_msh_path(name: Union[str, Path]) -> str:
    """Return a filesystem path with a .msh suffix, ensuring parent directories exist."""
    path = Path(name)
    if path.suffix.lower() != ".msh":
        path = path.with_suffix(".msh")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.as_posix()


def write_legacy_xdmf(msh_path: Union[str, Path]) -> Path:
    """Write the mesh as two XDMF files, the way legacy FEniCS reads them.

    ``<stem>_volume.xdmf``  tetrahedra, ``name_to_read`` = volume tag
    ``<stem>_surface.xdmf`` boundary triangles, ``name_to_read`` = boundary tag
    (the same split poisson/model_poisson.py makes from the .msh).

    A single file holding both cell types is a "Mixed" topology: legacy dolfin
    cannot read it, and ParaView draws the tets' outer faces on top of the
    tagged boundary triangles, so the boundary colouring looks patchy.
    Returns the surface file path.
    """
    msh_path = Path(msh_path)
    msh = meshio.read(msh_path)
    tags = msh.cell_data.get("gmsh:physical")
    if tags is None:
        raise ValueError(f"{msh_path} has no gmsh:physical cell data")

    def collect(kind):
        blocks = [(c.data, np.asarray(t, dtype=np.int32))
                  for c, t in zip(msh.cells, tags) if c.type == kind]
        if not blocks:
            return None, None
        return (np.concatenate([b[0] for b in blocks]),
                np.concatenate([b[1] for b in blocks]))

    written = []
    for kind, suffix in (("tetra", "_volume"), ("triangle", "_surface")):
        cells, data = collect(kind)
        if cells is None:
            continue
        path = msh_path.with_name(msh_path.stem + suffix + ".xdmf")
        meshio.write(path, meshio.Mesh(points=msh.points, cells=[(kind, cells)],
                                       cell_data={"name_to_read": [data]}),
                     file_format="xdmf")
        written.append(path)
    if not written:
        raise ValueError(f"{msh_path} contains no tetra or triangle cells to export.")
    # Drop a stale mixed-topology companion from earlier runs.
    old = msh_path.with_suffix(".xdmf")
    for stale in (old, old.with_suffix(".h5")):
        if stale.exists():
            stale.unlink()
    return written[-1]

# --- Mesh sizing and quality parameters (easy to tweak) ----------------------

# Shoreline-aware size band (units match mesh coordinates)
BAND_WIDTH_MIN = 0.0          # Distance keeping LC_MIN (0 ⇒ LC_MIN starts at the boundary)
BAND_WIDTH_MAX = 3000.0       # Distance where sizes transition back to LC_MAX; must exceed BAND_WIDTH_MIN
LC_MIN = 500.0               # Fine element size inside the refinement band
LC_MAX = 2500.0               # Default element size far from the refinement band
DISTANCE_SAMPLING = 40         # Distance field sampling density along curves

# 3D volumetric quality targets (used when optimize=True)
VOLUME_QUALITY_TARGET = 0.6        # Minimum acceptable mean-ratio quality after optimization
VOLUME_QUALITY_MAX_PASSES = 10      # Maximum targeted optimization passes
VOLUME_QUALITY_METHODS: Sequence[str] = (
    "",             # Default tetra optimizer (general smoothing)
    "Relocate3D",   # 3D node relocation smoothing
    # "Netgen",
    # "Lloyd",
)

# Optional global smoothing / robustness tweaks
MESH_SMOOTHING_ITERS = 100
INITIAL_DELAUNAY_TOL = 1e-13
MESH_ALGORITHM3D_QUALITY = 4   # Frontal 3D

# Mesh optimization controls (standard Gmsh smoothing / optimization)
ENABLE_VOLUME_OPTIMIZATION_DEFAULT = True  # Default for generate_mesh_mult(..., optimize=...)
ENABLE_VOLUME_SMOOTHING = True            # Apply Mesh.Smoothing before 3D generation
ENABLE_VOLUME_QUALITY_ALGORITHM = True    # Prefer quality-focused 3D algorithm

# Uniform mesh refinement controls
SURFACE_UNIFORM_REFINEMENT_PASSES = 0     # How many uniform refinement passes to run on the 2D mesh
VOLUME_UNIFORM_REFINEMENT_PASSES = 0      # How many uniform refinement passes to run on the 3D mesh

# 2D boundary distance-field controls
BOUNDARY_FIELD_SAMPLING = 100
BOUNDARY_FIELD_DIST_MIN = 100
BOUNDARY_FIELD_DIST_MAX = 250


def _curves_from_physical_lines(ptags: Sequence[int]) -> List[int]:
    curves: List[int] = []
    for tag in ptags:
        for ent in gmsh.model.getEntitiesForPhysicalGroup(1, int(tag)):
            curves.append(int(ent))
    return sorted(set(curves))

def _curves_from_surfaces(surface_tags: Sequence[int]) -> List[int]:
    """Return all boundary curves for the provided surfaces."""
    curves: set[int] = set()
    for surface_tag in surface_tags:
        boundary = gmsh.model.getBoundary([(2, int(surface_tag))], oriented=False, recursive=False)
        for dim, curve in boundary:
            if dim == 1:
                curves.add(int(curve))
    return sorted(curves)


def _min_volume_quality() -> Optional[float]:
    """Compute a robust mean-ratio quality for all tetrahedra.

    q_tet = 12 * (3 V)^{2/3} / sum_{edges} |e|^2, in [0, 1] for regular tets.
    Handles 4- and 10-node tets (uses corner nodes for the latter).
    """
    try:
        all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
    except Exception as err:
        print(f"[ryder_3D_meshing] Unable to access nodes for quality check: {err}", file=sys.stderr)
        return None

    coord = {
        int(tag): (
            float(all_node_coords[3 * idx]),
            float(all_node_coords[3 * idx + 1]),
            float(all_node_coords[3 * idx + 2]),
        )
        for idx, tag in enumerate(all_node_tags)
    }

    def tet_quality(p0, p1, p2, p3) -> float:
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)
        v3 = np.array(p3) - np.array(p0)
        vol6 = float(np.dot(v1, np.cross(v2, v3)))
        V = abs(vol6) / 6.0
        e01 = np.linalg.norm(np.array(p1) - np.array(p0))**2
        e02 = np.linalg.norm(np.array(p2) - np.array(p0))**2
        e03 = np.linalg.norm(np.array(p3) - np.array(p0))**2
        e12 = np.linalg.norm(np.array(p2) - np.array(p1))**2
        e13 = np.linalg.norm(np.array(p3) - np.array(p1))**2
        e23 = np.linalg.norm(np.array(p3) - np.array(p2))**2
        denom = float(e01 + e02 + e03 + e12 + e13 + e23)
        if denom <= 1e-30:
            return 0.0
        q = float(12.0 * (3.0 * V) ** (2.0 / 3.0) / denom)
        if q < 0.0:
            return 0.0
        if q > 1.0:
            return 1.0
        return q

    try:
        elem_types, elem_tags_list, elem_nodes_list = gmsh.model.mesh.getElements(3)
    except Exception as err:
        print(f"[ryder_3D_meshing] Unable to list 3D elements: {err}", file=sys.stderr)
        return None

    found_any = False
    worst = float("inf")

    for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_nodes_list):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
        if num_nodes not in (4, 10):
            continue
        tags = elem_tags.tolist() if hasattr(elem_tags, "tolist") else list(elem_tags)
        nodes = elem_nodes.tolist() if hasattr(elem_nodes, "tolist") else list(elem_nodes)
        step = num_nodes
        for idx in range(len(tags)):
            start = idx * step
            node_ids = [int(n) for n in nodes[start:start + step]]
            n0, n1, n2, n3 = node_ids[0], node_ids[1], node_ids[2], node_ids[3]
            try:
                p0, p1, p2, p3 = coord[n0], coord[n1], coord[n2], coord[n3]
            except KeyError:
                continue
            q = tet_quality(p0, p1, p2, p3)
            found_any = True
            if q < worst:
                worst = q

    return worst if found_any else None


def _improve_volume_quality(
    target: float,
    max_passes: int,
    methods: Sequence[str],
) -> Optional[float]:
    """Iteratively apply Gmsh 3D optimizers focused on low-quality tets.

    Returns the final minimum quality if available; otherwise ``None``.
    """

    if target <= 0 or max_passes <= 0:
        return None

    worst = _min_volume_quality()
    if worst is None or worst >= target:
        return worst

    failed_methods: set[str] = set()

    for _ in range(int(max_passes)):
        if worst is not None and worst >= target:
            break
        for method in methods:
            if method in failed_methods:
                continue
            try:
                gmsh.model.mesh.optimize(method=method, niter=1)
            except Exception as err:
                print(f"[ryder_3D_meshing] Skipping optimizer '{method}': {err}", file=sys.stderr)
                failed_methods.add(method)
        worst = _min_volume_quality()
        if worst is None:
            break

    return worst


def _run_uniform_refinement(passes: int, label: str) -> None:
    """Apply Gmsh's uniform refinement ``passes`` times, logging progress."""
    total = int(max(0, passes))
    if total <= 0:
        return
    for step in range(1, total + 1):
        try:
            print(f"[{label}] Uniform refinement pass {step}/{total}")
            gmsh.model.mesh.refine()
        except Exception as err:
            print(f"[{label}] Uniform refinement failed on pass {step}: {err}", file=sys.stderr)
            break


def readraster(filename):
    """Load a raster with GDAL and write a working copy next to our scripts.

    The original geotiffs are kept immutable.  Creating a copy in
    `mesh/new_data/` lets us freely modify metadata (projections, footprints
    etc.) without touching the source dataset the user provided.
    """
    raster = gdal.Open(filename)
    driver = raster.GetDriver()
    copy_name_list = filename.split('.')
    main_name = (copy_name_list[-2].split('/'))[-1]
    copy_name = 'new_data/'+main_name+'_copy.'+copy_name_list[-1]
    raster_copy = driver.CreateCopy(copy_name, raster)
    return raster_copy

def readvector(filename):
    """Helper around GDAL/OGR to read vector layers such as shorelines."""
    vector = gdal.OpenEx(filename)
    return vector

def get_boundary_string(filename):
    """Return the WKT string describing the boundary of the first layer."""
    border = readvector(filename)
    layer = border.GetLayer()
    feature = layer.GetFeature(0)
    geometry = feature.GetGeometryRef()
    boundary = str(geometry.GetBoundary())
    return boundary
    
def get_main_boundary(boundary, no=0):
    """Pick one polyline from a MULTILINESTRING boundary description.

    Parameters
    ----------
    boundary: str
        WKT string (typically MULTILINESTRING) returned by GDAL.
    no: int
        Which curve to pick after sorting by length.  By default we take the
        outermost shoreline (the longest curve).
    """
    boundary = boundary.replace('MULTILINESTRING ','')
    curves = boundary.split(')')
    if no == 0:
        # The outer boundary is the longest polyline in the WKT description.
        main_boundary = max(curves, key=len) #Check for longest curve <- outer boundary
        main_boundary = main_boundary.strip(',(')
    else:
        sorted_bounds = sorted(curves, reverse=True, key=len)
        main_boundary = sorted_bounds[no]
        main_boundary = main_boundary.strip(',(')
    
    boundary_split = main_boundary.split(',')[:-1]
    coords = [(float(i.split(' ')[0]), float(i.split(' ')[1])) for i in boundary_split]
    
    return coords

def inflow_segments(coords, category_array, category_trans, offset=300.0,
                    step=150.0, min_fraction=0.5):
    """Flag each polygon edge (coords[i-1], coords[i]) that is an open-ocean inflow.

    An edge is inflow when the ground just *outside* it (sampled every `step`
    along the edge, `offset` beyond it) is category 6.  The old rule asked for
    both endpoints to sit on/next to category 6, which breaks whenever the
    polygon is redrawn: Footprint moves vertices by a pixel or two, and a mouth
    endpoint that lands on water drops the whole 16 km mouth.  Sampling along
    the edge does not depend on where the vertices fall.

    Returns a list `flags` with flags[i] for the edge ending at coords[i]
    (flags[0] is the closing edge coords[-1] -> coords[0]).
    """
    from matplotlib.path import Path as _MplPath
    pts = np.asarray(coords, dtype=float)
    poly = _MplPath(pts)
    ny, nx = category_array.shape

    def cat_at(x, y):
        c = int(np.floor((x - category_trans[0]) / category_trans[1]))
        r = int(np.floor((y - category_trans[3]) / category_trans[5]))
        return category_array[r, c] if 0 <= r < ny and 0 <= c < nx else -1

    flags = []
    for i in range(len(pts)):
        a, b = pts[i - 1], pts[i]
        length = float(np.hypot(*(b - a)))
        if length == 0:
            flags.append(False)
            continue
        normal = np.array([-(b - a)[1], (b - a)[0]]) / length
        k = max(2, int(length / step))
        hits = 0
        for t in (np.arange(k) + 0.5) / k:
            q = a + t * (b - a)
            out = q + offset * normal
            if poly.contains_point(out):
                out = q - offset * normal
            hits += cat_at(*out) == 6
        flags.append(hits / k >= min_fraction)
    return flags


def point_in_set(point, line, tol=1e-6):
    """Return True if `point` already exists (within `tol`) in `line`."""
    
    px, py = point
    for (x, y) in line:
        if abs(px - x) < tol and abs(py - y) < tol:
            return True
    return False

def rotate_coordinates(x, y, z):
    """Rotate a point around the configured x- then y-axis angles.

    Gmsh exports the fjord in its native projection.  Downstream solvers expect
    the domain to be aligned with the standard Cartesian axes.  We therefore
    pre-compute the sine/cosine values above and apply a two-step rotation
    (first about X, then Y) to every node before writing the final mesh.
    """
    x_after_x = x
    y_after_x = y * ROTATE_X_COS - z * ROTATE_X_SIN
    z_after_x = y * ROTATE_X_SIN + z * ROTATE_X_COS

    x_after_y = x_after_x * ROTATE_Y_COS + z_after_x * ROTATE_Y_SIN
    y_after_y = y_after_x
    z_after_y = -x_after_x * ROTATE_Y_SIN + z_after_x * ROTATE_Y_COS

    return x_after_y, y_after_y, z_after_y

def get_grounding_line(seaice, landice):
    """Return the shared points between the sea-ice and land-ice polylines."""
    grounding_line = []
    
    for coord in seaice:
        if point_in_set(coord, landice):
            grounding_line.append(coord)

    return grounding_line



def add_grounding_line(shoreline, grounding_line):
    """Splice the grounding line polyline into the shoreline outline.

    We want the final 2D mesh to contain a sharp feature where the ice sheet
    detaches from the bed.  Gmsh can only preserve that feature if it is part
    of the top-level boundary, so we rebuild the shoreline list to explicitly
    insert the grounding-line points between the two matching shoreline points.
    """
    
    p1 = grounding_line[0]
    p2 = grounding_line[-1]
    
    tag1 = find_closest_point(p1[0], p1[1], shoreline)
    tag2 = find_closest_point(p2[0], p2[1], shoreline)
    
    new_shoreline = []
    
    p3 = shoreline[tag1]
    if p1[0] == p3[0] and p1[1] == p3[1]:
        grounding_line = grounding_line[1:]
        
    p4 = shoreline[tag2]
    if p2[0] == p4[0] and p2[1] == p4[1]:
        grounding_line = grounding_line[:-1]
    
    # Remove shoreline points that sit “below” the grounding line so the
    # stitched boundary stays monotonic along the fjord wall.
    while p3[1] < p1[1] and p4[1] < p2[1]:
        shoreline = shoreline[:tag1] + shoreline[tag1+1:]
        shoreline = shoreline[:tag2-1] + shoreline[tag2:]
        p3 = shoreline[tag1]
        p4 = shoreline[tag2]
        
    while p3[1] < p1[1]:
        shoreline = shoreline[:tag1] + shoreline[tag1+1:]
        p3 = shoreline[tag1]
    while p4[1] < p2[1]:
        shoreline = shoreline[:tag2] + shoreline[tag2+1:]
        p4 = shoreline[tag2]
    
    for point in shoreline[0:(tag1+1)]:
        new_shoreline.append(point)
    for point in grounding_line:
        new_shoreline.append(point)
    for point in shoreline[tag2:]:
        new_shoreline.append(point)
    
    return new_shoreline

def find_closest_point(x, y, points):
    """Return the index of the entry in `points` closest to `(x, y)`."""
    
    distances = {}
    
    i = 0
    for point in points:
        delta_x = x - point[0]
        delta_y = y - point[1]
        distances[i] = math.sqrt(delta_x**2 + delta_y**2)
        i += 1
        
    closest_point_tag = min(distances, key = distances.get)
    
    return closest_point_tag

def clip_front_to_outline(outline, front):
    """Make the ice-front polyline end exactly on the domain outline.

    The front comes from the uncut water/sea-ice outlines, so its ends lie at
    the true fjord walls -- outside the clipped domain.  generate_2D_mesh joins
    each end to the *nearest outline vertex*; with a dense outline that join
    can cross the boundary ("Unable to recover the edge").  Here the front is
    cut to its longest stretch inside the outline and the two crossing points
    become outline vertices (reusing an existing vertex when the crossing is
    on one); the returned front starts and ends exactly on them, which
    generate_2D_mesh recognises.

    Returns (outline, front) with the front running from the lower to the
    higher inserted vertex index, as generate_2D_mesh expects.
    """
    from matplotlib.path import Path as _MplPath
    P = np.asarray(outline, dtype=float)
    F = np.asarray(front, dtype=float)
    inside = _MplPath(P).contains_points(F)
    if not inside.any():
        raise RuntimeError("ice front lies entirely outside the domain outline")
    # longest run of consecutive inside points
    best, cur = (0, 0), None
    for i, ok in enumerate(inside):
        if ok and cur is None:
            cur = i
        if (not ok or i == len(F) - 1) and cur is not None:
            end = i if ok else i - 1
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    i0, i1 = best

    def hit(a, b):
        """First crossing of segment a->b (b beyond a) with the outline."""
        d = b - a
        best_t, best_e, best_p = np.inf, None, None
        for e in range(len(P)):
            q0, q1 = P[e], P[(e + 1) % len(P)]
            r = q1 - q0
            den = d[0] * r[1] - d[1] * r[0]
            if abs(den) < 1e-12:
                continue
            w = q0 - a
            t = (w[0] * r[1] - w[1] * r[0]) / den
            u = (w[0] * d[1] - w[1] * d[0]) / den
            if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9 and t < best_t:
                best_t, best_e, best_p = t, e, a + t * d
        return best_e, best_p

    def outward(i, step):
        j = i - step
        if 0 <= j < len(F) and not inside[j]:
            return F[i], F[j]                    # real segment to the outside
        k = i + step                             # front ends inside: extend it
        nb = F[k] if 0 <= k < len(F) else F[i] - (F[i] - F[i - step])
        return F[i], F[i] + 1e4 * (F[i] - nb) / max(np.hypot(*(F[i] - nb)), 1e-9)

    e1, p1 = hit(*outward(i0, 1))
    e2, p2 = hit(*outward(i1, -1))
    if e1 is None or e2 is None:
        raise RuntimeError("could not intersect the ice front with the outline")

    pts = [tuple(x) for x in P.tolist()]
    snap = 1.0   # m: a crossing this close to an outline vertex *is* that vertex

    def place(p):
        """Index of p in the outline, inserting it on its edge if it is new."""
        d = np.hypot(*(np.asarray(pts) - p).T)
        if d.min() < snap:
            return pts[int(d.argmin())]
        for e in range(len(pts)):
            q0, q1 = np.asarray(pts[e]), np.asarray(pts[(e + 1) % len(pts)])
            r = q1 - q0
            t = float(np.dot(p - q0, r) / max(np.dot(r, r), 1e-12))
            if -1e-9 <= t <= 1 + 1e-9 and np.hypot(*(q0 + t * r - p)) < snap:
                pts.insert(e + 1, tuple(p))
                return tuple(p)
        raise RuntimeError(f"front crossing {p} is not on the outline")

    p1 = np.asarray(place(p1))
    p2 = np.asarray(place(p2))
    run = [tuple(x) for x in F[i0:i1 + 1].tolist()]

    O = np.asarray(pts, dtype=float)
    O2 = np.roll(O, -1, axis=0)

    def dist_to_outline(q):
        q = np.asarray(q)
        d = O2 - O
        t = np.clip(((q - O) * d).sum(1) / np.maximum((d ** 2).sum(1), 1e-12), 0, 1)
        return float(np.min(np.hypot(*(O + t[:, None] * d - q).T)))

    # Both lines come from pixel edges, so a front point can sit *on* the
    # outline; joining to it runs along the boundary and gives zero-area
    # triangles.  Keep only points clearly inside.
    clear = 1.0
    run = [q for q in run if dist_to_outline(q) >= clear]
    run = [tuple(p1)] + run + [tuple(p2)]
    if pts.index(tuple(p1)) == pts.index(tuple(p2)):
        raise RuntimeError("both ends of the ice front land on one outline vertex")
    k1, k2 = pts.index(tuple(p1)), pts.index(tuple(p2))
    if k1 > k2:
        run = run[::-1]
    print(f"[FRONT] ice front clipped to the outline: {len(run) - 2} interior points, "
          f"ends inserted as outline vertices {min(k1, k2)} and {max(k1, k2)}",
          flush=True)
    return [list(q) for q in pts], [list(q) for q in run]


def get_water_ice_intersect(seaice, water):
    """Return the shared polyline between the water and sea-ice outlines."""
    grounding_line = []
    
    for coord in seaice:
        if point_in_set(coord, water):
            grounding_line.append(coord)

    return grounding_line[:-1]

def plot_main_geoline(shoreline, ax=None, **kw):
    """Quick matplotlib diagnostic for one polyline (left commented out)."""
    

    if ax is None:
        ax = plt.gca()



    X = []
    Y = []
    for coord in shoreline:
        X.append(coord[0])
        Y.append(coord[1])
    
    #plt.figure(figsize=(5, 5))
    ax.scatter(X, Y)
    #ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_xlabel('Meters')
    #ax.set_ylabel('Meters')
    #plt.draw()

    return ax

    
def plot_full_outline(lines):
    """Debug helper to visualise multiple boundary polylines."""
    
    # plt.figure(figsize=(6,8))
    
    for line in lines.keys():
        X = []
        Y = []
        for coord in lines[line]:
            X.append(coord[0])
            Y.append(coord[1])
        # plt.plot(X, Y, marker = 'o', markersize = 5, linestyle = '--', label=line)
        
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xlabel('$x$ (m)', fontsize = 12)
    # ax.set_ylabel('$y$ (m)', fontsize = 12)
    # plt.title("Outline Boundary Points",
            #   fontsize = 14)
    # plt.legend(loc="best", frameon = True, facecolor = "white", edgecolor = "black", 
            #    fontsize=8)
    # plt.draw()
        

def check_if_inflow(point, category_array, category_trans):
    """Return True when the GIS raster marks this shoreline point as inflow."""
    
    x = point[0]
    y = point[1]
    
    top_left_x = category_trans[0]
    x_res = category_trans[1]
    top_left_y  = category_trans[3]
    y_res = category_trans[5]
        
    x_index = round((x - top_left_x)/x_res)
    y_index = round((y - top_left_y)/y_res)
        
    cat = category_array[y_index, x_index]
    
    if cat == 6:
        return True
    elif cat != 3:
        # Look in an 8-neighbour stencil so thin inflow channels do not get
        # missed because a single shoreline vertex landed just outside them.
        if category_array[y_index - 1][x_index] == 6:
            return True
        elif category_array[y_index - 1][x_index - 1] == 6:
            return True
        elif category_array[y_index][x_index - 1] == 6:
            return True
        elif category_array[y_index + 1][x_index] == 6:
            return True
        elif category_array[y_index + 1][x_index + 1] == 6:
            return True
        elif category_array[y_index][x_index + 1] == 6:
            return True
        elif category_array[y_index + 1][x_index - 1] == 6:
            return True
        elif category_array[y_index - 1][x_index + 1] == 6:
            return True
        else:
            return False
    else:
        return False

def get_shorelines(line_tags, intersect_tags, inflow_lines, grounding_lines):
    """Return the line entities that form the exterior shoreline only."""
    
    shorelines = []
    inflow_line_tags = [tag for (dim, tag) in inflow_lines]
    grounding_line_tags = [tag for (dim, tag) in grounding_lines]
    dont_include = grounding_line_tags + inflow_line_tags + intersect_tags
    for line in line_tags:
        if line not in dont_include:
            shorelines.append((1, line))
            
    return shorelines

def sanitize(val, eps):
    """Clamp bathymetry samples that are positive or marked as nodata."""
    if val == -9999 or val > 0:
        return 0
    else:
        return val

def bilinear_interpolate(data_array, i, j, dx, dy, eps, 
                         missing_value=-9999, sanitize=False):
    """Evaluate a raster at fractional indices using bilinear interpolation."""
    v00 = data_array[j    , i    ]
    v10 = data_array[j    , i + 1]
    v01 = data_array[j + 1, i    ]
    v11 = data_array[j + 1, i + 1]
    if sanitize:
        # For bathymetry we do not allow positive depths; clamp above sea level.
        v00 = min(v00, 0)
        v10 = min(v10, 0)
        v01 = min(v01, 0)
        v11 = min(v11, 0)

    # Bilinear interpolation
    interpolated = (
        (1 - dx) * (1 - dy) * v00 +
        dx * (1 - dy) * v10 +
        (1 - dx) * dy * v01 +
        dx * dy * v11
    )
    
    return interpolated

def get_bathymetric_depth(x, y, data_trans, data_array, highres_trans, 
                          highres_array, eps, highres = True, interpolate = True):
    """Sample the fjord bathymetry, favouring the high-resolution raster."""
    
    if highres:
        try:
            # Try the Sherard-Osborn 15 m product first – when available we get
            # a much sharper representation of channels and ridges.
            top_left_x = highres_trans[0]
            x_res = highres_trans[1]
            top_left_y  = highres_trans[3]
            y_res = highres_trans[5]
            
            x_index = round((x - top_left_x)/x_res)
            y_index = round((y - top_left_y)/y_res)
            
            z = highres_array[y_index, x_index]
            
            if z > eps:
                # Above sea level values happen near the coast.  Clamp to the
                # ice draft instead so the mesh remains watertight.
                return eps
            if z != -9999:
                return z
        except IndexError:
            pass
    
    top_left_x = data_trans[0]
    x_res = data_trans[1]
    top_left_y  = data_trans[3]
    y_res = data_trans[5]
    
    px = (x - top_left_x) / x_res
    py = (y - top_left_y) / y_res
    
    if interpolate:
        i = int(np.floor(px))
        j = int(np.floor(py))

        dx = px - i
        dy = py - j
        
        z = bilinear_interpolate(data_array, i, j, dx, dy, 
                                 eps, sanitize=True)
    else:
        
        x_index = round(px)
        y_index = round(py)
        
        z = data_array[y_index, x_index]
        
    if (z == -9999 or z > eps):
        return eps
    else:
        return z
    
def get_ice_depth(x, y, thic_trans, thic_array, surf_array, bath_array, eps, 
                  interpolate = True):
    """Compute the draft of the floating ice tongue at (x, y).

    The freeboard (surface minus thickness) can occasionally sit above the
    seabed bathymetry.  We guard against that configuration by blending towards
    the bathymetry value using the small `eps` offset.
    """
    
    top_left_x = thic_trans[0]
    x_res = thic_trans[1]
    top_left_y  = thic_trans[3]
    y_res = thic_trans[5]
    
    # Convert to pixel coordinates (float)
    px = (x - top_left_x) / x_res
    py = (y - top_left_y) / y_res
    
    if interpolate:
        # Evaluate thickness/surface rasters in the same bilinear fashion as
        # the bathymetry to avoid discontinuities across pixel boundaries.
        i = int(np.floor(px))
        j = int(np.floor(py))

        dx = px - i
        dy = py - j
        
        thickness = bilinear_interpolate(thic_array, i, j, dx, dy, eps)
        surface_position = bilinear_interpolate(surf_array, i, j, dx, dy, eps)
        bath_level = bilinear_interpolate(bath_array, i, j, dx, dy, eps, sanitize=True)
    else:
        x_index = round(px)
        y_index = round(py)
            
        thickness = thic_array[y_index, x_index]
        surface_position = surf_array[y_index, x_index]
        bath_level = bath_array[y_index, x_index]
        
    z = surface_position - thickness
    
    if z >= 0:
        return 0
    elif z - bath_level <= abs(eps) and bath_level <= 0:
        if min(bath_level - eps, 0) < bath_level:
            print(f'Warning: {min(bath_level - eps, 0)} returned for depth {bath_level}')
        return min(bath_level - eps, 0)
    else:
        if z < bath_level:
            print(f'Warning: {z} returned for depth {bath_level}')
        return z

def dump_gmsh_points(model, filename="gmsh_points.txt"):
    pts = sorted(model.getEntities(0), key=lambda x: x[1])   # sort by tag

    with open(filename, "w") as f:
        for dim, ptag in pts:
            x, y, z = model.getValue(0, ptag, [])
            f.write(f"{ptag:6d}   {x:15.8f}   {y:15.8f}   {z:15.8f}\n")

    print(f"Wrote {len(pts)} points → {filename}")



def plot_gmsh_geometry(model):
    """
    Plot the 2D geometry (points + lines) from the Gmsh CAD model
    BEFORE calling model.mesh.generate(...).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    # First plot all line entities (dim == 1)
    for dim, line_tag in model.getEntities(1):
        # Get endpoints of this line (2 point entities)
        boundary = model.getBoundary([(dim, line_tag)], oriented=True, recursive=False)
        pts = [tag for (d, tag) in boundary if d == 0]  # point tags

        xs = []
        ys = []
        for ptag in pts:
            x, y, z = model.getValue(0, ptag, [])  # coordinates of this 0D point
            xs.append(x)
            ys.append(y)

        ax.plot(xs, ys, '-k', linewidth=1)

    # Optionally also scatter all points for clarity
    all_x = []
    all_y = []
    for dim, ptag in model.getEntities(0):
        x, y, z = model.getValue(0, ptag, [])
        all_x.append(x)
        all_y.append(y)
    ax.scatter(all_x, all_y, s=5, color='red')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gmsh 2D geometry (before meshing)")
    plt.close('all')


def generate_2D_mesh(outline, intersect, grounding_line, category_data, 
                     m, eps, adapt, adaptive_scales = (1/4, 2),
                     num_of_layers: int = 1,
                     filename: Optional[str] = None, delta: Optional[float] = None,
                     band_lcmax_boundaries: Optional[Sequence[BandBoundarySpec]] = None,
                     planar_only: bool = False):
    """Create the surface mesh that will later be extruded to 3D.

    The 2D stage wires up every shoreline, grounding line, and inflow segment
    as explicit curves.  That makes it much easier to preserve important
    physical boundaries once we start extruding with Gmsh.
    When ``band_lcmax_boundaries`` is supplied (or configured globally) those
    boundaries keep the coarse ``LC_MAX`` target inside the boundary band.
    """
    
    gmsh.initialize()
    # The plan-view mesh takes well under a second, and multithreaded 2D
    # meshing is both nondeterministic (column count varied run to run) and
    # crashed intermittently (segfault right after "Meshing 2D"), so the
    # stacked path meshes it on one thread.
    gmsh.option.setNumber("General.NumThreads",
                          1 if planar_only else GMESH_NUM_THREADS)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    model = gmsh.model
    model.add("2D")
    if band_lcmax_boundaries is None:
        band_lcmax_boundaries = BAND_BOUNDARIES_USE_LC_MAX
    band_lcmax_boundaries = tuple(band_lcmax_boundaries or ())
    
    if adapt:
        # Use three characteristic lengths: fine on the grounding line, medium
        # along the inflow, and coarser away from the ice where gradients are
        # smaller.
        sizes = (adaptive_scales[0]*m, m, adaptive_scales[1]*m)
    else:
        sizes = (m,m,m)
    
    category_array = category_data.ReadAsArray()
    category_trans = category_data.GetGeoTransform()
    
    coords = outline
    intersect_points = intersect
    grounding_points = grounding_line

    map_fake_to_real = {}
    if delta is not None:
        # Add fake points on boundary in inflow
        link_fake_to_real = {}
        flat_points_fake = set()
        i = 0
        while i < len(coords):
            ptl, ptr = coords[i-1], coords[i]
            if (check_if_inflow(coords[i-1], category_array, category_trans)
            and check_if_inflow(coords[i], category_array, category_trans)):
                dir_segment = np.arctan2(ptr[1] - ptl[1], ptr[0] - ptl[0])
                add_left = not check_if_inflow(coords[i-2], category_array, category_trans)
                add_right = not check_if_inflow(coords[i+1], category_array, category_trans)
                if add_left:
                    coords = (coords[: i]
                              + [[ptl[j] + delta * fun(dir_segment)
                                         for j, fun in enumerate([np.cos, np.sin])]]
                              + coords[i :])
                    map_fake_to_real[i+1] = i
                    link_fake_to_real[i+1] = -i
                    flat_points_fake.update([(0, i+1)])
                    i += 1
                if add_right:
                    coords = (coords[: i]
                              + [[ptr[j] - delta * fun(dir_segment)
                                         for j, fun in enumerate([np.cos, np.sin])]]
                              + coords[i :])
                    map_fake_to_real[i+1] = i+2
                    link_fake_to_real[i+1] = i+1
                    flat_points_fake.update([(0, i+1)])
                    i += 1
            i += 1

    p1 = intersect_points[0]
    p2 = intersect_points[-1]
    
    tag1 = find_closest_point(p1[0], p1[1], coords) + 1
    tag2 = find_closest_point(p2[0], p2[1], coords) + 1
    
    first_point = coords[0]
    
    lines = []
    inflow_lines = []
    grounding_lines = []
    edge_is_inflow = inflow_segments(coords, category_array, category_trans)
    
    point = model.geo.addPoint(first_point[0], first_point[1], 0, sizes[1])
    i = 1
    for coord in coords[1:]:
        if point_in_set(coord, grounding_points):
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[0])
        elif point+1 > tag1-1 and point < tag2:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[1])
        else:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[2])

        line = model.geo.addLine(point - 1, point)
        lines.append(line)
        
        if edge_is_inflow[i]:
            inflow_lines.append((1, line))
        elif (point_in_set(coords[i-1], grounding_points)
              and point_in_set(coords[i], grounding_points)):
            grounding_lines.append((1, line))
        elif ((point_in_set(coords[i], grounding_points)
              or point_in_set(coords[i-1], grounding_points))
              and coords[i][0] > coords[i-1][0]):
            grounding_lines.append((1, line))
        i += 1
        
    line = model.geo.addLine(point, 1)
    lines.append(line)
    closing_seg_line = line  # the outer ocean boundary (16 km face at max X)

    if edge_is_inflow[0]:
        inflow_lines.append((1, line))

    lines_interface = []
    # Stitch the water/ice intersection onto the boundary loop so Gmsh can
    # extrude distinct inflow surfaces later.
    tag1_eff = tag1
    if delta is not None:
        q1 = coords[tag1-1]

        if np.linalg.norm([c1 - c2 for c1, c2 in zip(p1, q1)]) > delta:
            # Compute a local tangent on the OUTLINE near q1
            q1m = coords[(tag1-2) % len(coords)]
            q1p = coords[(tag1)   % len(coords)]

            tx = q1p[0] - q1m[0]
            ty = q1p[1] - q1m[1]
            tlen = math.hypot(tx, ty)
            if tlen == 0:
                tx, ty = 1.0, 0.0
            else:
                tx /= tlen
                ty /= tlen

            nx = -ty
            ny =  tx

            point = model.geo.addPoint(
                q1[0] + delta * nx,
                q1[1] + delta * ny,
                0, sizes[1]
            )
            line = model.geo.addLine(tag1, point)
            lines_interface.append(line)
            tag1_eff = point

            map_fake_to_real[point] = tag1
            link_fake_to_real[point] = -line

    # clip_front_to_outline makes the front start/end *on* outline vertices;
    # then the interface starts there instead of adding a joining line.
    same1 = (delta is None and
             math.hypot(p1[0] - coords[tag1 - 1][0], p1[1] - coords[tag1 - 1][1]) < 1e-6)
    same2 = (delta is None and
             math.hypot(p2[0] - coords[tag2 - 1][0], p2[1] - coords[tag2 - 1][1]) < 1e-6)
    if same1:
        point = tag1
    else:
        point = model.geo.addPoint(p1[0], p1[1], 0, sizes[1])
        line = model.geo.addLine(tag1_eff, point)
        lines_interface.append(line)
        if delta is not None and tag1_eff == tag1:
            map_fake_to_real[point] = tag1
            flat_points_fake.update([(0, point)])
            link_fake_to_real[point] = -line
    for q in intersect_points[1:len(intersect_points) - (1 if same2 else 0)]:
        prev = point
        point = model.geo.addPoint(q[0], q[1], 0, sizes[1])
        line = model.geo.addLine(prev, point)
        lines_interface.append(line)
    if delta is not None:
        q2 = coords[tag2-1]

        if np.linalg.norm([c1 - c2 for c1, c2 in zip(p2, q2)]) > delta:
            q2m = coords[(tag2-2) % len(coords)]
            q2p = coords[(tag2)   % len(coords)]

            tx = q2p[0] - q2m[0]
            ty = q2p[1] - q2m[1]
            tlen = math.hypot(tx, ty)
            if tlen == 0:
                tx, ty = 1.0, 0.0
            else:
                tx /= tlen
                ty /= tlen

            nx = -ty
            ny =  tx

            point_new = model.geo.addPoint(
                q2[0] + delta * nx,
                q2[1] + delta * ny,
                0, sizes[1]
            )
            line = model.geo.addLine(point, point_new)
            lines_interface.append(line)
            point = point_new

        map_fake_to_real[point] = tag2
        flat_points_fake.update([(0, point)])
    if point != tag2:
        line = model.geo.addLine(point, tag2)
        lines_interface.append(line)
        if delta is not None:
            link_fake_to_real[point] = line

    model.geo.synchronize()

    if delta is None:
        # Build two curve loops: one for the full shoreline, one for the inlet
        # patch carved out by the water/ice intersection.
        first_part = lines[:tag1-1]
        intersection = lines_interface
        second_part = lines[tag2-1:]
        bound = first_part + intersection + second_part
        loop = model.geo.addCurveLoop(bound)
        surface1 = model.geo.addPlaneSurface([loop])
        
        outer_part = lines[tag1-1:tag2-1]
        rev_intersection = [-i for i in intersection[::-1]]
        bound = outer_part + rev_intersection
        loop = model.geo.addCurveLoop(bound)
        surface2 = model.geo.addPlaneSurface([loop])
        
        surface_dim_tags = [(2, surface1), (2, surface2)]
        
        water = [surface1]
        ice = [surface2]
    else:
        # Add fake points and lines around any vertex in-between inflows
        surfaces_interior = []
        surfaces_boundary = []

        link_project_fake_to_fake = {}

        # go around water surface first, starting from tag1
        lines_interior_water = lines_interface[1 : -1]
        lines_boundary_water = [lines_interface[-1]]
        local_interior = []
        bound = model.getBoundary([(1, lines_interface[-1])], combined=True,
                                  oriented=True, recursive=False)
        interior_point = [b for _, b in bound if b in map_fake_to_real.keys()][0]
        link_fake_to_real[interior_point] = lines_interface[-1]
        skip_interior = False
        # now at tag2 (i = tag2-1)
        for i in range(tag2, len(coords)+tag1):
            i_ = i % len(coords)
            i_m = (i-1) % len(coords)
            i_p = (i+1) % len(coords)
            tag_i = i_ + 1
            tag_i_m = (i-1) % len(coords) + 1
            if (1, tag_i_m) in inflow_lines:
                if (1, tag_i) in inflow_lines: # within an inflow: extend interior
                    if not skip_interior:
                        interior_point = tag_i
                        lines_interior_water += [tag_i_m]
                else: # at end of an inflow: start new exterior region
                    lines_boundary_water = [tag_i_m]
                    local_interior = []
                skip_interior = False
            elif i == len(coords)+tag1-1 or (1, tag_i) in inflow_lines: # at start of an inflow: close off exterior
                if i == len(coords)+tag1-1:
                    tag_i = lines_interface[0]
                bound = model.getBoundary([(1, tag_i)], combined=True, oriented=True, recursive=False)
                point = [b for _, b in bound if b in map_fake_to_real.keys()][0]
                line = model.geo.addLine(interior_point, point)
                lines_interior_water += [line]
                skip_interior = True

                link_project_fake_to_fake[(interior_point, point)] = tag_i_m
                link_project_fake_to_fake[(point, interior_point)] = -tag_i_m
                lines_boundary_water += [tag_i_m, tag_i, -line] + [-tag for tag in local_interior[::-1]]
                loop = model.geo.addCurveLoop(lines_boundary_water)
                surfaces_boundary += [model.geo.addPlaneSurface([loop])]
            else: # far from any inflow: generate fake point
                dir_segment_l = np.arctan2(coords[i_][1] - coords[i_m][1],
                                           coords[i_][0] - coords[i_m][0])
                dir_segment_r = np.arctan2(coords[i_p][1] - coords[i_][1],
                                           coords[i_p][0] - coords[i_][0])
                if abs(dir_segment_l - dir_segment_r) > np.pi:
                    dir_segment_l -= 2 * np.pi
                    if abs(dir_segment_l - dir_segment_r) > np.pi:
                        dir_segment_l += 4 * np.pi
                    assert abs(dir_segment_l - dir_segment_r) <= np.pi
                dir_segment = (dir_segment_l + dir_segment_r) / 2
                delta_eff = delta / np.cos((dir_segment_l - dir_segment_r) / 2)
                point = model.geo.addPoint(coords[i_][0] - delta_eff * np.sin(dir_segment),
                                           coords[i_][1] + delta_eff * np.cos(dir_segment),
                                           0, sizes[0])
                map_fake_to_real[point] = tag_i
                line = model.geo.addLine(interior_point, point)
                lines_interior_water += [line]
                local_interior += [line]
                link_project_fake_to_fake[(interior_point, point)] = tag_i_m
                link_project_fake_to_fake[(point, interior_point)] = -tag_i_m
                lines_boundary_water += [tag_i_m]
                interior_point = point
        loop = model.geo.addCurveLoop(lines_interior_water)
        surfaces_interior += [model.geo.addPlaneSurface([loop])]

        # then go around ice surface, starting from tag2
        lines_interior_ice = [-tag for tag in lines_interface[1 : -1][::-1]]
        lines_boundary_ice = [-lines_interface[0]]
        local_interior = []
        bound = model.getBoundary([(1, lines_interface[0])], combined=True,
                                  oriented=True, recursive=False)
        interior_point = [b for _, b in bound if b in map_fake_to_real.keys()][0]
        link_fake_to_real[interior_point] = -lines_interface[0]
        skip_interior = False
        # now at tag1 (i = tag1-1)
        for i in range(tag1, tag2-1):
            tag_i = i + 1
            tag_i_m = i
            # far from interface: generate fake point
            dir_segment_l = np.arctan2(coords[i][1] - coords[i-1][1],
                                       coords[i][0] - coords[i-1][0])
            dir_segment_r = np.arctan2(coords[i+1][1] - coords[i][1],
                                       coords[i+1][0] - coords[i][0])
            if abs(dir_segment_l - dir_segment_r) > np.pi:
                dir_segment_l -= 2 * np.pi
                if abs(dir_segment_l - dir_segment_r) > np.pi:
                    dir_segment_l += 4 * np.pi
                assert abs(dir_segment_l - dir_segment_r) <= np.pi
            dir_segment = (dir_segment_l + dir_segment_r) / 2
            delta_eff = delta / np.cos((dir_segment_l - dir_segment_r) / 2)
            point = model.geo.addPoint(coords[i][0] - delta_eff * np.sin(dir_segment),
                                       coords[i][1] + delta_eff * np.cos(dir_segment),
                                       0, sizes[1])
            map_fake_to_real[point] = tag_i
            line = model.geo.addLine(interior_point, point)
            lines_interior_ice += [line]
            local_interior += [line]
            link_project_fake_to_fake[(interior_point, point)] = tag_i_m
            link_project_fake_to_fake[(point, interior_point)] = -tag_i_m
            lines_boundary_ice += [tag_i_m]
            interior_point = point

        bound = model.getBoundary([(1, lines_interface[-1])], combined=True,
                                  oriented=True, recursive=False)
        point = [b for _, b in bound if b in map_fake_to_real.keys()][0]
        line = model.geo.addLine(interior_point, point)
        lines_interior_ice += [line]
        loop = model.geo.addCurveLoop(lines_interior_ice)
        surfaces_interior += [model.geo.addPlaneSurface([loop])]
        
        link_project_fake_to_fake[(interior_point, point)] = tag2-1
        link_project_fake_to_fake[(point, interior_point)] = -(tag2-1)
        lines_boundary_ice += [tag2-1, -lines_interface[-1], -line] + [-tag for tag in local_interior[::-1]]
        loop = model.geo.addCurveLoop(lines_boundary_ice)
        surfaces_boundary += [model.geo.addPlaneSurface([loop])]

        lines_interior_set = set([(1, c) for c in lines_interior_water + lines_interior_ice])
        surface_dim_tags = [(2, tag) for tag in surfaces_interior]
        
        water = surfaces_interior[: -1] + surfaces_boundary[: -1]
        ice = [surfaces_interior[-1], surfaces_boundary[-1]]

    inflow = []
    bath = []
    
    shorelines = get_shorelines(lines, lines_interface, inflow_lines, grounding_lines)
    intersect_lines = [(1, line) for line in lines_interface]

    ice_line_ents = set(grounding_lines)
    bath_line_ents = set(shorelines)
    inflow_line_ents = set(inflow_lines)
    intersect_line_ents = set(intersect_lines)

    if delta is None:
        boundary_curve_tags = ([tag for (_, tag) in shorelines]
                             + [tag for (_, tag) in grounding_lines])
    else:
        boundary_curve_tags = list(set(lines_interior_water + lines_interior_ice)
                                 - set([tag for _, tag in intersect_lines + inflow_lines]))

    model.geo.synchronize()

    if planar_only and delta is None and HORIZONTAL_REFINE:
        h_open = float(m)
        h_ice = ICE_ELEMENT_SIZE or h_open / 2
        h_gl = GL_ELEMENT_SIZE or h_open / 4
        inflow_tags = {tag for _, tag in inflow_lines}
        gl_curves = [c for c in outer_part if abs(c) not in inflow_tags]
        front_curves = list(lines_interface)
        fa = gmsh.model.mesh.field

        ice_const = fa.add("Constant")
        fa.setNumber(ice_const, "VIn", h_ice)
        fa.setNumber(ice_const, "VOut", 1e22)
        fa.setNumbers(ice_const, "SurfacesList", ice)
        fa.setNumber(ice_const, "IncludeBoundary", 1)

        def graded(curves, size):
            d = fa.add("Distance")
            fa.setNumbers(d, "CurvesList", sorted(abs(c) for c in curves))
            fa.setNumber(d, "Sampling", BOUNDARY_FIELD_SAMPLING)
            t = fa.add("Threshold")
            fa.setNumber(t, "InField", d)
            fa.setNumber(t, "SizeMin", size)
            fa.setNumber(t, "SizeMax", h_open)
            fa.setNumber(t, "DistMin", 0.0)
            fa.setNumber(t, "DistMax", SIZE_GRADING)
            return t

        parts = [ice_const, graded(front_curves, h_ice)]
        if gl_curves:
            parts.append(graded(gl_curves, h_gl))
        fmin = fa.add("Min")
        fa.setNumbers(fmin, "FieldsList", parts)
        fa.setAsBackgroundMesh(fmin)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_open)
        print(f"[PLANAR] horizontal refinement: open water {h_open:g} m, under ice "
              f"{h_ice:g} m, grounding line {h_gl:g} m, grading over "
              f"{SIZE_GRADING:g} m ({len(gl_curves)} grounding-line curves, "
              f"{len(front_curves)} front curves)", flush=True)
    elif boundary_curve_tags:
        field_api = gmsh.model.mesh.field
        distance_field = field_api.add("Distance")
        field_api.setNumbers(distance_field, "CurvesList", sorted(boundary_curve_tags))
        field_api.setNumber(distance_field, "Sampling", BOUNDARY_FIELD_SAMPLING)

        boundary_field = field_api.add("Threshold")
        field_api.setNumber(boundary_field, "InField", distance_field)
        field_api.setNumber(boundary_field, "SizeMin", sizes[0])
        field_api.setNumber(boundary_field, "SizeMax", sizes[2])
        field_api.setNumber(boundary_field, "DistMin", float(BOUNDARY_FIELD_DIST_MIN))
        field_api.setNumber(boundary_field, "DistMax", float(BOUNDARY_FIELD_DIST_MAX))
        field_api.setAsBackgroundMesh(boundary_field)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_MAX)

    plot_gmsh_geometry(model)

    model.mesh.generate(2)

    if planar_only:
        # Stop here: the plan-view triangulation is all the direct prism-stacking
        # path needs (see stacked_volume.py).  Everything past this point builds
        # the closed skin that the TetGen path required.
        planar = _extract_planar_mesh(
            model, water, ice,
            [tag for dim, tag in inflow_line_ents if dim == 1])
        gmsh.finalize()
        return planar

    extrude_from = surface_dim_tags
    all_inflow_lines = inflow_line_ents

    if num_of_layers < 1:
        raise ValueError(f"num_of_layers must be >= 1, got {num_of_layers}")

    # Structured extrusion with num_of_layers subdivisions, applied uniformly
    # to every side wall (shoreline, grounding line, inflow, ocean face)
    # via Gmsh's native "Extruded" mesher -- reliable and battle-tested (this
    # is the same mechanism that already worked fine at numElements=[1]).
    # Real per-node depths get assigned later in generate_mesh_mult by
    # rescaling each node's reference-space fraction; that logic already
    # handles any number of levels correctly, so no other change is needed
    # to get genuine sigma-layer structure everywhere. This only works
    # because the domain boundary itself is now clipped at
    # SHORELINE_MIN_THICKNESS instead of the literal (zero-thickness)
    # coastline, so no column is ever thin enough to make this degenerate.
    extrustion = model.geo.extrude(extrude_from, 0, 0, eps, numElements=[num_of_layers])
    model.geo.synchronize()

    next_inflow = []
    inflow_new = []
    shallow_tags = []
    ocean_face = None  # 3D face extruded from the closing segment (outer ocean boundary)
    if delta is not None:
        map_top_to_bottom = {}
        _existing_curves = {t for _, t in model.getEntities(1)}
        _new_curve_pts = {}  # tag -> (start_pt, end_pt) for curves created in the loop below
    for entity in extrustion:
        # Gmsh returns both volume (dim=3) and surface (dim=2) entities; we
        # only keep the surfaces and classify them by adjacency.
        if entity[0] == 3:
            model.geo.remove([entity], recursive=False)
        else:
            bound = model.getBoundary([entity], oriented=False)
            bound_set = set(bound)

            if (delta is not None and len(bound_set.intersection(inflow_line_ents)) == 0
                                  and len(bound_set.intersection(intersect_line_ents)) == 0
                                  and len(bound_set.intersection(lines_interior_set)) > 0):
                #add slanted boundaries
                bound_points = np.sort([tag for _, tag in model.getBoundary([entity], combined=True, recursive=True)])
                # print(bound_points)
                top_points, bottom_points = bound_points[: 2], bound_points[2 :]
                new_bound = model.getBoundary([entity], combined=True, oriented=True, recursive=False)
                model.geo.remove([entity], recursive=False)
                new_bound = [tag for _, tag in [new_bound[0], new_bound[-1], new_bound[1], new_bound[2]]]

                shallow_tags += list(bottom_points)

                # check if sorted correctly
                for j in range(3):
                    old_points = [tag for _, tag in
                                  model.getBoundary([(1, new_bound[j])], combined=True, oriented=True, recursive=False)]
                    for k in range(j + 1, 4):
                        new_points = [tag for _, tag in
                                      model.getBoundary([(1, new_bound[k])], combined=True, oriented=True, recursive=False)]
                        if new_points[0] in old_points or new_points[1] in old_points:
                            if k > j + 1:
                                new_entry = new_bound.pop(k)
                                new_bound = new_bound[: j + 1] + [new_entry] + new_bound[j + 1 :]
                            break

                idx_down = None
                _both_fake_top = False
                _conn_flip_at_idx_down = False
                old_points = []
                for j, bound in enumerate(new_bound):
                    new_points = [tag for _, tag in
                                  model.getBoundary([(1, bound)], combined=True, oriented=True, recursive=False)]

                    if new_points[0] in top_points:
                        if new_points[1] in top_points:
                            _lv = link_project_fake_to_fake[(new_points[0], new_points[1])]
                            if _lv < 0 and -_lv in _existing_curves:
                                # Negative value: the stored tag is a valid reversed reference
                                # to an already-synchronized curve.  Use it as-is; creating a
                                # new addLine would duplicate the same edge with identical
                                # Vertex pointers, which Gmsh rejects when building the loop.
                                print(f"[DBG-NEG] entity={entity[1]} using reversed ref _lv={_lv}", flush=True)
                                pass
                            elif _lv not in _existing_curves:
                                # The stored value is an orphaned coord-index that doesn't
                                # correspond to any existing synchronized curve — create fresh.
                                _r0 = map_fake_to_real[new_points[0]]
                                _r1 = map_fake_to_real[new_points[1]]
                                print(f"[DBG-ORPH] entity={entity[1]} stored_lv={_lv} r0={_r0} r1={_r1}", flush=True)
                                _lv = model.geo.addLine(_r0, _r1)
                                print(f"[DBG-ORPH] new_lv={_lv}", flush=True)
                                _new_curve_pts[_lv] = (_r0, _r1)
                            new_bound[j] = _lv
                            if j == 0:
                                _both_fake_top = True
                        else:
                            map_top_to_bottom[new_points[0]] = model.geo.addLine(map_fake_to_real[new_points[0]], new_points[1])
                            _new_curve_pts[map_top_to_bottom[new_points[0]]] = (map_fake_to_real[new_points[0]], new_points[1])
                            new_bound[j] = map_top_to_bottom[new_points[0]]
                    elif new_points[1] in top_points:
                        map_top_to_bottom[new_points[1]] = model.geo.addLine(map_fake_to_real[new_points[1]], new_points[0])
                        _new_curve_pts[map_top_to_bottom[new_points[1]]] = (map_fake_to_real[new_points[1]], new_points[0])
                        new_bound[j] = - map_top_to_bottom[new_points[1]]
                    else:
                        idx_down = j
                        new_bound[j] *= -1
                    if j and old_points[1] != new_points[0]:
                        if j == 1 and old_points[0] in new_points:
                            new_bound[0] *= -1
                            old_points = old_points[::-1]
                        if old_points[1] == new_points[1]:
                            new_bound[j] *= -1
                            new_points = new_points[::-1]
                            if j == idx_down:
                                _conn_flip_at_idx_down = True
                    old_points = new_points

                # When both top endpoints are interior fake points, the else-branch
                # *=-1 on the bottom edge is wrong.  Undo it — but only when the
                # connectivity check inside the j-loop didn't already undo it.
                if _both_fake_top and idx_down is not None and not _conn_flip_at_idx_down:
                    new_bound[idx_down] *= -1

                print(f"[DBG] entity={entity[1]} both_fake={_both_fake_top} conn_flip={_conn_flip_at_idx_down} idx_down={idx_down} new_bound={new_bound}", flush=True)
                try:
                    new_loop = model.geo.addCurveLoop(new_bound, reorient=True)
                except Exception:
                    # Print the actual endpoint chain so we can see the gap.
                    # _new_curve_pts is used as a fallback for curves created above that
                    # haven't been synchronized yet (model.getBoundary would fail on those).
                    def _get_curve_pts(ct):
                        abs_ct = abs(ct)
                        if abs_ct in _new_curve_pts:
                            s, e = _new_curve_pts[abs_ct]
                        else:
                            _p = [_t for _, _t in model.getBoundary([(1, abs_ct)], combined=True, oriented=True)]
                            s, e = _p[0], _p[1]
                        return (s, e) if ct > 0 else (e, s)

                    print(f"[DBG-FAIL] entity={entity[1]} addCurveLoop failed — checking chain:", flush=True)
                    prev_end = None
                    for _k, _ct in enumerate(new_bound):
                        _start, _end = _get_curve_pts(_ct)
                        _ok = (prev_end is None or prev_end == _start)
                        print(f"  [{_k}] curve={_ct} {_start}->{_end} {'OK' if _ok else 'BREAK!'}", flush=True)
                        prev_end = _end
                    _start0, _ = _get_curve_pts(new_bound[0])
                    print(f"  close: {prev_end}->{_start0} {'OK' if prev_end == _start0 else 'BREAK!'}", flush=True)
                    # Try reversed winding as fallback before giving up.
                    new_bound_rev = [-t for t in new_bound[::-1]]
                    try:
                        new_loop = model.geo.addCurveLoop(new_bound_rev)
                        new_bound = new_bound_rev
                        print(f"[DBG-REV] entity={entity[1]} reversed winding worked, loop={new_loop}", flush=True)
                    except Exception as _e2:
                        print(f"[DBG-REV-FAIL] entity={entity[1]} reversed winding also failed: {_e2}", flush=True)
                        raise
                else:
                    print(f"[DBG-OK] entity={entity[1]} loop={new_loop}", flush=True)
                entity = (2, model.geo.addPlaneSurface([new_loop]))
                bound = [(1, l) for l in new_bound]
                bound_set = set(bound)
            
            if len(bound_set.intersection(ice_line_ents)) == len(bound_set):
                # Faces touching the grounding line belong to the ice shelf.
                ice.append(entity[1])
            elif (1, closing_seg_line) in bound_set:
                # The open-ocean face is added directly to inflow (no fake-point
                # triangle fixing needed) so it gets INFLOW_TAG in the physical group.
                inflow.append(entity[1])
                ocean_face = entity[1]
            elif len(bound_set.intersection(bath_line_ents)) > 0:
                # Faces attached to the outer shoreline become bathymetry walls.
                bath.append(entity[1])
            elif len(bound_set.intersection(inflow_line_ents)) > 0:
                # Inflow surfaces are tracked separately for boundary conditions.
                inflow_new.append(entity[1])
                next_inflow = next_inflow + bound
            elif len(bound_set.intersection(intersect_line_ents)) > 0:
                # The inlet cut-out is discarded after the first extrude pass.
                # Use recursive=False so bounding curves survive for adjacent surface loops.
                model.geo.remove([entity], recursive=False)
            else:
                # The bottom-most cap defaults to bathymetry.
                bath.append(entity[1])

    for ent in inflow_new:
        # fix inflow_surfaces by adding triangles if needed
        bound_points = set(model.getBoundary([(2, ent)], combined=True, recursive=True))

        bound_points_fake = bound_points.intersection(flat_points_fake) if delta is not None else set()
        if len(bound_points_fake) > 0:
            new_bound = model.getBoundary([(2, ent)], combined=True, oriented=True, recursive=False)
            new_bound = [tag for _, tag in [new_bound[0], new_bound[-1], new_bound[1], new_bound[2]]]
            # check if sorted correctly
            for j in range(3):
                old_points = [tag for _, tag in 
                              model.getBoundary([(1, new_bound[j])], combined=True, oriented=True, recursive=False)]
                for k in range(j + 1, 4):
                    new_points = [tag for _, tag in 
                                  model.getBoundary([(1, new_bound[k])], combined=True, oriented=True, recursive=False)]
                    if new_points[0] in old_points or new_points[1] in old_points:
                        if k > j + 1:
                            new_entry = new_bound.pop(k)
                            new_bound = new_bound[: j + 1] + [new_entry] + new_bound[j + 1 :]
                        break

            bound_l_points = model.getBoundary([(1, new_bound[-1])], combined=True, oriented=True, recursive=False)
            bound_r_points = model.getBoundary([(1, new_bound[1])], combined=True, oriented=True, recursive=False)
            if new_bound[0] < 0:
                bound_r_points, bound_l_points = bound_l_points, bound_r_points

            for point_fake in bound_points_fake:
                # Create a gap-filler triangle (BATH) between the vertical inflow wall
                # and the adjacent slant surface. Do NOT modify new_bound so the inflow
                # face stays a strict 4-curve rectangle (vertical cliff).
                if point_fake in bound_l_points:
                    side_edge = new_bound[-1]
                    gap_loop = model.geo.addCurveLoop(
                        [link_fake_to_real[point_fake[1]],
                         map_top_to_bottom[point_fake[1]],
                         side_edge],
                        reorient=True
                    )
                    bath.append(model.geo.addPlaneSurface([gap_loop]))
                if point_fake in bound_r_points:
                    side_edge = new_bound[1]
                    gap_loop = model.geo.addCurveLoop(
                        [link_fake_to_real[point_fake[1]],
                         map_top_to_bottom[point_fake[1]],
                         side_edge],
                        reorient=True
                    )
                    bath.append(model.geo.addPlaneSurface([gap_loop]))

        inflow.append(ent)
    inflow_line_ents = set(next_inflow)
    all_inflow_lines = all_inflow_lines.union(inflow_line_ents)
        
    model.geo.synchronize()
    
    all_inflow_lines = [int(tag) for dim, tag in all_inflow_lines]
    shoreline_curve_tags = [int(tag) for (_, tag) in shorelines]
    inflow_curve_tags = list(all_inflow_lines)
    grounding_curve_tags = [int(tag) for (_, tag) in grounding_lines]
    intersection_curve_tags = [int(tag) for tag in lines_interface]

    coarse_boundary_lookup = {
        "inflow": inflow_curve_tags,
        "inflow_lines": inflow_curve_tags,
        "shoreline": shoreline_curve_tags,
        "bathymetry": shoreline_curve_tags,
        "grounding": grounding_curve_tags,
        "grounding_line": grounding_curve_tags,
        "intersection": intersection_curve_tags,
    }

    coarse_curve_tags: set[int] = set()
    valid_lookup_keys = tuple(sorted(coarse_boundary_lookup))
    for spec in band_lcmax_boundaries:
        if isinstance(spec, str):
            key = spec.strip().lower()
            try:
                coarse_curve_tags.add(int(key))
                continue
            except ValueError:
                pass
            if key not in coarse_boundary_lookup:
                options = ", ".join(valid_lookup_keys) or "<none>"
                raise ValueError(
                    f"Unknown boundary '{spec}' requested for LC_MAX. "
                    f"Valid names: {options}, or provide explicit curve tags."
                )
            coarse_curve_tags.update(int(tag) for tag in coarse_boundary_lookup[key])
        else:
            coarse_curve_tags.add(int(spec))

    # Physical groups map mesh regions to boundary conditions in downstream
    # solvers (FEniCS, etc.).  Keep them descriptive so post-processing stays
    # readable.
    model.geo.addPhysicalGroup(2, water, tag=WATER_SURFACE_TAG, name="Water Surface")
    model.geo.addPhysicalGroup(2, ice, tag=ICE_TAG, name="Ice")
    model.geo.addPhysicalGroup(2, bath, tag=BATHYMETRY_TAG, name="Bathymetry")
    model.geo.addPhysicalGroup(2, inflow, tag=INFLOW_TAG, name="Inflow")
    model.geo.addPhysicalGroup(1, all_inflow_lines, tag=INFLOW_LINE_TAG, name="Inflow lines")
  
    model.geo.synchronize()

    # No per-face transfinite setup needed anymore: the single domain-wide
    # extrude() above already gives every side wall (inflow, ocean_face,
    # bath, ice) num_of_layers structure natively via Gmsh's "Extruded"
    # mesher, now that the domain boundary itself never gets thin enough to
    # make that degenerate (see SHORELINE_MIN_THICKNESS).

    # Clear the mesh from the first generate(2) and the extruded mesh that
    # model.geo.extrude() produced automatically.  Without this, Gmsh's
    # structured extruder leaves orphaned nodes on the deleted surfaces that it
    # then tries to find on the new slant surfaces (surface 181 etc.), causing
    # "Could not find extruded node" errors.
    model.mesh.clear()
    model.mesh.generate(2)
    if SURFACE_UNIFORM_REFINEMENT_PASSES > 0:
        _run_uniform_refinement(SURFACE_UNIFORM_REFINEMENT_PASSES, "2D")

    if ocean_face is not None:
        _d2_et, _d2_etags, _d2_en = model.mesh.getElements(2, ocean_face)
        _d2_nodes = set(int(t) for en in _d2_en for t in en)
        _d2_z = sorted(set(round(float(model.mesh.getNode(int(t))[0][2]), 4)
                           for t in _d2_nodes))
        print(f"[DIAG2D-B] ocean face after gen(2): {sum(len(t) for t in _d2_etags)} tris, "
              f"{len(_d2_nodes)} nodes, z-unique={_d2_z[:5]}...{_d2_z[-3:]}", flush=True)

    # Store the XY footprint of important curves so the 3D stage can spot when
    # it is adjusting nodes that lie on the shoreline or grounding line.
    xy_shoreline = []
    shore_tags = [tag for (dim, tag) in shorelines]
    for line in shore_tags:
        shoreline_tags, shoreline_coords, _ = model.mesh.getNodes(1, line, 
                                                                  includeBoundary=True, 
                                                                  returnParametricCoord=False)
        shoreline_coords = shoreline_coords.reshape((shoreline_tags.size, 3))
        for coord in shoreline_coords:
            x = coord[0]
            y = coord[1]
            xy_shoreline.append((x,y))
    
    xy_grounding = []            
    grounding_line_tags = [tag for (dim, tag) in grounding_lines]
    for line in grounding_line_tags:
        grounding_tags, grounding_coords, _ = model.mesh.getNodes(1, line, 
                                                                  includeBoundary=True, 
                                                                  returnParametricCoord=False)
        grounding_coords = grounding_coords.reshape((grounding_tags.size, 3))
        for coord in grounding_coords:
            x = coord[0]
            y = coord[1]
            xy_grounding.append((x,y))
    
    xy_shallow = []            
    for tag in shallow_tags:
        shallow_tags_arr, shallow_coords, _ = model.mesh.getNodes(0, tag, includeBoundary=True, 
                                                                  returnParametricCoord=False)
        shallow_coords = shallow_coords.reshape((shallow_tags_arr.size, 3))
        for coord in shallow_coords:
            x = coord[0]
            y = coord[1]
            xy_shallow.append((x,y))
    
    if ocean_face is not None:
        _d_et, _d_etags, _d_en = model.mesh.getElements(2, ocean_face)
        _d_nodes = set(int(t) for en in _d_en for t in en)
        _d_z = sorted(set(round(float(model.mesh.getNode(int(t))[0][2]), 4) for t in _d_nodes))
        print(f"[DIAG2D] ocean face in 2D mesh: {sum(len(t) for t in _d_etags)} tris, "
              f"{len(_d_nodes)} nodes, z-unique={_d_z}", flush=True)

    mesh_filename = ensure_msh_path(filename)

    gmsh.write(mesh_filename)
        
    gmsh.finalize()

    try:
        xdmf_path = write_legacy_xdmf(mesh_filename)
        print(f"Wrote XDMF for FEniCS/ParaView: {xdmf_path} (+ _volume.xdmf)")
    except Exception as err:
        print(f"[ryder_3D_meshing] Failed to write XDMF next to {mesh_filename}: {err}", file=sys.stderr)
    
    mesh2D = (mesh_filename, xy_shoreline, xy_grounding, xy_shallow, closing_seg_line, ocean_face)

    return mesh2D


def generate_mesh_mult(outline, intersect, grounding_line, m, eps, category_data,
                       bathymetry_data, highres_data, thickness_data, surface_pos_data, 
                       scale = 1, num_of_layers = 2, adapt = False, adaptive_scales = (1/4, 2),
                       optimize: bool = ENABLE_VOLUME_OPTIMIZATION_DEFAULT, stack = 25, interpolate = True,
                       mesh_filename: Optional[str] = None, delta: Optional[float] = None,
                       band_lcmax_boundaries: Optional[Sequence[BandBoundarySpec]] = None):
    """Extrude the 2D surface mesh into 3D and sculpt it with raster data."""
    
    eps_eff = min(eps, -1)

    mesh2D = generate_2D_mesh(
        outline, intersect, grounding_line, 
        category_data, m, eps_eff, adapt,
        adaptive_scales=adaptive_scales,
        num_of_layers=num_of_layers,
        filename=mesh_filename,
        delta=delta,
        band_lcmax_boundaries=band_lcmax_boundaries,
    )

    mesh2d_path, xy_shoreline, xy_grounding, xy_shallow, closing_seg_line, ocean_face = mesh2D

    if mesh_filename:
        output_path = mesh2d_path
        meshname = Path(output_path).stem
    else:
        if scale > 1 and num_of_layers == 2:
            folder = "scaled/"
            meshname = f"S_{scale}_{m}"
        elif num_of_layers > 2 and scale > 1:
            folder = "combo/"
            meshname = f"C_{num_of_layers}_{scale}_{m}"
        elif num_of_layers > 2:
            folder = "layered/"
            meshname = f"L_{num_of_layers}_{m}"
        elif stack > 0:
            folder = "stacked/"
            meshname = f"P_{stack}_{m}"
        else:
            folder = "unstructured/"
            meshname = f"M_{m}"
        
        if optimize:
            meshname = f"{meshname}_opt"
        if adapt:
            folder = "adaptive/"
            meshname = f"A_{meshname}"
        output_path = ensure_msh_path(Path(folder) / meshname)
    
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", GMESH_NUM_THREADS)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    model = gmsh.model
    model.add("3D")
    
    # Re-open the 2D mesh we just wrote so we can extrude it inside the fresh
    # 3D model.  `merge` imports both geometry and mesh entities.
    gmsh.merge(mesh2d_path)

    model.geo.synchronize()
    
    bathymetry_array = bathymetry_data.ReadAsArray()
    bathymetry_trans = bathymetry_data.GetGeoTransform()
    highres_array = highres_data.ReadAsArray()
    highres_trans = highres_data.GetGeoTransform()
    thickness_array = thickness_data.ReadAsArray()
    thickness_trans = thickness_data.GetGeoTransform()
    surface_pos_array = surface_pos_data.ReadAsArray()
    
    def floor_depth(x, y):
        """Sea-floor depth at (x, y), with a minimum water column enforced.

        Wraps get_bathymetric_depth and then, if MIN_WATER_COLUMN > 0, pushes
        the floor down so that the column between the local top of the water
        (the ice draft where there is ice, otherwise the sea surface at z=0)
        and the floor is at least MIN_WATER_COLUMN thick.  This is what keeps
        sigma layers non-degenerate at interior shallow nodes, which the
        shoreline clip cannot reach.  With MIN_WATER_COLUMN == 0 the result is
        identical to calling get_bathymetric_depth directly.
        """
        floor = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array,
                                      highres_trans, highres_array, -MIN_WATER_DEPTH,
                                      highres=True, interpolate=interpolate)
        if MIN_WATER_COLUMN <= 0.0:
            return floor
        top = get_ice_depth(x, y, thickness_trans, thickness_array,
                            surface_pos_array, bathymetry_array,
                            -MIN_ICE_THICKNESS, interpolate=interpolate)
        top = min(top, 0.0)
        return min(floor, top - MIN_WATER_COLUMN)

    bathymetry_node_tags, bathymetry_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, BATHYMETRY_TAG)
    surface_node_tags, surface_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, WATER_SURFACE_TAG)
    ice_node_tags, ice_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, ICE_TAG)
    inflow_node_tags, inflow_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, INFLOW_TAG)
    
    bathymetry_node_coords = bathymetry_node_coords.reshape((bathymetry_node_tags.size, 3))
    surface_node_coords = surface_node_coords.reshape((surface_node_tags.size, 3))
    ice_node_coords = ice_node_coords.reshape((ice_node_tags.size, 3))
    inflow_node_coords = inflow_node_coords.reshape((inflow_node_tags.size, 3))

    # Shared setup for both sigma-layer seeding and stack-interval seeding.
    xy_ice = [(i[0], i[1]) for i in ice_node_coords]
    xy_surface = [(i[0], i[1]) for i in surface_node_coords]
    xy_water_ice_intersect = set(xy_ice).intersection(set(xy_surface))
    stacked_points = []
    inflow_stack_points = {}
    if adapt:
        sizes = (adaptive_scales[0]*m, m, adaptive_scales[1]*m)
    else:
        sizes = (m, m, m)

    if stack > 0:
        pass  # stacked_points / inflow_stack_points populated in the loops below
        
    i = 0
    for tag in bathymetry_node_tags:
        if tag not in surface_node_tags:
            x = bathymetry_node_coords[i][0]
            y = bathymetry_node_coords[i][1]
            z_norm = bathymetry_node_coords[i][2]  # normalized depth from 2D mesh

            if np.isnan(x) or np.isnan(y):
                i += 1
                continue

            if True: #not point_in_set((x,y), xy_shoreline) or point_in_set((x,y), xy_grounding):
                z = floor_depth(x, y)

                # Proportion of the way from surface to floor in the normalized 2D
                # mesh.  For bottom-cap nodes z_norm==eps_eff so fraction==1 (unchanged).
                # For side-wall/slant intermediate nodes this distributes them evenly
                # instead of collapsing all layers to the same depth.
                fraction = z_norm / eps_eff
                model.mesh.setNode(tag, [x, y, scale * fraction * z], [])
                
                # Sigma-layer seed points: add one geometry point per intermediate
                # sigma level at this (x,y) floor location.  The 3D mesher is forced
                # to put a node at each of these positions, producing structured layers.
                if (num_of_layers > 1
                        and abs(fraction - 1.0) < 1e-6
                        and not point_in_set((x, y), xy_shoreline)
                        and (not point_in_set((x, y), xy_ice) or point_in_set((x, y), xy_water_ice_intersect))
                        and tag not in inflow_node_tags):
                    for k in range(1, num_of_layers):
                        frac = k / num_of_layers
                        point = model.geo.addPoint(x, y, scale * frac * z, sizes[2])
                        stacked_points.append(point)
                elif (stack > 0
                        and not point_in_set((x,y), xy_shallow)
                        and abs(fraction - 1.0) < 1e-6
                        and not point_in_set((x,y), xy_shoreline)):
                    if (((not point_in_set(((x,y)), xy_ice)) and (tag not in inflow_node_tags))
                            or (point_in_set((x,y), xy_water_ice_intersect))):
                        current_depth = 0
                        while current_depth - 1.5*stack >= z:
                            current_depth = current_depth - stack
                            geo_z = current_depth / z * eps_eff
                            point = model.geo.addPoint(x, y, geo_z, sizes[2])
                            stacked_points.append(point)
        i += 1

    i = 0
    for tag in ice_node_tags:
        if tag not in surface_node_tags and tag not in bathymetry_node_tags:
            x = ice_node_coords[i][0]
            y = ice_node_coords[i][1]
            z_norm = ice_node_coords[i][2]

            if np.isnan(x) or np.isnan(y):
                i += 1
                continue

            z = get_ice_depth(x, y, thickness_trans, thickness_array,
                              surface_pos_array, bathymetry_array,
                              -MIN_ICE_THICKNESS, interpolate = interpolate)
            fraction = z_norm / eps_eff
            gmsh.model.mesh.setNode(tag, [x, y, scale * fraction * z], [])
            
            if num_of_layers > 1 and not point_in_set((x, y), xy_shallow):
                floor = floor_depth(x, y)
                water_col = floor - z  # both negative; floor < z (deeper)
                for k in range(1, num_of_layers):
                    frac = k / num_of_layers
                    sigma_z = z + frac * water_col  # from ice draft toward floor
                    point = model.geo.addPoint(x, y, scale * sigma_z, sizes[1])
                    stacked_points.append(point)
            elif stack > 0 and not point_in_set((x,y), xy_shallow):
                current_depth = z
                floor = floor_depth(x, y)
                while current_depth - 1.5*stack >= floor:
                    current_depth = current_depth - stack
                    geo_z = current_depth / floor * eps_eff
                    point = model.geo.addPoint(x, y, geo_z, sizes[1])
                    stacked_points.append(point)
        i += 1

    # Inflow face: keep x,y fixed (vertical cliff) and distribute z proportionally
    # between 0 (surface) and real bathymetric depth.  Without this, the structured
    # extrusion nodes at z=-0.25, -0.5, -0.75 (normalized) are never repositioned,
    # so the 4 layers are wildly unequal in the final mesh.

    # Pre-collect valid floor depths from pure-inflow floor nodes (fraction≈1).
    # Sigma interior nodes near the outer closing segment may be at positions
    # outside the raster domain and would otherwise be clamped to -MIN_WATER_DEPTH
    # (≈ 0), collapsing all sigma levels to z≈0 and destroying the face structure.
    _inflow_floor_xy = []
    _inflow_floor_z  = []
    _j = 0
    for _tag in inflow_node_tags:
        _xj = inflow_node_coords[_j][0]
        _yj = inflow_node_coords[_j][1]
        _frac_j = inflow_node_coords[_j][2] / eps_eff
        if (abs(_frac_j - 1.0) < 1e-6
                and _tag not in surface_node_tags
                and _tag not in bathymetry_node_tags
                and _tag not in ice_node_tags
                and not np.isnan(_xj) and not np.isnan(_yj)):
            _zj = floor_depth(_xj, _yj)
            if _zj < -1.0:  # keep only genuinely deep reference values
                _inflow_floor_xy.append((_xj, _yj))
                _inflow_floor_z.append(_zj)
        _j += 1
    _inflow_floor_xy_arr = np.array(_inflow_floor_xy) if _inflow_floor_xy else None
    _inflow_floor_z_arr  = np.array(_inflow_floor_z)  if _inflow_floor_z  else None

    i = 0
    for tag in inflow_node_tags:
        x = inflow_node_coords[i][0]
        y = inflow_node_coords[i][1]
        z_norm = inflow_node_coords[i][2]  # normalized: 0=surface, eps_eff=floor
        if (tag not in surface_node_tags
                and tag not in bathymetry_node_tags
                and tag not in ice_node_tags
                and not np.isnan(x) and not np.isnan(y)):
            z_real = floor_depth(x, y)
            # Fall back to nearest inflow floor node depth when the raster returns
            # a clamped/invalid value (e.g. outer closing-segment positions that
            # lie outside the raster extent).
            if z_real > -1.0 and _inflow_floor_xy_arr is not None:
                dists = np.linalg.norm(_inflow_floor_xy_arr - np.array([x, y]), axis=1)
                z_real = float(_inflow_floor_z_arr[np.argmin(dists)])
            fraction = z_norm / eps_eff  # 0 at surface, 1 at floor
            model.mesh.setNode(tag, [x, y, scale * fraction * z_real], [])
        i += 1


    # Seed sigma-level volume points from the ocean face (and any other inflow
    # floor nodes).  Bath floor nodes are excluded by the bath loop's condition
    # `tag not in inflow_node_tags`, so these inflow floor nodes would otherwise
    # get no sigma seeds.
    if num_of_layers > 1:
        i = 0
        for tag in inflow_node_tags:
            x = inflow_node_coords[i][0]
            y = inflow_node_coords[i][1]
            z_norm = inflow_node_coords[i][2]
            if (not np.isnan(x) and not np.isnan(y)
                    and tag not in surface_node_tags
                    and tag not in bathymetry_node_tags
                    and tag not in ice_node_tags):
                fraction = z_norm / eps_eff
                if abs(fraction - 1.0) < 1e-6:
                    z_real = floor_depth(x, y)
                    for k in range(1, num_of_layers):
                        frac = k / num_of_layers
                        point = model.geo.addPoint(x, y, scale * frac * z_real, sizes[2])
                        stacked_points.append(point)
            i += 1

    if stack > 0:
        inflow_stack_points = {}
        inflow_ents = gmsh.model.getEntitiesForPhysicalGroup(2, INFLOW_TAG)
        for ent in inflow_ents:
            tags, coords, _ = model.mesh.getNodes(dim=2, tag=ent, includeBoundary=True)
            coords = coords.reshape((tags.size, 3))
            surface_stack_points = []
            for tag, coord in zip(tags, coords):
                if tag not in surface_node_tags:
                    x, y, floor_scaled = coord
                    # coord[2] is the scaled mesh z (scale * fraction * real_floor).
                    # Convert back to real depth for the loop, then map to geometry
                    # space for the embedded point z.
                    floor_real = floor_scaled / scale
                    current_depth = 0.
                    while current_depth - 1.5*stack >= floor_real:
                        current_depth = current_depth - stack
                        geo_z = current_depth / floor_real * eps_eff
                        point = model.geo.addPoint(x, y, geo_z, sizes[2])
                        surface_stack_points.append(point)
            inflow_stack_points[ent] = surface_stack_points
        
        gmsh.model.geo.synchronize()
        
        for surface in inflow_stack_points.keys():
            model.mesh.embed(0, inflow_stack_points[surface], 2, surface)

    gmsh.model.geo.synchronize()
    model.mesh.generate(2)
    gmsh.model.geo.synchronize()

    surf_tags = []
    for group_tag in SURFACE_PHYSICAL_TAGS:
        surf_tags.extend(model.getEntitiesForPhysicalGroup(2, group_tag))

    surf_loop = gmsh.model.geo.addSurfaceLoop(surf_tags)
    vol = gmsh.model.geo.addVolume([surf_loop])

    if stacked_points and EMBED_SIGMA_SEEDS:
        gmsh.model.geo.synchronize()
        model.mesh.embed(0, stacked_points, 3, vol)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        print(f"[SIGMA SEED] embedded {len(stacked_points)} sigma-level seed points in volume")
    elif stacked_points:
        print(f"[SIGMA SEED] SKIPPED embedding {len(stacked_points)} seed points "
              f"(EMBED_SIGMA_SEEDS=0)")
    gmsh.model.geo.addPhysicalGroup(3, [vol], 5, name='Water')

    model.geo.synchronize()

    if num_of_layers > 2:
        if num_of_layers in [15] or (num_of_layers in [20] and adapt):
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", INITIAL_DELAUNAY_TOL)
        if scale > 1:
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", INITIAL_DELAUNAY_TOL)

    # Slant surfaces can meet the flat water cap at very shallow dihedral angles
    # (physically valid for a nearly-flat fjord floor).  The default 0.1-degree
    # threshold flags these as "overlapping" and aborts generate(3).  Lower to
    # 0.01 degrees so only true overlaps are rejected.
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)

    # TetGen (default algorithm 1) respects embedded 0D seed points (needed for
    # sigma layers).  The AngleToleranceFacetOverlap=0.01 above is enough to
    # avoid false "overlapping" rejections on the slant surfaces.
    # Netgen (4) ignores embedded points — do NOT force it here.

    if optimize:
        if ENABLE_VOLUME_SMOOTHING:
            gmsh.option.setNumber("Mesh.Smoothing", MESH_SMOOTHING_ITERS)
        else:
            gmsh.option.setNumber("Mesh.Smoothing", 0)
    if optimize and ENABLE_VOLUME_QUALITY_ALGORITHM:
        try:
            gmsh.option.setNumber("Mesh.Algorithm3D", MESH_ALGORITHM3D_QUALITY)
            print(f"[3D] Using Mesh.Algorithm3D = {MESH_ALGORITHM3D_QUALITY} (quality-oriented)")
        except Exception:
            print("[3D] Could not set Mesh.Algorithm3D; proceeding with default.")

    model.mesh.generate(3)

    if VOLUME_UNIFORM_REFINEMENT_PASSES > 0:
        _run_uniform_refinement(VOLUME_UNIFORM_REFINEMENT_PASSES, "3D")
    
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    coords = list(coords)

    final_volume_quality = None
    if optimize:
        final_volume_quality = _improve_volume_quality(
            VOLUME_QUALITY_TARGET,
            VOLUME_QUALITY_MAX_PASSES,
            VOLUME_QUALITY_METHODS,
        )

    model.mesh.removeDuplicateNodes()
    model.mesh.removeDuplicateElements()
    model.mesh.reclassifyNodes()
    model.geo.synchronize()

    if final_volume_quality is not None:
        print(f"Volume min quality after targeted optimization: {final_volume_quality:.3f}")

    surface_node_tags, surface_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, WATER_SURFACE_TAG)
    ice_node_tags, ice_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, ICE_TAG)
    bath_node_tags, bath_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, BATHYMETRY_TAG)
    water_node_tags, water_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(3, 5)
    
    ice_node_coords = ice_node_coords.reshape((ice_node_tags.size, 3))
    water_node_coords = water_node_coords.reshape((water_node_tags.size, 3))
        
    xmin = min(water_node_coords[:,0])
    ymin = min(water_node_coords[:,1])
    
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.removeDuplicateElements()
    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.geo.synchronize()

    rotated_nodes = []
    min_rot_x = float("inf")
    min_rot_y = float("inf")

    for idx, tag in enumerate(water_node_tags):
        x = water_node_coords[idx][0]
        y = water_node_coords[idx][1]
        z = water_node_coords[idx][2]
        if ((tag not in surface_node_tags)):
            z = z/scale
        x_translated = x - xmin
        y_translated = y - ymin
        x_rot, y_rot, z_rot = rotate_coordinates(x_translated, y_translated, z)
        rotated_nodes.append((tag, x_rot, y_rot, z_rot))
        if x_rot < min_rot_x:
            min_rot_x = x_rot
        if y_rot < min_rot_y:
            min_rot_y = y_rot

    if rotated_nodes:
        shift_x = -min_rot_x
        shift_y = -min_rot_y
        for tag, x_rot, y_rot, z_rot in rotated_nodes:
            gmsh.model.mesh.setNode(tag, [x_rot + shift_x, y_rot + shift_y, z_rot], [])

    # Snap all inflow face nodes to a single x-plane so the inflow wall is
    # perfectly vertical. Query by entity (not physical group) so that boundary
    # and edge nodes reclassified to dim<2 by reclassifyNodes() are included.
    inflow_snap_tags = set()
    for ent in gmsh.model.getEntitiesForPhysicalGroup(2, INFLOW_TAG):
        tags, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=ent, includeBoundary=True)
        inflow_snap_tags.update(int(t) for t in tags)

    if inflow_snap_tags:
        snap_coords = {t: gmsh.model.mesh.getNode(t)[0] for t in inflow_snap_tags}
        target_x = max(c[0] for c in snap_coords.values())
        for t, coord in snap_coords.items():
            if coord[0] != target_x:
                gmsh.model.mesh.setNode(t, [target_x, coord[1], coord[2]], [])

    # Snap the outer ocean boundary face to a single x-plane so it is perfectly vertical.
    # Identify it as the BATHYMETRY entity whose nodes sit at the highest mean X
    # in the rotated coordinate system (the open-ocean end of the domain).
    # Snap the outer ocean boundary face to a single x-plane so it is perfectly vertical.
    # Use getElements to get ALL nodes on the face (reclassifyNodes moves boundary nodes
    # to dim<2, so getNodes misses them; getElements gets all corner nodes of each triangle).
    bath_ents = gmsh.model.getEntitiesForPhysicalGroup(2, BATHYMETRY_TAG)
    max_x_mean = -np.inf
    ocean_ent_id = None
    for ent in bath_ents:
        _, _, node_tags_per_elem = gmsh.model.mesh.getElements(dim=2, tag=ent)
        all_nodes = set(int(n) for ns in node_tags_per_elem for n in ns)
        if len(all_nodes) > 0:
            xs = np.array([gmsh.model.mesh.getNode(t)[0][0] for t in all_nodes])
            if xs.mean() > max_x_mean:
                max_x_mean = xs.mean()
                ocean_ent_id = ent
    if ocean_ent_id is not None:
        _, _, node_tags_per_elem = gmsh.model.mesh.getElements(dim=2, tag=ocean_ent_id)
        ocean_snap_tags = set(int(n) for ns in node_tags_per_elem for n in ns)
        snap_coords = {t: gmsh.model.mesh.getNode(t)[0] for t in ocean_snap_tags}
        target_x = max(c[0] for c in snap_coords.values())
        for t, coord in snap_coords.items():
            if coord[0] != target_x:
                gmsh.model.mesh.setNode(t, [target_x, coord[1], coord[2]], [])

    model.geo.synchronize()

    gmsh.write(output_path)
    gmsh.finalize()
    try:
        xdmf_path = write_legacy_xdmf(output_path)
        print(f"Wrote XDMF for FEniCS/ParaView: {xdmf_path} (+ _volume.xdmf)")
    except Exception as err:
        print(f"[ryder_3D_meshing] Failed to write XDMF next to {output_path}: {err}", file=sys.stderr)
    dof = len(water_node_tags)
    return dof, meshname



def createdomainmask(elevationB):

    # bathymetry is -9999 outside domain
    domainmask = gdal_calc.Calc(
        calc=["(A!=-9999)"],
        A=elevationB,
        outfile="new_data/domainmask_nonresamp.tif",
        NoDataValue=0,
        overwrite=True)

    seathickness_data = readraster('new_data/seathickness.tif')

    domainmask.ReadAsArray() # WTF, why do i need this..!

    # for recropping
    gt = seathickness_data.GetGeoTransform()

    nx = seathickness_data.RasterXSize
    ny = seathickness_data.RasterYSize

    xmin = gt[0]
    xmax = gt[0] + nx * gt[1]
    ymax = gt[3]
    ymin = gt[3] + ny * gt[5]   # gt[5] is negative


    gdal.Warp(
        "new_data/domainmask_noncropped.tif",
        "new_data/domainmask_nonresamp.tif",
        dstSRS=seathickness_data.GetProjection(),
        xRes=gt[1],
        yRes=abs(gt[5]),
        targetAlignedPixels=True,
        outputBounds=(xmin, ymin, xmax, ymax),
        resampleAlg='bilinear'
    )

    domainmask_resamp = gdal.Open("new_data/domainmask_noncropped.tif")

    # CROP

    arrSeaThick = seathickness_data.ReadAsArray()      # <-- this is a numeric np.ndarray
    domainmask_resamp_band = domainmask_resamp.GetRasterBand(1)
    arrMask_resamp = domainmask_resamp_band.ReadAsArray()

    # Get shapes
    nB_y, nB_x = arrMask_resamp.shape
    nT_y, nT_x = arrSeaThick.shape  # nonresampled data

    # Common size in each direction
    n_y = min(nB_y, nT_y)
    n_x = min(nB_x, nT_x)

    # Crop both arrays to the shared region
    arrMask_resamp_crop = arrMask_resamp[:n_y, :n_x]

    save_array_as_tiff(arrMask_resamp_crop, seathickness_data , 'new_data/domainmask.tif')

    return

def compute_new_grounding_line(shoreline, grounding_line, tol=1e-9):
    """
    Given:
        shoreline      : list of (x, y)
        grounding_line : list of (x, y)

    Returns:
        new_grounding_line : list of (x, y)

    new_grounding_line is the portion of the grounding_line between its
    first and last intersection with the shoreline. If fewer than 2
    intersections are found, the original grounding_line is returned.
    """
    shoreline = np.asarray(shoreline, float)
    grounding_line = np.asarray(grounding_line, float)

    def segment_intersection(p1, p2, q1, q2):
        """
        Intersection of segments p1--p2 and q1--q2.
        Returns (point, t_p) where:
            point : np.array([x, y]) intersection
            t_p   : param in [0,1] along p1->p2
        or (None, None) if no intersection.
        """
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        q1 = np.asarray(q1, float)
        q2 = np.asarray(q2, float)

        r = p2 - p1
        s = q2 - q1

        denom = r[0]*s[1] - r[1]*s[0]
        if abs(denom) < tol:
            return None, None

        qp = q1 - p1
        t = (qp[0]*s[1] - qp[1]*s[0]) / denom
        u = (qp[0]*r[1] - qp[1]*r[0]) / denom

        if -tol <= t <= 1+tol and -tol <= u <= 1+tol:
            t = min(max(t, 0.0), 1.0)
            point = p1 + t * r
            return point, t

        return None, None

    intersections = []

    for gi in range(len(grounding_line) - 1):
        gp1 = grounding_line[gi]
        gp2 = grounding_line[gi+1]

        for si in range(len(shoreline) - 1):
            sp1 = shoreline[si]
            sp2 = shoreline[si+1]

            point, t_g = segment_intersection(gp1, gp2, sp1, sp2)
            if point is not None:
                param_g = gi + t_g
                intersections.append((param_g, point))

        sp1 = shoreline[-1]
        sp2 = shoreline[0]
        point, t_g = segment_intersection(gp1, gp2, sp1, sp2)
        if point is not None:
            param_g = gi + t_g
            intersections.append((param_g, point))

    if len(intersections) < 2:
        return grounding_line.tolist()

    intersections.sort(key=lambda x: x[0])

    param_start, p_start = intersections[0]
    param_end,   p_end   = intersections[-1]

    i_start = int(np.floor(param_start))
    i_end   = int(np.floor(param_end))

    new_gl = []
    new_gl.append(p_start.tolist())

    for i in range(i_start + 1, i_end + 1):
        if 0 <= i < len(grounding_line):
            new_gl.append(grounding_line[i].tolist())

    new_gl.append(p_end.tolist())

    return new_gl


def build_seathickness_shoreline(
    seathickness_path: str,
    bathymetry_path: str,
    categories_path: str,
    domainmask_path: str,
    shoreline_res: float = 60,
    min_thickness: float = SHORELINE_MIN_THICKNESS,
    thickness_path: Optional[str] = None,
    shelf_min_thickness: Optional[float] = None,
    margin: float = 0.0,
    gl_tolerance: float = 0.0,
    tolerance: float = 600.0,
) -> list[tuple[float, float]]:
    """
    Build a shoreline based on where seathickness > min_thickness, but
    restricted to valid fjord bathymetry and category mask. Clipping the
    domain boundary at a small positive thickness (rather than exactly 0)
    means every remaining column has a real, non-degenerate water depth --
    so the 3D mesh can be uniformly sigma-layered everywhere without a
    separate unstructured near-shore region.
    Returns a list of (x, y) shoreline coordinates.
    """
    seathickness_ds = readraster(seathickness_path)
    bath_ds         = readraster(bathymetry_path)
    cat_ds          = readraster(categories_path)
    domainmask       = readraster(domainmask_path)

    if shelf_min_thickness is None or thickness_path is None:
        shelf_min_thickness = min_thickness
    if shelf_min_thickness == min_thickness:
        seathickness_mask = gdal_calc.Calc(
            calc=[f"(A>{min_thickness}) * (B!=-9999) * (B<=0) * (C==1)"],
            A=seathickness_ds,
            B=bath_ds,
            C=domainmask,
            outfile="new_data/seathickness_mask.tif",
            NoDataValue=0,
            overwrite=True,
        )
    else:
        # Separate threshold where there is ice on top (D = ice thickness).
        ice_ds = readraster(thickness_path)
        seathickness_mask = gdal_calc.Calc(
            calc=[f"(A > (D>0)*{shelf_min_thickness} + (D<=0)*{min_thickness})"
                  f" * (B!=-9999) * (B<=0) * (C==1)"],
            A=seathickness_ds,
            B=bath_ds,
            C=domainmask,
            D=ice_ds,
            outfile="new_data/seathickness_mask.tif",
            NoDataValue=0,
            overwrite=True,
        )

    if margin > 0:
        seathickness_mask = _erode_mask_from_shore(
            seathickness_mask, domainmask, margin,
            "new_data/seathickness_mask_margin.tif")

    gdal.Footprint(
        'new_data/seathickness_outline.geojson',
        seathickness_mask,
        format="GeoJSON",
        maxPoints="unlimited" if gl_tolerance > 0 else shoreline_res,
        srcNodata=0,
    )

    seathickness_lines = get_boundary_string('new_data/seathickness_outline.geojson')
    seathickness_shoreline = get_main_boundary(seathickness_lines)

    if gl_tolerance > 0:
        if thickness_path is None:
            raise ValueError("gl_tolerance needs thickness_path to find the ice")
        seathickness_shoreline = _simplify_near_ice(
            seathickness_shoreline, readraster(thickness_path),
            gl_tolerance, tolerance)

    return seathickness_shoreline


def _simplify_near_ice(ring, ice_ds, tol_ice, tol_open, ice_reach_px=3):
    """Douglas-Peucker on a closed ring with a per-vertex tolerance.

    Vertices within `ice_reach_px` pixels of ice (thickness > 0) use `tol_ice`,
    all others `tol_open`, so the grounding line keeps many vertices and the
    open shore few.
    """
    pts = np.asarray(ring, dtype=float)
    n = len(pts)
    gt = ice_ds.GetGeoTransform()
    ice = ice_ds.ReadAsArray() > 0
    for _ in range(ice_reach_px):
        grown = ice.copy()
        grown[1:] |= ice[:-1]; grown[:-1] |= ice[1:]
        grown[:, 1:] |= ice[:, :-1]; grown[:, :-1] |= ice[:, 1:]
        ice = grown
    c = np.clip(np.floor((pts[:, 0] - gt[0]) / gt[1]).astype(int), 0, ice.shape[1] - 1)
    r = np.clip(np.floor((pts[:, 1] - gt[3]) / gt[5]).astype(int), 0, ice.shape[0] - 1)
    tol = np.where(ice[r, c], tol_ice, tol_open).astype(float)

    keep = np.zeros(n, bool)
    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    keep[[0, far]] = True
    stack = [(0, far), (far, n)]
    while stack:
        a, b = stack.pop()
        idx = np.arange(a + 1, b)
        if len(idx) == 0:
            continue
        A, B = pts[a], pts[b % n]
        AB = B - A
        L = float(np.hypot(*AB))
        P = pts[idx % n]
        if L > 0:
            dev = np.abs(AB[0] * (P[:, 1] - A[1]) - AB[1] * (P[:, 0] - A[0])) / L
        else:
            dev = np.hypot(*(P - A).T)
        ratio = dev / tol[idx % n]
        k = int(np.argmax(ratio))
        if ratio[k] > 1:
            m = int(idx[k])
            keep[m % n] = True
            stack += [(a, m), (m, b)]
    out = pts[keep]
    print(f"[SHORELINE] simplified {n} -> {len(out)} vertices "
          f"({int((keep & (tol == tol_ice)).sum())} near ice; tolerance "
          f"{tol_ice:g} m near ice, {tol_open:g} m elsewhere)", flush=True)
    return out.tolist()


def _erode_mask_from_shore(mask_ds, domainmask_ds, margin, out_path):
    """Shrink a 0/1 mask by `margin` metres, but only against in-domain pixels.

    Pixels outside the domain mask count as inside while eroding, so the
    boundary does not move where the domain is cut at the data edge (the open
    ocean mouth) -- only where it meets land or too-shallow water.
    """
    m = mask_ds.ReadAsArray() > 0
    outside = domainmask_ds.ReadAsArray() != 1
    k = int(round(margin / abs(mask_ds.GetGeoTransform()[1])))
    grow = m | outside
    for _ in range(k):
        e = grow.copy()
        e[1:] &= grow[:-1]; e[:-1] &= grow[1:]
        e[:, 1:] &= grow[:, :-1]; e[:, :-1] &= grow[:, 1:]
        grow = e
    eroded = (grow & m).astype(np.uint8)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out_path, eroded.shape[1], eroded.shape[0], 1, gdal.GDT_Byte)
    ds.SetGeoTransform(mask_ds.GetGeoTransform())
    ds.SetProjection(mask_ds.GetProjection())
    band = ds.GetRasterBand(1)
    band.WriteArray(eroded)
    band.SetNoDataValue(0)
    ds.FlushCache()
    print(f"[SHORELINE] margin {margin:g} m ({k} px): mask {int(m.sum())} -> "
          f"{int(eroded.sum())} pixels", flush=True)
    return ds


def save_array_as_tiff(array, template_ds, out_path, dtype=gdal.GDT_Float32, nodata=None, compress=True): 
    """
    Save a numpy array as a GeoTIFF, copying geotransform and projection
    from a template GDAL dataset.
    """

    gt = template_ds.GetGeoTransform()
    proj = template_ds.GetProjection()

    ny, nx = array.shape

    driver = gdal.GetDriverByName("GTIFF")
    options = ["COMPRESS=LZW"] if compress else []

    out_ds = driver.Create(out_path, nx, ny, 1, dtype, options=options)

    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band = out_ds.GetRasterBand(1)

    if nodata is not None:
        band.SetNoDataValue(nodata)

    band.WriteArray(array)
    band.FlushCache()

    out_ds = None
    print(f"Saved GeoTIFF → {out_path}")

class DraggablePolyline:
    """Interactive editor for a closed polygon.

    Mouse / keys (with the toolbar's zoom and pan switched off):
      left-drag a vertex   move it
      i                    insert a vertex at the cursor, into the nearest edge
      d                    delete the vertex nearest the cursor
      u                    undo the last change
      R                    reset to the automatic contour
      close the window     accept and save
    """

    PICK_PIXELS = 10

    def __init__(self, ax, points, automatic=None):
        self.ax = ax
        self.points = np.array(points, dtype=float)
        self.automatic = np.array(automatic if automatic is not None else points,
                                  dtype=float)
        self.history = []
        self.selected_index = None
        self.line, = ax.plot([], [], "r.-", ms=7, lw=1.2, zorder=5)
        self._redraw()
        canvas = ax.figure.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("key_press_event", self.on_key)

    # -- helpers --------------------------------------------------------------
    def _redraw(self):
        closed = np.vstack([self.points, self.points[:1]])
        self.line.set_data(closed[:, 0], closed[:, 1])
        self.ax.set_xlabel(f"{len(self.points)} vertices   |   drag  i:insert  "
                           f"d:delete  u:undo  R:reset   |   close window to save")
        self.ax.figure.canvas.draw_idle()

    def _busy(self):
        tb = getattr(self.ax.figure.canvas, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    def _nearest_vertex(self, event):
        disp = self.ax.transData.transform(self.points)
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        i = int(d.argmin())
        return i, d[i]

    def _push(self):
        self.history.append(self.points.copy())

    # -- events ---------------------------------------------------------------
    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1 or self._busy():
            return
        i, d = self._nearest_vertex(event)
        if d < self.PICK_PIXELS:
            self._push()
            self.selected_index = i

    def on_motion(self, event):
        if self.selected_index is None or event.inaxes != self.ax:
            return
        self.points[self.selected_index] = [event.xdata, event.ydata]
        self._redraw()

    def on_release(self, event):
        self.selected_index = None

    def on_key(self, event):
        if event.inaxes != self.ax and event.key not in ("u", "R"):
            return
        if event.key == "i":
            p = np.array([event.xdata, event.ydata])
            a, b = self.points, np.roll(self.points, -1, axis=0)
            ab = b - a
            t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
            dist = np.hypot(*(a + t[:, None] * ab - p).T)
            k = int(dist.argmin())
            self._push()
            self.points = np.insert(self.points, k + 1, p, axis=0)
        elif event.key == "d":
            if len(self.points) <= 3:
                return
            i, _ = self._nearest_vertex(event)
            self._push()
            self.points = np.delete(self.points, i, axis=0)
        elif event.key == "u":
            if not self.history:
                return
            self.points = self.history.pop()
        elif event.key == "R":
            self._push()
            self.points = self.automatic.copy()
        else:
            return
        self._redraw()

    def get_points(self):
        return self.points.tolist()

def ComputeSpaceExtent(surface_pos_data,seathickness_array):
    gt = surface_pos_data.GetGeoTransform()
    ny, nx = seathickness_array.shape

    xmin = gt[0]
    xmax = gt[0] + gt[1] * nx
    ymax = gt[3]
    ymin = gt[3] + gt[5] * ny

    extent=(xmin, xmax, ymin, ymax)

    return extent

def PlotShoreOnRaster(line,extent,seathickness_array):

    fig, ax = plt.subplots(figsize=(10, 10))

    im= ax.imshow(
        seathickness_array,
        origin="upper",
        cmap="viridis", vmin=0, vmax=20,
        extent=extent,
        alpha=0.6,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sea thickness (m)")

    plot_main_geoline(line, ax=ax, color="k", linewidth=1.5)

    ax.set_aspect("equal", "box")
    ax.set_title("line on seathickness background")
    plt.close('all')


def Manual_Redraw(line, extent, seathickness_array, ice_thickness=None,
                  automatic=None, thresholds=None):
    """Open the shoreline editor (blocking) and return the edited polygon.

    The background is the water column; contours mark where it crosses the
    open-water and under-ice thresholds, the white line is the ice outline and
    the dashed grey line the automatic contour.
    """
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("SHORELINE_EDIT=1 needs a display (no DISPLAY set)")
    for backend in ("TkAgg", "QtAgg"):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    # 'r' would otherwise reset the view instead of being free for us
    plt.rcParams["keymap.home"] = [k for k in plt.rcParams["keymap.home"]
                                   if k != "r"]

    fig, ax = plt.subplots(figsize=(11, 11))
    water = np.where(np.isfinite(seathickness_array), seathickness_array, np.nan)
    im = ax.imshow(water, extent=extent, cmap="viridis", vmin=0, vmax=200,
                   origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="water column (m)")
    ny, nx = water.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[3], extent[2], ny)
    if thresholds is not None:
        t_open, t_shelf = thresholds
        under_ice = (ice_thickness > 0) if ice_thickness is not None else np.zeros_like(water, bool)
        ax.contour(xs, ys, np.where(under_ice, np.nan, water), [t_open],
                   colors="orange", linewidths=0.8)
        ax.contour(xs, ys, np.where(under_ice, water, np.nan), [t_shelf],
                   colors="magenta", linewidths=0.8)
    if ice_thickness is not None:
        ax.contour(xs, ys, (ice_thickness > 0).astype(float), [0.5],
                   colors="white", linewidths=0.8)
    if automatic is not None:
        auto = np.vstack([automatic, automatic[:1]])
        ax.plot(auto[:, 0], auto[:, 1], "--", color="0.6", lw=0.8)
    pts = np.asarray(line)
    pad = 3000
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    title = "Shoreline editor"
    if thresholds is not None:
        title += (f"   orange: {thresholds[0]:g} m (open water)   "
                  f"magenta: {thresholds[1]:g} m (under ice)   white: ice edge")
    ax.set_title(title, fontsize=9)

    editor = DraggablePolyline(ax, line, automatic=automatic)
    plt.show()          # blocks until the window is closed
    new_line = editor.get_points()
    plt.close("all")
    plt.switch_backend("Agg")
    return new_line


# --- Direct prism stacking (bypasses TetGen entirely) ------------------------
#
# The TetGen path forced interior sigma layers by embedding one seed point per
# level per column.  That scales badly: 2492 seeds at 15 layers works, 6586 at
# 38 segfaults TetGen's boundary recovery, and turning the seeds off produces a
# mesh with no interior layering at all.  Here the volume is built directly
# from the plan-view triangulation, so cost is linear in the layer count.

TET_TYPE = 4       # gmsh element type: 4-node tetrahedron
TRI_TYPE = 2       # gmsh element type: 3-node triangle
LINE_TYPE = 1      # gmsh element type: 2-node line


def _extract_planar_mesh(model, water_tags, ice_tags, inflow_curve_tags):
    """Pull the plan-view triangulation out of a meshed planar gmsh model.

    Returns a dict with compacted, zero-based column indexing:
      xy            (C, 2)  column coordinates
      tris          (T, 3)  triangles as column indices
      tri_is_ice    (T,)    True where the triangle sits under land ice
      node_is_ice   (C,)    True where the column belongs to any ice triangle
      inflow_edges  set of frozenset({col_a, col_b}) for 2D inflow boundary edges
    """
    ntags, ncoords, _ = model.mesh.getNodes()
    ncoords = np.asarray(ncoords).reshape(-1, 3)
    raw = {int(t): i for i, t in enumerate(ntags)}

    tri_blocks, ice_blocks = [], []
    for surf_tags, ice_flag in ((list(water_tags), False), (list(ice_tags), True)):
        for surf in surf_tags:
            etypes, _etags, enodes = model.mesh.getElements(2, int(surf))
            for etype, conn in zip(etypes, enodes):
                if etype != TRI_TYPE:
                    continue
                block = np.asarray(conn, dtype=np.int64).reshape(-1, 3)
                tri_blocks.append(block)
                ice_blocks.append(np.full(len(block), ice_flag, dtype=bool))
    if not tri_blocks:
        raise RuntimeError("planar stage produced no triangles")

    tris_raw = np.concatenate(tri_blocks)
    tri_is_ice = np.concatenate(ice_blocks)

    # Compact to only the nodes the triangles actually use, so column indices
    # are dense and 0-based -- build_stacked_volume assumes that.
    used_raw = np.unique(tris_raw)
    compact = {int(t): i for i, t in enumerate(used_raw)}
    tris = np.vectorize(compact.get)(tris_raw).astype(np.int64)
    xy = ncoords[[raw[int(t)] for t in used_raw], :2].copy()

    node_is_ice = np.zeros(len(xy), dtype=bool)
    node_is_ice[np.unique(tris[tri_is_ice])] = True

    inflow_edges = set()
    for curve in inflow_curve_tags:
        try:
            etypes, _etags, enodes = model.mesh.getElements(1, int(curve))
        except Exception:
            continue
        for etype, conn in zip(etypes, enodes):
            if etype != LINE_TYPE:
                continue
            for a, b in np.asarray(conn, dtype=np.int64).reshape(-1, 2):
                ca, cb = compact.get(int(a)), compact.get(int(b))
                if ca is not None and cb is not None:
                    inflow_edges.add(frozenset((ca, cb)))

    print(f"[PLANAR] {len(xy)} columns, {len(tris)} triangles, "
          f"{int(node_is_ice.sum())} ice columns, "
          f"{len(inflow_edges)} inflow boundary edges", flush=True)
    return {"xy": xy, "tris": tris, "tri_is_ice": tri_is_ice,
            "node_is_ice": node_is_ice, "inflow_edges": inflow_edges}


def _write_stacked_msh(sm, face_groups, output_path):
    """Write a StackedMesh to .msh via discrete entities (no meshing involved)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", GMESH_NUM_THREADS)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    model = gmsh.model
    model.add("stacked")

    node_tags = np.arange(1, len(sm.coords) + 1, dtype=np.int64)

    vol = model.addDiscreteEntity(3)
    model.mesh.addNodes(3, vol, node_tags, sm.coords.reshape(-1))
    model.mesh.addElementsByType(vol, TET_TYPE, [], (sm.tets + 1).reshape(-1))
    model.addPhysicalGroup(3, [vol], WATER_SURFACE_TAG, name="Water")

    for phys_tag, name, faces in face_groups:
        if len(faces) == 0:
            continue
        ent = model.addDiscreteEntity(2)
        faces = np.asarray(faces)
        # Discrete surface elements reference node tags that already exist on
        # the volume entity, but gmsh still wants the nodes registered on this
        # entity before the elements referring to them.
        used = np.unique(faces)
        model.mesh.addNodes(2, ent, (used + 1),
                            sm.coords[used].reshape(-1))
        model.mesh.addElementsByType(ent, TRI_TYPE, [], (faces + 1).reshape(-1))
        model.addPhysicalGroup(2, [ent], phys_tag, name=name)
        print(f"[STACKED] physical {name} (tag {phys_tag}): {len(faces)} triangles",
              flush=True)

    gmsh.write(str(output_path))
    gmsh.finalize()

    try:
        xdmf_path = write_legacy_xdmf(output_path)
        print(f"Wrote XDMF for FEniCS/ParaView: {xdmf_path} (+ _volume.xdmf)")
    except Exception as err:
        print(f"[stacked] Failed to write XDMF next to {output_path}: {err}",
              file=sys.stderr)


def generate_stacked_mesh(outline, intersect, grounding_line, m, eps,
                          category_data, bathymetry_data, highres_data,
                          thickness_data, surface_pos_data, scale=1,
                          num_of_layers=2, adapt=False,
                          adaptive_scales=(1/4, 2), interpolate=True,
                          mesh_filename=None, delta=None,
                          band_lcmax_boundaries=None, **_ignored):
    """Build the 3D mesh by stacking the plan-view triangulation into prisms.

    Same role as generate_mesh_mult, but the interior is constructed directly
    instead of being tetrahedralised, so there is no seed-point embedding and
    no dependence on TetGen surviving the layer count.
    """
    eps_eff = min(eps, -1)

    planar = generate_2D_mesh(
        outline, intersect, grounding_line, category_data, m, eps_eff, adapt,
        adaptive_scales=adaptive_scales, num_of_layers=num_of_layers,
        filename=mesh_filename, delta=delta,
        band_lcmax_boundaries=band_lcmax_boundaries, planar_only=True)

    xy = planar["xy"]
    tris = planar["tris"]
    tri_is_ice = planar["tri_is_ice"]
    node_is_ice = planar["node_is_ice"]
    inflow_edges = planar["inflow_edges"]

    bathymetry_array = bathymetry_data.ReadAsArray()
    bathymetry_trans = bathymetry_data.GetGeoTransform()
    highres_array = highres_data.ReadAsArray()
    highres_trans = highres_data.GetGeoTransform()
    thickness_array = thickness_data.ReadAsArray()
    thickness_trans = thickness_data.GetGeoTransform()
    surface_pos_array = surface_pos_data.ReadAsArray()

    # --- sample the two bounding surfaces at every column --------------------
    C = len(xy)
    z_top = np.zeros(C)
    z_bot = np.zeros(C)
    for c in range(C):
        x, y = float(xy[c, 0]), float(xy[c, 1])
        if node_is_ice[c]:
            top = get_ice_depth(x, y, thickness_trans, thickness_array,
                                surface_pos_array, bathymetry_array,
                                -MIN_ICE_THICKNESS, interpolate=interpolate)
            z_top[c] = min(float(top), 0.0)
        else:
            z_top[c] = 0.0
        z_bot[c] = float(get_bathymetric_depth(
            x, y, bathymetry_trans, bathymetry_array, highres_trans,
            highres_array, -MIN_WATER_DEPTH, highres=True,
            interpolate=interpolate))

    # --- enforce a usable column thickness -----------------------------------
    # build_stacked_volume refuses to mesh a non-positive column, by design: a
    # silently collapsed column is what produced hundreds of inverted tets in
    # the TetGen meshes.  Clamp explicitly here and report how many columns it
    # touched, so the cost is visible instead of hidden.
    floor_limit = MIN_WATER_COLUMN if MIN_WATER_COLUMN > 0 else MIN_STACK_THICKNESS
    thickness = z_top - z_bot
    thin = thickness < floor_limit
    if thin.any():
        print(f"[STACKED] clamping {int(thin.sum())} of {C} columns to "
              f"{floor_limit:g} m (thinnest was {float(thickness.min()):.3f} m)",
              flush=True)
        z_bot[thin] = z_top[thin] - floor_limit

    layers = None
    if MIN_LAYER_THICKNESS > 0:
        layers = layers_from_min_thickness(z_top - z_bot, num_of_layers,
                                           MIN_LAYER_THICKNESS)
        print(f"[STACKED] MIN_LAYER_THICKNESS={MIN_LAYER_THICKNESS:g} m: "
              f"{int((layers < num_of_layers).sum())} of {C} columns get fewer "
              f"than {num_of_layers} layers (min {int(layers.min())})", flush=True)
    front_kw = {}
    if CALVING_FRONT:
        # Front columns sit on the ice/water interface: used by both ice and
        # open-water triangles.  They get extra layers from the sea surface
        # down to the draft, so open water reaches z=0 right up to the front.
        in_water = np.zeros(C, bool)
        in_water[np.unique(tris[~tri_is_ice])] = True
        draft = -z_top
        front = node_is_ice & in_water & (draft >= MIN_FRONT_DRAFT)
        if MIN_LAYER_THICKNESS > 0:
            nu = np.clip(np.floor(draft / MIN_LAYER_THICKNESS), 1, num_of_layers)
        else:
            nu = np.clip(np.round(num_of_layers * draft / (draft + z_top - z_bot)),
                         1, num_of_layers)
        nu = np.where(front, nu, 0).astype(np.int64)
        if layers is None:
            layers = np.full(C, num_of_layers, np.int64)
        front_kw = dict(tri_is_ice=tri_is_ice, z_surface=np.zeros(C),
                        upper_layers=nu)
        print(f"[STACKED] calving front: {int(front.sum())} front columns, draft "
              f"{float(draft[front].min()) if front.any() else 0:.0f}-"
              f"{float(draft[front].max()) if front.any() else 0:.0f} m, "
              f"{int(nu[front].min()) if front.any() else 0}-"
              f"{int(nu[front].max()) if front.any() else 0} layers above the draft",
              flush=True)
    if ICE_TOP_LAYER > 0:
        if layers is None:
            layers = np.full(C, num_of_layers, np.int64)
        front_kw["top_layer"] = np.where(node_is_ice, ICE_TOP_LAYER, 0.0)
        front_kw["max_growth"] = LAYER_GROWTH
        if "tri_is_ice" not in front_kw:
            front_kw["tri_is_ice"] = tri_is_ice
        H = z_top - z_bot
        stretched = node_is_ice & (H > layers * ICE_TOP_LAYER) & (layers >= 2)
        print(f"[STACKED] stretching {int(stretched.sum())} of "
              f"{int(node_is_ice.sum())} ice columns: top layer "
              f"{ICE_TOP_LAYER:g} m, growth <= {LAYER_GROWTH:g}", flush=True)
    sm = build_stacked_volume(xy, tris, scale * z_top, scale * z_bot,
                              num_of_layers, layers_per_column=layers,
                              **{k: (scale * v if k in ("z_surface", "top_layer") else v)
                                 for k, v in front_kw.items()})
    L = sm.num_layers

    # --- classify the derived boundary into physical groups ------------------
    cols = sm.node_col[sm.boundary_faces]
    is_top = sm.boundary_kind == TOP
    is_bottom = sm.boundary_kind == BOTTOM
    is_side = sm.boundary_kind == SIDE

    if sm.boundary_from_ice is not None:
        ice_face = is_top & sm.boundary_from_ice
    else:
        ice_face = is_top & np.all(node_is_ice[cols], axis=1)
    water_face = is_top & ~ice_face
    calving_face = sm.boundary_kind == CALVING

    side_idx = np.flatnonzero(is_side)
    inflow_side, wall_side = [], []
    for i in side_idx:
        edge = frozenset(int(c) for c in cols[i])
        (inflow_side if edge in inflow_edges else wall_side).append(i)

    ice_faces = ice_face | (calving_face & (CALVING_FRONT_TAG == ICE_TAG))
    face_groups = [
        (ICE_TAG, "Ice", sm.boundary_faces[ice_faces]),
        (WATER_SURFACE_TAG, "Water Surface", sm.boundary_faces[water_face]),
        (BATHYMETRY_TAG, "Bathymetry",
         np.concatenate([sm.boundary_faces[is_bottom],
                         sm.boundary_faces[wall_side]])
         if wall_side else sm.boundary_faces[is_bottom]),
        (INFLOW_TAG, "Inflow", sm.boundary_faces[inflow_side]),
    ]
    if CALVING_FRONT_TAG != ICE_TAG:
        face_groups.append((CALVING_FRONT_TAG, "Calving front",
                            sm.boundary_faces[calving_face]))
    print(f"[STACKED] calving front: {int(calving_face.sum())} faces -> tag "
          f"{CALVING_FRONT_TAG}", flush=True)

    output_path = ensure_msh_path(mesh_filename or
                                  f"stacked_layers{num_of_layers}.msh")
    _write_stacked_msh(sm, face_groups, output_path)

    print(f"[STACKED] {len(sm.coords)} nodes, {len(sm.tets)} tets, "
          f"{L} layers, {sm.num_columns} columns", flush=True)
    return len(sm.coords), Path(output_path).stem


def main():
    """Batch-generate meshes for a selection of resolutions/settings."""
    elevation = readraster('../data/gis_data/bathymetry.tif')
    elevation_band = elevation.GetRasterBand(1)

    thickness_data = readraster('../data/thickness.tif')
    surface_pos_data = readraster('../data/surface_pos.tif')
    full_bathymetry = readraster('../data/elevation.tif')

    elevation_array = elevation_band.ReadAsArray()
    thickness_array = thickness_data.ReadAsArray()
    surface_array = surface_pos_data.ReadAsArray()
    fullbathymetry_array = full_bathymetry.ReadAsArray()

    icebottom_array = surface_array-thickness_array
    seasurface = np.minimum(icebottom_array, 0)
    seathickness_array = seasurface - fullbathymetry_array

    save_array_as_tiff(icebottom_array, surface_pos_data , 'new_data/icebottom.tif')
    save_array_as_tiff(seathickness_array, surface_pos_data , 'new_data/seathickness.tif')

    createdomainmask(elevation)

    # Vertex budget for gdal.Footprint on the thickness mask.  30 gives ~4.5 km
    # chords over the 135 km shoreline, which cut across the true contour in
    # both directions (deep water cut off, sub-threshold water enclosed).
    shoreline_res = SHORELINE_RES

    shoreline = build_seathickness_shoreline(
        seathickness_path="new_data/seathickness.tif",
        bathymetry_path="../data/elevation.tif",
        categories_path="../data/gis_data/categories.tif",
        domainmask_path="new_data/domainmask.tif",
        shoreline_res=shoreline_res,
        thickness_path="../data/thickness.tif",
        shelf_min_thickness=SHELF_MIN_THICKNESS,
        margin=SHORELINE_MARGIN,
        gl_tolerance=SHORELINE_GL_TOL,
        tolerance=SHORELINE_TOL,
    )

    extent=ComputeSpaceExtent(surface_pos_data,seathickness_array)

    # Cache is keyed on the thresholds and resolution: the saved boundary is
    # only valid for the settings it was derived with, so different settings
    # get their own file rather than silently reusing the old shoreline.  A
    # hand-edited polygon (SHORELINE_EDIT=1) is saved into the same file.
    import json
    _saved_shore = Path(f"new_data/saved_shoreline_t{SHORELINE_MIN_THICKNESS:g}"
                        + ("" if SHELF_MIN_THICKNESS == SHORELINE_MIN_THICKNESS
                           else f"_s{SHELF_MIN_THICKNESS:g}")
                        + ("" if shoreline_res == 30 else f"_r{shoreline_res:g}")
                        + (f"_m{SHORELINE_MARGIN:g}" if SHORELINE_MARGIN > 0 else "")
                        + (f"_g{SHORELINE_GL_TOL:g}o{SHORELINE_TOL:g}"
                           if SHORELINE_GL_TOL > 0 else "")
                        + ".json")
    automatic = shoreline
    if _saved_shore.exists():
        shoreline = json.loads(_saved_shore.read_text())
        print(f"[SHORELINE] Loaded saved boundary from {_saved_shore}  "
              f"(delete this file to re-draw, SHORELINE_EDIT=1 to edit)")
    if SHORELINE_EDIT:
        thickness_array = thickness_data.ReadAsArray()
        shoreline = Manual_Redraw(shoreline, extent, seathickness_array,
                                  ice_thickness=thickness_array,
                                  automatic=automatic,
                                  thresholds=(SHORELINE_MIN_THICKNESS,
                                              SHELF_MIN_THICKNESS))
    if SHORELINE_EDIT or not _saved_shore.exists():
        _saved_shore.parent.mkdir(parents=True, exist_ok=True)
        _saved_shore.write_text(json.dumps(shoreline))
        print(f"[SHORELINE] Saved boundary ({len(shoreline)} points) to {_saved_shore}")

    categories = readraster('../data/gis_data/categories.tif')

    seaice = gdal_calc.Calc(calc=["(A==3)"], A=categories, outfile="new_data/water.tif", 
                            NoDataValue=0, overwrite=True)

    landice = gdal_calc.Calc(calc=["(A==2)"], A=categories, outfile="new_data/ice.tif", 
                             NoDataValue=0, overwrite=True)

    ice_res = 0.8*shoreline_res

    gdal.Footprint('new_data/seaice.geojson', seaice, format="GeoJSON", 
                   maxPoints = ice_res, srcNodata = 0)
    seaice_lines = get_boundary_string('new_data/seaice.geojson')
    seaice_line = get_main_boundary(seaice_lines)

    gdal.Footprint('new_data/landice.geojson', landice, format="GeoJSON", 
                   maxPoints = 'unlimited', srcNodata = 0)
    landice_lines = get_boundary_string('new_data/landice.geojson')
    landice_line = get_main_boundary(landice_lines, no=1)
    
    grounding_line = get_grounding_line(seaice_line, landice_line)

    new_grounding_line = compute_new_grounding_line(shoreline, grounding_line, tol=1e-3)
    grounding_line = new_grounding_line

    outline = shoreline 

    with open("../outline", 'w') as f:
        f.write(str(outline))

    water = gdal_calc.Calc(calc=["(A==0)"], A=categories, outfile="new_data/water.tif", 
                           NoDataValue=0, overwrite=True)
    gdal.Footprint('new_data/water.geojson', water, format="GeoJSON", 
                   maxPoints = 'unlimited', srcNodata = 0)
    water_lines = get_boundary_string('new_data/water.geojson')
    water_line = get_main_boundary(water_lines)

    intersect = get_water_ice_intersect(water_line, seaice_line)
    if SHORELINE_GL_TOL > 0:
        outline, intersect = clip_front_to_outline(outline, intersect)
    
    all_lines = {'Main Outline': outline, 'Grounding Line': grounding_line, 
                'Water-Ice Intersect': intersect}

    full_bathymetry = readraster('../data/elevation.tif')

    sherard = readraster("../data/sherard-osborn-fjord-15m-3996.tiff")
    highres = gdal.Warp('new_data/highres.tif', sherard, dstSRS='EPSG:3413')
    thickness_data = readraster('../data/thickness.tif')
    surface_pos_data = readraster('../data/surface_pos.tif')
    
    dofs = []
    
    class Scenario(NamedTuple):
        element_size: float
        scale: float
        num_layers: int
        adapt: bool
        optimize: bool
        stack: float
        filename: Optional[str] = None

    adaptive = [
        # 38 layers over ~952m depth gives ~25m per layer (sigma-coordinate style).
        Scenario(2000, 6, 38, False, False, 0, "coarse_test_layers38_v6.msh"),
    ]

    # Fast coarse scenario for shaking out the new structured-prism / thick-thin
    # decomposition logic before the full production run (see plan verification
    # step 2). Remove once validated.
    prism_test = [
        Scenario(2000, 6, 4, False, False, 0, "prism_test_layers4.msh"),
    ]

    # Same horizontal footprint as prism_test, but full production layer count
    # (38) -- checking whether the corner self-intersection reproduces at
    # realistic vertical resolution or was an artifact of the coarse layer count.
    prism_test_full_layers = [
        Scenario(2000, 6, 38, False, False, 0, "prism_test_layers38.msh"),
    ]

    # Moderate layer count for a quick, reliable preview mesh.
    prism_test_moderate = [
        Scenario(2000, 6, 15, False, False, 0, "prism_test_layers15.msh"),
    ]

    # Sweep scenario: layer count from NUM_LAYERS, shoreline clip threshold from
    # SHORELINE_MIN_THICKNESS, minimum water column from MIN_WATER_COLUMN (all
    # env vars). The output filename encodes all three so runs at different
    # settings never overwrite each other.
    _n_layers = int(os.environ.get("NUM_LAYERS", 38))
    _element_size = float(os.environ.get("ELEMENT_SIZE", 500))
    # Vertical exaggeration written into the mesh coordinates.  1 = real
    # coordinates, which the solver needs; 6 was the old default.
    _z_scale = float(os.environ.get("Z_SCALE", 1))
    sweep = [
        Scenario(_element_size, _z_scale, _n_layers, False, False, 0,
                 ("stacked" if USE_STACKED else "sweep")
                 + f"_layers{_n_layers}_t{SHORELINE_MIN_THICKNESS:g}"
                 + ("" if SHELF_MIN_THICKNESS == SHORELINE_MIN_THICKNESS
                    else f"_s{SHELF_MIN_THICKNESS:g}")
                 + f"_c{MIN_WATER_COLUMN:g}_h{_element_size:g}"
                 + ("" if SHORELINE_RES == 30 else f"_r{SHORELINE_RES}")
                 + (f"_m{SHORELINE_MARGIN:g}" if SHORELINE_MARGIN > 0 else "")
                 + ("" if _z_scale == 1 else f"_zx{_z_scale:g}")
                 + (f"_dz{MIN_LAYER_THICKNESS:g}" if USE_STACKED and MIN_LAYER_THICKNESS > 0 else "")
                 + (f"_top{ICE_TOP_LAYER:g}" if USE_STACKED and ICE_TOP_LAYER > 0 else "")
                 + ((f"_hr{ICE_ELEMENT_SIZE or _element_size / 2:g}"
                     f"-{GL_ELEMENT_SIZE or _element_size / 4:g}")
                    if USE_STACKED and HORIZONTAL_REFINE else "")
                 + ".msh"),
    ]

    params = sweep

    eps_extrude = -1.
    builder = generate_stacked_mesh if USE_STACKED else generate_mesh_mult
    print(f"[PATH] using {builder.__name__} (USE_STACKED={USE_STACKED})", flush=True)
    for scenario in params:
        dof, meshname = builder(outline, intersect, grounding_line, 
                                 scenario.element_size, eps_extrude , categories, full_bathymetry, highres, 
                                 thickness_data, surface_pos_data, scale = scenario.scale, 
                                 num_of_layers = scenario.num_layers, adapt = scenario.adapt, 
                                 adaptive_scales = (1/2, 2), optimize = scenario.optimize, 
                                 stack = scenario.stack, interpolate = True,
                                 mesh_filename = scenario.filename,
                                 delta = None)
        dofs.append((meshname, dof))
    
    for meshname, dof in dofs:
        print(f"Mesh Name: {meshname}\t\tDOFS: {dof}")
    

if __name__ == "__main__":
    main()
