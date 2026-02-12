import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
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
from typing import Dict, NamedTuple, Sequence, List, Optional, Union
from pathlib import Path
import subprocess, sys


# plt.style.use('seaborn-v0_8')
gdal.UseExceptions()

MIN_ICE_THICKNESS = 0.05
MIN_WATER_DEPTH = 0.1
EPS_Z_CLAMP = 10.0

WATER_SURFACE_TAG = 5
ICE_TAG = 1
BATHYMETRY_TAG = 2
INFLOW_TAG = 4
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
    """Write an XDMF companion file with physical tags preserved for FEniCS."""

    msh_path = Path(msh_path)
    xdmf_path = msh_path.with_suffix(".xdmf")
    msh = meshio.read(msh_path)

    cells: List[tuple[str, np.ndarray]] = []
    cell_data: Dict[str, List[np.ndarray]] = {name: [] for name in msh.cell_data.keys()}

    for idx, cell_block in enumerate(msh.cells):
        if cell_block.type not in ("triangle", "tetra"):
            continue
        cells.append((cell_block.type, cell_block.data))
        for name, data_list in msh.cell_data.items():
            arr = data_list[idx]
            cell_data[name].append(np.asarray(arr, dtype=np.int32))

    if not cells:
        raise ValueError(f"{msh_path} contains no tetra or triangle cells to export.")

    if "name_to_read" not in cell_data and "gmsh:physical" in cell_data:
        cell_data["name_to_read"] = [np.array(arr, dtype=np.int32) for arr in cell_data["gmsh:physical"]]

    meshio.write(
        xdmf_path,
        meshio.Mesh(
            points=msh.points,
            cells=cells,
            cell_data=cell_data,
            field_data=msh.field_data,
        ),
        file_format="xdmf",
    )

    return xdmf_path

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
    plt.show()

def generate_2D_mesh(outline, intersect, grounding_line, category_data, 
                     m, eps, adapt, adaptive_scales = (1/4, 2),
                     filename: Optional[str] = None, delta: Optional[float] = None,
                     band_lcmax_boundaries: Optional[Sequence[BandBoundarySpec]] = None):
    """Create the surface mesh that will later be extruded to 3D.

    The 2D stage wires up every shoreline, grounding line, and inflow segment
    as explicit curves.  That makes it much easier to preserve important
    physical boundaries once we start extruding with Gmsh.
    When ``band_lcmax_boundaries`` is supplied (or configured globally) those
    boundaries keep the coarse ``LC_MAX`` target inside the boundary band.
    """
    
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", GMESH_NUM_THREADS)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    
    
    model = gmsh.model
    model.add("2D")
    if band_lcmax_boundaries is None:
        band_lcmax_boundaries = BAND_BOUNDARIES_USE_LC_MAX
    band_lcmax_boundaries = tuple(band_lcmax_boundaries or ())
    # All geometry entities created below live in this temporary 2D model; we
    # return only the mesh and a few helper lists for the 3D stage.
    
    if adapt:
        # Use three characteristic lengths: fine on the grounding line, medium
        # along the inflow, and coarser away from the ice where gradients are
        # smaller.
        sizes = (adaptive_scales[0]*m, m, adaptive_scales[1]*m)
    else:
        sizes = (m,m,m)
    
    category_array = category_data.ReadAsArray()
    category_trans = category_data.GetGeoTransform()
    # GIS categories mark inflow vs. ice vs. open water; they drive both
    # sizing choices and physical-group labelling below.
    
    coords = outline
    intersect_points = intersect
    grounding_points = grounding_line

    if delta is not None:
        # Add fake points on boundary in inflow
        map_fake_to_real = {}
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
    # These indices help us splice the water/ice intersection onto the outer
    # boundary loop in a consistent orientation.
    
    tag1 = find_closest_point(p1[0], p1[1], coords) + 1
    tag2 = find_closest_point(p2[0], p2[1], coords) + 1
    
    first_point = coords[0]
    
    lines = []
    inflow_lines = []
    grounding_lines = []
    # We accumulate every curve entity to assemble plane surfaces and later
    # tag physical groups.
    
    # Lay down the boundary polyline one vertex at a time, switching the target
    # element size depending on whether we are close to the grounding line or
    # an inflow channel.
    point = model.geo.addPoint(first_point[0], first_point[1], 0, sizes[1])
    i = 1
    # Trace the shoreline polyline, attaching a new point+edge per vertex.
    for coord in coords[1:]:
        if point_in_set(coord, grounding_points):
            # Grounding-line vertices need the smallest h so later extrusions
            # capture the contact curve sharply.
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[0])
        elif point+1 > tag1-1 and point < tag2:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[1])
        else:
            # Everything else is further away from strong gradients.
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[2])

        line = model.geo.addLine(point - 1, point)
        lines.append(line)
        
        # Store curve tags for later physical-group assignment.
        if (check_if_inflow(coords[i-1], category_array, category_trans)
            and check_if_inflow(coords[i], category_array, category_trans)):
            inflow_lines.append((1, line))
        elif (point_in_set(coords[i-1], grounding_points)
              and point_in_set(coords[i], grounding_points)):
            grounding_lines.append((1, line))
        elif ((point_in_set(coords[i], grounding_points)
              or point_in_set(coords[i-1], grounding_points))
              and coords[i][0] > coords[i-1][0]):
            grounding_lines.append((1, line))
        i += 1
        
    # Close the outer loop by linking the last shoreline vertex to the first.
    line = model.geo.addLine(point, 1)
    lines.append(line)

    if (check_if_inflow(first_point, category_array, category_trans)
        and check_if_inflow(coords[-1], category_array, category_trans)):
        inflow_lines.append((1, line))

    lines_interface = []
    # Stitch the water/ice intersection onto the boundary loop so Gmsh can
    # extrude distinct inflow surfaces later.
    tag1_eff = tag1
    if delta is not None:
        # consider interposing a fake point between coords[tag1-1] and p1
       # EDIT start
       # q1 = coords[tag1-1]
       # if np.linalg.norm([c1 - c2 for c1, c2 in zip(p1, q1)]) > delta:
       #     dir_segment = np.arctan2(p1[1] - q1[1], p1[0] - q1[0])
       #     point = model.geo.addPoint(q1[0] + delta * np.cos(dir_segment),
       #                                q1[1] + delta * np.sin(dir_segment), 0, sizes[1])
       #     line = model.geo.addLine(tag1, point)
       #     lines_interface.append(line)
       #     tag1_eff = point
        q1 = coords[tag1-1]

        if np.linalg.norm([c1 - c2 for c1, c2 in zip(p1, q1)]) > delta:
            # Compute a local tangent on the OUTLINE near q1
            # Use neighbors on the outline: tag1-2, tag1 (wrap indices if needed)
            q1m = coords[(tag1-2) % len(coords)]
            q1p = coords[(tag1)   % len(coords)]

            tx = q1p[0] - q1m[0]
            ty = q1p[1] - q1m[1]
            tlen = math.hypot(tx, ty)
            if tlen == 0:
                tx, ty = 1.0, 0.0  # fallback
            else:
                tx /= tlen
                ty /= tlen

            # Outward normal (choose sign based on your convention)
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

       # EDIT END
            map_fake_to_real[point] = tag1
            link_fake_to_real[point] = -line
    point = model.geo.addPoint(p1[0], p1[1], 0, sizes[1])
    line = model.geo.addLine(tag1_eff, point)
    lines_interface.append(line)
    if delta is not None and tag1_eff == tag1:
        map_fake_to_real[point] = tag1
        flat_points_fake.update([(0, point)])
        link_fake_to_real[point] = -line
    for point in intersect_points[1:]:
        point = model.geo.addPoint(point[0], point[1], 0, sizes[1])
        line = model.geo.addLine(point - 1, point)
        lines_interface.append(line)
    if delta is not None:
        # consider interposing a fake point between p2 (point) and coords[tag2-1]
        # START EDIT
        #q2 = coords[tag2-1]
        #if np.linalg.norm([c1 - c2 for c1, c2 in zip(p2, q2)]) > delta:
        #    dir_segment = np.arctan2(q2[1] - p2[1], q2[0] - p2[0])
        #    point_new = model.geo.addPoint(q2[0] - delta * np.cos(dir_segment),
        #                                   q2[1] - delta * np.sin(dir_segment), 0, sizes[1])
        #    line = model.geo.addLine(point, point_new)
        #    lines_interface.append(line)
        #    point = point_new
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

            #END EDIT
        map_fake_to_real[point] = tag2
        flat_points_fake.update([(0, point)])
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
                             # + [tag for (_, tag) in inflow_lines]
                             + [tag for (_, tag) in grounding_lines])
    else:
        boundary_curve_tags = list(set(lines_interior_water + lines_interior_ice)
                                 - set([tag for _, tag in intersect_lines + inflow_lines]))

    model.geo.synchronize()

    if boundary_curve_tags:
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
        
    extrude_from = surface_dim_tags
    
    all_inflow_lines = inflow_line_ents
    extrustion = model.geo.extrude(extrude_from, 0, 0, eps,
                                   numElements=[1])
    model.geo.synchronize()

    next_inflow = []
    inflow_new = []
    shallow_tags = []
    if delta is not None: map_top_to_bottom = {}
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
                model.removeEntities([entity], recursive=False)
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
                old_points = []
                for j, bound in enumerate(new_bound):
                    new_points = [tag for _, tag in 
                                  model.getBoundary([(1, bound)], combined=True, oriented=True, recursive=False)]

                    if new_points[0] in top_points:
                        if new_points[1] in top_points:
                            new_bound[j] = link_project_fake_to_fake[(new_points[0], new_points[1])]
                        else:
                            map_top_to_bottom[new_points[0]] = model.geo.addLine(map_fake_to_real[new_points[0]], new_points[1])
                            new_bound[j] = map_top_to_bottom[new_points[0]]
                            # new_bound[j] = model.geo.addLine(map_fake_to_real[new_points[0]], new_points[1])
                    elif new_points[1] in top_points:
                        map_top_to_bottom[new_points[1]] = model.geo.addLine(map_fake_to_real[new_points[1]], new_points[0])
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
                    old_points = new_points
                try:
                    new_loop = model.geo.addCurveLoop(new_bound)
                except Exception:
                    new_bound[idx_down] *= -1
                    new_loop = model.geo.addCurveLoop(new_bound)
                entity = (2, model.geo.addPlaneSurface([new_loop]))
                bound = [(1, l) for l in new_bound]
                bound_set = set(bound)
            
            if len(bound_set.intersection(ice_line_ents)) == len(bound_set):
                # Faces touching the grounding line belong to the ice shelf.
                ice.append(entity[1])
            elif len(bound_set.intersection(bath_line_ents)) > 0:
                # Faces attached to the outer shoreline become bathymetry walls.
                bath.append(entity[1])
            elif len(bound_set.intersection(inflow_line_ents)) > 0:
                # Inflow surfaces are tracked separately for boundary conditions.
                inflow_new.append(entity[1])
                next_inflow = next_inflow + bound
            elif len(bound_set.intersection(intersect_line_ents)) > 0:
                # The inlet cut-out is discarded after the first extrude pass.
                model.geo.remove([entity], recursive=True)
            else:
                # The bottom-most cap defaults to bathymetry.
                bath.append(entity[1])

    for ent in inflow_new:
        # fix inflow_surfaces by adding triangles if needed
        bound_points = set(model.getBoundary([(2, ent)], combined=True, recursive=True))

        bound_points_fake = bound_points.intersection(flat_points_fake)
        if len(bound_points_fake) > 0:
            new_bound = model.getBoundary([(2, ent)], combined=True, oriented=True, recursive=False)
            model.removeEntities([(2, ent)], recursive=False)
                    
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

            idx_down = 2
            for point_fake in bound_points_fake:
                if point_fake in bound_l_points:
                    new_bound.pop()
                    new_bound += [- map_top_to_bottom[point_fake[1]],
                                  - link_fake_to_real[point_fake[1]]]
                if point_fake in bound_r_points:
                    new_bound.pop(1)
                    new_bound = ([new_bound[0]]
                               + [link_fake_to_real[point_fake[1]],
                                  map_top_to_bottom[point_fake[1]]]
                               + new_bound[1 :])
                    idx_down += 1

            try:
                new_loop = model.geo.addCurveLoop(new_bound)
            except Exception:
                new_bound = [-tag for tag in new_bound[::-1]]
                try:
                    new_loop = model.geo.addCurveLoop(new_bound)
                except Exception:
                    new_bound = [-tag for tag in new_bound[::-1]]
                    new_bound[idx_down] *= -1
                    try:
                        new_loop = model.geo.addCurveLoop(new_bound)
                    except Exception:
                        new_bound = [-tag for tag in new_bound[::-1]]
                        new_loop = model.geo.addCurveLoop(new_bound)
            ent = model.geo.addPlaneSurface([new_loop])
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
    model.mesh.generate(2)
    if SURFACE_UNIFORM_REFINEMENT_PASSES > 0:
        _run_uniform_refinement(SURFACE_UNIFORM_REFINEMENT_PASSES, "2D")
    # gmsh.model.mesh.optimize(method="Laplace2D", niter=5)
    
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
        shallow_tags, shallow_coords, _ = model.mesh.getNodes(0, tag, includeBoundary=True, 
                                                              returnParametricCoord=False)
        shallow_coords = shallow_coords.reshape((shallow_tags.size, 3))
        for coord in shallow_coords:
            x = coord[0]
            y = coord[1]
            xy_shallow.append((x,y))
    
    mesh_filename = ensure_msh_path(filename)
   
    gmsh.write(mesh_filename)
        
    gmsh.finalize()

    try:
        xdmf_path = write_legacy_xdmf(mesh_filename)
        print(f"Wrote XDMF companion for FEniCS: {xdmf_path}")
    except Exception as err:
        print(f"[ryder_3D_meshing] Failed to write XDMF next to {mesh_filename}: {err}", file=sys.stderr)
    
    mesh2D = (mesh_filename, xy_shoreline, xy_grounding, xy_shallow)
    
    return mesh2D

def generate_mesh_mult(outline, intersect, grounding_line, m, eps, category_data, 
                       bathymetry_data, highres_data, thickness_data, surface_pos_data, 
                       scale = 1, num_of_layers = 2, adapt = False, adaptive_scales = (1/4, 2),
                       optimize: bool = ENABLE_VOLUME_OPTIMIZATION_DEFAULT, stack = 25, interpolate = True,
                       mesh_filename: Optional[str] = None, delta: Optional[float] = None,
                       band_lcmax_boundaries: Optional[Sequence[BandBoundarySpec]] = None):
    """Extrude the 2D surface mesh into 3D and sculpt it with raster data."""
    assert num_of_layers == 2, "not implemented"
    
    eps_eff = min(eps, -1)

    mesh2D = generate_2D_mesh(outline, intersect, grounding_line, 
                              category_data, m, eps_eff, adapt,
                              adaptive_scales=adaptive_scales, filename=mesh_filename,
                              delta = delta,
                              band_lcmax_boundaries=band_lcmax_boundaries)

    mesh2d_path, xy_shoreline, xy_grounding, xy_shallow = mesh2D

    
    
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
    
    bathymetry_node_tags, bathymetry_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, BATHYMETRY_TAG)
    surface_node_tags, surface_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, WATER_SURFACE_TAG)
    ice_node_tags, ice_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, ICE_TAG)
    inflow_node_tags, inflow_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, INFLOW_TAG)
    
    bathymetry_node_coords = bathymetry_node_coords.reshape((bathymetry_node_tags.size, 3))
    surface_node_coords = surface_node_coords.reshape((surface_node_tags.size, 3))
    ice_node_coords = ice_node_coords.reshape((ice_node_tags.size, 3))
    inflow_node_coords = inflow_node_coords.reshape((inflow_node_tags.size, 3))
    
    if stack > 0:
        # Stacked meshes include additional horizontal slices (synthetic layers
        # every `stack` metres).  These extra points become seeds for the later
        # Cartesian layers we embed in the final tetrahedral mesh.
        xy_ice = [(i[0], i[1]) for i in ice_node_coords]
        xy_surface = [(i[0], i[1]) for i in surface_node_coords]
        xy_water_ice_intersect = set(xy_ice).intersection(set(xy_surface))
        
        inflow_stack_points = {}
            
        stacked_points = []
        if adapt:
            sizes = (adaptive_scales[0]*m, m, adaptive_scales[1]*m)
        else:
            sizes = (m,m,m)
        
    i = 0
    for tag in bathymetry_node_tags:
        if tag not in surface_node_tags:
            x = bathymetry_node_coords[i][0]
            y = bathymetry_node_coords[i][1]
               
            if True: #not point_in_set((x,y), xy_shoreline) or point_in_set((x,y), xy_grounding):   # EDIT JOSI

            #if point_in_set((x,y), xy_grounding):
                # Drive each bathymetry vertex to the raster-derived depth and
                # optionally create extra points for stacked layers.
                z = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                              highres_trans, highres_array, -MIN_WATER_DEPTH, 
                                              highres = True, interpolate = interpolate)

                model.mesh.setNode(tag, [x, y, scale*z], [])
                
                if stack > 0 and not point_in_set((x,y), xy_shallow):
                    if (((not point_in_set(((x,y)), xy_ice)) and (tag not in inflow_node_tags)) 
                            or (point_in_set((x,y), xy_water_ice_intersect))):
                        current_depth = 0
                        while current_depth - 1.5*stack >= z:
                            current_depth = current_depth - stack
                            point = model.geo.addPoint(x, y, scale*current_depth, sizes[2])
                            stacked_points.append(point)
        i += 1

    i = 0
    for tag in ice_node_tags:
        if tag not in surface_node_tags and tag not in bathymetry_node_tags:
            x = ice_node_coords[i][0]
            y = ice_node_coords[i][1]

            # Floating ice uses the thickness and surface elevation rasters
            # to determine its draft before we rescale vertically.
            # if not (point_in_set((x,y), xy_shoreline, 50) and point_in_set((x,y), xy_grounding, 50)):
            z = get_ice_depth(x, y, thickness_trans, thickness_array, 
                              surface_pos_array, bathymetry_array, 
                              -MIN_ICE_THICKNESS, interpolate = interpolate)
            gmsh.model.mesh.setNode(tag, [x, y, scale*z], [])
            
            if stack > 0 and not point_in_set((x,y), xy_shallow):
                current_depth = z
                floor = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                              highres_trans, highres_array, -MIN_WATER_DEPTH, 
                                              highres = True, interpolate = interpolate)
                while current_depth - 1.5*stack >= floor:
                    current_depth = current_depth - stack
                    point = model.geo.addPoint(x, y, scale*current_depth, sizes[1])
                    stacked_points.append(point)
        i += 1
    
    if stack > 0:
        inflow_stack_points = {}
        # copied_inflow_data = []
        inflow_ents = gmsh.model.getEntitiesForPhysicalGroup(2, INFLOW_TAG)
        for ent in inflow_ents:
            # new_ents = model.getBoundary([(2,ent)], combined=True, oriented=True, recursive=False)
            # new_bound = [tag for dim, tag in new_ents]
            tags, coords, _ = model.mesh.getNodes(dim=2, tag=ent, includeBoundary=True)
            coords = coords.reshape((tags.size, 3))
            # copied_inflow_data.append((new_bound, coords))
            # gmsh.model.removeEntities([(2, ent)], recursive=False)
        # gmsh.model.geo.synchronize()
        # new_inflow = []
        # for new_bound, coords in copied_inflow_data:
            surface_stack_points = []
            for tag, coord in zip(tags, coords):
                if tag not in surface_node_tags:
                    x, y, floor = coord

                    current_depth = 0.
                    while current_depth - 1.5*stack >= floor:
                        current_depth = current_depth - stack
                        point = model.geo.addPoint(x, y, scale*current_depth, sizes[2])
                        surface_stack_points.append(point)
            # new_loop = model.geo.addCurveLoop(new_bound)
            # ent = model.geo.addPlaneSurface([new_loop])
            # new_inflow.append(ent)
            inflow_stack_points[ent] = surface_stack_points
        
        gmsh.model.geo.synchronize()
        
        for surface in inflow_stack_points.keys():
            model.mesh.embed(0, inflow_stack_points[surface], 2, surface)
        # model.geo.addPhysicalGroup(2, new_inflow, tag=INFLOW_TAG, name="Inflow")

    gmsh.model.geo.synchronize()
    model.mesh.generate(2)

   # gmsh.write("debug_2d.msh")     # saves the gmsh mesh
   # write_legacy_xdmf("debug_2d.msh")   # converts to XDMF
   # print("Wrote debug_2d.msh and debug_2d.xdmf for inspection")

    # model.mesh.optimize("Laplace2D", dimTags = [(2, tag) for tag in inflow_node_tags])
    gmsh.model.geo.synchronize()
    
    surf_tags = []
    for group_tag in SURFACE_PHYSICAL_TAGS:
        surf_tags.extend(model.getEntitiesForPhysicalGroup(2, group_tag))

    surf_loop = gmsh.model.geo.addSurfaceLoop(surf_tags)
    vol = gmsh.model.geo.addVolume([surf_loop])

    if stack > 0:
        gmsh.model.geo.synchronize()
        model.mesh.embed(0, stacked_points, 3, vol)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.geo.addPhysicalGroup(3, [vol], 5, name='Water')        

    model.geo.synchronize()

    if num_of_layers > 2:
        if num_of_layers in [15] or (num_of_layers in [20] and adapt):
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", INITIAL_DELAUNAY_TOL)
        if scale > 1:
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", INITIAL_DELAUNAY_TOL)

    if optimize:
        if ENABLE_VOLUME_SMOOTHING:
            # Light Laplacian smoothing when quality tuning is requested.
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
    # fix quality before and after?
    # final_volume_quality = None
    # if optimize:
    #     final_volume_quality = _improve_volume_quality(
    #         VOLUME_QUALITY_TARGET,
    #         VOLUME_QUALITY_MAX_PASSES,
    #         VOLUME_QUALITY_METHODS,
    #     )
    
    if VOLUME_UNIFORM_REFINEMENT_PASSES > 0:
        _run_uniform_refinement(VOLUME_UNIFORM_REFINEMENT_PASSES, "3D")
    
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    coords = list(coords)

    # def weight(x, y):
        # d = min(x, L - x, y, L - y)
        # if clamp <= 0:
            # return 1.0
        # t = max(0.0, min(1.0, d / clamp))
        # return t**p

    # for i in range(len(nodeTags)):
    #     tag = int(nodeTags[i])
    #     x = coords[3*i + 0]
    #     y = coords[3*i + 1]
    #     z = coords[3*i + 2]
    #     if abs(z) < EPS_Z_CLAMP:
    #         z_new = 0.0
    #     else:
    #         z_new = z
    #     # z_new = H * weight(x, y)
    #     # NOTE: pass empty parametric coords as third arg
    #     # print('z = ', z)
    #     gmsh.model.mesh.setNode(tag, [x, y, z_new], [])

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
    # Rotate and shift the mesh so that the inlet aligns with the solver axes
    # and the minimum corner starts at (0, 0, 0).
    rotated_nodes = []
    min_rot_x = float("inf")
    min_rot_y = float("inf")

    for idx, tag in enumerate(water_node_tags):
        x = water_node_coords[idx][0]
        y = water_node_coords[idx][1]
        z = water_node_coords[idx][2]
        #if ((tag not in surface_node_tags) and 
        #    (not point_in_set((x, y), xy_shoreline))):   #changing this removes the spike   # EDIT 2
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
    
    model.geo.synchronize()
    
    # if optimize:
        # Optional smoothing passes (first default, then Netgen) to improve
        # element quality when requested.
        # gmsh.model.mesh.optimize(method="")
        # gmsh.model.mesh.optimize(method="Lloyd") # Lukas: gave error for me 
        # gmsh.model.mesh.optimize(method="Netgen")
        # model.geo.synchronize()
        # water_node_tags, water_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(3, 5)
    
    
    gmsh.write(output_path)
    gmsh.finalize()
    try:
        xdmf_path = write_legacy_xdmf(output_path)
        print(f"Wrote XDMF companion for FEniCS: {xdmf_path}")
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

     #        outfile="new_data/domainmask_nonresamp.tif",

    #arrdomainmask = domainmask.ReadAsArray()      # <-- this is a numeric np.ndarray
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
            # Parallel or almost parallel
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

    # Loop over grounding line segments
    for gi in range(len(grounding_line) - 1):
        gp1 = grounding_line[gi]
        gp2 = grounding_line[gi+1]

        # Loop over shoreline segments
        for si in range(len(shoreline) - 1):
            sp1 = shoreline[si]
            sp2 = shoreline[si+1]

            point, t_g = segment_intersection(gp1, gp2, sp1, sp2)
            if point is not None:
                # parameter along grounding line: gi + t_g
                param_g = gi + t_g
                intersections.append((param_g, point))

        # If shoreline is (numerically) closed, also check last→first segment
        sp1 = shoreline[-1]
        sp2 = shoreline[0]
        point, t_g = segment_intersection(gp1, gp2, sp1, sp2)
        if point is not None:
            param_g = gi + t_g
            intersections.append((param_g, point))

    # Not enough intersections: just return original grounding line
    if len(intersections) < 2:
        return grounding_line.tolist()

    # Sort intersections along grounding_line
    intersections.sort(key=lambda x: x[0])

    param_start, p_start = intersections[0]
    param_end,   p_end   = intersections[-1]

    # Convert params to index + local interpolation param
    i_start = int(np.floor(param_start))
    i_end   = int(np.floor(param_end))

    # Build the new grounding line
    new_gl = []
    new_gl.append(p_start.tolist())

    for i in range(i_start + 1, i_end + 1):
        if 0 <= i < len(grounding_line):
            new_gl.append(grounding_line[i].tolist())

    new_gl.append(p_end.tolist())

    return new_gl


# EDIT, to compute better shoreline
def build_seathickness_shoreline(
    seathickness_path: str,
    bathymetry_path: str,
    categories_path: str,
    domainmask_path: str,
    shoreline_res: float = 60,
) -> list[tuple[float, float]]:
    """
    Build a shoreline based on where seathickness > 0, but restricted
    to valid fjord bathymetry and category mask.
    Returns a list of (x, y) shoreline coordinates.
    """
    # 1. Read rasters
    seathickness_ds = readraster(seathickness_path)        # ice-bottom thickness or draft > 0 in fjord
    bath_ds         = readraster(bathymetry_path)          # full bathymetry (with -9999 outside)
    cat_ds          = readraster(categories_path)          # categories: 0 water, 2 land ice, 3 sea ice, etc.
    domainmask       = readraster(domainmask_path)


    # 2. Build a clean binary mask:
    #    mask = 1 where seathickness>0 AND bathymetry valid & below sea level AND not outside-category junk.
    seathickness_mask = gdal_calc.Calc(
        calc=["(A>0) * (B!=-9999) * (B<=0) * (C==1)"],  # optionally multiply by (C==0) if you want only water cells
        A=seathickness_ds,
        B=bath_ds,
        C=domainmask,  # uncomment and extend calc if you want category filtering
        outfile="new_data/seathickness_mask.tif",
        NoDataValue=0,
        overwrite=True,
    )

    # 3. Build a footprint of the non-zero mask
    gdal.Footprint(
        'new_data/seathickness_outline.geojson',
        seathickness_mask,
        format="GeoJSON",
        maxPoints=shoreline_res,
        srcNodata=0,          # 0 is outside; only >0 area becomes polygons
    )

    # 4. Extract the main boundary polyline (longest curve)
    seathickness_lines = get_boundary_string('new_data/seathickness_outline.geojson')
    seathickness_shoreline = get_main_boundary(seathickness_lines)  # longest curve

    return seathickness_shoreline

# EDIT
def save_array_as_tiff(array, template_ds, out_path, dtype=gdal.GDT_Float32, nodata=None, compress=True): 
    """
    Save a numpy array as a GeoTIFF, copying geotransform and projection
    from a template GDAL dataset.

    Parameters
    ----------
    array : np.ndarray
        2D numpy array to save.
    template_ds : gdal.Dataset
        GDAL dataset providing georeferencing info (same size as array).
    out_path : str
        Output .tif file path.
    dtype : gdal datatype, optional
        Output raster type, e.g. gdal.GDT_Float32.
    nodata : float or int, optional
        Nodata value to set in the output band.
    compress : bool, optional
        If True, use LZW compression.
    """

    # Get spatial metadata from template
    gt = template_ds.GetGeoTransform()
    proj = template_ds.GetProjection()

    # Raster shape
    ny, nx = array.shape

    # Choose driver + compression options
    driver = gdal.GetDriverByName("GTIFF")
    options = ["COMPRESS=LZW"] if compress else []

    # Create output raster
    out_ds = driver.Create(out_path, nx, ny, 1, dtype, options=options)

    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    # Write array
    band = out_ds.GetRasterBand(1)

    # Nodata if provided
    if nodata is not None:
        band.SetNoDataValue(nodata)

    band.WriteArray(array)
    band.FlushCache()

    # Close
    out_ds = None
    print(f"Saved GeoTIFF → {out_path}")

class DraggablePolyline:
    def __init__(self, ax, points):
        self.ax = ax
        self.points = np.array(points, dtype=float)
        self.line, = ax.plot(self.points[:,0], self.points[:,1], "r.-", ms=8, picker=5)
        self.selected_index = None

        self.cid_press = self.line.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.line.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.line.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        # find nearest point
        d = np.hypot(self.points[:,0] - event.xdata,
                     self.points[:,1] - event.ydata)
        if d.min() < 500:   # pixel threshold — adjust to your scale
            self.selected_index = d.argmin()

    def on_motion(self, event):
        if self.selected_index is None or event.inaxes != self.ax:
            return
        # update the selected point
        self.points[self.selected_index] = [event.xdata, event.ydata]
        # update line visually
        self.line.set_data(self.points[:,0], self.points[:,1])
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self.selected_index = None

    def get_points(self):
        return self.points.tolist()

def ComputeSpaceExtent(surface_pos_data,seathickness_array):
    gt = surface_pos_data.GetGeoTransform()
    ny, nx = seathickness_array.shape

    xmin = gt[0]
    xmax = gt[0] + gt[1] * nx
    ymax = gt[3]
    ymin = gt[3] + gt[5] * ny     # gt[5] is negative

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
    cbar.set_label("Sea thickness (m)")   # optional label

    plot_main_geoline(line, ax=ax, color="k", linewidth=1.5)

    ax.set_aspect("equal", "box")
    ax.set_title("line on seathickness background")
    plt.show()


def Manual_Redraw(line,extent,seathickness_array):
    fig, ax = plt.subplots(figsize=(10,10))

    # show seathickness background
    im = ax.imshow(seathickness_array, extent=extent, cmap="viridis",vmin=0,vmax=100, origin="upper")
    fig.colorbar(im, ax=ax)

    # draw shoreline and enable dragging
    editor = DraggablePolyline(ax, line)

    plt.title("Drag shoreline points and close window to save")
    plt.show()

    # get modified points after closing the plot
    new_line = editor.get_points()

    return new_line


def main():
    """Batch-generate meshes for a selection of resolutions/settings."""
    elevation = readraster('../data/gis_data/bathymetry.tif')
    elevation_band = elevation.GetRasterBand(1)

    # EDIT
    thickness_data = readraster('../data/thickness.tif')
    surface_pos_data = readraster('../data/surface_pos.tif')
    full_bathymetry = readraster('../data/elevation.tif')

    elevation_array = elevation_band.ReadAsArray()  # EDIT
    thickness_array = thickness_data.ReadAsArray()      # <-- this is a numeric np.ndarray
    surface_array = surface_pos_data.ReadAsArray()      # <-- this is a numeric np.ndarray
    fullbathymetry_array = full_bathymetry.ReadAsArray()


    # Option 1: from the whole dataset
    
    icebottom_array = surface_array-thickness_array
    seasurface = np.minimum(icebottom_array, 0)
    seathickness_array = seasurface - fullbathymetry_array

    save_array_as_tiff(icebottom_array, surface_pos_data , 'new_data/icebottom.tif')
    save_array_as_tiff(seathickness_array, surface_pos_data , 'new_data/seathickness.tif')

    createdomainmask(elevation)  # create mask for the square

    shoreline_res = 30


    shoreline = build_seathickness_shoreline(
    seathickness_path="new_data/seathickness.tif",
    bathymetry_path="../data/elevation.tif",          # or your full_bathymetry path
    categories_path="../data/gis_data/categories.tif",
    domainmask_path="new_data/domainmask.tif",
    shoreline_res=shoreline_res,
    )  

    # plot shoreline

    extent=ComputeSpaceExtent(surface_pos_data,seathickness_array)

  

###############REDRAW SHORELINE###########


    new_shoreline=Manual_Redraw(shoreline,extent,seathickness_array)
    shoreline=new_shoreline


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

#    grounding_line=Manual_Redraw(grounding_line,extent,seathickness_array)


    #outline = add_grounding_line(shoreline, grounding_line)
    outline = shoreline 
   # PlotShoreOnRaster(outline,extent,seathickness_array)


    with open("../outline", 'w') as f:
        f.write(str(outline))

    water = gdal_calc.Calc(calc=["(A==0)"], A=categories, outfile="new_data/water.tif", 
                           NoDataValue=0, overwrite=True)
    gdal.Footprint('new_data/water.geojson', water, format="GeoJSON", 
                   maxPoints = 'unlimited', srcNodata = 0)
    water_lines = get_boundary_string('new_data/water.geojson')
    water_line = get_main_boundary(water_lines)

    intersect = get_water_ice_intersect(water_line, seaice_line)
    
    all_lines = {'Main Outline': outline, 'Grounding Line': grounding_line, 
                'Water-Ice Intersect': intersect}

    full_bathymetry = readraster('../data/elevation.tif')

    #gdal.ContourGenerateEx(elevation_band, shoreline_shp, options = ['LEVEL_INTERVAL = 100000', 
    #                                                               'FIXED_LEVELS=0', 
    #                                                               'NODATA=-9999'])

    #gdal.Rasterize(full_bathymetry, shoreline_shp.GetDataset(), burnValues=0)
    #shutil.rmtree("new_data/shoreline_py")

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

    #Unstructured:
    unstructured = [
        # Scenario(400, 1, 2, False, False, 0),
        # Scenario(300, 1, 2, False, False, 0),
        # Scenario(250, 1, 2, False, False, 0),
        # Scenario(200, 1, 2, False, False, 0),
        # Scenario(160, 1, 2, False, False, 0),
    ]
    #Scaled DOF-test 200m:
    
    scaled_200_dofs = [
        # Scenario(200, 2, 2, False, False, 0),
        # Scenario(200, 3, 2, False, False, 0),
        # Scenario(200, 4, 2, False, False, 0),
        # Scenario(200, 5, 2, False, False, 0),
        # Scenario(200, 6, 2, False, False, 0),
        # Scenario(200, 7, 2, False, False, 0),
        # Scenario(200, 8, 2, False, False, 0),
        # Scenario(200, 9, 2, False, False, 0),
        # Scenario(200, 10, 2, False, False, 0),
        # Scenario(200, 11, 2, False, False, 0),
        # Scenario(200, 12, 2, False, False, 0),
        # Scenario(200, 13, 2, False, False, 0),
        # Scenario(200, 14, 2, False, False, 0),
        # Scenario(200, 15, 2, False, False, 0),
    ]

    #Scaled DOF-test 220m:
    
    scaled_220_dofs = [
        # Scenario(220, 2, 2, False, False, 0),
        # Scenario(220, 3, 2, False, False, 0),
        # Scenario(220, 4, 2, False, False, 0),
        # Scenario(220, 5, 2, False, False, 0),
        # Scenario(220, 6, 2, False, False, 0),
        # Scenario(220, 7, 2, False, False, 0),
        # Scenario(220, 8, 2, False, False, 0),
        # Scenario(220, 9, 2, False, False, 0),
        # Scenario(220, 10, 2, False, False, 0),
        # Scenario(220, 11, 2, False, False, 0),
        # Scenario(220, 12, 2, False, False, 0),
        # Scenario(220, 13, 2, False, False, 0),
        # Scenario(220, 14, 2, False, False, 0),
        # Scenario(220, 15, 2, False, False, 0),
    ]
    
    #Scaled:
    
    scaled = [
        # Scenario(220, 6, 2, False, False, 0),
        # Scenario(220, 6, 2, False, True, 0),
        # Scenario(200, 11, 2, False, False, 0),
        # Scenario(200, 6, 2, False, True, 0),
    ]
    
    #Layered:
    
    layered = [
        # Scenario(295, 1, 9, False, False, 0),
        # Scenario(340, 1, 12, False, False, 0),
        # Scenario(410, 1, 16, False, False, 0),
        # Scenario(450, 1, 20, False, False, 0),
    ]
    
    #Stacked:
     
    stacked = [
        # Scenario(560, 1, 2, False, False, 15),
        # Scenario(560, 1, 2, False, False, 15),
        # Scenario(480, 1, 2, False, False, 20),
        # Scenario(450, 1, 2, False, False, 25),
        # Scenario(310, 1, 2, False, False, 50),
        # Scenario(255, 1, 2, False, False, 75),
    ]
    
    #Adaptive:
    adaptive = [
        # Scenario(50, 1, 2, False, True, 0), # unstructured
        
        
        # Lukas: element size does not do anything at the moment. Mesh size is controlled by LC_MIN, LC_MAX at the top of the script.
        Scenario(250, 6, 2, True, False, 250, "coarse_test.msh"),
        
        # Scenario(1000, 6, 2, False, True, 0), # coarse
        # Scenario(470, 1, 2, True, False, 15),
        # Scenario(400, 1, 2, True, False, 20),
        # Scenario(460, 1, 20, True, False, 0),
        # Scenario(190, 6, 2, True, False, 0),
    ]
    
    #Combo test:
    # combo_test = [
    #     Scenario(295, 11, 5, False, False, 0),
    #     Scenario(410, 39, 13, False, False, 0),
    # ]
    
    params = adaptive

    # eps_extrude = -LC_MIN/4.0
    eps_extrude = -1.
    for scenario in params:
        dof, meshname = generate_mesh_mult(outline, intersect, grounding_line, 
                                 scenario.element_size, eps_extrude , categories, full_bathymetry, highres, 
                                 thickness_data, surface_pos_data, scale = scenario.scale, 
                                 num_of_layers = scenario.num_layers, adapt = scenario.adapt, 
                                 adaptive_scales = (1/2, 2), optimize = scenario.optimize, 
                                 stack = scenario.stack, interpolate = True,
                                 mesh_filename = scenario.filename,
                                 delta = scenario.element_size/3/(1+scenario.adapt))
                                 # delta = None)
#
                                 # delta = None)
        dofs.append((meshname, dof))
    
    for meshname, dof in dofs:
        print(f"Mesh Name: {meshname}\t\tDOFS: {dof}")
    
    

if __name__ == "__main__":
    main()

