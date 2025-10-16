# import matplotlib.pyplot as plt
import math
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo_utils import gdal_calc
import shutil
import gmsh
from typing import NamedTuple, Sequence, List, Optional, Union
from pathlib import Path
import subprocess, sys


# plt.style.use('seaborn-v0_8')
gdal.UseExceptions()

EPS_Z_CLAMP = 10.0

WATER_SURFACE_TAG = 5
ICE_TAG = 1
BATHYMETRY_TAG = 2
INFLOW_TAG = 4
INFLOW_LINE_TAG = 3
SURFACE_PHYSICAL_TAGS = (ICE_TAG, BATHYMETRY_TAG, INFLOW_TAG, WATER_SURFACE_TAG)
GMESH_NUM_THREADS = 8
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

# --- General utilities ------------------------------------------------------

def ensure_msh_path(name: Union[str, Path]) -> str:
    """Return a filesystem path with a .msh suffix, ensuring parent directories exist."""
    path = Path(name)
    if path.suffix.lower() != ".msh":
        path = path.with_suffix(".msh")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.as_posix()

# --- Mesh sizing and quality parameters (easy to tweak) ----------------------

# Shoreline-aware size band (units match mesh coordinates)
BAND_WIDTH_MIN = 0.0          # Distance keeping LC_MIN (0 ⇒ LC_MIN starts at the boundary)
BAND_WIDTH_MAX = 3000.0       # Distance where sizes transition back to LC_MAX; must exceed BAND_WIDTH_MIN
LC_MIN = 100.0               # Fine element size inside the refinement band
LC_MAX = 500.0               # Default element size far from the refinement band
DISTANCE_SAMPLING = 40         # Distance field sampling density along curves

# 3D volumetric quality targets (used when optimize=True)
VOLUME_QUALITY_TARGET = 0.4        # Minimum acceptable mean-ratio quality after optimization
VOLUME_QUALITY_MAX_PASSES = 10      # Maximum targeted optimization passes
VOLUME_QUALITY_METHODS: Sequence[str] = (
    "",             # Default tetra optimizer (general smoothing)
    "Relocate3D",   # 3D node relocation smoothing
    # "Netgen",
)

# Optional global smoothing / robustness tweaks
MESH_SMOOTHING_ITERS = 5
INITIAL_DELAUNAY_TOL = 1e-13
MESH_ALGORITHM3D_QUALITY = 4   # Frontal 3D

# Local refinement configuration
LOCAL_REFINE_DEFAULT_ENABLED = False
LOCAL_REFINE_THRESHOLD_DEFAULT = 0.3
LOCAL_REFINE_MAX_CYCLES_DEFAULT = 5
LOCAL_REFINE_MIN_IMPROVEMENT_DEFAULT = -1
LOCAL_REFINE_MAX_BAD_FRACTION = 0.0005
LOCAL_REFINE_MAX_SEED_NODES = 10
LOCAL_REFINE_INNER_RADIUS_FACTOR = 2.5
LOCAL_REFINE_OUTER_RADIUS_FACTOR = 10.0
LOCAL_REFINE_SIZE_MIN_FACTOR = 10.5
LOCAL_REFINE_DISTANCE_SAMPLING = 50
LOCAL_REFINE_THRESHOLD_DECAY = 1.0
LOCAL_REFINE_QUALITY_MARGIN = 0.5
LOCAL_REFINE_BASE_SIZE_SCALE = 200.0
LOCAL_REFINE_RESET_BACKGROUND = False

# 2D boundary distance-field controls
BOUNDARY_FIELD_SAMPLING = 50
BOUNDARY_FIELD_DIST_MIN = 0.01
BOUNDARY_FIELD_DIST_MAX = 10.0
reset_background = LOCAL_REFINE_RESET_BACKGROUND


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


def _local_refine_bad_regions(
    base_size: float,
    quality_threshold: float = LOCAL_REFINE_THRESHOLD_DEFAULT,
    max_bad_fraction: float = LOCAL_REFINE_MAX_BAD_FRACTION,
    max_seed_nodes: int = LOCAL_REFINE_MAX_SEED_NODES,
    inner_radius_factor: float = LOCAL_REFINE_INNER_RADIUS_FACTOR,
    outer_radius_factor: float = LOCAL_REFINE_OUTER_RADIUS_FACTOR,
    size_min_factor: float = LOCAL_REFINE_SIZE_MIN_FACTOR,
) -> Optional[int]:
    """Install a background size field around the worst-quality tets and remesh.

    Returns number of seed nodes used (0 if none), or None on failure.
    """

    print(f"[local-refine] Start: base_size={base_size}, threshold={quality_threshold}, max_bad_fraction={max_bad_fraction}, max_seed_nodes={max_seed_nodes}")

    try:
        elem_types, elem_tags_list, elem_nodes_list = gmsh.model.mesh.getElements(3)
    except Exception as err:
        print(f"[ryder_3D_meshing] Unable to list 3D elements for local refine: {err}", file=sys.stderr)
        return None

    per_elem_nodes: dict[int, List[int]] = {}
    all_elem_tags: List[int] = []

    for etype, tags, nodes in zip(elem_types, elem_tags_list, elem_nodes_list):
        try:
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
        except Exception:
            continue
        if num_nodes not in (4, 10):
            continue
        tag_list = tags.tolist() if hasattr(tags, "tolist") else list(tags)
        node_list = nodes.tolist() if hasattr(nodes, "tolist") else list(nodes)
        for i, tag in enumerate(tag_list):
            start = i * num_nodes
            elem_nodes = [int(n) for n in node_list[start:start + num_nodes]]
            per_elem_nodes[int(tag)] = elem_nodes
        all_elem_tags.extend(int(t) for t in tag_list)

    if not all_elem_tags:
        print("[local-refine] No 3D elements found; skipping.")
        return 0

    try:
        qualities = gmsh.model.mesh.getElementQualities(all_elem_tags, qualityName="gamma")
    except Exception as err:
        print(f"[ryder_3D_meshing] Element quality computation failed: {err}", file=sys.stderr)
        return None

    # Normalize to a plain Python list for robust handling
    qual_list = qualities.tolist() if hasattr(qualities, "tolist") else list(qualities)

    # Report baseline quality statistics before refinement
    q_min = float(min(qual_list)) if len(qual_list) > 0 else float("inf")
    q_avg = float(sum(qual_list) / len(qual_list)) if len(qual_list) > 0 else float("nan")
    mr_before = _min_volume_quality()
    print(f"[local-refine] Elements: {len(all_elem_tags)}; min gamma={q_min:.4f}; avg gamma={q_avg:.4f}; min mean-ratio={('n/a' if mr_before is None else f'{mr_before:.4f}')}\n[local-refine] Selecting elements with gamma < {quality_threshold} (cap {int(len(all_elem_tags) * float(max_bad_fraction))})")

    tag_quality = list(zip(all_elem_tags, qual_list))
    tag_quality.sort(key=lambda tq: float(tq[1]))

    max_bad = max(1, int(len(tag_quality) * float(max_bad_fraction)))
    selected: List[int] = []
    for tag, q in tag_quality:
        if float(q) < float(quality_threshold):
            selected.append(int(tag))
            if len(selected) >= max_bad:
                break
        else:
            break

    if not selected:
        print("[local-refine] Nothing below threshold; skipping.")
        return 0

    seed_nodes: List[int] = []
    seen: set[int] = set()
    for tag in selected:
        for n in per_elem_nodes.get(tag, []):
            if n not in seen:
                seen.add(n)
                seed_nodes.append(n)
                if len(seed_nodes) >= int(max_seed_nodes):
                    break
        if len(seed_nodes) >= int(max_seed_nodes):
            break

    if not seed_nodes:
        print("[local-refine] No seed nodes collected; skipping.")
        return 0

    r_in = max(1.0, float(inner_radius_factor) * float(base_size))
    r_out = max(r_in + 1.0, float(outer_radius_factor) * float(base_size))
    size_min = max(1.0, float(size_min_factor) * float(base_size))
    size_max = max(size_min, float(base_size))

    print(f"[local-refine] Selected bad elements: {len(selected)}; seed nodes: {len(seed_nodes)}")
    print(f"[local-refine] Band radii: r_in={r_in:.2f}, r_out={r_out:.2f}; target sizes: min={size_min:.2f}, max={size_max:.2f}")

    try:
        fdist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fdist, "NodesList", seed_nodes)
        gmsh.model.mesh.field.setNumber(fdist, "Sampling", LOCAL_REFINE_DISTANCE_SAMPLING)

        fthr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMin", size_min)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMax", size_max)
        gmsh.model.mesh.field.setNumber(fthr, "DistMin", r_in)
        gmsh.model.mesh.field.setNumber(fthr, "DistMax", r_out)

        gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
        print("[local-refine] Background size field installed.")
        
        # 4. Install the size field as the new background field so the remesher
        #    honors the boundary band transition. We disable the default
        #    characteristic length heuristics so only our size field is used.
        # gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_MAX)
    except Exception as err:
        print(f"[ryder_3D_meshing] Local refine field setup failed: {err}", file=sys.stderr)
        return None

    try:
        print("[local-refine] Remeshing 3D ...")
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.removeDuplicateElements()
        gmsh.model.mesh.reclassifyNodes()
        gmsh.model.geo.synchronize()
        # Report post-remesh quality
        mr_after = _min_volume_quality()
        print(f"[local-refine] Done. min mean-ratio before={('n/a' if mr_before is None else f'{mr_before:.4f}')}, after={('n/a' if mr_after is None else f'{mr_after:.4f}')}.")
    except Exception as err:
        print(f"[ryder_3D_meshing] Remeshing after local refine failed: {err}", file=sys.stderr)
        return None
    finally:
        if reset_background:
            try:
                gmsh.model.mesh.field.remove(fthr)
            except Exception:
                pass
            try:
                gmsh.model.mesh.field.remove(fdist)
            except Exception:
                pass

    return len(seed_nodes)

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

def plot_main_geoline(shoreline):
    """Quick matplotlib diagnostic for one polyline (left commented out)."""
    
    X = []
    Y = []
    for coord in shoreline:
        X.append(coord[0])
        Y.append(coord[1])
    
    # plt.figure(figsize=(5, 5))
    # plt.scatter(X, Y)
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xlabel('Meters')
    # ax.set_ylabel('Meters')
    # plt.draw()
    
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

def generate_2D_mesh(outline, intersect, grounding_line, category_data, 
                     m, eps, num_of_layers, adapt, adaptive_scales = (1/4, 2),
                     filename: Optional[str] = None):
    """Create the surface mesh that will later be extruded to 3D.

    The 2D stage wires up every shoreline, grounding line, and inflow segment
    as explicit curves.  That makes it much easier to preserve important
    physical boundaries once we start extruding with Gmsh.
    """
    
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", GMESH_NUM_THREADS)
    
    
    
    model = gmsh.model
    model.add("2D")
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
    
    outline_len = len(coords)
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
    first_intersect_point = intersect_points[0]
    # Stitch the water/ice intersection onto the boundary loop so Gmsh can
    # extrude distinct inflow surfaces later.
    point = model.geo.addPoint(first_intersect_point[0], 
                               first_intersect_point[1], 0, sizes[1])
    line = model.geo.addLine(tag1, point)
    lines.append(line)
    for point in intersect_points[1:]:
        point = model.geo.addPoint(point[0], point[1], 0, sizes[1])
        line = model.geo.addLine(point - 1, point)
        lines.append(line)
    line = model.geo.addLine(point, tag2)
    lines.append(line)
    
    # Build two curve loops: one for the full shoreline, one for the inlet
    # patch carved out by the water/ice intersection.
    first_part = lines[:tag1-1]
    intersection = lines[outline_len:]
    second_part = lines[tag2-1:outline_len]
    bound = first_part + intersection + second_part
    loop = model.geo.addCurveLoop(bound)
    surface1 = model.geo.addPlaneSurface([loop])
    
    outer_part = lines[tag1-1:tag2-1]
    rev_intersection = [-i for i in intersection[::-1]]
    bound = outer_part + rev_intersection
    loop = model.geo.addCurveLoop(bound)
    surface2 = model.geo.addPlaneSurface([loop])
    
    surface_dim_tags = [(2, surface1), (2, surface2)]
    
    shorelines = get_shorelines(lines, intersection, inflow_lines, grounding_lines)
    intersect_lines = [(1, line) for line in intersection]

    boundary_curve_tags = set(lines)
    boundary_curve_tags.update(tag for (_, tag) in shorelines)
    boundary_curve_tags.update(tag for (_, tag) in inflow_lines)
    boundary_curve_tags.update(tag for (_, tag) in grounding_lines)
    boundary_curve_tags.update(intersection)

    model.geo.synchronize()

    if boundary_curve_tags:
        field_api = gmsh.model.mesh.field
        distance_field = field_api.add("Distance")
        field_api.setNumbers(distance_field, "CurvesList", sorted(boundary_curve_tags))
        field_api.setNumber(distance_field, "Sampling", BOUNDARY_FIELD_SAMPLING)

        boundary_field = field_api.add("Threshold")
        field_api.setNumber(boundary_field, "IField", distance_field)
        field_api.setNumber(boundary_field, "LcMin", sizes[0])
        field_api.setNumber(boundary_field, "LcMax", sizes[2])
        field_api.setNumber(boundary_field, "DistMin", float(BOUNDARY_FIELD_DIST_MIN))
        field_api.setNumber(boundary_field, "DistMax", float(BOUNDARY_FIELD_DIST_MAX))
        field_api.setAsBackgroundMesh(boundary_field)

    model.mesh.generate(2)
        
    ice = [surface2]
    inflow = []
    bath = []
    mid_layers = []
    # As we extrude layer by layer we keep track of which newly created
    # surfaces belong to the ice shelf, bathymetry, inflow, or interior slabs.
        
    ice_line_ents = set(grounding_lines)
    bath_line_ents = set(shorelines)
    inflow_line_ents = set(inflow_lines)
    intersect_line_ents = set(intersect_lines)
        
    extrude_from = surface_dim_tags
    
    mid_group_1 = []
    mid_group_2 = []
    
    all_inflow_lines = inflow_line_ents
    for layer in range(2, num_of_layers+1):
        # Step the 2D surface downward by `eps/(layers-1)` so each layer ends up
        # with an identical thickness when we later sculpt the bathymetry.
        extrustion = model.geo.extrude(extrude_from, 0, 0, 
                                       round(eps*(1/(num_of_layers-1))), #avoid float rounding error
                                       numElements=[1])
        model.geo.synchronize()
            
        next_extrusion = []
        next_ice = []
        next_bath = []
        next_inflow = []
        next_intersect = []
        
        for entity in extrustion:
            # Gmsh returns both volume (dim=3) and surface (dim=2) entities; we
            # only keep the surfaces and classify them by adjacency.
            if entity[0] == 3:
                model.geo.remove([entity], recursive=False)
            else:
                bound = model.getBoundary([entity], oriented=False)
                bound_set = set(bound)

                if len(bound_set.intersection(ice_line_ents)) > 0:
                    # Faces touching the grounding line belong to the ice shelf.
                    ice.append(entity[1])
                    next_ice = next_ice + bound
                elif len(bound_set.intersection(bath_line_ents)) > 0:
                    # Faces attached to the outer shoreline become bathymetry walls.
                    bath.append(entity[1])
                    next_bath = next_bath + bound
                elif len(bound_set.intersection(inflow_line_ents)) > 0:
                    # Inflow surfaces are tracked separately for boundary conditions.
                    inflow.append(entity[1])
                    next_inflow = next_inflow + bound
                elif len(bound_set.intersection(intersect_line_ents)) > 0:
                    # The inlet cut-out is discarded after the first extrude pass.
                    model.geo.remove([entity], recursive=True)
                    next_intersect = next_intersect + bound
                else:
                    if layer == num_of_layers:
                        # The bottom-most cap defaults to bathymetry.
                        bath.append(entity[1])
                    else:
                        # Everything in between becomes an internal slab so we
                        # can reposition it during the 3D morphing stage.
                        mid_layers.append(entity[1])
                    next_extrusion.append(entity)
        if layer != num_of_layers:
            # Record surface IDs for the next pair of internal layers; the 3D
            # stage uses these markers when redistributing vertical nodes.
            mid_group_1.append(min([tag for (dim, tag) in next_extrusion])) 
            mid_group_2.append(max([tag for (dim, tag) in next_extrusion]))            
        extrude_from = next_extrusion
        ice_line_ents = set(next_ice)
        bath_line_ents = set(next_bath)
        inflow_line_ents = set(next_inflow)
        all_inflow_lines = all_inflow_lines.union(inflow_line_ents)
        intersect_line_ents = set(next_intersect)
        model.geo.synchronize()
        
    model.geo.synchronize()
    
    all_inflow_lines = [tag for dim, tag in all_inflow_lines]
    # --- Build a boundary-aware size field -----------------------------------
    # 1. Collect the curve entities that bound the target surfaces. We prefer
    #    user-provided Physical Line tags but fall back to the geometric
    #    boundary of the selected surfaces.
    curve_tags = _curves_from_physical_lines(phys_line_targets)
    if not curve_tags:
        volumes = gmsh.model.getEntities(3)
        if volumes:
            boundary = gmsh.model.getBoundary(volumes, oriented=False, recursive=False)
            surface_tags_after = sorted({tag for dim, tag in boundary if dim == 2})
        else:
            surface_tags_after = [tag for (dim, tag) in gmsh.model.getEntities(2)]
        curve_tags = _curves_from_surfaces(surface_tags_after)

    if not curve_tags:
        gmsh.finalize()
        raise RuntimeError("Unable to detect boundary curves for the refinement band.")

    # 2. Evaluate the signed distance to those curves. The field sampling governs
    #    how finely the distance is interpolated along each curve; increase
    #    `DISTANCE_SAMPLING` if you need a sharper transition.
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field, "Sampling", DISTANCE_SAMPLING)

    # 3. Map distance -> element size with a Threshold field. Distances smaller
    #    than `inner_band` use `LC_MIN`, distances larger than `outer_band`
    #    use `LC_MAX`, and the values in between are linearly interpolated.
    inner_band = max(0.0, float(BAND_WIDTH_MIN))  # 0 => LC_MIN starts exactly on the boundary
    outer_band = float(BAND_WIDTH_MAX)
    if outer_band <= inner_band:
        gmsh.finalize()
        raise ValueError("BAND_WIDTH_MAX must be greater than BAND_WIDTH_MIN to create a transition zone.")

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", LC_MIN)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", LC_MAX)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", inner_band)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", outer_band)

    # 4. Install the size field as the new background field so the remesher
    #    honors the boundary band transition. We disable the default
    #    characteristic length heuristics so only our size field is used.
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_MAX)
    
    
    # Physical groups map mesh regions to boundary conditions in downstream
    # solvers (FEniCS, etc.).  Keep them descriptive so post-processing stays
    # readable.
    model.geo.addPhysicalGroup(2, [surface1], tag=WATER_SURFACE_TAG, name="Water Surface")
    model.geo.addPhysicalGroup(2, ice, tag=ICE_TAG, name="Ice")
    model.geo.addPhysicalGroup(2, bath, tag=BATHYMETRY_TAG, name="Bathymetry")
    model.geo.addPhysicalGroup(2, inflow, tag=INFLOW_TAG, name="Inflow")
    model.geo.addPhysicalGroup(1, all_inflow_lines, tag=INFLOW_LINE_TAG, name="Inflow lines")

    if num_of_layers > 2:
        model.geo.addPhysicalGroup(2, mid_layers, tag=6, name="Mid")
  
    model.geo.synchronize()
    model.mesh.generate(2)
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
    
    mesh_filename = ensure_msh_path(filename or f"2D_{num_of_layers}_layer_mesh.msh")
   
    gmsh.write(mesh_filename)
        
    gmsh.finalize()

    mesh2D = (mesh_filename, xy_shoreline, xy_grounding, mid_group_1, mid_group_2)
    
    return mesh2D

def generate_mesh_mult(outline, intersect, grounding_line, m, eps, category_data, 
                       bathymetry_data, highres_data, thickness_data, surface_pos_data, 
                       scale = 1, num_of_layers = 2, adapt = False, adaptive_scales = (1/4, 2),
                       optimize = False, stack = 25, interpolate = True,
                       mesh_filename: Optional[str] = None,
                       local_refine: bool = LOCAL_REFINE_DEFAULT_ENABLED,
                       local_refine_threshold: Optional[float] = LOCAL_REFINE_THRESHOLD_DEFAULT,
                       local_refine_max_cycles: int = LOCAL_REFINE_MAX_CYCLES_DEFAULT,
                       local_refine_min_improvement: float = LOCAL_REFINE_MIN_IMPROVEMENT_DEFAULT):
    """Extrude the 2D surface mesh into 3D and sculpt it with raster data."""
    
    if num_of_layers > 2:
        eps = eps*(num_of_layers - 1)

    mesh2D = generate_2D_mesh(outline, intersect, grounding_line, 
                              category_data, m, eps, num_of_layers, adapt,
                              adaptive_scales=adaptive_scales, filename=mesh_filename)

    mesh2d_path, xy_shoreline, xy_grounding, mid_group_1, mid_group_2 = mesh2D
    
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
    
    if stack > 0:
        # Stacked meshes include additional horizontal slices (synthetic layers
        # every `stack` metres).  These extra points become seeds for the later
        # Cartesian layers we embed in the final tetrahedral mesh.
        scale = 1
        xy_ice = [(i[0], i[1]) for i in ice_node_coords]
        xy_surface = [(i[0], i[1]) for i in surface_node_coords]
        xy_water_ice_intersect = set(xy_ice).intersection(set(xy_surface))
        
        inflow_stack_points = {}
            
        stacked_points = []
        if adapt:
            sizes = (adaptive_scales[0]*m, m, adaptive_scales[1]*m)
        else:
            sizes = (m,m,m)
        
        points = [tag for dim, tag in model.getEntities(0)]
        point_coords = [model.getValue(0, tag, []) for tag in points]
        point_xy = [(x, y) for x, y, z in point_coords]
    
    if num_of_layers > 2:
        scale = scale
    else:
        scale = scale
    
    i = 0
    for tag in bathymetry_node_tags:
        if tag not in surface_node_tags:
            x = bathymetry_node_coords[i][0]
            y = bathymetry_node_coords[i][1]
               
            if not point_in_set((x,y), xy_shoreline):#(x, y) not in xy_shoreline:
                # Drive each bathymetry vertex to the raster-derived depth and
                # optionally create extra points for stacked layers.
                z = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                              highres_trans, highres_array, eps, 
                                              highres = True, interpolate = interpolate)

                model.mesh.setNode(tag, [x, y, scale*z], [])
                
                if stack > 0:
                    if (((not point_in_set(((x,y)), xy_ice)) and (tag not in inflow_node_tags)) 
                            or (point_in_set((x,y), xy_water_ice_intersect))):
                        current_depth = 0
                        while current_depth - 1.5*stack >= z:
                            current_depth = current_depth - stack
                            point = model.geo.addPoint(x, y, current_depth, sizes[2])
                            stacked_points.append(point)
                        
        i += 1
    
    i = 0
    for tag in ice_node_tags:
        if tag not in surface_node_tags:
            x = ice_node_coords[i][0]
            y = ice_node_coords[i][1]
            
            if (point_in_set((x,y), xy_grounding)) and (tag not in bathymetry_node_tags):
                # Grounded ice merges smoothly into the bedrock; keep it just
                # above the seabed using the small epsilon offset.
                depth = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                              highres_trans, highres_array, eps, 
                                              highres = True, interpolate = interpolate)
                z = depth - eps
                gmsh.model.mesh.setNode(tag, [x, y, scale*z], [])
            elif tag not in bathymetry_node_tags:
                # Floating ice uses the thickness and surface elevation rasters
                # to determine its draft before we rescale vertically.
                z = get_ice_depth(x, y, thickness_trans, thickness_array, 
                                      surface_pos_array, bathymetry_array, 
                                      eps, interpolate = interpolate)
                gmsh.model.mesh.setNode(tag, [x, y, scale*z], [])
                
                if stack > 0:
                    current_depth = z
                    floor = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                  highres_trans, highres_array, eps, 
                                                  highres = True, interpolate = interpolate)
                    while current_depth - 1.5*stack >= floor:
                        current_depth = current_depth - stack
                        point = model.geo.addPoint(x, y, current_depth, sizes[1])
                        stacked_points.append(point)
        i += 1
    
    if num_of_layers == 2:
        # The 2-layer meshes dedicate one surface to bathymetry and one to the
        # water/ice interface.  In that case we rebuild inflow patches manually
        # so stacked points can tie into them cleanly.
        
        inflow_ents = gmsh.model.getEntitiesForPhysicalGroup(2, INFLOW_TAG)
        # Physical groups from the 2D mesh do not survive the manual stacking
        # tweaks.  Remove and recreate them once we have injected extra points.
        model.geo.removePhysicalGroups([(2, INFLOW_TAG)])
        # Layered meshes do not need the 1D inflow tags; clear them so we can
        # rebuild higher dimensional embeddings below.
        model.geo.removePhysicalGroups([(1, INFLOW_LINE_TAG)])
        new_inflow = []
        
        copied_inflow_data = []
        for ent in inflow_ents:
            new_ents = model.getBoundary([(2,ent)], combined=True, oriented=True, recursive=False)
            new_bound = [tag for dim, tag in new_ents]
            tags, coords, _ = model.mesh.getNodes(dim=2, tag=ent, includeBoundary=True)
            coords = coords.reshape((tags.size, 3))
            xy_coords = [(i[0], i[1]) for i in coords]
            inflow_coord_set = set(xy_coords)
            copied_inflow_data.append((new_bound, inflow_coord_set))
            gmsh.model.removeEntities([(2, ent)], recursive=False)
            
        gmsh.model.geo.synchronize()
        inflow_data = copied_inflow_data
        new_lines = []
        skip = []
        for new_bound, inflow_coord_set in inflow_data:
            # Gmsh returns the boundary of each inflow surface as a list of
            # curve IDs.  We rebuild the loop to insert extra stacked points,
            # ensuring vertical prisms line up with inflow faces.
            new_bound = [new_bound[0], new_bound[-1], new_bound[1], new_bound[2]]

            if stack > 0:

                surface_stack_points = []
                for x, y in inflow_coord_set:

                    if point_in_set((x,y), skip):
                        new_bound = [new_bound[0], new_bound[1], new_bound[2], [-i for i in new_lines[::-1]]]
                    elif (not point_in_set((x,y), xy_shoreline)) and (not point_in_set((x,y), point_xy)):
                        current_depth = 0
                        floor = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                      highres_trans, highres_array, eps, 
                                                      highres = True, interpolate = interpolate)

                        while current_depth - 1.5*stack >= floor:
                            current_depth = current_depth - stack
                            point = model.geo.addPoint(x, y, current_depth, sizes[2])
                            surface_stack_points.append(point)
                    elif (point_in_set((x,y), point_xy)) and (not point_in_set((x,y), xy_shoreline)):
                        current_depth = 0
                        floor = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                      highres_trans, highres_array, eps, 
                                                      highres = True, interpolate = interpolate)
                        if current_depth - 1.5*stack >= floor:
                            skip.append((x,y))
                            line_points = []
                            floor = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                          highres_trans, highres_array, eps, 
                                                          highres = True, interpolate = interpolate)
                            boundary_points = model.getBoundary([(1, new_bound[1])], combined=True, oriented=True, recursive=False)
                            gmsh.model.removeEntities([(1, new_bound[1])], recursive=False)
                        
                            gmsh.model.geo.translate([(0, boundary_points[1][1])], 0, 0, floor + 1)
                            while current_depth - 1.5*stack >= floor:
                                current_depth = current_depth - stack
                                
                                point = model.geo.addPoint(x, y, current_depth, sizes[2])
                                line_points.append(point)
                            new_lines = []
                            new_lines.append(model.geo.addLine(boundary_points[0][1], line_points[0]))
                            for point in line_points[:-1]:
                                new_lines.append(model.geo.addLine(point, point+1))
                            new_lines.append(model.geo.addLine(line_points[-1], boundary_points[1][1]))  
                            new_bound = [new_bound[0], new_lines, new_bound[2], new_bound[-1]]

                
                new_bound = [x for i in new_bound for x in (i if isinstance(i, list) else [i])]
                new_loop = model.geo.addCurveLoop(new_bound)
                new_surface = model.geo.addPlaneSurface([new_loop])
                new_inflow.append(new_surface)
                inflow_stack_points[new_surface] = surface_stack_points
            else:
                new_loop = model.geo.addCurveLoop(new_bound)
                new_surface = model.geo.addPlaneSurface([new_loop])
                new_inflow.append(new_surface)
        
        gmsh.model.geo.synchronize()
        
        if stack > 0:
            # Anchor the newly created vertical stack points into the inflow
            # surface so later tetrahedra follow those columns.
            for surface in inflow_stack_points.keys():
                model.mesh.embed(0, inflow_stack_points[surface], 2, surface)

        model.geo.addPhysicalGroup(2, new_inflow, tag=INFLOW_TAG, name="Inflow")
        
        gmsh.model.geo.synchronize()
        model.mesh.generate(2)
        gmsh.model.geo.synchronize()
        
        surf_tags = []
        for group_tag in SURFACE_PHYSICAL_TAGS:
            surf_tags.extend(model.getEntitiesForPhysicalGroup(2, group_tag))

        surf_loop = gmsh.model.geo.addSurfaceLoop(surf_tags)
        vol = gmsh.model.geo.addVolume([surf_loop])

        if stack > 0:
            gmsh.model.geo.synchronize()
            model.mesh.embed(0, stacked_points, 3, vol)
            gmsh.model.geo.addPhysicalGroup(3, [vol], 5, name='Water')
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        else:
            gmsh.model.geo.addPhysicalGroup(3, [vol], 5, name='Water')
    
    else:
        model.geo.removePhysicalGroups([(1, INFLOW_LINE_TAG)])
        gmsh.model.geo.synchronize()
        
        if scale > 1:
            inflow_ents = gmsh.model.getEntitiesForPhysicalGroup(2, INFLOW_TAG)
            model.geo.removePhysicalGroups([(2, INFLOW_TAG)])
            new_inflow = []
            
            copied_inflow_data = []
            for ent in inflow_ents:
                new_ents = model.getBoundary([(2,ent)], combined=True, oriented=True, recursive=False)
                new_bound = [tag for dim, tag in new_ents]
                tags, coords, _ = model.mesh.getNodes(dim=2, tag=ent, includeBoundary=True)
                coords = coords.reshape((tags.size, 3))
                xy_coords = [(i[0], i[1]) for i in coords]
                inflow_coord_set = set(xy_coords)
                copied_inflow_data.append((new_bound, inflow_coord_set))
                gmsh.model.removeEntities([(2, ent)], recursive=False)

            gmsh.model.geo.synchronize()
            inflow_data = copied_inflow_data
            
            new_lines = []
            skip = []
            for new_bound, inflow_coord_set in inflow_data:
                    
                new_bound = [new_bound[0], new_bound[-1], new_bound[1], new_bound[2]]
                new_loop = model.geo.addCurveLoop(new_bound)
                new_surface = model.geo.addPlaneSurface([new_loop])
                new_inflow.append(new_surface)
            
            gmsh.model.geo.synchronize()

            model.geo.addPhysicalGroup(2, new_inflow, tag=INFLOW_TAG, name="Inflow")
            
            gmsh.model.geo.synchronize()
            model.mesh.generate(2)
            gmsh.model.geo.synchronize()
        
        surf_tags = []
        for group_tag in SURFACE_PHYSICAL_TAGS:
            surf_tags.extend(model.getEntitiesForPhysicalGroup(2, group_tag))

        surf_loop = gmsh.model.geo.addSurfaceLoop(surf_tags)
        vol = gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.addPhysicalGroup(3, [vol], 5, name='Water')
        
        mid_layers = [tag for tag in model.getEntitiesForPhysicalGroup(2, 6)]
        gmsh.model.geo.synchronize()
        model.mesh.embed(2, mid_layers, 3, vol)
        gmsh.model.geo.synchronize()
        layer = 1
        for surf in mid_group_1:
            mid_node_tags, mid_node_coords, _ = model.mesh.getNodes(dim=2, tag=surf, 
                                                                    includeBoundary=True)
            mid_node_coords = mid_node_coords.reshape((mid_node_tags.size, 3))
            i = 0
            for tag in mid_node_tags:
                x = mid_node_coords[i][0]
                y = mid_node_coords[i][1]
                if not point_in_set((x,y), xy_shoreline):#(x, y) not in xy_shoreline:
                    depth = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                  highres_trans, highres_array, eps, 
                                                  highres = True, interpolate = interpolate)
                    z = depth*(layer/(num_of_layers-1))
                    model.mesh.setNode(tag, [x, y, scale*z], [])
                i += 1
            layer += 1
        
        layer = 1
        for surf in mid_group_2:
            skip_tags, _, _ = model.mesh.getNodes(dim=2, tag=mid_group_1[layer-1], 
                                                  includeBoundary=True)
            mid_node_tags, mid_node_coords, _ = model.mesh.getNodes(dim=2, tag=surf, 
                                                                    includeBoundary=True)
            mid_node_coords = mid_node_coords.reshape((mid_node_tags.size, 3))
            i = 0
            for tag in mid_node_tags:
                x = mid_node_coords[i][0]
                y = mid_node_coords[i][1]
                if (point_in_set((x,y), xy_grounding)) and (tag not in bathymetry_node_tags):
                    depth = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                  highres_trans, highres_array, eps, 
                                                  highres = True, interpolate = interpolate)
                    z = depth - eps*((num_of_layers - 1 - layer)/(num_of_layers-1))
                    gmsh.model.mesh.setNode(tag, [x, y, scale*z], [])
                elif (not point_in_set((x,y), xy_shoreline)) and (tag not in skip_tags):
                    bath = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                                 highres_trans, highres_array, eps, 
                                                 highres = True, interpolate = interpolate)
                    ice = get_ice_depth(x, y, thickness_trans, thickness_array, 
                                        surface_pos_array, bathymetry_array, 
                                        eps, interpolate = interpolate)
                    wc_height = bath - ice
                    z = wc_height*(layer/(num_of_layers-1)) + ice
                    model.mesh.setNode(tag, [x, y, scale*z], [])
                i += 1
            layer += 1
    
    
    model.geo.synchronize()

    if num_of_layers > 2:
        if num_of_layers in [15] or (num_of_layers in [20] and adapt):
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", INITIAL_DELAUNAY_TOL)
        if scale > 1:
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", INITIAL_DELAUNAY_TOL)

    if optimize:
        # Light Laplacian smoothing when quality tuning is requested.
        gmsh.option.setNumber("Mesh.Smoothing", MESH_SMOOTHING_ITERS)
    # Prefer quality-oriented 3D algorithm when tuning quality or local refine.
    if optimize or local_refine:
        try:
            gmsh.option.setNumber("Mesh.Algorithm3D", MESH_ALGORITHM3D_QUALITY)
            print(f"[3D] Using Mesh.Algorithm3D = {MESH_ALGORITHM3D_QUALITY} (quality-oriented)")
        except Exception:
            print("[3D] Could not set Mesh.Algorithm3D; proceeding with default.")

    model.mesh.generate(3)
    
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

    # Optional: locally refine around worst-quality tets, then re-optimize (multi-cycle).
    if (optimize and local_refine):
        refine_threshold = float(local_refine_threshold) if local_refine_threshold is not None else float(VOLUME_QUALITY_TARGET)
        cycles = max(1, int(local_refine_max_cycles))
        current_q = final_volume_quality if final_volume_quality is not None else _min_volume_quality()
        print(f"[local-refine] Requested with threshold={refine_threshold}. Current min mean-ratio={('n/a' if current_q is None else f'{current_q:.4f}')}. Max cycles={cycles}.")
        last_q = current_q if current_q is not None else 0.0
        for c in range(1, cycles + 1):
            if current_q is not None and current_q >= refine_threshold:
                print(f"[local-refine] Target reached (min={current_q:.4f} >= {refine_threshold}). Stopping at cycle {c-1}.")
                break
            cycle_threshold = max((current_q or 0.0) + LOCAL_REFINE_QUALITY_MARGIN, refine_threshold)
            cycle_base_size = max(1.0, float(m) * (LOCAL_REFINE_BASE_SIZE_SCALE ** (c - 1)))
            print(f"[local-refine] Cycle {c}/{cycles}: refining regions below {cycle_threshold:.4f} (base_size={cycle_base_size:.2f}) ...")
            seeds = _local_refine_bad_regions(
                base_size=cycle_base_size,
                quality_threshold=float(cycle_threshold),
            )
            if not seeds:
                print("[local-refine] No seeds applied in this cycle; stopping further refinement.")
                break
            print(f"[local-refine] Seeds used in cycle {c}: {seeds}. Re-running optimizer ...")
            final_volume_quality = _improve_volume_quality(
                VOLUME_QUALITY_TARGET,
                VOLUME_QUALITY_MAX_PASSES,
                VOLUME_QUALITY_METHODS,
            )
            current_q = final_volume_quality if final_volume_quality is not None else _min_volume_quality()
            if current_q is None:
                print("[local-refine] Could not evaluate quality after optimization; stopping.")
                break
            improvement = current_q - (last_q if last_q is not None else 0.0)
            print(f"[local-refine] After cycle {c}: min mean-ratio={current_q:.4f} (prev={('n/a' if last_q is None else f'{last_q:.4f}')}); Δ={improvement:.4e}.")
            # if improvement <= 0:
            #     print("[local-refine] No improvement observed; stopping further refinement.")
            #     break
            if local_refine_min_improvement > 0 and improvement < float(local_refine_min_improvement):
                print(f"[local-refine] Improvement {improvement:.4e} < {local_refine_min_improvement}; stopping early.")
                break
            last_q = current_q
            refine_threshold = max(current_q + LOCAL_REFINE_QUALITY_MARGIN, refine_threshold * LOCAL_REFINE_THRESHOLD_DECAY)

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
    
    print(meshname)
    


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
        if ((tag not in surface_node_tags) and 
            (not point_in_set((x, y), xy_shoreline))):
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
    dof = len(water_node_tags)
    return dof, meshname

def main():
    """Batch-generate meshes for a selection of resolutions/settings."""
    elevation = readraster('../data/gis_data/bathymetry.tif')
    elevation_band = elevation.GetRasterBand(1)

    SR = osr.SpatialReference(elevation.GetProjection())
    ogr_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource('new_data/shoreline_py')
    shoreline_shp = ogr_ds.CreateLayer('shoreline', SR)

    gdal.ContourGenerateEx(elevation_band, shoreline_shp, 
                           options = ['LEVEL_INTERVAL = 100000', 'FIXED_LEVELS=0', 'NODATA=-9999'])

    gdal.Rasterize(elevation, shoreline_shp.GetDataset(), burnValues=0)
    bathymetry = gdal_calc.Calc(calc=["-9999*(B>0) + A*(A<=0)"], A=elevation, 
                                B=elevation, outfile="new_data/bathymetry.tif", 
                                NoDataValue=-9999, overwrite=True)

    shoreline_res = 60

    gdal.Footprint('new_data/shorelines.geojson', bathymetry, format="GeoJSON", 
                   maxPoints = shoreline_res, srcNodata = -9999)
    shorelines = get_boundary_string('new_data/shorelines.geojson')
    shoreline = get_main_boundary(shorelines)

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

    outline = add_grounding_line(shoreline, grounding_line)
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
    plot_full_outline(all_lines)

    full_bathymetry = readraster('../data/elevation.tif')

    gdal.ContourGenerateEx(elevation_band, shoreline_shp, options = ['LEVEL_INTERVAL = 100000', 
                                                                   'FIXED_LEVELS=0', 
                                                                   'NODATA=-9999'])

    #gdal.Rasterize(full_bathymetry, shoreline_shp.GetDataset(), burnValues=0)
    shutil.rmtree("new_data/shoreline_py")

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
        Scenario(400, 1, 2, False, False, 0),
        Scenario(300, 1, 2, False, False, 0),
        Scenario(250, 1, 2, False, False, 0),
        Scenario(200, 1, 2, False, False, 0),
        Scenario(160, 1, 2, False, False, 0),
    ]
    #Scaled DOF-test 200m:
    
    scaled_200_dofs = [
        Scenario(200, 2, 2, False, False, 0),
        Scenario(200, 3, 2, False, False, 0),
        Scenario(200, 4, 2, False, False, 0),
        Scenario(200, 5, 2, False, False, 0),
        Scenario(200, 6, 2, False, False, 0),
        Scenario(200, 7, 2, False, False, 0),
        Scenario(200, 8, 2, False, False, 0),
        Scenario(200, 9, 2, False, False, 0),
        Scenario(200, 10, 2, False, False, 0),
        Scenario(200, 11, 2, False, False, 0),
        Scenario(200, 12, 2, False, False, 0),
        Scenario(200, 13, 2, False, False, 0),
        Scenario(200, 14, 2, False, False, 0),
        Scenario(200, 15, 2, False, False, 0),
    ]

    #Scaled DOF-test 220m:
    
    scaled_220_dofs = [
        Scenario(220, 2, 2, False, False, 0),
        Scenario(220, 3, 2, False, False, 0),
        Scenario(220, 4, 2, False, False, 0),
        Scenario(220, 5, 2, False, False, 0),
        Scenario(220, 6, 2, False, False, 0),
        Scenario(220, 7, 2, False, False, 0),
        Scenario(220, 8, 2, False, False, 0),
        Scenario(220, 9, 2, False, False, 0),
        Scenario(220, 10, 2, False, False, 0),
        Scenario(220, 11, 2, False, False, 0),
        Scenario(220, 12, 2, False, False, 0),
        Scenario(220, 13, 2, False, False, 0),
        Scenario(220, 14, 2, False, False, 0),
        Scenario(220, 15, 2, False, False, 0),
    ]
    
    #Scaled:
    
    scaled = [
        Scenario(220, 6, 2, False, False, 0),
        Scenario(220, 6, 2, False, True, 0),
        Scenario(200, 11, 2, False, False, 0),
        Scenario(200, 6, 2, False, True, 0),
    ]
    
    #Layered:
    
    layered = [
        Scenario(295, 1, 9, False, False, 0),
        Scenario(340, 1, 12, False, False, 0),
        Scenario(410, 1, 16, False, False, 0),
        Scenario(450, 1, 20, False, False, 0),
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
        Scenario(50, 1, 2, False, True, 0, "lukas-mesh/stacked_fine.msh"), # stacked fine
        # Scenario(1000, 6, 2, False, True, 0), # coarse
        # Scenario(470, 1, 2, True, False, 15),
        # Scenario(400, 1, 2, True, False, 20),
        # Scenario(460, 1, 20, True, False, 0),
        # Scenario(190, 6, 2, True, False, 0),
    ]
    
    #Combo test:
    combo_test = [
        Scenario(295, 11, 5, False, False, 0),
        Scenario(410, 39, 13, False, False, 0),
    ]
    
    params = adaptive#stacked#combo_test
    
    for scenario in params:
        dof, meshname = generate_mesh_mult(outline, intersect, grounding_line, 
                                 scenario.element_size, -1.0*LC_MIN/3.0, categories, full_bathymetry, highres, 
                                 thickness_data, surface_pos_data, scale = scenario.scale, 
                                 num_of_layers = scenario.num_layers, adapt = scenario.adapt, 
                                 adaptive_scales = (1/4, 2), optimize = scenario.optimize, 
                                 stack = scenario.stack, interpolate = True,
                                 mesh_filename = scenario.filename,
                                 local_refine = LOCAL_REFINE_DEFAULT_ENABLED,
                                 local_refine_threshold = LOCAL_REFINE_THRESHOLD_DEFAULT,
                                 local_refine_max_cycles = LOCAL_REFINE_MAX_CYCLES_DEFAULT,
                                 local_refine_min_improvement = LOCAL_REFINE_MIN_IMPROVEMENT_DEFAULT)
        dofs.append((meshname, dof))
    
    for meshname, dof in dofs:
        print(f"Mesh Name: {meshname}\t\tDOFS: {dof}")
    
    

if __name__ == "__main__":
    main()
