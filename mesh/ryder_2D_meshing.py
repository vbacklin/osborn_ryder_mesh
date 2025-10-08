# import matplotlib.pyplot as plt
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import gmsh
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo_utils import gdal_calc
import shutil
# plt.style.use('seaborn-v0_8')
gdal.UseExceptions()

BATHYMETRY_TAG = 2
INFLOW_LINE_TAG = 3
GROUNDING_LINE_TAG = 6
DEFAULT_SURFACE_TAG = 999

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

def _point_segment_distance(px: float, py: float,
                            ax: float, ay: float,
                            bx: float, by: float) -> float:
    """Return the minimal distance between point P and the segment AB."""
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _distance_to_outline(px: float, py: float, outline: Tuple[Tuple[float, float], ...]) -> float:
    """Return minimal Euclidean distance from point P to an outline polyline."""
    min_distance = float("inf")
    for (ax, ay), (bx, by) in zip(outline, outline[1:]):
        min_distance = min(min_distance,
                           _point_segment_distance(px, py, ax, ay, bx, by))
    # Ensure the loop is closed
    ax, ay = outline[-1]
    bx, by = outline[0]
    min_distance = min(min_distance,
                       _point_segment_distance(px, py, ax, ay, bx, by))
    return min_distance


def tag_boundary_band_elements(outline: Tuple[Tuple[float, float], ...],
                               band_width: float) -> Dict[int, Set[int]]:
    """Return surface triangles whose centroids lie within ``band_width``."""
    if not band_width or band_width <= 0:
        return {}

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coord_lookup = {}
    for idx, tag in enumerate(node_tags):
        coord_lookup[int(tag)] = (
            float(node_coords[3 * idx]),
            float(node_coords[3 * idx + 1])
        )

    boundary_triangles: Dict[int, Set[int]] = defaultdict(set)

    for _, surface_tag in gmsh.model.getEntities(2):
        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, surface_tag)
        for elem_type, elem_tags, elem_node_tags in zip(elem_types,
                                                        elem_tags_list,
                                                        elem_node_tags_list):
            if elem_tags.size == 0:
                continue
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
            if num_nodes != 3:
                # Only refine first-order triangles; skip other element types.
                continue
            tags = elem_tags.tolist()
            nodes = elem_node_tags.tolist()
            for idx, elem_tag in enumerate(tags):
                node_ids = nodes[idx * num_nodes:(idx + 1) * num_nodes]
                cx = sum(coord_lookup[int(n)][0] for n in node_ids) / num_nodes
                cy = sum(coord_lookup[int(n)][1] for n in node_ids) / num_nodes
                distance = _distance_to_outline(cx, cy, outline)
                if distance <= band_width:
                    boundary_triangles[int(surface_tag)].add(int(elem_tag))

    return {surf: triangles for surf, triangles in boundary_triangles.items() if triangles}

def generate_2D_mesh(
    outline,
    intersection,
    grounding_line,
    category_data,
    element_size,
    adapt=False,
    adaptive_scales=(0.25, 2.0),
    filename: Optional[str] = None,
    boundary_band_width: Optional[float] = None,
):
    """Build a 2D mesh with shoreline, inflow, and grounding-line markers.

    When ``boundary_band_width`` is set, triangles whose centroids lie within
    that distance from the shoreline are locally refined after the initial
    meshing step.  Existing physical groups remain untouched.
    """

    gmsh.initialize()
    model = gmsh.model
    model.add("2D")

    if adapt:
        sizes = (
            adaptive_scales[0] * element_size,
            element_size,
            adaptive_scales[1] * element_size,
        )
    else:
        sizes = (element_size,) * 3

    category_array = category_data.ReadAsArray()
    category_trans = category_data.GetGeoTransform()

    coords = outline
    intersect_points = intersection
    grounding_points = grounding_line

    outline_len = len(coords)
    p1 = intersect_points[0]
    p2 = intersect_points[-1]

    tag1 = find_closest_point(p1[0], p1[1], coords) + 1
    tag2 = find_closest_point(p2[0], p2[1], coords) + 1

    first_point = coords[0]

    lines = []
    inflow_lines = []
    grounding_lines = []

    point = model.geo.addPoint(first_point[0], first_point[1], 0, sizes[1])
    i = 1
    for coord in coords[1:]:
        if point_in_set(coord, grounding_points):
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[0])
        elif point + 1 > tag1 - 1 and point < tag2:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[1])
        else:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[2])

        line = model.geo.addLine(point - 1, point)
        lines.append(line)

        if (
            check_if_inflow(coords[i - 1], category_array, category_trans)
            and check_if_inflow(coords[i], category_array, category_trans)
        ):
            inflow_lines.append((1, line))
        elif (
            point_in_set(coords[i - 1], grounding_points)
            and point_in_set(coords[i], grounding_points)
        ):
            grounding_lines.append((1, line))
        elif (
            (point_in_set(coords[i], grounding_points)
             or point_in_set(coords[i - 1], grounding_points))
            and coords[i][0] > coords[i - 1][0]
        ):
            grounding_lines.append((1, line))
        i += 1

    line = model.geo.addLine(point, 1)
    lines.append(line)

    if (
        check_if_inflow(first_point, category_array, category_trans)
        and check_if_inflow(coords[-1], category_array, category_trans)
    ):
        inflow_lines.append((1, line))

    first_intersect_point = intersect_points[0]
    point = model.geo.addPoint(first_intersect_point[0], first_intersect_point[1], 0, sizes[1])
    line = model.geo.addLine(tag1, point)
    lines.append(line)
    for point in intersect_points[1:]:
        point = model.geo.addPoint(point[0], point[1], 0, sizes[1])
        line = model.geo.addLine(point - 1, point)
        lines.append(line)
    line = model.geo.addLine(point, tag2)
    lines.append(line)

    first_part = lines[: tag1 - 1]
    intersection_lines = lines[outline_len:]
    second_part = lines[tag2 - 1 : outline_len]
    loop = model.geo.addCurveLoop(first_part + intersection_lines + second_part)
    surface1 = model.geo.addPlaneSurface([loop])

    outer_part = lines[tag1 - 1 : tag2 - 1]
    rev_intersection = [-i for i in intersection_lines[::-1]]
    loop = model.geo.addCurveLoop(outer_part + rev_intersection)
    surface2 = model.geo.addPlaneSurface([loop])

    shorelines = get_shorelines(lines, intersection_lines, inflow_lines, grounding_lines)

    boundary_curve_tags = set(lines)
    boundary_curve_tags.update(tag for (_, tag) in shorelines)
    boundary_curve_tags.update(tag for (_, tag) in inflow_lines)
    boundary_curve_tags.update(tag for (_, tag) in grounding_lines)
    boundary_curve_tags.update(intersection_lines)

    model.geo.synchronize()

    # if boundary_curve_tags:
    #     field_api = gmsh.model.mesh.field
    #     distance_field = field_api.add("Distance")
    #     field_api.setNumbers(distance_field, "CurvesList", sorted(boundary_curve_tags))
    #     field_api.setNumber(distance_field, "Sampling", 50)

    #     boundary_field = field_api.add("Threshold")
    #     field_api.setNumber(boundary_field, "IField", distance_field)
    #     field_api.setNumber(boundary_field, "LcMin", sizes[0])
    #     field_api.setNumber(boundary_field, "LcMax", sizes[2])
    #     field_api.setNumber(boundary_field, "DistMin", 0.01)
    #     field_api.setNumber(boundary_field, "DistMax", 10.0)
    #     field_api.setAsBackgroundMesh(boundary_field)

    shoreline_tags = [tag for (_, tag) in shorelines]
    inflow_line_tags = [tag for (_, tag) in inflow_lines]
    grounding_line_tags = [tag for (_, tag) in grounding_lines]

    if shoreline_tags:
        model.geo.addPhysicalGroup(1, shoreline_tags, tag=BATHYMETRY_TAG, name="Shoreline")
    if inflow_line_tags:
        model.geo.addPhysicalGroup(1, inflow_line_tags, tag=INFLOW_LINE_TAG, name="Inflow lines")
    if grounding_line_tags:
        model.geo.addPhysicalGroup(1, grounding_line_tags, tag=GROUNDING_LINE_TAG, name="Grounding line")

    model.geo.synchronize()
    model.mesh.generate(2)

    band_triangles: Dict[int, Set[int]] = {}
    if boundary_band_width and boundary_band_width > 0:
        outline_tuple = tuple((float(x), float(y)) for (x, y) in coords)
        band_triangles = tag_boundary_band_elements(outline_tuple, boundary_band_width)

    if band_triangles:
        all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
        tag_list = all_node_tags.tolist() if hasattr(all_node_tags, "tolist") else list(all_node_tags)
        coord_list = all_node_coords.tolist() if hasattr(all_node_coords, "tolist") else list(all_node_coords)
        coord_lookup = {}
        for idx, tag in enumerate(tag_list):
            coord_lookup[int(tag)] = (
                float(coord_list[3 * idx]),
                float(coord_list[3 * idx + 1]),
                float(coord_list[3 * idx + 2]),
            )

        triangle_type = gmsh.model.mesh.getElementType("Triangle", 1)
        line_type = gmsh.model.mesh.getElementType("Line", 1)
        next_node_tag = gmsh.model.mesh.getMaxNodeTag() + 1
        next_element_tag = gmsh.model.mesh.getMaxElementTag() + 1

        surface_triangles = defaultdict(list)
        edge_to_surfaces = defaultdict(list)

        for _, surface_tag in gmsh.model.getEntities(2):
            elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, surface_tag)
            for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
                tags = elem_tags.tolist() if hasattr(elem_tags, "tolist") else list(elem_tags)
                nodes = elem_nodes.tolist() if hasattr(elem_nodes, "tolist") else list(elem_nodes)
                if not tags:
                    continue
                _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
                if num_nodes != 3:
                    continue
                for idx, elem_tag in enumerate(tags):
                    start = idx * num_nodes
                    node_ids = [int(n) for n in nodes[start:start + num_nodes]]
                    surface_triangles[surface_tag].append({
                        "tag": int(elem_tag),
                        "nodes": node_ids,
                    })
                    edges = (
                        (node_ids[0], node_ids[1]),
                        (node_ids[1], node_ids[2]),
                        (node_ids[2], node_ids[0]),
                    )
                    for edge in edges:
                        edge_key = tuple(sorted(edge))
                        edge_to_surfaces[edge_key].append((surface_tag, int(elem_tag)))

        edge_to_curves = defaultdict(list)
        for _, curve_tag in gmsh.model.getEntities(1):
            elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(1, curve_tag)
            for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
                tags = elem_tags.tolist() if hasattr(elem_tags, "tolist") else list(elem_tags)
                nodes = elem_nodes.tolist() if hasattr(elem_nodes, "tolist") else list(elem_nodes)
                if not tags:
                    continue
                _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
                if num_nodes != 2:
                    continue
                for idx, elem_tag in enumerate(tags):
                    start = idx * num_nodes
                    n1, n2 = [int(n) for n in nodes[start:start + num_nodes]]
                    edge_key = tuple(sorted((n1, n2)))
                    edge_to_curves[edge_key].append((curve_tag, int(elem_tag), [n1, n2]))

        edges_to_split = {}
        for surface_tag, triangle_set in band_triangles.items():
            for tri in surface_triangles.get(surface_tag, []):
                if tri["tag"] not in triangle_set:
                    continue
                nodes = tri["nodes"]
                for edge in (
                    (nodes[0], nodes[1]),
                    (nodes[1], nodes[2]),
                    (nodes[2], nodes[0]),
                ):
                    edge_key = tuple(sorted(edge))
                    adjacency = edge_to_surfaces.get(edge_key, [])
                    if not adjacency:
                        continue
                    targeted_adjacency = [
                        (surf, elem_tag)
                        for (surf, elem_tag) in adjacency
                        if elem_tag in band_triangles.get(surf, set())
                    ]
                    if len(adjacency) == 1:
                        if targeted_adjacency:
                            edges_to_split[edge_key] = targeted_adjacency
                    elif len(adjacency) == len(targeted_adjacency) and targeted_adjacency:
                        edges_to_split[edge_key] = targeted_adjacency

        entity_nodes_to_add: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        entity_node_sets: Dict[Tuple[int, int], Set[int]] = {}

        def schedule_node(dim: int, entity_tag: int, node_tag: int, coords: Tuple[float, float, float]) -> None:
            key = (dim, entity_tag)
            if key not in entity_nodes_to_add:
                entity_nodes_to_add[key] = {"tags": [], "coords": []}
                entity_node_sets[key] = set()
            if node_tag not in entity_node_sets[key]:
                entity_node_sets[key].add(node_tag)
                entity_nodes_to_add[key]["tags"].append(int(node_tag))
                entity_nodes_to_add[key]["coords"].extend(coords)

        edge_midpoints: Dict[Tuple[int, int], int] = {}
        curve_updates: Dict[int, List[Tuple[int, List[int], int]]] = defaultdict(list)

        for edge_key, adjacency in edges_to_split.items():
            n1, n2 = edge_key
            p1 = coord_lookup[n1]
            p2 = coord_lookup[n2]
            midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)

            midpoint_tag = next_node_tag
            next_node_tag += 1
            coord_lookup[midpoint_tag] = midpoint
            edge_midpoints[edge_key] = midpoint_tag

            surfaces = {surf for surf, _ in adjacency}
            for surface_tag in surfaces:
                schedule_node(2, surface_tag, midpoint_tag, midpoint)

            if len(adjacency) == 1:
                for curve_tag, elem_tag, node_seq in edge_to_curves.get(edge_key, []):
                    schedule_node(1, curve_tag, midpoint_tag, midpoint)
                    curve_updates[curve_tag].append((elem_tag, node_seq, midpoint_tag))

        triangle_centroids: Dict[Tuple[int, int], int] = {}
        for surface_tag, triangle_set in band_triangles.items():
            for tri in surface_triangles.get(surface_tag, []):
                if tri["tag"] not in triangle_set:
                    continue
                n1, n2, n3 = tri["nodes"]
                p1 = coord_lookup[n1]
                p2 = coord_lookup[n2]
                p3 = coord_lookup[n3]
                centroid = (
                    (p1[0] + p2[0] + p3[0]) / 3.0,
                    (p1[1] + p2[1] + p3[1]) / 3.0,
                    (p1[2] + p2[2] + p3[2]) / 3.0,
                )
                centroid_tag = next_node_tag
                next_node_tag += 1
                coord_lookup[centroid_tag] = centroid
                triangle_centroids[(surface_tag, tri["tag"])] = centroid_tag
                schedule_node(2, surface_tag, centroid_tag, centroid)

        for (dim, entity_tag), data in entity_nodes_to_add.items():
            gmsh.model.mesh.addNodes(dim, entity_tag, data["tags"], data["coords"])

        curve_removals: Dict[int, Set[int]] = defaultdict(set)
        curve_new_segments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for curve_tag, updates in curve_updates.items():
            for elem_tag, node_seq, midpoint_tag in updates:
                curve_removals[curve_tag].add(elem_tag)
                start, end = node_seq
                curve_new_segments[curve_tag].append((start, midpoint_tag))
                curve_new_segments[curve_tag].append((midpoint_tag, end))

        for curve_tag, removals in curve_removals.items():
            if removals:
                gmsh.model.mesh.removeElements(1, curve_tag, list(removals))
            new_tags = []
            new_nodes = []
            for start, end in curve_new_segments[curve_tag]:
                new_tags.append(next_element_tag)
                next_element_tag += 1
                new_nodes.extend([start, end])
            if new_tags:
                gmsh.model.mesh.addElements(1, curve_tag, [line_type], [new_tags], [new_nodes])

        for surface_tag, triangle_set in band_triangles.items():
            triangles = surface_triangles.get(surface_tag, [])
            if not triangles:
                continue

            removal_tags = [tri["tag"] for tri in triangles if tri["tag"] in triangle_set]
            if removal_tags:
                gmsh.model.mesh.removeElements(2, surface_tag, removal_tags)

            new_triangle_tags = []
            new_triangle_nodes = []

            for tri in triangles:
                if tri["tag"] not in triangle_set:
                    continue
                nodes = tri["nodes"]
                centroid_tag = triangle_centroids[(surface_tag, tri["tag"])]

                boundary_nodes: List[int] = []
                for idx in range(3):
                    current = nodes[idx]
                    if not boundary_nodes or boundary_nodes[-1] != current:
                        boundary_nodes.append(current)
                    next_idx = (idx + 1) % 3
                    edge_key = tuple(sorted((nodes[idx], nodes[next_idx])))
                    midpoint_tag = edge_midpoints.get(edge_key)
                    if midpoint_tag is not None:
                        boundary_nodes.append(midpoint_tag)

                boundary_length = len(boundary_nodes)
                for idx in range(boundary_length):
                    n1 = boundary_nodes[idx]
                    n2 = boundary_nodes[(idx + 1) % boundary_length]
                    if n1 == n2:
                        continue
                    new_triangle_tags.append(next_element_tag)
                    next_element_tag += 1
                    new_triangle_nodes.extend([n1, n2, centroid_tag])

            if new_triangle_tags:
                gmsh.model.mesh.addElements(
                    2,
                    surface_tag,
                    [triangle_type],
                    [new_triangle_tags],
                    [new_triangle_nodes],
                )

        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()

    if not gmsh.model.getPhysicalGroups(2):
        surface_entities = [tag for (_, tag) in gmsh.model.getEntities(2)]
        if surface_entities:
            try:
                group_tag = gmsh.model.addPhysicalGroup(2, surface_entities, tag=DEFAULT_SURFACE_TAG)
            except RuntimeError:
                group_tag = gmsh.model.addPhysicalGroup(2, surface_entities)
            gmsh.model.setPhysicalName(2, group_tag, "AllSurfaces")

    gmsh.option.setNumber("Mesh.SaveAll", 1)
    mesh_filename = filename or f"2D_mesh_{int(element_size)}m.msh"
    gmsh.write(mesh_filename)

    gmsh.finalize()
    return mesh_filename


class Scenario(NamedTuple):
    element_size: float
    adapt: bool = False
    adaptive_scales: Tuple[float, float] = (0.25, 2.0)
    filename: Optional[str] = None
    boundary_band_width: Optional[float] = None


def main():
    elevation = readraster('../data/gis_data/bathymetry.tif')
    elevation_band = elevation.GetRasterBand(1)

    spatial_ref = osr.SpatialReference(elevation.GetProjection())
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shoreline_dir = 'new_data/shoreline_py'
    if Path(shoreline_dir).exists():
        shutil.rmtree(shoreline_dir)
    ogr_ds = driver.CreateDataSource(shoreline_dir)
    shoreline_layer = ogr_ds.CreateLayer('shoreline', spatial_ref)

    gdal.ContourGenerateEx(
        elevation_band,
        shoreline_layer,
        options=['LEVEL_INTERVAL = 100000', 'FIXED_LEVELS=0', 'NODATA=-9999']
    )

    gdal.Rasterize(elevation, shoreline_layer.GetDataset(), burnValues=0)
    bathymetry = gdal_calc.Calc(
        calc=["-9999*(B>0) + A*(A<=0)"],
        A=elevation,
        B=elevation,
        outfile="new_data/bathymetry.tif",
        NoDataValue=-9999,
        overwrite=True,
    )

    shoreline_res = 60
    gdal.Footprint(
        'new_data/shorelines.geojson',
        bathymetry,
        format="GeoJSON",
        maxPoints=shoreline_res,
        srcNodata=-9999,
    )
    shoreline_wkt = get_boundary_string('new_data/shorelines.geojson')
    shoreline = get_main_boundary(shoreline_wkt)

    categories = readraster('../data/gis_data/categories.tif')

    seaice = gdal_calc.Calc(
        calc=["(A==3)"],
        A=categories,
        outfile="new_data/seaice.tif",
        NoDataValue=0,
        overwrite=True,
    )

    landice = gdal_calc.Calc(
        calc=["(A==2)"],
        A=categories,
        outfile="new_data/ice.tif",
        NoDataValue=0,
        overwrite=True,
    )

    ice_res = 0.8 * shoreline_res
    gdal.Footprint(
        'new_data/seaice.geojson',
        seaice,
        format="GeoJSON",
        maxPoints=ice_res,
        srcNodata=0,
    )
    seaice_lines = get_boundary_string('new_data/seaice.geojson')
    seaice_line = get_main_boundary(seaice_lines)

    gdal.Footprint(
        'new_data/landice.geojson',
        landice,
        format="GeoJSON",
        maxPoints='unlimited',
        srcNodata=0,
    )
    landice_lines = get_boundary_string('new_data/landice.geojson')
    landice_line = get_main_boundary(landice_lines, no=1)

    grounding_line = get_grounding_line(seaice_line, landice_line)
    outline = add_grounding_line(shoreline, grounding_line)

    water = gdal_calc.Calc(
        calc=["(A==0)"],
        A=categories,
        outfile="new_data/water.tif",
        NoDataValue=0,
        overwrite=True,
    )
    gdal.Footprint(
        'new_data/water.geojson',
        water,
        format="GeoJSON",
        maxPoints='unlimited',
        srcNodata=0,
    )
    water_lines = get_boundary_string('new_data/water.geojson')
    water_line = get_main_boundary(water_lines)
    intersection = get_water_ice_intersect(water_line, seaice_line)

    shoreline_layer = None
    ogr_ds = None
    shutil.rmtree(shoreline_dir)

    scenarios = [
        Scenario(560, True, boundary_band_width=500.0),
    ]

    for scenario in scenarios:
        mesh_path = generate_2D_mesh(
            outline,
            intersection,
            grounding_line,
            categories,
            element_size=scenario.element_size,
            adapt=scenario.adapt,
            adaptive_scales=scenario.adaptive_scales,
            filename=scenario.filename,
            boundary_band_width=scenario.boundary_band_width,
        )
        print(f"Wrote {mesh_path}")


if __name__ == "__main__":
    main()
