# import matplotlib.pyplot as plt
import math
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

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
BOUNDARY_BAND_TAG = 7
INTERIOR_SURFACE_TAG = 8

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
                               band_width: float) -> None:
    """Create two discrete surfaces based on distance from shoreline."""
    if not band_width or band_width <= 0:
        return

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coord_lookup = {}
    for idx, tag in enumerate(node_tags):
        coord_lookup[int(tag)] = (
            float(node_coords[3 * idx]),
            float(node_coords[3 * idx + 1])
        )

    boundary_elements = {}
    interior_elements = {}
    removal_map = {}

    for _, surface_tag in gmsh.model.getEntities(2):
        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, surface_tag)
        for elem_type, elem_tags, elem_node_tags in zip(elem_types,
                                                        elem_tags_list,
                                                        elem_node_tags_list):
            if elem_tags.size == 0:
                continue
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
            tags = elem_tags.tolist()
            nodes = elem_node_tags.tolist()
            for idx, elem_tag in enumerate(tags):
                node_ids = nodes[idx * num_nodes:(idx + 1) * num_nodes]
                cx = sum(coord_lookup[int(n)][0] for n in node_ids) / num_nodes
                cy = sum(coord_lookup[int(n)][1] for n in node_ids) / num_nodes
                distance = _distance_to_outline(cx, cy, outline)
                target = boundary_elements if distance <= band_width else interior_elements
                if elem_type not in target:
                    target[elem_type] = {"tags": [], "nodes": []}
                target[elem_type]["tags"].append(int(elem_tag))
                target[elem_type]["nodes"].append([int(n) for n in node_ids])

                removal_map.setdefault(surface_tag, []).append(int(elem_tag))

    if not boundary_elements and not interior_elements:
        return

    for surface_tag, elem_tags in removal_map.items():
        if elem_tags:
            gmsh.model.mesh.removeElements(2, surface_tag, elem_tags)

    def _create_surface(data_map, phys_tag, name):
        if not data_map:
            return None
        entity = gmsh.model.addDiscreteEntity(2)
        elem_types = sorted(data_map.keys())
        elem_tags = [data_map[et]["tags"] for et in elem_types]
        elem_nodes = [
            [node for nodes in data_map[et]["nodes"] for node in nodes]
            for et in elem_types
        ]
        gmsh.model.mesh.addElements(2, entity, elem_types, elem_tags, elem_nodes)
        gmsh.model.addPhysicalGroup(2, [entity], tag=phys_tag)
        gmsh.model.setPhysicalName(2, phys_tag, name)
        return entity

    _create_surface(boundary_elements, BOUNDARY_BAND_TAG, "BoundaryBand")
    _create_surface(interior_elements, INTERIOR_SURFACE_TAG, "InteriorSurface")

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

    When ``boundary_band_width`` is set, the generated Gmsh file is rewritten so
    that triangles within that distance from the shoreline are tagged with a
    dedicated "BoundaryBand" physical surface, while the remaining triangles
    are collected in "InteriorSurface".
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

    if boundary_band_width and boundary_band_width > 0:
        outline_tuple = tuple((float(x), float(y)) for (x, y) in coords)
        tag_boundary_band_elements(outline_tuple, boundary_band_width)

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
