import matplotlib.pyplot as plt
import math
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo_utils import gdal_calc
import shutil
import gmsh
plt.style.use('seaborn-v0_8')
gdal.UseExceptions()

def readraster(filename):
    raster = gdal.Open(filename)
    driver = raster.GetDriver()
    copy_name_list = filename.split('.')
    main_name = (copy_name_list[-2].split('/'))[-1]
    copy_name = 'new_data/'+main_name+'_copy.'+copy_name_list[-1]
    raster_copy = driver.CreateCopy(copy_name, raster)
    return raster_copy

def readvector(filename):
    vector = gdal.OpenEx(filename)
    return vector

def get_boundary_string(filename):
    border = readvector(filename)
    layer = border.GetLayer()
    feature = layer.GetFeature(0)
    geometry = feature.GetGeometryRef()
    boundary = str(geometry.GetBoundary())
    return boundary
    
def get_main_boundary(boundary, no=0):
    boundary = boundary.replace('MULTILINESTRING ','')
    curves = boundary.split(')')
    if no == 0:
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
    
    px, py = point
    for (x, y) in line:
        if abs(px - x) < tol and abs(py - y) < tol:
            return True
    return False

def get_grounding_line(seaice, landice):
    grounding_line = []
    
    for coord in seaice:
        if point_in_set(coord, landice):
            grounding_line.append(coord)

    return grounding_line

def add_grounding_line(shoreline, grounding_line):
    
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
    grounding_line = []
    
    for coord in seaice:
        if point_in_set(coord, water):
            grounding_line.append(coord)

    return grounding_line[:-1]

def plot_main_geoline(shoreline):
    
    X = []
    Y = []
    for coord in shoreline:
        X.append(coord[0])
        Y.append(coord[1])
    
    plt.figure(figsize=(5, 5))
    plt.scatter(X, Y)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Meters')
    ax.set_ylabel('Meters')
    plt.draw()
    
def plot_full_outline(lines):
    
    plt.figure(figsize=(6,8))
    
    for line in lines.keys():
        X = []
        Y = []
        for coord in lines[line]:
            X.append(coord[0])
            Y.append(coord[1])
        plt.plot(X, Y, marker = 'o', markersize = 5, linestyle = '--', label=line)
        
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x$ (m)', fontsize = 12)
    ax.set_ylabel('$y$ (m)', fontsize = 12)
    plt.title("Outline Boundary Points",
              fontsize = 14)
    plt.legend(loc="best", frameon = True, facecolor = "white", edgecolor = "black", 
               fontsize=8)
    plt.draw()
        

def check_if_inflow(point, category_array, category_trans):
    
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
    
    shorelines = []
    inflow_line_tags = [tag for (dim, tag) in inflow_lines]
    grounding_line_tags = [tag for (dim, tag) in grounding_lines]
    dont_include = grounding_line_tags + inflow_line_tags + intersect_tags
    for line in line_tags:
        if line not in dont_include:
            shorelines.append((1, line))
            
    return shorelines

def sanitize(val, eps):
    if val == -9999 or val > 0:
        return 0
    else:
        return val

def bilinear_interpolate(data_array, i, j, dx, dy, eps, 
                         missing_value=-9999, sanitize=False):
    v00 = data_array[j    , i    ]
    v10 = data_array[j    , i + 1]
    v01 = data_array[j + 1, i    ]
    v11 = data_array[j + 1, i + 1]
    
    if sanitize:
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
    
    if highres:
        try:
            top_left_x = highres_trans[0]
            x_res = highres_trans[1]
            top_left_y  = highres_trans[3]
            y_res = highres_trans[5]
            
            x_index = round((x - top_left_x)/x_res)
            y_index = round((y - top_left_y)/y_res)
            
            z = highres_array[y_index, x_index]
            
            if z > eps:
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
    
    top_left_x = thic_trans[0]
    x_res = thic_trans[1]
    top_left_y  = thic_trans[3]
    y_res = thic_trans[5]
    
    # Convert to pixel coordinates (float)
    px = (x - top_left_x) / x_res
    py = (y - top_left_y) / y_res
    
    if interpolate:
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
                     m, eps, num_of_layers, adapt, adaptive_scales = (1/4, 2)):
    
    gmsh.initialize()
    
    model = gmsh.model
    model.add("2D")
    
    if adapt:
        sizes = (adaptive_scales[0]*m, m, adaptive_scales[1]*m)
    else:
        sizes = (m,m,m)
    
    category_array = category_data.ReadAsArray()
    category_trans = category_data.GetGeoTransform()
    
    coords = outline
    intersect_points = intersect
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
        elif point+1 > tag1-1 and point < tag2:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[1])
        else:
            point = model.geo.addPoint(coord[0], coord[1], 0, sizes[2])
        
        line = model.geo.addLine(point - 1, point)
        lines.append(line)
        
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
        
    line = model.geo.addLine(point, 1)
    lines.append(line)
    
    if (check_if_inflow(first_point, category_array, category_trans)
        and check_if_inflow(coords[-1], category_array, category_trans)):
        inflow_lines.append((1, line))
    first_intersect_point = intersect_points[0]
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
    
    model.geo.synchronize()
    model.mesh.generate(2)
        
    ice = [surface2]
    inflow = []
    bath = []
    mid_layers = []
        
    ice_line_ents = set(grounding_lines)
    bath_line_ents = set(shorelines)
    inflow_line_ents = set(inflow_lines)
    intersect_line_ents = set(intersect_lines)
        
    extrude_from = surface_dim_tags
    
    mid_group_1 = []
    mid_group_2 = []
    
    all_inflow_lines = inflow_line_ents
    for layer in range(2, num_of_layers+1):
        
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
            if entity[0] == 3:
                model.geo.remove([entity], recursive=False)
            else:
                bound = model.getBoundary([entity], oriented=False)
                bound_set = set(bound)
                
                if len(bound_set.intersection(ice_line_ents)) > 0:
                    ice.append(entity[1])
                    next_ice = next_ice + bound
                elif len(bound_set.intersection(bath_line_ents)) > 0:
                    bath.append(entity[1])
                    next_bath = next_bath + bound
                elif len(bound_set.intersection(inflow_line_ents)) > 0:
                    inflow.append(entity[1])
                    next_inflow = next_inflow + bound
                elif len(bound_set.intersection(intersect_line_ents)) > 0:
                    model.geo.remove([entity], recursive=True)
                    next_intersect = next_intersect + bound
                else:
                    if layer == num_of_layers:
                        bath.append(entity[1])
                    else:
                        mid_layers.append(entity[1])
                    next_extrusion.append(entity)
        if layer != num_of_layers:
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
    
    model.geo.addPhysicalGroup(2, [surface1], tag=1, name="Water Surface")
    model.geo.addPhysicalGroup(2, ice, tag=2, name="Ice")
    model.geo.addPhysicalGroup(2, bath, tag=3, name="Bathymetry")
    model.geo.addPhysicalGroup(2, inflow, tag=4, name="Inflow")
    model.geo.addPhysicalGroup(1, all_inflow_lines, tag=12, name="Inflow lines")

    if num_of_layers > 2:
        model.geo.addPhysicalGroup(2, mid_layers, tag=6, name="Mid")
  
    model.geo.synchronize()
    model.mesh.generate(2)
    
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
    
    filename = f"2D_{num_of_layers}_layer_mesh.msh"
    
    gmsh.write(filename)
        
    gmsh.finalize()
    
    mesh2D = (filename, xy_shoreline, xy_grounding, mid_group_1, mid_group_2)
    
    return mesh2D

def generate_mesh_mult(outline, intersect, grounding_line, m, eps, category_data, 
                       bathymetry_data, highres_data, thickness_data, surface_pos_data, 
                       scale = 1, num_of_layers = 2, adapt = False, adaptive_scales = (1/4, 2),
                       optimize = False, stack = 25, interpolate = True):
    
    if num_of_layers > 2:
        eps = eps*(num_of_layers - 1)
    
    mesh2D = generate_2D_mesh(outline, intersect, grounding_line, 
                              category_data, m, eps, num_of_layers, adapt)

    filename, xy_shoreline, xy_grounding, mid_group_1, mid_group_2 = mesh2D
    
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
    
    gmsh.initialize()
    
    model = gmsh.model
    model.add("3D")
    
    gmsh.merge(filename)

    model.geo.synchronize()
    
    bathymetry_array = bathymetry_data.ReadAsArray()
    bathymetry_trans = bathymetry_data.GetGeoTransform()
    highres_array = highres_data.ReadAsArray()
    highres_trans = highres_data.GetGeoTransform()
    thickness_array = thickness_data.ReadAsArray()
    thickness_trans = thickness_data.GetGeoTransform()
    surface_pos_array = surface_pos_data.ReadAsArray()
    
    bathymetry_node_tags, bathymetry_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 3)
    surface_node_tags, surface_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)
    ice_node_tags, ice_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 2)
    inflow_node_tags, inflow_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 4)
    
    bathymetry_node_coords = bathymetry_node_coords.reshape((bathymetry_node_tags.size, 3))
    surface_node_coords = surface_node_coords.reshape((surface_node_tags.size, 3))
    ice_node_coords = ice_node_coords.reshape((ice_node_tags.size, 3))
    
    if stack > 0:
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
                depth = get_bathymetric_depth(x, y, bathymetry_trans, bathymetry_array, 
                                              highres_trans, highres_array, eps, 
                                              highres = True, interpolate = interpolate)
                z = depth - eps
                gmsh.model.mesh.setNode(tag, [x, y, scale*z], [])
            elif tag not in bathymetry_node_tags:
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
        
        inflow_ents = gmsh.model.getEntitiesForPhysicalGroup(2, 4)
        model.geo.removePhysicalGroups([(2,4)])
        model.geo.removePhysicalGroups([(1,12)])
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
            for surface in inflow_stack_points.keys():
                model.mesh.embed(0, inflow_stack_points[surface], 2, surface)

        model.geo.addPhysicalGroup(2, new_inflow, tag=4, name="Inflow")
        
        gmsh.model.geo.synchronize()
        model.mesh.generate(2)
        gmsh.model.geo.synchronize()
        
        surf_tags = []
        for i in range(1, 5):
            surf_tags = surf_tags + [tag for tag in model.getEntitiesForPhysicalGroup(2, i)]

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
        model.geo.removePhysicalGroups([(1,12)])
        gmsh.model.geo.synchronize()
        
        if scale > 1:
            inflow_ents = gmsh.model.getEntitiesForPhysicalGroup(2, 4)
            model.geo.removePhysicalGroups([(2,4)])
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

            model.geo.addPhysicalGroup(2, new_inflow, tag=4, name="Inflow")
            
            gmsh.model.geo.synchronize()
            model.mesh.generate(2)
            gmsh.model.geo.synchronize()
        
        surf_tags = []
        for i in range(1, 5):
            surf_tags = surf_tags + [tag for tag in model.getEntitiesForPhysicalGroup(2, i)]

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
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-13)
        if scale > 1:
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-13)

    model.mesh.generate(3)
        
    model.mesh.reclassifyNodes()
    model.geo.synchronize()
    
    surface_node_tags, surface_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)
    ice_node_tags, ice_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 2)
    bath_node_tags, bath_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 3)
    water_node_tags, water_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(3, 5)
    
    ice_node_coords = ice_node_coords.reshape((ice_node_tags.size, 3))
    water_node_coords = water_node_coords.reshape((water_node_tags.size, 3))
        
    xmin = min(water_node_coords[:,0])
    ymin = min(water_node_coords[:,1])
    
    i = 0
    print(meshname)
    
    for tag in water_node_tags:
        x = water_node_coords[i][0]
        y = water_node_coords[i][1]
        z = water_node_coords[i][2]
        if ((tag not in surface_node_tags) and 
            (not point_in_set((x,y), xy_shoreline))):
            gmsh.model.mesh.setNode(tag, [x - xmin, y - ymin, z/scale], [])
        else:
            gmsh.model.mesh.setNode(tag, [x - xmin, y - ymin, z], [])
        i += 1
    
    model.geo.synchronize()
    
    if optimize:
        gmsh.model.mesh.optimize(method="")
        gmsh.model.mesh.optimize(method="Netgen")
        model.geo.synchronize()
        water_node_tags, water_node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(3, 5)
    
    filename = f"{folder}{meshname}.msh"
    gmsh.write(filename)
    gmsh.finalize()
    dof = len(water_node_tags)
    return dof, meshname

def main():
    elevation = readraster('../data/gis_data/elevation.tif')
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
    
    #Unstructured:
    unstructured = [(400, 1, 2, False, False, 0), (300, 1, 2, False, False, 0), 
                    (250, 1, 2, False, False, 0), (200, 1, 2, False, False, 0), 
                    (160, 1, 2, False, False, 0)]
    #Scaled DOF-test 200m:
    
    scaled_200_dofs = [(200, 2, 2, False, False, 0), (200, 3, 2, False, False, 0), 
                       (200, 4, 2, False, False, 0), (200, 5, 2, False, False, 0), 
                       (200, 6, 2, False, False, 0), (200, 7, 2, False, False, 0),
                       (200, 8, 2, False, False, 0), (200, 9, 2, False, False, 0),
                       (200, 10, 2, False, False, 0), (200, 11, 2, False, False, 0),
                       (200, 12, 2, False, False, 0), (200, 13, 2, False, False, 0),
                       (200, 14, 2, False, False, 0), (200, 15, 2, False, False, 0)]

    #Scaled DOF-test 220m:
    
    scaled_220_dofs = [(220, 2, 2, False, False, 0), (220, 3, 2, False, False, 0), 
                       (220, 4, 2, False, False, 0), (220, 5, 2, False, False, 0), 
                       (220, 6, 2, False, False, 0), (220, 7, 2, False, False, 0),
                       (220, 8, 2, False, False, 0), (220, 9, 2, False, False, 0),
                       (220, 10, 2, False, False, 0), (220, 11, 2, False, False, 0),
                       (220, 12, 2, False, False, 0), (220, 13, 2, False, False, 0),
                       (220, 14, 2, False, False, 0), (220, 15, 2, False, False, 0)]
    
    #Scaled:
    
    scaled = [(220, 6, 2, False, False, 0), (220, 6, 2, False, True, 0), 
              (200, 11, 2, False, False, 0), (200, 6, 2, False, True, 0)]
    
    #Layered:
    
    layered = [(295, 1, 9, False, False, 0), (340, 1, 12, False, False, 0), 
               (410, 1, 16, False, False, 0), (450, 1, 20, False, False, 0)]
    
    #Stacked:
     
    stacked = [(730, 1, 2, False, False, 10), (560, 1, 2, False, False, 15), 
               (480, 1, 2, False, False, 20), (450, 1, 2, False, False, 25), 
               (310, 1, 2, False, False, 50), (255, 1, 2, False, False, 75)]
    
    #Adaptive:
    adaptive = [(560, 1, 2, True, False, 10), (470, 1, 2, True, False, 15), 
                (400, 1, 2, True, False, 20), (460, 1, 20, True, False, 0),
                (190, 6, 2, True, False, 0)]
    
    #Combo test:
    combo_test = [(295, 11, 5, False, False, 0), (410, 39, 13, False, False, 0)]     
    
    params = combo_test
    
    for m, scale, num_of_layers, adapt, opt, stack in params:
        dof, meshname = generate_mesh_mult(outline, intersect, grounding_line, 
                                 m, -1, categories, full_bathymetry, highres, 
                                 thickness_data, surface_pos_data, scale = scale, 
                                 num_of_layers = num_of_layers, adapt = adapt, 
                                 adaptive_scales = (1/4, 2), optimize = opt, 
                                 stack = stack, interpolate = True)
        dofs.append((meshname, dof))
    
    for meshname, dof in dofs:
        print(f"Mesh Name: {meshname}\t\tDOFS: {dof}")
    
    

if __name__ == "__main__":
    main()