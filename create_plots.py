import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import meshio
import matplotlib as mpl
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import re
plt.style.use('seaborn-v0_8')

def get_outline(filename):
    outline_x = []
    outline_y = []
    with open(filename, 'r') as f:
        full_outline = f.read()
        full_outline = full_outline[1:-1]
        points = full_outline.split('), ')
        for point in points[:-1]:
            point = point[1:]
            x, y = point.split(', ')
            outline_x.append(float(x))
            outline_y.append(float(y))
        point = points[-1]
        point = point[1:-1]
        x, y = point.split(', ')
        outline_x.append(float(x))
        outline_y.append(float(y))
    
    xmin = min(outline_x)
    ymin = min(outline_y)
    
    outline_x = [x - xmin for x in outline_x]
    outline_y = [y - ymin for y in outline_y]
    
    return outline_x, outline_y

def extract_surface_geometry_points(filename, target_marker=None):
    msh = meshio.read(filename)
    
    all_points = msh.points 
    surface_points = []
    
    triangle_cells = []
    for cell in msh.cells:
        if cell.type == 'triangle':
            if len(triangle_cells) == 0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])

    markers = msh.cell_data_dict["gmsh:physical"]['triangle']
    

    for tri, marker in zip(triangle_cells, markers):
        if (target_marker is None) or (marker in target_marker):
            for vertex_index in tri:
                surface_points.append(all_points[vertex_index])

    surface_points = np.unique(np.array(surface_points), axis=0)
    return surface_points 

def format_latex_name(mesh_name):
    # Adaptive stacked mesh: A_P_10_600 → P_{10/(150,600,1200)}
    match = re.match(r"A_([A-Z]+)_(\d+)_(\d+)", mesh_name)
    if match:
        prefix = match.group(1)
        part1 = match.group(2)
        part2 = int(match.group(3))
        g = part2 / 4
        i = part2
        o = part2 * 2
        if g == int(g):
            g = int(g)
        if i == int(i):
            i = int(i)
        return rf"${prefix}_{{{{{part1}}}/({g},{i},{o})}}$"

    # Optimized scaled mesh: S_6_250_opt → S_{6/250/opt}
    match = re.match(r"S_(\d+)_(\d+)_opt", mesh_name)
    if match:
        part1, part2 = match.group(1), match.group(2)
        return rf"$S_{{{part1}/{part2}/\text{{opt}}}}$"

    match = re.match(r"([A-Z]+)_?(\d+)_?(\d+)?", mesh_name)
    if match:
        prefix = match.group(1)
        part1 = match.group(2)
        part2 = match.group(3)
        if part2:
            return rf"${prefix}_{{{part1}/{part2}}}$"
        else:
            return rf"${prefix}_{{{part1}}}$"

    # Default fallback
    return mesh_name

def plot_relative_error(csv_path, mesh_type, limit_y_scale = None, figsize=(10, 6)):
    # Load the data
    df = pd.read_csv(csv_path)

    # Ensure correct data types
    df["Frequency Scale"] = pd.to_numeric(df["Frequency Scale"], errors="coerce")
    df["Rel. Error (%)"] = pd.to_numeric(df["Rel. Error (%)"], errors="coerce")

    # Get unique meshes
    meshes = df["Mesh"].unique()
    
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = plt.get_cmap('Dark2').colors  # 8 distinct colors

    plt.figure()

    # Plot each mesh
    for i, mesh in enumerate(meshes):
        mesh_df = df[df["Mesh"] == mesh]
        plt.plot(
            mesh_df["Frequency Scale"],
            mesh_df["Rel. Error (%)"],
            linestyle=linestyles[i],
            marker="o",
            color=colors[i],
            markersize = 5,
            label=format_latex_name(mesh)
        )
        
    if limit_y_scale:
        plt.ylim(0, limit_y_scale)
        
    plt.xlabel("Frequency Scale ($k$)", fontsize=12)
    plt.ylabel("Relative $L^2$ Error (%)", fontsize=12)
    if mesh_type != 'high_f':
        plt.title(f"Poisson Error - {mesh_type.capitalize()} Meshes", fontsize=14)
    else:
        plt.title("Poisson Error - High Frequency Test", fontsize=14)
        

    plt.legend(loc="best", frameon = True, facecolor = "white", edgecolor = "black", 
               title="Mesh", fontsize=9)
    plt.show()

def plot_node_count(txt_files, Title = None, figsize=(10,6)):
    
    plt.figure()
    
    linestyles = ['-', '--']
    colors = plt.get_cmap('Dark2').colors  # 8 distinct colors
    i = 0
    for txt_file in txt_files:
        xs = []
        ys = []
        with open(txt_file) as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.split('\t')
                mesh = line_split[0]
                dofs = line_split[-1]
                mesh_split = mesh.split('_')
                dofs_split = dofs.split(' ')
                scale = mesh_split[-2]
                mesh_size = mesh_split[-1]
                dof_count = dofs_split[-1]
                xs.append(int(scale))
                ys.append(int(dof_count))
                
            plt.plot(
                xs,
                ys,
                linestyle=linestyles[i],
                marker='o',
                markersize = 5,
                color=colors[i],
                label=mesh_size
            )
        i += 1
    
    dof_limit_x = [1,15]
    dof_limit_y = [100000, 100000]
    plt.plot(dof_limit_x, dof_limit_y, color='red', linestyle=':')
    plt.legend(loc="best", frameon = True, facecolor = "white", edgecolor = "black", 
               title="Mesh size ($m$)", fontsize=9)
    plt.ylabel("Total Node Count", fontsize=12)
    plt.xlabel("Scaling Factor ($a$)", fontsize=12)
    plt.title("How Vertical Scaling Affects Node Generation", fontsize=14)
        
def parse_gmsh_quality(filename):
    """
    Parse Gmsh scalar view histogram of element quality (clean SP format).
    
    Format expected:
    View "# Elements" {
        SP(<quality>, 0, 0){<count>};
        ...
    }

    Returns:
        qualities (np.ndarray): Element quality values (Γ)
        counts (np.ndarray): Corresponding element counts
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    qualities = []
    counts = []

    pattern = re.compile(r"SP\(([\d\.Ee+-]+),\s*[\d\.Ee+-]+,\s*[\d\.Ee+-]+\)\s*\{\s*([\d\.Ee+-]+)\s*\}")

    for line in lines:
        match = pattern.search(line)
        if match:
            quality = float(match.group(1))
            count = int(float(match.group(2)))  # sometimes it's written as float
            if count != 0:
                qualities.append(quality)
                counts.append(count)

    qualities = np.array(qualities)
    counts = np.array(counts)

    return qualities, counts

def plot_mesh_qualities(filenames, mesh_type, figsize=(10,6)):
    
    all_qualities = []
    all_counts = []
    mesh_names = []
    
    for file in filenames:
        qualities, counts = parse_gmsh_quality(file)
        all_qualities.append(qualities)
        all_counts.append(counts)
        file_split = file.split('/')
        mesh_name = file_split[-1].split('.')[0]
        mesh_names.append(format_latex_name(mesh_name))
    
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = plt.get_cmap('Dark2').colors 
    plt.figure()
    i = 0

    for qualities in all_qualities:
        plt.plot(
            qualities,
            all_counts[i],
            linestyle=linestyles[i],
            color=colors[i],
            label=mesh_names[i]
        )
        i += 1
    
    plt.xlabel("Element Quality ($\gamma$)", fontsize = 12)
    plt.ylabel("Element Count", fontsize = 12)
    plt.title(f"Element Quality Distribution - {mesh_type.capitalize()} Meshes",
              fontsize = 14)
    plt.legend(loc="best", frameon = True, facecolor = "white", edgecolor = "black", 
               title="Mesh", fontsize=9)

def tick_format(x,y):
    if x == 0:
        return 0
    elif x < 0.01:
        return f"{x:.0e}".replace("e-0", "e-")
    elif x < 10:
        return f"{x:.2f}"
    elif x < 100:
        return f"{x:.1f}"
    else:
        return f"{round(x)}"

def interpolate_averaged_pointwise_error(filename, resolution=500, csv_path = None, limit=None):
    
    file_list = filename.split('/')
    error_file = file_list[-1].split('_error_')
    mesh_name = format_latex_name(error_file[0])

    if len(error_file[-1]) == 12:
        k = 1
    elif len(error_file[-1]) == 14:
        k = 105
    else:
        suffix = error_file[-1]
        k = suffix[1:3]
    
    if csv_path:
        df = pd.read_csv(csv_path)
        mf = df.loc[lambda df: df['Mesh'] == error_file[0], :]
        rel_error = mf.loc[lambda df: df['Frequency Scale'] == int(k)]["Rel. Error (%)"].iloc[0]

    # Load mesh and pointwise error data
    mesh = meshio.read(filename)
    points = mesh.points  # shape (n_points, 3)
    data = mesh.point_data
    field_names = list(data.keys())
    error = data[field_names[0]]
    
    # Average error vertically per (x, y)
    xy = np.round(points[:, :2], 5)
    error_map = {}
    for pt, err in zip(xy, error):
        key = tuple(pt)
        error_map.setdefault(key, []).append(err)
    
    xs, ys, zs = [], [], []
    for (x, y), errs in error_map.items():
        xs.append(x)
        ys.append(y)
        zs.append(np.mean(errs))
    
    x = np.array(xs)
    y = np.array(ys)
    values = np.array(zs)

    # Interpolation grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), resolution),
        np.linspace(np.min(y), np.max(y), resolution)
    )
    points_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Create boundary mask using outline
    outline_x, outline_y = get_outline('outline')
    domain_path = Path(np.column_stack((outline_x, outline_y)))
    inside = domain_path.contains_points(points_grid)

    # Interpolate only inside region
    interp_values = griddata((x, y), values, points_grid[inside], method='linear')

    # Fill result with NaNs, write interpolated values inside
    grid_z = np.full(grid_x.shape, np.nan)
    grid_z.ravel()[inside] = interp_values
    
    # Plot
    cmap = 'afmhot'
    cmap = 'nipy_spectral'
    cmap = 'gist_stern'
    
    nipy_spectral = mpl.colormaps['nipy_spectral'].resampled(256)
    newcolors = nipy_spectral(np.linspace(0, 0.95, 256))
    cmap = ListedColormap(newcolors)
    cmap = 'afmhot'
    
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_axis_off()
    pc = plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap=cmap, vmax=limit)

    tick_form = FuncFormatter(lambda x, pos: tick_format(x, pos))
    cb = plt.colorbar(pc, format = tick_form)
    cb.set_label(label='Vertically Averaged Pointwise Error', fontsize=14)

    if csv_path:
        ax.text(0.15*np.nanmax(x), 0.35*np.nanmax(y), 
                f"{mesh_name}\n$k = {k}$\n{rel_error}% Error", dict(size=16))
    else:
        ax.text(0.15*np.nanmax(x), 0.35*np.nanmax(y), 
                f"{mesh_name}\n$k = {k}$", dict(size=16))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    print(np.nanmax(grid_z))
    
def plot_gradient(filename, target_marker=None, resolution=2000, limit = None):
    
    file_list = filename.split('/')
    mesh_file = file_list[-1].split('.')
    mesh_name = format_latex_name(mesh_file[0])
    
    surface_pts = extract_surface_geometry_points(filename, target_marker=target_marker)  # e.g., bathymetry marker
    x, y, z = surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2]
    
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), resolution),
                                 np.linspace(y.min(), y.max(), resolution)
                                 )
    
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    dz_dx, dz_dy = np.gradient(grid_z, np.mean(np.diff(grid_x[0])), np.mean(np.diff(grid_y[:,0])))
    outline_x, outline_y = get_outline('outline')
    hull_path = Path(np.column_stack((outline_x, outline_y)))

    # Flatten grid for domain masking
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside = hull_path.contains_points(grid_points)

    # Mask gradients outside domain
    grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    masked_grad = np.full(grad_mag.shape, np.nan)
    masked_grad.ravel()[inside] = grad_mag.ravel()[inside]
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    cmap = 'cool'
    norm = TwoSlopeNorm(vmin=0., vcenter=0.25, vmax=limit)
    pc = plt.pcolormesh(grid_x, grid_y, masked_grad, shading='auto', cmap=cmap, 
                        norm=norm)#, vmax=limit)
    ax.set_axis_off()
    tick_form = FuncFormatter(lambda x, pos: tick_format(x, pos))
    cb = plt.colorbar(pc)
    cb.ax.set_yscale('linear')
    cb.ax.yaxis.set_major_formatter(tick_form)
    cb.set_label(label='Gradient Magnitude', fontsize=14)
    if target_marker == [3]:
        ax.text(0.15*np.nanmax(x), 0.35*np.nanmax(y), 
                f"{mesh_name}\nBathymetry Slope", dict(size=16))
    elif target_marker == [1,2]:
        ax.text(0.15*np.nanmax(x), 0.35*np.nanmax(y), 
                f"{mesh_name}\nIce Tongue Slope", dict(size=16))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    return np.nanmax(masked_grad)

def plot_vertical_extent(filename, resolution=2000):
    
    file_list = filename.split('/')
    mesh_file = file_list[-1].split('.')
    mesh_name = format_latex_name(mesh_file[0])
    
    bathymetry_pts = extract_surface_geometry_points(filename, target_marker=[3])  # bathymetry
    surface_pts = extract_surface_geometry_points(filename, target_marker=[1,2])  # water surface and ice tongue
    
    x, y, bath_z = bathymetry_pts[:, 0], bathymetry_pts[:, 1], bathymetry_pts[:, 2]
    surf_x, surf_y, surf_z = surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2]
    
    # Interpolation grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), resolution),
        np.linspace(np.min(y), np.max(y), resolution)
    )
    points_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Create boundary mask using outline
    outline_x, outline_y = get_outline('outline')
    domain_path = Path(np.column_stack((outline_x, outline_y)))
    inside = domain_path.contains_points(points_grid)

    # Interpolate only inside region
    interp_bath_values = griddata((x, y), bath_z, points_grid[inside], method='linear')
    interp_surf_values = griddata((surf_x, surf_y), surf_z, points_grid[inside], method='linear')
    # Fill result with NaNs, write interpolated values inside
    grid_z = np.full(grid_x.shape, np.nan)
    grid_z.ravel()[inside] = abs(interp_bath_values - interp_surf_values)
    # Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_axis_off()
    viridis = mpl.colormaps['viridis']
    cmap = viridis.reversed()
    pc = plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap=cmap)
    tick_form = FuncFormatter(lambda x, pos: tick_format(x, pos))
    cb = plt.colorbar(pc, format = tick_form)
    cb.set_label(label='Meters', fontsize=14)
    ax.text(0.15*np.nanmax(x), 0.35*np.nanmax(y), 
            f"{mesh_name}\nVertical Extent", dict(size=16))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    print(np.nanmax(grid_z))
    
def final_format_table(infile_name):
    outfile_name = infile_name[:-4] + '.txt'
    with open(infile_name, "r") as in_file:
        with open(outfile_name, "w") as out_file:
            lines = in_file.readlines()
            all_lines = ""
            i = 0
            for line in lines:
                entries = line.split(',')
                if i == 0:
                    new_line = entries[0] 
                    final_entry = f'{entries[-1].replace('\n','')} \\\ \n'
                else:
                    new_line = format_latex_name(entries[0]) 
                    final_entry = f"{entries[-2]},{entries[-1].replace('\n','')}"
                    final_entry = final_entry.replace('"','')
                    final_entry = final_entry.replace("'","")
                    final_entry.replace('/n','')
                    final_entry = f"{final_entry} \\\ \n"
                    entries = entries[:-1]
                for entry in entries[4:6] + entries[1:2] + entries[6:-1] + entries[2:4]:
                    new_line = new_line + f" & {entry}"    
                all_lines = all_lines + new_line + ' & ' + final_entry
                i += 1
            out_file.write(all_lines)
    

mesh_limits = {"unstructured": 100, "scaled": 100, "layered": 55, 
               "stacked": None, "adaptive": None}

for mesh_type in mesh_limits.keys():
    csv_path = f"poisson/results/{mesh_type}/error_data.csv"
    #csv_path = f"poisson/results/analytical/{mesh_type}.csv"
    #plot_relative_error(csv_path, mesh_type, limit_y_scale=None)
    
csv_path = "poisson/results/analytical/high_f.csv"
#plot_relative_error(csv_path, "high_f", limit_y_scale=None)

#plot_node_count(["mesh/scaled/dof_test_200.txt", "mesh/scaled/dof_test_220.txt"])

mesh_types = ["unstructured", "scaled", "layered", "stacked", "adaptive"]

for mesh_type in mesh_types:
    folder = f"mesh/element_qualities/{mesh_type}/"
    if mesh_type == 'unstructured':
        mesh_names = ["M_400", "M_300", "M_250", "M_200", "M_160"]
    elif mesh_type == 'scaled':
        mesh_names = ['S_6_220', 'S_6_220_opt', 'S_3_200', 'S_11_200', 'S_6_200_opt']
        mesh_names = ['S_6_220', 'S_11_200', 'S_6_220_opt', 'S_6_200_opt']
    elif mesh_type == 'layered':
        mesh_names = ['L_9_295', 'L_12_340', 'L_16_410', 'L_20_450']
    elif mesh_type == 'stacked':
        mesh_names = ['P_10_730', 'P_15_560', 'P_20_480', 'P_25_450', 'P_50_310',
                      'P_75_255', 'P_100_225']
        mesh_names = ['P_10_730', 'P_15_560', 'P_20_480', 'P_25_450', 'P_50_310',
                      'P_75_255']
    elif mesh_type == 'adaptive':
        mesh_names = ['A_P_10_560', 'A_P_15_470', 'A_P_20_400', 'A_L_20_460', 
                      'A_S_6_190']
    filenames = [f"{folder}{mesh}.pos" for mesh in mesh_names]
        
    #plot_mesh_qualities(filenames, mesh_type)

res=500

mesh = "stacked/P_15_560"
#ks = [1, 10, 20, 30, 40, 50]
ks = [1, 10, 20, 30, 40, 50, 53, 70, 90, 105]

for k in ks:
    if k <= 50 or mesh.split('/')[0] == "combo":
        csv_path = f"poisson/results/analytical/{mesh.split('/')[0]}.csv"
    elif mesh.split('/')[0] == "adaptive":
        csv_path = "poisson/results/analytical/high_f_adapt.csv"
    else:
        csv_path = "poisson/results/analytical/high_f.csv"
    #interpolate_averaged_pointwise_error(f"poisson/paraview_outputs/{mesh}_error_k{k}000000.vtu",
    #                                     resolution=res, csv_path=csv_path, limit=4.033701720146616)


#lim = plot_gradient(f"mesh/{mesh}.msh", target_marker=[3], resolution = res)
#plot_gradient(f"mesh/{mesh}.msh", target_marker=[1, 2], resolution = res, limit=lim)

#plot_vertical_extent(f"mesh/{mesh}.msh", resolution = res)

final_format_table("poisson/results/analytical/final_all.csv")
