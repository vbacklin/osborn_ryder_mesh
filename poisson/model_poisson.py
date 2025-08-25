import meshio
import dolfin as df
import numpy as np
import pandas as pd

def convert_msh_to_xdmf(filename):
    
    msh = meshio.read(filename)
    
    triangle_cells = []
    for cell in msh.cells:
        if cell.type == 'triangle':
            if len(triangle_cells) == 0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])
        elif cell.type == 'tetra':
            tetra_cells = cell.data

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == 'triangle':
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == 'tetra':
            tetra_data = msh.cell_data_dict["gmsh:physical"][key]

    triangle_mesh = meshio.Mesh(points = msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"name_to_read":[triangle_data]})
    tetra_mesh = meshio.Mesh(points = msh.points, cells = {'tetra': tetra_cells},
                             cell_data={"name_to_read":[tetra_data]})
    
    vmesh_name = "volume_mesh.xdmf"
    meshio.write(vmesh_name, tetra_mesh)
    smesh_name = "surface_mesh.xdmf"
    meshio.write(smesh_name, triangle_mesh)
    
    return vmesh_name, smesh_name

def loadmesh(filename):

    xdmf_mesh = filename
    with df.XDMFFile(xdmf_mesh) as xdmf_infile:
        mesh = df.Mesh()
        xdmf_infile.read(mesh)

    return mesh

def get_boundary_nodes(V):
    dofmap = V.dofmap()
    mesh = V.mesh()
    
    boundary_dofs = set()
    
    for facet in df.facets(mesh):
        if facet.exterior():
            for vertex in df.vertices(facet):
                dofs = dofmap.entity_dofs(mesh, 0, [vertex.index()])
                boundary_dofs.update(dofs)
    
    return boundary_dofs
    
def create_reference_solution(k = 1, space = 1, m=220, scale=6, num_of_layers = 2, 
                              adapt = False, opt = False, stack = 0, cube=False):
    
    if cube:
        filename = "../gmsh example/ref_cube_m001.msh"
        out_file = f"../gmsh example/ref_cube_k{k}.pvd"
    else:   
        prefix = '../mesh/'
        if scale > 1:
            folder = "scaled/"
            meshname = f"S_{scale}_{m}"
            if opt:
                meshname = f"{meshname}_opt"
        elif num_of_layers > 2:
            folder = "layered/"
            meshname = f"L_{num_of_layers}_{m}"
        elif stack > 0:
            folder = "stacked/"
            meshname = f"P_{stack}_{m}"
        else:
            folder = "unstructured/"
            meshname = f"M_{m}"
        
        if adapt:
            folder = "adaptive/"
            meshname = f"A_{meshname}"
        filename = f"{prefix}{folder}{meshname}.msh"
        
        out_file = f"paraview_outputs/k{k}_ref_sol_{meshname}.pvd"
    
    volume, surfaces = convert_msh_to_xdmf(filename)
    water_body = loadmesh(volume)
    
    coords = water_body.coordinates()
    dx = np.max(coords[:, 0])
    dy = np.max(coords[:, 1])
    if cube:
        dz = np.max(coords[:, 2])
    else:
        dz = abs(round(np.min(coords[:, 2]), 2))
        
    x_freq = k/dx 
    y_freq = k/dy 
    z_freq = k/dz 
    analytical = f"sin(x[0]*{x_freq}) + sin(x[1]*{y_freq}) + sin(x[2]*{z_freq})"
    
    V = df.FunctionSpace(water_body, "P", space)
    
    u_D = df.Expression(analytical, degree=5)
    u_exact = df.interpolate(u_D, V)
    
    df.File(out_file) << u_exact

def ryder_poisson(k = 1, space = 1, m=400, scale=1, num_of_layers = 2, 
                  adapt = False, opt = False, stack = 0):
    
    prefix = '../mesh/'
    if scale > 1:
        if num_of_layers > 2:
            folder = "combo/"
            meshname = f"C_{num_of_layers}_{scale}_{m}"
        else:
            folder = "scaled/"
            meshname = f"S_{scale}_{m}"
        if opt:
            meshname = f"{meshname}_opt"
    elif num_of_layers > 2:
        folder = "layered/"
        meshname = f"L_{num_of_layers}_{m}"
    elif stack > 0:
        folder = "stacked/"
        meshname = f"P_{stack}_{m}"
    else:
        folder = "unstructured/"
        meshname = f"M_{m}"
    
    if adapt:
        folder = "adaptive/"
        meshname = f"A_{meshname}"
    filename = f"{prefix}{folder}{meshname}.msh"
    out_file = f"paraview_outputs/{folder}{meshname}_poisson_k{k}.pvd"
    error_file = f"paraview_outputs/{folder}{meshname}_error_k{k}.pvd"
    if k <= 50:
        csv_path = f"results/{folder}/error_data.csv"
    else:
        csv_path = "results/high_f/error_data_high_f.csv"
    
    volume, surfaces = convert_msh_to_xdmf(filename)
    water_body = loadmesh(volume)
    
    coords = water_body.coordinates()
    dx = np.max(coords[:, 0])
    dy = np.max(coords[:, 1])
    dz = abs(round(np.min(coords[:, 2]), 2))
        
    x_freq = k/dx 
    y_freq = k/dy 
    z_freq = k/dz 
    analytical = f"sin(x[0]*{x_freq}) + sin(x[1]*{y_freq}) + sin(x[2]*{z_freq})"
    lhs = f"(sin(x[0]*{x_freq})*{x_freq**2} + sin(x[1]*{y_freq})*{y_freq**2} + sin(x[2]*{z_freq})*{z_freq**2})"
    
    u_D = df.Expression(analytical, degree=3)
    f = df.Expression(lhs, degree=3)
    
    V = df.FunctionSpace(water_body, "P", space)
    
    dofs = V.dim()
    boundary_nodes = get_boundary_nodes(V)
    
    n_boundary_nodes = len(boundary_nodes)
    boundary_ratio = (n_boundary_nodes/dofs)
    
    mvc = df.MeshValueCollection("size_t", water_body, 2)
    with df.XDMFFile(surfaces) as infile:
        infile.read(mvc, "name_to_read")
    boundary_surfaces = df.cpp.mesh.MeshFunctionSizet(water_body, mvc)
    
    u0 = df.Constant(0.0)
    
    bc1 = df.DirichletBC(V, u0, boundary_surfaces, 1) #Water surface
    bc2 = df.DirichletBC(V, u0, boundary_surfaces, 2) #Ice
    bc3 = df.DirichletBC(V, u0, boundary_surfaces, 3) #Bathymetry
    bc4 = df.DirichletBC(V, u0, boundary_surfaces, 4) #Inflow
    
    bc5 = df.DirichletBC(V, u_D, "on_boundary") #For test only
    
    # Define variational problem
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = f*v*df.dx 
    
    # Compute solution
    u_h = df.Function(V)
    
    print(f"Solving for mesh {meshname}, k = {k}.")
    if dofs >= 150000:
        solver_parameters = {"linear_solver": "cg", "preconditioner": "hypre_amg"}
        df.solve(a == L, u_h, [bc5], solver_parameters=solver_parameters)
    else:
        df.solve(a == L, u_h, [bc5])#, solver_parameters=solver_parameters)

    space_ind = f"P{space}"       
    
    # Save solution in VTK format
    if dofs < 300000 and space == 1:
        df.File(out_file) << u_h
    
    print(f"Calculating error for mesh {meshname}, k = {k}.")
    u_exact = df.Expression(analytical, degree=5)
    if dofs > 300000 and space == 1:
        error_L2 = df.sqrt(df.assemble((u_exact - u_h)**2 * df.dx))
        norm_L2 = df.sqrt(df.assemble(u_exact**2 * df.dx(water_body)))
        rel_error = error_L2 / norm_L2
    else:
        error_L2 = df.errornorm(u_exact, u_h, norm_type='L2')
        norm_L2 = df.sqrt(df.assemble(u_exact**2 * df.dx(water_body)))
        rel_error = error_L2 / norm_L2
    
    
    V_error = df.FunctionSpace(water_body, "P", space) 
    
    error_func = df.Function(V_error)
    u_interp = df.interpolate(u_exact, V)
    u_h_vec = u_h.vector().get_local()
    u_D_vec = u_interp.vector().get_local()
    
    error_func.vector()[:] = np.abs(u_h_vec - u_D_vec)
    error_vec = error_func.vector().get_local()
    
    max_error = np.max(error_vec)
    mean_error = np.mean(error_vec)
    median_error = np.median(error_vec)
    min_max_radius_ratio = df.MeshQuality.radius_ratio_min_max(water_body)
    min_max_radius_ratio = (format_float(min_max_radius_ratio[0]), 
                            format_float(min_max_radius_ratio[1]))
    
    print(f"Done with mesh {meshname}, k = {k}.")
    if dofs < 300000:
        df.File(error_file) << error_func
    
    return [meshname, dofs, n_boundary_nodes, format_float(boundary_ratio), k, 
            space_ind, format_float(error_L2), format_float(100*rel_error), 
            format_float(max_error), format_float(mean_error), 
            format_float(median_error), min_max_radius_ratio], csv_path

def format_float(x):
    if isinstance(x, float):
        return f"{x:.3g}"
    return x

def main():
        
    csv_path = "results/analytical/all.csv"
    
    ks = [1, 10, 20, 30, 40, 50]
    #ks = [53, 70, 90, 105]
    #ks = [1, 10, 20, 30, 40, 50, 53, 70, 90, 105]
    #ks = [50]
    space = 1
    
    results = []
    
    #Unstructured:
    unstructured = [(400, 1, 2, False, False, 0), (300, 1, 2, False, False, 0), 
                    (250, 1, 2, False, False, 0), (200, 1, 2, False, False, 0), 
                    (160, 1, 2, False, False, 0)]
    
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
    
    #High freq test:
    high_freq = [(730, 1, 2, False, False, 10), (560, 1, 2, False, False, 15), 
                 (480, 1, 2, False, False, 20), (305, 1, 2, False, False, 10),
                 (450, 1, 20, False, False, 0), (460, 1, 20, True, False, 0), 
                 (560, 1, 2, True, False, 10), (470, 1, 2, True, False, 15), 
                 (400, 1, 2, True, False, 20)]
    
    combo = [(295, 11, 5, False, False, 0), (410, 39, 13, False, False, 0)] 
    
    params = unstructured
    
    for m, scale, num_of_layers, adapt, opt, stack in params:
        for k in ks:
            res, csv_path2 = ryder_poisson(k = k, space = space, m = m, scale = scale, 
                                          num_of_layers = num_of_layers,
                                          adapt = adapt, opt = opt, stack = stack)
            results.append(res)

    
    column_headers = ["Mesh", "DOFs", "Boundary Node Count", "Boundary Ratio", 
                      "Frequency Scale", "Function Space", "Global Error", 
                      "Rel. Error (%)", "Max Error", "Mean Error", "Median Error", 
                      "Min/Max Radius Ratio"]
    
    dataframe = pd.DataFrame(results, columns=column_headers)
    
    dataframe.to_csv(csv_path, index=False) 

    
    params = [(480, 1, 2, False, False, 20)]
    
    for m, scale, num_of_layers, adapt, opt, stack in params:
        create_reference_solution(k = 50, space = 1, m=m, scale=scale, 
                                  num_of_layers = num_of_layers, adapt = adapt, 
                                  opt = opt, stack = stack, cube=True)
    
if __name__ == "__main__":
    main()
