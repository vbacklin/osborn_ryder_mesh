import gmsh
# -------------------
# Tunables (only tets)
# -------------------
L = 1.0   # square side length
H = 0.5   # max interior lift
h = 0.2   # target 2D mesh size (smaller = finer)
gmsh.initialize()
gmsh.model.add("clamped_tets")
# -------------------
# Geometry: square in XY at z=0
# -------------------
p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
p2 = gmsh.model.geo.addPoint(L, 0, 0, h)
p3 = gmsh.model.geo.addPoint(L, L, 0, h)
p4 = gmsh.model.geo.addPoint(0, L, 0, h)
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)
cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s = gmsh.model.geo.addPlaneSurface([cl])
gmsh.model.geo.synchronize()
# -------------------
# Triangle surface mesh (no recombine)
# -------------------
gmsh.option.setNumber("Mesh.RecombineAll", 0)  # absolutely no quads/hexes
gmsh.model.mesh.generate(2)
# -------------------
# Geometric extrude (creates volume)
# -------------------
# We just extrude the CAD by H; meshing to tets happens later.
out = gmsh.model.geo.extrude([(2, s)], 0, 0, H)  # no numElements/heights -> pure geometry
gmsh.model.geo.synchronize()
# Grab the volume tag (dim=3)
vols = [ent[1] for ent in out if ent[0] == 3]
if not vols:
    raise RuntimeError("No volume created by extrusion.")
vol = vols[0]
# -------------------
# 3D tetrahedral mesh
# -------------------
# Pick a robust 3D tet algorithm and enable quality-improving passes.
gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay (tet)
gmsh.option.setNumber("Mesh.Optimize", 1)  # enable optimizer
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # netgen-style improvement
gmsh.model.mesh.generate(3)
nodeTags, coords, _ = gmsh.model.mesh.getNodes()
coords = list(coords)
set_node = gmsh.model.mesh.setNode  # newer API: setNode(tag, [x,y,z], paramCoords)
for i, tag in enumerate(nodeTags):
    x = coords[3 * i + 0]
    y = coords[3 * i + 1]
    if x < 0.001:
        set_node(int(tag), [x, y, 0.0], [])
gmsh.model.geo.synchronize()
gmsh.write("clamped_tets.msh")
# gmsh.fltk.run()
gmsh.finalize()