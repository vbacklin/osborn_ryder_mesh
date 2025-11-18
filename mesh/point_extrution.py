import gmsh
import sys
import numpy as np

lc = 0.1       # target mesh size
eps_z = 0.05   # thickness in z

gmsh.initialize()
gmsh.model.add("square_manual_3d")

# --- Bottom square geometry (z = 0) ---
p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

loop_b = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s_b    = gmsh.model.geo.addPlaneSurface([loop_b])

gmsh.model.geo.synchronize()

# Mesh the 2D surface to discover interior nodes
gmsh.model.mesh.generate(2)

# --- Collect interior surface nodes (exclude boundary) ---
surf_node_tags, surf_node_coords, _ = gmsh.model.mesh.getNodes(
    dim=2, tag=s_b, includeBoundary=True
)

boundary_tags = set()
for lt in [l1, l2, l3, l4]:
    tags, _, _ = gmsh.model.mesh.getNodes(dim=1, tag=lt, includeBoundary=True)
    boundary_tags.update(tags.tolist())

surf_tags_set = set(surf_node_tags.tolist())
interior_tags = sorted(surf_tags_set.difference(boundary_tags))

coords = np.array(surf_node_coords, dtype=float).reshape(-1, 3)
tag_to_xyz = {int(t): tuple(coords[i]) for i, t in enumerate(surf_node_tags)}

# --- Build the top boundary (manually; no extrude) ---
# Copy the 4 corner points to z = eps_z
p1t = gmsh.model.geo.addPoint(0.0, 0.0, eps_z, lc)
p2t = gmsh.model.geo.addPoint(1.0, 0.0, eps_z, lc)
p3t = gmsh.model.geo.addPoint(1.0, 1.0, eps_z, lc)
p4t = gmsh.model.geo.addPoint(0.0, 1.0, eps_z, lc)

# Top edges
l1t = gmsh.model.geo.addLine(p1t, p2t)
l2t = gmsh.model.geo.addLine(p2t, p3t)
l3t = gmsh.model.geo.addLine(p3t, p4t)
l4t = gmsh.model.geo.addLine(p4t, p1t)
loop_t = gmsh.model.geo.addCurveLoop([l1t, l2t, l3t, l4t])
s_t    = gmsh.model.geo.addPlaneSurface([loop_t])

# Vertical edges at the corners
lv1 = gmsh.model.geo.addLine(p1, p1t)
lv2 = gmsh.model.geo.addLine(p2, p2t)
lv3 = gmsh.model.geo.addLine(p3, p3t)
lv4 = gmsh.model.geo.addLine(p4, p4t)

# Side faces (each is a planar quad loop)
loop_s1 = gmsh.model.geo.addCurveLoop([l1, lv2, -l1t, -lv1])
s_s1    = gmsh.model.geo.addPlaneSurface([loop_s1])

loop_s2 = gmsh.model.geo.addCurveLoop([l2, lv3, -l2t, -lv2])
s_s2    = gmsh.model.geo.addPlaneSurface([loop_s2])

loop_s3 = gmsh.model.geo.addCurveLoop([l3, lv4, -l3t, -lv3])
s_s3    = gmsh.model.geo.addPlaneSurface([loop_s3])

loop_s4 = gmsh.model.geo.addCurveLoop([l4, lv1, -l4t, -lv4])
s_s4    = gmsh.model.geo.addPlaneSurface([loop_s4])

# --- Make a closed volume from the six faces ---
gmsh.model.geo.synchronize()
sl = gmsh.model.geo.addSurfaceLoop([s_b, s_t, s_s1, s_s2, s_s3, s_s4])
vol = gmsh.model.geo.addVolume([sl])

# --- Duplicate ONLY interior surface nodes to z = eps_z and embed them ---
lifted_pts = []
for t in interior_tags:
    x, y, z = tag_to_xyz[t]
    lifted_pts.append(gmsh.model.geo.addPoint(x, y, eps_z, lc))

gmsh.model.geo.synchronize()

# Embed the lifted points into the volume so they appear in the 3D mesh
# (dim=0 points into inDim=3 entity)
if lifted_pts:
    gmsh.model.mesh.embed(0, lifted_pts, 3, vol)

# 3D mesh (no extrusion used anywhere)
gmsh.model.mesh.generate(3)

gmsh.write("square_manual_3d.msh")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
