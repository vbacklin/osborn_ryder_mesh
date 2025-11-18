import gmsh
import sys

gmsh.initialize()
gmsh.model.add("square_multi_extrude")

# --- Geometry kernel: use OCC (robust for booleans) ---
occ = gmsh.model.occ

# Base square [0,1]x[0,1], split into two regions by x=0.5:
# Left region:  [0, 0.5] x [0, 1]
# Right region: [0.5, 1] x [0, 1]
sL = occ.addRectangle(0.0, 0.0, 0.0, 0.5, 1.0)   # tag of a 2D surface
sR = occ.addRectangle(0.5, 0.0, 0.0, 0.5, 1.0)

gmsh.model.occ.synchronize()

# If you already have a single surface and want to split it by a line,
# you could use occ.fragment() with a wire; here we just built two surfaces.

# --- Extrude each surface by a different amount ---
h_left  = 0.20
h_right = 0.50

# Extrude returns a list of (dim, tag) of created entities; we’ll grab the volumes.
outL = occ.extrude([(2, sL)], 0, 0, h_left)
outR = occ.extrude([(2, sR)], 0, 0, h_right)

# Helper to pick the (3, volTag) from the extrusion return
def last_volume(extrude_out):
    vols = [t for t in extrude_out if t[0] == 3]
    if not vols:
        raise RuntimeError("No volume produced by extrusion.")
    return vols[-1]  # the volume is typically the last 3D entity

vL = last_volume(outL)[1]
vR = last_volume(outR)[1]

# --- Imprint/fragment so the common vertical interface is shared ---
# This ensures a conformal mesh where the shorter block ends (at z = h_left)
# the taller block's side face is split accordingly.
frag_out, frag_map = occ.fragment([(3, vL)], [(3, vR)])

# Optional: clean up duplicates (harmless if nothing to clean)
occ.removeAllDuplicates()
occ.synchronize()

# --- (Optional) Define physical groups to keep nice tags after meshing ---
# Volumes
pg_vL = gmsh.model.addPhysicalGroup(3, [t[1] for t in frag_out if t[0] == 3][:1])  # one of the resulting left fragments
pg_vR = gmsh.model.addPhysicalGroup(3, [t[1] for t in frag_out if t[0] == 3][1:2]) # and one right fragment
gmsh.model.setPhysicalName(3, pg_vL, "LeftVol_h=0.20")
gmsh.model.setPhysicalName(3, pg_vR, "RightVol_h=0.50")

# Surfaces (example: bottom face(s) at z=0)
bottom_sfs = gmsh.model.getEntitiesInBoundingBox(0, 0, -1e-9, 1, 1, 1e-9, 2)
if bottom_sfs:
    pg_bottom = gmsh.model.addPhysicalGroup(2, [t[1] for t in bottom_sfs])
    gmsh.model.setPhysicalName(2, pg_bottom, "BottomFaces_z0")

# You can similarly pick side or top faces using bounding boxes.

# --- Mesh ---
gmsh.model.mesh.generate(3)

# Uncomment to inspect in the GUI:
# gmsh.fltk.run()

gmsh.write("square_multi_extrude.brep")
gmsh.write("square_multi_extrude.msh")
gmsh.finalize()
