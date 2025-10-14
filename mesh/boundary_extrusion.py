import gmsh
import math

# -------------------
# Tunables (tets only)
# -------------------
L = 1.0          # base square side length
H = 0.5          # height
h = 0.15         # target mesh size (global)
margin_top = 0.30 * L   # total clipping (both sides) at the very top
n_sections = 5          # number of intermediate sections (>= 2: base & top)
profile_pow = 2.0       # 1 = linear frustum, >1 = gentler near edges (clamped feel)
                         # try 2..4 for a rounder clamp

# Derived: keep sane
if margin_top >= L:
    raise ValueError("margin_top must be < L")

gmsh.initialize()
gmsh.model.add("clamped_tets")

occ = gmsh.model.occ

# Helper: add a square wire of side 'side' centered at (L/2, L/2) at height z
def add_square_wire(side, z, hsize):
    cx, cy = L/2, L/2
    s2 = side / 2.0
    x0, x1 = cx - s2, cx + s2
    y0, y1 = cy - s2, cy + s2
    p = [
        occ.addPoint(x0, y0, z, hsize),
        occ.addPoint(x1, y0, z, hsize),
        occ.addPoint(x1, y1, z, hsize),
        occ.addPoint(x0, y1, z, hsize),
    ]
    l = [
        occ.addLine(p[0], p[1]),
        occ.addLine(p[1], p[2]),
        occ.addLine(p[2], p[3]),
        occ.addLine(p[3], p[0]),
    ]
    return occ.addWire(l)

# Build section wires from bottom (t=0) to top (t=1)
# side(t) = L - margin_top * t^profile_pow  (0 at base → L; 1 at top → L - margin_top)
sections = []
for k in range(n_sections):
    t = k / (n_sections - 1) if n_sections > 1 else 1.0
    side = L - margin_top * (t ** profile_pow)
    # ensure side doesn't go negative due to floating point
    side = max(1e-12, side)
    z = H * t
    sections.append(add_square_wire(side, z, h))

occ.synchronize()

# Loft (thru sections) → solid
# Prefer keyword form; fall back to positional for older Gmsh
try:
    vol = occ.addThruSections(sections, makeSolid=True, makeRuled=True)
except TypeError:
    vol = occ.addThruSections(sections, True, True)

occ.synchronize()

# -------------------
# Mesh: pure tets
# -------------------
gmsh.option.setNumber("Mesh.RecombineAll", 0)   # no quads/hex/prisms
gmsh.option.setNumber("Mesh.Algorithm3D", 4)    # Frontal-Delaunay (tet)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# global size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

gmsh.model.mesh.generate(3)

# No Laplace smoothing (as requested). Netgen optimize only is kept to clean tets.
try:
    gmsh.model.mesh.optimize("Netgen")
except:
    pass

gmsh.write("clamped_tets.msh")
# gmsh.fltk.run()
gmsh.finalize()
