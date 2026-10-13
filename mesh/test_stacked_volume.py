"""Self-contained checks for stacked_volume.py.  No gmsh, no GDAL, no rasters.

Run: python -u test_stacked_volume.py
"""
import numpy as np
from stacked_volume import (build_stacked_volume, split_prism_to_tets,
                            layers_from_min_thickness, TOP, BOTTOM, SIDE)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def tri_area(xy, tris):
    a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    return 0.5 * np.abs(np.cross(b - a, c - a))


def tet_volumes(coords, tets):
    p = coords[tets]
    return np.einsum('ij,ij->i',
                     np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                     p[:, 3] - p[:, 0]) / 6.0


def boundary_edges_of(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    k = np.sort(e, axis=1)
    _, cnt = np.unique(k, axis=0, return_counts=True)
    uniq, idx, cnt = np.unique(k, axis=0, return_index=True, return_counts=True)
    return uniq[cnt == 1]


def verify(name, xy, tris, z_top, z_bot, L):
    print(f"\n--- {name}: {len(xy)} columns, {len(tris)} triangles, {L} layers ---")
    m = build_stacked_volume(xy, tris, z_top, z_bot, L)

    vols = tet_volumes(m.coords, m.tets)

    check("tet count == 3 * tris * layers",
          len(m.tets) == 3 * len(tris) * L, f"{len(m.tets)}")
    check("node count == columns * (layers+1)",
          len(m.coords) == len(xy) * (L + 1), f"{len(m.coords)}")
    check("all tets strictly positive volume",
          np.all(vols > 0), f"min {vols.min():.6g}")
    check("no inverted tets", (vols < 0).sum() == 0, f"{(vols < 0).sum()}")
    check("no zero-volume tets", (vols == 0).sum() == 0, f"{(vols == 0).sum()}")

    # Volume must equal the analytic prism volume exactly (to round-off).
    analytic = float(np.sum(tri_area(xy, tris) *
                            ((z_top - z_bot)[tris].mean(axis=1))))
    got = float(vols.sum())
    rel = abs(got - analytic) / max(abs(analytic), 1e-30)
    check("total volume == analytic prism volume", rel < 1e-10,
          f"rel err {rel:.3e}")

    # Conformity: every face is used once (boundary) or twice (interior).
    faces = np.concatenate([m.tets[:, [0, 2, 1]], m.tets[:, [0, 1, 3]],
                            m.tets[:, [0, 3, 2]], m.tets[:, [1, 2, 3]]])
    _, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    check("every face used once or twice", set(np.unique(cnt)) <= {1, 2},
          f"counts {sorted(set(cnt.tolist()))}")

    # The boundary surface must be closed: each undirected edge exactly twice.
    be = np.concatenate([m.boundary_faces[:, [0, 1]],
                         m.boundary_faces[:, [1, 2]],
                         m.boundary_faces[:, [2, 0]]])
    _, ecnt = np.unique(np.sort(be, axis=1), axis=0, return_counts=True)
    check("boundary surface is closed (no open edges)",
          np.all(ecnt == 2), f"edges with count!=2: {int((ecnt != 2).sum())}")

    # And consistently oriented: each *directed* edge exactly once.
    _, dcnt = np.unique(be, axis=0, return_counts=True)
    check("boundary orientation consistent", np.all(dcnt == 1),
          f"directed edges repeated: {int((dcnt != 1).sum())}")

    # Boundary face budget: caps are one triangle per 2D triangle each;
    # walls are 2 triangles per boundary edge per layer.
    nb_edge = len(boundary_edges_of(tris))
    check("top cap == #triangles",
          (m.boundary_kind == TOP).sum() == len(tris),
          f"{(m.boundary_kind == TOP).sum()} vs {len(tris)}")
    check("bottom cap == #triangles",
          (m.boundary_kind == BOTTOM).sum() == len(tris),
          f"{(m.boundary_kind == BOTTOM).sum()} vs {len(tris)}")
    check("side walls == 2 * boundary edges * layers",
          (m.boundary_kind == SIDE).sum() == 2 * nb_edge * L,
          f"{(m.boundary_kind == SIDE).sum()} vs {2 * nb_edge * L}")

    # Degeneracy checks matching mesh_diagnostics/degeneracy.py.
    p = m.coords[m.tets]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    emin = np.min([np.linalg.norm(p[:, a] - p[:, b], axis=1) for a, b in pairs],
                  axis=0)
    check("no zero-length edges", (emin == 0).sum() == 0, f"{(emin == 0).sum()}")

    # Every node must be distinct.
    uniq_nodes = np.unique(np.round(m.coords, 9), axis=0)
    check("no duplicate nodes", len(uniq_nodes) == len(m.coords),
          f"{len(m.coords) - len(uniq_nodes)} duplicates")

    # Vertical structure: this is the property the seeded meshes lacked.
    key = np.round(m.coords[:, :2], 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    per = np.bincount(inv)
    check("every column carries layers+1 nodes",
          np.all(per == L + 1), f"min {per.min()} max {per.max()}")
    return m


# --- 1. one prism, both diagonal cases ------------------------------------
print("=== single prism ===")
for label, cols in (("ordered", (0, 1, 2)), ("rotated", (2, 0, 1)),
                    ("reversed", (2, 1, 0))):
    tets = split_prism_to_tets([0, 1, 2], [3, 4, 5], cols)
    allv = set()
    for t in tets:
        allv |= set(t)
    check(f"{label}: 3 tets covering all 6 vertices",
          len(tets) == 3 and allv == {0, 1, 2, 3, 4, 5})

# Two prisms sharing a quad face must pick the same diagonal.
# Columns 5 and 9 shared; prism A has third column 2, prism B has third 7.
a = split_prism_to_tets([0, 1, 2], [3, 4, 5], (2, 5, 9))
b = split_prism_to_tets([10, 1, 2], [13, 4, 5], (7, 5, 9))


def quad_diagonal(tets, u_bot, v_bot, u_top, v_top):
    """Which diagonal splits quad (u_bot,v_bot,v_top,u_top)?"""
    got = set()
    for t in tets:
        s = set(t)
        if {u_bot, v_top} <= s and not {v_bot, u_top} <= s:
            got.add("u_bot-v_top")
        if {v_bot, u_top} <= s and not {u_bot, v_top} <= s:
            got.add("v_bot-u_top")
    return got


da = quad_diagonal(a, 1, 2, 4, 5)
db = quad_diagonal(b, 1, 2, 4, 5)
check("neighbouring prisms agree on the shared quad diagonal",
      da == db and len(da) == 1, f"{da} vs {db}")

# --- 2. structured grid, flat ---------------------------------------------
nx, ny = 6, 5
gx, gy = np.meshgrid(np.linspace(0, 5000, nx), np.linspace(0, 4000, ny))
xy = np.column_stack([gx.ravel(), gy.ravel()])
tris = []
for j in range(ny - 1):
    for i in range(nx - 1):
        a0 = j * nx + i
        tris += [[a0, a0 + 1, a0 + nx], [a0 + 1, a0 + nx + 1, a0 + nx]]
tris = np.array(tris)
verify("flat slab, 38 layers", xy, tris,
       np.zeros(len(xy)), np.full(len(xy), -600.0), 38)

# --- 3. sloped floor + ice draft, the realistic shape ----------------------
rng = np.random.default_rng(0)
z_top = np.where(xy[:, 0] < 2500, -40 - 0.01 * xy[:, 1], 0.0)
z_bot = -300 - 0.06 * xy[:, 0] - 0.02 * xy[:, 1] + rng.normal(0, 5, len(xy))
verify("sloped floor under ice, 38 layers", xy, tris, z_top, z_bot, 38)
verify("same geometry, 15 layers", xy, tris, z_top, z_bot, 15)
verify("same geometry, 1 layer", xy, tris, z_top, z_bot, 1)

# --- 4a. same mesh with column indices randomly permuted ------------------
# The diagonal rule keys on column *indices*, so an arbitrary relabelling is
# the sharpest test that neighbouring prisms still agree.  A mesh generator
# gives no guarantee that a triangle's indices are ordered or local.
perm = rng.permutation(len(xy))
inv_perm = np.empty_like(perm)
inv_perm[perm] = np.arange(len(perm))
verify("permuted column indices, 38 layers", xy[perm], inv_perm[tris],
       z_top[perm], z_bot[perm], 38)

# --- 4b. genuinely unstructured triangulation from gmsh -------------------
try:
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disc")
    lc = 300.0
    pts_tags = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in
                [(0, 0), (6000, 0), (6000, 4000), (2500, 5200), (0, 4000)]]
    lines = [gmsh.model.geo.addLine(pts_tags[i], pts_tags[(i + 1) % len(pts_tags)])
             for i in range(len(pts_tags))]
    surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    ncoords = np.asarray(ncoords).reshape(-1, 3)
    remap = {int(t): i for i, t in enumerate(ntags)}
    etypes, etags, enodes = gmsh.model.mesh.getElements(2)
    conn = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 3)
    utris = np.vectorize(remap.get)(conn)
    gmsh.finalize()
    upts = ncoords[:, :2]
    zt = np.where(upts[:, 0] < 3000, -30.0, 0.0)
    zb = -250 - 0.03 * upts[:, 0] + rng.normal(0, 6, len(upts))
    verify("unstructured gmsh triangulation, 38 layers", upts, utris, zt, zb, 38)
except ImportError:
    print("\n(gmsh unavailable -- skipping unstructured case)")

# --- 5. degenerate input must be rejected, not silently meshed -------------
print("\n--- degenerate input handling ---")
bad_bot = np.full(len(xy), -600.0)
bad_bot[3] = 0.0   # zero-thickness column
try:
    build_stacked_volume(xy, tris, np.zeros(len(xy)), bad_bot, 10)
    check("zero-thickness column raises", False, "no exception")
except ValueError as e:
    check("zero-thickness column raises", "non-positive thickness" in str(e))

# --- 6. variable layer count per column (sweep split) ----------------------
def verify_variable(name, xy, tris, z_top, z_bot, L, n):
    print(f"\n--- {name}: {len(xy)} columns, {len(tris)} triangles, "
          f"layers {n.min()}..{n.max()} (max {L}) ---")
    m = build_stacked_volume(xy, tris, z_top, z_bot, L, layers_per_column=n)
    vols = tet_volumes(m.coords, m.tets)

    check("tet count == sum over tris of the 3 columns' layers",
          len(m.tets) == int(n[tris].sum()), f"{len(m.tets)} vs {int(n[tris].sum())}")
    check("node count == sum(layers+1)", len(m.coords) == int((n + 1).sum()))
    check("all tets strictly positive volume", np.all(vols > 0),
          f"min {vols.min():.6g}")
    analytic = float(np.sum(tri_area(xy, tris) *
                            ((z_top - z_bot)[tris].mean(axis=1))))
    rel = abs(float(vols.sum()) - analytic) / abs(analytic)
    check("total volume == analytic prism volume", rel < 1e-10, f"rel err {rel:.3e}")

    faces = np.concatenate([m.tets[:, [0, 2, 1]], m.tets[:, [0, 1, 3]],
                            m.tets[:, [0, 3, 2]], m.tets[:, [1, 2, 3]]])
    _, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    check("every face used once or twice", set(np.unique(cnt)) <= {1, 2},
          f"counts {sorted(set(cnt.tolist()))}")
    be = np.concatenate([m.boundary_faces[:, [0, 1]], m.boundary_faces[:, [1, 2]],
                         m.boundary_faces[:, [2, 0]]])
    _, ecnt = np.unique(np.sort(be, axis=1), axis=0, return_counts=True)
    check("boundary surface is closed", np.all(ecnt == 2),
          f"{int((ecnt != 2).sum())} open")
    _, dcnt = np.unique(be, axis=0, return_counts=True)
    check("boundary orientation consistent", np.all(dcnt == 1))

    bed = boundary_edges_of(tris)
    check("top cap == #triangles", (m.boundary_kind == TOP).sum() == len(tris))
    check("bottom cap == #triangles", (m.boundary_kind == BOTTOM).sum() == len(tris))
    want_side = int(n[bed].sum())
    check("side walls == sum over boundary edges of both columns' layers",
          (m.boundary_kind == SIDE).sum() == want_side,
          f"{(m.boundary_kind == SIDE).sum()} vs {want_side}")

    check("node_col matches coordinates",
          np.allclose(m.coords[:, :2], xy[m.node_col]))
    per = np.bincount(m.node_col, minlength=len(xy))
    check("column c carries layers[c]+1 nodes", np.array_equal(per, n + 1))
    top = m.coords[m.node_level == 0]
    check("top level sits on z_top",
          np.allclose(top[:, 2], z_top[m.node_col[m.node_level == 0]]))
    return m


rng = np.random.default_rng(1)
z_top = np.where(xy[:, 0] < 2500, -40 - 0.01 * xy[:, 1], 0.0)
z_bot = -300 - 0.06 * xy[:, 0] - 0.02 * xy[:, 1] + rng.normal(0, 5, len(xy))
thk = z_top - z_bot
n_var = layers_from_min_thickness(thk, 38, 12.0)
verify_variable("sloped, layers from min thickness 12 m", xy, tris, z_top, z_bot,
                38, n_var)
n_rand = rng.integers(1, 39, len(xy))
verify_variable("random 1..38 layers per column", xy, tris, z_top, z_bot, 38, n_rand)
verify_variable("random layers, permuted column indices", xy[perm], inv_perm[tris],
                z_top[perm], z_bot[perm], 38, n_rand[perm])
n_alt = np.where(np.arange(len(xy)) % 2 == 0, 1, 38)
verify_variable("alternating 1 and 38 layers", xy, tris, z_top, z_bot, 38, n_alt)
m_u = verify_variable("all columns at max via sweep", xy, tris, z_top, z_bot, 15,
                      np.full(len(xy), 15))
check("sweep with uniform layers has 3 tets per prism-layer",
      len(m_u.tets) == 3 * len(tris) * 15)
try:
    zt_u = np.where(upts[:, 0] < 3000, -30.0, 0.0)
    zb_u = -250 - 0.03 * upts[:, 0] + rng.normal(0, 6, len(upts))
    verify_variable("unstructured gmsh triangulation, variable layers", upts, utris,
                    zt_u, zb_u, 38, layers_from_min_thickness(zt_u - zb_u, 38, 10.0))
except NameError:
    print("\n(no unstructured triangulation -- skipping)")

print("\n--- invalid layer counts ---")
for bad, label in ((np.zeros(len(xy), np.int64), "0 layers"),
                   (np.full(len(xy), 39), "above max")):
    try:
        build_stacked_volume(xy, tris, z_top, z_bot, 38, layers_per_column=bad)
        check(f"{label} raises", False, "no exception")
    except ValueError:
        check(f"{label} raises", True)
check("layers_from_min_thickness clips to [1, L]",
      list(layers_from_min_thickness([0.5, 25, 1e6], 38, 5.0)) == [1, 5, 38])

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
raise SystemExit(1 if FAIL else 0)
