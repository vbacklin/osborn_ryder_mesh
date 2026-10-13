"""Direct sigma-layer volume construction: stack a plan-view triangulation into
prisms and split each prism into tetrahedra.

Why this exists
---------------
The previous 3D path built a closed boundary skin and handed it to TetGen,
forcing interior vertical resolution by embedding one seed point per sigma
level per column.  That works at 15 layers (2492 seeds) and segfaults TetGen's
boundary recovery at 38 (6586 seeds).  Turning the seeds off lets the mesh
generate, but then the interior has no layering at all -- `extrude()` only
structures the *boundary* surfaces.

Here the interior is built directly and TetGen is never involved, so cost is
linear in the layer count and there is nothing to "recover".

Design notes
------------
* Node layout is a regular (column, level) grid: node index for column `c` at
  level `k` is `c * (L + 1) + k`, with `k = 0` at the top (sea surface or ice
  draft) and `k = L` at the floor.
* Each prism is split into 3 tets.  The three quad side faces each need a
  diagonal, and two prisms sharing a quad face must choose the *same* diagonal
  or the mesh is not conforming.  The rule is purely local: rotate the triangle
  so its smallest column index comes first, then pick each quad's diagonal
  through the endpoint with the smaller column index.  Both prisms sharing a
  face see the same pair of column indices, so they always agree.  No global
  coordination, which is what made the earlier per-face attempt collapse.
* The boundary is *derived* from the tets (faces used by exactly one tet)
  rather than constructed alongside them.  A separately-built boundary can
  disagree with the volume; a derived one cannot.
"""

from __future__ import annotations

import numpy as np

__all__ = ["split_prism_to_tets", "sweep_prism_to_tets", "build_stacked_volume", "CALVING",
           "stretched_fractions",
           "layers_from_min_thickness", "StackedMesh"]


def split_prism_to_tets(bot, top, cols):
    """Split one prism into 3 tetrahedra.

    `bot` / `top` are the three node indices of the lower and upper triangle,
    corresponding column-wise (bot[i] and top[i] are the same column).  `cols`
    are the three *column* indices, which is what the diagonal rule keys on --
    they are shared by neighbouring prisms, node indices are not.

    Returns a list of three 4-tuples of node indices.  Orientation is not
    guaranteed here; the caller fixes sign from the coordinates.
    """
    # Rotate so the smallest column index leads.  This is what makes the choice
    # reproducible from any neighbouring prism's point of view.
    r = int(np.argmin(cols))
    order = [r, (r + 1) % 3, (r + 2) % 3]
    c0, c1, c2 = (cols[i] for i in order)
    v0, v1, v2 = (bot[i] for i in order)
    v3, v4, v5 = (top[i] for i in order)

    # Quad faces (v0,v1,v4,v3) and (v2,v0,v3,v5) both take their diagonal
    # through column c0, which is the smallest -- so v0-v4 and v0-v5.
    # The remaining quad (v1,v2,v5,v4) is decided by c1 vs c2.
    if c1 < c2:
        # diagonal v1-v5
        return [(v0, v1, v2, v5), (v0, v1, v5, v4), (v0, v4, v5, v3)]
    # diagonal v2-v4
    return [(v0, v1, v2, v4), (v0, v4, v2, v5), (v0, v4, v5, v3)]


def sweep_prism_to_tets(cols, node_ids, sigmas):
    """Split one prism whose three columns may carry different layer counts.

    `cols` are the three column indices, `node_ids[q]` / `sigmas[q]` the node
    ids and fractional depths (0 at top, 1 at bottom) of column q, top first.

    The prism is swept from top to bottom.  The current "front" is one node per
    column, starting at the top triangle.  Each step advances the column whose
    next node is shallowest -- ties broken by column index -- and emits the tet
    (front, next node).  Restricted to any two columns, that order depends
    only on those two columns' sigmas and indices, so the prism on the other
    side of a shared quad advances them in the same order and triangulates the
    face identically: conforming without global coordination, the same
    property split_prism_to_tets gets from its diagonal rule.

    Every tet is (plan triangle) x (one vertical step on one column), so its
    volume is |dz| * area / 6 > 0 -- it cannot degenerate.
    """
    i = [0, 0, 0]
    n = [len(node_ids[q]) - 1 for q in range(3)]
    out = []
    while True:
        cand = [(sigmas[q][i[q] + 1], cols[q], q) for q in range(3) if i[q] < n[q]]
        if not cand:
            return out
        q = min(cand)[2]
        out.append((node_ids[0][i[0]], node_ids[1][i[1]], node_ids[2][i[2]],
                    node_ids[q][i[q] + 1]))
        i[q] += 1


def layers_from_min_thickness(thickness, num_layers, min_layer_thickness):
    """Layer count per column so no layer is thinner than `min_layer_thickness`
    (where possible): clip(floor(thickness / min_layer_thickness), 1, L)."""
    n = np.floor(np.asarray(thickness, float) / float(min_layer_thickness))
    return np.clip(n, 1, int(num_layers)).astype(np.int64)


class StackedMesh:
    """Result of :func:`build_stacked_volume`.

    `node_col[i]` / `node_level[i]` give the plan column and level (0 = top)
    of node i.  `layers[c]` is the layer count of column c; `num_layers` is
    the maximum.  Use node_col rather than index arithmetic -- the node layout
    is only `c * (L + 1) + k` when every column has the same count.
    """

    __slots__ = ("coords", "tets", "boundary_faces", "boundary_kind",
                 "num_layers", "num_columns", "node_col", "node_level", "layers",
                 "boundary_from_ice")

    def __init__(self, coords, tets, boundary_faces, boundary_kind,
                 num_layers, num_columns, node_col, node_level, layers,
                 boundary_from_ice=None):
        self.coords = coords
        self.tets = tets
        self.boundary_faces = boundary_faces
        self.boundary_kind = boundary_kind
        self.num_layers = num_layers
        self.num_columns = num_columns
        self.node_col = node_col
        self.node_level = node_level
        self.layers = layers
        # For each boundary face: True if it belongs to an ice prism (only set
        # when build_stacked_volume was given tri_is_ice).
        self.boundary_from_ice = boundary_from_ice

    def faces_of(self, kind):
        return self.boundary_faces[self.boundary_kind == kind]


# Boundary classification codes.
TOP = 0      # sea surface or ice draft
BOTTOM = 1   # sea floor
SIDE = 2     # vertical wall (shoreline or inflow)
CALVING = 3  # vertical ice front: sea surface down to the ice draft


def _derive_boundary(tets):
    """Faces used by exactly one tet, with the outward-consistent node order.

    Gmsh's tet face convention for a positively-oriented tet (0,1,2,3) gives
    these four faces with outward normals.
    """
    faces = np.concatenate([
        tets[:, [0, 2, 1]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 3, 2]],
        tets[:, [1, 2, 3]],
    ])
    key = np.sort(faces, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True,
                                 return_counts=True)
    interior_bad = counts > 2
    if interior_bad.any():
        raise ValueError(
            f"{int(interior_bad.sum())} faces are shared by more than two tets "
            "-- the prism split is not conforming")
    return faces[first[counts == 1]]


def build_stacked_volume(xy, tris, z_top, z_bot, num_layers,
                         layers_per_column=None, tri_is_ice=None,
                         z_surface=None, upper_layers=None,
                         top_layer=None, max_growth=1.2):
    """Stack a plan-view triangulation into a layered tetrahedral volume.

    Parameters
    ----------
    xy : (C, 2) float
        Plan-view node coordinates, one per column.
    tris : (T, 3) int
        Triangles indexing into `xy`.
    z_top, z_bot : (C,) float
        Upper and lower surface at each column.  `z_top > z_bot` is required;
        a non-positive column thickness is an error rather than something to
        silently collapse, because that is exactly the degeneracy that has been
        contaminating these meshes.
    num_layers : int
        Number of sigma layers (>= 1).  Produces `num_layers + 1` levels.
    layers_per_column : (C,) int, optional
        Per-column layer count in [1, num_layers].  When given, prisms are
        split with sweep_prism_to_tets so neighbouring columns may differ;
        when omitted every column gets `num_layers` and the original
        diagonal-rule split is used unchanged.
    top_layer : (C,) float, optional
        Vertical stretching.  Where top_layer[c] > 0 and the column is thick
        enough, its (lower-part) layers grow geometrically from top_layer[c]
        at the top (ice base) to the floor, instead of being evenly spaced.
        Only node heights change; the topology is the same.
    tri_is_ice, z_surface, upper_layers : optional
        Vertical calving front.  `tri_is_ice` (T,) marks plan triangles under
        ice.  A column with `upper_layers[c] > 0` is a *front* column: it
        carries extra nodes from `z_surface[c]` (sea surface) down to
        `z_top[c]` (ice draft) in `upper_layers[c]` layers.  Open-water prisms
        use the whole front column; ice prisms only the part below the draft.
        Between two front columns the part above the draft is then a vertical
        boundary face, classified CALVING.

    Returns
    -------
    StackedMesh
    """
    xy = np.asarray(xy, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    z_top = np.asarray(z_top, dtype=float)
    z_bot = np.asarray(z_bot, dtype=float)
    L = int(num_layers)

    if L < 1:
        raise ValueError("num_layers must be >= 1")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (C, 2)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("tris must be (T, 3)")
    C = xy.shape[0]
    if z_top.shape != (C,) or z_bot.shape != (C,):
        raise ValueError("z_top and z_bot must be (C,)")

    thickness = z_top - z_bot
    bad = thickness <= 0
    if bad.any():
        worst = float(thickness.min())
        raise ValueError(
            f"{int(bad.sum())} of {C} columns have non-positive thickness "
            f"(min {worst:.6g}). Clamp the floor or clip the domain before "
            "stacking -- this construction will not hide a degenerate column.")

    if (tri_is_ice is not None or top_layer is not None) and layers_per_column is None:
        layers_per_column = np.full(C, L, np.int64)
    if layers_per_column is not None:
        return _build_variable(xy, tris, z_top, z_bot, L,
                               np.asarray(layers_per_column, np.int64),
                               tri_is_ice, z_surface, upper_layers, top_layer,
                               max_growth)

    # --- nodes: regular (column, level) grid, level 0 at the top -------------
    sigma = np.linspace(0.0, 1.0, L + 1)                 # (L+1,)
    z = z_top[:, None] - thickness[:, None] * sigma[None, :]   # (C, L+1)
    coords = np.empty((C * (L + 1), 3), dtype=float)
    coords[:, 0] = np.repeat(xy[:, 0], L + 1)
    coords[:, 1] = np.repeat(xy[:, 1], L + 1)
    coords[:, 2] = z.reshape(-1)

    def nid(col, lvl):
        return col * (L + 1) + lvl

    # --- tets ----------------------------------------------------------------
    tets = np.empty((tris.shape[0] * L * 3, 4), dtype=np.int64)
    n = 0
    for tri in tris:
        cols = (int(tri[0]), int(tri[1]), int(tri[2]))
        for k in range(L):
            # level k is shallower than level k+1, so k+1 is the "bottom".
            bot = [nid(c, k + 1) for c in cols]
            top = [nid(c, k) for c in cols]
            for t in split_prism_to_tets(bot, top, cols):
                tets[n] = t
                n += 1
    tets = tets[:n]

    # --- orientation ---------------------------------------------------------
    p = coords[tets]
    vol = np.einsum('ij,ij->i',
                    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0]) / 6.0
    flip = vol < 0
    if flip.any():
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    if not np.all(np.abs(vol) > 0):
        raise ValueError(f"{int((vol == 0).sum())} tets have zero volume")

    # --- boundary, derived from the tets themselves --------------------------
    bfaces = _derive_boundary(tets)
    lvl = bfaces % (L + 1)
    kind = np.full(bfaces.shape[0], SIDE, dtype=np.int8)
    kind[np.all(lvl == 0, axis=1)] = TOP
    kind[np.all(lvl == L, axis=1)] = BOTTOM

    node_col = np.repeat(np.arange(C), L + 1)
    node_level = np.tile(np.arange(L + 1), C)
    return StackedMesh(coords, tets, bfaces, kind, L, C, node_col, node_level,
                       np.full(C, L, np.int64))


def stretched_fractions(n, thickness, top, max_growth=1.2):
    """Level depths as fractions of the column, top (0) to floor (1).

    The first layer is `top` thick and each layer below is a constant factor
    r >= 1 thicker, r chosen so the n layers fill `thickness` exactly.  r is
    capped at `max_growth`; where the cap binds, the first layer comes out
    thicker than `top` instead.  A column with thickness <= n * top stays
    uniform.
    """
    if top <= 0 or thickness <= n * top or n < 2:
        return np.linspace(0.0, 1.0, n + 1)
    target = thickness / top                     # sum_{k<n} r^k = target
    geo = lambda r: (r ** n - 1) / (r - 1)
    if geo(max_growth) <= target:
        r = max_growth
    else:
        lo, hi = 1.0, max_growth
        for _ in range(100):
            r = 0.5 * (lo + hi)
            (lo, hi) = (r, hi) if geo(r) < target else (lo, r)
        r = 0.5 * (lo + hi)
    dz = r ** np.arange(n)
    f = np.concatenate([[0.0], np.cumsum(dz)])
    return f / f[-1]


def _build_variable(xy, tris, z_top, z_bot, L, n, tri_is_ice=None,
                    z_surface=None, upper_layers=None, top_layer=None,
                    max_growth=1.2):
    C = len(xy)
    T = len(tris)
    if n.shape != (C,):
        raise ValueError("layers_per_column must be (C,)")
    if n.min() < 1 or n.max() > L:
        raise ValueError(f"layers_per_column must lie in [1, {L}], "
                         f"got [{n.min()}, {n.max()}]")
    ice_tri = (np.zeros(T, bool) if tri_is_ice is None
               else np.asarray(tri_is_ice, bool))
    nu = (np.zeros(C, np.int64) if upper_layers is None
          else np.asarray(upper_layers, np.int64))
    if ice_tri.shape != (T,) or nu.shape != (C,) or nu.min() < 0:
        raise ValueError("tri_is_ice must be (T,), upper_layers (C,) and >= 0")
    front = nu > 0
    if front.any():
        if z_surface is None:
            raise ValueError("upper_layers needs z_surface")
        z_surface = np.asarray(z_surface, float)
        if not np.all(z_surface[front] > z_top[front]):
            raise ValueError("front columns need z_surface above the ice draft")
    thickness = z_top - z_bot

    # Column c holds nu[c] upper layers (surface -> draft) above n[c] lower
    # layers (draft -> floor).  The node at index nu[c] is the draft (or the
    # top, for a column without an upper part).
    per = n + 1 + nu
    off = np.concatenate([[0], np.cumsum(per)])
    node_col = np.repeat(np.arange(C), per)
    node_level = np.arange(off[-1]) - off[node_col]
    k = node_level - nu[node_col]               # <0 upper, 0 draft, >0 lower
    up = k < 0
    # Sweep key: upper part in [-1, 0), draft 0, lower part sigma in (0, 1].
    key = np.where(up, k / np.maximum(nu[node_col], 1), k / n[node_col])
    depth_frac = np.where(up, 0.0, key)
    if top_layer is not None:
        top_layer = np.asarray(top_layer, float)
        for c in np.flatnonzero(top_layer > 0):
            lo = off[c] + nu[c]
            depth_frac[lo:off[c + 1]] = stretched_fractions(
                int(n[c]), float(thickness[c]), float(top_layer[c]), max_growth)
    z = z_top[node_col] - depth_frac * thickness[node_col]
    if front.any():
        zs = np.where(front, z_surface, z_top)
        z = np.where(up, z_top[node_col] + (-key) * (zs - z_top)[node_col], z)
    coords = np.column_stack([xy[node_col, 0], xy[node_col, 1], z])

    full_ids = [np.arange(off[c], off[c + 1]) for c in range(C)]
    low_ids = [np.arange(off[c] + nu[c], off[c + 1]) for c in range(C)]

    tets, owner = [], []
    for t, tri in enumerate(tris):
        cols = (int(tri[0]), int(tri[1]), int(tri[2]))
        lists = [low_ids[c] if ice_tri[t] else full_ids[c] for c in cols]
        new = sweep_prism_to_tets(cols, lists, [key[l] for l in lists])
        tets.extend(new)
        owner.extend([t] * len(new))
    tets = np.asarray(tets, np.int64)
    owner = np.asarray(owner, np.int64)

    p = coords[tets]
    vol = np.einsum('ij,ij->i',
                    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0]) / 6.0
    if not np.all(np.abs(vol) > 0):
        bad_t = np.unique(owner[vol == 0])
        q = xy[tris[bad_t]]
        area = 0.5 * np.abs((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                            - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))
        raise ValueError(
            f"{int((vol == 0).sum())} tets have zero volume, in plan triangles "
            f"{bad_t[:10].tolist()} (plan areas {np.round(area[:10], 3).tolist()}, "
            f"centroids {np.round(q.mean(1)[:3]).tolist()})")
    flip = vol < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]

    bfaces, bowner = _derive_boundary_with_owner(tets)
    fcol = node_col[bfaces]
    bottom = np.all(node_level[bfaces] == (per - 1)[fcol], axis=1)
    ncols = np.array([len(set(r)) for r in fcol.tolist()])
    kind = np.full(bfaces.shape[0], SIDE, dtype=np.int8)
    kind[(ncols == 3) & bottom] = BOTTOM
    kind[(ncols == 3) & ~bottom] = TOP
    if front.any():
        # Side faces lying wholly in the upper part, across a plan edge that
        # is interior (shared by two triangles), are the calving front.
        e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                                    tris[:, [2, 0]]]), axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        interior = {tuple(x) for x in ue[cnt == 2].tolist()}
        upper_face = np.all(k[bfaces] <= 0, axis=1) & np.any(k[bfaces] < 0, axis=1)
        for i in np.flatnonzero((kind == SIDE) & upper_face):
            pair = tuple(sorted(set(fcol[i].tolist())))
            if pair in interior:
                kind[i] = CALVING
    from_ice = ice_tri[owner[bowner]] if tri_is_ice is not None else None
    return StackedMesh(coords, tets, bfaces, kind, L, C, node_col, node_level, n,
                       from_ice)


def _derive_boundary_with_owner(tets):
    """Like _derive_boundary, but also return the owning tet of each face."""
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                            tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]])
    owner = np.tile(np.arange(len(tets)), 4)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError(f"{int((counts > 2).sum())} faces are shared by more "
                         "than two tets -- the prism split is not conforming")
    keep = first[counts == 1]
    return faces[keep], owner[keep]
