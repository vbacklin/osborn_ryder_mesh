# refine_band.py
"""Refine a surface band of a (potentially 3D) Gmsh mesh and export a FEniCS-ready skin.

The script re-uses the original boundary triangles to preserve physical tags,
computes a distance/threshold size field along the detected boundary curves,
remeshes the surface, and finally writes a triangle-only `.msh` and `.xdmf`
pair that Dolfin/FEniCS can consume without the “mixed cell type” error.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import gmsh
import meshio
import numpy as np

# ----------------- user params -----------------
in_msh = sys.argv[1] if len(sys.argv) > 1 else "in2d.msh"
out_msh = sys.argv[2] if len(sys.argv) > 2 else "out2d_refined.msh"

# Physical groups to target (optional).
phys_line_targets: Sequence[int] = []     # Physical Line tags to follow (1D)
phys_surface_targets: Sequence[int] = []  # Physical Surface tags to follow (2D)

# Band + sizes (units match the mesh coordinates)
band_width_min = 0.0       # Distance from the boundary that keeps lc_min (0 => lc_min exactly on the boundary)
band_width_max = 4000.0     # Distance where size transitions back to lc_max (must be > band_width_min)
lc_min = 50.0             # Fine size inside the band
lc_max = 1000.0            # Default size far from the band

# Surface remeshing controls
feature_angle = 0.0        # Feature detection angle (degrees) (40 ?)
seam_angle = 90.0          # Seam detection angle (degrees) (180 ?)
distance_sampling = 5    # Sampling density for the distance field along curves
optimize_iterations = 1    # Laplacian smoothing iterations after remeshing (0 disables smoothing)
quality_target = 0.1       # Aim to raise the worst triangle quality above this threshold
quality_max_passes = 5     # Maximum improvement passes targeting poor-quality elements
# Optimizer sequence for surface cleanup. Keep methods widely supported
# across Gmsh versions to avoid "unknown method" errors.
quality_methods: Sequence[str] = ("Relocate2D", "Laplace2D")
# ------------------------------------------------



def _prepare_field_data(snapshot: List[Tuple[int, int, str, Iterable[int]]]) -> Dict[str, np.ndarray]:
    """Return meshio-style field data mapping physical names to (tag, dim).

    Includes both 1D (curves) and 2D (surfaces) PhysicalGroups so that
    the output file preserves those definitions, even if only triangles
    are exported as cells.
    """
    field: Dict[str, Tuple[int, int]] = {}
    for dim, tag, name, _ in snapshot:
        if dim not in (1, 2):
            continue
        if not name:
            name = f"PhysicalGroup_{tag}"
        base = name
        suffix = 1
        while name in field and (field[name][0] != tag or field[name][1] != dim):
            name = f"{base}_{suffix}"
            suffix += 1
        field[name] = (int(tag), dim)
    return {name: np.array(value, dtype=np.int32) for name, value in field.items()}


def _surface_physical_map(snapshot: List[Tuple[int, int, str, Iterable[int]]]) -> Dict[int, List[int]]:
    """Map surface entity tags to their physical tags."""
    mapping: Dict[int, List[int]] = {}
    for dim, tag, _, ents in snapshot:
        if dim != 2:
            continue
        if hasattr(ents, "tolist"):
            ent_iter = ents.tolist()
        else:
            ent_iter = list(ents)
        for ent in ent_iter:
            mapping.setdefault(int(ent), []).append(int(tag))
    return mapping


def _collect_triangle_data(
    surface_tags: Sequence[int],
    coord_lookup: Dict[int, Tuple[float, float, float]],
    surface_phys: Dict[int, List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect original triangle centroids and tags on the selected surfaces."""
    centers: List[Tuple[float, float, float]] = []
    phys_tags: List[int] = []
    geom_tags: List[int] = []

    for surface_tag in surface_tags:
        elem = gmsh.model.mesh.getElements(2, surface_tag)
        if not elem[0]:
            continue
        elem_types, elem_tags_list, elem_nodes_list = elem
        for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_nodes_list):
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
            if num_nodes not in (3, 4):
                continue
            tags = elem_tags.tolist() if hasattr(elem_tags, "tolist") else list(elem_tags)
            nodes = elem_nodes.tolist() if hasattr(elem_nodes, "tolist") else list(elem_nodes)
            for idx, _ in enumerate(tags):
                start = idx * num_nodes
                node_ids = [int(n) for n in nodes[start:start + num_nodes]]
                if num_nodes == 4:
                    tri_batches = [
                        node_ids[:3],
                        [node_ids[0], node_ids[2], node_ids[3]],
                    ]
                else:
                    tri_batches = [node_ids]
                for tri in tri_batches:
                    coords = np.array([coord_lookup[n] for n in tri], dtype=float)
                    centers.append(tuple(coords.mean(axis=0)))
                    phys_candidates = surface_phys.get(int(surface_tag), [])
                    phys_tag = phys_candidates[0] if phys_candidates else 0
                    phys_tags.append(int(phys_tag))
                    geom_tags.append(int(surface_tag))

    return (
        np.array(centers, dtype=float),
        np.array(phys_tags, dtype=np.int32),
        np.array(geom_tags, dtype=np.int32),
    )


def _assign_tags(
    new_centers: np.ndarray,
    base_centers: np.ndarray,
    base_phys: np.ndarray,
    base_geom: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign physical and geometrical tags to new triangles via nearest-centroid mapping."""
    if base_centers.size == 0:
        return (np.zeros(len(new_centers), dtype=np.int32), np.zeros(len(new_centers), dtype=np.int32))

    phys_out = np.empty(len(new_centers), dtype=np.int32)
    geom_out = np.empty(len(new_centers), dtype=np.int32)
    batch_size = 512
    base = base_centers
    for start in range(0, len(new_centers), batch_size):
        end = min(len(new_centers), start + batch_size)
        batch = new_centers[start:end]
        diff = base[None, :, :] - batch[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest = np.argmin(dist2, axis=1)
        phys_out[start:end] = base_phys[nearest]
        geom_out[start:end] = base_geom[nearest]
    return phys_out, geom_out


def _min_surface_quality() -> Optional[float]:
    """Return the minimum triangle 'equilateral' quality computed geometrically.

    Uses q = 4*sqrt(3)*A / (a^2 + b^2 + c^2) in [0, 1].
    Quads are split into two triangles consistently with export.
    """
    def tri_quality(p0, p1, p2) -> float:
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)
        a = float(np.linalg.norm(v1))
        b = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        c = float(np.linalg.norm(np.array(p0) - np.array(p2)))
        area = 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
        denom = a*a + b*b + c*c
        if denom <= 1e-30:
            return 0.0
        return float(4.0 * math.sqrt(3.0) * area / denom)

    try:
        all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
    except Exception as err:
        print(f"[refine_band] Unable to access nodes for quality check: {err}", file=sys.stderr)
        return None

    coord_lookup = {
        int(tag): (
            float(all_node_coords[3 * idx]),
            float(all_node_coords[3 * idx + 1]),
            float(all_node_coords[3 * idx + 2]),
        )
        for idx, tag in enumerate(all_node_tags)
    }

    try:
        elem_types, elem_tags_list, elem_nodes_list = gmsh.model.mesh.getElements(2)
    except Exception as err:
        print(f"[refine_band] Unable to list 2D elements: {err}", file=sys.stderr)
        return None

    found_any = False
    worst = float("inf")

    for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_nodes_list):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
        tags = elem_tags.tolist() if hasattr(elem_tags, "tolist") else list(elem_tags)
        nodes = elem_nodes.tolist() if hasattr(elem_nodes, "tolist") else list(elem_nodes)
        for idx in range(len(tags)):
            start = idx * num_nodes
            node_ids = [int(n) for n in nodes[start:start + num_nodes]]
            if num_nodes == 3:
                tri_batches = [node_ids]
            elif num_nodes == 4:
                tri_batches = [node_ids[:3], [node_ids[0], node_ids[2], node_ids[3]]]
            else:
                continue
            for tri in tri_batches:
                try:
                    p0 = coord_lookup[tri[0]]
                    p1 = coord_lookup[tri[1]]
                    p2 = coord_lookup[tri[2]]
                except KeyError:
                    continue
                q = tri_quality(p0, p1, p2)
                found_any = True
                if q < worst:
                    worst = q

    return worst if found_any else None


def _improve_surface_quality(
    target: float,
    max_passes: int,
    methods: Sequence[str],
) -> Optional[float]:
    """Iteratively run Gmsh optimizers until the worst triangle meets `target`."""
    if target <= 0 or max_passes <= 0:
        return None

    worst = _min_surface_quality()
    if worst is None or worst >= target:
        return worst

    failed_methods: set[str] = set()

    for _ in range(int(max_passes)):
        if worst is not None and worst >= target:
            break
        for method in methods:
            if method in failed_methods:
                continue
            try:
                gmsh.model.mesh.optimize(method=method, niter=1)
            except Exception as err:
                # Gracefully skip unsupported optimizers instead of aborting.
                print(f"[refine_band] Skipping optimizer '{method}': {err}", file=sys.stderr)
                failed_methods.add(method)
        worst = _min_surface_quality()
        if worst is None:
            break

    return worst


def _curves_from_surfaces(surface_tags: Sequence[int]) -> List[int]:
    """Return all boundary curves for the provided surfaces."""
    curves: set[int] = set()
    for surface_tag in surface_tags:
        boundary = gmsh.model.getBoundary([(2, int(surface_tag))], oriented=False, recursive=False)
        for dim, curve in boundary:
            if dim == 1:
                curves.add(int(curve))
    return sorted(curves)


def _curves_from_physical_lines(ptags: Sequence[int]) -> List[int]:
    curves: List[int] = []
    for tag in ptags:
        for ent in gmsh.model.getEntitiesForPhysicalGroup(1, int(tag)):
            curves.append(int(ent))
    return sorted(set(curves))


def _surfaces_from_physical_surfaces(ptags: Sequence[int]) -> List[int]:
    surfaces: List[int] = []
    for tag in ptags:
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, int(tag)):
            surfaces.append(int(ent))
    return sorted(set(surfaces))


gmsh.initialize()
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.open(in_msh)

# Take a physical snapshot before altering the model.
phys_snapshot: List[Tuple[int, int, str, Iterable[int]]] = []
for dim, ptag in gmsh.model.getPhysicalGroups():
    name = gmsh.model.getPhysicalName(dim, ptag)
    ents = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
    phys_snapshot.append((dim, ptag, name, ents))

field_data = _prepare_field_data(phys_snapshot)
surface_phys_map = _surface_physical_map(phys_snapshot)

# Determine surfaces to track.
surface_tags_initial = _surfaces_from_physical_surfaces(phys_surface_targets)
if not surface_tags_initial:
    volumes = gmsh.model.getEntities(3)
    if volumes:
        boundary = gmsh.model.getBoundary(volumes, oriented=False, recursive=False)
        surface_tags_initial = sorted({tag for dim, tag in boundary if dim == 2})
    else:
        surface_tags_initial = [tag for (dim, tag) in gmsh.model.getEntities(2)]

if not surface_tags_initial:
    gmsh.finalize()
    raise RuntimeError("Unable to detect any surface entities to refine.")

# Original triangle centroids and tags for later mapping.
all_node_tags0, all_node_coords0, _ = gmsh.model.mesh.getNodes()
coord_lookup0 = {
    int(tag): (
        float(all_node_coords0[3 * idx]),
        float(all_node_coords0[3 * idx + 1]),
        float(all_node_coords0[3 * idx + 2]),
    )
    for idx, tag in enumerate(all_node_tags0)
}

base_centers, base_phys, base_geom = _collect_triangle_data(
    surface_tags_initial,
    coord_lookup0,
    surface_phys_map,
)

# Remeshing preparation.
gmsh.model.mesh.classifySurfaces(
    math.radians(feature_angle),
    True,
    True,
    math.radians(seam_angle),
)
gmsh.model.mesh.createGeometry()
gmsh.model.geo.synchronize()

# --- Build a boundary-aware size field -----------------------------------
# 1. Collect the curve entities that bound the target surfaces. We prefer
#    user-provided Physical Line tags but fall back to the geometric
#    boundary of the selected surfaces.
curve_tags = _curves_from_physical_lines(phys_line_targets)
if not curve_tags:
    volumes = gmsh.model.getEntities(3)
    if volumes:
        boundary = gmsh.model.getBoundary(volumes, oriented=False, recursive=False)
        surface_tags_after = sorted({tag for dim, tag in boundary if dim == 2})
    else:
        surface_tags_after = [tag for (dim, tag) in gmsh.model.getEntities(2)]
    curve_tags = _curves_from_surfaces(surface_tags_after)

if not curve_tags:
    gmsh.finalize()
    raise RuntimeError("Unable to detect boundary curves for the refinement band.")

# 2. Evaluate the signed distance to those curves. The field sampling governs
#    how finely the distance is interpolated along each curve; increase
#    `distance_sampling` if you need a sharper transition.
distance_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", curve_tags)
gmsh.model.mesh.field.setNumber(distance_field, "Sampling", distance_sampling)

# 3. Map distance -> element size with a Threshold field. Distances smaller
#    than `inner_band` use `lc_min`, distances larger than `outer_band`
#    use `lc_max`, and the values in between are linearly interpolated.
inner_band = max(0.0, float(band_width_min))  # 0 => lc_min starts exactly on the boundary
outer_band = float(band_width_max)
if outer_band <= inner_band:
    gmsh.finalize()
    raise ValueError("band_width_max must be greater than band_width_min to create a transition zone.")

threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", lc_min)
gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", lc_max)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", inner_band)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", outer_band)

# 4. Install the size field as the new background field so the remesher
#    honors the boundary band transition. We disable the default
#    characteristic length heuristics so only our size field is used.
gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

# Generate the new surface mesh.
gmsh.model.mesh.generate(2)

# Target poor-quality triangles before any global smoothing to avoid
# segfaults observed when running optimizers on degenerate elements.
final_quality = _improve_surface_quality(quality_target, quality_max_passes, quality_methods)

# Optionally run an additional smoothing pass (typically Laplacian).
if optimize_iterations > 0:
    gmsh.model.mesh.optimize(method="Laplace2D", niter=int(optimize_iterations))

if final_quality is not None:
    print(f"Surface min quality after targeted optimization: {final_quality:.3f}")

# Extract the refined mesh.
all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
coord_lookup = {
    int(tag): (
        float(all_node_coords[3 * idx]),
        float(all_node_coords[3 * idx + 1]),
        float(all_node_coords[3 * idx + 2]),
    )
    for idx, tag in enumerate(all_node_tags)
}

triangles: List[List[int]] = []
centers: List[Tuple[float, float, float]] = []

for _, surface_tag in gmsh.model.getEntities(2):
    elem_types, elem_tags_list, elem_nodes_list = gmsh.model.mesh.getElements(2, surface_tag)
    for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_nodes_list):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
        if num_nodes not in (3, 4):
            continue
        tags = elem_tags.tolist() if hasattr(elem_tags, "tolist") else list(elem_tags)
        nodes = elem_nodes.tolist() if hasattr(elem_nodes, "tolist") else list(elem_nodes)
        for idx in range(len(tags)):
            start = idx * num_nodes
            node_ids = [int(n) for n in nodes[start:start + num_nodes]]
            if num_nodes == 4:
                tri_batches = [
                    node_ids[:3],
                    [node_ids[0], node_ids[2], node_ids[3]],
                ]
            else:
                tri_batches = [node_ids]
            for tri in tri_batches:
                triangles.append(tri)
                coords = np.array([coord_lookup[n] for n in tri], dtype=float)
                centers.append(tuple(coords.mean(axis=0)))

if not triangles:
    gmsh.finalize()
    raise RuntimeError("Remeshing produced no surface triangles.")

new_centers = np.array(centers, dtype=float)
phys_tags, geom_tags = _assign_tags(new_centers, base_centers, base_phys, base_geom)

unique_nodes = sorted({node for tri in triangles for node in tri})
node_index = {tag: idx for idx, tag in enumerate(unique_nodes)}
points = np.array([coord_lookup[tag] for tag in unique_nodes], dtype=float)
tri_connectivity = np.array([[node_index[n] for n in tri] for tri in triangles], dtype=np.int64)

cell_data = {
    "gmsh:physical": [phys_tags.astype(np.int32)],
    "gmsh:geometrical": [geom_tags.astype(np.int32)],
    "name_to_read": [phys_tags.astype(np.int32)],
}

surface_mesh = meshio.Mesh(
    points=points,
    cells=[("triangle", tri_connectivity)],
    cell_data=cell_data,
    field_data=field_data,
)

out_path = Path(out_msh)
out_path.parent.mkdir(parents=True, exist_ok=True)
meshio.write(out_path, surface_mesh, file_format="gmsh22")
meshio.write(out_path.with_suffix(".xdmf"), surface_mesh, file_format="xdmf")

gmsh.finalize()
print(f"Wrote {out_path} and {out_path.with_suffix('.xdmf')}")
