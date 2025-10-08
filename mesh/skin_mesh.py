"""Extract the boundary surface of a 3D Gmsh mesh and emit FEniCS-friendly output.

The original script rebuilt the boundary with a secondary meshing pass, which
introduced mixed element types (triangles and quads).  Mixed surface cells lead
to ``dolfin`` raising ``Unknown value "mixed"`` when reading the generated
XDMF file.  This rewrite keeps only the existing triangular boundary elements,
repackages them into a dedicated surface model, and converts the result to
XDMF with strictly single-type cells.

Usage
-----
    python mesh/skin_mesh.py [input_mesh.msh] [--out skin_clean]

The script writes two files next to the output prefix:

``<prefix>.msh``
    Gmsh surface mesh containing only first-order triangles.

``<prefix>.xdmf`` (+ ``.h5`` sidecar)
    FEniCS-compatible mesh generated via ``meshio``.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gmsh
import meshio
import numpy as np

# Tag used when the input mesh does not carry any surface physical groups.
DEFAULT_SURFACE_TAG = 999


def _to_list(values: Iterable[int]) -> List[int]:
    """Return ``values`` as a plain Python ``list`` of ints."""

    if hasattr(values, "tolist"):
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def _collect_surface_groups() -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[str, Tuple[int, int]], int]:
    """Return surface→physical group mapping and field data metadata."""

    surface_groups: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    field_data: Dict[str, Tuple[int, int]] = {}
    used_tags: set[int] = set()

    for dim, phys_tag in gmsh.model.getPhysicalGroups(2):
        phys_tag = int(phys_tag)
        used_tags.add(phys_tag)
        try:
            name = gmsh.model.getPhysicalName(dim, phys_tag)
        except RuntimeError:
            name = ""
        if not name:
            name = f"PhysicalGroup_{phys_tag}"

        base_name = name
        suffix = 1
        while name in field_data and field_data[name][0] != phys_tag:
            name = f"{base_name}_{suffix}"
            suffix += 1

        field_data[name] = (phys_tag, dim)

        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
        entity_list = entities.tolist() if hasattr(entities, "tolist") else list(entities)
        for entity in entity_list:
            surface_groups[int(entity)].append((phys_tag, name))

    return surface_groups, field_data, max(used_tags) if used_tags else 0


def _gather_boundary_triangles(
    coord_lookup: Dict[int, Tuple[float, float, float]]
) -> Tuple[
    List[int],
    List[List[int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Tuple[int, int]],
]:
    """Collect boundary triangles and physical tags from the current Gmsh model."""

    surface_groups, field_data, max_used_tag = _collect_surface_groups()
    default_tag = DEFAULT_SURFACE_TAG
    if default_tag <= max_used_tag:
        default_tag = max_used_tag + 1

    default_name = "SkinSurface"
    base_default_name = default_name
    suffix = 1
    while default_name in field_data:
        default_name = f"{base_default_name}_{suffix}"
        suffix += 1

    triangles_nodes: List[List[int]] = []
    triangles_physical: List[int] = []
    triangles_geometrical: List[int] = []
    triangles_centers: List[Tuple[float, float, float]] = []
    node_tags_set: set[int] = set()
    seen_elements: set[int] = set()
    default_used = False

    volumes = gmsh.model.getEntities(3)
    surface_candidates = gmsh.model.getBoundary(volumes, oriented=False, recursive=False)

    for _, surface_tag in surface_candidates:
        elem_types, elem_tags_list, elem_nodes_list = gmsh.model.mesh.getElements(2, surface_tag)
        for elem_type, elem_tags, elem_nodes in zip(elem_types, elem_tags_list, elem_nodes_list):
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
            if num_nodes != 3:
                # Drop quads or high-order polygons; FEniCS expects simplices.
                continue

            tags = _to_list(elem_tags)
            nodes = _to_list(elem_nodes)

            if not tags:
                continue

            for local_idx, tag in enumerate(tags):
                if tag in seen_elements:
                    continue
                seen_elements.add(tag)

                start = local_idx * num_nodes
                tri_nodes = nodes[start:start + num_nodes]
                node_tags_set.update(tri_nodes)

                phys_entries = surface_groups.get(int(surface_tag))
                if phys_entries:
                    if len(phys_entries) > 1:
                        # If a surface belongs to multiple physical groups, keep the first.
                        phys_tag = phys_entries[0][0]
                    else:
                        phys_tag = phys_entries[0][0]
                else:
                    phys_tag = default_tag
                    default_used = True

                triangles_nodes.append([int(n) for n in tri_nodes])
                triangles_physical.append(int(phys_tag))
                triangles_geometrical.append(int(surface_tag))

                centroid = np.mean([coord_lookup[int(n)] for n in tri_nodes], axis=0)
                triangles_centers.append(tuple(float(v) for v in centroid))

    if not triangles_nodes:
        raise RuntimeError("No triangular surface elements detected in the source mesh.")

    node_tags = sorted(node_tags_set)
    if default_used:
        field_data.setdefault(default_name, (default_tag, 2))
    else:
        # Remove default tag entry if unused.
        field_data = {name: value for name, value in field_data.items() if value[0] != default_tag}

    return (
        node_tags,
        triangles_nodes,
        np.array(triangles_physical, dtype=np.int32),
        np.array(triangles_geometrical, dtype=np.int32),
        np.array(triangles_centers, dtype=float),
        field_data,
    )


def _build_discrete_surface(
    node_tags: List[int],
    coord_lookup: Dict[int, Tuple[float, float, float]],
    triangles_nodes: List[List[int]],
) -> None:
    """Create a discrete surface model in Gmsh with the provided triangles."""

    gmsh.model.add("skin_surface")

    surface_tag = 1
    triangle_type = gmsh.model.mesh.getElementType("Triangle", 1)

    coords_flat: List[float] = []
    for tag in node_tags:
        coords_flat.extend(coord_lookup[int(tag)])

    gmsh.model.addDiscreteEntity(2, surface_tag)
    gmsh.model.mesh.addNodes(2, surface_tag, node_tags, coords_flat)

    elem_tags = list(range(1, len(triangles_nodes) + 1))
    elem_nodes_flat: List[int] = []
    for tri in triangles_nodes:
        elem_nodes_flat.extend(tri)

    gmsh.model.mesh.addElements(
        2,
        surface_tag,
        [triangle_type],
        [elem_tags],
        [elem_nodes_flat],
    )


def _remesh_surface(mesh_size: float | None, angle: float, curve_angle: float) -> None:
    """Run a surface re-meshing pass on the current Gmsh model."""

    include_boundary = True
    force_param = True

    gmsh.model.mesh.classifySurfaces(
        math.radians(angle),
        include_boundary,
        force_param,
        math.radians(curve_angle),
    )
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    if mesh_size and mesh_size > 0:
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize(method="", niter=5)
    gmsh.model.mesh.optimize(method="Netgen", niter=1)


def _extract_remeshed_surface() -> Tuple[np.ndarray, np.ndarray]:
    """Extract node coordinates and triangle connectivity from the remeshed model."""

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    node_coords = node_coords[order]

    node_index = {int(tag): idx for idx, tag in enumerate(node_tags)}

    triangles: List[List[int]] = []

    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2)
    for elem_type, elem_nodes in zip(elem_types, elem_node_tags_list):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
        if num_nodes != 3:
            continue

        nodes = _to_list(elem_nodes)
        for idx in range(len(nodes) // num_nodes):
            start = idx * num_nodes
            tri = [node_index[int(nodes[start + k])] for k in range(num_nodes)]
            triangles.append(tri)

    if not triangles:
        raise RuntimeError("Remeshing did not produce any triangular elements.")

    return node_coords.astype(float), np.array(triangles, dtype=np.int64)


def _assign_tags(
    new_centers: np.ndarray,
    original_centers: np.ndarray,
    original_phys: np.ndarray,
    original_geom: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign physical and geometrical tags to remeshed triangles via nearest match."""

    phys_tags = np.empty(len(new_centers), dtype=np.int32)
    geom_tags = np.empty(len(new_centers), dtype=np.int32)

    batch_size = 256
    for start in range(0, len(new_centers), batch_size):
        end = min(len(new_centers), start + batch_size)
        batch = new_centers[start:end]
        # Compute squared distances in a memory-conscious way.
        diff = original_centers[None, :, :] - batch[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        indices = np.argmin(dist2, axis=1)
        phys_tags[start:end] = original_phys[indices]
        geom_tags[start:end] = original_geom[indices]

    return phys_tags, geom_tags


def _build_surface_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    phys_tags: np.ndarray,
    geom_tags: np.ndarray,
    field_data: Dict[str, Tuple[int, int]],
) -> meshio.Mesh:
    """Compose a meshio surface mesh with cell data and field definitions."""

    cell_data = {
        "gmsh:physical": [phys_tags.astype(np.int32)],
        "gmsh:geometrical": [geom_tags.astype(np.int32)],
        "name_to_read": [phys_tags.astype(np.int32)],
    }

    mesh_field_data = {
        name: np.array(value, dtype=np.int32)
        for name, value in field_data.items()
    }

    return meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data=cell_data,
        field_data=mesh_field_data,
    )


def build_skin_mesh(
    input_mesh: Path,
    output_prefix: Path,
    mesh_size: float | None,
    classify_angle: float,
    seam_angle: float,
) -> Tuple[Path, Path]:
    """Extract a boundary surface mesh and export both .msh and .xdmf flavours."""

    gmsh.initialize()
    try:
        gmsh.open(str(input_mesh))
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.removeDuplicateNodes()

        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        coord_lookup = {
            int(tag): (
                float(all_coords[3 * idx]),
                float(all_coords[3 * idx + 1]),
                float(all_coords[3 * idx + 2]),
            )
            for idx, tag in enumerate(all_tags)
        }

        (
            node_tags,
            triangles_nodes,
            triangles_physical,
            triangles_geometrical,
            triangles_centers,
            field_data,
        ) = _gather_boundary_triangles(coord_lookup)

        _build_discrete_surface(node_tags, coord_lookup, triangles_nodes)
        _remesh_surface(mesh_size, classify_angle, seam_angle)

        points, triangles = _extract_remeshed_surface()
        triangle_coords = points[triangles]
        new_centers = triangle_coords.mean(axis=1)

        phys_tags, geom_tags = _assign_tags(
            new_centers,
            triangles_centers,
            triangles_physical,
            triangles_geometrical,
        )

        surface_mesh = _build_surface_mesh(points, triangles, phys_tags, geom_tags, field_data)
    finally:
        gmsh.finalize()

    xdmf_path = output_prefix.with_suffix(".xdmf")
    msh_path = output_prefix.with_suffix(".msh")
    meshio.write(msh_path, surface_mesh, file_format="gmsh22")
    meshio.write(xdmf_path, surface_mesh, file_format="xdmf")

    return msh_path, xdmf_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_mesh",
        nargs="?",
        default=Path("mesh/stacked/P_10_560.msh"),
        type=Path,
        help="Path to the source 3D Gmsh mesh (defaults to mesh/stacked/P_10_560.msh).",
    )
    parser.add_argument(
        "--out",
        dest="output_prefix",
        default=Path("skin_clean"),
        type=Path,
        help="Output file prefix (without extension). Defaults to 'skin_clean'.",
    )
    parser.add_argument(
        "--mesh-size",
        dest="mesh_size",
        type=float,
        default=None,
        help="Target mesh size for the remeshed skin (set both min and max).",
    )
    parser.add_argument(
        "--classify-angle",
        dest="classify_angle",
        type=float,
        default=40.0,
        help="Feature angle (degrees) used when classifying surfaces.",
    )
    parser.add_argument(
        "--seam-angle",
        dest="seam_angle",
        type=float,
        default=180.0,
        help="Maximum dihedral angle (degrees) for seam detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    msh_path, xdmf_path = build_skin_mesh(
        args.input_mesh,
        args.output_prefix,
        mesh_size=args.mesh_size,
        classify_angle=args.classify_angle,
        seam_angle=args.seam_angle,
    )
    print(f"Wrote {msh_path}")
    print(f"Wrote {xdmf_path} (and companion HDF5 file)")


if __name__ == "__main__":
    main()
