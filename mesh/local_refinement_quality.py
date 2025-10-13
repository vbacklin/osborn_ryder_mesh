#!/usr/bin/env python3
# local_refinement_quality.py
#
# Local refinement by splitting the worst tetrahedra after inserting interior
# centroid nodes.
#
# Usage:
#   python local_refinement_quality.py input.msh -o refined.msh --iters 8 --worst-frac 0.01 \
#          --hmin 0.15 --min-improve 1e-4
#
import argparse
import math
from statistics import median
import gmsh

# ---------- small helpers ----------

def _try_set_option(name, value):
    try:
        gmsh.option.setNumber(name, value); return True
    except Exception:
        return False

def _edge_lengths(pts):
    L=[]
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            L.append(math.dist(pts[i], pts[j]))
    return L

def _triangle_quality(p1,p2,p3):
    a=math.dist(p2,p3); b=math.dist(p1,p3); c=math.dist(p1,p2)
    s=0.5*(a+b+c)
    A2=max(s*(s-a)*(s-b)*(s-c),0.0)
    if A2<=0: return 0.0
    A=math.sqrt(A2)
    d=a*a+b*b+c*c
    return 0.0 if d<=0 else min(1.0,(2.0*math.sqrt(3.0)*A)/d)

def _tet_quality(p1,p2,p3,p4):
    import numpy as np
    pts=[p1,p2,p3,p4]
    e2=sum(l*l for l in _edge_lengths(pts))
    a,b,c,d = map(np.array, pts)
    V=abs(np.dot(a-d, np.cross(b-d, c-d)))/6.0
    if e2<=0: return 0.0
    q = 12.0*(3.0*V)**(2.0/3.0)/e2
    return min(1.0, max(0.0, q))

def _count_elements(dim):
    _t, eByT, _n = gmsh.model.mesh.getElements(dim)
    return sum(len(a) for a in eByT)

def _detect_dim(announce=True):
    n3=_count_elements(3); n2=_count_elements(2); n1=_count_elements(1)
    dim = 3 if n3 else 2 if n2 else 1 if n1 else 0
    if announce:
        print(f"Element counts — dim3:{n3}, dim2:{n2}, dim1:{n1} -> using dim={dim}" if dim else
              f"Element counts — dim3:{n3}, dim2:{n2}, dim1:{n1} -> no elements")
    return dim

def _node_coord_map():
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    return {int(nt):(nodeCoords[3*i], nodeCoords[3*i+1], nodeCoords[3*i+2])
            for i, nt in enumerate(nodeTags)}

def _barycenter(pts):
    n=len(pts); return (sum(p[0] for p in pts)/n, sum(p[1] for p in pts)/n, sum(p[2] for p in pts)/n)

def _tet_signed_volume(p0, p1, p2, p3):
    ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    cx, cy, cz = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
    return (
        ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
    )

def _orient_positive_tet(nodes, coord_map):
    n0, n1, n2, n3 = nodes
    p0, p1, p2, p3 = coord_map[n0], coord_map[n1], coord_map[n2], coord_map[n3]
    vol = _tet_signed_volume(p0, p1, p2, p3)
    if vol < 0.0:
        nodes = (n0, n1, n3, n2)
    return nodes

def _split_tet_with_centroid(node_ids, centroid_tag, centroid_coord, coord_map):
    coord_map = dict(coord_map)
    coord_map[centroid_tag] = centroid_coord
    a, b, c, d = node_ids
    raw = [
        (b, c, d, centroid_tag),
        (a, d, c, centroid_tag),
        (a, b, d, centroid_tag),
        (a, c, b, centroid_tag),
    ]
    oriented=[]
    for tet in raw:
        oriented.append(_orient_positive_tet(tuple(tet), coord_map))
    return oriented

# ---------- quality & worst elements ----------

def compute_qualities(dim):
    types, tagsByT, nodesByT = gmsh.model.mesh.getElements(dim)

    # Native per-type qualities
    try:
        all_tags=[]; all_q=[]
        for et, tags in zip(types, tagsByT):
            try:
                q = gmsh.model.mesh.getElementQualities(et)
                if len(q)!=len(tags): raise RuntimeError
                all_tags.extend(tags); all_q.extend(q)
            except Exception:
                raise
        if all_tags: return list(all_tags), list(all_q)
    except Exception:
        pass

    # Whole-mesh variant
    try:
        q = gmsh.model.mesh.getElementQualities()
        all_tags=[t for arr in tagsByT for t in arr]
        if all_tags and len(q)==len(all_tags): return list(all_tags), list(q)
    except Exception:
        pass

    # Fallback: tri/tet exact-ish, others neutral
    coord=_node_coord_map()
    all_tags=[]; all_q=[]
    for et, tags, nds in zip(types, tagsByT, nodesByT):
        tags=list(tags)
        _,_,_, nPer, _, _ = gmsh.model.mesh.getElementProperties(et)
        is_tri=(nPer==3 and dim==2)
        is_tet=(nPer==4 and dim==3)
        for e in range(len(tags)):
            block=nds[e*nPer:(e+1)*nPer]
            pts=[coord[int(nid)] for nid in block]
            if is_tri: q=_triangle_quality(*pts)
            elif is_tet: q=_tet_quality(*pts)
            else: q=0.5
            all_tags.append(int(tags[e])); all_q.append(q)
    return all_tags, all_q

def worst_elements_geometry(dim, frac):
    coord=_node_coord_map()
    types, tagsByT, ndsByT = gmsh.model.mesh.getElements(dim)
    elem_tags, q = compute_qualities(dim)
    if not elem_tags: return [], [], [], [], []

    # map elemTag->(centroid, avgEdge)
    info={}
    for et, tags, nds in zip(types, tagsByT, ndsByT):
        tags=list(tags)
        _,_,_, nPer, _, _ = gmsh.model.mesh.getElementProperties(et)
        for e in range(len(tags)):
            t=int(tags[e])
            block=nds[e*nPer:(e+1)*nPer]
            pts=[coord[int(nid)] for nid in block]
            bc=_barycenter(pts)
            L=_edge_lengths(pts); h=sum(L)/len(L) if L else 0.0
            info[t]=(bc, h, [int(nid) for nid in block])

    n=len(elem_tags); k=max(1,int(math.ceil(frac*n)))
    worst_idx=sorted(range(n), key=lambda i:q[i])[:k]
    worst_tags=[int(elem_tags[i]) for i in worst_idx]
    centers=[info[t][0] for t in worst_tags]
    scales=[max(info[t][1],1e-12) for t in worst_tags]
    node_sets=[info[t][2] for t in worst_tags]
    return worst_tags, centers, scales, node_sets, q

# ---------- local refinement: split tetrahedra ----------

def local_refine_iter_by_points(dim, worst_frac, hmin, verbose=True, use_ball_field=False, ball_mult=2.0):
    _ = (hmin, ball_mult)  # retained for CLI compatibility
    worst, centers, _scales, _node_sets, q = worst_elements_geometry(dim, worst_frac)
    if not worst:
        raise RuntimeError("No elements to refine.")
    if verbose:
        print(f"  splitting {len(centers)} tetrahedra by inserting centroid nodes")
        if use_ball_field:
            print("  (ignoring --use-ball-field for direct splitting refinement)")

    coord_map = _node_coord_map()
    if not coord_map:
        raise RuntimeError("Failed to retrieve node coordinates.")
    max_node_tag = max(coord_map.keys())

    new_nodes_by_vol = {}
    new_tets_by_vol = {}
    remove_by_vol = {}
    skipped = 0

    for idx, elem_tag in enumerate(worst):
        elem_type, elem_nodes, elem_dim, vol_tag = gmsh.model.mesh.getElement(int(elem_tag))
        if elem_dim != 3 or elem_type != 4:
            skipped += 1
            continue

        elem_nodes = [int(n) for n in elem_nodes]
        center = centers[idx]
        max_node_tag += 1
        centroid_tag = max_node_tag

        new_nodes_by_vol.setdefault(vol_tag, []).append((centroid_tag, center))
        remove_by_vol.setdefault(vol_tag, []).append(int(elem_tag))
        sub_tets = _split_tet_with_centroid(elem_nodes, centroid_tag, center, coord_map)
        new_tets_by_vol.setdefault(vol_tag, []).extend(sub_tets)

        coord_map[centroid_tag] = center  # keep map consistent for subsequent orientation

    if skipped and verbose:
        print(f"  skipped {skipped} elements that are not 4-node tetrahedra")

    for vol_tag, elem_tags in remove_by_vol.items():
        gmsh.model.mesh.removeElements(3, vol_tag, elem_tags)

    for vol_tag, nodes in new_nodes_by_vol.items():
        if not nodes:
            continue
        node_tags = [tag for tag, _ in nodes]
        coords_flat = []
        for _, (x, y, z) in nodes:
            coords_flat.extend([x, y, z])
        gmsh.model.mesh.addNodes(3, vol_tag, node_tags, coords_flat)

    for vol_tag, tets in new_tets_by_vol.items():
        if not tets:
            continue
        nodes_flat = []
        for tet in tets:
            nodes_flat.extend(tet)
        gmsh.model.mesh.addElementsByType(vol_tag, 4, [], nodes_flat)

    try:
        gmsh.model.mesh.reclassifyNodes()
    except Exception:
        pass

    return q  # previous qualities (for ∆median)

# ---------- driver ----------

def run(input_path, output_path, iters, worst_frac, min_improve, hmin, dim_opt, quiet,
        use_ball_field, point_radius_mult):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # Make sure point sizing is actually used (ignore if missing on old builds)
    _try_set_option("Mesh.CharacteristicLengthFromPoints", 0)
    _try_set_option("Mesh.CharacteristicLengthFromCurvature", 0)
    _try_set_option("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    try:
        gmsh.open(input_path)

        dim = dim_opt if dim_opt else _detect_dim(announce=not quiet)
        if dim != 3:
            # We can also do 2D, but your use-case is 3D; keep simple
            raise RuntimeError("Expecting a 3D mesh; found dim != 3.")
        if not quiet:
            print(f"Loaded '{input_path}'. Working in dim={dim}.")

        # baseline
        tags, q = compute_qualities(3)
        if not tags: raise RuntimeError("Could not read 3D elements/qualities.")
        q_med=median(q); q_min=min(q)
        if not quiet:
            print(f"[iter 0] elements={len(tags)}, minQ={q_min:.6f}, medianQ={q_med:.6f}")

        for it in range(1, iters+1):
            if not quiet:
                n_worst=int(max(1, math.ceil(worst_frac*len(tags))))
                print(f"[iter {it}] local refinement of worst {n_worst} elements...")
            prev_med=q_med

            # do one refinement round via embedded points (+ optional ball field)
            local_refine_iter_by_points(3, worst_frac, hmin, verbose=not quiet,
                                        use_ball_field=use_ball_field, ball_mult=point_radius_mult)

            # recompute stats
            tags, q = compute_qualities(3)
            if not tags:
                gmsh.model.mesh.renumberNodes()
                tags, q = compute_qualities(3)
            if not tags: raise RuntimeError("Empty 3D element list after remesh.")
            q_med=median(q); q_min=min(q)
            dmed=q_med-prev_med
            if not quiet:
                print(f"  elements={len(tags)}, minQ={q_min:.6f}, medianQ={q_med:.6f}, Δmedian={dmed:.3e}")

            if dmed < min_improve:
                if not quiet:
                    print(f"[iter {it}] Early stop: Δmedian {dmed:.3e} < {min_improve:.3e}")
                # if min_improve is negative (your test), we never early-stop
                if min_improve >= 0: break

        gmsh.write(output_path)
        if not quiet: print(f"Wrote '{output_path}'.")

    finally:
        gmsh.finalize()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Local quality-driven refinement by embedding interior size points.")
    ap.add_argument("input", help=".msh file to load")
    ap.add_argument("-o","--output", default="refined.msh", help="output .msh file")
    ap.add_argument("--iters", type=int, default=8, help="max iterations")
    ap.add_argument("--worst-frac", type=float, default=0.01, help="fraction of worst elements per iter")
    ap.add_argument("--min-improve", type=float, default=1e-4, help="early stop on median gain; set <0 to disable")
    ap.add_argument("--hmin", type=float, default=0.15, help="target element size near bad elements")
    ap.add_argument("--dim", type=int, choices=[3], default=None, help="force 3D (expects 3D mesh)")
    # optional: widen influence with a ball field as well
    ap.add_argument("--use-ball-field", action="store_true", help="also add Ball size fields around points")
    ap.add_argument("--point-radius-mult", type=float, default=2.0, help="Ball radius multiplier if --use-ball-field")
    ap.add_argument("--quiet", action="store_true", help="less console output")
    args = ap.parse_args()

    run(args.input, args.output, args.iters, args.worst_frac, args.min_improve,
        args.hmin, args.dim, args.quiet, args.use_ball_field, args.point_radius_mult)

if __name__ == "__main__":
    main()
