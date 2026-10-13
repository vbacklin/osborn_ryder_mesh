# Ryder mesh script: user guide

Oct 13, 2026

## What the script does

`mesh/ryder_3D_meshing_manylayers.py` builds a tetrahedral mesh of the Sherard Osborn / Ryder fjord water body, including the cavity under the ice shelf, for FEniCS. Run with no settings, it produces the current default mesh: about 218,000 nodes and 1.23 million tetrahedra, in real coordinates (metres, z up, sea surface at z = 0).

How the mesh is built:

1. The domain outline is traced from the rasters: water deeper than a threshold, pulled a little inward from the fjord walls.
2. A 2D triangle mesh of that outline is made with gmsh, finer under the ice shelf and along the grounding line.
3. Every 2D node becomes a vertical column from the top (sea surface or ice base) down to the sea floor, split into layers. The layer stack is turned into tetrahedra directly, so no 3D mesher (TetGen) is involved.
4. The boundary faces are tagged (ice, sea floor and walls, ocean inflow, open water surface) and written as `.msh` plus two XDMF files.

It reads these rasters (150 m grid, EPSG:3413) and never modifies them; working copies go to `mesh/new_data/`:

| File | Used for |
| --- | --- |
| `data/surface_pos.tif` | ice surface elevation |
| `data/thickness.tif` | ice thickness (ice base = surface − thickness) |
| `data/elevation.tif` | bed / sea floor (150 m) |
| `data/sherard-osborn-fjord-15m-3996.tiff` | high-resolution sea floor (15 m) where available |
| `data/gis_data/categories.tif` | 0 water, 2 grounded ice, 3 floating ice shelf, 6 open ocean beyond the inflow |

## How to run it

Run it from the `mesh/` directory, inside the conda environment that has the packages below. GDAL needs the two PROJ variables to point at that environment's `share/proj` folder. Often conda's `proj` package sets them on activation; set them only if you get a PROJ error such as "proj.db not found":

```bash
cd osborn_ryder_mesh/mesh
conda activate <your-env>
export PROJ_DATA=$CONDA_PREFIX/share/proj
export PROJ_LIB=$PROJ_DATA
python -u ryder_3D_meshing_manylayers.py
```

Every setting is an environment variable, put in front of the command (all on one line):

```bash
NUM_LAYERS=20 ELEMENT_SIZE=1000 python -u ryder_3D_meshing_manylayers.py
```

- **Runtime:** about 10 s for the coarse test mesh and under a minute for the default. Memory stays under 3 GB.
- **Repeatable:** the same settings always give the same mesh, byte for byte.
- **Use `python -u`:** output is unbuffered, so if anything crashes you see how far it got.
- **One run at a time:** every run rewrites `mesh/new_data/`, so don't start two in parallel.
- **Python packages:** `gmsh` (4.13), `meshio`, `numpy`, `matplotlib`, and GDAL (`osgeo`).

## Output files

Each run writes three files into `mesh/`, all sharing one name that encodes the settings:

| File | Contents |
| --- | --- |
| `<name>.msh` | the full mesh for gmsh (version 2.2): tetrahedra plus tagged boundary triangles |
| `<name>_volume.xdmf` (+ `.h5`) | tetrahedra only, cell array `name_to_read` = 5 |
| `<name>_surface.xdmf` (+ `.h5`) | boundary triangles only, cell array `name_to_read` = boundary tag |

The default name is `stacked_layers38_t30_s75_c0_h500_m300_dz10_top5_hr250-125`. Each part records a setting from the parameter table below: `layers38` = `NUM_LAYERS`, `t30` = `SHORELINE_MIN_THICKNESS`, `s75` = `SHELF_MIN_THICKNESS`, `h500` = `ELEMENT_SIZE`, `m300` = `SHORELINE_MARGIN`, `dz10` = `MIN_LAYER_THICKNESS`, `top5` = `ICE_TOP_LAYER`, `hr250-125` = ice and grounding-line element sizes. A part is left out when the setting is off. Same settings, same name, so a rerun overwrites the old files.

**Boundary tags** (`gmsh:physical` in the `.msh`, `name_to_read` in the XDMF):

| Tag | Name | Where |
| --- | --- | --- |
| 1 | Ice | underside of the ice shelf, and the vertical calving front |
| 2 | Bathymetry | sea floor and the side walls along the shore and the grounding line |
| 4 | Inflow | vertical open-ocean boundary at the fjord mouth (and a small side channel) |
| 5 | Water Surface | open water surface at z = 0 |
| 5 (volume) | Water | all tetrahedra |

In legacy FEniCS:

```python
import dolfin as df
name = "stacked_layers38_t30_s75_c0_h500_m300_dz10_top5_hr250-125"
mesh = df.Mesh()
with df.XDMFFile(name + "_volume.xdmf") as f:
    f.read(mesh)
mvc = df.MeshValueCollection("size_t", mesh, 2)
with df.XDMFFile(name + "_surface.xdmf") as f:
    f.read(mvc, "name_to_read")
boundaries = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)   # ds(1) = ice, ds(2) = bed, ...
```

`poisson/model_poisson.py` does the same split itself, starting from the `.msh`. To look at a mesh in ParaView, open `_surface.xdmf` (or `_volume.xdmf`, never the `.h5`) and colour by `name_to_read`.

## Parameters

All parameters are environment variables with sensible defaults; set only the ones you want to change. Lengths are in metres. For on/off switches, 1 = on, 0 = off.

### Resolution

| Parameter | Default | What it does |
| --- | --- | --- |
| `ELEMENT_SIZE` | 500 | Triangle edge length (m) in the plan-view mesh in open water. The main horizontal resolution knob; mesh size grows about 4× per halving. |
| `NUM_LAYERS` | 38 | Maximum number of vertical layers per column. Deep columns get this many; thinner ones get fewer (see `MIN_LAYER_THICKNESS`). |

### Domain outline (where the mesh ends)

| Parameter | Default | What it does |
| --- | --- | --- |
| `SHORELINE_MIN_THICKNESS` | 30 | Open water shallower than this (m) is left out of the domain. |
| `SHELF_MIN_THICKNESS` | 75 | Same, for water under ice. Higher because the shelf data near the walls is not measured (filler bed of "ice base − 10 m"); 75 m keeps all of that out. |
| `SHORELINE_MARGIN` | 300 | Pulls the boundary this far (m) inward from the threshold contour, so the side walls sit on the deep side of the steep fjord walls. 0 = no margin. Not applied at the ocean mouth. 450 m breaks the 2D meshing. |
| `SHORELINE_GL_TOL` | 150 | How closely (m) the outline follows the contour near ice, i.e. along the grounding line. Smaller = more vertices; below ~100 m it traces the 150 m pixel staircase. 0 = old method (`SHORELINE_RES`). |
| `SHORELINE_TOL` | 600 | Same, away from the ice. |
| `SHORELINE_RES` | 30 | Only used when `SHORELINE_GL_TOL=0`: fixed number of outline vertices. |
| `SHORELINE_EDIT` | 0 | 1 = open the interactive outline editor before meshing (see the section below). |

### Vertical layering

| Parameter | Default | What it does |
| --- | --- | --- |
| `MIN_LAYER_THICKNESS` | 10 | No layer thinner than this (m) where avoidable: a column of thickness H gets min(floor(H/10), `NUM_LAYERS`) layers, at least 1. 0 = every column gets `NUM_LAYERS`. |
| `ICE_TOP_LAYER` | 5 | In ice-covered columns, the first layer under the ice base is this thick (m), and layers grow downward. Resolves the plume boundary layer. 0 = evenly spaced layers. |
| `LAYER_GROWTH` | 1.2 | Largest allowed thickness ratio between two consecutive layers. Where it binds (thin columns), the top layer comes out a bit thicker than `ICE_TOP_LAYER`. |
| `MIN_STACK_THICKNESS` | 10 | Safety floor (m): a column thinner than this is deepened to this. With the default outline no column needs it. |

### Horizontal refinement

| Parameter | Default | What it does |
| --- | --- | --- |
| `HORIZONTAL_REFINE` | 1 | 1 = finer triangles under the ice and along the grounding line; 0 = uniform `ELEMENT_SIZE` everywhere. |
| `ICE_ELEMENT_SIZE` | 0 (= h/2) | Triangle size (m) under the ice shelf. 0 = half of `ELEMENT_SIZE`. |
| `GL_ELEMENT_SIZE` | 0 (= h/4) | Triangle size (m) at the grounding line (the walls around the ice). 0 = a quarter of `ELEMENT_SIZE`. |
| `SIZE_GRADING` | 3000 | Distance (m) over which sizes grow back to `ELEMENT_SIZE`, from the grounding line and from the ice front into open water. |

### Calving front, tags, coordinates

| Parameter | Default | What it does |
| --- | --- | --- |
| `CALVING_FRONT` | 1 | 1 = the ice front is a vertical face from the sea surface down to the ice draft. 0 = old behaviour, a ramp one element wide. |
| `CALVING_FRONT_TAG` | 1 | Boundary tag of the calving face. Default = Ice; set 7 to give it its own tag. |
| `Z_SCALE` | 1 | Vertical exaggeration written into the coordinates. Keep 1 for simulations (real coordinates); 6 was the old default. |

### Old path (do not use unless comparing)

| Parameter | Default | What it does |
| --- | --- | --- |
| `USE_STACKED` | 1 | 0 = old TetGen-based path. It produces degenerate elements and has no real layering; kept only for reference. |
| `MIN_WATER_COLUMN`, `EMBED_SIGMA_SEEDS` | 0, 1 | Settings of the old path only. |

Fixed constants, set in the file itself: `MIN_ICE_THICKNESS = 0.05`, `MIN_WATER_DEPTH = 0.1`, and `MIN_FRONT_DRAFT = 1.0` (front columns with less draft get no vertical face).

## Editing the outline by hand

The automatic outline can be corrected by hand before meshing. Start the script with `SHORELINE_EDIT=1` on a machine with a display:

```bash
SHORELINE_EDIT=1 python -u ryder_3D_meshing_manylayers.py
```

A window shows the water column with the outline on top: the open-water threshold contour (orange), the under-ice contour (magenta), the ice edge (white), and the automatic outline (dashed grey) for reference.

| Action | Effect |
| --- | --- |
| left-drag a red vertex | move it |
| `i` | insert a vertex at the mouse, into the nearest edge |
| `d` | delete the vertex nearest the mouse |
| `u` | undo |
| `R` | reset to the automatic outline |
| close the window | save and continue meshing |

Turn off the toolbar's zoom/pan tool before editing; edits are ignored while it is active.

The outline is cached in `mesh/new_data/saved_shoreline_<settings>.json`, and the file name encodes the outline settings. Later runs with the same settings reuse it, edits included. Changing an outline setting starts from a fresh automatic outline, and the old file stays. To discard your edits, delete the file.

## Typical meshes and how to check them

| Mesh | Command (after the `export` lines) | Size | Edge: open / ice / grounding line |
| --- | --- | --- | --- |
| Coarse test | `NUM_LAYERS=20 ELEMENT_SIZE=1000 python -u ryder_3D_meshing_manylayers.py` | 37k nodes, 197k tets | 1000 / 410 / 250 m |
| Coarse, no grounding-line refinement | `NUM_LAYERS=20 ELEMENT_SIZE=1000 GL_ELEMENT_SIZE=500 python -u ...` | smaller than the above | 1000 / 500 / 500 m |
| Default | `python -u ryder_3D_meshing_manylayers.py` | 218k nodes, 1.23M tets | 500 / 225 / 130 m |
| Uniform, for comparison | `HORIZONTAL_REFINE=0 python -u ...` | 99k nodes, 560k tets | 500 m everywhere |

All of these have a 5 m first layer under the ice, no inverted elements, and gamma ≥ 0.011.

The scripts in `mesh/mesh_diagnostics/` check a mesh. Trust them, not the run log:

| Script | What it reports |
| --- | --- |
| `degeneracy.py MESH.msh` | inverted, zero-volume and zero-length-edge elements (should all be 0) |
| `mesh_quality.py MESH.msh` | minimum gamma, edge lengths |
| `layer_structure.py MESH.msh` | nodes per vertical column |
| `quality_by_region.py MESH.msh H` | quality by distance to grounding line, shore, inflow and ice front (`H` = element size) |
| `plot_transects.py MESH.msh out.png Y0 Y1 STEP` | cross-sections at y = const (km): the actual element cuts over the raster ice and bed |
| `plot_wall.py out.png MESH.msh` | the side walls unrolled flat, showing the layers on them |

The geometry code is tested by `python -u test_stacked_volume.py`, which should end with `ALL CHECKS PASSED`.

## Things to know

- **The domain is cut back from the real coast.** It stops where open water reaches 30 m and under-ice water 75 m, then another 300 m inward. So the side walls are vertical and 100–200 m tall rather than tapering to zero depth.
- **The calving front is a vertical face.** Its height is the ice draft, 10–90 m. It carries the Ice tag (1) by default.
- **Shelf data near the walls is synthetic.** The ice base is computed by assuming the shelf floats, and about 13% of shelf pixels have a filler bed exactly 10 m under it. The 75 m under-ice threshold keeps those out of the domain.
- **Gamma looks low; that's expected.** Thin layers under wide triangles are flat on purpose, so median gamma is about 0.1. The worst elements sit where the sea floor is very steep, not at the walls.
- **Large settings changes need checking.** Margins above 300 m, or very small tolerances, can make the 2D meshing fail. Run the diagnostics after any big change.
- **The mesh is in GIS coordinates (EPSG:3413, metres).** x and y are around −110 000 to −80 000 and −909 000 to −857 000. Shift the origin in the solver if large coordinate values are a problem.
