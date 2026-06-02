# Generate a small Oceananigans.jl tripolar-grid example dataset for xgcm.
#
# Produces `oceananigans_tripolar.nc`: a coarse global tripolar grid with the
# longitude/latitude of the C-grid points (tracer centre + the u/v faces + the
# F-point corners) plus a smooth tracer and a horizontal velocity field. xgcm
# uses these to demonstrate fold-aware (`face_connections`, `fold=True`)
# operations on the Oceananigans north-fold convention.
#
# Pinned environment (reproducible):
#   julia 1.10+
#   Oceananigans v0.96.2  (TripolarGrid is exported directly; the old
#                          OrthogonalSphericalShellGrids.jl is now a submodule)
#   NCDatasets   v0.14+
#
# Run:  julia --project=. scripts/generate_oceananigans_tripolar.jl
# (or `julia scripts/generate_oceananigans_tripolar.jl` against a depot that
#  already has Oceananigans + NCDatasets installed.)

using Oceananigans
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NCDatasets

# Coarse grid: keep the file tiny (<1 MB). Longitude count MUST be even (fold).
Nx, Ny, Nz = 60, 40, 1

grid = TripolarGrid(size = (Nx, Ny, Nz),
                    z = (-10, 0),
                    north_poles_latitude = 60,
                    first_pole_longitude = 70)

@info "Built tripolar grid" grid

# --- coordinates at the staggered C-grid locations -------------------------
# OrthogonalSphericalShellGrid stores 2-D (λ, φ) at each stagger location.
λcc = λnodes(grid, Center(), Center(), Center())
φcc = φnodes(grid, Center(), Center(), Center())
λff = λnodes(grid, Face(),   Face(),   Center())
φff = φnodes(grid, Face(),   Face(),   Center())
λfc = λnodes(grid, Face(),   Center(), Center())  # u points (east face)
φfc = φnodes(grid, Face(),   Center(), Center())
λcf = λnodes(grid, Center(), Face(),   Center())  # v points (north face)
φcf = φnodes(grid, Center(), Face(),   Center())

# --- smooth analytic fields on the grid ------------------------------------
c = CenterField(grid)   # tracer at (Center, Center)
u = XFaceField(grid)    # zonal velocity at (Face,  Center)
v = YFaceField(grid)    # meridional velocity at (Center, Face)

# Functions of geographic (λ, φ); deg-trig keeps them smooth across the fold.
# Use zonal wavenumber-2 patterns that do NOT vanish at the poles, so that the
# velocity (and hence the relative vorticity) is non-trivial right at the fold.
set!(c, (λ, φ, z) -> sind(2λ))
set!(u, (λ, φ, z) -> sind(2λ))
set!(v, (λ, φ, z) -> cosd(2λ))

# Fill halos so the north-fold (zipper) relationship is applied internally.
fill_halo_regions!(c)
fill_halo_regions!(u)
fill_halo_regions!(v)

C = Array(interior(c, :, :, 1))
U = Array(interior(u, :, :, 1))
V = Array(interior(v, :, :, 1))

@info "Field shapes" size(C) size(U) size(V) size(λcc) size(λff)

# --- write NetCDF ----------------------------------------------------------
# Each variable carries its own (x, y) index dims; xgcm reconstructs the
# staggered Arakawa-C positions and the fold from these coordinates.
outfile = joinpath(@__DIR__, "..", "oceananigans_tripolar.nc")
isfile(outfile) && rm(outfile)

ds = NCDataset(outfile, "c")
defDim(ds, "x_c", size(C, 1)); defDim(ds, "y_c", size(C, 2))
defDim(ds, "x_f", size(λff, 1)); defDim(ds, "y_f", size(λff, 2))

ds.attrib["title"] = "Oceananigans.jl tripolar-grid example for xgcm"
ds.attrib["source"] = "Oceananigans TripolarGrid; scripts/generate_oceananigans_tripolar.jl"
ds.attrib["north_poles_latitude"] = 60
ds.attrib["first_pole_longitude"] = 70

defVar(ds, "lon_cc", λcc, ("x_c", "y_c")); defVar(ds, "lat_cc", φcc, ("x_c", "y_c"))
defVar(ds, "lon_ff", λff, ("x_f", "y_f")); defVar(ds, "lat_ff", φff, ("x_f", "y_f"))
defVar(ds, "lon_fc", λfc, ("x_f", "y_c")); defVar(ds, "lat_fc", φfc, ("x_f", "y_c"))
defVar(ds, "lon_cf", λcf, ("x_c", "y_f")); defVar(ds, "lat_cf", φcf, ("x_c", "y_f"))

defVar(ds, "tracer", C, ("x_c", "y_c"))
defVar(ds, "u", U, ("x_f", "y_c"))
defVar(ds, "v", V, ("x_c", "y_f"))

close(ds)
@info "Wrote $outfile"
