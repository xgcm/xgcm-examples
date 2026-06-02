# Convert the latest surface snapshot from the ClimaOcean 1-degree run
# (`run_one_degree_simulation.jl`, JLD2 output) into a NetCDF file for xgcm:
# surface u, v, T plus the tripolar grid's longitude/latitude at the tracer
# centre, the u/v faces, and the cell corner.
#
#   julia --project=scripts/climaocean_env scripts/jld2_to_netcdf.jl

using Oceananigans
using Oceananigans.Grids: λnodes, φnodes, Center, Face
using JLD2
using NCDatasets

infile  = joinpath(@__DIR__, "ocean_one_degree_surface_fields.jld2")
outfile = joinpath(@__DIR__, "..", "oceananigans_tripolar.nc")

ut = FieldTimeSeries(infile, "u"; backend = OnDisk())
vt = FieldTimeSeries(infile, "v"; backend = OnDisk())
Tt = FieldTimeSeries(infile, "T"; backend = OnDisk())

n = length(ut.times)
@info "Found $n snapshot(s); using the last one at t = $(ut.times[n]) s ($(ut.times[n]/86400) days)"

grid = ut.grid

U = Array(interior(ut[n], :, :, 1))
V = Array(interior(vt[n], :, :, 1))
T = Array(interior(Tt[n], :, :, 1))

# Longitudes/latitudes at the staggered C-grid locations (2-D on the tripolar grid).
λcc = λnodes(grid, Center(), Center(), Center()); φcc = φnodes(grid, Center(), Center(), Center())
λfc = λnodes(grid, Face(),   Center(), Center()); φfc = φnodes(grid, Face(),   Center(), Center())
λcf = λnodes(grid, Center(), Face(),   Center()); φcf = φnodes(grid, Center(), Face(),   Center())
λff = λnodes(grid, Face(),   Face(),   Center()); φff = φnodes(grid, Face(),   Face(),   Center())

@info "Field shapes" size(T) size(U) size(V) size(λcc) size(λff)

isfile(outfile) && rm(outfile)
ds = NCDataset(outfile, "c")
defDim(ds, "x_c", size(T, 1)); defDim(ds, "y_c", size(T, 2))
defDim(ds, "x_f", size(λff, 1)); defDim(ds, "y_f", size(λff, 2))

ds.attrib["title"] = "ClimaOcean 1-degree global ocean (Oceananigans TripolarGrid) surface snapshot"
ds.attrib["source"] = "scripts/run_one_degree_simulation.jl (ECCO4 init, JRA55 forcing)"
ds.attrib["snapshot_time_days"] = ut.times[n] / 86400

defVar(ds, "lon_cc", λcc, ("x_c", "y_c")); defVar(ds, "lat_cc", φcc, ("x_c", "y_c"))
defVar(ds, "lon_fc", λfc, ("x_f", "y_c")); defVar(ds, "lat_fc", φfc, ("x_f", "y_c"))
defVar(ds, "lon_cf", λcf, ("x_c", "y_f")); defVar(ds, "lat_cf", φcf, ("x_c", "y_f"))
defVar(ds, "lon_ff", λff, ("x_f", "y_f")); defVar(ds, "lat_ff", φff, ("x_f", "y_f"))

defVar(ds, "tracer", T, ("x_c", "y_c"))   # surface temperature [°C]
defVar(ds, "u", U, ("x_f", "y_c"))        # zonal velocity [m/s]
defVar(ds, "v", V, ("x_c", "y_f"))        # meridional velocity [m/s]

close(ds)
@info "Wrote $outfile"
