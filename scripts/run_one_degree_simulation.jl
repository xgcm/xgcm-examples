# Realistic 1-degree global ocean--sea-ice simulation (ClimaOcean) on CPU.
#
# Adapted from ClimaOcean's `examples/one_degree_simulation.jl` to run **locally
# on a CPU** (no GPU). It builds the realistic TripolarGrid with ETOPO bathymetry,
# initializes from the ECCO4 state estimate, forces with JRA55-do, and time-steps
# the coupled ocean--sea-ice model to 10 model-days (long enough to spin up the
# circulation past the initial barotropic adjustment). Surface ocean fields
# (u, v, T, S) are written to JLD2 once per model-day, so the day-10 snapshot is
# saved (and earlier days remain as a safety net for an interrupted run).
# On a CPU this takes several hours. Convert the final snapshot to NetCDF with
# `jld2_to_netcdf.jl` for use in the xgcm tripolar-fold example.
#
#   julia --project=scripts/climaocean_env scripts/run_one_degree_simulation.jl

using ClimaOcean
using Oceananigans
using Oceananigans.Units
using Dates
using Printf
using Statistics

arch = CPU()                 # local Mac: no GPU
Nx, Ny, Nz = 360, 180, 50

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map = true)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3,
                                                 skew_flux_formulation=AdvectiveFormulation())
@inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) =
    Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true, parameters=15days)
vertical_mixing = ClimaOcean.Oceans.default_ocean_closure()

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(eddy_closure, horizontal_viscosity, vertical_mixing))

sea_ice = sea_ice_simulation(grid, ocean; advection=tracer_advection)

# Initial condition: ECCO4 monthly state estimate.
date = DateTime(1993, 1, 1)
dataset = ECCO4Monthly()
set!(ocean.model, T=Metadatum(:temperature; date, dataset),
                  S=Metadatum(:salinity; date, dataset))
set!(sea_ice.model, h=Metadatum(:sea_ice_thickness; date, dataset),
                    ℵ=Metadatum(:sea_ice_concentration; date, dataset))

# JRA55-do atmospheric forcing.
radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(80),
                                       include_rivers_and_icebergs = false)

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=20minutes, stop_time=10days)

wall_time = Ref(time_ns())
function progress(sim)
    o = sim.model.ocean
    u, v, w = o.model.velocities
    T = o.model.tracers.T
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))
    step_time = 1e-9 * (time_ns() - wall_time[])
    @info @sprintf("time: %s, iter: %d, max|u|: (%.1e, %.1e, %.1e) m/s, wall/iter-block: %s",
                   prettytime(sim), iteration(sim), umax..., prettytime(step_time))
    wall_time[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, TimeInterval(12hours))

# Surface ocean output, one snapshot per model-day (day 10 is the final one).
ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
    schedule = TimeInterval(1days),
    filename = joinpath(@__DIR__, "ocean_one_degree_surface_fields"),
    indices = (:, :, grid.Nz),
    overwrite_existing = true)

@info "Starting run! (stop_time = 10 days; expect several hours on CPU)"
run!(simulation)
