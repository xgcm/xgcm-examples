import Pkg
Pkg.activate(".")
Pkg.add(["ClimaOcean", "Oceananigans", "NCDatasets", "CFTime"])
Pkg.precompile()
println("CLIMAOCEAN_INSTALL_DONE")
