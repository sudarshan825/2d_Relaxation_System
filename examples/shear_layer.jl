# Doubly periodic double shear layer on [0, 1]²:
#   ρ = 1, T = 0,  u1 = tanh(30(y - 1/4)) for y ≤ 1/2,  tanh(30(3/4 - y)) otherwise,
#   u2 = 0.05 sin(2πx),  up to t = 1.5 on 100 × 100 points.
#
#   julia -t auto examples/shear_layer.jl                       # ε = 0.1, τ = 0.001 (default)
#   julia -t auto examples/shear_layer.jl 0.001 0.001           # ε τ
#   julia -t auto examples/shear_layer.jl 0.1 0.001 200         # ε τ Nx
#   julia -t auto examples/shear_layer.jl 0.1 0.001 100 ARS443  # ε τ Nx scheme
#
# Data (and, if Plots.jl is installed, a vorticity plot) are written to results/.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
using RelaxWLS2D
if Base.find_package("Plots") !== nothing          # plot every run if Plots.jl is installed
    include(joinpath(@__DIR__, "plot_results.jl"))
    RelaxWLS2D.PLOT_HOOK[] = file -> (plot_file(file); nothing)
end

arg(i, T, default) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

run_case(Params(eps    = arg(1, Float64, 0.1),
                tau    = arg(2, Float64, 0.001),
                Nx     = arg(3, Int, 100),
                scheme = length(ARGS) >= 4 ? ARGS[4] : "ARS222",
                tfinal = 1.5,
                nsave  = 5))
