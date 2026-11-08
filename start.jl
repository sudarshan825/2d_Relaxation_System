# One-line start for RelaxWLS2D in the Julia REPL or VS Code:
#
#   include("path/to/RelaxWLS2D/start.jl")
#
# It activates this package, installs what is missing (Plots.jl once, into the
# default environment), loads the solver and the plot functions, and sets
#   R = <package>/results
# Then e.g.
#   res = run_case(Params())                            # ε = 0.1, τ = 0.001: data + PNG in R
#   plot_file(res.file; field = :u1)                    # other fields
#   run_tests()
import Pkg

const PKGDIR = @__DIR__

Pkg.activate(PKGDIR; io = devnull)
isfile(joinpath(PKGDIR, "Manifest.toml")) || Pkg.instantiate(; io = devnull)

if Base.find_package("Plots") === nothing
    println("Installing Plots.jl into the default environment (only once, takes a few minutes) ...")
    Pkg.activate(; io = devnull)
    Pkg.add("Plots")
    Pkg.activate(PKGDIR; io = devnull)
end

using RelaxWLS2D
include(joinpath(PKGDIR, "examples", "plot_results.jl"))

# every run_case now also plots the vorticity -> <results>/<tag>_omega.png
RelaxWLS2D.PLOT_HOOK[] = file -> (plot_file(file); nothing)

R = joinpath(PKGDIR, "results")
mkpath(R)

"Run the package tests."
run_tests() = (Pkg.activate(PKGDIR; io = devnull); Pkg.test())

println("""
RelaxWLS2D ready.  Results folder R = $R
Threads: $(Threads.nthreads())$(Threads.nthreads() == 1 ? "   (start Julia with  julia -t auto  to use all cores)" : "")

  res = run_case(Params())                                   # ε = 0.1, τ = 0.001, 100×100, t = 1.5
  res = run_case(Params(eps = 1e-3, nsave = 5))              # near the Navier–Stokes limit, 5 snapshots
  plot_file(res.file; field = :u1)                           # other fields: :rho :u1 :u2 :T
  compare_files(fileA, fileB)                                # same contour levels
  plot_all(R)                                                # every run in R
  run_tests()""")
