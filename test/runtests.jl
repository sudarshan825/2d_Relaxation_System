if Base.find_package("RelaxWLS2D") === nothing      # include()d without activating
    import Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
end
using RelaxWLS2D, Test

# small, fast configurations (reference values from an independent prototype)
small(; kw...) = Params(; Nx = 24, tfinal = 0.1, write_output = false, kw...)
rest(x, y, ε) = (0.0, 0.0, 0.0, 0.0)
amp(res) = sum(res.u1 .* sinpi.(2 .* res.y)) / sum(sinpi.(2 .* res.y) .^ 2)
rates(e) = log2.(e[1:end-1] ./ e[2:end])

@testset "RelaxWLS2D" begin
    @testset "WLS operators" begin
        x, y = RelaxWLS2D.lattice(32, 1.0)
        o = RelaxWLS2D.WLSOps2D(x, y)
        f = sinpi.(2 .* x) .* cospi.(2 .* y)
        @test maximum(abs, o.Dx * f .- 2π .* cospi.(2 .* x) .* cospi.(2 .* y)) < 0.05 * 2π
        @test maximum(abs, o.Dy * f .+ 2π .* sinpi.(2 .* x) .* sinpi.(2 .* y)) < 0.05 * 2π
        @test maximum(abs, o.Lap * f .+ 8π^2 .* f) < 0.08 * 8π^2      # prototype 0.049
        @test maximum(abs, o.Lc  * f .+ 8π^2 .* f) < 0.05 * 8π^2      # prototype 0.018
        @test maximum(abs, o.Dx * ones(o.N)) < 1e-10
    end

    @testset "global equilibrium stays at rest" begin
        res = run_case(small(eps = 0.1, tau = 0.01, Nx = 12, tfinal = 0.05, init = rest);
                       verbose = false)
        for f in (res.rho, res.u1, res.u2, res.T, res.G1, res.theta_a)
            @test maximum(abs, f) < 1e-12
        end
    end

    # shear wave u1 = sin(2πy), τ = 0.05, t = 0.1: limit amplitude e^{-4π²τt} = 0.8209;
    # prototype at Nx = 24: 0.8272 (ε = 1e-3) and 0.8270 (ε = 0.1)
    limit = exp(-4π^2 * 0.05 * 0.1)
    for ε in (1e-3, 0.1)
        @testset "shear wave decay ε = $ε" begin
            res = run_case(small(eps = ε, tau = 0.05, init = shear_wave()); verbose = false)
            @test all(isfinite, res.u1)
            @test abs(amp(res) - limit) < 0.02 * limit
            @test abs(amp(res) - 0.827) < 0.002
            @test maximum(abs, res.u2) < 0.01
            @test abs(sum(res.rho)) / length(res.rho) < 1e-12        # mass
        end
    end

    @testset "Taylor–Green, asymptotic preserving (ε = 1e-4)" begin
        # prototype: max error 0.021, max|∇·u| 1.2e-7 with Δt independent of ε
        res = run_case(small(eps = 1e-4, tau = 0.05, init = taylor_green()); verbose = false)
        ex = exp(-8π^2 * 0.05 * 0.1) .* sinpi.(2 .* res.x) .* cospi.(2 .* res.y)
        @test maximum(abs, res.u1 .- ex) < 0.03
        x, y = res.x, res.y
        o = RelaxWLS2D.WLSOps2D(x, y)
        @test maximum(abs, o.Dx * res.u1 .+ o.Dy * res.u2) < 1e-5
        @test maximum(abs, res.rho .+ res.T) < 1e-3                # p = ρ + T → 0 (prototype 3.4e-5)
    end

    @testset "temporal order of the IMEX schemes" begin
        init(x, y, ε) = (0.3 * sinpi(2x), sinpi(2y), 0.2 * cospi(2x), 0.5 * cospi(4x))
        base(; kw...) = Params(; Nx = 16, eps = 0.5, tau = 0.1, tfinal = 0.05, stab = false,
                               init = init, write_output = false, kw...)
        err(a, b) = maximum(maximum(abs, getfield(a, k) .- getfield(b, k))
                            for k in (:rho, :u1, :u2, :T, :G1, :G2, :theta_a, :theta_b))
        for (sch, order) in (("ARS111", 1), ("ARS222", 2), ("ARS443", 3))
            ref = run_case(base(scheme = sch, dt = 0.05 / 800); verbose = false)
            e = [err(run_case(base(scheme = sch, dt = 0.05 / m); verbose = false), ref)
                 for m in (25, 50, 100)]
            @test all(rates(e) .> order - 0.25)
        end
    end

    @testset "shear layer: short run, mass, output file" begin
        dir = mktempdir()
        res = run_case(small(eps = 0.1, tau = 0.001, tfinal = 0.1, init = shear_layer,
                             write_output = true, outdir = dir, nsave = 2, plot = false);
                       verbose = false)
        @test all(isfinite, res.omega)
        @test abs(sum(res.rho) / length(res.rho) - 1) < 1e-12       # ρ = 1 initially
        @test 13 < maximum(abs, res.omega) < 17                      # prototype: 15.06
        @test isfile(res.file) && length(res.snapshots) == 1
        lines = readlines(res.file)
        @test startswith(lines[1], "# x y rho u1 u2 T omega")
        @test occursin("model=relaxation", lines[1])
        @test length(lines) == 24^2 + 1
        @test length(split(lines[2])) == 7
    end
end
