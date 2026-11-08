# =============================================================================
#  RelaxWLS2D.jl
#
#  Meshfree Eulerian solver for the 2D relaxation (moment) system of the BGK
#  model in the low-Mach / small-Knudsen scaling, d = 2, periodic box [0,L]²:
#
#        ε ∂t ρ + ∇·u                         = 0
#        ε ∂t u + ε ∇·Θ + ∇p                  = 0,     p = ρ + T
#        ∂t T + (2/(εd)) ∇·u + (1/d) ∇·Γ      = 0
#        ε² ∂t Θ + D(∇u) = -(1/τ) (Θ - A(u))
#        ε² ∂t Γ + (d+2) ∇T = -(1/τ) (Γ - (d+2) u T)
#
#  with D(∇u) = ∇u + ∇uᵀ - (2/d)(∇·u) I and A(u) = u⊗u - |u|²/d I.
#  ρ, u, T are fluctuations around the global equilibrium, Γ is the heat flux
#  and Θ the trace-free stress, Θ = [θa θb; θb -θa].
#  As ε → 0: ∇·u = 0, ρ + T = 0 and the incompressible Navier–Stokes equations
#  with viscosity τ, ∂t u + ∇·(u⊗u) + ∇Π = τ Δu, ∂t T + ∇·(uT) = τ ΔT.
#
#  Space : weighted least squares (GFDM) on fixed points (uniform lattice),
#          Gaussian weight exp(-α d²/r²), quadratic Taylor fit; sparse
#          matrices ∂x, ∂y; Laplacian ∂x∂x + ∂y∂y in the pressure step.
#  Time  : globally stiffly accurate IMEX Runge-Kutta (ARS type). Implicit:
#          acoustics (one Helmholtz problem per stage for p~ = p/ε), ∇·Θ, ∇·Γ
#          and the relaxation; explicit: D(∇u), A(u), (d+2)∇T, (d+2)uT.
#          Δt does not depend on ε (asymptotic preserving).
#  Validity: the system is hyperbolic with shear-wave speed 1/ε; it needs
#          ε·max|u| < 1 (subcharacteristic condition).
#  Stabilisation: third-order upwind-type dissipation -a h³/12 Δ² of all
#          fields (explicit), a = max|u| + (√((d+2)/d)/ε) ε²τ/(ε²τ+Δt).
# =============================================================================
module RelaxWLS2D

using LinearAlgebra, SparseArrays, Printf
using Base.Threads

export Params, run_case, write_results, file_tag,
       shear_layer, make_shear_layer, step_shear_layer, make_step_shear_layer,
       shear_wave, taylor_green

const DIM = 2
const CP  = (DIM + 2) / DIM      # (d+2)/d = 2
const CG  = DIM + 2.0            # d+2 = 4

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Base.@kwdef mutable struct Params
    eps::Float64     = 0.1       # scaling parameter ε (needs ε·max|u| < 1)
    tau::Float64     = 0.001     # relaxation time τ
    L::Float64       = 1.0       # periodic box [0, L]²
    Nx::Int          = 100       # points per direction (Nx² points)
    tfinal::Float64  = 1.5
    scheme::String   = "ARS222"  # ARS111 (1st), ARS222 (2nd), ARS443 (3rd order)
    hratio::Float64  = 0.35      # neighbour radius r = Δx / hratio
    min_nb::Int      = 12        # minimum number of neighbours
    alpha::Float64   = 6.0       # Gaussian weight parameter
    stab::Bool       = true      # third-order upwind-type dissipation
    cfl::Float64     = 0.8       # diffusive time-step factor
    cadv::Float64    = 0.3       # transport time-step factor (see stable_dt)
    dt::Float64      = 0.0       # > 0 overrides the automatic time step
    init::Function   = shear_layer # (x, y, ε) -> (ρ, u1, u2, T) initial fluctuations
    outdir::String   = joinpath(dirname(@__DIR__), "results")   # <package>/results
    nprint::Int      = 0         # progress every nprint steps (0: ~20 messages)
    write_output::Bool = true
    nsave::Int       = 0         # > 0: also save nsave equally spaced snapshots
    save_times::Vector{Float64} = Float64[]   # and/or snapshots at these times
    plot::Bool       = true      # call the plot hook after each output (see start.jl)
end

# -----------------------------------------------------------------------------
# Periodic cell list
# -----------------------------------------------------------------------------
@inline pdist(d, L) = d - L * round(d / L)

struct CellList
    head::Matrix{Int}
    nxt::Vector{Int}
    nc::Int
    cs::Float64
    L::Float64
end

@inline cellidx(q, cs, nc) = clamp(floor(Int, q / cs) + 1, 1, nc)

function CellList(x, y, L, h)
    nc = max(1, floor(Int, L / h)); cs = L / nc; N = length(x)
    head = zeros(Int, nc, nc); nxt = zeros(Int, N)
    for i in 1:N
        a = cellidx(x[i], cs, nc); b = cellidx(y[i], cs, nc)
        nxt[i] = head[a, b]; head[a, b] = i
    end
    return CellList(head, nxt, nc, cs, L)
end

"Neighbours of (qx, qy) within r (periodic), without `self`: indices and offsets."
function query!(js, dxs, dys, cl::CellList, x, y, qx, qy, r, self = 0)
    empty!(js); empty!(dxs); empty!(dys)
    L = cl.L; nc = cl.nc; r2 = r * r
    m = ceil(Int, r / cl.cs)
    a = cellidx(qx, cl.cs, nc); b = cellidx(qy, cl.cs, nc)
    full = 2m + 1 >= nc
    ra = full ? (1:nc) : (a-m:a+m)
    rb = full ? (1:nc) : (b-m:b+m)
    for aa in ra, bb in rb
        j = cl.head[mod1(aa, nc), mod1(bb, nc)]
        while j != 0
            if j != self
                dx = pdist(x[j] - qx, L); dy = pdist(y[j] - qy, L)
                if dx * dx + dy * dy < r2
                    push!(js, j); push!(dxs, dx); push!(dys, dy)
                end
            end
            j = cl.nxt[j]
        end
    end
    return js
end

# -----------------------------------------------------------------------------
# Meshfree WLS operators
# -----------------------------------------------------------------------------
struct WLSOps2D
    x::Vector{Float64}
    y::Vector{Float64}
    N::Int
    L::Float64
    h::Float64                       # mean spacing Δx
    Dx::SparseMatrixCSC{Float64,Int}
    Dy::SparseMatrixCSC{Float64,Int}
    Lap::SparseMatrixCSC{Float64,Int}   # Dx Dx + Dy Dy (pressure / Helmholtz step)
    Lc::SparseMatrixCSC{Float64,Int}    # compact ∂xx + ∂yy of the fit (dissipation)
end

"""
    WLSOps2D(x, y; L, hratio, min_nb, alpha)

For each point a quadratic Taylor fit
f_j - f_i ≈ f_x dx + f_y dy + f_xx dx²/2 + f_xy dx dy + f_yy dy²/2
to the neighbours within r = Δx/hratio (enlarged until `min_nb` neighbours),
weighted least squares with weight exp(-α d²/r²). Returns sparse ∂x, ∂y, the
Laplacian `Lap = Dx Dx + Dy Dy` used in the implicit pressure step (it is
consistent with the discrete divergence and gradient, which keeps the IMEX
scheme at its full order in time) and the compact Laplacian `Lc = ∂xx + ∂yy`
of the fit, used for the numerical dissipation.
"""
function WLSOps2D(x, y; L = 1.0, hratio = 0.35, min_nb = 12, alpha = 6.0)
    N  = length(x); h0 = L / sqrt(N); r0 = h0 / hratio
    cl = CellList(x, y, L, r0)
    lists = Vector{Vector{Int}}(undef, N)
    cx = Vector{Vector{Float64}}(undef, N); cy = similar(cx); cl2 = similar(cx)
    @threads for i in 1:N
        js = Int[]; dxs = Float64[]; dys = Float64[]
        r = r0
        for _ in 1:20
            query!(js, dxs, dys, cl, x, y, x[i], y[i], r, i)
            length(js) >= min_nb && break
            r *= 1.2
        end
        n = length(js)
        M = zeros(n, 5); w = zeros(n)
        for e in 1:n
            sx = dxs[e] / h0; sy = dys[e] / h0
            M[e, 1] = sx; M[e, 2] = sy; M[e, 3] = sx^2 / 2; M[e, 4] = sx * sy; M[e, 5] = sy^2 / 2
            w[e] = exp(-alpha * (dxs[e]^2 + dys[e]^2) / r^2)
        end
        C = (M' * (w .* M)) \ (M' .* w')            # 5 × n
        lists[i] = copy(js)
        cx[i] = C[1, :] ./ h0; cy[i] = C[2, :] ./ h0
        cl2[i] = (C[3, :] .+ C[5, :]) ./ h0^2
    end
    I = Int[]; J = Int[]; Vx = Float64[]; Vy = Float64[]; Vl = Float64[]
    for i in 1:N
        append!(I, fill(i, length(lists[i]) + 1)); append!(J, vcat(lists[i], i))
        append!(Vx, vcat(cx[i], -sum(cx[i])))
        append!(Vy, vcat(cy[i], -sum(cy[i])))
        append!(Vl, vcat(cl2[i], -sum(cl2[i])))
    end
    Dx = sparse(I, J, Vx, N, N); Dy = sparse(I, J, Vy, N, N)
    return WLSOps2D(collect(Float64, x), collect(Float64, y), N, L, h0,
                    Dx, Dy, Dx * Dx + Dy * Dy, sparse(I, J, Vl, N, N))
end

# -----------------------------------------------------------------------------
# IMEX Runge-Kutta tableaux (globally stiffly accurate, ARS type): (Ã, A)
# -----------------------------------------------------------------------------
function imex_tableau(name::AbstractString)
    name = uppercase(name)
    if name == "ARS111"
        At = [0.0 0.0; 1.0 0.0]
        A  = [0.0 0.0; 0.0 1.0]
    elseif name == "ARS222"
        g = 1.0 - 1.0 / sqrt(2.0)
        d = 1.0 - 1.0 / (2.0 * g)
        At = [0.0 0.0 0.0; g 0.0 0.0; d 1-d 0.0]
        A  = [0.0 0.0 0.0; 0.0 g 0.0; 0.0 1-g g]
    elseif name == "ARS443"
        At = [0.0    0.0   0.0   0.0  0.0;
              1/2    0.0   0.0   0.0  0.0;
              11/18  1/18  0.0   0.0  0.0;
              5/6   -5/6   1/2   0.0  0.0;
              1/4    7/4   3/4  -7/4  0.0]
        A  = [0.0  0.0   0.0  0.0  0.0;
              0.0  1/2   0.0  0.0  0.0;
              0.0  1/6   1/2  0.0  0.0;
              0.0 -1/2   1/2  1/2  0.0;
              0.0  3/2  -3/2  1/2  1/2]
    else
        error("unknown IMEX tableau $name (use ARS111, ARS222, ARS443)")
    end
    return Matrix{Float64}(At), Matrix{Float64}(A)
end

# -----------------------------------------------------------------------------
# AP IMEX solver
# -----------------------------------------------------------------------------
mutable struct Relaxation2D
    ops::WLSOps2D
    eps::Float64
    tau::Float64
    At::Matrix{Float64}
    A::Matrix{Float64}
    s::Int
    stab::Bool
    S::SparseMatrixCSC{Float64,Int}               # -(h³/12) Δ²
    lu_cache::Dict{Tuple{Float64,Float64},Any}
end

function Relaxation2D(ops::WLSOps2D, eps, tau; scheme = "ARS222", stab = true)
    At, A = imex_tableau(scheme)
    S = (-(ops.h^3) / 12.0) .* (ops.Lc * ops.Lc)
    return Relaxation2D(ops, Float64(eps), Float64(tau), At, A, size(A, 1), stab, S,
                        Dict{Tuple{Float64,Float64},Any}())
end

"Dissipation speed a = max|u| + (√((d+2)/d)/ε) ε²τ/(ε²τ+Δt)."
function stab_speed(sv::Relaxation2D, dt, umax)
    sv.stab || return 0.0
    e2t = sv.eps^2 * sv.tau
    th = sv.tau > 0 ? e2t / (e2t + dt) : 0.0
    return sqrt(CP) / sv.eps * th + umax
end

"Spectral radius of the dissipation operator S (power iteration)."
function spectral_radius(S; iters = 60)
    v = [sin(1.3 * i) + cos(0.7 * i^1.1) for i in 1:size(S, 1)]
    λ = 0.0
    for _ in 1:iters
        w = S * v
        λ = norm(w) / norm(v)
        v = w ./ norm(w)
    end
    return λ
end

"""
    stable_dt(sv, umax; cfl = 0.8, cadv = 0.3)

Time step from three bounds:
- explicit diffusion through the relaxed Γ, Θ: 2d (d+2)/d τ Δt²/(ε²τ+Δt) ≤ cfl Δx²,
- explicit transport: Δt ≤ cadv Δx / (max|u| + c), c = (√((d+2)/d)/ε) ε²τ/(ε²τ+Δt),
- explicit dissipation: Δt ≤ 1 / ((max|u| + c) ρ(S)).
"""
function stable_dt(sv::Relaxation2D, umax; cfl = 0.8, cadv = 0.3)
    h = sv.ops.h; e2t = sv.eps^2 * sv.tau
    dt = 1.0
    if sv.tau > 0
        a = 2 * DIM * CP * sv.tau; b = cfl * h * h
        dt = (b + sqrt(b * b + 4 * a * b * e2t)) / (2 * a)
    end
    rS = sv.stab ? spectral_radius(sv.S) : 0.0
    for _ in 1:30                      # c and the dissipation depend on Δt
        th = sv.tau > 0 ? e2t / (e2t + dt) : 0.0
        sp = umax + sqrt(CP) / sv.eps * th
        dtn = min(dt, cadv * h / sp)
        sv.stab && (dtn = min(dtn, 1 / (sp * rS)))
        abs(dtn - dt) < 1e-14 && break
        dt = dtn
    end
    return dt
end

# bordered system [α I - β Δ, q; qᵀ, 0] for the zero-mean part of p~
function helmholtz_lu(sv::Relaxation2D, α, β)
    get!(sv.lu_cache, (α, β)) do
        N = sv.ops.N
        q = fill(1.0 / N, N)
        K = α .* sparse(1.0I, N, N) .- β .* sv.ops.Lap
        lu([K sparse(reshape(q, N, 1)); sparse(reshape(q, 1, N)) spzeros(1, 1)])
    end
end

const FIELDS = (:rho, :u1, :u2, :T, :G1, :G2, :ta, :tb)

"""
Explicit parts: F = (F_Γ1, F_Γ2, F_θa, F_θb) in ε²τ ∂t X = -(F + X), and the
dissipation sources R = a S (ρ, u1, u2, T).
"""
function explicit_terms(sv::Relaxation2D, U, a)
    o = sv.ops; τ = sv.tau; e2t = sv.eps^2 * sv.tau
    FG1 = CG .* (τ .* (o.Dx * U.T) .- U.u1 .* U.T)
    FG2 = CG .* (τ .* (o.Dy * U.T) .- U.u2 .* U.T)
    xu1 = o.Dx * U.u1; yu1 = o.Dy * U.u1; xu2 = o.Dx * U.u2; yu2 = o.Dy * U.u2
    Fa = τ .* (xu1 .- yu2) .- 0.5 .* (U.u1 .^ 2 .- U.u2 .^ 2)
    Fb = τ .* (yu1 .+ xu2) .- U.u1 .* U.u2
    if a > 0
        S = sv.S
        FG1 = FG1 .- (e2t * a) .* (S * U.G1); FG2 = FG2 .- (e2t * a) .* (S * U.G2)
        Fa  = Fa  .- (e2t * a) .* (S * U.ta); Fb  = Fb  .- (e2t * a) .* (S * U.tb)
        R = (a .* (S * U.rho), a .* (S * U.u1), a .* (S * U.u2), a .* (S * U.T))
    else
        z = zeros(length(U.rho)); R = (z, z, z, z)
    end
    return (FG1, FG2, Fa, Fb), R
end

"One IMEX-RK step; U is a named tuple with the fields $(FIELDS)."
function imex_step(sv::Relaxation2D, U, dt)
    o = sv.ops; ε = sv.eps; e2t = ε^2 * sv.tau
    At, A, s = sv.At, sv.A, sv.s
    N = o.N
    p_n = U.rho .+ U.T
    umax = maximum(hypot.(U.u1, U.u2))
    a = stab_speed(sv, dt, umax)
    divu_n = o.Dx * U.u1 .+ o.Dy * U.u2

    @assert A[1, 1] == 0.0 "only ARS-type tableaux are implemented"
    st = Vector{Any}(undef, s)
    let
        E, R = explicit_terms(sv, U, a)
        qx = o.Dx * U.ta .+ o.Dy * U.tb; qy = o.Dx * U.tb .- o.Dy * U.ta     # ∇·Θ
        pt = p_n ./ ε
        st[1] = (U = U, E = E, R = R, divG = o.Dx * U.G1 .+ o.Dy * U.G2,
                 qx = qx, qy = qy, divq = o.Dx * qx .+ o.Dy * qy,
                 Lpt = o.Lap * pt, gx = o.Dx * pt, gy = o.Dy * pt, div = divu_n)
    end
    Xn = (U.G1, U.G2, U.ta, U.tb)
    for i in 2:s
        aii = A[i, i]; den = e2t + aii * dt
        # 1. Γ and Θ (relaxation implicit, the rest explicit)
        X = ntuple(4) do k
            num = e2t .* Xn[k]
            for j in 1:i-1
                At[i, j] != 0 && (num = num .- (dt * At[i, j]) .* st[j].E[k])
                A[i, j]  != 0 && (num = num .- (dt * A[i, j]) .* getfield(st[j].U, FIELDS[4 + k]))
            end
            num ./ den
        end
        G1, G2, ta, tb = X
        divG = o.Dx * G1 .+ o.Dy * G2
        qx = o.Dx * ta .+ o.Dy * tb; qy = o.Dx * tb .- o.Dy * ta
        divq = o.Dx * qx .+ o.Dy * qy
        # 2. predictors p*, u* and the discrete divergence of u*
        p_star = p_n .- (dt / DIM * aii) .* divG
        u1s = U.u1 .- (dt * aii) .* qx; u2s = U.u2 .- (dt * aii) .* qy
        div_star = divu_n .- (dt * aii) .* divq
        sum_div = zeros(N); Rrho = zeros(N)
        for j in 1:i-1
            sj = st[j]
            if A[i, j] != 0
                c = dt * A[i, j]
                p_star .-= (c / DIM) .* sj.divG
                u1s .-= c .* (sj.qx .+ sj.gx); u2s .-= c .* (sj.qy .+ sj.gy)
                div_star .-= c .* (sj.divq .+ sj.Lpt)
                sum_div .+= A[i, j] .* sj.div
            end
            if At[i, j] != 0 && a > 0
                c = dt * At[i, j]
                Rr, Ru1, Ru2, RT = sj.R
                Rrho .+= c .* Rr
                p_star .+= c .* (Rr .+ RT)
                u1s .+= c .* Ru1; u2s .+= c .* Ru2
                div_star .+= c .* (o.Dx * Ru1 .+ o.Dy * Ru2)
            end
        end
        # 3. Helmholtz problem  ε² p~ - c Δt² aᵢᵢ² Δ p~ = rhs   (mean of p conserved)
        rhs = ε .* p_star .- (CP * dt) .* sum_div .- (CP * dt * aii) .* div_star
        α = ε^2; β = CP * (dt * aii)^2
        pm = sum(p_star) / N / ε
        sol = helmholtz_lu(sv, α, β) \ vcat(rhs .- α * pm, 0.0)
        pt = pm .+ sol[1:end-1]
        Lpt = o.Lap * pt; gx = o.Dx * pt; gy = o.Dy * pt
        div = div_star .- (dt * aii) .* Lpt
        # 4. velocity
        u1 = u1s .- (dt * aii) .* gx; u2 = u2s .- (dt * aii) .* gy
        # 5. density and temperature
        rho = U.rho .+ Rrho .+ (ε .* pt .- p_star) ./ CP
        T = ε .* pt .- rho
        Ui = (rho = rho, u1 = u1, u2 = u2, T = T, G1 = G1, G2 = G2, ta = ta, tb = tb)
        E, R = explicit_terms(sv, Ui, a)
        st[i] = (U = Ui, E = E, R = R, divG = divG, qx = qx, qy = qy, divq = divq,
                 Lpt = Lpt, gx = gx, gy = gy, div = div)
    end
    return st[s].U                       # globally stiffly accurate: last stage
end

# -----------------------------------------------------------------------------
# Initial data (fluctuations ρ, u1, u2, T)
# -----------------------------------------------------------------------------
"""
    shear_layer(x, y, ε)

Doubly periodic double shear layer: ρ = 1, T = 0,
u1 = tanh(30(y - 1/4)) for y ≤ 1/2, tanh(30(3/4 - y)) otherwise, u2 = 0.05 sin(2πx).
`make_shear_layer(k, δ)` for other steepness k and perturbation δ.
"""
shear_layer(x, y, ε) = make_shear_layer(30.0, 0.05)(x, y, ε)

make_shear_layer(k, δ) = (x, y, ε) -> (1.0,
    y <= 0.5 ? tanh(k * (y - 0.25)) : tanh(k * (0.75 - y)),
    δ * sin(2π * x), 0.0)

"""
    make_step_shear_layer(U = 0.5, a = 0.01)

Double vortex sheet: u1 = U for y ≤ 1/4 and y > 3/4, -U in between, plus
a·sin(2πx) in u1 and u2; ρ = T = 0. `step_shear_layer` uses the defaults.
"""
make_step_shear_layer(U = 0.5, a = 0.01) = function (x, y, ε)
    aa = a * sin(2π * x)
    return (0.0, (y <= 0.25 || y > 0.75) ? U + aa : -U + aa, aa, 0.0)
end
step_shear_layer(x, y, ε) = make_step_shear_layer()(x, y, ε)

"Shear wave u1 = A sin(2πy): incompressible limit u1 = A e^{-4π²τt} sin(2πy)."
shear_wave(A = 1.0) = (x, y, ε) -> (0.0, A * sin(2π * y), 0.0, 0.0)

"Taylor–Green vortex: incompressible limit decays like e^{-8π²τt}."
taylor_green(A = 1.0) = (x, y, ε) ->
    (0.0, A * sin(2π * x) * cos(2π * y), -A * cos(2π * x) * sin(2π * y), 0.0)

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
lattice(n, L) = ([(a - 0.5) * L / n for b in 1:n for a in 1:n],
                 [(b - 0.5) * L / n for b in 1:n for a in 1:n])

function init_state(x, y, p::Params)
    N = length(x)
    v = [p.init(x[i], y[i], p.eps) for i in 1:N]
    z = zeros(N)
    return (rho = [t[1] for t in v], u1 = [t[2] for t in v], u2 = [t[3] for t in v],
            T = [t[4] for t in v], G1 = copy(z), G2 = copy(z), ta = copy(z), tb = copy(z))
end

"Assemble the result (fields at the lattice points, vorticity, kinetic fields)."
function make_result(o::WLSOps2D, U, p, t)
    ω = o.Dx * U.u2 .- o.Dy * U.u1
    ε = p.eps
    ρF = 1 .+ ε .* U.rho
    return (x = o.x, y = o.y, rho = U.rho, u1 = U.u1, u2 = U.u2, T = U.T, omega = ω,
            G1 = U.G1, G2 = U.G2, theta_a = U.ta, theta_b = U.tb,
            rhoF = ρF, uxF = ε .* U.u1 ./ ρF, uyF = ε .* U.u2 ./ ρF, t = t)
end

"""
    run_case(p::Params; verbose = true) -> named tuple

Run one case on the Nx×Nx lattice. The result holds `x, y`, the fluctuations
`rho, u1, u2, T`, the vorticity `omega`, the moments `G1, G2, theta_a, theta_b`,
the kinetic fields `rhoF, uxF, uyF`, the final time `t`, the time step `dt`,
and, if files were written, `file` and `snapshots`.
"""
function run_case(p::Params; verbose = true)
    x, y = lattice(p.Nx, p.L)
    tb = time()
    o  = WLSOps2D(x, y; L = p.L, hratio = p.hratio, min_nb = p.min_nb, alpha = p.alpha)
    sv = Relaxation2D(o, p.eps, p.tau; scheme = p.scheme, stab = p.stab)
    U  = init_state(x, y, p)
    umax = max(1.0, maximum(hypot.(U.u1, U.u2)))
    Ma = p.eps * maximum(hypot.(U.u1, U.u2))
    Ma >= 0.6 && @warn @sprintf("ε·max|u| = %.2f. The relaxation system needs ε·max|u| < 1 (shear waves travel with speed 1/ε); close to 1 the solution is poor and above 1 it blows up. Use a smaller ε.", Ma)
    dt  = p.dt > 0 ? p.dt : stable_dt(sv, umax; cfl = p.cfl, cadv = p.cadv)
    nst = ceil(Int, p.tfinal / dt - 1e-9); dt = p.tfinal / nst
    np  = p.nprint > 0 ? p.nprint : max(1, nst ÷ 20)
    ts  = vcat(p.save_times, p.nsave > 0 ? [k * p.tfinal / p.nsave for k in 1:p.nsave-1] : Float64[])
    savesteps = Set(clamp(round(Int, τs / dt), 1, nst - 1) for τs in ts if 0 < τs < p.tfinal)
    snaps = String[]
    m0 = sum(U.rho) / o.N

    verbose && @printf("relaxation 2D  ε=%g τ=%g  Nx=%d  %s  dt=%.3e  steps=%d  threads=%d  (operators %.1fs)\n",
                       p.eps, p.tau, p.Nx, p.scheme, dt, nst, nthreads(), time() - tb)
    t = 0.0; tstart = time()
    for n in 1:nst
        U = imex_step(sv, U, dt)
        t += dt
        all(isfinite, U.u1) || error(@sprintf("non-finite solution at step %d, t = %.4f (ε·max|u| must stay below 1; try a smaller ε or dt)", n, t))
        if verbose && (n % np == 0 || n == nst)
            @printf("  step %6d  t=%.4f  max|u|=%.3e  max|∇·u|=%.2e  mean ρ err=%.2e  (%.1fs)\n",
                    n, t, maximum(hypot.(U.u1, U.u2)),
                    maximum(abs, o.Dx * U.u1 .+ o.Dy * U.u2), sum(U.rho) / o.N - m0,
                    time() - tstart)
        end
        if p.write_output && n in savesteps
            push!(snaps, output(make_result(o, U, p, t), p, @sprintf("_t%.4g", t); verbose = verbose))
        end
    end
    res = merge(make_result(o, U, p, t), (dt = dt,))
    if p.write_output
        res = merge(res, (file = output(res, p, ""; verbose = verbose), snapshots = snaps))
    end
    return res
end

"Base name of the output files: relax2d_eps<ε>_tau<τ>_Nx<Nx>."
file_tag(p::Params) = @sprintf("relax2d_eps%g_tau%g_Nx%d", p.eps, p.tau, p.Nx)

"""
    write_results(res, p; suffix = "", verbose = true) -> path

Write `<tag><suffix>_uniform.dat` to `p.outdir` (lattice values, x fastest)
with columns x  y  ρ  u1  u2  T  ω. The first line records the run parameters.
The format is the same as in ALEBGK2D, so its plot functions can be used.
"""
function write_results(res, p::Params; suffix = "", verbose = true)
    mkpath(p.outdir)
    file = abspath(joinpath(p.outdir, file_tag(p) * suffix * "_uniform.dat"))
    hdr = @sprintf("# x y rho u1 u2 T omega   eps=%g tau=%g t=%g Nx=%d scheme=%s frame=Eulerian model=relaxation",
                   p.eps, p.tau, res.t, p.Nx, p.scheme)
    cols = (res.x, res.y, res.rho, res.u1, res.u2, res.T, res.omega)
    open(file, "w") do io
        println(io, hdr)
        for i in eachindex(res.x)
            println(io, join((@sprintf("%.10e", c[i]) for c in cols), " "))
        end
    end
    verbose && println("  wrote ", file)
    return file
end

"Function `file -> nothing` called after each output (set by start.jl to plot)."
const PLOT_HOOK = Ref{Any}(nothing)

function output(res, p, suffix; verbose = true)
    file = write_results(res, p; suffix = suffix, verbose = verbose)
    if p.plot && PLOT_HOOK[] !== nothing
        try
            Base.invokelatest(PLOT_HOOK[], file)
        catch err
            @warn "plotting failed" exception = err
        end
    end
    return file
end

end # module
