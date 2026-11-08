# RelaxWLS2D.jl

A meshfree Eulerian solver for the two-dimensional relaxation (moment) system of
the BGK model in the low-Mach / small-Knudsen scaling on a doubly periodic box,
written in pure Julia (standard library only). It is the two-dimensional
extension of RelaxWLS1D: weighted least squares (GFDM) derivatives on fixed
points and an asymptotic-preserving IMEX Runge–Kutta scheme whose time step
does not depend on $\varepsilon$.

## Model

On $[0,L]^2$ with periodic boundary conditions, $d = 2$:

$$
\begin{aligned}
\varepsilon\,\partial_t\rho + \nabla\cdot u &= 0,\\
\varepsilon\,\partial_t u + \varepsilon\,\nabla\cdot\Theta + \nabla p &= 0, \qquad p=\rho+T,\\
\partial_t T + \tfrac{2}{\varepsilon d}\,\nabla\cdot u + \tfrac{1}{d}\,\nabla\cdot\Gamma &= 0,\\
\varepsilon^2\partial_t\Theta + D(\nabla u) &= -\tfrac{1}{\tau}\big(\Theta - A(u)\big),\\
\varepsilon^2\partial_t\Gamma + (d+2)\nabla T &= -\tfrac{1}{\tau}\big(\Gamma - (d+2)\,uT\big),
\end{aligned}
$$

with $D(\nabla u) = \nabla u + \nabla u^{\mathsf T} - \tfrac{2}{d}(\nabla\cdot u)I$ and
$A(u) = u\otimes u - \tfrac{|u|^2}{d}I$. Here $\rho, u, T$ are fluctuations around the
global equilibrium, $\Gamma$ is the heat flux and
$\Theta = \begin{pmatrix}\theta_a & \theta_b\\ \theta_b & -\theta_a\end{pmatrix}$
the trace-free stress. As $\varepsilon\to0$, $\nabla\cdot u = 0$, $\rho+T = 0$, and
$u$ solves the incompressible Navier–Stokes equations with viscosity $\tau$:

$$
\partial_t u + \nabla\cdot(u\otimes u) + \nabla\Pi = \tau\,\Delta u,\qquad
\partial_t T + \nabla\cdot(uT) = \tau\,\Delta T .
$$

The system is hyperbolic. Its shear waves travel with speed $1/\varepsilon$, so it
is a good approximation of the kinetic model only as long as
$\varepsilon\,\max|u| < 1$ (subcharacteristic condition). For larger values the
equations themselves become unstable, whatever the discretisation. `run_case`
warns when $\varepsilon\max|u_0| \ge 0.6$.

## Method

| Ingredient | Choice |
|---|---|
| Points | Fixed uniform lattice, `Nx × Nx` points |
| Neighbours | Periodic cell lists, radius $r=\Delta x/0.35$, enlarged until each point has `min_nb` neighbours |
| Derivatives | Quadratic weighted least-squares Taylor fit (GFDM), Gaussian weight $e^{-\alpha d^2/r^2}$, $\alpha = 6$; sparse matrices $\partial_x$, $\partial_y$ |
| Laplacian | $\partial_x\partial_x+\partial_y\partial_y$ (products of the first-derivative matrices) in the pressure step, so that it is consistent with the discrete divergence and gradient. The compact $\partial_{xx}+\partial_{yy}$ of the fit would reduce the scheme to first order in time |
| Time | Globally stiffly accurate IMEX Runge–Kutta: `ARS111` (1st), `ARS222` (2nd, default), `ARS443` (3rd order). Implicit: acoustics, $\nabla\cdot\Theta$, $\nabla\cdot\Gamma$ and the relaxation; explicit: $D(\nabla u)$, $A(u)$, $(d+2)\nabla T$, $(d+2)uT$ |
| Pressure | One Helmholtz problem per stage for $\tilde p = p/\varepsilon$, $\varepsilon^2\tilde p - \tfrac{d+2}{d}\,a_{ii}^2\Delta t^2\,\Delta\tilde p = \dots$, sparse LU (factorised once), mean of $p$ conserved |
| Stabilisation | Explicit third-order upwind-type dissipation $-a\,\tfrac{\Delta x^3}{12}\Delta^2$ on all fields, $a = \max|u| + \tfrac{\sqrt{(d+2)/d}}{\varepsilon}\tfrac{\varepsilon^2\tau}{\varepsilon^2\tau+\Delta t}$ (switch off with `stab = false`) |
| Time step | $2d\tfrac{d+2}{d}\tau\Delta t^2/(\varepsilon^2\tau+\Delta t)\le$ `cfl`$\,\Delta x^2$, $\Delta t\le$ `cadv`$\,\Delta x/(\max|u|+a_\varepsilon)$ and the explicit dissipation bound. None of them depends on small $\varepsilon$ |

## Installation

```bash
git clone https://github.com/sudarshan825/2d_relaxation_system.git
cd 2d_relaxation_system
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The solver needs only the Julia standard library. Plotting uses
[Plots.jl](https://github.com/JuliaPlots/Plots.jl).

## Quick start

Start Julia with all cores, `julia -t auto`, then

```julia
include("path/to/RelaxWLS2D/start.jl")
res = run_case(Params())                     # ε = 0.1, τ = 0.001, 100×100, t = 1.5
res = run_case(Params(eps = 1e-3))           # close to the Navier–Stokes limit
```

`start.jl` activates the package, installs Plots.jl the first time, loads the
solver and the plot functions and sets `R` to the folder `results/` of the
package. Every run writes its data and a vorticity plot to `R`.

`res` is a named tuple with the point positions `x, y`, the fluctuations
`rho, u1, u2, T`, the vorticity `omega`, the moments `G1, G2, theta_a, theta_b`,
the kinetic fields `rhoF = 1 + ερ`, `uxF, uyF = εu/ρF`, the final time `t`, the
time step `dt`, the output file `file` and the snapshot files `snapshots`.

From a terminal:

```bash
julia -t auto examples/shear_layer.jl                     # ε = 0.1, τ = 0.001
julia -t auto examples/shear_layer.jl 0.001 0.001         # ε τ [Nx [scheme]]
julia -t auto examples/shear_layer.jl 0.1 0.001 200 ARS443
julia examples/plot_results.jl                            # plot every run in results/
```

### Initial data

The default is the doubly periodic double shear layer: $\rho = 1$, $T = 0$,

$$
u_1 = \begin{cases}\tanh\big(30(y-\tfrac14)\big), & y\le \tfrac12,\\[2pt] \tanh\big(30(\tfrac34-y)\big), & y>\tfrac12,\end{cases}
\qquad u_2 = 0.05\,\sin(2\pi x).
$$

| `init =` | Data |
|---|---|
| `shear_layer` (default), `make_shear_layer(k, δ)` | shear layer with steepness `k` and perturbation `δ` |
| `step_shear_layer`, `make_step_shear_layer(U, a)` | vortex sheet u₁ = ±U, perturbation a·sin(2πx) in u₁ and u₂ |
| `shear_wave(A)` | u₁ = A sin(2πy); for small ε it decays like $e^{-4\pi^2\tau t}$ |
| `taylor_green(A)` | Taylor–Green vortex; for small ε it decays like $e^{-8\pi^2\tau t}$ |
| `(x, y, ε) -> (ρ, u1, u2, T)` | any function returning the fluctuations |

### Plotting

```julia
plot_file(res.file)                    # vorticity (also saved as PNG)
plot_file(res.file; field = :u1)       # :omega, :rho, :u1, :u2, :T
compare_files(fileA, fileB; save = joinpath(R, "compare.png"))   # same contour levels
plot_all(R)                            # every run in a folder
```

The output format is the same as that of the BGK solver ALEBGK2D
(`sudarshan825/2d_BGK`), so `compare_files(relaxation_file, bgk_file)` puts the
two models side by side.

## Parameters

| Field | Default | Meaning |
|---|---|---|
| `eps` | `0.1` | scaling parameter ε (needs ε·max\|u\| < 1) |
| `tau` | `0.001` | relaxation time τ (viscosity of the limit) |
| `L` | `1.0` | side of the periodic box |
| `Nx` | `100` | points per direction (`Nx²` points) |
| `tfinal` | `1.5` | final time |
| `scheme` | `"ARS222"` | `"ARS111"`, `"ARS222"` or `"ARS443"` |
| `hratio` | `0.35` | `Δx = hratio·r` |
| `min_nb` | `12` | minimum number of neighbours |
| `alpha` | `6.0` | Gaussian weight parameter |
| `stab` | `true` | third-order upwind-type dissipation |
| `cfl` | `0.8` | diffusive time-step factor |
| `cadv` | `0.3` | transport time-step factor |
| `dt` | `0.0` | fixed time step if `> 0` |
| `init` | `shear_layer` | initial fluctuations `(x, y, ε) -> (ρ, u1, u2, T)` |
| `outdir` | `<package>/results` | output folder (created if missing) |
| `write_output` | `true` | write `.dat` files |
| `nsave` | `0` | number of equally spaced snapshots |
| `save_times` | `[]` | snapshots at these times |
| `plot` | `true` | plot each output when a plot function is set (`start.jl`) |
| `nprint` | `0` | progress interval in steps (`0`: about 20 messages) |

## Output

Each output is written to `outdir` as `<tag>_uniform.dat`, with columns
`x  y  rho  u1  u2  T  omega` (x fastest) and
`<tag> = relax2d_eps<ε>_tau<τ>_Nx<Nx>` (snapshots add `_t<t>`). The first line
records the run parameters.

## Remarks

- Double shear layer, τ = 0.001, t = 1.5, 100×100 points: for ε = 10⁻³ the
  maximal vorticity is 17.4, against 17.5 for a pseudo-spectral solution
  of the incompressible Navier–Stokes equations on 128² modes, and the contours agree. For
  ε = 0.1 it is 17.2. About 500 time steps for every ε.
- For ε = 0.5 compressibility effects are clearly visible. Around ε = 0.7 the
  shear layer (max|u| ≈ 1.3) comes close to the limit ε·max|u| = 1, and for
  ε = 1 the relaxation system blows up. Use the kinetic solver there.
- The cost per step is one sparse back substitution per stage and a few
  sparse matrix–vector products; the LU factorisation is done once.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The tests check the WLS operators, the global equilibrium, the decay of a shear
wave (ε = 10⁻³ and 0.1) and of the Taylor–Green vortex (ε = 10⁻⁴, asymptotic
preserving), mass conservation, the temporal order 1, 2, 3 of the IMEX
schemes, and the output files.

## License

MIT – see [LICENSE](LICENSE).
