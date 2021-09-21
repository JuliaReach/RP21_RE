# ==================
# Dependencies
# ==================
using ReachabilityAnalysis, CarlemanLinearization
using Plots, LaTeXStrings, LinearAlgebra, SparseArrays
using Plots.PlotMeasures
using DifferentialEquations

using CarlemanLinearization: _error_bound_specabs_R

include("../utils.jl")

# ==================
# Model definition
# ==================

# the following are constants of the model
# the domain of x is [-L0/2, L0/2]
L0 = 1.0     # domain length
ν = 0.05     # kinematic viscosity
U0 = 1.0
Re = 20
ν = U0 * L0 / Re
Tnl = L0 / U0
Tmax = Tnl/2

# ================================
# Spatial flowpipe visualization
# ================================

# no-op
ReachabilityAnalysis.convexify(R::ReachabilityAnalysis.AbstractReachSet) = R

function flowpipe_spatial(sol, t, n)
    solt = sol(t) |> convexify

    xdom = range(-L0/2, L0/2, length=n)
    xdomred = xdom[2:end-1]

    X = [set(overapproximate(Projection(solt, i:i), Interval)) for i in 1:dim(sol)]
    R = ReachSet(CartesianProductArray(X), tspan(solt))
    arr = [Interval(xdomred[i]) × CartesianProductArray(X).array[i] for i in 1:length(xdomred)]
    arr = vcat(Interval(xdom[1]) × Interval(0), arr, Interval(xdom[end]) × Interval(0))
    U = UnionSetArray(arr)
    Uch = [ConvexHull(U.array[i], U.array[i+1]) for i in 1:length(U.array)-1] |> UnionSetArray

    return Uch
end

# model for solution with TM
@taylorize function burgers!(du, u, p, t)
    local nf = length(u)
    n = nf + 2
    local Δx = L0/(n-1)
    local c1 = ν/Δx^2
    local c2 = 1/(4Δx)

    du[1] = c1*(-2*u[1]+u[2]) - c2*(u[2]^2)
    du[nf] = c1*(u[nf-1]-2*u[nf]) - c2*(-u[nf-1]^2)

    for i in 2:nf-1
        du[i] = c1*(u[i-1]-2*u[i]+u[i+1]) - c2*(u[i+1]^2 - u[i-1]^2)
    end
end


# ====================
# Carleman method
# ====================

using DynamicPolynomials
using MultivariatePolynomials

# FIXME refactor to RA / CarlemanLinearization
function ___findfirst(y::Vector{<:AbstractMonomialLike}, x)
    ypow = powers.(y)

    xvars = variables(x)
    xpow = exponents(x)

    for (i, pi) in enumerate(ypow)
        if pi.is[1] == xvars && pi.is[2] == xpow
            return i
        end
    end
    return nothing
end

# we remove the endpoints
function burguers_carlin(nf)
    n = nf + 2
    Δx = L0/(n-1)
    c1 = ν/Δx^2
    c2 = 1/(4Δx)

    # fix the endpoints
    n = nf # rename to simplify the rest of the code
    dv = c1 * fill(-2.0, n)
    ev = c1 * fill(1.0, n-1)

    F1 = SymTridiagonal(dv, ev)

    F2 = zeros(n, n^2)
    x, = @polyvar x[1:n]
    y = kron(x, x)

    idx = ___findfirst(y, x[2]^2)
    F2[1, idx] = 1.0

    idx = ___findfirst(y, x[n]^2)
    F2[n, idx] = -1.0

    for i in 2:n-1
        idx = ___findfirst(y, x[i-1]^2)
        F2[i, idx] = -1.0

        idx = ___findfirst(y, x[i+1]^2)
        F2[i, idx] = 1.0
    end
    F2 .*= -c2
    F2 = sparse(F2)

    return F1, F2
end

# ===================
# Solution method
# ===================

function _solve_burgers_carlin(; n, N, width0=0, δ=0.02)

    nf = n-2
    F1, F2 = burguers_carlin(nf)

    A = build_matrix(F1, F2, N)

    xdom = range(-L0/2, L0/2, length=n)
    c0 = Singleton(-U0*sin.(2*π*xdom[2:end-1]/L0))
    if width0 == 0
        X0 = overapproximate(c0, Hyperrectangle);
    else
        H0 = Hyperrectangle(zeros(nf), fill(width0, nf))
        X0 = overapproximate(c0 ⊕ H0, Hyperrectangle)
    end

    dirs = _template(n=nf, N=N)
    # amodel = Forward(inv=true, setops=:lazy, sih=:concrete) # FIXME set options
    alg = LGG09(δ=δ, template=dirs, approx_model=Forward())

    sol_carlin = _solve_CARLIN(X0, F1, F2; alg=alg, N=N, T=Tmax, bloat=false, kron_pow_algorithm="symbolic");
    sol_carlin_spatial = flowpipe_spatial(sol_carlin, Tnl/2, n);
end

# ===================
# TM solution method
# ===================

function _solve_burgers_TM(; n, width0=0)

    nf = n - 2

    xdom = range(-L0/2, L0/2, length=n)
    c0 = Singleton(-U0*sin.(2*π*xdom[2:end-1]/L0))
    if width0 == 0
        X0 = overapproximate(c0, Hyperrectangle)
    else
        H0 = Hyperrectangle(zeros(nf), fill(width0, nf))
        X0 = overapproximate(c0 ⊕ H0, Hyperrectangle)
    end

    prob = IVP(BlackBoxContinuousSystem(burgers!, nf), X0)
    solTM = solve(prob, tspan=(0.0, Tmax), alg=TMJets20(), ensemble=true, trajectories=1);
    solzTM = overapproximate(solTM, Zonotope);

    sol_spatial = flowpipe_spatial(solzTM, Tnl/2, n);
end

function _get_R(; n)
    nf = n-2
    F1, F2 = burguers_carlin(nf);
    xdom = range(-L0/2, L0/2, length=n)
    x0 = -U0*sin.(2*π*xdom[2:end-1]/L0);
    R, Re_λ1 = _error_bound_specabs_R(x0, Matrix(F1), Matrix(F2); check=true)
end

# ================================
# Results
# ================================

n = 10

# Initial point
_solve_burgers_TM(n=n); tt = @elapsed _solve_burgers_TM(n=n)
print(io, "Burgers, TMJets, initial point, -, $(tt)\n")

_solve_burgers_carlin(n=n, N=2, δ=0.01); tt = @elapsed _solve_burgers_carlin(n=n, N=2, δ=0.01)
print(io, "Burgers, Carleman, initial point, N=2, $(tt)\n")

_solve_burgers_carlin(n=n, N=3, δ=0.01); tt = @elapsed _solve_burgers_carlin(n=n, N=3, δ=0.01)
print(io, "Burgers, Carleman, initial point, N=3, $(tt)\n")

# Initial set
width0 = 0.03
_solve_burgers_TM(n=n, width0=width0); tt = @elapsed _solve_burgers_TM(n=n, width0=width0)
print(io, "Burgers, TMJets, initial set, -, $(tt)\n")

_solve_burgers_carlin(n=n, N=2, δ=0.01, width0=width0); tt = @elapsed _solve_burgers_carlin(n=n, N=2, δ=0.01, width0=width0)
print(io, "Burgers, Carleman, initial set, N=2, $(tt)\n")

_solve_burgers_carlin(n=n, N=3, δ=0.01, width0=width0); tt = @elapsed _solve_burgers_carlin(n=n, N=3, δ=0.01, width0=width0)
print(io, "Burgers, Carleman, initial set, N=3, $(tt)\n")

# ==================
# Figures
# ==================

# initial point
function figure_burger_point()

    fig = plot(legend=:topright, xlab=L"x", ylab=L"u(x, 0.5)",
               legendfontsize=25,
               tickfont=font(25, "Times"),
               guidefontsize=25,
               xguidefont=font(35, "Times"),
               yguidefont=font(35, "Times"),
               xtick = xticklatex([-0.50, -0.25, 0.0, 0.25, 0.50], 2),
               ytick = xticklatex([-1.0, -0.5, 0.0, 0.5, 1.0], 2),
               xlims=(-0.5, 0.5),
               ylims=(-1.05, 1.0),
               bottom_margin=10mm, left_margin=6mm, right_margin=12mm, top_margin=3mm, size=(900, 600))

    # model values
    n = 10

    # initial condition (smooth)
    xdom = range(-L0/2, L0/2, length=200)
    c0 = Singleton(-U0*sin.(2*π*xdom/L0))
    plot!(fig, xdom, element(c0), seriestype=:path, c=:black, marker=:none, ls=:dash, lab=L"u(x, 0)", lw=2.0, alpha=1.)

    # initial cond, samples
    xdom = range(-L0/2, L0/2, length=n)
    c0 = Singleton(-U0*sin.(2*π*xdom/L0))
    plot!(fig, xdom, element(c0), seriestype=:path, c=:grey, marker=:square, ls=:dash, lab="", lw=2.0, alpha=1.)

    solN2_carlin = _solve_burgers_carlin(n=n, N=2, δ=0.01)
    solN3_carlin = _solve_burgers_carlin(n=n, N=3, δ=0.01);

    sol_TM = _solve_burgers_TM(n=n);

    plot!(fig, solN3_carlin, c=:orange, lab=L"N = 3", alpha=1., lc=:orange)
    plot!(fig, solN2_carlin, c=:red, lab=L"N = 2", alpha=1., lc=:red)
    plot!(fig, sol_TM, c=:blue, lab=L"\textrm{TM}", alpha=1., lc=:blue)

    return fig
end

fig = figure_burger_point()
savefig(fig, joinpath(TARGET_FOLDER, "figure_burger_point.pdf"))

# ==================
# initial set
# ==================

function figure_burger_set()

    fig = plot(legend=:topright, xlab=L"x", ylab=L"u(x, 0.5)",
               legendfontsize=25,
               tickfont=font(25, "Times"),
               guidefontsize=25,
               xguidefont=font(35, "Times"),
               yguidefont=font(35, "Times"),
               xtick = xticklatex([-0.50, -0.25, 0.0, 0.25, 0.50], 2),
               ytick = xticklatex([-1.0, -0.5, 0.0, 0.5, 1.0], 2),
               xlims=(-0.5, 0.5),
               ylims=(-1.05, 1.0),
               bottom_margin=10mm, left_margin=6mm, right_margin=12mm, top_margin=3mm, size=(900, 600))

    # model values
    n = 10
    width0 = 0.03

    # initial condition (smooth)
    xdom = range(-L0/2, L0/2, length=200)
    c0 = Singleton(-U0*sin.(2*π*xdom/L0))
    plot!(fig, xdom, element(c0), seriestype=:path, c=:black, marker=:none, ls=:dash, lab=L"u(x, 0)", lw=2.0, alpha=1.)

    # initial cond, samples
    xdom = range(-L0/2, L0/2, length=n)
    c0 = Singleton(-U0*sin.(2*π*xdom/L0))
    plot!(fig, xdom, element(c0), seriestype=:path, c=:grey, marker=:square, ls=:dash, lab="", lw=2.0, alpha=1.)

    solN2_carlin = _solve_burgers_carlin(n=n, N=2, width0=width0, δ=0.01)
    solN3_carlin = _solve_burgers_carlin(n=n, N=3, width0=width0, δ=0.01);

    sol_TM = _solve_burgers_TM(n=n, width0=width0);

    plot!(fig, solN3_carlin, c=:orange, lab=L"N = 3", alpha=1., lc=:orange)
    plot!(fig, solN2_carlin, c=:red, lab=L"N = 2", alpha=1., lc=:red)
    plot!(fig, sol_TM, c=:blue, lab=L"\textrm{TM}", alpha=1., lc=:blue)

    return fig
end

fig = figure_burger_set()
savefig(fig, joinpath(TARGET_FOLDER, "figure_burger_set.pdf"))
