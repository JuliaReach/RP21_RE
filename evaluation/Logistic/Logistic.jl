# =================
# Dependencies
# =================

using ReachabilityAnalysis, CarlemanLinearization
using Plots, LaTeXStrings, LinearAlgebra, SparseArrays
using Plots.PlotMeasures

using LazySets: center
using CarlemanLinearization: _error_bound_specabs_R

include("../utils.jl")

# =================
# Model definition
# =================

function quadratic(; a, b, N)
    dl = zeros(N-1)
    d = [a*i for i in 1:N]
    du = [b*i for i in 1:N-1]
    return Tridiagonal(dl, d, du)
end

function logistic(; r, K, N)
    return quadratic(a=r, b=-r/K, N=N)
end

# define model
r = -0.5
K = 0.8

# standard form
F1 = hcat(r)
F2 = hcat(-r/K)
a = r
b = -r/K
A = logistic(r=r, K=K, N=4);

x0 = interval(0.5)
R, Re_λ1 = _error_bound_specabs_R(x0, F1, F2; check=true)

using Revise
Tmax = 10.0

x0set = interval(0.47, 0.53)

NSTEPS = 1000

# analytic solution
u0 = mid(x0)
tsp = range(0.0, Tmax, length=NSTEPS+1)
f_analytic(t, u0) = u0 * a * exp(a * t) / (a + b*(1 - exp(a*t))*u0)
sol_a = f_analytic.(tsp, u0)

# solution with Carleman, discrete time
function sol_carlin_discrete_time(; N)
    A = logistic(r=r, K=K, N=N)
    y0 = center(kron_pow_stack(x0, N)) |> Singleton
    prob = @ivp(y' = A * y, y(0) ∈ y0, dim=N)

    return solve(prob, T=Tmax, alg=ORBIT(δ=0.5, approx_model=NoBloating()))
end

# solution with Carleman, continuous time
function sol_carlin_continuous_time(; N)
    F1 = hcat(r); F2 = hcat(-r/K)
    dirs = _template(n=1, N=N)
    alg = LGG09(δ=0.1, template=dirs, approx_model=Forward())
    return _solve_CARLIN(x0set, F1, F2; alg=alg, N=N, T=Tmax, bloat=false)
end

# ============
# Figure 1a
# ============

function figure_1a()
    sol_discr_N1 = sol_carlin_discrete_time(N=1)
    sol_discr_N2 = sol_carlin_discrete_time(N=2)
    sol_discr_N3 = sol_carlin_discrete_time(N=3)
    sol_discr_N4 = sol_carlin_discrete_time(N=4)
    sol_discr_N5 = sol_carlin_discrete_time(N=5)
    sol_discr_N6 = sol_carlin_discrete_time(N=6);

    fig = plot(legend=:topright, xlab=L"t", ylab=L"x(t)",
               legendfontsize=25,
               tickfont=font(25, "Times"),
               guidefontsize=25,
               xguidefont=font(35, "Times"),
               yguidefont=font(35, "Times"),
               xtick=([0.0, 2.5, 5.0, 7.5, 10.0], [L"0", L"2.5", L"5.0", L"7.5", L"10.0"]),
               ytick=([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [L"0", L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"]),
               xlims=(0.0, 10.1),
               ylims=(-0.01, 0.5),
               bottom_margin=10mm, left_margin=2mm, right_margin=12mm, top_margin=3mm, size=(900, 600))

    plot!(fig, tsp, sol_a, c=:magenta, lab=L"\textrm{Analytic}", lw=3.0, alpha=1.)

    plot!(fig, sol_discr_N1, vars=(0, 1), lab=L"N=1", seriestype=:path, linestyle=:dash, c=:green, markersize=8, markershape=:circle, lw=2.0, alpha=1.)
    plot!(fig, sol_discr_N2, vars=(0, 1), lab=L"N=2", seriestype=:path, linestyle=:dash, c=:red, markersize=8, markershape=:utriangle, lw=2.0, alpha=1.)
    #plot!(fig, sol_discr_N3, vars=(0, 1), lab="N=3", seriestype=:path, c=:red, marker=:circle, lw=2.0, alpha=1.)
    plot!(fig, sol_discr_N4, vars=(0, 1), lab=L"N=4", seriestype=:path, linestyle=:dash, c=:blue, markersize=8, markershape=:rect, lw=2.0, alpha=1.)
    #plot!(fig, sol_discr_N5, vars=(0, 1), lab="N=5", seriestype=:path, c=:blue, marker=:star, alpha=1.)

    plot!(fig, sol_discr_N6, vars=(0, 1), lab=L"N=6", seriestype=:path, linestyle=:dash, c=:orange, markersize=8, markershape=:star, lw=2.0, alpha=1.)



end

fig = figure_1a()
savefig(fig, joinpath(TARGET_FOLDER, "figure_1a.pdf"))

# ============
# Figure 1b
# ============

function figure_1b()
    sol_discr_N1 = sol_carlin_discrete_time(N=1)
    sol_discr_N2 = sol_carlin_discrete_time(N=2)
    sol_discr_N3 = sol_carlin_discrete_time(N=3)
    sol_discr_N4 = sol_carlin_discrete_time(N=4)
    sol_discr_N5 = sol_carlin_discrete_time(N=5)
    sol_discr_N6 = sol_carlin_discrete_time(N=6);

    fig = plot(legend=:topright, xlab=L"t", ylab=L"x(t)",
               legendfontsize=25,
               tickfont=font(25, "Times"),
               guidefontsize=25,
               xguidefont=font(35, "Times"),
               yguidefont=font(35, "Times"),
               xtick=([0.0, 2.5, 5.0, 7.5, 10.0], [L"0", L"2.5", L"5.0", L"7.5", L"10.0"]),
               ytick=([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [L"0", L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"]),
               xlims=(0.0, 10.1),
               ylims=(-0.01, 0.6),
               bottom_margin=10mm, left_margin=2mm, right_margin=12mm, top_margin=3mm, size=(900, 600))

    tsp = range(0.0, Tmax, length=NSTEPS+1)

    sol_a_high = f_analytic.(tsp, x0set.hi)
    sol_a_low = f_analytic.(tsp, x0set.lo)

    # initial set
    plot!(Singleton([0.0]) × Interval(x0set), lw=8.0, alpha=1., c=:black, lab="", markershape=:none, seriestype=:shape, xlims=(-0.05, 10.1))

    plot!(fig, sol_carlin_continuous_time(N=1), vars=(0, 1), lab="", c=:green, lc=:green)
    plot!(fig, sol_carlin_continuous_time(N=2), vars=(0, 1), lab="", c=:red, lc=:red)
    plot!(fig, sol_carlin_continuous_time(N=6), vars=(0, 1), lab="", c=:orange, lc=:orange)

    plot!(fig, sol_discr_N1, vars=(0, 1), lab=L"N=1", seriestype=:path, linestyle=:dash, c=:green, markersize=8, markershape=:circle, lw=2.0, alpha=1.)
    plot!(fig, sol_discr_N2, vars=(0, 1), lab=L"N=2", seriestype=:path, linestyle=:dash, c=:red, markersize=8, markershape=:utriangle, lw=2.0, alpha=1.)
    plot!(fig, sol_discr_N6, vars=(0, 1), lab=L"N=6", seriestype=:path, linestyle=:dash, c=:orange, markersize=8, markershape=:star, lw=2.0, alpha=1.)

    plot!(fig, tsp, sol_a_high, c=:magenta, lab=L"\textrm{Analytic}", lw=3.0, alpha=1.)
    plot!(fig, tsp, sol_a_low, c=:magenta, lab="", lw=3.0, alpha=1.)

end

fig = figure_1b()
savefig(fig, joinpath(TARGET_FOLDER, "figure_1b.pdf"))
