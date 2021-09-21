# =================
# Dependencies
# =================

using ReachabilityAnalysis
using CarlemanLinearization
using Plots
using LaTeXStrings
using Plots.PlotMeasures
using LinearAlgebra
using SparseArrays
using DifferentialEquations # to plot trajectories on top

using CarlemanLinearization: _error_bound_specabs_R

include("../utils.jl")

# =================
# Model definition
# =================

# model params
r_vac = 0.19
r_tra = 0.13
T_lat = 5.2
T_inf = 2.3

# Model params

Λ = 0.0
P = 1e7 # total

@show Ptot = P
@show PS0 = Ptot * 0.6
@show PI0 = Ptot * 0.03
@show PE0 = Ptot - PS0 - PI0

function seir_model_carlin()

    F1 = zeros(3, 3)
    F1[1, 1] = -Λ/P -r_vac
    F1[2, 2] = -Λ/P -1/T_lat
    F1[3, 2] = 1/T_lat
    F1[3, 3] = -Λ/P -1/T_inf

    F2 = zeros(3, 9) # [x, x⊗x]
    F2[1, 3] = -r_tra/P
    F2[2, 3] = r_tra/P;

    return F1, F2
end

# =================
# Solution method
# =================

## Solution with CARLIN

function _solve_seir_carlin(; N=4, T=30.0, δ=0.1, radius0=0, bloat=false, resets=nothing)

    x0c = [PS0, PE0, PI0] # initial condition

    F1, F2 = seir_model_carlin()
    R, Re_λ1 = _error_bound_specabs_R(x0c, F1, F2; check=true)

    n = 3
    dirs = _template(n=n, N=N)
    alg = LGG09(δ=δ, template=dirs)

    if radius0 == 0
        X0 = convert(Hyperrectangle, Singleton(x0c))
    else
        X0 = Hyperrectangle(x0c, radius0)
    end

    if isnothing(resets)
        @time sol = _solve_CARLIN(X0, F1, F2; alg=alg, N=N, T=T, bloat=bloat);
    else
        @time sol = _solve_CARLIN_resets(X0, F1, F2; resets=resets, alg=alg, N=N, T=T, bloat=bloat);
    end
    return sol
end

## Solution with TMJETS

@taylorize function seir!(dx, x, p, t)
    P_S, P_E, P_I = x

    aux1 = Λ / P

    dx[1] = -aux1 * P_S - r_vac * P_S + Λ - r_tra * P_S * P_I / P
    dx[2] = -aux1 * P_E - P_E / T_lat + r_tra * P_S * P_I / P
    dx[3] = -aux1 * P_I + P_E / T_lat - P_I / T_inf
end

function _solve_seir_carlin_TM(; T=30.0, radius0=0, trajectories=-1)
    x0c = [PS0, PE0, PI0] # initial condition
    if radius0 == 0
        X0 = convert(Hyperrectangle, Singleton(x0c))
    else
        X0 = Hyperrectangle(x0c, radius0)
    end
    prob = @ivp(x' = seir!(x), x(0) ∈ X0, dim=3)

    if trajectories == -1
        @time sol_tm = solve(prob, T=T, alg=TMJets())
    else
        sol_tm = solve(prob, T=T, alg=TMJets(), ensemble=true, trajectories=trajectories)
    end
    return sol_tm
end

# ===============
# Results
# ===============

# parameters
Tmax = 30.0
rr0  = fill(1.0 * 1e5, 3)

# taylor models solution
_solve_seir_carlin_TM(T=Tmax, radius0=rr0) # warm-up
time_TM = @elapsed _solve_seir_carlin_TM(T=Tmax, radius0=rr0)

# no error bounds, N = 2
_solve_seir_carlin(N=2, T=Tmax, δ=0.1, radius0=rr0, bloat=false) # warm-up
time_NoError_N2 = @elapsed _solve_seir_carlin(N=2, T=Tmax, δ=0.1, radius0=rr0, bloat=false)

# including error bounds, N = 5
_solve_seir_carlin(N=5, T=Tmax, δ=0.1, radius0=rr0, bloat=true, resets=[4.0]) # warm-up
time_Error_N5 = @elapsed _solve_seir_carlin(N=5, T=Tmax, δ=0.1, radius0=rr0, bloat=true, resets=[4.0])

# save times
print(io, "SEIR, Carleman, no error bound, N=2, $(time_NoError_N2)\n")
print(io, "SEIR, Carleman, including error bound, N=5, $(time_Error_N5)\n")
print(io, "SEIR, TMJets, -, -, $(time_TM)\n")

# ===============
# Figures
# ===============

# figure with NO error bounds
function figure_seir()

    Tmax = 30.0
    rr0  = fill(1.0 * 1e5, 3)
    solN2 = _solve_seir_carlin(N=2, T=Tmax, δ=0.1, radius0=rr0, bloat=false);

    # taylor models solution
    sol_tm = _solve_seir_carlin_TM(T=Tmax, radius0=rr0)

    # trajectories
    sol_tm_traj = _solve_seir_carlin_TM(T=Tmax, radius0=rr0, trajectories=5)

    fig = plot(legend=:topright, xlab=L"\textrm{Time (days)}", ylab=L"\textrm{Individuals } (\times 10^6)",
               legendfontsize=25,
               tickfont=font(25, "Times"),
               guidefontsize=25,
               xguidefont=font(35, "Times"),
               yguidefont=font(35, "Times"),
               xtick = ([0, 10, 20, 30], [L"0", L"10", L"20", L"30"]),
               ytick = ([0, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6],
                        [L"0", L"1", L"2", L"3", L"4", L"5", L"6"]),
               xlims=(0, 30),
               ylims=(0, 6e6),
               bottom_margin=10mm, left_margin=6mm, right_margin=6mm, top_margin=3mm, size=(900, 600))


    # carleman linearization solution
    plot!(fig, solN2, vars=(0, 1), lab=L"S~(N = 2)", alpha=1., c=:blue, lc=:blue)
    plot!(fig, sol_tm, vars=(0, 1), lab=L"S~(TM)", alpha=1., c=:aquamarine, lc=:aquamarine)

    plot!(fig, solN2, vars=(0, 2), lab=L"E~(N=2)", alpha=1., c=:red, lc=:red)
    plot!(fig, sol_tm, vars=(0, 2), lab=L"E~(TM)", alpha=1., c=:darksalmon, lc=:darksalmon)

    plot!(fig, solN2, vars=(0, 3), lab=L"I~(N=2)", alpha=1., c=:orange, lc=:orange)
    plot!(fig, sol_tm, vars=(0, 3), lab=L"I~(TM)", alpha=1., c=:darkseagreen, lc=:darkseagreen)


    # lens
    lens!(fig, [18, 22], [7.5e4, 1.75e5], inset = (1, bbox(0.32, 0.34, 0.28, 0.24)),
           tickfont=font(14, "Times"),
           subplot=2,
           xticks=xticklatex([18., 20., 22.], 2),
           yticks=([7.5e4, 1.25e5, 1.75e5], [L"0.075", L"0.125", L"0.175"]))

    return fig
end

fig = figure_seir()
savefig(fig, joinpath(TARGET_FOLDER, "figure_seir_noerror.pdf"))

# figure with error bounds
function figure_seir_errors()

    Tmax = 30.0
    rr0  = fill(1.0 * 1e5, 3)

    N = 5
    solN5 = _solve_seir_carlin(N=N, T=Tmax, δ=0.1, radius0=rr0, bloat=false);

    solN5bloat = _solve_seir_carlin(N=N, T=Tmax, δ=0.1, radius0=rr0, bloat=true, resets=[4.0]);

    fig = plot(legend=:topright, xlab=L"\textrm{Time (days)}", ylab=L"\textrm{Individuals } (\times 10^6)",
               legendfontsize=25,
               tickfont=font(25, "Times"),
               guidefontsize=25,
               xguidefont=font(35, "Times"),
               yguidefont=font(35, "Times"),
               xtick = ([0, 10, 20, 30], [L"0", L"10", L"20", L"30"]),
               ytick = ([0, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6],
                        [L"0", L"1", L"2", L"3", L"4", L"5", L"6"]),
               xlims=(0, 30),
               ylims=(0, 6e6),
               bottom_margin=10mm, left_margin=6mm, right_margin=6mm, top_margin=3mm, size=(900, 600))


    # carleman linearization solution

    plot!(fig, solN5bloat, vars=(0, 1), lab=L"err(S)", alpha=1., c=:blue, lc=:blue)

    plot!(fig, solN5bloat, vars=(0, 2), lab=L"err(E)", alpha=1., c=:red, lc=:red)

    plot!(fig, solN5bloat, vars=(0, 3), lab=L"err(I)", alpha=1., c=:orange, lc=:orange)

    plot!(fig, solN5, vars=(0, 1), lab=L"S~(N = %$N)", alpha=1., c=:aquamarine, lc=:aquamarine)
    plot!(fig, solN5, vars=(0, 2), lab=L"E~(N = %$N)", alpha=1., c=:darksalmon, lc=:darksalmon)
    plot!(fig, solN5, vars=(0, 3), lab=L"I~(N = %$N)", alpha=1., c=:darkseagreen, lc=:darkseagreen)

    ylims!(0, 6e6)

    a = [4.0, 0.0]; b = [4.0, 1.3e6]
    plot!(LineSegment(a, b), markershape=:none, seriestype=:shape, alpha=1., c=:black, ls=:dash, lw=2.0)

    # lens
    lens!(fig, [3.5, 4.5], [0.6e6, 1.3e6], inset = (1, bbox(0.32, 0.34, 0.28, 0.24)),
           tickfont=font(14, "Times"),
           subplot=2,
           xticks=xticklatex([3.5, 4.0, 4.5], 2),
           yticks=([0.7e6, 1.0e6, 1.3e6], [L"0.7", L"1.0", L"1.3"]))

    return fig
end

fig = figure_seir_errors()
savefig(fig, joinpath(TARGET_FOLDER, "figure_seir_error.pdf"))
