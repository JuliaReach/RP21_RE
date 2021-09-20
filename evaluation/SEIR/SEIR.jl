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
# Results
# =================

## Solution with CARLIN

function _solve_seir_carlin(; N=4, T=30.0, δ=0.1, radius0=0, bloat=false, resets=nothing)

    x0c = [PS0, PE0, PI0] # initial condition
    @show x0c
    F1, F2 = seir_model_carlin()
    R, Re_λ1 = _error_bound_specabs_R(x0c, F1, F2; check=true)
    @show R
    @show Re_λ1

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

@time _solve_seir_carlin() # warm-up
@time _solve_seir_carlin() # run

@time _solve_seir_carlin_TM() # warm-up
@time _solve_seir_carlin_TM() # run
