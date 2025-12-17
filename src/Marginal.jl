# CorrectMatch.jl
# Copyright © 2019 Université catholique de Louvain, UCLouvain
# Copyright © 2019 Imperial College London
# by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

module Marginal

import ..CorrectMatch as CM

export fit_histogram, extract_marginal, fit_entropy

include("logarithmic.jl")
using StatsBase, Distributions, Optim
using Discreet
using Roots: fzero

import Distributions: DiscreteUnivariateDistribution

### Utils

function number_bins(d::DiscreteUnivariateDistribution)::Int
    isa(d, Categorical) && return ncategories(d)
    return typemax(Int)
end

function llog(d::DiscreteUnivariateDistribution, counts::AbstractVector{Int})::Float64
    slide = minimum(d)
    return sum(logpdf(d, i - 1 + slide) * v for (i, v) in enumerate(counts))
end

import StatsBase.bic
function bic(d::DiscreteUnivariateDistribution, counts::AbstractVector{Int})::Float64
    nb_params = isa(d, Categorical) ? ncategories(d) : length(params(d))
    return - 2 * llog(d, counts) + log(sum(counts)) * nb_params
end

### Fitting routines

function fit_entropy(h::Float64, S::Int; ϵ::Float64 = 1e-6)::Float64
    ee(p) = estimate_entropy(rand(Logarithmic(p), S))

    b_min = ee(ϵ)
    b_max = ee(1-ϵ)
    h >= b_max && return 1-ϵ
    h <= b_min && return ϵ

    try
        fzero(p -> ee(p) - h, ϵ, 1-ϵ)
    catch ArgumentError
        abs(h - b_min) < abs(h - b_max) ? ϵ : 1-ϵ
    end
end

""" Fit a count distribution from a sorted histogram sample"""
function fit_histogram(counts::AbstractVector{Int}; ϵ::Float64 = 1e-5)::DiscreteUnivariateDistribution
    # Validate input
    isempty(counts) && throw(CM.DataValidationError("counts cannot be empty"))
    any(<(0), counts) && throw(CM.DataValidationError("counts must be non-negative"))

    S = sum(counts)
    rvs = Vector{DiscreteUnivariateDistribution}()

    try
        # Categorical
        rv = Categorical(counts ./ S)
        push!(rvs, rv)

        # Logarithmic
        rv = Logarithmic(fit_entropy(entropy(counts / S), S))
        push!(rvs, rv)

        # NegativeBinom
        df = OnceDifferentiable(x::Vector -> -llog(NegativeBinomial(x[1], x[2]), counts), [1.0, 0.5])
        results = optimize(df, [ϵ, ϵ], [Inf, 1.0 - ϵ], [1.0, 0.5], Fminbox(LBFGS()))
        if Optim.converged(results)
            push!(rvs, NegativeBinomial(results.minimizer...))
        end

        # Geometric distribution
        results = optimize(x -> -llog(Geometric(x), counts), ϵ, 1 - ϵ)
        if Optim.converged(results)
            push!(rvs, Geometric(results.minimizer))
        end

        # Return the best distribution according to BIC
        bics = [bic(rv, counts) for rv in rvs]

        best_idx = argmin(bics)
        return rvs[best_idx]
    catch e
        throw(CM.ModelFittingError("Failed to fit marginal distribution", e))
    end
end

function extract_marginal(row::AbstractVector{T}; exact::Bool = true)::DiscreteUnivariateDistribution where {T}
    # Validate input
    isempty(row) && throw(CM.DataValidationError("row cannot be empty"))
    any(!isfinite, row) && throw(CM.DataValidationError("row contains non-finite values"))

    cm = collect(values(countmap(row; alg = :dict)))
    sort!(cm; rev = true)

    if exact
        return Categorical(cm / sum(cm))
    else
        return fit_histogram(cm)
    end
end

end
