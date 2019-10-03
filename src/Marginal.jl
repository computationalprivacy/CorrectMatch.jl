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

include("logarithmic.jl")
using StatsBase, Distributions, Optim
using Discreet
using Compat: argmin

export fit_histogram, extract_marginal, number_bins


### Utils

function number_bins(d::Distribution)
  isa(d, Categorical) && return ncategories(d)
  typemax(Int)
end

function llog(d::Distribution, counts::AbstractVector{Int})
  slide = minimum(d)
  sum(logpdf(d, i - 1 + slide) * v for (i, v) in enumerate(counts))
end

import StatsBase.bic
function bic(d::Distribution, counts::AbstractVector{Int})
  nb_params = isa(d, Categorical) ? ncategories(d) : length(params(d))
  - 2 * llog(d, counts) + log(sum(counts)) * nb_params
end


### Fitting routines

function fit_entropy(h::Float64, S; ϵ::Float64=1e-6)
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
function fit_histogram(counts::AbstractVector{Int}; ϵ = 1e-5)
    rvs = Distribution[]
    S = sum(counts)

    # Categorical
    rv = Categorical(counts ./ S)
    push!(rvs, rv)

    # Logarithmic
    rv = Logarithmic(fit_entropy(entropy(counts / S), S))
    push!(rvs, rv)

    # NegativeBinom
    f = x::Vector -> -llog(NegativeBinomial(x[1], x[2]), counts)
    df = OnceDifferentiable(f, [1., .5])
    results = optimize(df, [ϵ, ϵ], [Inf, 1. - ϵ], [1., .5], Fminbox(LBFGS()))
    push!(rvs, NegativeBinomial(results.minimizer...))

    # Geometric distribution
    results = optimize(x -> -llog(Geometric(x), counts), ϵ, 1 - ϵ)
    push!(rvs, Geometric(results.minimizer))

    # Return the best distribution according to BIC
    bics = [bic(rv, counts) for rv in rvs]
    rvs[argmin(bics)]
end

function extract_marginal(row::AbstractVector; exact::Bool=true)
  cm = collect(values(countmap(row; alg=:dict)))
  sort!(cm, rev=true)

  if exact
    return Categorical(cm / sum(cm))
  else
    return fit_histogram(cm)
  end
end

end
