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

module Individual

import ..CorrectMatch as CM

using ..Copula
using StatsFuns, Distributions

export smooth_weight, individual_uniqueness

function smooth_weight(
    p::Copula.GaussianCopula{T},
    indiv::AbstractVector{Int};
    iter::Int = 100,
    kwargs...,
)::Vector{Float64} where {T<:Real}
    M = length(indiv)
    ΔI::Vector{Float64} = [pdf(p.marginals[j], indiv[j]) for j in 1:M]

    cell_probs = Vector{Float64}(undef, iter)
    for i in 1:iter
        lower::Vector{Float64} = rand(M) .* (1 .- ΔI)
        upper::Vector{Float64} = lower .+ ΔI

        # Convert to gaussian marginals
        lower = norminvcdf.(lower)
        upper = norminvcdf.(upper)

        # Estimate the cell probability for this arrangement
        _, cell_probs[i], _ = Copula.call_mvndst(lower, upper, p.Σ.mat; kwargs...)
    end

    return cell_probs
end

"""Estimate the probability that a specific record is unique within a population of given size."""
function individual_uniqueness(
    p::Copula.GaussianCopula{T},
    indiv::AbstractVector{Int},
    n::Int;
    iter::Int = 100,
)::Float64 where {T<:Real}
    cells::Vector{Float64} = smooth_weight(p, indiv; iter = iter)
    p_avg::Float64 = mean(cells)

    # Additional numerical checks
    if !isfinite(p_avg) || p_avg < 0 || p_avg > 1
        throw(
            CM.NumericalInstabilityError(
                "Average cell probability $p_avg is invalid (must be in [0,1])",
                "probability estimation",
            ),
        )
    end
    return (1 - p_avg) ^ (n - 1)
end

end
