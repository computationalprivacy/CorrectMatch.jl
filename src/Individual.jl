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

using ..Copula
using StatsFuns, Distributions
using Compat: undef

export smooth_weight, individual_uniqueness

function smooth_weight(p::GaussianCopula, indiv::AbstractVector{Int}; iter::Int=100, kwargs...)
  M = length(indiv)
  ΔI =[pdf(p.marginals[j], indiv[j]) for j=1:M]

  cell_probs = Array{Float64}(undef, iter)
  for i=1:iter
    lower = rand(M) .* (1 .- ΔI)
    upper = lower .+ ΔI

    # Convert to gaussian marginals
    lower = norminvcdf.(lower)
    upper = norminvcdf.(upper)

    # Estimate the cell probability for this arrangement
    _, cell_probs[i], _ = Copula.call_mvndst(lower, upper, p.Σ.mat; kwargs...)
  end

  cell_probs
end

""" Estimate individual uniqueness for one given record. """
function individual_uniqueness(p::GaussianCopula, indiv::AbstractVector{Int}, n::Int; iter::Int=100)
  cells = smooth_weight(p, indiv)
  p_avg = mean(cells)
  (1 - p_avg) ^ (n - 1)
end

# function individual_uniqueness(p::GaussianCopula, n::Int=100; iter::Int=10)
#   scores = Array{Float64}(n)
#   for i=1:n
#     cells = smooth_weight(p, rrand(p))
#     scores[i] = (1)
#   end
#   # cells =   uniqueness_cell(c) = (1 - c / correction)^(N-1)
# end

end
