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

struct GaussianCopula <: DiscreteMultivariateDistribution
  Σ::PDMat{Float64}
  marginals::Vector{DiscreteUnivariateDistribution}
end


### Parameters
params(d::GaussianCopula) = (d.Σ, d.marginals)


### Sampling
function gaussian_rvs(Σ::AbstractPDMat{Float64}, n::Int)
  X = rand(MvNormal(Σ), n)'
  StatsFuns.normcdf.(X)
end

function encode(d::Distribution, column::AbstractVector{Float64})
  [quantile(d, c) for c in column]
end

function apply_marginals(data::AbstractMatrix{Float64},
                         marginals::AbstractVector{<:Distribution})
  discrete_data = Array{Int}(undef, size(data))
  for i = 1:size(data, 2)
    discrete_data[:, i] = encode(marginals[i], @view(data[:, i]))
  end
  discrete_data
end

import Compat.Random: rand
function rand(d::GaussianCopula, n::Int)
    continuous_sample = gaussian_rvs(d.Σ, n)
    apply_marginals(continuous_sample, d.marginals)
end

rand(d::GaussianCopula) = rand(d, 1)
