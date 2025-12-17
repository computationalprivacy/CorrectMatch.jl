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

"""A discrete multivariate distribution modeled using a Gaussian copula."""
struct GaussianCopula{T<:Real} <: DiscreteMultivariateDistribution
    Σ::PDMat{T}
    marginals::Vector{DiscreteUnivariateDistribution}

    # Inner constructor to ensure type consistency
    function GaussianCopula{T}(Σ::PDMat{T}, marginals::AbstractVector{<:DiscreteUnivariateDistribution}) where {T<:Real}
        return new{T}(Σ, collect(marginals))
    end
end

# Convenience constructors
function GaussianCopula(Σ::PDMat{T}, marginals::AbstractVector{<:DiscreteUnivariateDistribution}) where {T<:Real}
    return GaussianCopula{T}(Σ, marginals)
end

### Parameters
params(d::GaussianCopula{T}) where {T} = (d.Σ, d.marginals)

### Sampling
function gaussian_rvs(Σ::AbstractPDMat{T}, n::Int)::Matrix{T} where {T<:Real}
    X = rand(MvNormal(Σ), n)'
    return StatsFuns.normcdf.(X)
end

function encode(d::DiscreteUnivariateDistribution, column::AbstractVector{T})::Vector{Int} where {T<:Real}
    return [quantile(d, c) for c in column]
end

function apply_marginals(
    data::AbstractMatrix{T},
    marginals::AbstractVector{<:DiscreteUnivariateDistribution},
)::Matrix{Int} where {T<:Real}
    discrete_data = Matrix{Int}(undef, size(data))
    for i in 1:size(data, 2)
        discrete_data[:, i] = encode(marginals[i], @view(data[:, i]))
    end
    return discrete_data
end

import Random: rand
function rand(d::GaussianCopula{T}, n::Int)::Matrix{Int} where {T<:Real}
    continuous_sample = gaussian_rvs(d.Σ, n)
    return apply_marginals(continuous_sample, d.marginals)
end

rand(d::GaussianCopula)::Matrix{Int} = rand(d, 1)
