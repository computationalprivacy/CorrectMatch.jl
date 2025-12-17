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

module Copula

import ..CorrectMatch as CM

using ..Marginal
using Distributions, PDMats, StatsBase, StatsFuns, Roots
using Discreet
using mvndst_jll
using LinearAlgebra

using Distributions: DiscreteMultivariateDistribution, DiscreteUnivariateDistribution

#=
================================================================================
MVNDST Fortran Interface
================================================================================
=#

"""
Call the Fortran 'mvndst' routine to integrate multivariate Gaussian distribution.
"""
function call_mvndst(lo::Vector{Float64}, hi::Vector{Float64}, corr_mat::AbstractMatrix{<:Real}; kwargs...)
    N = length(lo)

    # Compute infin flags for bound types
    infin = [
        if lo[i] == -Inf && hi[i] == Inf
            -1
        elseif lo[i] == -Inf
            0
        elseif hi[i] == Inf
            1
        else
            2
        end for i in 1:N
    ]

    flat_corr = [corr_mat[i, j] for i in 1:N for j in 1:(i-1)]
    return mvndst(lo, hi, infin, flat_corr; kwargs...)
end

function mvndst(
    lower::Vector{Float64},
    upper::Vector{Float64},
    infin::Vector{Int},
    correl::Vector{Float64};
    maxpts::Int = 2000,
    abseps::Float64 = 1e-6,
    releps::Float64 = 1e-6,
)
    n = length(lower)
    @assert n == length(upper)
    @assert (n * (n - 1) / 2) == length(correl)

    err, value, inform = Ref(1.0), Ref(1.0), Ref(1)
    ccall(
        (:mvndst_, libmvndst),
        Cvoid,
        (
            Ref{Int},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Int},
            Ptr{Float64},
            Ref{Int},
            Ref{Float64},
            Ref{Float64},
            Ref{Float64},
            Ref{Float64},
            Ref{Int},
        ),
        n,
        lower,
        upper,
        infin,
        correl,
        maxpts,
        abseps,
        releps,
        err,
        value,
        inform,
    )

    return err[], value[], inform[]
end

#=
================================================================================
GaussianCopula Distribution
================================================================================
=#

"""A discrete multivariate distribution modeled using a Gaussian copula."""
struct GaussianCopula{T<:Real} <: DiscreteMultivariateDistribution
    Σ::PDMat{T}
    marginals::Vector{DiscreteUnivariateDistribution}

    function GaussianCopula{T}(Σ::PDMat{T}, marginals::AbstractVector{<:DiscreteUnivariateDistribution}) where {T<:Real}
        return new{T}(Σ, collect(marginals))
    end
end

function GaussianCopula(Σ::PDMat{T}, marginals::AbstractVector{<:DiscreteUnivariateDistribution}) where {T<:Real}
    return GaussianCopula{T}(Σ, marginals)
end

params(d::GaussianCopula{T}) where {T} = (d.Σ, d.marginals)

### Sampling

function gaussian_rvs(Σ::AbstractPDMat{T}, n::Int) where {T<:Real}
    X = rand(MvNormal(Σ), n)'
    return StatsFuns.normcdf.(X)
end

function encode(d::DiscreteUnivariateDistribution, column::AbstractVector{T}) where {T<:Real}
    return [quantile(d, c) for c in column]
end

function apply_marginals(
    data::AbstractMatrix{T},
    marginals::AbstractVector{<:DiscreteUnivariateDistribution},
) where {T<:Real}
    discrete_data = Matrix{Int}(undef, size(data))
    @inbounds for (i, (col, marginal)) in enumerate(zip(eachcol(data), marginals))
        discrete_data[:, i] = encode(marginal, col)
    end
    return discrete_data
end

import Random: rand

function rand(d::GaussianCopula{T}, n::Int) where {T<:Real}
    continuous_sample = gaussian_rvs(d.Σ, n)
    return apply_marginals(continuous_sample, d.marginals)
end

rand(d::GaussianCopula) = rand(d, 1)

#=
================================================================================
Maximum Likelihood Estimation
================================================================================
=#

### Helper functions

"""Extract marginal distributions from each column of a data matrix."""
function _extract_marginals(data::AbstractMatrix{<:Integer}, exact::Bool)
    return [extract_marginal(@view(data[:, i]); exact = exact) for i in axes(data, 2)]
end

"""Objective function for correlation estimation via mutual information matching."""
function obj_using_samples(
    theta::Float64,
    nb_rows::Int,
    marginals::AbstractVector{<:DiscreteUnivariateDistribution},
    mi_opt::Float64,
)
    (theta < 0) && return -Inf
    (theta >= 1) && return Inf

    G = GaussianCopula(PDMat([1.0 theta; theta 1.0]), marginals)
    discrete_sample = rand(G, nb_rows)
    mi_sample = mutual_information(discrete_sample; normalize = false, adjusted = true)[2, 1]
    return mi_sample - mi_opt
end

### Fit data

import Distributions.fit_mle

"""Fit a Gaussian copula model to discrete multivariate data using maximum likelihood estimation."""
function fit_mle(
    d::Type{GaussianCopula},
    data::AbstractMatrix{T};
    samples::Int = 10000,
    exact_marginal::Bool = false,
    adaptative_threshold::Int = 100,    
    mi_abs_tol::Float64 = 1e-5,
) where {T<:Integer}
    CM.validate_discrete_data(data, "data")

    try
        marginals = _extract_marginals(data, exact_marginal)
        fit_mle(GaussianCopula, marginals, data; samples = samples, adaptative_threshold = adaptative_threshold, mi_abs_tol = mi_abs_tol)
    catch e
        CM.wrap_fitting_error(e, "Failed to fit Gaussian copula model")
    end
end

function fit_mle(
    d::Type{GaussianCopula},
    Σ::PDMat{S},
    data::AbstractMatrix{T};
    exact_marginal::Bool = false,
) where {S<:Real,T<:Integer}
    CM.validate_discrete_data(data, "data")
    CM.validate_correlation_matrix(Σ.mat, "correlation matrix")

    if size(Σ, 1) != size(data, 2)
        throw(
            CM.DataValidationError(
                "Correlation matrix dimension ($(size(Σ, 1))) must match number of variables ($(size(data, 2)))",
            ),
        )
    end

    try
        marginals = _extract_marginals(data, exact_marginal)
        return GaussianCopula(Σ, marginals)
    catch e
        CM.wrap_fitting_error(e, "Failed to create Gaussian copula with provided correlation matrix")
    end
end

function fit_mle(
    d::Type{GaussianCopula},
    marginals::Vector{<:DiscreteUnivariateDistribution},
    data::AbstractMatrix{T};
    samples::Int = 10000,
    adaptative_threshold::Int = 100,    
    mi_abs_tol::Float64 = 1e-5,
) where {T<:Integer}
    CM.validate_discrete_data(data, "data")

    N, M = size(data)

    if M != length(marginals)
        throw(CM.DataValidationError("Number of marginals ($(length(marginals))) must match number of variables ($M)"))
    end

    # Handle edge case of no variables
    if M == 0
        return GaussianCopula(PDMat(Matrix{Float64}(I, 0, 0)), marginals)
    end

    # Handle degenerate marginals (all variables constant)
    all_degenerate = all(marginals) do m
        try
            ncategories(m) == 1
        catch
            false
        end
    end
    if all_degenerate
        return GaussianCopula(PDMat(Matrix{Float64}(I, M, M)), marginals)
    end

    try
        mi_matrix = mutual_information(data; normalize = false, adjusted = true)

        if any(!isfinite, mi_matrix)
            throw(
                CM.NumericalInstabilityError(
                    "Mutual information matrix contains non-finite values",
                    "mutual information estimation",
                ),
            )
        end

        _fit_mle_mi_matrix(
            d,
            marginals,
            mi_matrix,
            data;
            adaptative_threshold = adaptative_threshold,
            samples = samples,
            mi_abs_tol = mi_abs_tol,
        )
    catch e
        CM.wrap_fitting_error(e, "Failed to fit copula parameters")
    end
end

function _fit_mle_mi_matrix(
    d::Type{GaussianCopula},
    marginals::Vector{<:DiscreteUnivariateDistribution},
    mi_matrix::AbstractMatrix{Float64},
    data::AbstractMatrix{T};
    adaptative_threshold::Int = 100,
    samples::Int = 10000,
    mi_abs_tol::Float64 = 1e-5,
) where {T<:Integer}
    N, M = size(data)
    corr_up = zeros(Float64, (M, M))

    for i in 1:M, j in (i+1):M
        m_i, m_j = marginals[i], marginals[j]

        if isnan(mi_matrix[i, j])
            corr_up[i, j] = NaN
        elseif mi_matrix[i, j] < mi_abs_tol
            corr_up[i, j] = 0.0
        else
            try
                ϵ = eps(Float64)
                obj_fun(t) = obj_using_samples(t, samples, [m_i, m_j], mi_matrix[i, j])
                # Set slightly wider bounds to avoid edge-case issues:
                results = find_zero(obj_fun, (-ϵ, 1.0 + ϵ), Bisection())

                if !isfinite(results) || results < -1 || results > 1
                    throw(CM.NumericalInstabilityError("Invalid correlation estimate: $results"))
                end

                corr_up[i, j] = results
            catch e
                if e isa CM.NumericalInstabilityError
                    rethrow(e)
                else
                    throw(CM.NumericalInstabilityError("Failed to estimate correlation for variables $i and $j", e))
                end
            end
        end
    end

    # Correct numerical errors: ensure symmetric positive-definite correlation matrix
    try
        B = corr_up + corr_up' + diagm(ones(M))
        F = eigen(B)
        D, V = F.values, F.vectors

        if any(!isfinite, D) || any(!isfinite, V)
            throw(
                CM.NumericalInstabilityError("Eigendecomposition produced non-finite values", "matrix diagonalization"),
            )
        end

        corr_mat_fixed = V * diagm(max.(D, 1e-10)) * V'
        corr_mat_fixed = (corr_mat_fixed + corr_mat_fixed') / 2

        if any(!isfinite, corr_mat_fixed)
            throw(
                CM.NumericalInstabilityError(
                    "Correlation matrix contains non-finite values after correction",
                    "matrix correction",
                ),
            )
        end

        return GaussianCopula(PDMat(corr_mat_fixed), marginals)
    catch e
        CM.wrap_fitting_error(e, "Failed to construct valid correlation matrix")
    end
end

end
