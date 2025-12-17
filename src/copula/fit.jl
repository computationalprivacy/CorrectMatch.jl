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

### Optimization routines

function obj_using_samples(
    theta::Float64,
    nb_rows::Int,
    marginals::AbstractVector{<:DiscreteUnivariateDistribution},
    mi_opt::Float64,
)::Float64
    (theta < 0) && (return -Inf)
    (theta >= 1) && (return Inf)

    G = GaussianCopula(PDMat([1.0 theta; theta 1.0]), marginals)
    discrete_sample::Matrix{Int} = rand(G, nb_rows)
    mi_sample::Float64 = mutual_information(discrete_sample; normalize = false, adjusted = true)[2, 1]
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
)::GaussianCopula{Float64} where {T<:Integer}
    # Validate input data
    CM.validate_discrete_data(data, "data")

    try
        marginals::Vector{DiscreteUnivariateDistribution} =
            [extract_marginal(@view(data[:, i]); exact = exact_marginal) for i in 1:size(data, 2)]
        fit_mle(GaussianCopula, marginals, data; samples = samples)
    catch e
        isa(e, CM.CorrectMatchError) && rethrow(e)
        throw(CM.ModelFittingError("Failed to fit Gaussian copula model", e))
    end
end

function fit_mle(
    d::Type{GaussianCopula},
    Σ::PDMat{S},
    data::AbstractMatrix{T};
    exact_marginal::Bool = false,
)::GaussianCopula{S} where {S<:Real,T<:Integer}
    # Validate inputs
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
        marginals::Vector{DiscreteUnivariateDistribution} =
            [extract_marginal(@view(data[:, i]); exact = exact_marginal) for i in 1:size(data, 2)]
        return GaussianCopula(Σ, marginals)
    catch e
        isa(e, CM.CorrectMatchError) && rethrow(e)
        throw(CM.ModelFittingError("Failed to create Gaussian copula with provided correlation matrix", e))
    end
end

function fit_mle(
    d::Type{GaussianCopula},
    marginals::Vector{<:DiscreteUnivariateDistribution},
    data::AbstractMatrix{T};
    adaptative_threshold::Int = 100,
    samples::Int = 10000,
    mi_abs_tol::Float64 = 1e-5,
)::GaussianCopula{Float64} where {T<:Integer}
    # Validate inputs
    CM.validate_discrete_data(data, "data")

    N, M = size(data)

    # Check dimension compatibility
    if M != length(marginals)
        throw(CM.DataValidationError("Number of marginals ($(length(marginals))) must match number of variables ($M)"))
    end

    # Handle edge case of no variables
    if M == 0
        return GaussianCopula(PDMat(Matrix{Float64}(I, 0, 0)), marginals)
    end

    # Handle degenerate marginals (all variables constant)
    all_degenerate = all(m -> try
        ncategories(m) == 1
    catch
        ; false
    end, marginals)
    if all_degenerate
        return GaussianCopula(PDMat(Matrix{Float64}(I, M, M)), marginals)
    end

    try
        mi_matrix::Matrix{Float64} = mutual_information(data; normalize = false, adjusted = true)

        # Validate mutual information matrix
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
        if isa(e, CM.CorrectMatchError)
            rethrow(e)
        else
            throw(CM.ModelFittingError("Failed to fit copula parameters", e))
        end
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
)::GaussianCopula{Float64} where {T<:Integer}
    N, M = size(data)

    corr_up = zeros(Float64, (M, M))

    for i in 1:M, j in (i+1):M
        m_i, m_j = marginals[i], marginals[j]

        if isnan(mi_matrix[i, j])
            corr_up[i, j] = NaN
        elseif mi_matrix[i, j] < mi_abs_tol
            # Mutual information is effectively zero, variables are independent
            corr_up[i, j] = 0.0
        else
            try
                ϵ = eps(Float64)
                obj_fun(t) = obj_using_samples(t, samples, [m_i, m_j], mi_matrix[i, j])
                results = find_zero(obj_fun, (ϵ, 1.0 - ϵ), Bisection())  # always converge

                # Validate the correlation estimate
                if !isfinite(results) || results < -1 || results > 1
                    throw(CM.NumericalInstabilityError("Invalid correlation estimate: $results", "root finding"))
                end

                corr_up[i, j] = results
            catch e
                if isa(e, CM.NumericalInstabilityError)
                    rethrow(e)
                else
                    throw(
                        CM.NumericalInstabilityError(
                            "Failed to estimate correlation for variables $i and $j",
                            "optimization",
                        ),
                    )
                end
            end
        end
    end

    # Correct numerical errors by ensuring that the correlation matrix
    # is symmetric and that every eigenvalue is positive
    try
        B = corr_up + corr_up' + diagm(0 => ones(M))
        D, V = eigvals(B), eigvecs(B) # diagonalize B

        # Check for numerical issues in eigendecomposition
        if any(!isfinite, D) || any(!isfinite, V)
            throw(
                CM.NumericalInstabilityError("Eigendecomposition produced non-finite values", "matrix diagonalization"),
            )
        end

        corr_mat_fixed = V * diagm(0 => max.(D, 1e-10)) * V' # better than eps(Float64)
        # Numerical floating-point errors can make the matrix non-Hermitian
        corr_mat_fixed = (corr_mat_fixed + corr_mat_fixed') / 2

        # Final validation of the correlation matrix
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
        isa(e, CM.CorrectMatchError) && rethrow(e)
        throw(CM.NumericalInstabilityError("Failed to construct valid correlation matrix", "matrix correction"))
    end
end
