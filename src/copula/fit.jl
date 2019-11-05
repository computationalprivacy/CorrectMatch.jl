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

function obj_using_samples(theta::Float64, nb_rows::Int, marginals::AbstractVector{<:Distribution}, mi_opt::Float64)
    (theta < 0) && (return -Inf)
    (theta >= 1) && (return Inf)

    G = GaussianCopula(PDMat([1. theta; theta 1.]), marginals)
    discrete_sample = rand(G, nb_rows)
    mi_sample = mutual_information(discrete_sample; normalize=false, adjusted=true)[2,1]
    mi_sample - mi_opt
end


### Fit data

import Distributions.fit_mle
function fit_mle(d::Type{GaussianCopula}, data::AbstractMatrix;
                 samples::Int=10000, exact_marginal=false)
    marginals = [extract_marginal(@view(data[:, i]); exact=exact_marginal) for i=1:size(data, 2)]
    fit_mle(GaussianCopula, marginals, data; samples=samples)
end

function fit_mle(d::Type{GaussianCopula}, Σ::PDMat, data::AbstractMatrix; exact_marginal=false)
    marginals = [extract_marginal(@view(data[:, i]); exact=exact_marginal) for i=1:size(data, 2)]
    GaussianCopula(Σ, marginals)
end

function fit_mle(d::Type{GaussianCopula},
                 marginals::Vector{<:UnivariateDistribution},
                 data::AbstractMatrix;
                 adaptative_threshold::Int=100,
                 samples::Int=10000,
                 mi_abs_tol::Float64=1e-5)
    N, M = size(data)
    if M == 0
        @assert M == length(marginals)
        return GaussianCopula(PDMat(Matrix{Float64}(0, 0)), marginals)
    end

    mi_matrix = mutual_information(data; normalize=false, adjusted=true)
    _fit_mle_mi_matrix(d, marginals, mi_matrix, data;
                       adaptative_threshold=adaptative_threshold,
                       samples=samples,
                       mi_abs_tol=mi_abs_tol)
end

function _fit_mle_mi_matrix(d::Type{GaussianCopula},
                 marginals::Vector{<:UnivariateDistribution},
                 mi_matrix::AbstractMatrix{Float64},
                 data::AbstractMatrix;
                 adaptative_threshold::Int=100,
                 samples::Int=10000,
                 mi_abs_tol::Float64=1e-5)
  N, M = size(data)

  corr_up = zeros(Float64, (M, M))

  for i = 1:M, j = i+1:M
    m_i, m_j = marginals[i], marginals[j]

    if isnan(mi_matrix[i,j])
      corr_up[i,j] = NaN
    else
      ϵ = eps(Float64)
      obj_fun(t) = obj_using_samples(t, samples, [m_i, m_j], mi_matrix[i, j])
      results = find_zero(obj_fun, (-ϵ, 1.), Bisection())  # always converge
      corr_up[i,j] = results
    end
  end

  # Correct numerical errors by ensuring that the corelation matrix
  # is diagonal and that every eigenvalue is positive
  B = corr_up + corr_up' + diagm(0 => ones(M))
  D, V = eigvals(B), eigvecs(B) # diagonalize B
  corr_mat_fixed = V * diagm(0 => max.(D, 1e-10)) * V' # better than eps(Float64)
  # Numerical floating-point errors can make the matrix non-Hermitian
  corr_mat_fixed = (corr_mat_fixed + corr_mat_fixed') / 2
  GaussianCopula(PDMat(corr_mat_fixed), marginals)
end
