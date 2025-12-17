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

"""Validate that data contains only discrete (integer) values suitable for analysis."""
function validate_discrete_data(data::AbstractArray, name::String = "data")
    isempty(data) && throw(DataValidationError("$name cannot be empty"))
    any(!isfinite, data) && throw(DataValidationError("$name contains non-finite values"))
    if isa(data, AbstractMatrix)
        n_rows, n_cols = size(data)
        n_rows == 0 && throw(InsufficientDataError("$name must have at least one observation (row)"))
        n_cols == 0 && throw(DataValidationError("$name must have at least one variable (column)"))
    end
    return isa(data, AbstractArray{<:AbstractFloat}) &&
        @warn "$name contains floating-point values. CorrectMatch works best with discrete (integer) data."
end

"""Validate that a matrix is a valid correlation matrix."""
function validate_correlation_matrix(Σ::AbstractMatrix{T}, name::String = "correlation matrix") where {T<:Real}
    n = size(Σ, 1)
    size(Σ, 2) != n && throw(DataValidationError("$name must be square, got size $(size(Σ))"))

    # diagonal ≈ 1
    diag_diff = abs.(diag(Σ) .- 1)
    if any(diag_diff .> 1e-10)
        i = findfirst(>(1e-10), diag_diff)
        throw(DataValidationError("$name diagonal element [$i,$i] = $(Σ[i,i]) should be 1.0"))
    end

    # symmetry
    if !issymmetric(Σ)
        max_asym = maximum(abs(Σ[i, j] - Σ[j, i]) for i in 1:n for j in 1:n if i != j)
        max_asym > 1e-10 && throw(DataValidationError("$name is not symmetric (max asymmetry: $max_asym)"))
    end

    # off-diagonal bounds
    if any(abs.(Σ) .> 1 .+ eps(Float64)) # allow tiny numerical slack
        ind = findfirst(x -> abs(x) > 1 + eps(Float64), Σ)
        i, j = ind
        throw(DataValidationError("$name element [$i,$j] = $(Σ[i,j]) is outside [-1, 1]"))
    end

    # Check positive definiteness
    try
        min_eig = minimum(eigvals(Hermitian(Σ)))
        min_eig ≤ 0 && throw(
            NumericalInstabilityError(
                "Matrix is not positive definite (minimum eigenvalue: $min_eig)",
                "eigenvalue decomposition",
            ),
        )
    catch e
        isa(e, NumericalInstabilityError) && rethrow(e)
        throw(NumericalInstabilityError("Failed to check positive definiteness", "eigenvalue decomposition"))
    end
end
