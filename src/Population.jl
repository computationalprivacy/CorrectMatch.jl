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

module Population

import ..CorrectMatch as CM

using StatsBase: countmap, values


# Hashing function to create unique identifiers per row in a matrix

function _inline_hash(x::AbstractVector{T}, h::UInt) where {T}
    @inbounds for i in eachindex(x)
        h = hash(x[i], h)
    end
    h
end

function _inline_hash(x::AbstractVector{T}) where {T}
    _inline_hash(x, zero(UInt))
end

# Compute frequencies from 1D or 2D data

function frequencies_from_data(data::AbstractVector{T}) where {T}
    CM.validate_discrete_data(data, "data")
    values(countmap(data))
end

function frequencies_from_data(data::AbstractMatrix{T}) where {T}
    CM.validate_discrete_data(data, "data")
    values(countmap([_inline_hash(@view(data[i, :])) for i in axes(data, 1)]))
end

# Compute uniqueness and correctness from frequencies

function uniqueness_from_freqs(freqs, total_size::Int)
    total_unique = count(==(1), freqs)
    total_unique / total_size
end

function correctness_from_freqs(freqs, total_size::Int)
    length(freqs) / total_size
end

"""Calculate the empirical uniqueness of a vector, or a matrix where each row represents a multivariate record."""
function uniqueness(data::Union{AbstractVector{T}, AbstractMatrix{T}}) where {T}
    N = isa(data, AbstractVector{T}) ? length(data) : size(data, 1)
    freqs = frequencies_from_data(data)
    uniqueness_from_freqs(freqs, N)
end


"""Calculate the empirical correctness of a vector, or a matrix where each row represents a multivariate record."""
function correctness(data::Union{AbstractVector{T}, AbstractMatrix{T}}) where {T}
    N = isa(data, AbstractVector{T}) ? length(data) : size(data, 1)
    freqs = frequencies_from_data(data)
    correctness_from_freqs(freqs, N)
end

end