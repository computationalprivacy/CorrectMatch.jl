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

module Uniqueness

import ..CorrectMatch as CM

using StatsBase

function _inline_hash(x::AbstractVector{T}, h::UInt) where {T}
    @inbounds for i in eachindex(x)
        h = hash(x[i], h)
    end
    return h
end

function uniqueness_from_freqs(freqs, total_size::Int)
    total_unique = count(==(1), freqs)
    return total_unique / total_size
end

"""Calculate the empirical uniqueness of a vector of discrete values."""
function uniqueness(data::AbstractVector{T}) where {T}
    CM.validate_discrete_data(data, "data")
    N = length(data)
    freqs = values(countmap(data))
    return uniqueness_from_freqs(freqs, N)
end

"""Calculate the empirical uniqueness of a matrix where each row represents a multivariate record."""
function uniqueness(data::AbstractMatrix{T}) where {T}
    CM.validate_discrete_data(data, "data")
    N = size(data, 1)
    freqs = values(countmap([_inline_hash(@view(data[i, :]), zero(UInt)) for i in axes(data, 1)]))
    return uniqueness_from_freqs(freqs, N)
end

end
