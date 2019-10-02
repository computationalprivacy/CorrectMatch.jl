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

using StatsBase
export uniqueness


function _inline_hash(x::AbstractVector, h::UInt)
  for i=1:length(x)
    h = hash(x, h)
  end
  h
end

function uniqueness_from_freqs(freqs::Base.ValueIterator, total_size::Int)
    total_unique = 0
    for f in freqs
      if f == 1
        total_unique += 1
      end
    end
    total_unique / total_size
end

""" Return the number of unique elements in a vector. """
function uniqueness(data::AbstractVector)
  N = length(data)
  freqs = values(countmap(data))

  uniqueness_from_freqs(freqs, N)
end

""" Return the number of unique rows in a matrix. """
function uniqueness(data::AbstractMatrix)
  N = size(data, 1)
  freqs = values(countmap([_inline_hash(@view(data[i, :]), zero(UInt))
                 for i=1:N]))

  uniqueness_from_freqs(freqs, N)
end

end
