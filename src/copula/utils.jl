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

using Pkg
import CorrectMatch

"""
Call the Fortran 'mvndst' routine to integrate multivariate Gaussian
distribution.
"""
function call_mvndst(
    lo::Vector{Float64},
    hi::Vector{Float64},
    corr_mat::AbstractMatrix{<:Real};
    kwargs...,
)::Tuple{Float64,Float64,Int}
    N::Int = length(lo)

    infin = Vector{Int}()
    for i in 1:N
        lowinf::Bool = lo[i] == -Inf
        uppinf::Bool = hi[i] == Inf

        if lowinf && uppinf
            push!(infin, -1)
        elseif lowinf
            push!(infin, 0)
        elseif uppinf
            push!(infin, 1)
        else
            push!(infin, 2)
        end
    end

    flat_corr::Vector{Float64} = [corr_mat[i, j] for i in 1:N for j in 1:(i-1)]
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
)::Tuple{Float64,Float64,Int}
    n = length(lower)
    @assert n == length(upper)
    @assert (n*(n-1)/2) == length(correl)

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
