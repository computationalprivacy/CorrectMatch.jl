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

using Distributions, Roots

import Distributions: @check_args, @distr_support, partype, pdf, cdf, logpdf


import Compat.Statistics: mean, median, var, quantile
import Base: convert, minimum, maximum
import StatsBase: params, mode, fit
import Compat.Random: rand, GLOBAL_RNG, AbstractRNG


struct Logarithmic{T<:Real} <: DiscreteUnivariateDistribution
  p::T

  function Logarithmic{T}(p::T) where T
    @check_args(Logarithmic, zero(p) < p < one(p))
    new{T}(p)
  end

end

Logarithmic(p::T) where {T<:Real} = Logarithmic{T}(p)
Logarithmic() = Logarithmic(0.5)

@distr_support Logarithmic 1 Inf


### Conversions
convert(::Type{Logarithmic{T}}, p::Real) where {T<:Real} = Logarithmic(T(p))
convert(::Type{Logarithmic{T}}, d::Logarithmic{S}) where {T <: Real, S <: Real} = Logarithmic(T(d.p))


### Parameters
params(d::Logarithmic) = (d.p,)
@inline partype(d::Logarithmic{T}) where {T<:Real} = T


### Statistics
mean(d::Logarithmic) = -d.p / (log(1 - d.p) * (1 - d.p))
mode(d::Logarithmic{T}) where {T<:Real} = one(T)
var(d::Logarithmic) = -d.p * (d.p + log1p(-d.p)) / ((1 - d.p)^2 * log1p(-d.p)^2)


### Evaluations

function pdf(d::Logarithmic{T}, x::Int) where T<:Real
  x < 1 && zero(T)
  -d.p^x / x / log1p(-d.p)
end

function logpdf(d::Logarithmic{T}, x::Int) where T<:Real
  x < 1 && -T(Inf)
  log(-1 / log1p(-d.p)) + log(d.p) * x - log(x)
end


function cdf(d::Logarithmic{T}, x::Int) where T<:Real
  x < 1 && return zero(T)
  a = -1 / log1p(-d.p)
  b = sum(d.p^k / k for k=1:x)
  a * b
end


function quantile(d::Logarithmic{T}, p::Real; cap=10000) where T <: Real
  d.p == 0 && return one(T)
  d.p == 1 && return T(Inf)

  pk = - d.p / log1p(-d.p)
  k = 1

  while (p > pk) && (k < cap)
    p -= pk
    pk *= d.p * k / (k+1)
    k += 1
  end
  k
end


### Sampling

rand(d::Logarithmic) = rand(GLOBAL_RNG, d)

function rand(rng::AbstractRNG, d::Logarithmic)
  (d.p == 0) && (return 1)

  # Automatic selection between the LS and LK algorithms
  if d.p < 0.95
     s = -d.p / log1p(-d.p)
     x = 1
     u = rand(rng)

    while u > s
      u -= s
      x += 1
      s *= d.p * (x - 1) / x
    end

    return x
  end

  r = log1p(-d.p)
	v = rand(rng)
  (v >= d.p) && (return 1)

	u = rand(rng)
	q = -expm1(r * u)

  (v <= (q * q)) && (return (round(1 + log(v) / log(q))))
	(v <= q) && (return 1)
	2
end
