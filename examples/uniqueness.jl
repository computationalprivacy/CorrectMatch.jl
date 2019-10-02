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

using CorrectMatch: Copula, Uniqueness
using StatsBase
using CSV, GZip

df = CSV.read(GZip.open("adult.csv.gz"))
data = Array{Int}(df)
N = size(data, 1)

# True population uniqueness
u = uniqueness(data)
println("True population uniqueness: $u")

# Fit model and estimate uniqueness
G = fit_mle(GaussianCopula, data; exact_marginal=true)
u = uniqueness(rand(G, N))
println("Estimated population uniqueness: $u")

# Fit model on 325 records (1% of the original data) and estimate uniqueness
ix = sample(1:N, 325; replace=false);
G = fit_mle(GaussianCopula, data[ix, :]; exact_marginal=false)
u = uniqueness(rand(G, N))
println("Estimated population uniqueness (1% sample): $u")
