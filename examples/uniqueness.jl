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

using CorrectMatch
using DataFrames
using StatsBase
using CSV
using CodecZlib

# Read gzipped CSV file
df = CSV.read(transcode(GzipDecompressor, read("adult.csv.gz")), DataFrame)
N = nrow(df)

# True population uniqueness
u = uniqueness(df)
println("True population uniqueness: $u")

# Fit model from DataFrame
G = fit_mle(GaussianCopula, df)
u = uniqueness(rand(G, N))
println("Estimated population uniqueness: $u")

# Individual uniqueness
indiv = df[1, :]
u_indiv = individual_uniqueness(G, indiv, N)
println("Individual uniqueness (record 1): $u_indiv")

# Fit model on 325 records (1% of the original data) and estimate uniqueness
ix = sample(1:N, 325; replace=false)
G_sample = fit_mle(GaussianCopula, df[ix, :])
u = uniqueness(rand(G_sample, N))
println("Estimated population uniqueness (1% sample): $u")
