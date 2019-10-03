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
using CorrectMatch.Marginal, CorrectMatch.Individual
using Distributions, PDMats, StatsFuns

using Compat
using Compat.Test, Compat.LinearAlgebra


@testset "Uniqueness" begin
    M1 = [1 0;
          1 0;
          0 1]
    @test uniqueness(M1) == 1/3

    M2 = [1 0;
          1 0;
          1 0]
    @test uniqueness(M2) == 0

    M3 = [1 0;
          1 1;
          1 2]
    @test uniqueness(M3) == 1

    M4 = [1; 2; 3]
    @test uniqueness(M4) == 1

    M5 = [1; 1; 2]
    @test uniqueness(M5) == 1/3
end


@testset "Marginals" begin
    d = fit_histogram([10, 10, 10, 10, 10])
    @test isa(d, Categorical)
    @test probs(d) == [.2, .2, .2, .2, .2]

    d = fit_histogram([721, 180, 60, 23, 9, 4, 2, 1])
    @test isa(d, Marginal.Logarithmic)
end


@testset "Copula" begin
    data = [1 1 1; 1 1 1; 1 1 1; 1 1 1]
    G = fit(GaussianCopula, data)
    @test isa(G, GaussianCopula)
    @test G.Σ.mat ≈ diagm(0 => ones(3)) atol=1e-3  # doesn't matter, for regression
    @test ncategories(G.marginals[1]) == 1
    @test probs(G.marginals[1]) ≈ [1.]
    @test rand(G, 4) == data


    data = [1 2 3; 3 4 5; 6 7 9; 9 10 11]
    G = fit_mle(GaussianCopula, data; exact_marginal=true)
    @test G.Σ.mat ≈ diagm(0 => ones(3)) atol=1e-3
    @test probs(G.marginals[1]) == probs(G.marginals[2]) == probs(G.marginals[3])
end


@testset "Individual uniqueness" begin
    data = [1 1 1; 1 1 1; 1 1 1; 1 1 1]
    G = fit(GaussianCopula, data)
    @test smooth_weight(G, [1, 1, 1]; iter=1) ≈ [1.]

    @test 1. == individual_uniqueness(G, [1, 1, 1], 1)
    @test 0. == individual_uniqueness(G, [1, 1, 1], 2)
end
