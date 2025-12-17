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
using DataFrames

using LinearAlgebra
using Test

@testset "Population" begin
    M1 = [
        1 0;
        1 0;
        0 1
    ]
    @test uniqueness(M1) == 1/3
    @test correctness(M1) == (1 + 1/2 * 2) / 3

    M2 = [
        1 0;
        1 0;
        1 0
    ]
    @test uniqueness(M2) == 0
    @test correctness(M2) == 1/3

    M3 = [
        1 0;
        1 1;
        1 2
    ]
    @test uniqueness(M3) == 1
    @test correctness(M3) == 1

    M4 = [1; 2; 3]
    @test uniqueness(M4) == 1
    @test correctness(M4) == 1

    M5 = [1; 1; 2]
    @test uniqueness(M5) == 1/3
    @test correctness(M5) == (1/2 * 2 + 1) / 3
end

@testset "Marginals" begin
    d = fit_histogram([10, 10, 10, 10, 10])
    @test isa(d, Categorical)
    @test probs(d) == [0.2, 0.2, 0.2, 0.2, 0.2]

    d = fit_histogram([721, 180, 60, 23, 9, 4, 2, 1])
    @test isa(d, Marginal.Logarithmic)
end

@testset "Copula" begin
    data = [1 1 1; 1 1 1; 1 1 1; 1 1 1]
    G = fit(GaussianCopula, data)
    @test isa(G, GaussianCopula)
    @test G.Σ.mat ≈ diagm(0 => ones(3)) atol=1e-3  # doesn't matter, for regression
    @test ncategories(G.marginals[1]) == 1
    @test probs(G.marginals[1]) ≈ [1.0]
    @test Matrix(rand(G, 4)) == data

    data = [1 2 3; 3 4 5; 6 7 9; 9 10 11]
    G = fit_mle(GaussianCopula, data; exact_marginal = true)
    @test G.Σ.mat ≈ diagm(0 => ones(3)) atol=1e-3
    @test probs(G.marginals[1]) == probs(G.marginals[2]) == probs(G.marginals[3])
end

@testset "Individual uniqueness and correctness" begin
    data = [1 1 1; 1 1 1; 1 1 1; 1 1 1]
    G = fit(GaussianCopula, data)
    @test smooth_weight(G, [1, 1, 1]; iter = 1) ≈ [1.0]

    @test 1.0 == individual_uniqueness(G, [1, 1, 1], 1)
    @test 0.0 == individual_uniqueness(G, [1, 1, 1], 2)
    @test 1.0 == individual_correctness(G, [1, 1, 1], 1)
    @test 0.5 == individual_correctness(G, [1, 1, 1], 2)
    @test 1/3 == individual_correctness(G, [1, 1, 1], 3)
end

@testset "DataFrame integration" begin
    # Test with non-1-indexed values
    df = DataFrame(
        a = [5, 10, 5, 10, 5],
        b = [10, 20, 10, 30, 10]
    )

    # Population metrics work on DataFrame
    @test uniqueness(df) == 2/5
    @test correctness(df) == (2 + 1/3 * 3) / 5

    # fit_mle works on DataFrame
    G = fit_mle(GaussianCopula, df)
    @test isa(G, GaussianCopula)
    @test length(G.marginals) == 2
    @test length(G.categories) == 2

    # encode_record works
    @test encode_record(G, df[1, :]) == [1, 1]  # 5 -> 1, 10 -> 1
    @test encode_record(G, [5, 10]) == [1, 1]
    @test encode_record(G, [10, 30]) == [2, 3]

    N = nrow(df)
    @test 0 <= individual_uniqueness(G, df[1, :], N) <= 1
    @test 0 <= individual_correctness(G, df[1, :], N) <= 1

    # Can also use raw values directly
    @test 0 <= individual_uniqueness(G, [5, 10], N) <= 1
end

@testset "DataFrame categorical columns" begin
    # Test that string columns work too
    df = DataFrame(
        color = ["red", "blue", "red", "green", "blue"],
        size = ["S", "M", "L", "S", "M"]
    )

    G = fit_mle(GaussianCopula, df)
    @test isa(G, GaussianCopula)

    @test 0 <= individual_uniqueness(G, df[1, :], nrow(df)) <= 1
    @test 0 <= individual_uniqueness(G, ["red", "S"], nrow(df)) <= 1
end

@testset "DataFrame integer columns sampling" begin
    # Test that sampling from DataFrames with integer columns doesn't produce
    # out-of-bounds indices (regression test for quantile edge cases)

    # Create DataFrame with integer values 1-9
    df = DataFrame(
        col1 = repeat(1:9, 10),
        col2 = repeat(1:9, 10),
        col3 = repeat(1:9, 10)
    )

    G = fit_mle(GaussianCopula, df)
    @test isa(G, GaussianCopula)

    # Sampling should not throw any errors
    samples = rand(G, 1000)
    @test size(samples) == (1000, 3)

    # All sampled values should be within the original value range
    for col in eachcol(samples)
        @test all(v -> v in 1:9, col)
    end
end

@testset "encode function edge cases" begin
    # Test the encode function handles edge probability values correctly
    using CorrectMatch.Copula: encode

    # Create a simple categorical distribution
    d = Categorical([0.25, 0.25, 0.25, 0.25])
    K = ncategories(d)

    # Test with probability 0.0 (edge case that could return 0)
    result = encode(d, [0.0])
    @test result[1] >= 1
    @test result[1] <= K

    # Test with probability 1.0 (edge case that could return K+1)
    result = encode(d, [1.0])
    @test result[1] >= 1
    @test result[1] <= K

    # Test with very small probability
    result = encode(d, [1e-15])
    @test result[1] >= 1
    @test result[1] <= K

    # Test with probability very close to 1.0
    result = encode(d, [1.0 - 1e-15])
    @test result[1] >= 1
    @test result[1] <= K

    # Test normal range
    result = encode(d, [0.5])
    @test result[1] >= 1
    @test result[1] <= K
end

@testset "Large sample DataFrame integer columns" begin
    # Stress test: large samples increase likelihood of hitting edge cases
    df = DataFrame(
        a = rand(1:5, 100),
        b = rand(1:10, 100),
        c = rand(1:3, 100)
    )

    G = fit_mle(GaussianCopula, df)

    # Generate many samples to increase chance of edge probability values
    for _ in 1:10
        samples = rand(G, 10000)
        @test size(samples, 1) == 10000

        # Verify all values are within expected bounds
        @test all(v -> v in levels(G.categories[1]), samples[:, 1])
        @test all(v -> v in levels(G.categories[2]), samples[:, 2])
        @test all(v -> v in levels(G.categories[3]), samples[:, 3])
    end
end

@testset "returns_dataframe and rand output type" begin
    # Model fitted from Matrix should return Matrix
    data_matrix = [1 1; 2 2; 3 3; 1 2]
    G_matrix = fit_mle(GaussianCopula, data_matrix)
    @test returns_dataframe(G_matrix) == false
    samples_matrix = rand(G_matrix, 10)
    @test isa(samples_matrix, Matrix{Int})
    @test size(samples_matrix) == (10, 2)

    # Model fitted from DataFrame should return DataFrame
    df = DataFrame(a = [1, 2, 3, 1], b = [1, 2, 3, 2])
    G_df = fit_mle(GaussianCopula, df)
    @test returns_dataframe(G_df) == true
    samples_df = rand(G_df, 10)
    @test isa(samples_df, DataFrame)
    @test size(samples_df) == (10, 2)

    # Model fitted from DataFrame with strings should return DataFrame with original values
    df_strings = DataFrame(
        color = ["red", "blue", "green", "red"],
        size = ["S", "M", "L", "S"]
    )
    G_strings = fit_mle(GaussianCopula, df_strings)
    @test returns_dataframe(G_strings) == true
    samples_strings = rand(G_strings, 100)
    @test isa(samples_strings, DataFrame)
    # Verify that sampled values are from the original categories
    @test all(v -> v in ["red", "blue", "green"], samples_strings[:, 1])
    @test all(v -> v in ["S", "M", "L"], samples_strings[:, 2])
end
