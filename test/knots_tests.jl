using SimpleSplines
using Test

@testset "$(rpad("Mesh Tests",80))" begin

    L = 2π

    @testset "$(rpad("uniform",76))" begin
        m = UniformMesh(8, 1.0)
        @test ncells(m) == 8
        @test domainlength(m) == 1.0
        @test length(m) == 8
        @test eltype(m) == Float64
        @test breakpoints(m) ≈ collect(0:7) ./ 8
        @test cellbounds(m) ≈ collect(0:8) ./ 8
        @test meshwidth(m) ≈ 1/8
        @test first(breakpoints(m)) == 0
        @test UniformMesh(8) == UniformMesh(8, 2π)
    end

    @testset "$(rpad("graded",76))" begin
        m = GradedMesh(16, L)
        y = breakpoints(m)
        @test length(y) == 16
        @test issorted(y)
        @test first(y) == 0
        @test last(y) < L
        # the map does not depend on n, so refining gives a genuine mesh family: every
        # breakpoint of the coarse mesh is a breakpoint of the refined one
        y2 = breakpoints(GradedMesh(32, L))
        @test all(any(z -> isapprox(z, yi; atol = 1e-12), y2) for yi in y)
        # the cell widths really do vary, or the family would be the uniform one
        w = diff(cellbounds(m))
        @test maximum(w) / minimum(w) > 1.2
        @test_throws ArgumentError GradedMesh(8, L; amplitude = 1.5)
    end

    @testset "$(rpad("random",76))" begin
        m = RandomMesh(16, L)
        y = breakpoints(m)
        @test length(y) == 16
        @test issorted(y)
        @test first(y) == 0
        @test last(y) < L
        @test sum(diff(cellbounds(m))) ≈ L
        # seeded, hence reproducible
        @test breakpoints(RandomMesh(16, L; seed = 7)) == breakpoints(RandomMesh(16, L; seed = 7))
        @test breakpoints(RandomMesh(16, L; seed = 7)) != breakpoints(RandomMesh(16, L; seed = 8))
        @test_throws ArgumentError RandomMesh(8, L; spread = -1)
    end

    @testset "$(rpad("invalid arguments",76))" begin
        @test_throws ArgumentError UniformMesh(0, 1.0)
        @test_throws ArgumentError UniformMesh(4, 0.0)
        @test_throws ArgumentError UniformMesh(4, -1.0)
        @test_throws ArgumentError UniformMesh(4, Inf)
    end

    @testset "$(rpad("equality and hashing",76))" begin
        @test UniformMesh(8, L) == UniformMesh(8, L)
        @test hash(UniformMesh(8, L)) == hash(UniformMesh(8, L))
        @test isequal(UniformMesh(8, L), UniformMesh(8, L))
        @test UniformMesh(8, L) != UniformMesh(9, L)
        @test UniformMesh(8, L) ≈ UniformMesh(8, L)
    end

end
