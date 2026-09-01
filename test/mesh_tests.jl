using SimpleSplines
using Test

@testset "$(rpad("Mesh Tests",80))" begin
    L = 2π

    @testset "$(rpad("uniform",76))" begin
        m = UniformMesh(8, 0 .. 1)
        @test ncells(m) == 8
        @test domain(m) == (0.0 .. 1.0)
        @test domainlength(m) == 1.0
        @test length(m) == 8
        @test eltype(m) == Float64
        # n+1 breakpoints for n cells, both endpoints included
        @test length(breakpoints(m)) == 9
        @test breakpoints(m) ≈ collect(0:8) ./ 8
        @test meshwidth(m) ≈ 1/8
        @test first(m) == 0
        @test last(m) == 1
        @test first(breakpoints(m)) == 0
        @test last(breakpoints(m)) == 1

        # a domain given three ways
        @test UniformMesh(8, 0 .. 1) == UniformMesh(8, (0, 1)) == UniformMesh(8, 1)
        @test UniformMesh(8) == UniformMesh(8, 2π)

        # an offset domain
        m2 = UniformMesh(4, -1 .. 3)
        @test breakpoints(m2) ≈ [-1.0, 0.0, 1.0, 2.0, 3.0]
        @test domainlength(m2) == 4
        @test meshwidth(m2) == 1

        # an integer domain is promoted rather than throwing from inside `breakpoints`
        @test eltype(UniformMesh(4, 0 .. 1)) == Float64
        @test eltype(UniformMesh(4, -1 .. 2)) == Float64
    end

    @testset "$(rpad("graded",76))" begin
        m = GradedMesh(16, 0 .. L)
        y = breakpoints(m)
        @test length(y) == 17
        @test issorted(y)
        @test first(y) == 0
        @test last(y) ≈ L                 # the grading map fixes both endpoints
        # the map does not depend on n, so refining gives a genuine mesh family: every
        # breakpoint of the coarse mesh is a breakpoint of the refined one
        y2 = breakpoints(GradedMesh(32, 0 .. L))
        @test all(any(z -> isapprox(z, yi; atol = 1e-12), y2) for yi in y)
        # the cell widths really do vary, or the family would be the uniform one
        w = diff(y)
        @test maximum(w) / minimum(w) > 1.2
        @test sum(w) ≈ L
        @test_throws ArgumentError GradedMesh(8, 0 .. L; amplitude = 1.5)
    end

    @testset "$(rpad("random",76))" begin
        m = RandomMesh(16, 0 .. L)
        y = breakpoints(m)
        @test length(y) == 17
        @test issorted(y)
        @test first(y) == 0
        # the right endpoint is set exactly, not accumulated over n additions -- a basis whose
        # knot vector ends a few eps short of b evaluates to zero at b
        @test last(y) == L
        @test sum(diff(y)) ≈ L
        # seeded, hence reproducible
        @test breakpoints(RandomMesh(16, 0 .. L; seed = 7)) ==
              breakpoints(RandomMesh(16, 0 .. L; seed = 7))
        @test breakpoints(RandomMesh(16, 0 .. L; seed = 7)) !=
              breakpoints(RandomMesh(16, 0 .. L; seed = 8))
        @test_throws ArgumentError RandomMesh(8, 0 .. L; spread = -1)
    end

    @testset "$(rpad("general",76))" begin
        # the shape the particle codes need: uniform inside, one oversized cell at each end
        y = [-20.0; collect(range(-10, 10; length = 5)); 20.0]
        m = GeneralMesh(y)
        @test ncells(m) == 6
        @test breakpoints(m) == y
        @test domain(m) == (-20.0 .. 20.0)
        @test domainlength(m) == 40
        @test meshwidth(m) == 10
        @test diff(breakpoints(m)) ≈ [10.0, 5.0, 5.0, 5.0, 5.0, 10.0]

        # integers are accepted and promoted
        @test eltype(GeneralMesh([0, 1, 3])) == Float64
        @test breakpoints(GeneralMesh([0, 1, 3])) == [0.0, 1.0, 3.0]

        @test_throws ArgumentError GeneralMesh([1.0])
        @test_throws ArgumentError GeneralMesh([0.0, 1.0, 1.0])      # repeated
        @test_throws ArgumentError GeneralMesh([0.0, 2.0, 1.0])      # not increasing
        @test_throws ArgumentError GeneralMesh([0.0, Inf])
    end

    @testset "$(rpad("invalid arguments",76))" begin
        @test_throws ArgumentError UniformMesh(0, 0 .. 1)
        @test_throws ArgumentError UniformMesh(4, 0 .. 0)
        @test_throws ArgumentError UniformMesh(4, 1 .. 0)
        @test_throws ArgumentError UniformMesh(4, 0 .. Inf)
    end

    @testset "$(rpad("equality and hashing",76))" begin
        @test UniformMesh(8, 0 .. L) == UniformMesh(8, 0 .. L)
        @test hash(UniformMesh(8, 0 .. L)) == hash(UniformMesh(8, 0 .. L))
        @test isequal(UniformMesh(8, 0 .. L), UniformMesh(8, 0 .. L))
        @test UniformMesh(8, 0 .. L) != UniformMesh(9, 0 .. L)
        @test UniformMesh(8, 0 .. L) != UniformMesh(8, 1 .. L)
        @test UniformMesh(8, 0 .. L) ≈ UniformMesh(8, 0 .. L)
        # a GeneralMesh with the same breakpoints compares equal to the uniform one, since
        # equality is about the subdivision and not about which family produced it
        @test GeneralMesh(breakpoints(UniformMesh(8, 0 .. 1))) == UniformMesh(8, 0 .. 1)
    end
end
