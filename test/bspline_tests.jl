using SimpleSplines
using LinearAlgebra
using Random
using Test

const MESHES = ((:uniform, n -> UniformMesh(n, 2π)),
                (:graded,  n -> GradedMesh(n, 2π)),
                (:random,  n -> RandomMesh(n, 2π)))

@testset "$(rpad("Periodic B-Spline Basis Tests",80))" begin

    @testset "$(rpad("construction and accessors",76))" begin
        b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        @test nbasis(b) == 16                    # N = n, not n + p
        @test degree(b) == 3
        @test order(b) == 4                      # spline order k = p + 1
        @test nnodes(b) == nbasis(b)
        @test length(nodes(b)) == nbasis(b)
        @test eltype(b) == Float64
        @test eachindex(b) == 1:16
        @test length(knotvector(b)) == 5 * 16
        @test PeriodicBSplineBasis(16, 3) == PeriodicBSplineBasis(UniformMesh(16, 2π), 3)

        # a basis function spans p+1 cells, so it would wrap onto itself for n ≤ p
        @test_throws ArgumentError PeriodicBSplineBasis(UniformMesh(3, 2π), 3)
        @test_throws ArgumentError PeriodicBSplineBasis(UniformMesh(16, 2π), -1)
    end

    @testset "$(rpad("dimension is n, not n + p",76))" begin
        for p in 0:4, n in (8, 13, 16)
            @test nbasis(PeriodicBSplineBasis(UniformMesh(n, 2π), p)) == n
        end
    end

    @testset "$(rpad("partition of unity",76))" begin
        xs = collect(range(0, 2π, length = 97)[1:96])
        for p in 0:4, (nm, mk) in MESHES
            b = PeriodicBSplineBasis(mk(16), p)
            @test maximum(abs, [sum(b[x, j] for j in eachindex(b)) - 1 for x in xs]) < 1e-13
        end
    end

    @testset "$(rpad("non-negativity",76))" begin
        xs = collect(range(0, 2π, length = 97)[1:96])
        for p in 0:4, (nm, mk) in MESHES
            b = PeriodicBSplineBasis(mk(16), p)
            @test all(≥(-1e-14), b[xs, :])
        end
    end

    @testset "$(rpad("support is exactly p+1 cells",76))" begin
        for p in 0:4
            b = PeriodicBSplineBasis(UniformMesh(12, 2π), p)
            bnds = cellbounds(b)
            live = count(1:12) do k
                xs = range(bnds[k], bnds[k+1], length = 7)[2:6]
                maximum(abs, [evaluate(b, 1, x) for x in xs]) > 1e-13
            end
            @test live == p + 1
        end
    end

    @testset "$(rpad("periodicity",76))" begin
        # the basis is defined on the torus: evaluation is invariant under x -> x + L, and
        # accepts arguments outside [0,L) by reducing them
        for p in 1:4, (nm, mk) in MESHES
            b = PeriodicBSplineBasis(mk(16), p)
            L = domainlength(b)
            for x in range(0.1, L - 0.1, length = 11), j in (1, 5, 16), d in 0:2
                @test evaluate(b, j, x, d) ≈ evaluate(b, j, x + L, d) atol = 1e-13
                @test evaluate(b, j, x, d) ≈ evaluate(b, j, x - 2L, d) atol = 1e-13
            end
        end
    end

    @testset "$(rpad("translation identity phi_{j+1}(x) = phi_j(x - h)",76))" begin
        # on a uniform mesh the basis functions are translates of one cardinal spline
        for p in 1:4
            n = 16
            b = PeriodicBSplineBasis(UniformMesh(n, 2π), p)
            h = 2π / n
            for x in range(0, 2π, length = 23), j in 1:n-1
                @test evaluate(b, j + 1, x + h) ≈ evaluate(b, j, x) atol = 1e-13
            end
        end
    end

    @testset "$(rpad("smoothness is C^{p-1} across the seam and no more",76))" begin
        for p in 1:4
            b = PeriodicBSplineBasis(UniformMesh(12, 2π), p)
            L = domainlength(b)
            ε = 1e-9
            for d in 0:p-1
                @test abs(evaluate(b, 1, ε, d) - evaluate(b, 1, L - ε, d)) < 1e-6
            end
            # the p-th derivative jumps: the basis is not smoother than C^{p-1}
            @test abs(evaluate(b, 1, ε, p) - evaluate(b, 1, L - ε, p)) > 1e-3
        end
    end

    @testset "$(rpad("derivative recursion against finite differences",76))" begin
        for p in 2:4, (nm, mk) in MESHES
            b = PeriodicBSplineBasis(mk(16), p)
            h = 1e-5
            for x in range(0.37, 5.9, length = 13), j in (1, 4, 11), d in 0:p-2
                fd = (evaluate(b, j, x + h, d) - evaluate(b, j, x - h, d)) / 2h
                @test evaluate(b, j, x, d + 1) ≈ fd atol = 1e-6
            end
        end
    end

    @testset "$(rpad("lazy derivative agrees with evaluate",76))" begin
        b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        D = b'
        xs = collect(range(0.1, 6.0, length = 17))
        @test D[xs, 3] ≈ [evaluate(b, 3, x, 1) for x in xs]
        @test D[0.7, :] ≈ [evaluate(b, j, 0.7, 1) for j in eachindex(b)]
        @test D[xs, :] ≈ [evaluate(b, j, x, 1) for x in xs, j in eachindex(b)]
        @test D[0.7, 3] ≈ evaluate(b, 3, 0.7, 1)
    end

    @testset "$(rpad("indexing forms agree",76))" begin
        b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        xs = collect(range(0.1, 6.0, length = 17))
        @test b[0.7, 3] == evaluate(b, 3, 0.7)
        @test b[0.7, :] == [b[0.7, j] for j in eachindex(b)]
        @test b[xs, 3] == [b[x, 3] for x in xs]
        @test b[xs, :] == [b[x, j] for x in xs, j in eachindex(b)]
        @test b(0.7, 3) == b[0.7, 3]
    end

    @testset "$(rpad("expansion evaluation",76))" begin
        b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        û = randn(nbasis(b))
        for x in range(0.1, 6.0, length = 11)
            @test evaluate(b, û, x) ≈ sum(û[j] * b[x, j] for j in eachindex(b))
            @test evaluate(b, û, x, 1) ≈ sum(û[j] * evaluate(b, j, x, 1) for j in eachindex(b))
        end
        @test evaluate(b, û, [0.3, 1.1]) ≈ [evaluate(b, û, 0.3), evaluate(b, û, 1.1)]
        @test_throws DimensionMismatch evaluate(b, randn(3), 0.5)
    end

    @testset "$(rpad("bounds and argument checks",76))" begin
        b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        @test_throws BoundsError evaluate(b, 0, 0.5)
        @test_throws BoundsError evaluate(b, 17, 0.5)
        @test_throws ArgumentError evaluate(b, 1, 0.5, -1)
    end

    @testset "$(rpad("Greville nodes",76))" begin
        # on a uniform mesh the Greville abscissae are the breakpoints shifted by (p+1)/2
        # cells, and they lie in [0,L)
        for p in 1:4
            b = PeriodicBSplineBasis(UniformMesh(8, 8.0), p)
            ξ = nodes(b)
            @test all(0 .≤ ξ .< 8.0)
            @test length(unique(round.(ξ; digits = 10))) == nbasis(b)
        end
        # for p = 1 the Greville point of phi_j is where the hat function peaks, i.e. the
        # breakpoint one cell over, so the node set is the breakpoint set cyclically shifted
        @test sort(nodes(PeriodicBSplineBasis(UniformMesh(8, 8.0), 1))) ≈ breakpoints(UniformMesh(8, 8.0))
        @test nodes(PeriodicBSplineBasis(UniformMesh(8, 8.0), 1)) ≈ circshift(breakpoints(UniformMesh(8, 8.0)), -1)
        @test grid(PeriodicBSplineBasis(UniformMesh(8, 8.0), 3)) == nodes(PeriodicBSplineBasis(UniformMesh(8, 8.0), 3))
    end

    @testset "$(rpad("equality and hashing",76))" begin
        b1 = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        b2 = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        b3 = PeriodicBSplineBasis(UniformMesh(16, 2π), 2)
        @test b1 == b2
        @test hash(b1) == hash(b2)
        @test isequal(b1, b2)
        @test isapprox(b1, b2)
        @test b1 != b3
        @test !isapprox(b1, b3)
    end

end
