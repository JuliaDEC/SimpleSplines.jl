using SimpleSplines
using LinearAlgebra
using SparseArrays
using Test

@testset "$(rpad("Mass Operator Tests",80))" begin
    @testset "$(rpad("the mesh decides the representation",76))" begin
        # a uniform mesh makes the basis functions translates of one cardinal spline, so M
        # is circulant; a graded or random mesh does not
        for p in 1:4
            @test mass_operator(SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), p))) isa
                  CirculantMass
            @test mass_operator(SplineQuadrature(PeriodicBSplineBasis(GradedMesh(16, 2π), p))) isa
                  FactorizedMass
            @test mass_operator(SplineQuadrature(PeriodicBSplineBasis(RandomMesh(16, 2π), p))) isa
                  FactorizedMass
        end
    end

    @testset "$(rpad("the mass matrix really is circulant on a uniform mesh",76))" begin
        for p in 1:4, n in (12, 16, 24)

            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2π), p))
            M = Matrix(mass_matrix(q))
            c = M[:, 1]
            @test maximum(abs, [M[i, j] - c[mod1(i - j + 1, n)] for i in 1:n, j in 1:n]) <
                  1e-12
        end
    end

    @testset "$(rpad("and is NOT circulant on a graded mesh",76))" begin
        # a check that the test above is not vacuous, and that the dispatch matters
        n = 16
        q = SplineQuadrature(PeriodicBSplineBasis(GradedMesh(n, 2π), 3))
        M = Matrix(mass_matrix(q))
        c = M[:, 1]
        @test maximum(abs, [M[i, j] - c[mod1(i - j + 1, n)] for i in 1:n, j in 1:n]) > 1e-3
        @test_throws ArgumentError CirculantMass(M, n)
    end

    @testset "$(rpad("the FFT solve agrees with the factorised one",76))" begin
        for p in 1:4, n in (12, 15, 16, 32)

            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2π), p))
            M = mass_matrix(q)
            circ = mass_operator(q)
            fact = FactorizedMass(M)
            @test circ isa CirculantMass
            for _ in 1:3
                b = randn(n)
                @test circ \ b ≈ fact \ b atol = 1e-10
                @test M * (circ \ b) ≈ b atol = 1e-10
            end
            # and on several right-hand sides at once
            B = randn(n, 3)
            @test circ \ B ≈ fact \ B atol = 1e-10
        end
    end

    @testset "$(rpad("in-place solve, aliasing, and no allocation",76))" begin
        q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(32, 2π), 3))
        op = mass_operator(q)
        b = randn(32)
        y = similar(b)
        @test mass_solve!(y, op, b) ≈ op \ b
        # y and x may alias
        z = copy(b)
        @test mass_solve!(z, op, z) ≈ op \ b
        @test ldiv!(y, op, b) ≈ op \ b
        # the plans and the spectral buffer are preallocated, so a solve allocates nothing
        mass_solve!(y, op, b)
        @test (@allocated mass_solve!(y, op, b)) == 0
    end

    @testset "$(rpad("the representation follows the basis, not the mesh",76))" begin
        # a bounded basis has no seam, so its mass matrix is banded outright; a periodic one
        # on a non-uniform mesh wraps and is only banded modulo N, which is what keeps it on
        # the sparse Cholesky
        u = UniformMesh(32, 0 .. 1)
        g = GradedMesh(32, 0 .. 1)
        @test mass_operator(SplineQuadrature(BSplineBasis(u, 3))) isa BandedMass
        @test mass_operator(SplineQuadrature(BSplineBasis(g, 3))) isa BandedMass
        @test mass_operator(SplineQuadrature(BSplineBasis(u, 3, Dirichlet()))) isa
              BandedMass
        @test mass_operator(SplineQuadrature(BSplineBasis(g, 3, Neumann()))) isa BandedMass
        @test mass_operator(SplineQuadrature(PeriodicBSplineBasis(u, 3))) isa CirculantMass
        @test mass_operator(SplineQuadrature(PeriodicBSplineBasis(g, 3))) isa FactorizedMass
    end

    @testset "$(rpad("the banded solve matches the dense one and does not allocate",76))" begin
        for p in 1:4, bc in (Free(), Dirichlet(), Neumann(), Robin(1.0, 2.0)),
            m in (UniformMesh(24, 0 .. 1), GradedMesh(24, 0 .. 1), RandomMesh(24, 0 .. 1))
            q = SplineQuadrature(BSplineBasis(m, p, bc))
            op = mass_operator(q)
            @test op isa BandedMass
            N = nbasis(q)
            x = randn(N)
            @test op \ x ≈ Matrix(mass_matrix(q)) \ x

            y = similar(x)
            @test mass_solve!(y, op, x) ≈ op \ x
            z = copy(x)
            @test mass_solve!(z, op, z) ≈ op \ x        # y and x may alias
            # the banded factor solves in place, so unlike CHOLMOD this allocates nothing
            mass_solve!(y, op, x)
            @test (@allocated mass_solve!(y, op, x)) == 0
        end
    end

    @testset "$(rpad("accessors and errors",76))" begin
        q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), 3))
        op = mass_operator(q)
        @test size(op) == (16, 16)
        @test size(op, 1) == 16
        @test eltype(op) == Float64
        @test Matrix(op) ≈ Matrix(mass_matrix(q))
        @test mass_matrix(op) === mass_matrix(q)
        @test_throws DimensionMismatch CirculantMass(Matrix(mass_matrix(q)), 15)
    end

    @testset "$(rpad("the basis tabulation is sparse",76))" begin
        # (p+1)*nq structurally nonzero entries per row is the whole point: a dense
        # tabulation makes every assembly cost N times more than it should
        for p in 2:4
            n, nq = 32, quadrature_order(p)
            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2π), p))
            Φ = basis_values(q, 0)
            @test Φ isa SparseMatrixCSC
            @test nnz(Φ) == n * (p + 1) * nq
            @test nnz(Φ) < 0.5 * length(Φ)
        end
    end

    @testset "$(rpad("the constant assemblies are memoised, not reassembled",76))" begin
        q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), 3))
        A = mixed_matrix(q, 1, 1)
        B = mixed_matrix(q, 1, 1)
        @test A === B                       # the same object, not merely equal
        @test stiffness_matrix(q) === A
        @test derivative_matrix(q) === mixed_matrix(q, 0, 1)
    end
end
