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

    @testset "$(rpad("a singular assembly is deflated rather than shifted",76))" begin
        # the stiffness matrix of a periodic basis has the constants in its kernel, on every
        # mesh. The textbook cure is the rank-one shift by the mean projector, which is what
        # the deflation is checked against -- it must agree with it without ever forming it.
        for meshtype in (UniformMesh, GradedMesh), p in 1:4, n in (16, 33)
            b = PeriodicBSplineBasis(meshtype(n, 2π), p)
            S = stiffness_matrix(SplineQuadrature(b))
            @test norm(S * ones(n), Inf) < 1e-10

            op = mass_operator(S, b; kernel = :project)
            @test op isa (meshtype === UniformMesh ? CirculantMass : FactorizedMass)
            @test size(op) == (n, n)

            shifted = Matrix(S) .+ inv(n)
            for _ in 1:3
                rhs = randn(n)
                meanfree = rhs .- sum(rhs) / n
                x = op \ rhs

                @test x ≈ shifted \ meanfree atol = 1e-10
                @test S * x ≈ meanfree atol = 1e-10      # it solves the system
                @test sum(x) / n ≈ 0 atol = 1e-12        # in the complement of the kernel

                # the mean of the right-hand side is dropped by the deflation itself, so
                # taking it out first changes nothing
                @test op \ meanfree ≈ x atol = 1e-12

                y = copy(rhs)                            # y and x may alias
                @test mass_solve!(y, op, y) ≈ x atol = 1e-12
            end
        end
    end

    @testset "$(rpad("the deflation asserts a kernel, and both paths check it",76))" begin
        # :project says what the kernel is rather than asking what it happens to be, so a
        # mass matrix -- invertible, and with M𝟙 nowhere near zero -- is refused by it on
        # either representation. That is what keeps the two meanings the same one.
        for m in (UniformMesh(32, 2π), GradedMesh(32, 2π)), p in 1:4

            b = PeriodicBSplineBasis(m, p)
            M = mass_matrix(SplineQuadrature(b))
            @test_throws ArgumentError mass_operator(M, b; kernel = :project)
        end
    end

    @testset "$(rpad("what the deflation refuses",76))" begin
        n = 16
        b = PeriodicBSplineBasis(UniformMesh(n, 2π), 3)
        S = stiffness_matrix(SplineQuadrature(b))

        # the default rejects a singular matrix, and the tolerance is relative: the zero
        # eigenvalue of a stiffness matrix comes out of the transform at ~1e-14, not at zero,
        # so an absolute bound of eps() would call it invertible and divide by it
        @test_throws ArgumentError mass_operator(S, b)
        @test_throws ArgumentError mass_operator(S, b; kernel = :reject)

        # a bounded basis has no deflation to reach
        @test_throws ArgumentError mass_operator(
            Matrix(mass_matrix(SplineQuadrature(BSplineBasis(UniformMesh(n, 2π), 3)))),
            BSplineBasis(UniformMesh(n, 2π), 3); kernel = :project)

        # and neither mode accepts a name that is neither
        @test_throws ArgumentError mass_operator(S, b; kernel = :ignore)
        @test_throws ArgumentError mass_operator(
            stiffness_matrix(SplineQuadrature(PeriodicBSplineBasis(GradedMesh(n, 2π), 3))),
            PeriodicBSplineBasis(GradedMesh(n, 2π), 3); kernel = :ignore)

        # the deflation removes the constants specifically, so it verifies that they are
        # what the kernel holds rather than assuming it
        @test_throws ArgumentError SimpleSplines.FactorizedMass(
            sparse(Diagonal([1.0, 0.0, 1.0, 1.0])); kernel = :project)
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
