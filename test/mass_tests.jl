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

        # a name that is not a mode is reported as one on a bounded basis too, rather than as
        # a mode that some other basis implements
        bb = BSplineBasis(UniformMesh(n, 2π), 3)
        Mb = Matrix(mass_matrix(SplineQuadrature(bb)))
        err = try
            mass_operator(Mb, bb; kernel = :ignore)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("must be :reject or :project", err.msg)
        @test !occursin("periodic basis only", err.msg)

        # the deflation removes the constants specifically, so it verifies that they are
        # what the kernel holds rather than assuming it
        @test_throws ArgumentError SimpleSplines.FactorizedMass(
            sparse(Diagonal([1.0, 0.0, 1.0, 1.0])); kernel = :project)
    end

    @testset "$(rpad("a kernel larger than the constants is refused by both paths",76))" begin
        # the projector onto the complement of span{𝟙, v} with v = (1,-1,1,-1): it has 𝟙 in
        # its kernel, so it passes the constant-kernel check, and v there as well, so the
        # deflation of the constants alone does not leave an invertible problem behind. Both
        # v and 𝟙 are constant along diagonals, so the same matrix exercises both
        # representations.
        n = 4
        v = [1.0, -1.0, 1.0, -1.0]
        P = Matrix(I, n, n) .- ones(n, n) ./ n .- (v * v') ./ n

        @test norm(P * ones(n), Inf) < 1e-14        # the constants are in the kernel
        @test norm(P * v, Inf) < 1e-14              # and so is a second vector

        @test_throws ArgumentError CirculantMass(P, n; kernel = :project)
        @test_throws ArgumentError SimpleSplines.FactorizedMass(
            sparse(P); kernel = :project)

        # both say so, rather than reporting the kernel assertion they did accept
        for e in (try
            CirculantMass(P, n; kernel = :project)
        catch e
            e
        end,
            try
            SimpleSplines.FactorizedMass(sparse(P); kernel = :project)
        catch e
            e
        end)
            @test occursin("singular beyond the constants", e.msg)
        end
    end

    @testset "$(rpad("the tolerances are relative, so scaling changes no verdict",76))" begin
        # an absolute tolerance floored at one is absolute for every matrix below that scale,
        # and it then accepts an invertible matrix as having the constants in its kernel
        n = 32
        for meshtype in (UniformMesh, GradedMesh), p in 1:4,
            s in (1e-12, 1e-8, 1.0, 1e8)
            b = PeriodicBSplineBasis(meshtype(n, 2π), p)
            q = SplineQuadrature(b)

            # an invertible assembly is refused at every scale ...
            @test_throws ArgumentError mass_operator(
                mass_matrix(q) .* s, b; kernel = :project)

            # ... and a singular one is accepted at every scale, and solves
            S = stiffness_matrix(q) .* s
            op = mass_operator(S, b; kernel = :project)
            rhs = randn(n)
            rhs .-= sum(rhs) / n
            # `x` scales like 1/s, so the residual of `S * x` does not depend on s at all
            x = op \ rhs
            @test norm(S * x .- rhs, Inf) ≤ 1e-10 * norm(rhs, Inf)
            @test abs(sum(x)) ≤ 1e-10 * norm(x, Inf) * n
        end

        # circulance is judged relatively too: a uniform mesh stays circulant under scaling,
        # and a graded one stays refused
        M = Matrix(mass_matrix(SplineQuadrature(
            PeriodicBSplineBasis(UniformMesh(16, 2π), 3))))
        G = Matrix(mass_matrix(SplineQuadrature(
            PeriodicBSplineBasis(GradedMesh(16, 2π), 3))))
        for s in (1e-12, 1.0, 1e12)
            @test CirculantMass(M .* s, 16) isa CirculantMass
            @test_throws ArgumentError CirculantMass(G .* s, 16)
        end
    end

    @testset "$(rpad("the default tolerances follow the element type",76))" begin
        # an absolute default fixes the answer to one element type: at 1e-10 no Float32
        # assembly is circulant at all, because Float32 rounding alone exceeds it
        for p in 1:4, n in (16, 32, 64)

            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2π), p))
            M32 = Float32.(mass_matrix(q))
            op = CirculantMass(M32, n)
            @test op isa CirculantMass{Float32}
            @test eltype(op) === Float32

            b32 = randn(Float32, n)
            @test norm(M32 * (op \ b32) .- b32, Inf) < 1.0f-3

            # and the deflation works there too
            S32 = Float32.(stiffness_matrix(q))
            opp = CirculantMass(S32, n; kernel = :project)
            rhs = randn(Float32, n)
            rhs .-= sum(rhs) / n
            x = opp \ rhs
            @test norm(S32 * x .- rhs, Inf) < 1.0f-3
            @test abs(sum(x)) < 1.0f-3 * n
        end
    end

    @testset "$(rpad("the deflated construct and solve infer and do not allocate",76))" begin
        n = 32
        bu = PeriodicBSplineBasis(UniformMesh(n, 2π), 3)
        bg = PeriodicBSplineBasis(GradedMesh(n, 2π), 3)
        Su = stiffness_matrix(SplineQuadrature(bu))
        Sg = stiffness_matrix(SplineQuadrature(bg))

        # the keyword is a literal at the call site, so it must constant-propagate into a
        # concrete type -- a two-way Union here propagates into every consumer downstream.
        # Written as a function rather than a closure: a closure over a local infers as Any
        # on the 1.10 compat floor.
        project(M, b) = mass_operator(M, b; kernel = :project)
        @test @inferred(project(Su, bu)) isa CirculantMass
        @test @inferred(project(Sg, bg)) isa FactorizedMass

        # the circulant deflation is a stored zero factor, not a branch, so the solve
        # allocates no more than the undeflated one does: nothing
        op = project(Su, bu)
        x = randn(n)
        y = similar(x)
        mass_solve!(y, op, x)
        @test (@allocated mass_solve!(y, op, x)) == 0
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

    @testset "$(rpad("a first column describes the matrix, and indexes like it",76))" begin
        # The baseline is the assembled matrix, not the indexing formula the type uses: a
        # comprehension written the way `getindex` is written would pass by construction.
        for p in 2:4, n in (12, 16, 31)

            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2π), p))
            M = Matrix(mass_matrix(q))
            C = SimpleSplines.Circulant(M[:, 1])
            @test size(C) == (n, n)
            @test eltype(C) == Float64
            @test maximum(abs, Matrix(C) - M) < 1e-12
            @test C[3, 1] == M[3, 1]
        end
    end

    @testset "$(rpad("the column and the matrix build the same operator",76))" begin
        for p in 2:4, n in (12, 16, 32)

            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2π), p))
            b = basis(q)
            M = mass_matrix(q)

            opM = mass_operator(M, b)
            opc = mass_operator(Vector(M[:, 1]), b)

            @test opc isa CirculantMass
            @test mass_matrix(opc) isa SimpleSplines.Circulant
            @test size(opc) == size(opM)
            @test maximum(abs, Matrix(opc) - Matrix(opM)) < 1e-12

            for _ in 1:3
                x = randn(n)
                @test opc \ x ≈ opM \ x atol = 1e-10
                @test M * (opc \ x) ≈ x atol = 1e-10
            end
        end
    end

    @testset "$(rpad("the column path is O(n) where the matrix path is O(n²)",76))" begin
        # The case from the issue: a periodic stiffness matrix shifted by the rank-one mean
        # projector is circulant like the matrix it came from, but forming it turns an O(n)
        # sparse assembly into an O(n²) dense one, and `CirculantMass` then keeps it.
        p, n = 5, 128
        b = PeriodicBSplineBasis(UniformMesh(n, 2π), p)
        S = stiffness_matrix(SplineQuadrature(b))
        shifted = Matrix(S) .+ inv(n)
        column = Vector(S[:, 1]) .+ inv(n)

        opM = mass_operator(shifted, b)
        opc = mass_operator(column, b)

        # Same operator, reached two ways.
        x = randn(n)
        @test opc \ x ≈ opM \ x atol = 1e-10
        @test shifted * (opc \ x) ≈ x atol = 1e-8

        # The saving is the point of the change, so it is measured rather than asserted. The
        # dense n×n matrix is 8n² bytes and the column is 8n; a factor of ten is far inside
        # that and leaves room for the plans and buffers both operators carry.
        @test Base.summarysize(opc) < Base.summarysize(opM) / 10

        # It is a saving in *storage*, and the operator keeps it: the whole point is that the
        # matrix is not retained for the life of the operator. `mass_matrix` proves which of
        # the two is held.
        @test Base.summarysize(mass_matrix(opc)) < Base.summarysize(mass_matrix(opM)) / 10

        # Construction is O(n) rather than O(n²) as well, because the circulance check has
        # nothing to verify on a `Circulant`. That is a saving in time, not in allocation —
        # the caller allocated the matrix either way, and the check is an allocation-free
        # loop over n² entries — so it is measured in `scripts/circulant_column_cost.jl`
        # rather than asserted here, where a timing would be flaky.
    end

    @testset "$(rpad("the column path did not weaken the circulance check",76))" begin
        # A `Circulant` is circulant by construction, so its check has nothing to verify.
        # That must not have disabled the check for everything else.
        n = 16
        G = Matrix(mass_matrix(SplineQuadrature(PeriodicBSplineBasis(GradedMesh(n, 2π), 3))))
        @test_throws ArgumentError CirculantMass(G, n)

        # and a banded-but-not-circulant matrix is still refused
        B = Matrix(mass_matrix(SplineQuadrature(BSplineBasis(UniformMesh(n, 2π), 3))))
        @test_throws ArgumentError CirculantMass(B, nbasis(BSplineBasis(UniformMesh(n, 2π), 3)))
    end

    @testset "$(rpad("what the column form refuses",76))" begin
        n = 16
        u = PeriodicBSplineBasis(UniformMesh(n, 2π), 3)
        c = Vector(mass_matrix(SplineQuadrature(u))[:, 1])

        # a mesh on which a first column does not describe the matrix at all
        for meshtype in (GradedMesh, RandomMesh)
            g = PeriodicBSplineBasis(meshtype(n, 2π), 3)
            @test_throws ArgumentError mass_operator(c, g)
        end

        # a column of the wrong length
        @test_throws DimensionMismatch CirculantMass(c, n - 1)
        @test_throws DimensionMismatch mass_operator(c[1:(n - 1)], u)

        # and no column at all
        @test_throws ArgumentError SimpleSplines.Circulant(Float64[])
    end

    @testset "$(rpad("the column form carries the deflation and the element type",76))" begin
        n = 32
        b = PeriodicBSplineBasis(UniformMesh(n, 2π), 3)
        S = stiffness_matrix(SplineQuadrature(b))

        # the constants are in the kernel of a periodic stiffness matrix, so `:reject` must
        # refuse the column exactly as it refuses the matrix
        @test_throws ArgumentError mass_operator(Vector(S[:, 1]), b)

        opc = mass_operator(Vector(S[:, 1]), b; kernel = :project)
        opM = mass_operator(Matrix(S), b; kernel = :project)
        x = randn(n)
        x .-= sum(x) / n
        @test opc \ x ≈ opM \ x atol = 1e-10
        @test abs(sum(opc \ x)) < 1e-10          # the solution has no constant component

        # Float32 all the way through, rather than promoting to Float64 somewhere inside
        q32 = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(n, 2.0f0π), 3))
        c32 = Vector(mass_matrix(q32)[:, 1])
        @test eltype(c32) == Float32
        op32 = mass_operator(c32, basis(q32))
        @test op32 isa CirculantMass{Float32}
        @test eltype(mass_matrix(op32)) == Float32
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
