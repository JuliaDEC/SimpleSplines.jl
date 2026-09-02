using SimpleSplines
using LinearAlgebra
using Random
using Test

# A vector whose axes do not start at 1, so that the sweep over a vector of points is tested
# against one. Written out here rather than taken from OffsetArrays, which is not a dependency
# of this package and would be one for four lines.
struct ShiftedVector{T} <: AbstractVector{T}
    data::Vector{T}
    offset::Int
end

Base.size(v::ShiftedVector) = size(v.data)
Base.axes(v::ShiftedVector) = (v.offset .+ axes(v.data, 1),)
Base.getindex(v::ShiftedVector, i::Int) = v.data[i - v.offset]
Base.IndexStyle(::Type{<:ShiftedVector}) = IndexLinear()

@testset "$(rpad("Clamped and Recombined Basis Tests",80))" begin
    @testset "$(rpad("clamped: dimension, partition of unity, endpoints",76))" begin
        for p in 0:5, n in max(1, p):8

            b = BSplineBasis(UniformMesh(n, -1 .. 2), p)
            @test nbasis(b) == n + p
            @test degree(b) == p
            @test order(b) == p + 1
            @test ncells(b) == n
            @test domain(b) == (-1.0 .. 2.0)
            @test length(knotvector(b)) == n + 2p + 1

            # a partition of unity on the whole closed interval, the right endpoint included
            for x in range(-1, 2; length = 17)
                @test sum(evaluate(b, j, x) for j in eachindex(b)) ≈ 1
            end

            # interpolatory at both ends: this is what the p+1-fold end knot buys, and what
            # `_bspline`'s `xmax` argument exists to make true at b as well as at a
            @test evaluate(b, 1, -1.0) ≈ 1
            @test evaluate(b, nbasis(b), 2.0) ≈ 1
            @test all(abs(evaluate(b, j, -1.0)) < 1e-14 for j in 2:nbasis(b))
            @test all(abs(evaluate(b, j, 2.0)) < 1e-14 for j in 1:(nbasis(b) - 1))

            # outside the domain a compactly supported basis is zero
            @test all(evaluate(b, j, -1.5) == 0 for j in eachindex(b))
            @test all(evaluate(b, j, 2.5) == 0 for j in eachindex(b))

            @test polynomial_reproduction(b) == p
            @test local_width(b) == p + 1
        end
    end

    @testset "$(rpad("clamped: Greville nodes",76))" begin
        for p in 1:4
            b = BSplineBasis(UniformMesh(10, 0 .. 1), p)
            ξ = nodes(b)
            @test length(ξ) == nbasis(b) == nnodes(b)
            @test issorted(ξ)
            @test ξ[begin] ≈ 0
            @test ξ[end] ≈ 1
            # each abscissa lies in the support of its own function
            @test all(evaluate(b, j, ξ[j]) > 0 for j in eachindex(b))
        end
    end

    @testset "$(rpad("clamped: derivatives against finite differences",76))" begin
        b = BSplineBasis(UniformMesh(12, 0 .. 1), 4)
        h = 1e-5
        for j in eachindex(b), x in (0.13, 0.37, 0.61, 0.88)

            fd = (evaluate(b, j, x + h) - evaluate(b, j, x - h)) / 2h
            @test isapprox(evaluate(b, j, x, 1), fd; atol = 1e-6)
            fd2 = (evaluate(b, j, x + h) - 2evaluate(b, j, x) + evaluate(b, j, x - h)) / h^2
            @test isapprox(evaluate(b, j, x, 2), fd2; atol = 1e-3)
        end
        # every derivative above the degree vanishes identically
        @test all(evaluate(b, j, 0.4, 5) == 0 for j in eachindex(b))
    end

    @testset "$(rpad("recombined: the condition holds for any coefficients",76))" begin
        m = UniformMesh(10, -2 .. 3)
        for p in 1:5
            bd = BSplineBasis(m, p, Dirichlet())
            @test nbasis(bd) == 10 + p - 2
            û = randn(nbasis(bd))
            @test abs(evaluate(bd, û, -2.0)) < 1e-12
            @test abs(evaluate(bd, û, 3.0)) < 1e-12

            bn = BSplineBasis(m, p, Neumann())
            @test nbasis(bn) == 10 + p - 2
            v̂ = randn(nbasis(bn))
            @test abs(evaluate(bn, v̂, -2.0, 1)) < 1e-10
            @test abs(evaluate(bn, v̂, 3.0, 1)) < 1e-10

            # one condition per end, whatever its order: a mixed pair still costs two
            bm = BSplineBasis(m, p, (Dirichlet(), Neumann()))
            @test nbasis(bm) == 10 + p - 2
            ŵ = randn(nbasis(bm))
            @test abs(evaluate(bm, ŵ, -2.0)) < 1e-12
            @test abs(evaluate(bm, ŵ, 3.0, 1)) < 1e-10

            # one end only
            b1 = BSplineBasis(m, p, (Dirichlet(), Free()))
            @test nbasis(b1) == 10 + p - 1
            x̂ = randn(nbasis(b1))
            @test abs(evaluate(b1, x̂, -2.0)) < 1e-12
            @test abs(evaluate(b1, x̂, 3.0)) > 1e-8          # unconstrained at the right

            if p ≥ 2
                bnat = BSplineBasis(m, p, Natural())
                @test nbasis(bnat) == 10 + p - 2
                n̂ = randn(nbasis(bnat))
                @test abs(evaluate(bnat, n̂, -2.0, 2)) < 1e-8
                @test abs(evaluate(bnat, n̂, 3.0, 2)) < 1e-8
            end

            α, β = 1.5, -0.7
            br = BSplineBasis(m, p, Robin(α, β))
            r̂ = randn(nbasis(br))
            @test abs(α * evaluate(br, r̂, -2.0) + β * evaluate(br, r̂, -2.0, 1)) < 1e-10
            @test abs(α * evaluate(br, r̂, 3.0) + β * evaluate(br, r̂, 3.0, 1)) < 1e-10

            # a Constraint with no standard name
            bc = BSplineBasis(m, p, Constraint(1, -2))
            ĉ = randn(nbasis(bc))
            @test abs(evaluate(bc, ĉ, -2.0) - 2evaluate(bc, ĉ, -2.0, 1)) < 1e-10
        end
    end

    @testset "$(rpad("recombined: Dirichlet is the textbook elimination",76))" begin
        # for a Dirichlet condition the recombination reduces to dropping the end function,
        # so R is the interior identity block and nothing more
        b = BSplineBasis(UniformMesh(6, 0 .. 1), 3, Dirichlet())
        R = recombination_matrix(b)
        @test size(R) == (9, 7)
        @test Matrix(R) == Matrix(1.0I, 9, 9)[:, 2:8]
    end

    @testset "$(rpad("recombined: the matrix conjugates the parent assembly",76))" begin
        b = BSplineBasis(UniformMesh(8, 0 .. 1), 3, Neumann())
        R = recombination_matrix(b)
        qp = SplineQuadrature(parent(b))
        q = SplineQuadrature(b)
        # M̃ = R' M R -- this is why the quadrature needs no recombined special case
        @test maximum(abs, Matrix(mass_matrix(q)) -
                           Matrix(R' * mass_matrix(qp) * R)) < 1e-12
        @test maximum(abs, Matrix(basis_values(q, 0)) -
                           Matrix(R' * basis_values(qp, 0))) < 1e-12
    end

    @testset "$(rpad("local evaluation agrees with the reference recursion",76))" begin
        # `evaluate_all` runs de Boor's triangular scheme plus derivative lifting;
        # `evaluate` runs the Cox-de Boor recursion written out as it stands. They must
        # agree at every degree, every derivative order, every mesh family and every
        # boundary condition, and nothing outside the reported block may be nonzero.
        meshes = [UniformMesh(9, -1 .. 2), GradedMesh(9, -1 .. 2), RandomMesh(9, -1 .. 2),
            GeneralMesh([-3.0, -1.0, -0.5, 0.0, 0.7, 1.0, 2.0, 5.0])]
        for m in meshes, p in 0:4

            for bc in (Free(), Dirichlet(), Neumann(), Natural(), Robin(1.0, 2.0),
                Constraint(1, -2))
                p < constraint_order(bc) && continue
                b = BSplineBasis(m, p, bc)
                w = local_width(b)
                buf = zeros(w)
                for d in 0:p
                    for x in range(first(m) + 1e-9, last(m) - 1e-9; length = 13)
                        j₀ = evaluate_all!(buf, b, x, d)
                        block = Int[]
                        for t in 1:w
                            j = basis_index(b, j₀ + t - 1)
                            ref = (1 ≤ j ≤ nbasis(b)) ? evaluate(b, j, x, d) : 0.0
                            @test isapprox(buf[t], ref; atol = 1e-9 * max(1, abs(ref)))
                            1 ≤ j ≤ nbasis(b) && push!(block, j)
                        end
                        # everything not in the block really is zero
                        for j in eachindex(b)
                            j in block && continue
                            @test abs(evaluate(b, j, x, d)) < 1e-12
                        end
                    end
                end
            end

            # periodic, where the block wraps
            p < ncells(m) || continue
            bp = BSplineBasis(m, p, Periodic())
            buf = zeros(local_width(bp))
            for d in 0:p, x in range(first(m), last(m); length = 13)

                j₀ = evaluate_all!(buf, bp, x, d)
                for t in eachindex(buf)
                    j = basis_index(bp, j₀ + t - 1)
                    @test 1 ≤ j ≤ nbasis(bp)
                    @test isapprox(buf[t], evaluate(bp, j, x, d); atol = 1e-9)
                end
            end
        end
    end

    @testset "$(rpad("the local block is zero outside the domain",76))" begin
        # `findcell` clamps to the nearest cell, so de Boor's scheme would evaluate that
        # cell's polynomial at a point outside it -- an extrapolation, and O(1) wrong rather
        # than slightly wrong: on a degree-3 basis on 0 .. 1 the block at x = -0.5 came out
        # as [125.0, -196.0, 82.67, -10.67] against the correct zero. This is the property
        # every caller of `evaluate_all!` depends on, `evaluate(b, û, x)` and the
        # tensor-product paths among them, so it is checked here at the root.
        meshes = [UniformMesh(9, -1 .. 2), GradedMesh(9, -1 .. 2), RandomMesh(9, -1 .. 2),
            GeneralMesh([-3.0, -1.0, -0.5, 0.0, 0.7, 1.0, 2.0, 5.0])]
        for m in meshes, p in 0:4

            for bc in (Free(), Dirichlet(), Neumann(), Natural(), Robin(1.0, 2.0),
                Constraint(1, -2))
                p < constraint_order(bc) && continue
                b = BSplineBasis(m, p, bc)
                a, z = first(m), last(m)
                buf = zeros(local_width(b))
                for d in 0:p,
                    x in (a - 1e-9, a - 0.5, a - 100.0, z + 1e-9, z + 0.5,
                        z + 100.0)

                    j₀ = evaluate_all!(buf, b, x, d)
                    @test all(iszero, buf)
                    # the reference recursion agrees, which is what makes zero the right
                    # answer rather than merely the convenient one
                    @test all(evaluate(b, j, x, d) == 0 for j in eachindex(b))
                    # and the documented deposition recipe therefore adds nothing, which is
                    # the property a particle loop actually depends on
                    coeffs = zeros(nbasis(b))
                    for t in eachindex(buf)
                        j = basis_index(b, j₀ + t - 1)
                        1 ≤ j ≤ nbasis(b) || continue
                        coeffs[j] += buf[t]
                    end
                    @test all(iszero, coeffs)
                end

                # the closed domain belongs to the basis: both endpoints are inside, so the
                # guard must not swallow the interpolatory value there
                for d in 0:p, x in (a, z)

                    j₀ = evaluate_all!(buf, b, x, d)
                    ref = map(1:local_width(b)) do t
                        j = basis_index(b, j₀ + t - 1)
                        1 ≤ j ≤ nbasis(b) ? evaluate(b, j, x, d) : 0.0
                    end
                    @test isapprox(buf, ref; atol = 1e-9 * max(1, maximum(abs, ref)))
                end
            end

            # a periodic basis reduces its argument, so no real point is outside and the
            # block there is the periodic extension rather than zero
            p < ncells(m) || continue
            bp = BSplineBasis(m, p, Periodic())
            bufp = zeros(local_width(bp))
            L = last(m) - first(m)
            for d in 0:p, x in (first(m) - 0.5, last(m) + 0.5, first(m) - 3L)

                j₀ = evaluate_all!(bufp, bp, x, d)
                ref = zeros(local_width(bp))
                k₀ = evaluate_all!(ref, bp, first(m) + mod(x - first(m), L), d)
                @test j₀ == k₀
                @test bufp == ref
            end
        end
    end

    @testset "$(rpad("out-of-domain evaluation is allocation-free",76))" begin
        # The guard must not cost the particle loop an allocation, and it is on the path a
        # straying particle takes every step.
        b = BSplineBasis(UniformMesh(32, -10 .. 10), 3)
        buf = zeros(4)
        evaluate_all!(buf, b, -12.0, 0)                     # warm up
        evaluate_all!(buf, b, -12.0, 1)
        @test @allocated(evaluate_all!(buf, b, -12.5, 0)) == 0
        @test @allocated(evaluate_all!(buf, b, -12.5, 1)) == 0
        @test @allocated(evaluate_all!(buf, b, 12.5, 0)) == 0

        br = BSplineBasis(UniformMesh(32, -10 .. 10), 3, Dirichlet())
        bufr = zeros(local_width(br))
        evaluate_all!(bufr, br, -12.0, 0)
        @test @allocated(evaluate_all!(bufr, br, -12.5, 0)) == 0
    end

    @testset "$(rpad("evaluating a spline uses the local block",76))" begin
        # `evaluate(b, û, x)` sums the local block rather than the whole basis. The two
        # agree wherever the spline is defined, and the local path must not extrapolate:
        # outside a bounded domain `findcell` clamps and de Boor would happily continue the
        # nearest cell's polynomial, where the answer is zero.
        for p in 0:4, bc in (Free(), Dirichlet(), Neumann(), Periodic())

            # a condition involving D^k with k > p constrains nothing and is rejected
            p < constraint_order(bc) && continue
            b = BSplineBasis(UniformMesh(16, 0 .. 1), p, bc)
            û = randn(nbasis(b))
            reference(x, d) = sum(û[j] * evaluate(b, j, x, d) for j in eachindex(û))

            # Relative, at the same 1e-9 scale the testset above uses for `evaluate_all!`
            # against `_bspline`: the two sides evaluate each basis function by the two
            # different recursions, and then sum a different number of terms, so they agree
            # to a few ULP of a value whose size is set by `randn` and by the length of the
            # sum rather than to any fixed absolute figure. The `atol` is the floor for the
            # points where cancellation makes a relative comparison meaningless. A real
            # indexing fault moves this by O(1), not by 1e-13.
            for x in (0.0, 0.019, 0.25, 0.5, 0.5 + eps(), 0.937, 1.0), d in 0:min(p, 2)

                @test isapprox(evaluate(b, û, x, d), reference(x, d);
                    rtol = 1e-9, atol = 1e-12)
            end
        end

        for bc in (Free(), Dirichlet(), Neumann())
            b = BSplineBasis(UniformMesh(16, 0 .. 1), 3, bc)
            û = randn(nbasis(b))
            for x in (-1.0, -1e-9, 1 + 1e-9, 2.0)
                @test evaluate(b, û, x) == 0
            end
        end

        # a periodic basis takes any real argument and is the periodic extension
        bp = BSplineBasis(UniformMesh(16, 0 .. 1), 3, Periodic())
        v̂ = randn(nbasis(bp))
        for x in (-2.3, -0.1, 0.4, 1.7, 5.0)
            @test evaluate(bp, v̂, x) ≈ evaluate(bp, v̂, mod(x, 1.0))
        end

        # the sweep over a vector of points shares one block buffer, so it must agree with
        # the point-by-point path exactly -- same arithmetic, one allocation instead of many
        xs = collect(range(-0.2, 1.2; length = 41))
        for bc in (Free(), Dirichlet(), Neumann(), Periodic()), d in 0:2

            b = BSplineBasis(UniformMesh(16, 0 .. 1), 3, bc)
            û = randn(nbasis(b))
            @test evaluate(b, û, xs, d) == [evaluate(b, û, x, d) for x in xs]
            @test evaluate(b, û, xs, d) isa Vector{Float64}
        end

        # the sweep counts its own output rather than reusing the keys of `X`, so a vector
        # whose axes do not start at 1 is evaluated instead of raising a `BoundsError`
        bo = BSplineBasis(UniformMesh(16, 0 .. 1), 3)
        ûo = randn(nbasis(bo))
        Xoff = ShiftedVector(xs, 3)
        @test axes(Xoff) == (4:(3 + length(xs)),)
        @test evaluate(bo, ûo, Xoff) == evaluate(bo, ûo, xs)
        @test axes(evaluate(bo, ûo, Xoff)) == (Base.OneTo(length(xs)),)

        # `s(v)` routes to that sweep rather than broadcasting the scalar method
        s = Spline(BSplineBasis(UniformMesh(16, 0 .. 1), 3), randn(19))
        @test s(xs) == [s(x) for x in xs]

        @test_throws DimensionMismatch evaluate(
            BSplineBasis(UniformMesh(16, 0 .. 1), 3), zeros(3), xs)
    end

    @testset "$(rpad("local evaluation is allocation-free",76))" begin
        b = BSplineBasis(UniformMesh(32, -10 .. 10), 3)
        buf = zeros(4)
        evaluate_all!(buf, b, 0.3, 0)                       # warm up
        evaluate_all!(buf, b, 0.3, 1)
        @test @allocated(evaluate_all!(buf, b, 0.31, 0)) == 0
        @test @allocated(evaluate_all!(buf, b, 0.31, 1)) == 0

        bp = BSplineBasis(UniformMesh(32, 0 .. 2π), 3, Periodic())
        evaluate_all!(buf, bp, 0.3, 0)
        evaluate_all!(buf, bp, 0.3, 1)
        @test @allocated(evaluate_all!(buf, bp, 0.31, 0)) == 0
        @test @allocated(evaluate_all!(buf, bp, 0.31, 1)) == 0
    end

    @testset "$(rpad("findcell",76))" begin
        m = UniformMesh(4, 0 .. 1)
        @test findcell(m, 0.0) == 1
        @test findcell(m, 0.1) == 1
        @test findcell(m, 0.25) == 2
        @test findcell(m, 0.99) == 4
        # the right endpoint belongs to the last cell, not to a cell of its own
        @test findcell(m, 1.0) == 4
        # outside, clamped to the nearest
        @test findcell(m, -5.0) == 1
        @test findcell(m, 5.0) == 4

        mg = GeneralMesh([-20.0, -10.0, 0.0, 10.0, 20.0])
        @test findcell(mg, -15.0) == 1
        @test findcell(mg, -5.0) == 2
        @test findcell(mg, 5.0) == 3
        @test findcell(mg, 20.0) == 4
    end

    @testset "$(rpad("polynomial reproduction is the conservation predicate",76))" begin
        m = UniformMesh(12, -10 .. 10)

        # the clamped cubic basis both manuscripts use: 1, v and v² are in the span, which
        # is what makes mass, momentum and energy conservation survive the discretisation
        b = BSplineBasis(m, 3)
        q = SplineQuadrature(b)
        @test polynomial_reproduction(b) == 3
        for k in 0:3
            û = l2_projection(q, x -> x^k)
            for x in range(-10, 10; length = 11)
                @test isapprox(evaluate(b, û, x), x^k; rtol = 1e-9, atol = 1e-7)
            end
        end

        # a periodic basis reproduces the constants and nothing more: v is not periodic
        bp = BSplineBasis(m, 3, Periodic())
        qp = SplineQuadrature(bp)
        @test polynomial_reproduction(bp) == 0
        û = l2_projection(qp, x -> 1.0)
        @test all(isapprox(evaluate(bp, û, x), 1; atol = 1e-10)
        for x in range(-10, 9; length = 11))

        # a Dirichlet-recombined basis loses even the constants
        @test polynomial_reproduction(BSplineBasis(m, 3, Dirichlet())) == -1
        @test polynomial_reproduction(BSplineBasis(m, 3, Neumann())) == 0
        @test polynomial_reproduction(BSplineBasis(m, 3, Natural())) == 0
        @test polynomial_reproduction(BSplineBasis(m, 3, Robin(1.0, 1.0))) == -1
        @test polynomial_reproduction(BSplineBasis(m, 3, Robin(0.0, 1.0))) == 0
    end

    @testset "$(rpad("quadrature on all three bases",76))" begin
        m = UniformMesh(12, 0 .. 1)
        for bc in (Free(), Periodic(), Dirichlet(), Neumann())
            b = BSplineBasis(m, 3, bc)
            q = SplineQuadrature(b)
            M = mass_matrix(q)
            @test size(M) == (nbasis(b), nbasis(b))
            @test maximum(abs, M - M') < 1e-14
            @test minimum(eigvals(Symmetric(Matrix(M)))) > 0        # SPD
            Φ = basis_values(q, 0)
            w = quadrature_weights(q)
            @test maximum(abs, Matrix(M) - Matrix(Φ * Diagonal(w) * Φ')) < 1e-12
            rhs = randn(nbasis(b))
            @test maximum(abs, Matrix(M) * (mass_operator(q) \ rhs) - rhs) < 1e-8
            # the weights sum to the domain width whatever the basis
            @test sum(w) ≈ domainlength(b)
        end

        # the representation is a property of the basis, not of the mesh: only the periodic
        # basis on a uniform mesh gets the Fourier one, and only a periodic basis on any
        # other mesh is left with the sparse Cholesky -- a bounded basis has no seam, so its
        # mass matrix is banded outright
        @test mass_operator(SplineQuadrature(BSplineBasis(m, 3, Periodic()))) isa
              CirculantMass
        @test mass_operator(SplineQuadrature(BSplineBasis(m, 3))) isa BandedMass
        @test mass_operator(SplineQuadrature(BSplineBasis(m, 3, Dirichlet()))) isa
              BandedMass
        @test mass_operator(SplineQuadrature(
            BSplineBasis(GradedMesh(12, 0 .. 1), 3, Periodic()))) isa FactorizedMass
    end
end
