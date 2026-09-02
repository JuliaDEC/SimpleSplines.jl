using SimpleSplines
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "$(rpad("Tensor Product Tests",80))" begin
    @testset "$(rpad("construction: degree, domain and condition are per-axis",76))" begin
        bx = BSplineBasis(UniformMesh(10, 0 .. 2π), 3, Periodic())
        bv = BSplineBasis(UniformMesh(8, -10 .. 10), 4, Free())
        B = bx ⊗ bv

        @test B isa TensorProductBasis
        @test ndims(B) == 2
        @test size(B) == (10, 12)
        @test size(B, 1) == 10
        @test size(B, 2) == 12
        @test nbasis(B) == length(B) == 120
        @test degree(B) == (3, 4)
        @test order(B) == (4, 5)
        @test ncells(B) == (10, 8)
        @test bases(B) === (bx, bv)
        @test eltype(B) == Float64
        @test boundary(B) == (Periodic(), (Free(), Free()))

        # ⊗ associates flat rather than nesting
        b3 = BSplineBasis(UniformMesh(4, 0 .. 1), 1)
        @test ndims(bx ⊗ bv ⊗ b3) == 3
        @test bases(bx ⊗ bv ⊗ b3) === (bx, bv, b3)
        @test TensorProductBasis(bx, bv) == B
        @test hash(TensorProductBasis(bx, bv)) == hash(B)

        # a one-factor product is legal and behaves like the axis it wraps
        @test ndims(TensorProductBasis(bx)) == 1
        @test_throws ArgumentError TensorProductBasis()

        # `kron` has no one-argument method, so assembling a one-factor product by splatting
        # into it threw. Nothing else in the suite builds a quadrature on a `D = 1` basis.
        B1 = TensorProductBasis(BSplineBasis(UniformMesh(7, 0 .. 1), 3))
        op1 = mass_operator(TensorProductQuadrature(B1))
        @test mass_matrix(op1) == mass_matrix(only(mass_factors(op1)))
        @test Matrix(op1) == Matrix(mass_matrix(op1))

        # `size(op, d)` answers like `Base` at both ends of the range
        @test size(op1, 1) == size(op1, 2) == prod(size(B1))
        @test size(op1, 3) == 1
        @test_throws BoundsError size(op1, 0)
    end

    @testset "$(rpad("domain is a DomainSets product",76))" begin
        B = BSplineBasis(UniformMesh(4, 0 .. 1), 2) ⊗
            BSplineBasis(UniformMesh(4, -2 .. 3), 2)
        d = domain(B)
        @test SVector(0.5, 1.0) ∈ d
        @test SVector(0.5, 9.0) ∉ d
        @test [0.5, 1.0] ∈ d
        @test [-0.5, 1.0] ∉ d
        @test domain(bases(B)[1]) == (0.0 .. 1.0)
        @test domain(bases(B)[2]) == (-2.0 .. 3.0)
        @test leftendpoint(domain(bases(B)[2])) == -2.0
        @test rightendpoint(domain(bases(B)[2])) == 3.0
    end

    @testset "$(rpad("evaluation factorises",76))" begin
        bx = BSplineBasis(UniformMesh(10, 0 .. 2π), 3, Periodic())
        bv = BSplineBasis(UniformMesh(8, -10 .. 10), 4)
        B = bx ⊗ bv
        Random.seed!(0x1234)
        for _ in 1:40
            x = (2π * rand(), 20rand() - 10)
            I = (rand(1:size(B, 1)), rand(1:size(B, 2)))
            @test evaluate(B, I, x) ≈ evaluate(bx, I[1], x[1]) * evaluate(bv, I[2], x[2])
            @test evaluate(B, CartesianIndex(I), x) ≈ evaluate(B, I, x)
            @test evaluate(B, LinearIndices(B)[I...], x) ≈ evaluate(B, I, x)
            @test B[x, I] ≈ evaluate(B, I, x)
            # a mixed derivative
            @test evaluate(B, I, x, (1, 2)) ≈
                  evaluate(bx, I[1], x[1], 1) * evaluate(bv, I[2], x[2], 2)
        end
    end

    @testset "$(rpad("local evaluation of a spline matches the full sum",76))" begin
        B = TensorProductBasis(
            BSplineBasis(UniformMesh(5, 0 .. 1), 1),
            BSplineBasis(UniformMesh(4, -1 .. 1), 2, Periodic()),
            BSplineBasis(UniformMesh(6, 2 .. 5), 3, Dirichlet()))
        @test size(B) == (6, 4, 7)
        @test degree(B) == (1, 2, 3)
        Random.seed!(0xbeef)
        û = randn(size(B)...)
        for _ in 1:15
            x = (rand(), 2rand() - 1, 3rand() + 2)
            ref = sum(û[I] * evaluate(B, I, x) for I in CartesianIndices(B))
            @test isapprox(evaluate(B, û, x), ref; atol = 1e-10)
            for k in 1:3
                d = ntuple(i -> i == k ? 1 : 0, 3)
                refd = sum(û[I] * evaluate(B, I, x, d) for I in CartesianIndices(B))
                @test isapprox(evaluate(B, û, x, d), refd; atol = 1e-9)
            end
        end

        # the per-axis blocks are the factors of the D-dimensional block
        bufs = ntuple(k -> zeros(local_width(bases(B)[k])), 3)
        x = (0.37, 0.11, 3.4)
        j₀ = evaluate_all!(bufs, B, x)
        for k in 1:3
            jk, vk = evaluate_all(bases(B)[k], x[k])
            @test j₀[k] == jk
            @test bufs[k] ≈ vk
        end

        # The local sum stays inferable, and allocates its per-axis buffers and nothing else,
        # whatever mix of basis types the axes carry. Two separate faults cost both at once
        # on exactly this basis: a weight accumulated from inside the index `ntuple` is
        # boxed, and `size(B, k)` reads the tuple of bases with a loop variable, which
        # dispatches dynamically once per term when the axes differ in type. Together they
        # were 72 832 bytes and 34x the runtime here, while `evaluate_all!` on the same bases
        # was already allocation-free -- which is why the suite could not see it.
        @test (@inferred evaluate(B, û, x)) isa Float64
        @test (@inferred evaluate(B, û, x, (0, 1, 0))) isa Float64
        @test (@allocated evaluate(B, û, x)) < 1024
    end

    @testset "$(rpad("a point outside the domain evaluates to zero",76))" begin
        # The fast path must agree with the full sum *outside* the domain as well, not only
        # inside it. It did not: with a one-hot coefficient array on a 3 ⊗ 2 basis over
        # 0 .. 1, `evaluate(B, û, (-0.5, 0.4))` gave 7.0542 where `evaluate(B, I, (-0.5,
        # 0.4))` gives 0.0 -- the local block on a bounded axis was the nearest cell's
        # polynomial extrapolated. Random coefficients reached -418 where the answer is
        # zero, and this is exactly the straying-particle case the one-dimensional
        # `evaluate` docstring calls out.
        bx = BSplineBasis(UniformMesh(8, 0 .. 1), 3)
        bv = BSplineBasis(UniformMesh(8, 0 .. 1), 2, Dirichlet())
        bp = BSplineBasis(UniformMesh(6, 0 .. 1), 3, Periodic())

        for B in (bx ⊗ bv, bx ⊗ bv ⊗ bp, bv ⊗ bx)
            D = ndims(B)
            Random.seed!(0x0d0d0d)
            û = randn(size(B)...)
            reference(x, d) = sum(û[I] * evaluate(B, I, x, d) for I in CartesianIndices(B))

            # a point outside on one axis, on all of them, and just outside
            outside = [ntuple(k -> k == 1 ? -0.5 : 0.4, D),
                ntuple(k -> k == 1 ? 2.0 : 0.5, D),
                ntuple(_ -> -1.0, D),
                ntuple(k -> k == 1 ? -1e-9 : 0.5, D),
                ntuple(k -> k == 1 ? 1 + 1e-9 : 0.5, D)]
            # every point above is outside on axis 1, which is bounded in all three
            # products, so the whole outer product vanishes however the other axes wrap
            for x in outside
                d = ntuple(_ -> 0, D)
                @test evaluate(B, û, x, d) == reference(x, d) == 0
                @test Spline(B, û)(x) == 0
            end

            # and the identity in the `evaluate` docstring holds off-domain too, which is
            # the doctest that was failing there
            I = ntuple(k -> 2 + k, D)
            ê = zeros(size(B)...)
            ê[I...] = 1.0
            for x in vcat(outside, [ntuple(_ -> 0.3, D)])
                @test evaluate(B, ê, x) ≈ evaluate(B, I, x)
            end
        end
    end

    @testset "$(rpad("Kronecker mass: exact against the dense product",76))" begin
        B = BSplineBasis(UniformMesh(10, 0 .. 2π), 3, Periodic()) ⊗
            BSplineBasis(UniformMesh(8, -10 .. 10), 4)
        q = TensorProductQuadrature(B)
        op = mass_operator(q)

        @test op isa KroneckerMass
        @test ndims(op) == 2
        @test size(op) == (nbasis(B), nbasis(B))
        @test length(mass_factors(op)) == 2
        # each axis keeps its own representation: the periodic uniform one takes the Fourier
        # path, the clamped one the banded Cholesky
        @test mass_factors(op)[1] isa CirculantMass
        @test mass_factors(op)[2] isa BandedMass

        Mdense = Matrix(op)
        @test Mdense ≈ kron(Matrix(mass_matrix(mass_factors(op)[2])),
            Matrix(mass_matrix(mass_factors(op)[1])))

        Random.seed!(0x5eed)
        bb = randn(size(B)...)
        @test maximum(abs, Mdense * vec(bb) - vec(op * bb)) < 1e-9
        @test maximum(abs, vec(op \ bb) - Mdense \ vec(bb)) < 1e-8
        # a vector argument is reshaped on the documented convention
        @test maximum(abs, op \ vec(bb) - vec(op \ bb)) < 1e-12
        # round trip
        @test maximum(abs, op \ (op * bb) - bb) < 1e-9
        # in place
        y = copy(bb)
        ldiv!(op, y)
        @test maximum(abs, y - (op \ bb)) < 1e-12
    end

    @testset "$(rpad("Kronecker mass: three axes, mixed representations",76))" begin
        B = TensorProductBasis(
            BSplineBasis(UniformMesh(5, 0 .. 1), 2),
            BSplineBasis(UniformMesh(6, -1 .. 1), 2, Periodic()),
            BSplineBasis(UniformMesh(5, 2 .. 5), 3, Dirichlet()))
        q = TensorProductQuadrature(B)
        op = mass_operator(q)
        Random.seed!(0xfeed)
        b3 = randn(size(B)...)
        # the middle-axis fibres are strided, which is what the contiguous buffer in
        # `_apply_along!` exists for: an FFTW plan rejects a wrong-strides array outright
        @test maximum(abs, vec(op \ b3) - Matrix(op) \ vec(b3)) < 1e-7
        @test maximum(abs, Matrix(op) * vec(b3) - vec(op * b3)) < 1e-9
    end

    @testset "$(rpad("projection: polynomials are exact",76))" begin
        B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗
            BSplineBasis(UniformMesh(6, 0 .. 2), 2)
        q = TensorProductQuadrature(B)
        @test polynomial_reproduction(B) == 2               # min(3, 2)
        f = x -> x[1]^2 * x[2] + 3x[1] - x[2]^2 + 1
        v̂ = l2_projection(q, f)
        Random.seed!(0xabcd)
        for _ in 1:15
            x = (rand(), 2rand())
            @test isapprox(evaluate(B, v̂, x), f(x); atol = 1e-9)
        end

        # the in-place form agrees
        ŵ = similar(v̂)
        l2_projection!(ŵ, q, f)
        @test maximum(abs, ŵ - v̂) < 1e-12

        # a sample given on the grid rather than as a function gives the same answer
        F = quadrature_sample(q, f)
        @test size(F) == quadrature_grid_size(q)
        @test maximum(abs, l2_projection(q, F) - v̂) < 1e-12
    end

    @testset "$(rpad("projection: a non-separable function converges",76))" begin
        # Checked by convergence, not by a tolerance plucked from the air: a wrong
        # contraction gives an error that does not converge at all, which a fixed tolerance
        # on one coarse mesh cannot tell apart from ordinary discretisation error.
        #
        # The test function is periodic in x[1]. A term such as x[1]*x[2] would be
        # discontinuous across the seam of the periodic axis and the projection would then
        # converge at first order however smooth the formula looks -- that is
        # `polynomial_reproduction == 0` on that axis, not a defect.
        g = x -> sin(x[1]) * exp(-x[2]^2 / 8) + cos(2x[1]) * x[2] / 20
        Random.seed!(0x0f0f)
        pts = [(2π * rand(), 6rand() - 3) for _ in 1:60]
        errs = Float64[]
        for n in (8, 16, 32)
            Bn = BSplineBasis(UniformMesh(n, 0 .. 2π), 3, Periodic()) ⊗
                 BSplineBasis(UniformMesh(n, -10 .. 10), 3)
            qn = TensorProductQuadrature(Bn)
            ûn = l2_projection(qn, g)
            push!(errs, maximum(abs(evaluate(Bn, ûn, x) - g(x)) for x in pts))
        end
        @test errs[1] > errs[2] > errs[3]
        # cubic splines are fourth order, so halving h should cut the error by about 16
        @test errs[1] / errs[2] > 6
        @test errs[2] / errs[3] > 6
        @test errs[3] < 1e-4
    end

    @testset "$(rpad("projection: agrees with the dense Kronecker assembly",76))" begin
        # the whole point of the factored path is that it computes the same thing as the
        # textbook one, so at a size where both are possible they are compared directly
        B = BSplineBasis(UniformMesh(5, 0 .. 1), 2) ⊗
            BSplineBasis(UniformMesh(4, 0 .. 1), 3)
        q = TensorProductQuadrature(B)
        f = x -> exp(x[1]) * cos(3x[2])

        û = l2_projection(q, f)

        # the same projection, assembled entry by entry from the one-dimensional tables
        q1, q2 = quadratures(q)
        x1, x2 = quadrature_nodes(q), quadrature_weights(q)
        X1, X2 = quadrature_nodes(q)
        W1, W2 = quadrature_weights(q)
        Φ1, Φ2 = basis_values(q1, 0), basis_values(q2, 0)
        L = zeros(size(B)...)
        for i in 1:size(B, 1), j in 1:size(B, 2)

            s = 0.0
            for r in eachindex(X1), t in eachindex(X2)

                s += f((X1[r], X2[t])) * Φ1[i, r] * Φ2[j, t] * W1[r] * W2[t]
            end
            L[i, j] = s
        end
        Mdense = kron(Matrix(mass_matrix(quadratures(q)[2].mass)),
            Matrix(mass_matrix(quadratures(q)[1].mass)))
        @test maximum(abs, vec(û) - Mdense \ vec(L)) < 1e-10
    end

    @testset "$(rpad("contract with derivatives",76))" begin
        B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗
            BSplineBasis(UniformMesh(8, 0 .. 1), 3)
        q = TensorProductQuadrature(B)
        # ∫ ∂₁φ_I · 1 dx should vanish for every interior function and telescope at the ends
        F = ones(quadrature_grid_size(q)...)
        G = contract(q, F, (1, 0))
        @test size(G) == size(B)
        # ∫ ∂₁(φ_i(x)φ_j(y)) dx dy = [φ_i]₀¹ ∫φ_j, so only the two end rows are nonzero
        @test maximum(abs, G[2:(end - 1), :]) < 1e-12
        @test maximum(abs, G[1, :]) > 1e-3
    end

    @testset "$(rpad("contract does not consume its sample",76))" begin
        # The weights are applied in place, so `contract` must copy first. It did not: when
        # the sample was already an `Array` of the working element type the conversion was
        # the identity, and the caller's array came back scaled by the quadrature weights.
        # The visible symptom was that a second projection of the same array gave a
        # different answer -- and `quadrature_sample` returns exactly such an array.
        B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗
            BSplineBasis(UniformMesh(6, 0 .. 1), 2)
        q = TensorProductQuadrature(B)

        F = quadrature_sample(q, x -> sin(x[1]) * x[2])
        F₀ = copy(F)
        contract(q, F)
        @test F == F₀

        @test l2_projection(q, F) ≈ l2_projection(q, F)
        @test F == F₀

        û = zeros(size(B)...)
        l2_projection!(û, q, F)
        @test F == F₀

        # an integer sample is promoted and must be left alone just the same
        G = ones(Int, quadrature_grid_size(q)...)
        G₀ = copy(G)
        contract(q, G)
        @test G == G₀
    end

    @testset "$(rpad("dimension mismatches are reported",76))" begin
        B = BSplineBasis(UniformMesh(4, 0 .. 1), 2) ⊗
            BSplineBasis(UniformMesh(4, 0 .. 1), 2)
        q = TensorProductQuadrature(B)
        @test_throws DimensionMismatch evaluate(B, zeros(3, 3), (0.5, 0.5))
        @test_throws DimensionMismatch l2_projection!(zeros(3, 3), q, x -> 1.0)
        @test_throws DimensionMismatch contract(q, zeros(3, 3))
        @test_throws DimensionMismatch TensorProductQuadrature(B; nq = (2, 3, 4))
    end

    @testset "$(rpad("per-axis nq and dmax",76))" begin
        B = BSplineBasis(UniformMesh(6, 0 .. 1), 2) ⊗
            BSplineBasis(UniformMesh(6, 0 .. 1), 4)
        q = TensorProductQuadrature(B; nq = (3, 7), dmax = (1, 2))
        @test length(quadrature_nodes(q)[1]) == 6 * 3
        @test length(quadrature_nodes(q)[2]) == 6 * 7
        @test_nowarn basis_values(quadratures(q)[1], 1)
        @test_throws ArgumentError basis_values(quadratures(q)[1], 2)
        @test_nowarn basis_values(quadratures(q)[2], 2)
    end
end
