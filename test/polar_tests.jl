using SimpleSplines
using LinearAlgebra
using Random
using SparseArrays
using Test

@testset "$(rpad("Polar Spline Tests",80))" begin
    radial = BSplineBasis(UniformMesh(10, 0 .. 1), 3)
    angular = PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3)
    B = PolarSplineBasis(radial, angular)

    Ns = nbasis(radial)
    Nθ = nbasis(angular)
    angles = range(0, 2π; length = 17)[1:16]

    @testset "$(rpad("construction: three functions replace the first two rows",76))" begin
        @test B isa PolarSplineBasis
        @test ndims(B) == 2
        @test eltype(B) == Float64
        @test nbasis(B) == length(B) == 3 + (Ns - 2) * Nθ
        @test nbasis(parent(B)) == Ns * Nθ
        @test nbasis(B) == nbasis(parent(B)) - 2Nθ + 3
        @test degree(B) == (3, 3)
        @test order(B) == (4, 4)
        @test ncells(B) == (10, 16)
        @test bases(B) === (radial, angular)
        @test meshwidth(B) == meshwidth(parent(B))
        @test mesh(B) == mesh(parent(B))
        @test domain(B) == domain(parent(B))
        @test pole(B) == 0.0

        s = repr(B)
        @test occursin("PolarSplineBasis{Float64}", s)
        @test occursin("pole triangle of 3 for 2×$(Nθ)", s)
        @test B == PolarSplineBasis(radial ⊗ angular)
        @test hash(B) == hash(PolarSplineBasis(radial ⊗ angular))

        R = recombination_matrix(B)
        @test size(R) == (Ns * Nθ, nbasis(B))
        # The three pole columns reach the whole of the first two rows; every other column is
        # a single tensor-product function passed through.
        @test all(length(nzrange(R, k)) == 2Nθ for k in 1:3)
        @test all(length(nzrange(R, k)) == 1 for k in 4:nbasis(B))
        @test rank(Matrix(R)) == nbasis(B)
    end

    @testset "$(rpad("construction: what is rejected, and why",76))" begin
        @test_throws ArgumentError PolarSplineBasis(
            BSplineBasis(UniformMesh(10, 0 .. 1), 1), angular)              # degree < 2
        @test_throws ArgumentError PolarSplineBasis(
            PeriodicBSplineBasis(UniformMesh(10, 0 .. 1), 3), angular)      # no pole
        @test_throws ArgumentError PolarSplineBasis(
            BSplineBasis(UniformMesh(10, 0 .. 1), 3, Dirichlet()), angular) # recombined
        @test_throws ArgumentError PolarSplineBasis(radial, radial)         # not periodic
        @test_throws ArgumentError PolarSplineBasis(
            radial, PeriodicBSplineBasis(UniformMesh(2, 0 .. 2π), 3))       # Nθ < 3

        # A *clamped* radial basis needs no counterpart to the last one: it has `ncells + p`
        # functions and both are bounded below already, so at least one radial row always
        # survives the pole triangle.
        @test minimum(nbasis(BSplineBasis(UniformMesh(n, 0 .. 1), p))
        for p in 2:6, n in 1:6) ==
              3

        # A rim condition removes one, and then the guard is reachable: the smallest case is
        # one cell at degree two, which leaves the two rows the triangle replaces and nothing
        # else. This is why that guard exists rather than being an unreachable assertion.
        @test nbasis(RecombinedBSplineBasis(
            BSplineBasis(UniformMesh(1, 0 .. 1), 2), Free(), Dirichlet())) == 2
        @test_throws ArgumentError PolarSplineBasis(
            RecombinedBSplineBasis(
                BSplineBasis(UniformMesh(1, 0 .. 1), 2), Free(), Dirichlet()), angular)

        # Recombining the *pole* end is what is rejected, at either order of the two
        # conditions, and a rim-only recombination is what is accepted.
        @test_throws ArgumentError PolarSplineBasis(
            RecombinedBSplineBasis(radial, Dirichlet(), Free()), angular)
        @test_throws ArgumentError PolarSplineBasis(
            RecombinedBSplineBasis(radial, Neumann(), Dirichlet()), angular)
        @test PolarSplineBasis(
            RecombinedBSplineBasis(radial, Free(), Dirichlet()), angular) isa
              PolarSplineBasis
    end

    @testset "$(rpad("the premise: only the first two rows reach the pole",76))" begin
        # The construction rests on these, and they are a theorem about a clamped knot vector
        # of degree ≥ 2 rather than something the code arranges. If they were to stop holding
        # the pole triangle would be constraining the wrong coefficients silently.
        @test evaluate(radial, 1, 0.0) == 1
        @test all(evaluate(radial, i, 0.0) == 0 for i in 2:Ns)
        @test evaluate(radial, 1, 0.0, 1) ≈ -evaluate(radial, 2, 0.0, 1)
        @test evaluate(radial, 2, 0.0, 1) ≈ degree(radial) * ncells(radial)
        @test all(evaluate(radial, i, 0.0, 1) == 0 for i in 3:Ns)
    end

    @testset "$(rpad("the pole triangle",76))" begin
        V = pole_triangle(B)
        @test size(V) == (3, 2)
        # Equilateral, centred on the pole, of the radius that makes the basis non-negative.
        r = 2 / evaluate(radial, 2, 0.0, 1)
        @test all(≈(r), [norm(V[k, :]) for k in 1:3])
        @test norm(sum(V; dims = 1)) < 1e-14
        @test V[1, :] ≈ [r, 0.0]

        # Ψ_k restricted to the pole is the barycentric coordinate that is one at vertex k:
        # value 1/3 there, and a gradient that takes it to one at its own vertex and to zero
        # at the others.
        @test all(evaluate(B, k, (0.0, θ)) ≈ 1 / 3 for k in 1:3, θ in angles)
        for k in 1:3, l in 1:3
            # ℓ_k(v_l) through the chart: value at the pole plus ∇ℓ_k · v_l, and ∇ℓ_k is read
            # off the radial derivative along the direction of v_l.
            g = [2 / (3r) * V[k, 1] / r, 2 / (3r) * V[k, 2] / r]
            @test 1 / 3 + dot(g, V[l, :]) ≈ (k == l ? 1.0 : 0.0) atol=1e-14
        end
    end

    @testset "$(rpad("C⁰ at the pole, against the tensor-product control",76))" begin
        û = randn(nbasis(B))
        vals = [evaluate(B, û, (0.0, θ)) for θ in angles]
        @test maximum(vals) - minimum(vals) < 1e-14

        # Every basis function, not only a random combination.
        for k in 1:nbasis(B)
            v = [evaluate(B, k, (0.0, θ)) for θ in angles]
            @test maximum(v) - minimum(v) < 1e-14
        end

        # The control that must fail: the parent, whose θ-dependence at s = 0 is free.
        P = parent(B)
        v̂ = randn(size(P)...)
        pv = [evaluate(P, v̂, (0.0, θ)) for θ in angles]
        @test maximum(pv) - minimum(pv) > 1e-2
    end

    @testset "$(rpad("C¹ at the pole, against the C⁰-only control",76))" begin
        θnodes = nodes(angular)
        C(θ) = evaluate(angular, cos.(θnodes), θ)
        S(θ) = evaluate(angular, sin.(θnodes), θ)
        A = [C.(angles) S.(angles)]

        # One gradient explains the radial derivative at every angle exactly when the spline
        # is C¹ at the pole in the pseudo-Cartesian chart.
        residual(∂s) = (
            b = [∂s(θ) for θ in angles]; norm(A * (A \ b) - b, Inf) /
                                         max(1, norm(b, Inf)))

        û = randn(nbasis(B))
        @test residual(θ -> evaluate(B, û, (0.0, θ), (1, 0))) < 1e-12

        # The control: C⁰ imposed and nothing more — the first radial row tied to a single
        # coefficient, the second left free. It is C⁰ by construction, which is what makes it
        # a control for C¹ alone, and its radial derivative at the pole is a generic angular
        # spline that no single gradient fits.
        P = parent(B)
        R₀ = spzeros(Ns * Nθ, 1 + Nθ + (Ns - 2) * Nθ)
        for j in 1:Nθ
            R₀[1 + (j - 1) * Ns, 1] = 1
            R₀[2 + (j - 1) * Ns, 1 + j] = 1
        end
        for j in 1:Nθ, i in 3:Ns

            R₀[i + (j - 1) * Ns, 1 + Nθ + (i - 2) + (j - 1) * (Ns - 2)] = 1
        end
        ĉ = R₀ * randn(size(R₀, 2))
        c0only(x, d) = evaluate(P, reshape(ĉ, size(P)), x, d)

        v = [c0only((0.0, θ), (0, 0)) for θ in angles]
        @test maximum(v) - minimum(v) < 1e-14                     # it is C⁰ …
        @test residual(θ -> c0only((0.0, θ), (1, 0))) > 1e-2      # … and it is not C¹
    end

    @testset "$(rpad("the chart's constants and linears are in the space exactly",76))" begin
        θnodes = nodes(angular)
        greville = nodes(radial)
        V = pole_triangle(B)

        function chart(α, β)
            û = zeros(nbasis(B))
            for k in 1:3
                û[k] = α * V[k, 1] + β * V[k, 2]
            end
            for i in 3:Ns, j in 1:Nθ

                û[3 + (i - 2) + (j - 1) * (Ns - 2)] = greville[i] * (α * cos(θnodes[j]) +
                                                       β * sin(θnodes[j]))
            end
            û
        end

        pts = [(s, θ) for s in range(0, 1; length = 7), θ in angles]
        # An absolute tolerance, not `≈`: at the pole both sides are zero, and a relative
        # comparison of two numbers that are both round-off says nothing.
        @test all(abs(evaluate(B, ones(nbasis(B)), x) - 1) < 1e-13 for x in pts)
        @test all(abs(evaluate(B, chart(1.0, 0.0), x) -
                      pseudo_cartesian(B, x)[1]) < 1e-13 for x in pts)
        @test all(abs(evaluate(B, chart(0.0, 1.0), x) -
                      pseudo_cartesian(B, x)[2]) < 1e-13 for x in pts)

        @test pseudo_cartesian(B, (0.0, 1.3)) == (0.0, 0.0)
    end

    @testset "$(rpad("evaluation: the paths agree, and evaluate_all misses nothing",76))" begin
        û = randn(nbasis(B))
        pts = [(0.0, 0.7), (1e-9, 2.1), (0.03, 4.0), (1 / 10, 0.0), (0.4, 5.9), (1.0, 3.0)]

        for x in pts, d in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
            # The spline through the parent's local block, against the sum over every index.
            direct = sum(û[k] * evaluate(B, k, x, d) for k in eachindex(B))
            @test evaluate(B, û, x, d) ≈ direct atol=1e-10

            # `evaluate_all` must report every function that is nonzero at the point; if it
            # dropped one the contraction below would be short by that term.
            idx, vals = evaluate_all(B, x, d)
            @test allunique(idx)
            @test sum(û[idx] .* vals) ≈ direct atol=1e-10
            @test all(abs(evaluate(B, k, x, d) - v) < 1e-10 for (k, v) in zip(idx, vals))
        end

        # Away from the pole the block is the tensor-product one and nothing else.
        idx, _ = evaluate_all(B, (0.55, 2.2))
        @test length(idx) == 16
        @test !any(≤(3), idx)

        # In the first two radial cells the three pole functions join it.
        for s in (0.0, 0.05, 0.15)
            idx, _ = evaluate_all(B, (s, 2.2))
            @test idx[1:3] == [1, 2, 3]
        end

        @test evaluate(B, û, [(0.3, 1.0), (0.6, 2.0)]) ≈
              [evaluate(B, û, (0.3, 1.0)), evaluate(B, û, (0.6, 2.0))]
        # A single point written as a vector is a point, not a one-element list of points.
        @test evaluate(B, û, [0.3, 1.0]) == evaluate(B, û, (0.3, 1.0))
        @test evaluate(B, û, [[0.3, 1.0], [0.6, 2.0]]) ≈
              [evaluate(B, û, (0.3, 1.0)), evaluate(B, û, (0.6, 2.0))]
        @test B[(0.3, 1.0), 7] == evaluate(B, 7, (0.3, 1.0))
        @test B((0.3, 1.0), 7) == evaluate(B, 7, (0.3, 1.0))
        @test_throws DimensionMismatch parent_coefficients(B, randn(nbasis(B) + 1))
        @test parent_coefficients(B, û) ≈ reshape(recombination_matrix(B) * û, Ns, Nθ)
    end

    @testset "$(rpad("assembly: the tables, the mass matrix and the integrals",76))" begin
        q = PolarSplineQuadrature(B)

        @test basis(q) === B
        @test nbasis(q) == nbasis(B)
        @test ndims(q) == 2
        @test degree(q) == (3, 3)
        @test quadrature_nodes(q) === quadrature_nodes(parent(q))
        @test length(quadrature_weights(q)) == prod(quadrature_grid_size(q))

        Φ = basis_values(q, (0, 0))
        @test size(Φ) == (nbasis(B), prod(quadrature_grid_size(q)))
        @test basis_values(q) === Φ                           # memoised, and d = 0 is scalar
        @test basis_values(q, 0) === Φ
        @test_throws ArgumentError basis_values(q, 1)

        # The table is the tabulation, point by point.
        grid = [x for x in Iterators.product(quadrature_nodes(q)...)][:]
        for (r, x) in zip([1, 37, length(grid)], grid[[1, 37, end]])
            @test all(abs(Φ[k, r] - evaluate(B, k, x)) < 1e-12 for k in 1:nbasis(B))
        end

        M = mass_matrix(q)
        # Exactly, not to round-off: the assembly symmetrises the triple product, as the
        # one-dimensional one does, so a caller may hand `M` to anything that demands it.
        @test issymmetric(M)
        @test isposdef(Symmetric(Matrix(M)))
        @test M ≈ mixed_matrix(q, (0, 0), (0, 0))
        @test M ≈
              recombination_matrix(B)' *
              mass_matrix(TensorProductQuadrature(parent(B))) *
              recombination_matrix(B) atol=1e-12
        @test mass_operator(q) === mass_factorization(q)

        𝟙 = ones(nbasis(B))
        @test 𝟙' * M * 𝟙 ≈ 2π                                  # the area of the square
        @test M * 𝟙 ≈ basis_integrals(q)                       # the partition of unity
        @test sum(basis_integrals(q)) ≈ 2π

        # ∫ f φ_k φ_l with f ≡ 1 is the mass matrix, and the sampled and functional forms of
        # the weight agree.
        @test weighted_matrix(q, x -> 1.0, (0, 0), (0, 0)) ≈ M
        w = ones(length(quadrature_weights(q)))
        @test weighted_matrix(q, w, (0, 0), (0, 0)) ≈ M
        @test_throws DimensionMismatch weighted_matrix(q, w[1:(end - 1)], (0, 0), (0, 0))

        K = stiffness_matrix(q)
        # The stiffness matrix is a sum of two `mixed_matrix` products and is not symmetrised,
        # so this is symmetry to round-off rather than exact symmetry.
        @test norm(K - K', Inf) < 1e-14 * norm(K, Inf)
        @test norm(K * 𝟙, Inf) < 1e-10                         # the constants are its kernel
        @test K ≈ mixed_matrix(q, (1, 0), (1, 0)) + mixed_matrix(q, (0, 1), (0, 1))
    end

    @testset "$(rpad("projection: the constants and the chart's linears are exact",76))" begin
        q = PolarSplineQuadrature(B)
        pts = [(s, θ) for s in range(0, 1; length = 7), θ in angles]

        û = l2_projection(q, x -> 1.0)
        @test all(abs(evaluate(B, û, x) - 1) < 1e-11 for x in pts)

        v̂ = l2_projection(q, x -> pseudo_cartesian(B, x)[2])
        @test all(abs(evaluate(B, v̂, x) - pseudo_cartesian(B, x)[2]) < 1e-11 for x in pts)

        ŵ = similar(û)
        l2_projection!(ŵ, q, x -> 1.0)
        @test ŵ ≈ û
        @test_throws DimensionMismatch l2_projection!(zeros(nbasis(B) + 1), q, x -> 1.0)
    end

    @testset "$(rpad("a homogeneous-Dirichlet rim composes with the pole triangle",76))" begin
        rim = RecombinedBSplineBasis(radial, Free(), Dirichlet())
        D = PolarSplineBasis(rim, angular)

        # One angular row of degrees of freedom is what a rim condition costs, and the pole
        # triangle is untouched: three functions still, reaching the same two rows.
        @test nbasis(D) == nbasis(B) - Nθ
        @test nbasis(D) == 3 + (Ns - 3) * Nθ
        RD = recombination_matrix(D)
        @test size(RD) == (nbasis(rim) * Nθ, nbasis(D))
        @test all(length(nzrange(RD, k)) == 2Nθ for k in 1:3)
        @test rank(Matrix(RD)) == nbasis(D)

        # The premise of the composition: the two functions the triangle is built from are the
        # clamped parent's, unchanged, so the number the whole construction rests on is too.
        @test all(evaluate(rim, k, x) == evaluate(radial, k, x)
        for k in 1:2, x in range(0, 1; length = 41))
        @test evaluate(rim, 2, 0.0, 1) == evaluate(radial, 2, 0.0, 1)
        @test pole_triangle(D) == pole_triangle(B)

        # The rim itself: every basis function vanishes there, so no spline in the space has a
        # value at s = 1. This is the property the space is built for.
        @test all(evaluate(D, k, (1.0, θ)) == 0 for k in 1:nbasis(D), θ in angles)
        û = randn(nbasis(D))
        @test all(abs(evaluate(D, û, (1.0, θ))) < 1e-14 for θ in angles)

        # C⁰ and C¹ at the pole are unaffected — nothing the triangle reads has changed.
        spread(v) = maximum(v) - minimum(v)
        @test maximum(k -> spread([evaluate(D, k, (0.0, θ)) for θ in angles]),
            1:nbasis(D)) < 1e-14
        θn = nodes(angular)
        A = [
             [evaluate(angular, cos.(θn), θ) for θ in angles] [evaluate(angular, sin.(θn), θ)
                                                               for θ in angles]]
        b = [evaluate(D, û, (0.0, θ), (1, 0)) for θ in angles]
        @test norm(A * (A \ b) - b, Inf) / max(1, norm(b, Inf)) < 1e-14

        # The partition of unity is now false, and that is the point: it survives everywhere
        # except the last radial cell, and the constant is therefore no longer in the space.
        h = meshwidth(radial)
        @test all(abs(sum(evaluate(D, k, (s, θ)) for k in 1:nbasis(D)) - 1) < 1e-13
        for s in range(0, 1 - h; length = 21), θ in angles)
        @test all(sum(evaluate(D, k, (1.0, θ)) for k in 1:nbasis(D)) == 0 for θ in angles)

        # evaluate_all rebuilds the index formula from `Ns` rather than reading `R`, so it is
        # the one path a rim condition could break silently. Both its indices and its values.
        for x in ((0.01, 1.1), (0.5, 2.3), (0.97, 5.0)), d in ((0, 0), (1, 0), (0, 1))

            idx, vals = evaluate_all(D, x, d)
            full = [evaluate(D, k, x, d) for k in 1:nbasis(D)]
            @test all(1 .<= idx .<= nbasis(D))
            @test all(vals[t] ≈ full[idx[t]] for t in eachindex(idx))
            # Absolute, not `≈`: for d ≠ (0,0) both sums are a derivative of a partition of
            # unity, which is zero, and a relative tolerance between two round-off-sized
            # numbers compares nothing. A dropped function would show as its own O(1) value.
            @test abs(sum(vals) - sum(full)) < 1e-12
        end

        # Assembly goes through `R` alone, so it needs nothing of its own — but the mass
        # matrix must still be nonsingular on the smaller space.
        qd = PolarSplineQuadrature(D)
        M = mass_matrix(qd)
        @test size(M) == (nbasis(D), nbasis(D))
        @test issymmetric(M)
        @test isposdef(Matrix(M))

        # And the constant has left: it is in the free space to round-off and not in this one.
        @test all(abs(evaluate(B, l2_projection(PolarSplineQuadrature(B), x -> 1.0),
                      (s, 0.7)) - 1) < 1e-11 for s in range(0, 1; length = 9))
        v̂ = l2_projection(qd, x -> 1.0)
        @test maximum(abs(evaluate(D, v̂, (s, 0.7)) - 1) for s in range(0, 1; length = 9)) >
              0.5
    end
end
