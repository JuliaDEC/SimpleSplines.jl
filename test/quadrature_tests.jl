using SimpleSplines
using LinearAlgebra
using Random
using Test

@testset "$(rpad("Spline Quadrature Tests",80))" begin

    @testset "$(rpad("quadrature_order",76))" begin
        # nq points are exact to degree 2nq-1; degree 3p-1 is what consistency needs
        @test quadrature_order.(1:4) == [2, 3, 5, 6]
        @test all(2 * quadrature_order(p) - 1 ≥ 3p - 1 for p in 1:8)
        @test all(2 * (quadrature_order(p) - 1) - 1 < 3p - 1 for p in 1:8)
    end

    @testset "$(rpad("nodes and weights",76))" begin
        for (nm, mk) in MESHES, p in 1:4
            b = PeriodicBSplineBasis(mk(16), p)
            q = SplineQuadrature(b)
            @test length(quadrature_nodes(q)) == 16 * q.nq
            @test length(quadrature_weights(q)) == length(quadrature_nodes(q))
            @test sum(quadrature_weights(q)) ≈ domainlength(b)      # weights sum to |Omega|
            @test all(>(0), quadrature_weights(q))
            @test issorted(quadrature_nodes(q))
            @test all(0 .≤ quadrature_nodes(q) .≤ domainlength(b))
        end
    end

    @testset "$(rpad("basis_values matches evaluate",76))" begin
        b = PeriodicBSplineBasis(RandomMesh(12, 2π), 3)
        q = SplineQuadrature(b)
        x = quadrature_nodes(q)
        for d in 0:3
            Φ = basis_values(q, d)
            @test size(Φ) == (nbasis(b), length(x))
            for j in (1, 5, 12), r in (1, 17, length(x))
                @test Φ[j, r] ≈ evaluate(b, j, x[r], d) atol = 1e-14
            end
        end
        @test_throws ArgumentError basis_values(q, 4)
    end

    @testset "$(rpad("mass matrix",76))" begin
        for (nm, mk) in MESHES, p in 1:4
            q = SplineQuadrature(PeriodicBSplineBasis(mk(16), p))
            M = mass_matrix(q)
            @test M ≈ transpose(M)
            @test isposdef(Matrix(M))
            @test sum(M) ≈ domainlength(basis(q))       # partition of unity twice over
            @test mass_factorization(q) \ (M * ones(16)) ≈ ones(16)
        end
    end

    @testset "$(rpad("basis integrals",76))" begin
        for (nm, mk) in MESHES, p in 1:4
            q = SplineQuadrature(PeriodicBSplineBasis(mk(16), p))
            Iv = basis_integrals(q)
            @test sum(Iv) ≈ domainlength(basis(q))
            # M 1 = int phi_i, because the basis is a partition of unity
            @test mass_matrix(q) * ones(16) ≈ Iv
            @test all(>(0), Iv)
        end
        # a constant of the discretisation, assembled once and returned by reference rather
        # than recomputed on every call
        q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16), 3))
        @test basis_integrals(q) === basis_integrals(q)
        basis_integrals(q)
        @test (@allocated basis_integrals(q)) == 0
    end

    @testset "$(rpad("S is already antisymmetric, given nq >= p",76))" begin
        # int d_x (phi_k phi_l) = 0 on a periodic domain: the basis is C^{p-1} across the
        # seam and there are no boundary terms, so S needs no skew-symmetrisation. This is
        # why the first discrete bracket carries a factor 1/2 that does NOT cancel against
        # a doubling: S - S^T is 2S, not S.
        #
        # The proviso is that the quadrature integrate d_x (phi_k phi_l), of degree 2p-1,
        # exactly -- that is nq >= p. Below it the identity fails outright, by 9e-3 at
        # p = 3, nq = 2.
        for (nm, mk) in MESHES, p in 1:4, nq in p:quadrature_order(p)+1
            q = SplineQuadrature(PeriodicBSplineBasis(mk(16), p); nq = max(nq, 2))
            S = derivative_matrix(q)
            @test maximum(abs, S + transpose(S)) < 1e-12
        end

        # On a UNIFORM mesh it holds at any nq, the assemblies being circulant. A test that
        # used only a uniform mesh would therefore confirm the identity for the wrong
        # reason and hide the requirement above.
        for p in 3:4, nq in 2:3
            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), p); nq = nq)
            @test maximum(abs, derivative_matrix(q) + transpose(derivative_matrix(q))) < 1e-12
        end
        for p in 3:4
            q = SplineQuadrature(PeriodicBSplineBasis(RandomMesh(16, 2π), p); nq = 2)
            @test maximum(abs, derivative_matrix(q) + transpose(derivative_matrix(q))) > 1e-4
        end
    end

    @testset "$(rpad("stiffness matrix",76))" begin
        for (nm, mk) in MESHES, p in 1:4
            q = SplineQuadrature(PeriodicBSplineBasis(mk(16), p))
            K = stiffness_matrix(q)
            @test K ≈ transpose(K)
            @test minimum(eigvals(Symmetric(Matrix(K)))) > -1e-12   # positive semi-definite
            @test maximum(abs, K * ones(16)) < 1e-12            # constants in the kernel
        end
    end

    @testset "$(rpad("integration by parts: -int phi_k phi_l''' = int phi_k' phi_l''",76))" begin
        # This holds for p >= 3. At p = 2 the third derivative vanishes identically inside
        # every cell, so the un-integrated form is zero while the integrated one is not --
        # the integration by parts is then the only correct reading, not a convenience.
        for (nm, mk) in MESHES, p in 3:4
            q = SplineQuadrature(PeriodicBSplineBasis(mk(16), p))
            @test maximum(abs, mixed_matrix(q, 1, 2) + mixed_matrix(q, 0, 3)) < 1e-10
        end
        for (nm, mk) in MESHES
            q = SplineQuadrature(PeriodicBSplineBasis(mk(16), 2))
            @test maximum(abs, mixed_matrix(q, 0, 3)) < 1e-12       # vanishes identically
            @test maximum(abs, mixed_matrix(q, 1, 2)) > 1          # this one does not
        end
    end

    @testset "$(rpad("weighted_matrix",76))" begin
        q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), 3))
        @test weighted_matrix(q, one, 0, 0) ≈ mass_matrix(q)
        @test weighted_matrix(q, one, 0, 1) ≈ derivative_matrix(q)
        # the vector form is the same as the function form
        @test weighted_matrix(q, sin, 0, 1) ≈ weighted_matrix(q, sin.(quadrature_nodes(q)), 0, 1)
        @test_throws DimensionMismatch weighted_matrix(q, [1.0, 2.0], 0, 1)
        # against a direct quadrature of int sin(x) phi_k phi_l
        Φ = basis_values(q, 0); w = quadrature_weights(q); x = quadrature_nodes(q)
        A = weighted_matrix(q, sin, 0, 0)
        @test A[3, 4] ≈ sum(w[r] * sin(x[r]) * Φ[3, r] * Φ[4, r] for r in eachindex(w))
    end

    @testset "$(rpad("exactness of the quadrature",76))" begin
        # the default rule integrates degree 3p-1 exactly, which is what a triple product
        # of basis functions with one derivative needs
        for p in 1:4
            q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), p))
            qq = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), p); nq = 20)
            Φ = basis_values(q, 0); Φ1 = basis_values(q, 1); w = quadrature_weights(q)
            T = [sum(w[r] * Φ[m, r] * Φ[k, r] * Φ1[l, r] for r in eachindex(w))
                 for m in 1:3, k in 1:3, l in 1:3]
            Ψ = basis_values(qq, 0); Ψ1 = basis_values(qq, 1); v = quadrature_weights(qq)
            Tex = [sum(v[r] * Ψ[m, r] * Ψ[k, r] * Ψ1[l, r] for r in eachindex(v))
                   for m in 1:3, k in 1:3, l in 1:3]
            @test maximum(abs, T - Tex) < 1e-13
        end
    end

    @testset "$(rpad("L2 projection",76))" begin
        q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(32, 2π), 3))
        b = basis(q)
        û = l2_projection(q, sin)
        @test maximum(abs, [evaluate(b, û, x) - sin(x) for x in range(0, 2π, length = 61)]) < 1e-5
        # the projection of a spline is that spline
        v̂ = randn(nbasis(b))
        @test l2_projection(q, evaluate(b, v̂, quadrature_nodes(q))) ≈ v̂
        # in-place agrees with out-of-place
        ŵ = similar(û)
        @test l2_projection!(ŵ, q, sin) ≈ û
        @test_throws DimensionMismatch l2_projection(q, [1.0, 2.0])
        @test_throws DimensionMismatch l2_projection!(zeros(3), q, sin)
        @test_throws DimensionMismatch l2_projection!(similar(û), q, [1.0, 2.0])

        # on a uniform mesh the whole projection is allocation-free: the f ⊙ w buffer lives
        # in the quadrature, the load vector is formed with mul! straight into û, and the
        # CirculantMass solve allocates nothing
        fq = sin.(quadrature_nodes(q))
        l2_projection!(ŵ, q, fq)
        @test (@allocated l2_projection!(ŵ, q, fq)) == 0
        @test ŵ ≈ û

        # The in-place and allocating methods are separate implementations -- the buffer is
        # the whole point of the split -- so they are checked against each other on every
        # mesh family, and not only on the uniform mesh the allocation test above needs.
        for (nm, mk) in MESHES, p in 1:4
            qm = SplineQuadrature(PeriodicBSplineBasis(mk(16), p))
            fm = sin.(quadrature_nodes(qm))
            ûm = Vector{Float64}(undef, nbasis(basis(qm)))
            @test l2_projection!(ûm, qm, fm) ≈ l2_projection(qm, fm)
        end

        # A sample wider than the quadrature's element type gets its own f ⊙ w product
        # instead of being narrowed into the shared buffer, which would be an InexactError
        # here and a silent loss of precision for a BigFloat. Graded rather than uniform:
        # the rfft plan of a CirculantMass takes a real argument only.
        qc = SplineQuadrature(PeriodicBSplineBasis(GradedMesh(16, 2π), 3))
        fc = cis.(quadrature_nodes(qc))
        ûc = Vector{ComplexF64}(undef, nbasis(basis(qc)))
        @test l2_projection!(ûc, qc, fc) ≈ l2_projection(qc, fc)
        @test eltype(l2_projection(qc, fc)) == ComplexF64
    end

    @testset "$(rpad("L2 projection converges at order p+1",76))" begin
        for p in 1:4
            errs = Float64[]; hs = Float64[]
            for n in (16, 32, 64)
                b = PeriodicBSplineBasis(GradedMesh(n, 2π), p)
                q = SplineQuadrature(b; nq = quadrature_order(p) + 3)
                û = l2_projection(q, sin)
                xs = range(0, 2π, length = 401)[1:400]
                push!(errs, maximum(abs, [evaluate(b, û, x) - sin(x) for x in xs]))
                push!(hs, meshwidth(b))
            end
            rate = log(errs[end-1] / errs[end]) / log(hs[end-1] / hs[end])
            @test rate > p + 0.8
        end
    end

    @testset "$(rpad("argument checks",76))" begin
        b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
        @test_throws ArgumentError SplineQuadrature(b; nq = 0)
        @test_throws ArgumentError SplineQuadrature(b; dmax = -1)
    end

end
