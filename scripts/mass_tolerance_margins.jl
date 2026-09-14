# What margin each of the three tolerances in `src/mass.jl` actually has, and therefore
# whether the form they take is the right one.
#
# All three answer a structural question about an assembled matrix -- is this matrix
# circulant, are the constants in its kernel, is the minor that drops one degree of freedom
# invertible -- and the answer has to survive two things a written-down figure does not
# survive: a rescaling of the assembly, and a change of element type. An absolute bound gets
# both wrong. It calls an invertible matrix singular once the matrix is scaled below the
# bound, and it calls every `Float32` assembly non-circulant because `Float32` rounding alone
# exceeds a figure chosen for `Float64`.
#
# So each bound is relative and carries `eps(T)`. This script measures the two sides of each
# one -- how close the accepted case comes to the bound, and how far the refused case sits
# above it -- so that the form is chosen from evidence. The figures quoted in the comments in
# `src/mass.jl` come from here.
#
# Run: julia --project=. scripts/mass_tolerance_margins.jl

using SimpleSplines
using LinearAlgebra
using Printf
using SparseArrays

const MESHES = (UniformMesh, GradedMesh, RandomMesh)

# ---------------------------------------------------------------------------------------
# `_check_constant_kernel`: ‖M𝟙‖∞ against `n * eps(T) * ‖M‖∞`
#
# A stiffness matrix on a periodic basis has M𝟙 = 0 exactly in exact arithmetic -- the
# integrand ∑ᵢ Bᵢ'(x) Bⱼ'(x) vanishes pointwise, since ∑ᵢ Bᵢ ≡ 1 -- so its residual is pure
# rounding, and the question is only whether that rounding stays below the bound as n grows.
function constant_kernel_margins(::Type{T}) where {T}
    worst_singular, at_singular = zero(T), ""
    best_invertible, at_invertible = T(Inf), ""

    for meshtype in MESHES, p in 1:8, n in (2p + 2, 16, 33, 64, 256, 512, 1024)
        n ≤ 2p + 1 && continue
        q = SplineQuadrature(PeriodicBSplineBasis(meshtype(n, 2π), p))
        for (A, singular) in ((T.(stiffness_matrix(q)), true), (T.(mass_matrix(q)), false))
            ratio = norm(A * ones(T, n), Inf) / (n * eps(T) * norm(A, Inf))
            if singular
                ratio > worst_singular && ((worst_singular, at_singular) = (
                    ratio, "$(meshtype) p=$p n=$n"))
            else
                ratio < best_invertible && ((best_invertible, at_invertible) = (
                    ratio, "$(meshtype) p=$p n=$n"))
            end
        end
    end
    return (worst_singular, at_singular, best_invertible, at_invertible)
end

# ---------------------------------------------------------------------------------------
# `_check_circulant`: max|M[i,j] - c[mod1(i-j+1,n)]| against `rtol * max|c|`
#
# The accepted side is a uniform periodic mesh, whose residual is assembly rounding. The
# refused side is a graded mesh, which misses by a fraction of the entries themselves.
function circulance_margins(::Type{T}) where {T}
    worst_circulant, at_circulant = zero(T), ""
    best_refused, at_refused = T(Inf), ""

    for p in 1:8, n in (max(12, 2p + 2), 16, 33, 64, 256)

        n ≤ 2p + 1 && continue
        for (meshtype, circulant) in ((UniformMesh, true), (GradedMesh, false))
            M = T.(Matrix(mass_matrix(SplineQuadrature(
                PeriodicBSplineBasis(meshtype(n, 2π), p)))))
            c = M[:, 1]
            r = maximum(abs, [M[i, j] - c[mod1(i - j + 1, n)] for i in 1:n, j in 1:n]) /
                maximum(abs, c)
            if circulant
                r > worst_circulant && ((worst_circulant, at_circulant) = (r, "p=$p n=$n"))
            else
                r < best_refused && ((best_refused, at_refused) = (r, "p=$p n=$n"))
            end
        end
    end
    return (worst_circulant, at_circulant, best_refused, at_refused)
end

# ---------------------------------------------------------------------------------------
# `_check_definite_minor`: the pivot ratio of the Cholesky factor of the minor
#
# CHOLMOD reports success for a positive *semi*definite factorisation, so `issuccess` does not
# separate a minor that is invertible from one that is not. The pivots do. `diag` of an LLᵀ
# factor is the diagonal of L, so the pivot ratio is the square of the diagonal ratio; the
# check compares the diagonal ratio against `sqrt(n * eps(T))`.
#
# The refused case is the projector onto the complement of span{𝟙, v} with v = (1,-1,1,-1):
# it has the constants in its kernel, so it passes `_check_constant_kernel`, and v there as
# well, so deflating the constants alone leaves a singular problem behind.
function minor_pivot_margins()
    worst_genuine, at_genuine = Inf, ""

    for meshtype in MESHES, p in 1:4, n in (16, 33, 64, 256)
        S = stiffness_matrix(SplineQuadrature(PeriodicBSplineBasis(meshtype(n, 2π), p)))
        d = diag(cholesky(Symmetric(S[2:end, 2:end]); check = false))
        lo, hi = extrema(abs, d)
        ratio = (lo / hi)^2
        ratio < worst_genuine && ((worst_genuine, at_genuine) = (
            ratio, "$(meshtype) p=$p n=$n"))
    end

    v = [1.0, -1.0, 1.0, -1.0]
    P = Matrix(I, 4, 4) .- ones(4, 4) ./ 4 .- (v * v') ./ 4
    F = cholesky(Symmetric(sparse(P[2:end, 2:end])); check = false)
    lo, hi = extrema(abs, diag(F))

    return (worst_genuine, at_genuine, (lo / hi)^2, issuccess(F))
end

function main()
    for T in (Float64, Float32)
        ws, as, bi, ai = constant_kernel_margins(T)
        println("\n", T, " -- the constants in the kernel, ‖M𝟙‖∞ / (n·eps·‖M‖∞)")
        @printf("  accepted (stiffness) worst  %10.3e   at %s\n", ws, as)
        @printf("  refused  (mass)      best   %10.3e   at %s\n", bi, ai)
        @printf("  the gate is at 10; margins are %.1f× below and %.1e× above\n",
            10 / ws, bi / 10)

        wc, ac, br, ar = circulance_margins(T)
        println("\n", T, " -- circulance, max deviation relative to max|c|")
        @printf("  accepted (uniform)   worst  %10.3e = %7.1f·eps   at %s\n",
            wc, wc / eps(T), ac)
        @printf("  refused  (graded)    best   %10.3e                at %s\n", br, ar)
        @printf("  the default rtol is sqrt(eps) = %.3e\n", sqrt(eps(T)))
    end

    wg, ag, bad, success = minor_pivot_margins()
    println("\nthe minor's pivot ratio, min/max")
    @printf("  genuine assembly     worst  %10.3e   at %s\n", wg, ag)
    @printf("  kernel span{𝟙, v}           %10.3e   (issuccess reports %s)\n", bad, success)
    @printf("  the bound is n·eps, i.e. %.3e at n = 3\n", 3 * eps())

    return nothing
end

main()
