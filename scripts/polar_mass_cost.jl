# What the pole costs: the mass factorisation and solve, against the Kronecker structure it
# gives up.
#
# A tensor-product mass matrix is never formed — a solve is one one-dimensional solve per axis,
# O(N Σ_d p_d) in time and O(Σ_d N_d²) in storage. The pole rows destroy that: a pole function
# is a sum over the whole angular axis, so `KroneckerMass` does not apply and the polar mass
# matrix is assembled and factorised.
#
# This measures that trade at the sizes a nonlinear flow on a mapped disk would use, and
# separates the two things that are easy to conflate: the one-off assembly and factorisation,
# paid once when the space is built, and the per-solve cost, paid at every Newton iteration.
#
# Timings need a cold process, so this is a script and not something to run in a warm session.
# Each figure is the best of several runs after one warm-up call, which is the usual guard
# against measuring compilation instead of the code.
#
# Run: julia --project=. --startup-file=no scripts/polar_mass_cost.jl

using LinearAlgebra
using Printf
using Random
using SimpleSplines
using SparseArrays

Random.seed!(20260917)

const P = 3
const LEVELS = [(16, 32), (32, 64), (64, 128)]
const REPEATS = 20

function best(f, n = REPEATS)
    f()                                  # warm-up: the first call compiles
    minimum(1:n) do _
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end
end

@printf("%10s %8s %8s | %11s %11s %11s | %11s %11s\n",
    "ns × nθ", "N polar", "N tens.", "assemble", "factorise", "nnz(M)",
    "solve polar", "solve kron")

for (ns, nθ) in LEVELS
    radial = BSplineBasis(UniformMesh(ns, 0 .. 1), P)
    angular = PeriodicBSplineBasis(UniformMesh(nθ, 0 .. 2π), P)

    B = PolarSplineBasis(radial, angular)
    T = radial ⊗ angular

    # The two are at the same mesh, so the polar space is 2Nθ − 3 functions smaller; that
    # difference is the pole, and it is what the comparison is about.
    qt = TensorProductQuadrature(T)
    q = PolarSplineQuadrature(B)

    Φ = basis_values(q, (0, 0))
    w = quadrature_weights(q)
    M = SparseMatrixCSC{Float64, Int}(Φ * Diagonal(w) * Φ')

    t_assemble = best(() -> SparseMatrixCSC{Float64, Int}(Φ * Diagonal(w) * Φ'), 5)
    t_factorise = best(() -> cholesky(Symmetric(M)), 5)

    x = randn(nbasis(B))
    y = similar(x)
    op = mass_operator(q)
    t_solve = best(() -> mass_solve!(y, op, x))

    X = randn(size(T)...)
    Y = similar(X)
    opt = mass_operator(qt)
    t_kron = best(() -> mass_solve!(Y, opt, X))

    @printf("%4d × %-4d %8d %8d | %8.2f ms %8.2f ms %11d | %8.3f ms %8.3f ms\n",
        ns, nθ, nbasis(B), nbasis(T), 1e3t_assemble, 1e3t_factorise, nnz(M),
        1e3t_solve, 1e3t_kron)
end

println()
println("""
assemble   = Φ₀ diag(w) Φ₀ᵀ, the sparse triple product, paid once per space
factorise  = the sparse Cholesky of it, paid once per space
solve      = one mass solve, the figure a Newton iteration pays
solve kron = the same on the tensor-product space at the same mesh, for scale""")
println()

## ---------------------------------------------------------------------------------------
## A homogeneous-Dirichlet rim, at the finest level
## ---------------------------------------------------------------------------------------

# A rim condition removes N_θ functions and leaves the pole rows alone, so it should cost a
# little less on every line and change nothing structural. Measured rather than argued,
# because a sparse factorisation's cost is not a function of the matrix size alone — and at
# one level only, since the point is the comparison and not a second scaling study.

let (ns, nθ) = LEVELS[end]
    radial = BSplineBasis(UniformMesh(ns, 0 .. 1), P)
    angular = PeriodicBSplineBasis(UniformMesh(nθ, 0 .. 2π), P)
    q = PolarSplineQuadrature(PolarSplineBasis(
        RecombinedBSplineBasis(radial, Free(), Dirichlet()), angular))

    Φ = basis_values(q, (0, 0))
    w = quadrature_weights(q)
    M = SparseMatrixCSC{Float64, Int}(Φ * Diagonal(w) * Φ')

    x = randn(nbasis(q))
    y = similar(x)
    op = mass_operator(q)

    @printf("%4d × %-4d %8d %8s | %8.2f ms %8.2f ms %11d | %8.3f ms %8s   (rim)\n",
        ns, nθ, nbasis(q), "—",
        1e3best(() -> SparseMatrixCSC{Float64, Int}(Φ * Diagonal(w) * Φ'), 5),
        1e3best(() -> cholesky(Symmetric(M)), 5), nnz(M),
        1e3best(() -> mass_solve!(y, op, x)), "—")
end
