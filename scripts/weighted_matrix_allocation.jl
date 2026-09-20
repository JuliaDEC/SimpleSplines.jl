# Where the allocation of `weighted_matrix` goes, and whether it is worth removing.
#
# `weighted_matrix` is the one assembly that cannot be memoised — it depends on the field, so a
# time integrator asks for a different one inside every Newton iteration of every step. That
# call pattern is what makes a per-call allocation worth measuring.
#
# The figure is easy to attribute to the wrong thing. `Φₐ diag(f ⊙ w) Φᵦᵀ` builds a sparse
# intermediate and then a sparse triple product, and the returned matrix is a small part of
# what the two of them allocate. This separates the three, and prints the result matrix's own
# size beside them, so the attribution is read off rather than assumed.
#
# It also prints the per-call time against one mass solve, because the allocation only matters
# if the call is on the critical path at all.
#
# Timings need a cold process, so this is a script and not something to run in a warm session.
# Each figure is the best of several runs after one warm-up call.
#
# Run: julia --project=. --startup-file=no scripts/weighted_matrix_allocation.jl

using LinearAlgebra
using Printf
using SimpleSplines
using SparseArrays

const P = 3
const LEVELS = [128, 256, 512]
const REPEATS = 20

# The three arrays a `SparseMatrixCSC` owns. `Base.summarysize` walks the type and overcounts
# here; these are the bytes the matrix is.
storage(A::SparseMatrixCSC) = sizeof(A.colptr) + sizeof(A.rowval) + sizeof(A.nzval)

function best(f, n = REPEATS)
    f()                                  # warm-up: the first call compiles
    minimum(1:n) do _
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end
end

@printf("%6s %7s | %10s %10s %10s %10s %10s | %9s %9s\n",
    "N", "nbasis", "f ⊙ w", "Φₐ D", "product", "result", "TOTAL", "call", "mass \\")

for N in LEVELS
    q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(N, 2π), P))
    x = quadrature_nodes(q)
    w = quadrature_weights(q)
    f = sin.(x)

    Φₐ = basis_values(q, 0)
    Φᵦ = basis_values(q, 1)
    D = Diagonal(f .* w)
    L = Φₐ * D
    A = weighted_matrix(q, f, 0, 1)

    weighted_matrix(q, f, 0, 1)          # warm-up before the allocation counts

    # A single `@allocated` reading is not always in family: one component can come out above
    # the total it is part of. The smallest of several readings is the figure.
    total = minimum(_ -> @allocated(weighted_matrix(q, f, 0, 1)), 1:REPEATS)
    temp = minimum(_ -> @allocated(f .* w), 1:REPEATS)
    left = minimum(_ -> @allocated(Φₐ * D), 1:REPEATS)
    product = minimum(_ -> @allocated(L * Φᵦ'), 1:REPEATS)

    rhs = ones(nbasis(q.basis))

    @printf("%6d %7d | %10d %10d %10d %10d %10d | %7.1f µs %7.1f µs\n",
        N, nbasis(q.basis), temp, left, product, storage(A), total,
        best(() -> weighted_matrix(q, f, 0, 1)) * 1e6,
        best(() -> mass_matrix(q) \ rhs) * 1e6)
end

# The sparsity pattern does not depend on the field, which is the premise of the cached-pattern
# fix. Checked rather than asserted.
let q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(128, 2π), P))
    x = quadrature_nodes(q)
    A = weighted_matrix(q, sin.(x), 0, 1)
    B = weighted_matrix(q, exp.(x), 0, 1)
    C = weighted_matrix(q, ones(length(x)), 0, 1)
    println()
    println("pattern independent of f: ",
        A.colptr == B.colptr == C.colptr && A.rowval == B.rowval == C.rowval,
        "  (nnz = ", nnz(A), ")")
end
