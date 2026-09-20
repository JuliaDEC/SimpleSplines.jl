# What a first column saves against a materialised matrix, for a circulant mass operator.
#
# The case is the one from issue #10: a periodic stiffness matrix shifted by the rank-one mean
# projector. `S + 𝟙𝟙ᵀ/n` is circulant like `S`, and its first column is `S[:,1] .+ 1/n` — but
# forming it turns an O(n) sparse assembly into an O(n²) dense one, and `CirculantMass` then
# keeps that matrix for the life of the operator while reading n numbers out of it.
#
# Two costs are separated here, because they behave differently and only one of them is
# visible to an allocation counter.
#
#   * Storage. The dense matrix is 8n² bytes against the column's 8n, and the operator holds
#     whichever it was given. `Base.summarysize` measures it.
#   * Construction time. The circulance check probes all n² positions of the matrix, an
#     allocation-free loop, so it costs time and no bytes. On a `Circulant` there is nothing
#     to check and the loop is gone. `@allocated` cannot see this difference — it reports the
#     column path as the *more* allocating of the two, because `Circulant` copies its column
#     while the matrix path only reads a matrix the caller already built.
#
# Timings need a cold process, so this is a script and not something to run in a warm session.
# Each figure is the best of several runs after one warm-up call.
#
# Run: julia --project=. --startup-file=no scripts/circulant_column_cost.jl

using LinearAlgebra
using Printf
using SimpleSplines
using SparseArrays

const P = 5
const LEVELS = [64, 128, 256, 512]
const REPEATS = 20

function best(f, n = REPEATS)
    f()                                  # warm-up: the first call compiles
    minimum(1:n) do _
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end
end

@printf("%6s | %12s %12s %8s | %11s %11s %8s\n",
    "n", "op, matrix", "op, column", "ratio", "build matrix", "build column", "ratio")

for n in LEVELS
    b = PeriodicBSplineBasis(UniformMesh(n, 2π), P)
    S = stiffness_matrix(SplineQuadrature(b))

    shifted = Matrix(S) .+ inv(n)
    column = Vector(S[:, 1]) .+ inv(n)

    opM = CirculantMass(shifted, n)
    opc = CirculantMass(column, n)

    # The solves must agree, or the rest of the line means nothing.
    x = randn(n)
    @assert maximum(abs, (opM \ x) - (opc \ x)) < 1e-10

    sM = Base.summarysize(opM)
    sc = Base.summarysize(opc)
    tM = best(() -> CirculantMass(shifted, n))
    tc = best(() -> CirculantMass(column, n))

    @printf("%6d | %10d B %10d B %7.1fx | %8.1f µs %8.1f µs %7.1fx\n",
        n, sM, sc, sM / sc, tM * 1e6, tc * 1e6, tM / tc)
end

# What an allocation counter reports, which is the opposite of the conclusion and is why the
# test suite asserts storage rather than allocation.
let n = 128
    b = PeriodicBSplineBasis(UniformMesh(n, 2π), P)
    S = stiffness_matrix(SplineQuadrature(b))
    shifted = Matrix(S) .+ inv(n)
    column = Vector(S[:, 1]) .+ inv(n)

    CirculantMass(shifted, n)
    CirculantMass(column, n)

    aM = minimum(_ -> @allocated(CirculantMass(shifted, n)), 1:REPEATS)
    ac = minimum(_ -> @allocated(CirculantMass(column, n)), 1:REPEATS)

    println()
    @printf("n = %d, construction allocations: matrix %d B, column %d B\n", n, aM, ac)
    println("The column path allocates more, because `Circulant` copies its column and the")
    println("matrix path only reads a matrix the caller had already built. The saving is in")
    println("what the operator then holds, and in the check it no longer runs.")
end
