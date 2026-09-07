# How the cost of `evaluate` and of `evaluate_all` grows with the degree, and therefore which
# complexity their docstrings should state.
#
# The docstrings used to give the single-function path as `O(p^2)`. That is wrong. `_bspline`
# is the Cox-de Boor recursion written out as it stands, with no memoisation, so it splits
# into two subproblems at every level and the cost of one value is `O(2^p)`. `evaluate_all!`
# runs de Boor's triangular scheme instead, which is `O(p^2)` for all `p+1` nonzero functions
# *together* -- so the block form is not merely `p+1` times cheaper, it has a different
# exponent. At the cubics almost everything here uses the difference hardly shows, which is
# why the wrong figure survived.
#
# This script is what the figures quoted in `evaluate`'s and `_bspline`'s docstrings, and in
# the cost table of `docs/src/theory/bsplines.md`, are read off. Absolute timings are a
# property of the machine, so the assertion at the end is on the *shape*: doubling the degree
# has to cost the single-function path far more than it costs the block.
#
# Run: julia --project=. scripts/evaluate_cost_scaling.jl

using SimpleSplines
using Printf

const DEGREES = (3, 6, 8, 12)
const CALLS = 20_000
const REPEATS = 5
const NCELLS = 40
const X = 0.317                         # interior, and not a breakpoint of the mesh below

# The minimum over `REPEATS` runs, which is the estimator least polluted by whatever else the
# machine is doing. The value is accumulated so that the loop cannot be optimised away, and a
# zero total means the chosen function does not reach `X` and the timing is of nothing.
function nanoseconds_per_call(f)
    f()                                 # compile
    best = Inf
    for _ in 1:REPEATS
        acc = 0.0
        t = @elapsed for _ in 1:CALLS
            acc += f()
        end
        iszero(acc) && error("the accumulated value is zero; nothing was measured")
        best = min(best, t)
    end
    return best / CALLS * 1e9
end

function measure(p)
    b = BSplineBasis(UniformMesh(NCELLS, 0 .. 1), p)
    # Any function nonzero at X. On a clamped basis those on cell c are c .. c+p.
    j = first(local_indices(b, findcell(b, X)))
    one_function = nanoseconds_per_call(() -> evaluate(b, j, X))
    # `evaluate_all`, not `evaluate_all!`: it is the form the docstrings compare against, and
    # its freshly allocated buffer is part of what a caller pays.
    whole_block = nanoseconds_per_call(() -> sum(last(evaluate_all(b, X))))
    return (; p, one_function, whole_block)
end

rows = [measure(p) for p in DEGREES]

@printf("%3s  %14s  %14s  %8s\n", "p", "evaluate [ns]", "block [ns]", "ratio")
for r in rows
    @printf("%3d  %14.1f  %14.1f  %8.1f\n",
        r.p, r.one_function, r.whole_block, r.one_function / r.whole_block)
end

# The claim under test. Between p = 6 and p = 12 the degree doubles, so an `O(2^p)` path pays
# a factor of about 2^6 = 64 while an `O(p^2)` one pays about 4. The bounds are loose enough
# to survive a noisy machine and tight enough that the two cannot be confused: an `O(p^2)`
# single-function path would come nowhere near 20x.
lo = only(filter(r -> r.p == 6, rows))
hi = only(filter(r -> r.p == 12, rows))

growth_one = hi.one_function / lo.one_function
growth_block = hi.whole_block / lo.whole_block

@printf("\np = 6 -> 12:  evaluate x%.1f   block x%.1f\n", growth_one, growth_block)

if growth_one < 20
    error("`evaluate` grew only $(round(growth_one; digits = 1))x from p = 6 to p = 12; " *
          "the O(2^p) the docstrings claim would grow about 64x")
end
if growth_block > 8
    error("`evaluate_all` grew $(round(growth_block; digits = 1))x from p = 6 to p = 12; " *
          "the O(p^2) the docstrings claim would grow about 4x")
end

println("\nOK: the single-function path grows exponentially in the degree and the block " *
        "path polynomially, as the docstrings state.")
