# How closely `evaluate(b, û, x, d)` agrees with a sum over the whole basis, and therefore
# what kind of tolerance the test in `test/basis_tests.jl` may use.
#
# `evaluate(b, û, x, d)` sums the local block of `local_width(b)` functions, obtained from
# de Boor's triangular scheme plus derivative lifting. The reference sums `û[j] * evaluate(b,
# j, x, d)` over the whole basis, each term coming from the Cox-de Boor recursion written out
# as it stands. The two are mathematically identical and cannot agree to the last bit: they
# take different recursions to each value and then sum a different number of terms.
#
# The question the test needs answered is which *kind* of tolerance is right. A fixed `atol`
# is not, because the magnitude of the sum is set by `randn` and by the length of the sum;
# the first version of that test used `atol = 1e-12` and failed on deviations of 1.9e-12 at
# a value of 12, i.e. a relative error of 1.6e-13 -- a handful of ULP. This script measures
# the distribution so that the relative bound is chosen from evidence rather than from
# whatever made the failure go away.
#
# Run: julia --project=. scripts/local_evaluation_agreement.jl

using SimpleSplines
using Printf
using Random

const DRAWS = 400
const RTOL = 1e-9                       # the bound the test uses
const POINTS = (0.0, 0.019, 0.25, 0.5, 0.5 + eps(), 0.937, 1.0)

function sweep(; draws = DRAWS, seed = 0x2f7a91c4)
    rng = Xoshiro(seed)

    worst_rel = 0.0
    worst_abs = 0.0
    worst_at = ()
    ncomparisons = 0

    for p in 0:4, bc in (Free(), Dirichlet(), Neumann(), Periodic())

        # a condition involving D^k with k > p constrains nothing and is rejected
        p < constraint_order(bc) && continue
        b = BSplineBasis(UniformMesh(16, 0 .. 1), p, bc)

        for _ in 1:draws
            û = randn(rng, nbasis(b))
            for x in POINTS, d in 0:min(p, 2)

                local_value = evaluate(b, û, x, d)
                reference = sum(û[j] * evaluate(b, j, x, d) for j in eachindex(û))

                absdev = abs(local_value - reference)
                scale = max(abs(local_value), abs(reference))
                reldev = scale > 0 ? absdev / scale : 0.0

                ncomparisons += 1
                absdev > worst_abs && (worst_abs = absdev)
                if reldev > worst_rel
                    worst_rel = reldev
                    worst_at = (p, bc, x, d, scale, absdev)
                end
            end
        end
    end

    return (; worst_rel, worst_abs, worst_at, ncomparisons)
end

r = sweep()
p, bc, x, d, scale, absdev = r.worst_at

@printf("comparisons      = %d\n", r.ncomparisons)
@printf("max rel deviation = %.3e   (p=%d bc=%s x=%g d=%d |value|=%.6g absdev=%.3e)\n",
    r.worst_rel, p, bc, x, d, scale, absdev)
@printf("max abs deviation = %.3e\n", r.worst_abs)
@printf("headroom vs rtol=%.0e : %.0fx\n", RTOL, RTOL / r.worst_rel)

# The bound the test relies on. A genuine indexing fault in the local-block path moves the
# result by O(1), so this margin is what separates rounding from a defect.
if r.worst_rel > RTOL
    error("the relative deviation $(r.worst_rel) exceeds the rtol = $(RTOL) the test uses")
end
println("\nOK: every comparison is inside the rtol the test asserts.")
