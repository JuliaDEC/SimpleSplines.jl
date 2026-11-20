# How far the argument reduction of a periodic basis moves from `a + mod(x - a, L)`, and what
# it costs `evaluate_all!`. The figures in the changelog entry on the reduction are read off
# this script.
#
# The bound compares the package's `_reduce_argument` with `a + mod(x - a, L)`, the reduction
# that it replaced, over 10^6 points with |x| up to about 630 on a domain of length 2π, for six
# values of `a`. Pairs on opposite sides of a seam differ by about `L` and are counted apart.
#
# The timing is of the package as loaded, so the comparison with the old reduction runs this
# same file against a checkout of the base commit, each in its own process:
#
#   julia --startup-file=no --project=. scripts/periodic_reduction.jl
#   julia --startup-file=no --project=<checkout of the base> scripts/periodic_reduction.jl
#
# Each figure is the median over REPEATS passes of 10^6 points, in ns per point.

using SimpleSplines
using Printf
using Random
using Statistics

const NPOINTS = 10^6
const REPEATS = 7

function reduction_bound()
    rng = Xoshiro(20261120)
    worst, opposite = 0.0, 0
    for a in (0.0, -0.3, 0.1, 1.0, -2.0, 3.7)
        b = PeriodicBSplineBasis(UniformMesh(8, (a, a + 2π)), 3)
        L = domainlength(b)
        for _ in 1:NPOINTS
            x = a + (2rand(rng) - 1) * 100L
            δ = abs(SimpleSplines._reduce_argument(b, x) - (a + mod(x - a, L)))
            δ > L / 2 ? (opposite += 1) : (worst = max(worst, δ))
        end
    end
    return worst, opposite
end

function sweep(b, xs)
    v = Vector{Float64}(undef, degree(b) + 1)
    acc = 0.0
    for x in xs
        evaluate_all!(v, b, x, 0)
        acc += v[1]
    end
    iszero(acc) && error("the accumulated value is zero; nothing was measured")
    return acc
end

function timings()
    rng = Xoshiro(1)
    rows = String[]
    for p in (1, 3)
        b = PeriodicBSplineBasis(UniformMesh(64, (0.0, 2π)), p)
        L = domainlength(b)
        inside = rand(rng, NPOINTS) .* L
        outside = (rand(rng, NPOINTS) .* 8 .- 3.5) .* L     # seven in eight outside [0, L)
        for (name, xs) in (("inside", inside), ("7/8 outside", outside))
            sweep(b, xs)                                     # compile
            t = [(@elapsed sweep(b, xs)) / NPOINTS * 1e9 for _ in 1:REPEATS]
            push!(rows,
                @sprintf("degree %d, %-11s  %6.2f ns/point (min %6.2f)", p, name,
                    median(t), minimum(t)))
        end
    end
    return rows
end

worst, opposite = reduction_bound()
report = [
    "SimpleSplines at $(pkgdir(SimpleSplines)), Julia $(VERSION), threads $(Threads.nthreads())",
    "load average $(join(round.(Sys.loadavg(); digits = 2), " "))",
    @sprintf("largest difference from a + mod(x - a, L): %.3g; pairs a period apart: %d of %d",
        worst, opposite, 6NPOINTS),
    timings()...
]
println(join(report, "\n"))
