# The approximation order of the polar spline space is not reduced at the pole.
#
# The claim to reproduce is Zoni & Güçlü's, that a polar spline discretisation converges at
# high order "uniformly across the computational domain, without effects of order reduction
# due to the singularity". Two things have to be shown, and the second is the one a norm alone
# cannot say:
#
#   1. the L² error of the projection of a smooth function converges at the rate the degree
#      implies, p+1;
#   2. the error is not concentrated in the pole cells — the per-cell error density there is
#      comparable to the rest of the domain, rather than the global rate being held up by a
#      large domain around a bad patch.
#
# The target is a fixed smooth function of the *true* Cartesian coordinates of the disk,
# f(s cos θ, s sin θ), not of the pseudo-Cartesian chart: the chart moves with the angular
# mesh, so a target defined through it would change under refinement and the rate would be
# measured against a moving object.
#
# The projection is the parameter-measure one, `l2_projection`, and the error is reported in
# both measures. The parameter measure ds dθ is the more demanding of the two here, because it
# gives the pole neighbourhood a weight its physical area s ds dθ does not.
#
# Run: julia --project=. --startup-file=no scripts/polar_approximation_order.jl

using LinearAlgebra
using Printf
using SimpleSplines

const P = 3                              # degree on both axes
const LEVELS = [(8, 16), (16, 32), (32, 64), (64, 128)]

# Smooth on the closed disk and not even nearly a polynomial, so the rate measured is the
# basis's and not the target's.
f(x, y) = exp(x / 2) * cos(2y) + sin(x * y)

target(s, θ) = f(s * cos(θ), s * sin(θ))

function level(ns, nθ)
    B = PolarSplineBasis(BSplineBasis(UniformMesh(ns, 0 .. 1), P),
        PeriodicBSplineBasis(UniformMesh(nθ, 0 .. 2π), P))
    q = PolarSplineQuadrature(B)

    sq, θq = quadrature_nodes(q)
    w = quadrature_weights(q)
    grid = [(s, θ) for s in sq, θ in θq]

    F = [target(x...) for x in grid][:]
    û = l2_projection(q, F)
    r = basis_values(q, (0, 0))' * û .- F

    # The Jacobian of the polar chart, s, is the physical area element of the unit disk. The
    # parameter measure is the bare weight.
    jac = [x[1] for x in grid][:]

    err_param = sqrt(abs(sum(r .^ 2 .* w)))
    err_disk = sqrt(abs(sum(r .^ 2 .* w .* jac)))

    # The per-radial-cell error density: the RMS of the residual over each cell's own
    # quadrature points, which is an error per unit parameter area and so is comparable
    # between cells of the same size. Every cell here has the same size, so the numbers are
    # directly comparable and a pole effect would be visible as the first one or two standing
    # out from the rest.
    nq = length(sq) ÷ ns
    cellrms = [begin
                   rows = ((c - 1) * nq + 1):(c * nq)
                   block = reshape(r, length(sq), length(θq))[rows, :]
                   bw = reshape(w, length(sq), length(θq))[rows, :]
                   sqrt(abs(sum(block .^ 2 .* bw) / sum(bw)))
               end
               for c in 1:ns]

    return (nbasis = nbasis(B), err_param = err_param, err_disk = err_disk,
        cellrms = cellrms)
end

results = [level(ns, nθ) for (ns, nθ) in LEVELS]

println("degree ", P, " on both axes, L² projection of a smooth function of the disk")
println()
@printf("%10s %8s  %12s %7s  %12s %7s\n",
    "ns × nθ", "N", "‖e‖_dsdθ", "order", "‖e‖_sdsdθ", "order")
for (i, ((ns, nθ), r)) in enumerate(zip(LEVELS, results))
    if i == 1
        @printf("%4d × %-4d %8d  %12.4e %7s  %12.4e %7s\n",
            ns, nθ, r.nbasis, r.err_param, "—", r.err_disk, "—")
    else
        op = log2(results[i - 1].err_param / r.err_param)
        od = log2(results[i - 1].err_disk / r.err_disk)
        @printf("%4d × %-4d %8d  %12.4e %7.3f  %12.4e %7.3f\n",
            ns, nθ, r.nbasis, r.err_param, op, r.err_disk, od)
    end
end
println()
println("expected order ", P + 1)
println()

## ---------------------------------------------------------------------------------------
## Where the error sits
## ---------------------------------------------------------------------------------------

println("per-radial-cell RMS residual, innermost cells first (the pole is cell 1)")
println()
for ((ns, nθ), r) in zip(LEVELS, results)
    head = r.cellrms[1:min(6, ns)]
    bulk = maximum(r.cellrms[3:end])
    @printf("%4d × %-4d  cells 1..%d: ", ns, nθ, length(head))
    for v in head
        @printf("%9.2e ", v)
    end
    @printf("   max over cells 3..%d: %9.2e   ratio pole/bulk: %5.2f\n",
        ns, bulk, maximum(r.cellrms[1:2]) / bulk)
end
println()

## ---------------------------------------------------------------------------------------

orders_param = [log2(results[i - 1].err_param / results[i].err_param)
                for i in 2:length(results)]
orders_disk = [log2(results[i - 1].err_disk / results[i].err_disk)
               for i in 2:length(results)]

# The rate is read off the finest pair, where the asymptotic regime is reached; the coarser
# ones are printed so that a pre-asymptotic level cannot be mistaken for the answer.
rate_ok = orders_param[end] > P + 0.7 && orders_disk[end] > P + 0.7

# The pole cells are not a bad patch: their error density is within a small factor of the
# worst cell anywhere else. A factor, not equality — the residual of a projection is not
# uniform across cells for any basis, and the first cells carry the largest curvature of the
# polar chart.
pole_ratios = [maximum(r.cellrms[1:2]) / maximum(r.cellrms[3:end]) for r in results]
pole_ok = all(<(3), pole_ratios)

println("order at the finest pair:  ds dθ ", round(orders_param[end]; digits = 3),
    "   s ds dθ ", round(orders_disk[end]; digits = 3), "   (expected ", P + 1, ")")
println("pole/bulk error ratio:     ", round.(pole_ratios; digits = 3))
println()
println(rate_ok && pole_ok ? "ALL CHECKS PASS" : "SOME CHECK FAILED")
exit(rate_ok && pole_ok ? 0 : 1)
