# The polar basis is a partition of unity across the pole rows, and its mass matrix integrates
# correctly on a mapped domain.
#
# Two claims, and the second is the one that matters for an assembly on a mapped disk:
#
#   1. Σ_k Ψ_k ≡ 1 everywhere, the pole triangle included. This is not automatic — it is what
#      the choice of pole-triangle vertex radius buys, and a triangle of any other size spans
#      the same space while breaking it.
#
#   2. The mass matrix reproduces the area: 𝟙ᵀ 𝕄 𝟙 = ∫ 1. On the parameter square that is
#      2π; on a mapped domain it is ∫ J ds dθ with J the Jacobian of the map, assembled
#      through `weighted_matrix`. The map used here is the one of the Grad-Shafranov case
#      C2 — `eq:mapping`, from `Experiments/MetriplecticRelaxation/src/takeda.jl` — whose area
#      is known independently from that experiment's own P₁ triangulation, so the number has
#      something to disagree with.
#
# Run: julia --project=. --startup-file=no scripts/polar_partition_of_unity.jl

using LinearAlgebra
using Printf
using SimpleSplines

B = PolarSplineBasis(BSplineBasis(UniformMesh(24, 0 .. 1), 3),
    PeriodicBSplineBasis(UniformMesh(48, 0 .. 2π), 3))
q = PolarSplineQuadrature(B)

println("basis     ", B)
println("functions ", nbasis(B))
println()

## ---------------------------------------------------------------------------------------
## Σ_k Ψ_k ≡ 1
## ---------------------------------------------------------------------------------------

# Points chosen to sit on the pole, inside the two cells the pole triangle is supported on,
# across the seam of that support, and well away from it.
pts = vcat(
    [(0.0, θ) for θ in range(0, 2π; length = 33)[1:32]],
    [(s, θ)
     for s in [1e-8, 0.01, 1 / 24, 0.06, 2 / 24, 0.1, 0.5, 0.97, 1.0],
    θ in range(0, 2π; length = 17)[1:16]][:]
)

# Through `evaluate_all`, which is the path an assembly or a deposition takes, and through the
# sum over every index, which is the definition. The two must agree, or `evaluate_all` is
# dropping a function that is nonzero at the point — the failure mode an offset-and-block
# interface has at a non-product index set.
pu_all = maximum(x -> abs(sum(evaluate_all(B, x)[2]) - 1), pts)
pu_def = maximum(x -> abs(sum(evaluate(B, k, x) for k in eachindex(B)) - 1), pts)

# The pole triangle on its own reproduces the first two rows' own sum, which is what makes the
# whole basis add to one.
radial, angular = bases(B)
function tworow_sum(x)
    sum(evaluate(radial, i, x[1]) for i in 1:2) *
    sum(evaluate(angular, j, x[2]) for j in 1:nbasis(angular))
end
pole_sum = maximum(x -> abs(sum(evaluate(B, k, x) for k in 1:3) - tworow_sum(x)), pts)

println("partition of unity")
@printf("  max |Σ_k Ψ_k − 1|            over %3d points, via evaluate_all = %.3e\n",
    length(pts), pu_all)
@printf("  max |Σ_k Ψ_k − 1|            over %3d points, via evaluate    = %.3e\n",
    length(pts), pu_def)
@printf("  max |Σ_{k≤3} Ψ_k − Σ(N₁+N₂)Mⱼ|                              = %.3e\n", pole_sum)
pu_pass = max(pu_all, pu_def, pole_sum) < 1e-13
println("  passes: ", pu_pass)
println()

# Non-negativity, which the vertex radius also buys and which a larger triangle would keep but
# a smaller one would not.
λmin = minimum(k -> minimum(x -> evaluate(B, k, x), pts), 1:3)
@printf("  least value of a pole function over those points = %.3e (must be ≥ 0)\n", λmin)
nonneg_pass = λmin ≥ -1e-15
println("  passes: ", nonneg_pass)
println()

## ---------------------------------------------------------------------------------------
## The constant, and the area of the parameter square
## ---------------------------------------------------------------------------------------

M = mass_matrix(q)
𝟙 = ones(nbasis(B))

area_param = 𝟙' * M * 𝟙
integrals = basis_integrals(q)

println("the parameter square")
@printf("  𝟙ᵀ 𝕄 𝟙          = %.12f     exact 2π = %.12f   error %.3e\n",
    area_param, 2π, abs(area_param - 2π))
@printf("  Σ ∫Ψ_k           = %.12f                        error %.3e\n",
    sum(integrals), abs(sum(integrals) - 2π))
@printf("  max |𝕄𝟙 − ∫Ψ|    = %.3e   (the partition of unity, seen in the assembly)\n",
    maximum(abs, M * 𝟙 .- integrals))
param_pass = abs(area_param - 2π) < 1e-11 &&
             maximum(abs, M * 𝟙 .- integrals) < 1e-13
println("  passes: ", param_pass)
println()

## ---------------------------------------------------------------------------------------
## The mapped domain
## ---------------------------------------------------------------------------------------

# `eq:mapping`, the map of the Grad-Shafranov case C2. Written as a function of the Cartesian
# pseudo-coordinates u = s cos θ, v = s sin θ, because r depends on u alone and that is what
# makes the Jacobian short enough to be checked by eye:
#
#   q(u) = √(1 + ε(ε + 2u)),   r = a[b + (1−q)/ε],   z = c e ξ v /(2 − q) ,
#
#   ∂r/∂u = −a/q ,  ∂r/∂v = 0 ,  ∂z/∂v = c e ξ/(2 − q) ,
#
# so det ∂(r,z)/∂(u,v) = −a c e ξ / (q (2 − q)), and det ∂(u,v)/∂(s,θ) = s, giving
#
#   J(s,θ) = a c e ξ s / (q (2 − q)) .
#
# The sign is dropped: the measure is |J|, and the map reverses orientation.
const MAPC = (e = 1.4, ε = 0.3, a = 4.0, b = 3.0, c = 6.3, ξ = 1 / sqrt(1 - 0.3^2 / 4))

function disk_map(s, θ)
    (; e, ε, a, b, c, ξ) = MAPC
    qq = sqrt(1 + ε * (ε + 2s * cos(θ)))
    return (a * (b + (1 - qq) / ε), c * e * ξ * s * sin(θ) / (2 - qq))
end

function disk_jacobian(s, θ)
    (; e, ε, a, b, c, ξ) = MAPC
    qq = sqrt(1 + ε * (ε + 2s * cos(θ)))
    return a * c * e * ξ * s / (qq * (2 - qq))
end

# The analytic Jacobian against a central difference of the map itself, so that a slip in the
# derivation above cannot pass as agreement with the reference area.
function fd_jacobian(s, θ; h = 1e-6)
    r₊, z₊ = disk_map(s + h, θ)
    r₋, z₋ = disk_map(s - h, θ)
    rθ₊, zθ₊ = disk_map(s, θ + h)
    rθ₋, zθ₋ = disk_map(s, θ - h)
    abs(((r₊ - r₋) * (zθ₊ - zθ₋) - (rθ₊ - rθ₋) * (z₊ - z₋)) / (4h^2))
end

fd_err = maximum([abs(disk_jacobian(s, θ) - fd_jacobian(s, θ)) /
                  max(1, disk_jacobian(s, θ))
                  for s in 0.1:0.1:0.9, θ in range(0, 2π; length = 13)])

# `weighted_matrix` samples the coefficient on the flattened grid, which is exactly the shape
# the assembly of a mapped domain needs; the same call carries the metric of a bracket.
MJ = weighted_matrix(q, x -> disk_jacobian(x...), (0, 0), (0, 0))
area_mapped = 𝟙' * MJ * 𝟙

# The independent value: the area quoted in `Experiments/MetriplecticRelaxation/src/takeda.jl`
# for the image of `eq:mapping`, obtained there from that experiment's own triangulation.
const AREA_REFERENCE = 114.777

println("the mapped domain — `eq:mapping`, the Grad-Shafranov case C2")
@printf("  analytic vs finite-difference Jacobian, worst relative error = %.3e\n", fd_err)
@printf("  𝟙ᵀ 𝕄_J 𝟙        = %.6f\n", area_mapped)
@printf("  reference        = %.6f    (takeda.jl, from the P₁ triangulation)\n",
    AREA_REFERENCE)
@printf("  relative difference = %.3e\n",
    abs(area_mapped - AREA_REFERENCE) / AREA_REFERENCE)

# The reference is quoted to six figures, so agreement is asked for to that and no further.
mapped_pass = fd_err < 1e-7 && abs(area_mapped - AREA_REFERENCE) / AREA_REFERENCE < 1e-5
println("  passes: ", mapped_pass)
println()

## ---------------------------------------------------------------------------------------

allpass = pu_pass && nonneg_pass && param_pass && mapped_pass
println(allpass ? "ALL CHECKS PASS" : "SOME CHECK FAILED")
exit(allpass ? 0 : 1)
