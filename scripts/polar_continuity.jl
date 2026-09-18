# C⁰ and C¹ of the polar spline space at the pole, measured rather than asserted, each
# against a control that must fail.
#
# The claim is that a `PolarSplineBasis` is single-valued at s = 0 and has a single-valued
# gradient there in the pseudo-Cartesian chart, by construction. A measurement that cannot
# tell a space with the property from one without it measures nothing, so each check is run on
# the polar basis, which must pass, and on a control that must fail.
#
# Three checks, three controls:
#
#   C⁰   control: the plain `TensorProductBasis` the polar space is built from. Its
#        θ-dependence at s = 0 is unconstrained, so a spline in it takes an O(1) range of
#        values over the circle that the map sends to one point.
#
#   C¹   control: the space that imposes C⁰ and nothing more — the first radial row tied to a
#        constant, the second row left free. It passes the C⁰ check by construction, which is
#        what makes it a control for C¹ alone, and its radial derivative at the pole is a
#        generic angular spline, which no single gradient fits.
#
#   span control: the pole triangle with one of its three functions dropped. Its span still
#        contains the constants and one linear direction, so it is *still C¹* — a subspace of
#        a C¹ space is C¹ — and it therefore cannot be a control for the C¹ check. What it
#        fails is completeness: ỹ leaves that span, and the projection error onto the pole
#        cells jumps from round-off to a few per cent. That is what the third of the three
#        functions earns, and this is where it is measured.
#
# A fourth section measures the same two properties on a space with a homogeneous-Dirichlet
# **rim**, where the expected answer is that nothing changes. A rim condition recombines the
# outer end of the radial axis and the pole triangle reads only the first two functions of the
# inner end, so the two compose with nothing to reconcile. What the rim does change is the
# partition of unity, which `polar_partition_of_unity.jl` measures; here it appears only as the
# positive control that the constant has actually left the space, without which "C⁰ and C¹ still
# hold" would be consistent with the rim condition having done nothing at all.
#
# Run: julia --project=. --startup-file=no scripts/polar_continuity.jl

using LinearAlgebra
using Printf
using Random
using SimpleSplines
using SparseArrays

Random.seed!(20260917)

const NANGLES = 32
const ANGLES = range(0, 2π; length = NANGLES + 1)[1:NANGLES]

radial = BSplineBasis(UniformMesh(12, 0 .. 1), 3)
angular = PeriodicBSplineBasis(UniformMesh(24, 0 .. 2π), 3)

B = PolarSplineBasis(radial, angular)
P = parent(B)

Ns = nbasis(radial)
Nθ = nbasis(angular)
θnodes = nodes(angular)
greville = nodes(radial)

println("polar basis          ", B)
println("parent basis         ", P)
println("functions            ", nbasis(B), " polar, ", nbasis(P), " tensor-product")
println()

spread(v) = maximum(v) - minimum(v)

# A spline given by a recombination of the parent, evaluated through the parent. This is how
# the two control spaces are reached without giving either of them a type of its own.
recombined(R, û, x, d = (0, 0)) = evaluate(P, reshape(R * û, size(P)), x, d)

## ---------------------------------------------------------------------------------------
## C⁰: the value at the pole does not depend on the angle
## ---------------------------------------------------------------------------------------

û_polar = randn(nbasis(B))
û_tensor = randn(size(P)...)

s0_polar = spread([evaluate(B, û_polar, (0.0, θ)) for θ in ANGLES])
s0_tensor = spread([evaluate(P, û_tensor, (0.0, θ)) for θ in ANGLES])

# Every basis function, not only one random spline: a spread hiding in one function and
# cancelling in a random combination would pass the test above.
s0_each = maximum(k -> spread([evaluate(B, k, (0.0, θ)) for θ in ANGLES]), 1:nbasis(B))

println("C⁰ at the pole")
@printf("  polar,   random spline          spread = %.3e\n", s0_polar)
@printf("  polar,   worst basis function   spread = %.3e\n", s0_each)
@printf("  CONTROL  tensor product         spread = %.3e   (must be O(1))\n", s0_tensor)
c0_pass = s0_polar < 1e-14 && s0_each < 1e-14
c0_control_fails = s0_tensor > 1e-2
println("  polar passes: ", c0_pass, "        control fails as it must: ", c0_control_fails)
println()

## ---------------------------------------------------------------------------------------
## C¹: the pseudo-Cartesian gradient at the pole does not depend on the angle
## ---------------------------------------------------------------------------------------

# The chart is x̃ = s C(θ), ỹ = s S(θ) with C, S the angular splines whose coefficients are
# cos θⱼ and sin θⱼ. Along the ray of fixed θ the chart is a straight line through the origin
# with direction (C(θ), S(θ)), so
#
#     ∂_s u(0,θ) = ∇u · (C(θ), S(θ)) ,
#
# and one gradient explains every angle exactly when u is C¹ there. Fitting (∇u) by least
# squares over many angles and reporting the residual is that statement: zero residual means
# one gradient suffices, an O(1) residual means the "gradient" depends on which angles it was
# read from.
Cspl(θ) = evaluate(angular, cos.(θnodes), θ)
Sspl(θ) = evaluate(angular, sin.(θnodes), θ)

function gradient_residual(∂s)
    A = [Cspl.(ANGLES) Sspl.(ANGLES)]
    b = [∂s(θ) for θ in ANGLES]
    g = A \ b
    return (g, norm(A * g - b, Inf) / max(1, norm(b, Inf)))
end

# The radial derivative at the pole, taken analytically rather than by a difference quotient:
# a quotient over a shrinking s would measure the limit and the round-off together.
g_polar, r_polar = gradient_residual(θ -> evaluate(B, û_polar, (0.0, θ), (1, 0)))

# CONTROL: C⁰ and nothing more. Column 1 ties the whole first radial row to one coefficient;
# every function of the second row keeps its own.
function c0_only_recombination(Ns, Nθ)
    Is, Js, Vs = Int[], Int[], Float64[]
    for j in 1:Nθ
        push!(Is, 1 + (j - 1) * Ns)
        push!(Js, 1)
        push!(Vs, 1.0)
    end
    for j in 1:Nθ
        push!(Is, 2 + (j - 1) * Ns)
        push!(Js, 1 + j)
        push!(Vs, 1.0)
    end
    for j in 1:Nθ, i in 3:Ns

        push!(Is, i + (j - 1) * Ns)
        push!(Js, 1 + Nθ + (i - 2) + (j - 1) * (Ns - 2))
        push!(Vs, 1.0)
    end
    sparse(Is, Js, Vs, Ns * Nθ, 1 + Nθ + (Ns - 2) * Nθ)
end

R₀ = c0_only_recombination(Ns, Nθ)
û₀ = randn(size(R₀, 2))

g_c0, r_c0 = gradient_residual(θ -> recombined(R₀, û₀, (0.0, θ), (1, 0)))
s0_c0 = spread([recombined(R₀, û₀, (0.0, θ)) for θ in ANGLES])

println("C¹ at the pole — the relative residual of one gradient fitted over ", NANGLES,
    " angles")
@printf("  polar,   random spline          residual = %.3e   ∇u = (% .4f, % .4f)\n",
    r_polar, g_polar[1], g_polar[2])
@printf("  CONTROL  C⁰-only space          residual = %.3e   (must be O(1))\n", r_c0)
@printf("  CONTROL  C⁰-only space, C⁰      spread   = %.3e   (must be at round-off)\n",
    s0_c0)
c1_pass = r_polar < 1e-12
c1_control_fails = r_c0 > 1e-2 && s0_c0 < 1e-14
println("  polar passes: ", c1_pass, "        control fails as it must: ", c1_control_fails)
println()

## ---------------------------------------------------------------------------------------
## The linear functions of the chart are in the space exactly
## ---------------------------------------------------------------------------------------

# A stronger statement than the gradient being single-valued, and the reason it is: the polar
# space contains 1, x̃ and ỹ exactly, not to the accuracy of the angular basis. The
# coefficients are written down rather than projected, so what is measured is the span and not
# the projection.

function chart_coefficients(α, β)
    û = zeros(nbasis(B))
    # A linear function of the chart, expanded in the barycentric coordinates of the pole
    # triangle, has as its k-th weight its own value at vertex k.
    V = pole_triangle(B)
    for k in 1:3
        û[k] = α * V[k, 1] + β * V[k, 2]
    end
    for i in 3:Ns, j in 1:Nθ

        û[3 + (i - 2) + (j - 1) * (Ns - 2)] = greville[i] *
                                              (α * cos(θnodes[j]) + β * sin(θnodes[j]))
    end
    return û
end

pts = [(s, θ) for s in range(0, 1; length = 11), θ in ANGLES]

û_one = ones(nbasis(B))
err_1 = maximum(x -> abs(evaluate(B, û_one, x) - 1), pts)
err_x = maximum(
    x -> abs(evaluate(B, chart_coefficients(1.0, 0.0), x) -
             pseudo_cartesian(B, x)[1]), pts)
err_y = maximum(
    x -> abs(evaluate(B, chart_coefficients(0.0, 1.0), x) -
             pseudo_cartesian(B, x)[2]), pts)

println("the chart's constants and linears are in the space exactly")
@printf("  max |u - 1|   = %.3e\n", err_1)
@printf("  max |u - x̃|   = %.3e\n", err_x)
@printf("  max |u - ỹ|   = %.3e\n", err_y)
linear_pass = max(err_1, err_x, err_y) < 1e-13
println("  passes: ", linear_pass)
println()

## ---------------------------------------------------------------------------------------
## Three pole functions and not two: the span control
## ---------------------------------------------------------------------------------------

# Dropping one of the three leaves a space that is still C¹ — a subspace of a C¹ space is C¹ —
# so it cannot fail the check above. What it fails is completeness: the L² projection of ỹ
# onto it carries an O(1) error, while the full space reproduces ỹ to round-off.

q = PolarSplineQuadrature(B)
M = mass_matrix(q)
Φ = basis_values(q, (0, 0))
w = quadrature_weights(q)
xq = [pt for pt in Iterators.product(quadrature_nodes(q)...)][:]

ỹ = [pseudo_cartesian(B, x)[2] for x in xq]

# The two cells the pole functions are supported on. The global norm is reported too, but it
# is the wrong yardstick for this control: outside the pole cells the tensor-product rows
# represent ỹ perfectly whether Ψ₃ is there or not, so a global relative error dilutes an O(1)
# local failure down to the volume fraction of the first two cells.
pole_support = 2 * meshwidth(radial)
pole_cells = [x[1] ≤ pole_support for x in xq]

l2(v, mask) = sqrt(abs(sum(v[mask] .^ 2 .* w[mask])))
all_of_it = trues(length(xq))

function projection_errors(rows)
    û = M[rows, rows] \ (Φ[rows, :] * (ỹ .* w))
    r = Φ[rows, :]' * û .- ỹ
    (l2(r, all_of_it) / l2(ỹ, all_of_it), l2(r, pole_cells) / l2(ỹ, pole_cells))
end

gl_full, pole_full = projection_errors(1:nbasis(B))
gl_red, pole_red = projection_errors([1; 2; 4:nbasis(B)])

println("the third pole function is not redundant — relative L² error of the projection of ỹ")
@printf("  polar,   all three pole functions  global = %.3e   pole cells = %.3e\n",
    gl_full, pole_full)
@printf("  CONTROL  Ψ₃ dropped                global = %.3e   pole cells = %.3e",
    gl_red, pole_red)

println("")
# Not "must be O(1)": outside the pole cells the outer rows still represent ỹ exactly, and
# even inside them Ψ₁ and Ψ₂ recover the part of ỹ that lies along the one linear direction
# they span. What is left is a few per cent of the local norm — against a full space that is
# exact to round-off, which is a discrimination of thirteen orders of magnitude, and that is
# what a control has to establish.
span_pass = max(gl_full, pole_full) < 1e-12
span_control_fails = pole_red > 1e10 * pole_full && pole_red > 1e-3
println("  polar passes: ", span_pass, "        control fails as it must: ",
    span_control_fails)
println()

## ---------------------------------------------------------------------------------------
## A homogeneous-Dirichlet rim leaves the pole alone
## ---------------------------------------------------------------------------------------

# The expected answer here is "the same numbers as above". That is the claim: a rim condition
# is a boundary condition on the outer end of the radial axis, the pole triangle is built from
# the first two functions of the inner end, and a right-recombined basis's first two functions
# are the clamped parent's unchanged. So C⁰ and C¹ at the pole are not merely preserved to
# round-off, they are preserved because nothing they are computed from has moved.
#
# The positive control is the constant. Without it this section would pass equally well against
# a rim condition that did nothing, which is the failure mode of a check that only reports that
# something is still true.

rim = RecombinedBSplineBasis(radial, Free(), Dirichlet())
D = PolarSplineBasis(rim, angular)

û_rim = randn(nbasis(D))

s0_rim = spread([evaluate(D, û_rim, (0.0, θ)) for θ in ANGLES])
s0_rim_each = maximum(k -> spread([evaluate(D, k, (0.0, θ)) for θ in ANGLES]), 1:nbasis(D))
r_rim = last(gradient_residual(θ -> evaluate(D, û_rim, (0.0, θ), (1, 0))))

# The two functions the triangle is read from, and the number it rests on.
row_shift = maximum(abs(evaluate(rim, k, x) - evaluate(radial, k, x))
for k in 1:2, x in range(0, 1; length = 81))
dN2_shift = abs(evaluate(rim, 2, 0.0, 1) - evaluate(radial, 2, 0.0, 1))

# CONTROL: the constant. It is in the free space to round-off and must leave the rim space.
qr = PolarSplineQuadrature(D)
Φr = basis_values(qr, (0, 0))
wr = quadrature_weights(qr)
one_rim = let û = l2_projection(qr, x -> 1.0)
    r = Φr' * û .- 1
    sqrt(abs(sum(r .^ 2 .* wr))) / sqrt(abs(sum(wr)))
end
one_free = let û = l2_projection(q, x -> 1.0)
    r = Φ' * û .- 1
    sqrt(abs(sum(r .^ 2 .* w))) / sqrt(abs(sum(w)))
end

# And the rim itself: no function in the space has a value at s = 1.
rim_value = maximum(abs(evaluate(D, k, (1.0, θ))) for k in 1:nbasis(D), θ in ANGLES)

println("a homogeneous-Dirichlet rim leaves the pole alone")
println("  functions              ", nbasis(B), " free → ", nbasis(D), " with the rim, a ",
    "difference of ", nbasis(B) - nbasis(D), " = Nθ")
@printf("  rim,     random spline          C⁰ spread = %.3e   (free: %.3e)\n",
    s0_rim, s0_polar)
@printf("  rim,     worst basis function   C⁰ spread = %.3e   (free: %.3e)\n",
    s0_rim_each, s0_each)
@printf("  rim,     random spline          C¹ resid  = %.3e   (free: %.3e)\n", r_rim,
    r_polar)
@printf("  the two rows the triangle reads          shift = %.3e   N₂'(a) shift = %.3e\n",
    row_shift, dN2_shift)
@printf("  max |Ψ_k(1, θ)|                                = %.3e\n", rim_value)
@printf("  CONTROL  relative L² error of 1    rim = %.3e   free = %.3e   (rim must be O(1))\n",
    one_rim, one_free)
rim_pass = s0_rim < 1e-14 && s0_rim_each < 1e-14 && r_rim < 1e-12 &&
           row_shift == 0 && dN2_shift == 0 && rim_value == 0 &&
           nbasis(B) - nbasis(D) == Nθ
rim_control_fails = one_rim > 1e-2 && one_free < 1e-12
println("  rim passes: ", rim_pass, "        control fails as it must: ", rim_control_fails)
println()

## ---------------------------------------------------------------------------------------

allpass = c0_pass && c0_control_fails && c1_pass && c1_control_fails &&
          linear_pass && span_pass && span_control_fails &&
          rim_pass && rim_control_fails
println(allpass ? "ALL CHECKS PASS" : "SOME CHECK FAILED")
exit(allpass ? 0 : 1)
