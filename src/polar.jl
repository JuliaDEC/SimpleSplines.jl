@doc raw"""
    PolarSplineBasis(radial, angular)
    PolarSplineBasis(B::TensorProductBasis)

The polar spline space on the parameter square ``[a,b] \times [c,c+L)``, whose left radial
endpoint ``s = a`` is a **pole**: one point of the physical domain, reached from every angle.

The first two rows of the radial basis — ``2 N_\theta`` tensor-product functions — are
replaced by **three**, the *pole triangle*, which together span the constants and the two
linear functions of the pseudo-Cartesian chart at the pole. Every other row passes through
unchanged, so

```math
N = 3 + (N_s - 2) \, N_\theta .
```

```jldoctest
julia> B = PolarSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3),
                            PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3));

julia> nbasis(B), nbasis(parent(B))
(147, 176)

julia> û = zeros(nbasis(B)); û[1:3] .= 1;      # the three pole functions, added up

julia> abs(evaluate(B, û, (0.0, 0.3)) - evaluate(B, û, (0.0, 2.9))) < 1e-15
true
```

# Why this is not a boundary condition

A [`BoundaryCondition`](@ref) constrains one axis at one end, and
[`BSplineBasis`](@ref)`(mesh, p, bc)` builds the one-dimensional basis that satisfies it.
A pole is not of that kind: single-valuedness at ``s = a`` says that the ``\theta``-dependence
there is constant, which **couples the two axes**. There is no radial basis whose tensor
product with anything is a polar space, so this is a two-dimensional basis type beside
[`TensorProductBasis`](@ref) rather than a fourth boundary condition — and the
independent-per-axis promise of the tensor-product layer stays true, because the polar space
is not one.

# The construction

Write a tensor-product spline as ``u = \sum_{ij} \hat{u}_{ij} N_i(s) M_j(\theta)`` with the
radial basis clamped of degree ``p \ge 2`` and the angular basis periodic. A clamped basis has
``N_1(a) = 1`` and ``N_i(a) = 0`` for ``i > 1``, and, at degree two or more, only ``N_1`` and
``N_2`` have a nonzero derivative there. So the value and the radial derivative at the pole
read

```math
u(a,\theta) = \sum_j \hat{u}_{1j} M_j(\theta) ,
\qquad
\partial_s u(a,\theta) = \sum_j \bigl( \hat{u}_{1j} N_1'(a) + \hat{u}_{2j} N_2'(a) \bigr)
                         M_j(\theta) ,
```

and nothing else in the basis can affect either. Imposing

```math
\hat{u}_{1j} = c ,
\qquad
\hat{u}_{2j} = c + \frac{\alpha \cos \theta_j + \beta \sin \theta_j}{N_2'(a)} ,
```

with ``\theta_j`` the Greville abscissae of the angular basis ([`nodes`](@ref)), makes
``u(a,\theta) = c`` a constant — the partition of unity of the angular basis — and

```math
\partial_s u(a,\theta) = \alpha \, C(\theta) + \beta \, S(\theta) ,
\qquad
C = \sum_j \cos\theta_j \, M_j , \quad S = \sum_j \sin\theta_j \, M_j .
```

That is a linear function of the **pseudo-Cartesian** coordinates

```math
\tilde{x} = (s-a) \, C(\theta) , \qquad \tilde{y} = (s-a) \, S(\theta)
```

— see [`pseudo_cartesian`](@ref) — so ``u`` is ``C^1`` at the pole in that chart, by
construction and not by refinement. The constrained set is three-dimensional, parametrised by
``(c, \alpha, \beta)``.

# The pole triangle

The three functions that span it are chosen to be the barycentric coordinates of a triangle in
the ``(\tilde{x}, \tilde{y})`` plane, with vertices

```math
v_k = \frac{2}{N_2'(a)} \bigl( \cos \psi_k, \, \sin \psi_k \bigr) ,
\qquad \psi_k = \frac{2\pi (k-1)}{3} ,
```

so that ``\Psi_k`` has the value ``1/3`` at the pole and the gradient
``\nabla \ell_k = \tfrac{2}{3}\|v\|^{-1}(\cos\psi_k, \sin\psi_k)``. Written out, the
coefficients of ``\Psi_k`` on the first two rows are

```math
\lambda^{(1)}_{kj} = \tfrac{1}{3} ,
\qquad
\lambda^{(2)}_{kj} = \tfrac{1}{3} \bigl( 1 + \cos(\psi_k - \theta_j) \bigr) ,
```

which is why that vertex radius and no other: it is the smallest for which
``\lambda^{(2)} \ge 0``, so the basis stays **non-negative**, and the three vertices sum to
zero, so ``\sum_k \Psi_k = \sum_j (N_1 + N_2) M_j`` and the whole basis is still a
**partition of unity**. Any other radius spans the same three-dimensional space and is only a
worse-conditioned basis of it. See [`pole_triangle`](@ref).

# Indexing and coefficients

The index set is **not** a product, so a spline in this basis carries a coefficient
**vector** of length `nbasis(B)`, not an array: `1:3` are the pole functions and
`3 + (i-2) + (j-1)*(N_s-2)` is the outer function ``N_i M_j``, the radial index running
fastest as it does in `LinearIndices(parent(B))`.

Every basis function is a combination of the parent's, ``\Psi_k = \sum_I R_{Ik} \Phi_I``, and
every assembly is the parent's conjugated by that matrix — exactly as for a
[`RecombinedBSplineBasis`](@ref) one dimension down. See [`recombination_matrix`](@ref) and
[`parent_coefficients`](@ref).

# Requirements

The radial basis must be **clamped and unconstrained at the pole end**, of degree at least
two; degree one has no ``C^1`` to impose, and a periodic radial axis has no pole. A plain
[`BSplineBasis`](@ref) qualifies, and so does a [`RecombinedBSplineBasis`](@ref) carrying
[`Free`](@ref) at the pole end — see *The rim* below. A rim condition of order ``m`` must
also leave the first two radial functions alone, which needs ``m + 3`` functions in the
parent it recombines; only a condition of order two or more can fail that. At least one
radial row must survive the pole triangle, which a clamped basis gives for free and a rim
condition can take away.
The angular basis must be a [`PeriodicBSplineBasis`](@ref) with at least three functions, or
``C``, ``S`` and the constants are not independent and the triangle is degenerate.

# The rim

The pole is not a boundary condition — it couples the two axes, which is why this type exists
beside [`TensorProductBasis`](@ref) rather than as a fourth `BSplineBasis(mesh, p, bc)`
method. The **rim**, the outer end ``s = b``, is an ordinary boundary condition on the radial
axis and is imposed where every other one is, by recombining that axis:

```julia
radial = RecombinedBSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3), Free(), Dirichlet())
B = PolarSplineBasis(radial, PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3))
```

The two compose with nothing to reconcile. A rim condition changes only rows the pole
triangle does not read: the triangle is built from the first two functions and their
derivatives at ``s = a``, and a right-recombined basis's first two functions **are** the
clamped parent's, unchanged — as long as the rim block stays clear of them, which is what the
``m + 3`` requirement above asks for. So ``R`` is built exactly as before, on the smaller
``N_s``.

What this costs is the **partition of unity**: a homogeneous-Dirichlet rim removes the
constant from the space by construction, so ``\sum_k \Psi_k \equiv 1`` becomes false — near
the rim, and only there. The ``C^0`` and ``C^1`` properties at the pole are untouched, since
no function the triangle is built from has changed. [`polynomial_reproduction`](@ref) of the
radial axis says which of the two regimes a basis is in.
"""
# The pole end needs two things and neither is "the basis is a `BSplineBasis`": the knot
# vector must be clamped, so that exactly two functions reach the pole, and that end must be
# unrecombined, so that those two functions are the ones the triangle is built from. A basis
# recombined at the *outer* end only satisfies both — its first two functions are the clamped
# parent's, unchanged — which is what lets a rim condition compose with the pole triangle
# instead of having to rebuild it.
_pole_end_is_clamped(radial) = radial isa BSplineBasis
function _pole_end_is_clamped(radial::RecombinedBSplineBasis)
    boundary(radial)[1] isa Free
end

_pole_end_description(radial) = "a $(nameof(typeof(radial)))"
function _pole_end_description(radial::RecombinedBSplineBasis)
    "a RecombinedBSplineBasis carrying $(boundary(radial)[1]) at the pole end"
end

# An unrecombined pole end is not enough on its own: the triangle needs its first two
# functions to be the parent's, and a rim condition of order `m` replaces the parent's last
# `m + 1` functions by `m` combinations of them. Only functions `1 : Np - m - 1` pass through
# unchanged, so on a coarse enough parent the rim block reaches function two, a *third*
# function then has a derivative at the pole, and the C¹ constraint the triangle imposes is
# built on the wrong rows — silently, since C⁰ survives it. `Np ≥ m + 3` is what keeps the
# first two clear; only a condition of order two or more can reach past the `Ns ≥ 3` guard
# below and fail it.
_check_rim_clears_pole(radial) = nothing
function _check_rim_clears_pole(radial::RecombinedBSplineBasis)
    m = constraint_order(boundary(radial)[2])
    Np = nbasis(parent(radial))
    Np ≥ m + 3 || throw(ArgumentError(
        "the rim condition of the radial axis reaches the pole: $(boundary(radial)[2]) is " *
        "of order $(m), so it recombines the last $(m + 1) of the parent's $(Np) " *
        "functions, and the first two — the ones the pole triangle is built from — are not " *
        "among the ones left unchanged. A parent of at least $(m + 3) functions keeps them " *
        "clear; refine the radial mesh or raise its degree"))
    return nothing
end

struct PolarSplineBasis{T, PT <: TensorProductBasis{T, 2}}
    parent::PT
    λ::Matrix{T}
    R::SparseMatrixCSC{T, Int}
    triangle::Matrix{T}
    cosθ::Vector{T}
    sinθ::Vector{T}

    function PolarSplineBasis(parent::PT) where {T, PT <: TensorProductBasis{T, 2}}
        radial, angular = bases(parent)

        _pole_end_is_clamped(radial) || throw(ArgumentError(
            "the pole end of the radial axis of a polar spline basis must be clamped and " *
            "unconstrained, and $(_pole_end_description(radial)) is not: the construction " *
            "reads the value and the derivative of the first two functions at the pole, " *
            "which only a clamped knot vector localises there, and recombining that end " *
            "would change the two rows the pole triangle replaces. The *rim* — the outer " *
            "end — may carry any boundary condition; pass " *
            "`RecombinedBSplineBasis(parent, Free(), Dirichlet())`"))
        _check_rim_clears_pole(radial)
        angular isa PeriodicBSplineBasis || throw(ArgumentError(
            "the angular axis of a polar spline basis must be a PeriodicBSplineBasis, not a " *
            "$(nameof(typeof(angular))): the pole is reached from every angle, so the " *
            "angular axis has no ends"))

        p = degree(radial)
        p ≥ 2 || throw(ArgumentError(
            "a polar spline basis needs a radial degree of at least two, got $(p): at " *
            "degree one the first two functions are the only ones with a derivative at the " *
            "pole *and* the only ones with a value there, so imposing C¹ would leave the " *
            "constants alone and there would be no space left to refine"))

        Ns = nbasis(radial)
        Nθ = nbasis(angular)

        # A clamped basis has `ncells + p` functions and `p ≥ 2` is checked above, so `Ns ≥ 3`
        # follows and this cannot fail. A rim condition removes one function, and then it can:
        # `ncells = 1` at `p = 2` leaves `Ns = 2`, the two rows the pole triangle replaces and
        # nothing else, i.e. a basis of the three pole functions alone.
        Ns ≥ 3 || throw(ArgumentError(
            "the radial basis has $(Ns) functions, too few for the pole triangle to leave " *
            "anything behind: it replaces the first two rows, so a third is needed for the " *
            "space to hold more than the triangle itself; refine the radial mesh or raise " *
            "its degree"))
        Nθ ≥ 3 || throw(ArgumentError(
            "the angular basis has $(Nθ) functions, too few for the pole triangle: the " *
            "constants, C and S must be independent in it, and on fewer than three " *
            "functions they are not"))

        a = leftendpoint(domain(radial))

        # The number the whole construction rests on, read off the basis rather than written
        # out as p/h₁: that formula is right for a clamped uniform knot vector and the
        # measurement is right whatever the mesh does. The test suite checks the two against
        # each other, and checks N₁'(a) = -N₂'(a), which is what makes the constant `c` drop
        # out of the second constraint above.
        dN2 = evaluate(radial, 2, a, 1)

        θ = nodes(angular)
        ψ = T[2π * (k - 1) / 3 for k in 1:3]

        # The smallest vertex radius that keeps λ⁽²⁾ non-negative; see the docstring.
        r = 2 / dN2
        triangle = T[k == 1 ? r * cos(ψ[j]) : r * sin(ψ[j]) for j in 1:3, k in 1:2]

        λ = T[(1 + cos(ψ[k] - θ[j])) / 3 for k in 1:3, j in 1:Nθ]

        R = _polar_recombination(λ, Ns, Nθ)

        # The coefficients of the chart splines C and S; see `pseudo_cartesian`. They are
        # fixed for the basis and a per-point rebuild is O(Nθ) in a routine called once per
        # quadrature point.
        cosθ = T[cos(θj) for θj in θ]
        sinθ = T[sin(θj) for θj in θ]

        new{T, PT}(parent, λ, R, triangle, cosθ, sinθ)
    end
end

function PolarSplineBasis(radial::AbstractBSplineBasis, angular::AbstractBSplineBasis)
    PolarSplineBasis(TensorProductBasis(radial, angular))
end

# The sparse `nbasis(parent) × N` matrix with `Ψ_k = Σ_I R[I,k] Φ_I`. Parent rows are indexed
# `i + (j-1)*Ns`, the radial index fastest, which is `LinearIndices(parent)`; the polar columns
# are the three pole functions and then the outer rows in the same order.
function _polar_recombination(λ::Matrix{T}, Ns::Integer, Nθ::Integer) where {T}
    N = 3 + (Ns - 2) * Nθ

    Is = Int[]
    Js = Int[]
    Vs = T[]

    for k in 1:3, j in 1:Nθ

        push!(Is, 1 + (j - 1) * Ns)
        push!(Js, k)
        push!(Vs, one(T) / 3)

        push!(Is, 2 + (j - 1) * Ns)
        push!(Js, k)
        push!(Vs, λ[k, j])
    end

    for j in 1:Nθ, i in 3:Ns

        push!(Is, i + (j - 1) * Ns)
        push!(Js, 3 + (i - 2) + (j - 1) * (Ns - 2))
        push!(Vs, one(T))
    end

    return sparse(Is, Js, Vs, Ns * Nθ, N)
end

"""
    parent(B::PolarSplineBasis)

The [`TensorProductBasis`](@ref) the polar space is a subspace of.
"""
Base.parent(B::PolarSplineBasis) = B.parent

"""
    recombination_matrix(B::PolarSplineBasis)

The sparse matrix `R` with `Ψ_k = Σ_I R[I,k] Φ_I`, of size `nbasis(parent(B)) × nbasis(B)`.

Its columns `1:3` are the pole triangle and carry `2 * Nθ` nonzeros each; every other column
is a single one. As for a [`RecombinedBSplineBasis`](@ref), every assembly of the polar basis
is the parent's conjugated by this matrix — `M̃ = R' * M * R`, `Φ̃ = R' * Φ` — which is why
[`PolarSplineQuadrature`](@ref) needs no quadrature rule of its own.
"""
recombination_matrix(B::PolarSplineBasis) = B.R

@doc raw"""
    pole_triangle(B::PolarSplineBasis)

The `3 × 2` matrix of the pole-triangle vertices ``v_k`` in the pseudo-Cartesian chart.

The ``k``-th pole function restricted to the pole — its value and its gradient there — is the
barycentric coordinate of this triangle that is one at ``v_k`` and zero at the other two.
"""
pole_triangle(B::PolarSplineBasis) = B.triangle

@doc raw"""
    pseudo_cartesian(B::PolarSplineBasis, x)

The chart ``(\tilde{x}, \tilde{y}) = (s-a) \, (C(\theta), S(\theta))`` in which the polar
space is ``C^1`` at the pole, with

```math
C = \sum_j \cos \theta_j \, M_j , \qquad S = \sum_j \sin \theta_j \, M_j
```

the angular splines whose coefficients are the cosine and sine of the Greville abscissae.

``C`` and ``S`` are splines rather than the trigonometric functions themselves, and that is
the point: ``\cos\theta`` is not in the angular spline space, so a chart built from it would
make the ``C^1`` property hold only up to the approximation error of that space. Built from
``C`` and ``S`` it holds exactly, and the space contains ``1``, ``\tilde{x}`` and
``\tilde{y}`` to round-off. A geometry map that is itself represented in this space — the
isogeometric case — therefore carries the smoothness to the physical domain.

The price is that the unit circle of this chart is the spline through the Greville values of
the cosine and the sine, which sits a little inside the true one — some 2.5% at 16 angular
cells, a gap that closes at the angular order under refinement. The chart is a chart, not a
measurement: distances in it are not distances on the disk.

```jldoctest
julia> B = PolarSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3),
                            PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3));

julia> pseudo_cartesian(B, (0.0, 1.2))
(0.0, 0.0)

julia> pseudo_cartesian(B, (1.0, 0.7)) .≈ 2 .* pseudo_cartesian(B, (0.5, 0.7))
(true, true)

julia> x̃, ỹ = pseudo_cartesian(B, (0.5, 0.0)); abs(x̃ - 0.5) < 0.02, abs(ỹ) < 1e-15
(true, true)
```
"""
function pseudo_cartesian(B::PolarSplineBasis{T}, x) where {T}
    radial, angular = bases(B.parent)
    s = x[1] - leftendpoint(domain(radial))
    C = evaluate(angular, B.cosθ, x[2])
    S = evaluate(angular, B.sinθ, x[2])
    return (s * C, s * S)
end

Base.eltype(::PolarSplineBasis{T}) where {T} = T
Base.eltype(::Type{<:PolarSplineBasis{T}}) where {T} = T
Base.ndims(::PolarSplineBasis) = 2
Base.ndims(::Type{<:PolarSplineBasis}) = 2
Base.length(B::PolarSplineBasis) = size(B.R, 2)
Base.eachindex(B::PolarSplineBasis) = Base.OneTo(length(B))

nbasis(B::PolarSplineBasis) = length(B)
degree(B::PolarSplineBasis) = degree(B.parent)
order(B::PolarSplineBasis) = order(B.parent)
ncells(B::PolarSplineBasis) = ncells(B.parent)
meshwidth(B::PolarSplineBasis) = meshwidth(B.parent)
mesh(B::PolarSplineBasis) = mesh(B.parent)
domain(B::PolarSplineBasis) = domain(B.parent)
bases(B::PolarSplineBasis) = bases(B.parent)

"""
    pole(B::PolarSplineBasis)

The radial coordinate of the pole, i.e. the left endpoint of the radial axis.
"""
pole(B::PolarSplineBasis) = leftendpoint(domain(bases(B.parent)[1]))

function Base.show(io::IO, B::PolarSplineBasis{T}) where {T}
    radial, angular = bases(B.parent)
    print(io, "PolarSplineBasis{", T, "}(")
    print(io, "s: p=", degree(radial), ", n=", ncells(radial))
    print(io, " ⊕ θ: p=", degree(angular), ", n=", ncells(angular))
    print(io, ", pole triangle of 3 for 2×", nbasis(angular), ")")
end

Base.:(==)(B1::PolarSplineBasis, B2::PolarSplineBasis) = (B1.parent == B2.parent)
Base.hash(B::PolarSplineBasis, h::UInt) = hash(B.parent, hash(:PolarSplineBasis, h))

## ---------------------------------------------------------------------------------------
## Evaluation
## ---------------------------------------------------------------------------------------

@doc raw"""
    parent_coefficients(B::PolarSplineBasis, û)

The coefficient **array** of `û` in the parent [`TensorProductBasis`](@ref), of size
`size(parent(B))`.

`R * û` reshaped, which is the one place the flattening convention of the polar index set
meets the array shape of the parent. Every evaluation goes through it, and it is exported so
that a caller with a spline in hand can reach the parent's own machinery — plotting, a
tensor-product projection, `evaluate_all!` — without rewriting the index arithmetic.
"""
function parent_coefficients(B::PolarSplineBasis{T}, û::AbstractVector{S}) where {T, S}
    length(û) == nbasis(B) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the basis has $(nbasis(B))"))
    reshape(B.R * û, size(B.parent))
end

@doc raw"""
    evaluate(B::PolarSplineBasis, k::Integer, x, d = (0, 0))
    evaluate(B::PolarSplineBasis, û::AbstractVector, x, d = (0, 0))

The mixed derivative ``\partial_s^{d_1} \partial_\theta^{d_2}`` at `x = (s, θ)` of the `k`-th
basis function, or of the spline with coefficient vector `û`.

```jldoctest
julia> B = PolarSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3),
                            PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3));

julia> sum(evaluate(B, k, (0.3, 1.1)) for k in eachindex(B)) ≈ 1     # partition of unity
true

julia> maximum(abs, [evaluate(B, 1, (0.0, θ)) - 1/3 for θ in range(0, 2π, 17)]) < 1e-15
true
```

The `k`-th basis function is evaluated through the ``k``-th column of
[`recombination_matrix`](@ref), so a pole function costs ``O(N_\theta)`` and every other one
costs the same as the tensor-product function it is. The spline is not: it goes through
[`parent_coefficients`](@ref) and then the parent's local block, which is
``O(\prod_d (p_d+1))`` per point — but it forms `R * û` on every call, so a sweep over many
points should pass the whole vector of them, which lifts that out of the loop.
"""
function evaluate(B::PolarSplineBasis{T}, k::Integer, x,
        d::NTuple{2, Int} = (0, 0)) where {T}
    @boundscheck (1 ≤ k ≤ nbasis(B)) || throw(BoundsError(B, k))
    R = B.R
    rows = rowvals(R)
    vals = nonzeros(R)
    S = _evaltype(T, eltype(x))
    v = zero(S)
    for t in nzrange(R, Int(k))
        v += vals[t] * evaluate(B.parent, rows[t], x, d)
    end
    return v
end

function evaluate(B::PolarSplineBasis{T}, û::AbstractVector{S}, x,
        d::NTuple{2, Int} = (0, 0)) where {T, S}
    evaluate(B.parent, parent_coefficients(B, û), x, d)
end

# Dispatch on the element type rather than on `AbstractVector` alone: a single point written
# as `[s, θ]` is an `AbstractVector` too, and it belongs to the method above.
function evaluate(B::PolarSplineBasis{T}, û::AbstractVector{S},
        X::AbstractVector{<:Union{Tuple, AbstractVector}},
        d::NTuple{2, Int} = (0, 0)) where {T, S}
    ĉ = parent_coefficients(B, û)
    [evaluate(B.parent, ĉ, x, d) for x in X]
end

(B::PolarSplineBasis)(x, k) = evaluate(B, k, x)

Base.getindex(B::PolarSplineBasis, x, k) = evaluate(B, k, x)

@doc raw"""
    evaluate_all(B::PolarSplineBasis, x, d = (0, 0))

The basis functions that are nonzero at `x = (s, θ)`, as a pair `(idx, values)` of the global
indices and the matching values of ``\partial_s^{d_1} \partial_\theta^{d_2} \Psi``.

Unlike the tensor-product and one-dimensional forms, which return the **first index** of a
contiguous block, this returns the indices themselves. The polar index set is not a product
and the block is not contiguous in it: in the first two radial cells the three pole functions
are nonzero alongside the outer rows, and they sit at the front of the index set rather than
beside them. An offset cannot say that, so the indices are given.

```jldoctest
julia> B = PolarSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3),
                            PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3));

julia> idx, vals = evaluate_all(B, (0.02, 1.1));

julia> idx[1:3], sum(vals) ≈ 1
([1, 2, 3], true)

julia> idx, vals = evaluate_all(B, (0.9, 1.1));      # away from the pole: 4 × 4 functions

julia> length(idx), sum(vals) ≈ 1
(16, true)
```
"""
function evaluate_all(B::PolarSplineBasis{T}, x, d::NTuple{2, Int} = (0, 0)) where {T}
    radial, angular = bases(B.parent)
    Ns = nbasis(radial)
    R = _evaltype(T, eltype(x))

    i₀, bs = evaluate_all(radial, x[1], d[1])
    j₀, bθ = evaluate_all(angular, x[2], d[2])

    # The block holds at most the three pole functions and the parent's local block, so one
    # allocation of that size each replaces the handful `push!` would grow through.
    nmax = 3 + length(bs) * length(bθ)
    idx = Int[]
    vals = R[]
    sizehint!(idx, nmax)
    sizehint!(vals, nmax)

    # The pole functions, if the radial block reaches either of the first two rows. Ψ_k is
    # N_1(s)/3 + N_2(s) * Σ_j λ_kj M_j(θ), and both angular sums are taken over the block
    # rather than over all of N_θ, which is what keeps this O(p²) rather than O(N_θ).
    if i₀ ≤ 2
        for k in 1:3
            v = zero(R)
            for ti in eachindex(bs)
                i = basis_index(radial, i₀ + ti - 1)
                (i == 1 || i == 2) || continue
                aθ = zero(R)
                for tj in eachindex(bθ)
                    j = basis_index(angular, j₀ + tj - 1)
                    aθ += bθ[tj] * (i == 1 ? one(R) / 3 : B.λ[k, j])
                end
                v += bs[ti] * aθ
            end
            push!(idx, k)
            push!(vals, v)
        end
    end

    for ti in eachindex(bs)
        i = basis_index(radial, i₀ + ti - 1)
        (3 ≤ i ≤ Ns) || continue
        for tj in eachindex(bθ)
            j = basis_index(angular, j₀ + tj - 1)
            push!(idx, 3 + (i - 2) + (j - 1) * (Ns - 2))
            push!(vals, bs[ti] * bθ[tj])
        end
    end

    return (idx, vals)
end

## ---------------------------------------------------------------------------------------
## The quadrature
## ---------------------------------------------------------------------------------------

@doc raw"""
    PolarSplineQuadrature(B; nq = map(quadrature_order, degree(B)), dmax = 3)

The assembly table of a [`PolarSplineBasis`](@ref): the parent's
[`TensorProductQuadrature`](@ref), together with the tabulations and the mass factorisation
of the polar basis itself.

```jldoctest
julia> B = PolarSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3),
                            PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3));

julia> q = PolarSplineQuadrature(B);

julia> size(basis_values(q, (0, 0)))
(147, 3200)

julia> sum(basis_integrals(q)) ≈ 2π       # ∫ Σ_k Ψ_k over the parameter square
true
```

# Why there is no `KroneckerMass`

The mass matrix of a tensor-product basis is a Kronecker product and is never formed; its
solve is one one-dimensional solve per axis. The pole rows destroy that structure — a pole
function is a sum over the whole angular axis — so the polar mass matrix is assembled and
factorised. It is sparse: the three pole columns are dense in the ``2 N_\theta`` parent rows
they touch, and every other column has the ``(p_s+1)(p_\theta+1)`` neighbours of a
tensor-product function. Only the ``3 \times 3`` pole block is dense, so the factorisation is
that of a banded matrix with three border rows rather than of a dense one.

# Storage

The table ``\Phi_d[k,r] = \partial_s^{d_1} \partial_\theta^{d_2} \Psi_k(x_r)`` over the
flattened quadrature grid is `R' * kron(Φ_θ, Φ_s)`, sparse, formed on first use and memoised.
The memo is what makes [`weighted_matrix`](@ref) affordable inside a Newton loop, where the
coefficient changes at every iteration and the tabulation does not. The Kronecker product is
in reverse axis order because the flattening runs the radial axis fastest — the convention of
[`parent_coefficients`](@ref), of `vec` of a [`quadrature_sample`](@ref) array, and of the
index set itself.
"""
struct PolarSplineQuadrature{T, BT <: PolarSplineBasis{T},
    QT <: TensorProductQuadrature{T, 2}, MO <: MassOperator{T}}
    basis::BT
    parent::QT
    mass::MO
    w::Vector{T}
    Φ::Dict{NTuple{2, Int}, SparseMatrixCSC{T, Int}}
    cache::Dict{Tuple{NTuple{2, Int}, NTuple{2, Int}}, SparseMatrixCSC{T, Int}}
    integrals::Vector{T}

    function PolarSplineQuadrature(B::PolarSplineBasis{T};
            nq = map(quadrature_order, degree(B)), dmax = 3) where {T}
        parent = TensorProductQuadrature(B.parent; nq = nq, dmax = dmax)

        ws, wθ = quadrature_weights(parent)
        w = kron(wθ, ws)

        Φ = Dict{NTuple{2, Int}, SparseMatrixCSC{T, Int}}()
        cache = Dict{Tuple{NTuple{2, Int}, NTuple{2, Int}}, SparseMatrixCSC{T, Int}}()

        Φ₀ = _polar_table(B, parent, (0, 0))
        Φ[(0, 0)] = Φ₀
        M = Φ₀ * Diagonal(w) * Φ₀'
        M = SparseMatrixCSC{T, Int}((M + M') / 2)   # symmetric by construction; enforce it
        mass = FactorizedMass(M)
        integrals = Φ₀ * w

        new{T, typeof(B), typeof(parent), typeof(mass)}(
            B, parent, mass, w, Φ, cache, integrals)
    end
end

function _polar_table(B::PolarSplineBasis{T}, parent::TensorProductQuadrature{T, 2},
        d::NTuple{2, Int}) where {T}
    Φs = basis_values(quadratures(parent)[1], d[1])
    Φθ = basis_values(quadratures(parent)[2], d[2])
    SparseMatrixCSC{T, Int}(B.R' * kron(Φθ, Φs))
end

"""
    basis(q::PolarSplineQuadrature)

The [`PolarSplineBasis`](@ref) the quadrature was built for.
"""
basis(q::PolarSplineQuadrature) = q.basis

"""
    parent(q::PolarSplineQuadrature)

The [`TensorProductQuadrature`](@ref) of the parent basis, which holds the per-axis nodes,
weights and tabulations everything here is built from.
"""
Base.parent(q::PolarSplineQuadrature) = q.parent

nbasis(q::PolarSplineQuadrature) = nbasis(q.basis)
degree(q::PolarSplineQuadrature) = degree(q.basis)
Base.ndims(::PolarSplineQuadrature) = 2
Base.eltype(::PolarSplineQuadrature{T}) where {T} = T

"""
    quadrature_nodes(q::PolarSplineQuadrature)

The per-axis node vectors, as a tuple `(s, θ)`, exactly as for the parent
[`TensorProductQuadrature`](@ref). The two-dimensional grid is their product and the
flattening runs the radial axis fastest.
"""
quadrature_nodes(q::PolarSplineQuadrature) = quadrature_nodes(q.parent)

"""
    quadrature_weights(q::PolarSplineQuadrature)

The **flattened** weight vector of the two-dimensional grid, `kron(w_θ, w_s)`, of length
`prod(quadrature_grid_size(parent(q)))`.

This is a vector where the parent's is a tuple of per-axis vectors: the polar tabulation has
one column per grid point, so the weight that goes with it is one number per grid point too.
"""
quadrature_weights(q::PolarSplineQuadrature) = q.w

"""
    quadrature_grid_size(q::PolarSplineQuadrature)

The per-axis size of the quadrature grid, `(n_d * nq_d)` per axis.
"""
quadrature_grid_size(q::PolarSplineQuadrature) = quadrature_grid_size(q.parent)

@doc raw"""
    basis_values(q::PolarSplineQuadrature, d::NTuple{2,Int})
    basis_values(q::PolarSplineQuadrature, d::Integer = 0)

The table ``\Phi_d[k,r] = \partial_s^{d_1} \partial_\theta^{d_2} \Psi_k(x_r)`` over the
flattened quadrature grid, sparse, formed on first use and memoised.

`d` is a **per-axis multi-index**: `(1,0)` is ``\partial_s``, `(0,1)` is
``\partial_\theta``. The scalar form is accepted only for `d = 0`, where the multi-index is
unambiguous; a scalar `d ≥ 1` names no derivative on a two-dimensional space and is rejected
rather than resolved to one of the axes.
"""
function basis_values(q::PolarSplineQuadrature, d::NTuple{2, Int})
    get!(q.Φ, d) do
        _polar_table(q.basis, q.parent, d)
    end
end

function basis_values(q::PolarSplineQuadrature, d::Integer = 0)
    d == 0 || throw(ArgumentError(
        "the derivative order of a polar spline space is a per-axis multi-index, not the " *
        "scalar $(d): ∂_s and ∂_θ are different tables. Write basis_values(q, (1, 0)) for ∂_s"))
    basis_values(q, (0, 0))
end

@doc raw"""
    mixed_matrix(q::PolarSplineQuadrature, a::NTuple{2,Int}, b::NTuple{2,Int})

The matrix ``\int D^a \Psi_k \, D^b \Psi_l \, ds \, d\theta`` over the parameter square,
memoised.

The integral is against the **parameter** measure. A polar space is used for a mapped domain,
where the measure carries the Jacobian of the map; that is [`weighted_matrix`](@ref) with the
Jacobian as its coefficient, and it is the caller's to supply because the map is.
"""
function mixed_matrix(q::PolarSplineQuadrature, a::NTuple{2, Int}, b::NTuple{2, Int})
    get!(q.cache, (a, b)) do
        A = basis_values(q, a) * Diagonal(q.w) * basis_values(q, b)'
        SparseMatrixCSC{eltype(q), Int}(A)
    end
end

@doc raw"""
    weighted_matrix(q::PolarSplineQuadrature, f, a::NTuple{2,Int}, b::NTuple{2,Int})

The matrix ``\int f(x) \, D^a \Psi_k \, D^b \Psi_l \, ds \, d\theta``, with `f` either a
function of the coordinate pair `(s, θ)` or a vector already sampled on the flattened
quadrature grid.

Not memoised, unlike [`mixed_matrix`](@ref): the coefficient of a mapped assembly or of a
metric bracket depends on the state and changes at every Newton iteration.
"""
function weighted_matrix(q::PolarSplineQuadrature, f, a::NTuple{2, Int}, b::NTuple{2, Int})
    x = quadrature_nodes(q)
    weighted_matrix(q, vec([f(pt) for pt in Iterators.product(x...)]), a, b)
end

function weighted_matrix(q::PolarSplineQuadrature, f::AbstractVector, a::NTuple{2, Int},
        b::NTuple{2, Int})
    length(f) == length(q.w) || throw(DimensionMismatch(
        "the coefficient was sampled at $(length(f)) points but the quadrature grid has " *
        "$(length(q.w))"))
    basis_values(q, a) * Diagonal(f .* q.w) * basis_values(q, b)'
end

"""
    mass_matrix(q::PolarSplineQuadrature)

The mass matrix ``\\mathbb{M}_{kl} = \\int \\Psi_k \\Psi_l \\, ds \\, d\\theta``, symmetric
positive definite, assembled when the quadrature is built and returned by reference.
"""
mass_matrix(q::PolarSplineQuadrature) = mass_matrix(q.mass)

"""
    mass_operator(q::PolarSplineQuadrature)
    mass_factorization(q::PolarSplineQuadrature)

The [`FactorizedMass`](@ref) of the polar mass matrix — a sparse Cholesky, not a
[`KroneckerMass`](@ref); see [`PolarSplineQuadrature`](@ref) for why the Kronecker structure
is not available.
"""
mass_operator(q::PolarSplineQuadrature) = q.mass
mass_factorization(q::PolarSplineQuadrature) = q.mass

@doc raw"""
    stiffness_matrix(q::PolarSplineQuadrature)

The matrix ``\int \nabla \Psi_k \cdot \nabla \Psi_l \, ds \, d\theta``, the sum over the two
axes of the parameter square, symmetric positive semi-definite with the constants in its
kernel.

This is the gradient of the parameter square, not of the mapped domain: the metric of the map
belongs in the weight, as it does for [`mixed_matrix`](@ref).
"""
function stiffness_matrix(q::PolarSplineQuadrature)
    mixed_matrix(q, (1, 0), (1, 0)) + mixed_matrix(q, (0, 1), (0, 1))
end

@doc raw"""
    basis_integrals(q::PolarSplineQuadrature)

The vector ``\int \Psi_k \, ds \, d\theta``.

Equal to ``\mathbb{M} \mathbf{1}`` because the polar basis is a partition of unity — the pole
triangle included, which is what the choice of vertex radius buys. Assembled once and
returned by reference; do not mutate the result.
"""
basis_integrals(q::PolarSplineQuadrature) = q.integrals

@doc raw"""
    l2_projection(q::PolarSplineQuadrature, f)
    l2_projection!(û, q::PolarSplineQuadrature, f)

The coefficient vector of the ``L^2`` projection of `f` onto the polar spline space,
``\hat{u} = \mathbb{M}^{-1} \int f \, \Psi_k \, ds \, d\theta``.

`f` may be a function of the coordinate pair `(s, θ)`, or a vector already sampled on the
flattened quadrature grid.

```jldoctest
julia> B = PolarSplineBasis(BSplineBasis(UniformMesh(16, 0 .. 1), 3),
                            PeriodicBSplineBasis(UniformMesh(32, 0 .. 2π), 3));

julia> q = PolarSplineQuadrature(B);

julia> û = l2_projection(q, x -> 1.0);         # the constants are in the space exactly

julia> abs(evaluate(B, û, (0.0, 0.7)) - 1) < 1e-12
true
```

The load is one sparse matrix-vector product against the tabulation, and the solve is the
sparse Cholesky of [`mass_operator`](@ref). Neither has the Kronecker shortcut of a
tensor-product projection, and that is the cost of the pole.
"""
function l2_projection(q::PolarSplineQuadrature, f)
    l2_projection!(Vector{eltype(q)}(undef, nbasis(q)), q, f)
end

function l2_projection!(û::AbstractVector, q::PolarSplineQuadrature, f)
    length(û) == nbasis(q) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the basis has $(nbasis(q))"))
    F = f isa AbstractVector ? f :
        vec([f(pt) for pt in Iterators.product(quadrature_nodes(q)...)])
    length(F) == length(q.w) || throw(DimensionMismatch(
        "the sample has $(length(F)) entries but the quadrature grid has $(length(q.w))"))
    copyto!(û, basis_values(q, (0, 0)) * (F .* q.w))
    mass_solve!(û, q.mass, û)
    return û
end
