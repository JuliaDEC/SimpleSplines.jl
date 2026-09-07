
@inline _evaltype(::Type{T}, ::Type{S}) where {T, S} = promote_type(T, S)

# Whether `x` sits at the value the topmost knot span is to be closed at. `nothing` is the
# answer for a basis that has no such point -- a periodic one, whose knot vector extends
# beyond the domain in both directions -- and it is a type, not a value, so the comparison
# resolves when the method is compiled.
@inline _atmax(x, ::Nothing) = false
@inline _atmax(x, xmax) = (x == xmax)

@doc raw"""
    _bspline(kv, k, p, x, d, xmax = nothing)

The `d`-th derivative at `x` of the B-spline of degree `p` starting at knot index `k` of the
knot vector `kv`, i.e. the one supported on `kv[k] .. kv[k+p+1]`.

This is the Cox-de Boor recursion written out directly,

```math
\phi_j^p = w_j^p \, \phi_j^{p-1} + (1 - w_{j+1}^p) \, \phi_{j+1}^{p-1} ,
\qquad w_j^p = \frac{x - x_j}{x_{j+p} - x_j} ,
```

with ``\phi_j^0`` the indicator of ``[x_j, x_{j+1})``, together with the derivative
recursion

```math
\frac{d}{dx} \phi_j^p = p \left( \frac{\phi_j^{p-1}}{x_{j+p} - x_j}
                               - \frac{\phi_{j+1}^{p-1}}{x_{j+p+1} - x_{j+1}} \right) .
```

Both are evaluated as written rather than through a faster equivalent, so that the assembly
built on them doubles as a check of the formulae themselves. It is the *reference*
implementation: [`evaluate_all`](@ref) computes the same numbers in ``O(p^2)`` for the whole
local block, and the test suite checks the two against each other.

Written as it stands the recursion is *unmemoised*, so it splits into two subproblems at every
level and the cost of one value is ``O(2^p)`` rather than ``O(p^2)``. That is deliberate — the
point of this function is to be the formula, not to be fast — but it is the reason
[`evaluate_all!`](@ref) exists rather than a convenience wrapper around a loop over this one.
Measured at 20 000 evaluations on a 40-cell mesh: 27 ns per call at ``p = 3`` against 17 ns for
the *whole block* through [`evaluate_all`](@ref), and 15.8 µs against 103 ns at ``p = 12``.

A knot span of zero length — a repeated knot — contributes nothing, which is what makes the
recursion well defined at a knot of multiplicity greater than one. The guard is on the span
being positive rather than on the numerator, because it is the division that fails.

# The right endpoint

``\phi_j^0`` is the indicator of the *half-open* span, which makes every B-spline
right-continuous and makes the partition of unity hold on ``[a,b)`` but not at ``b``: at
``x = b`` every indicator is false and the whole basis evaluates to zero. That is the wrong
answer for a clamped basis, which is meant to be interpolatory at both ends,
``\phi_N (b) = 1``.

`xmax` closes the topmost span on the right: when `x == xmax`, the span `kv[k] .. kv[k+1]` is
treated as containing `x` if it is nonempty and `kv[k] < x ≤ kv[k+1]`. Since this fires only
at that one point, the result is exactly the limit from the left, and every value and
derivative at ``b`` is the one the polynomial on the last cell takes there. Passing
`nothing` — the default, and what a periodic basis passes — leaves the half-open convention
untouched.
"""
function _bspline(kv::AbstractVector{T}, k::Int, p::Int, x::S, d::Int,
        xmax = nothing) where {T, S}
    R = _evaltype(T, S)

    if d > 0
        p == 0 && return zero(R)
        a = kv[k + p] - kv[k]
        b = kv[k + p + 1] - kv[k + 1]
        out = zero(R)
        a > 0 && (out += _bspline(kv, k, p - 1, x, d - 1, xmax) / a)
        b > 0 && (out -= _bspline(kv, k + 1, p - 1, x, d - 1, xmax) / b)
        return p * out
    end

    if p == 0
        inside = (kv[k] ≤ x < kv[k + 1]) ||
                 (_atmax(x, xmax) && kv[k] < x ≤ kv[k + 1])
        return inside ? one(R) : zero(R)
    end

    a = kv[k + p] - kv[k]
    b = kv[k + p + 1] - kv[k + 1]
    out = zero(R)
    a > 0 && (out += (x - kv[k]) / a * _bspline(kv, k, p - 1, x, 0, xmax))
    b > 0 && (out += (kv[k + p + 1] - x) / b * _bspline(kv, k + 1, p - 1, x, 0, xmax))
    return out
end

@doc raw"""
    AbstractBSplineBasis{T} <: Basis{T}

A B-spline basis of some degree on a [`Mesh`](@ref), closed at the ends according to a
[`BoundaryCondition`](@ref).

Three concrete types implement it, and which one a construction produces is decided by the
boundary condition rather than chosen by name — see [`BSplineBasis`](@ref):

  - [`BSplineBasis`](@ref) — the clamped basis, ``N = n + p``, no constraint at either end.
  - [`PeriodicBSplineBasis`](@ref) — the basis on the torus, ``N = n``.
  - [`RecombinedBSplineBasis`](@ref) — a clamped basis with one homogeneous constraint per
    constrained end, ``N = n + p - \#\text{constraints}``.

All three answer [`nbasis`](@ref), [`degree`](@ref), [`order`](@ref), [`nodes`](@ref),
[`evaluate`](@ref), [`evaluate_all`](@ref), `getindex` and `axes`, so the assembly in
[`SplineQuadrature`](@ref) is written once against this interface.
"""
abstract type AbstractBSplineBasis{T} <: Basis{T} end

"""
    mesh(b::AbstractBSplineBasis)

The [`Mesh`](@ref) the basis is built on.
"""
mesh(b::AbstractBSplineBasis) = b.mesh

"""
    knotvector(b::AbstractBSplineBasis)

The knot sequence the Cox-de Boor recursion runs on.
"""
knotvector(b::AbstractBSplineBasis) = b.knots

@doc raw"""
    degree(b::AbstractBSplineBasis)

The polynomial degree ``p`` of the basis: every basis function is piecewise of degree ``p`` and
the space is ``\mathcal{C}^{p-1}`` across an interior breakpoint.

The generic belongs to `GeometricBase` and is extended here rather than redefined. See
[`order`](@ref) for ``k = p+1``, the other convention, and why both exist.
"""
degree(b::AbstractBSplineBasis) = b.p

@doc raw"""
    order(b::AbstractBSplineBasis)

The order ``k = p + 1`` of the spline basis `b`.

Note that this is the *spline* meaning of the word — a B-spline of order `k` is piecewise of
degree `k-1`, and the knot vector of a basis of `N` functions has `N + k` entries. It is not
the meaning `order` carries for the bases of `CompactBasisFunctions`, where it is the number
of basis functions. For a spline basis the two differ.
"""
order(b::AbstractBSplineBasis) = degree(b) + 1

ncells(b::AbstractBSplineBasis) = ncells(mesh(b))
domain(b::AbstractBSplineBasis) = domain(mesh(b))
domainlength(b::AbstractBSplineBasis) = domainlength(mesh(b))
meshwidth(b::AbstractBSplineBasis) = meshwidth(mesh(b))

@doc raw"""
    breakpoints(b::AbstractBSplineBasis)

The `n+1` cell boundaries of the basis, held by the basis rather than asked of the mesh on
every call.

This is what keeps [`evaluate_all!`](@ref) allocation-free, `findcell` being on the innermost
loop of a particle deposition where one allocation per particle per step is the whole cost of
the routine. [`UniformMesh`](@ref) builds its vector on demand, its breakpoints being a closed
form and its own `findcell` a division that needs none of them; the other mesh families hold
theirs, and for those the basis and the mesh share one array.

As for a mesh, the result must not be mutated: it is the basis's own storage, and where the
mesh stores its breakpoints too it is the mesh's as well.
"""
breakpoints(b::AbstractBSplineBasis) = b.breaks

Base.eltype(::AbstractBSplineBasis{T}) where {T} = T
# Also on the type, so that a downstream package can compute the element type of a derived
# container from type parameters alone rather than from an instance -- which is what a
# `similar_type`-style function needs in order to stay inferable.
Base.eltype(::Type{<:AbstractBSplineBasis{T}}) where {T} = T
Base.eachindex(b::AbstractBSplineBasis) = Base.OneTo(nbasis(b))
Base.axes(b::AbstractBSplineBasis) = (Inclusion(domain(b)), eachindex(b))
"""
    grid(b::AbstractBSplineBasis)

The [`nodes`](@ref) of `b`, under the name `ContinuumArrays` uses for the points a quasi-array
is sampled at. The generic belongs to that package and is extended here rather than redefined.
"""
ContinuumArrays.grid(b::AbstractBSplineBasis) = nodes(b)

"""
    nnodes(b::AbstractBSplineBasis)

The number of [`nodes`](@ref), which for a spline basis is [`nbasis`](@ref)`(b)`: there is one
Greville abscissa per basis function.

The generic belongs to `GeometricBase` and is extended here rather than redefined.
"""
nnodes(b::AbstractBSplineBasis) = nbasis(b)

@doc raw"""
    boundary(b::AbstractBSplineBasis)

The boundary condition of `b`: [`Periodic`](@ref) for a periodic basis, otherwise the pair
`(left, right)`.
"""
function boundary end

@doc raw"""
    polynomial_reproduction(b::AbstractBSplineBasis)

The largest ``m`` such that every polynomial of degree ``\le m`` lies in the span of `b`, or
`-1` if not even the constants do.

This is the predicate that decides which moments an ``L^2`` projection preserves. Projecting a
distribution onto `b` reproduces ``\int \pi(v) f \, dv`` for a polynomial ``\pi`` exactly when
``\pi`` lies in the span, so a scheme whose mass, momentum and energy diagnostics are computed
from the projected distribution — ``1``, ``v`` and ``v^2`` — needs
`polynomial_reproduction(b) ≥ 2`.

!!! note "This is about the projection, not automatically about the dynamics"
    A conserved quantity of a *particle* scheme may be conserved by construction — for
    instance where a drift coefficient is solved from the conservation constraints themselves,
    which are conditions on particle sums and involve the basis only through ``f_s'/f_s``. What
    fails on a basis with too little reproduction is the agreement between the two
    representations: the moments of the projected ``f_s`` are then not the moments of the
    particles, and any diagnostic or entropy read off ``f_s`` describes a distribution with the
    wrong moments. Check the specific claim; do not read this number as a conservation
    guarantee on its own.

| basis | value |
|:--|:--|
| [`BSplineBasis`](@ref) | `p` — the clamped basis reproduces every polynomial it can represent |
| [`PeriodicBSplineBasis`](@ref) | `0` — a partition of unity, but ``v`` is not periodic |
| [`RecombinedBSplineBasis`](@ref) | one less than the order of the lowest derivative either condition involves — `-1` for [`Dirichlet`](@ref), since every function vanishes at the ends, `0` for [`Neumann`](@ref), `1` for [`Natural`](@ref) |

```jldoctest
julia> m = UniformMesh(16, -10 .. 10);

julia> polynomial_reproduction(BSplineBasis(m, 3))
3

julia> polynomial_reproduction(BSplineBasis(m, 3, Periodic()))
0

julia> polynomial_reproduction(BSplineBasis(m, 3, Dirichlet()))
-1

julia> polynomial_reproduction(BSplineBasis(m, 3, Natural()))
1
```

!!! warning "This is why the boundary condition is not a free choice"
    Choosing [`Dirichlet`](@ref) because the distribution function decays at the edge of the
    velocity domain removes the constants from the span, and with them the conservation of
    mass, momentum and energy that the scheme was built to have. The decay is a property of
    the solution; imposing it on the *space* is a different and much stronger statement.
"""
function polynomial_reproduction end

## ---------------------------------------------------------------------------------------
## The clamped basis
## ---------------------------------------------------------------------------------------

@doc raw"""
    BSplineBasis(mesh, p)
    BSplineBasis(mesh, p, bc)
    BSplineBasis{T}(mesh, p)

The B-spline basis of degree `p` on the [`Mesh`](@ref) `mesh`, closed at the ends according
to the [`BoundaryCondition`](@ref) `bc`.

With no `bc`, or with [`Free`](@ref), this is the *clamped* (or open) basis: the end knots
are repeated ``p+1`` times, the dimension is ``N = n + p``, and the basis is interpolatory at
both ends, ``\phi_1(a) = \phi_N(b) = 1``.

```jldoctest
julia> b = BSplineBasis(UniformMesh(8, 0 .. 1), 3);

julia> nbasis(b), degree(b), order(b)
(11, 3, 4)

julia> sum(b[0.3, j] for j in eachindex(b)) ≈ 1          # partition of unity
true

julia> b[0.0, 1], b[1.0, nbasis(b)]                      # interpolatory at both ends
(1.0, 1.0)
```

!!! note "The constructor dispatches on the boundary condition"
    `BSplineBasis(mesh, p, bc)` does not always return a `BSplineBasis`. `Periodic()` gives a
    [`PeriodicBSplineBasis`](@ref), which is a different construction with a different
    dimension, and any local constraint gives a [`RecombinedBSplineBasis`](@ref). This is
    deliberate: it lets one call site select any of the three, which is what a tensor product
    with a different condition on each axis needs.

    ```jldoctest
    julia> typeof(BSplineBasis(UniformMesh(8, 0 .. 1), 3, Periodic())).name.name
    :PeriodicBSplineBasis

    julia> typeof(BSplineBasis(UniformMesh(8, 0 .. 1), 3, Dirichlet())).name.name
    :RecombinedBSplineBasis
    ```

# Knot vector

``[\,\underbrace{a, \dots, a}_{p+1}, y_2, \dots, y_n, \underbrace{b, \dots, b}_{p+1}\,]``,
of length ``n + 2p + 1``, so that basis function `j` is supported on
`knotvector(b)[j] .. knotvector(b)[j+p+1]` and there are ``N = n + p`` of them.

Repeating the end knot ``p+1`` times drops the continuity there to ``\mathcal{C}^{-1}``,
which is what lets the basis take a nonzero value at the endpoint at all; inside, it is
``\mathcal{C}^{p-1}``.

# Requirements

`p ≥ 0`, and `n ≥ 1`. Unlike the periodic case there is no lower bound on the number of cells
in terms of the degree: a single cell carries the full polynomial space of degree `p`.

See also [`polynomial_reproduction`](@ref), which is `p` here and is what conservation
depends on, and [`evaluate_all`](@ref) for the local evaluation a particle loop needs.
"""
struct BSplineBasis{T, MT <: Mesh{T}} <: AbstractBSplineBasis{T}
    mesh::MT
    p::Int
    knots::Vector{T}
    breaks::Vector{T}

    function BSplineBasis{T}(mesh::MT, p::Integer) where {T, MT <: Mesh{T}}
        p ≥ 0 || throw(ArgumentError(
            "the degree of a B-spline basis must be non-negative, got p = $(p)"))

        y = breakpoints(mesh)
        a, b = y[begin], y[end]

        # [a ×(p+1), y_2..y_n, b ×(p+1)]. The interior breakpoints appear once, the two end
        # knots p+1 times; length n + 2p + 1, hence N = n + p basis functions.
        knots = vcat(fill(a, p + 1), y[(begin + 1):(end - 1)], fill(b, p + 1))

        new{T, MT}(mesh, Int(p), knots, y)
    end
end

BSplineBasis(mesh::Mesh{T}, p::Integer) where {T} = BSplineBasis{T}(mesh, p)

@doc raw"""
    nbasis(b::AbstractBSplineBasis)

The dimension ``N`` of the spline space — the number of basis functions, and the length a
coefficient vector must have.

It is decided by the boundary condition, not by the degree alone:

| basis | ``N`` |
|:--|:--|
| [`BSplineBasis`](@ref), clamped | ``n + p`` |
| [`PeriodicBSplineBasis`](@ref) | ``n`` |
| [`RecombinedBSplineBasis`](@ref) | ``n + p`` less one per constrained end |

The generic belongs to `CompactBasisFunctions` and is extended here rather than redefined.
"""
nbasis(b::BSplineBasis) = ncells(b) + b.p
boundary(::BSplineBasis) = (Free(), Free())
polynomial_reproduction(b::BSplineBasis) = b.p

# The knot index of the span that is cell `c`: knots[p+c] .. knots[p+c+1] == y_c .. y_{c+1}.
# The p+1 basis functions nonzero on that cell are c, c+1, ..., c+p.
_spanindex(b::BSplineBasis, c::Integer) = b.p + c
_firstindex(b::BSplineBasis, c::Integer) = c
_closeat(b::BSplineBasis) = b.knots[end]

"""
    basis_index(b::AbstractBSplineBasis, j::Integer)

The index of basis function `j`, wrapped onto `1:nbasis(b)` where the basis is periodic and
returned unchanged otherwise.

[`evaluate_all`](@ref) reports the first index of its local block *before* wrapping, because
the block is contiguous only before it. Pass each index through this function rather than
writing `mod1` at the call site: whether the wrap is needed is a property of the basis, and
getting it wrong on a non-periodic basis silently folds the two ends of the domain together.
"""
basis_index(::AbstractBSplineBasis, j::Integer) = j

@doc raw"""
    nodes(b::BSplineBasis)

The Greville abscissae of `b`,

```math
\xi_j = \frac{1}{p} \sum_{i=1}^{p} x_{j+i} ,
```

the averages of the `p` interior knots of each basis function.

These are the points a spline basis is naturally interpolated at: ``\xi_j`` lies in the
support of ``\phi_j`` and the collocation matrix ``\phi_j(\xi_i)`` is invertible
(Schoenberg-Whitney). For a clamped basis ``\xi_1 = a`` and ``\xi_N = b``.
"""
function nodes(b::BSplineBasis{T}) where {T}
    p = b.p
    p == 0 && return T[(b.knots[j] + b.knots[j + 1]) / 2 for j in 1:nbasis(b)]
    T[sum(b.knots[j + i] for i in 1:p) / p for j in 1:nbasis(b)]
end

## ---------------------------------------------------------------------------------------
## The periodic basis
## ---------------------------------------------------------------------------------------

@doc raw"""
    PeriodicBSplineBasis(mesh, p)
    PeriodicBSplineBasis{T}(mesh, p)
    PeriodicBSplineBasis(n, p; L = 2π)

The periodic B-spline basis of degree `p` on the [`Mesh`](@ref) `mesh`, spanning the spline
space ``\mathcal{S}^p_n`` of ``\mathcal{C}^{p-1}`` piecewise polynomials of degree `p` on the
torus obtained by identifying the two ends of ``[a,b]``.

```jldoctest
julia> b = PeriodicBSplineBasis(UniformMesh(8, 0 .. 1), 3);

julia> nbasis(b), degree(b), order(b)
(8, 3, 4)

julia> sum(b[0.3, j] for j in eachindex(b)) ≈ 1      # partition of unity
true
```

# Dimension

The number of degrees of freedom is ``N = n``, the number of *cells* — not ``n + p``, as it
is on a bounded interval. The construction differs from the clamped case only in how the
knot vector is closed up: instead of repeating the end knots to clamp the basis at the two
ends, the breakpoints are continued periodically, ``y_{i+n} = y_i + L``, and the recursion is
applied to the resulting bi-infinite sequence. The splines it produces satisfy
``\phi_{j+n}^p (x) = \phi_j^p (x - L)``, so only `n` of them are distinct on ``\Omega``, and
the periodic basis is obtained by wrapping these onto the torus,

```math
\phi_j^p \big\vert_\Omega (x) = \sum_{r \in \mathbb{Z}} \phi_j^p (x + rL) ,
\qquad 1 \le j \le n ,
```

a finite sum, since ``\phi_j^p`` is supported on the `p+1` cells
``[y_j, y_{j+p+1}]``. The `p` extra functions of the clamped case are precisely those the
clamping introduces at the two ends, and the wrapping identifies them in pairs.

# Why this matters

There are no boundary functions at all: every basis function spans `p+1` cells, and the
basis is ``\mathcal{C}^{p-1}`` across the seam ``a \equiv b`` as well as inside. Integration by
parts over ``\Omega`` therefore leaves no boundary terms, which is what makes the discrete
brackets assembled from this basis exactly antisymmetric.

On a [`UniformMesh`](@ref) the basis functions are in addition translates of a single
cardinal B-spline, and the mass, stiffness and derivative matrices are circulant.

What is lost is polynomial reproduction beyond the constants: ``v`` and ``v^2`` are not
periodic, so a scheme that conserves momentum or energy because they lie in the span of the
basis does not do so here. See [`polynomial_reproduction`](@ref).

# Requirements

`p ≥ 0` and `n > p`. The second is what makes the wrapping well defined: a basis function
spans `p+1` cells, so with `n ≤ p` it would wrap onto itself and the sum above would not
terminate.

See also [`SplineQuadrature`](@ref) for the assembly built on this basis, and
[`evaluate`](@ref) for derivatives of arbitrary order.
"""
struct PeriodicBSplineBasis{T, MT <: Mesh{T}} <: AbstractBSplineBasis{T}
    mesh::MT
    p::Int
    knots::Vector{T}
    breaks::Vector{T}
    offset::Int

    function PeriodicBSplineBasis{T}(mesh::MT, p::Integer) where {T, MT <: Mesh{T}}
        p ≥ 0 || throw(ArgumentError(
            "the degree of a B-spline basis must be non-negative, got p = $(p)"))

        n = ncells(mesh)

        # A basis function spans p+1 cells. With n ≤ p it would wrap onto itself, the sum
        # over the periodic images would not terminate, and the dimension count N = n would
        # be wrong. The check is here rather than at the wrap because this is where the
        # offending pair (n, p) is still in hand.
        n > p || throw(ArgumentError(
            "a periodic B-spline basis of degree $(p) needs more than $(p) cells, " *
            "got n = $(n); refine the mesh or lower the degree"))

        L = domainlength(mesh)

        breaks = breakpoints(mesh)

        # The n distinct breakpoints, dropping the right endpoint: on a torus y_{n+1} is the
        # periodic image of y_1 and not a breakpoint of its own.
        y = breaks[begin:(end - 1)]

        # Five periodic images of the breakpoints. The recursion for basis function j
        # reaches from knot 2n+j to knot 2n+j+p+1 ≤ 3n+p+1, and evaluation shifts the
        # argument by ±L, so the two outer blocks are what keeps every index in range.
        knots = vcat(y .- 2L, y .- L, y, y .+ L, y .+ 2L)

        new{T, MT}(mesh, Int(p), knots, breaks, 2n)
    end
end

PeriodicBSplineBasis(mesh::Mesh{T}, p::Integer) where {T} = PeriodicBSplineBasis{T}(mesh, p)
function PeriodicBSplineBasis(n::Integer, p::Integer; L = 2π)
    PeriodicBSplineBasis(UniformMesh(n, L), p)
end

nbasis(b::PeriodicBSplineBasis) = ncells(b.mesh)
boundary(::PeriodicBSplineBasis) = Periodic()
polynomial_reproduction(::PeriodicBSplineBasis) = 0

_spanindex(b::PeriodicBSplineBasis, c::Integer) = b.offset + c
_firstindex(b::PeriodicBSplineBasis, c::Integer) = c - b.p
# The knot vector runs beyond the domain in both directions, so there is no topmost span to
# close: x is reduced onto [a,b) before evaluation and never reaches a knot vector end.
_closeat(::PeriodicBSplineBasis) = nothing

basis_index(b::PeriodicBSplineBasis, j::Integer) = mod1(j, nbasis(b))

@doc raw"""
    nodes(b::PeriodicBSplineBasis)

The Greville abscissae of `b`, reduced onto ``[a,b)``,

```math
\xi_j = \frac{1}{p} \sum_{i=1}^{p} x_{j+i} ,
```

the averages of the `p` interior knots of each basis function.
"""
function nodes(b::PeriodicBSplineBasis{T}) where {T}
    a = leftendpoint(domain(b))
    L = domainlength(b)
    p = b.p
    p == 0 && return T[a + mod(_cellcentre(b, j) - a, L) for j in 1:nbasis(b)]
    T[a + mod(sum(b.knots[b.offset + j + i] for i in 1:p) / p - a, L)
      for j in 1:nbasis(b)]
end

function _cellcentre(b::PeriodicBSplineBasis, j::Integer)
    (b.knots[b.offset + j] + b.knots[b.offset + j + 1]) / 2
end

## ---------------------------------------------------------------------------------------
## Evaluation
## ---------------------------------------------------------------------------------------

@doc raw"""
    evaluate(b::AbstractBSplineBasis, j, x, d = 0)
    evaluate(b::AbstractBSplineBasis, û::AbstractVector, x, d = 0)

The `d`-th derivative of the `j`-th basis function of `b` at `x`, or of the spline
``u_h = \sum_j \hat{u}_j \phi_j`` with coefficients `û`.

`x` may also be a vector, in which case a vector is returned.

The explicit derivative order is what the lazy `b'` products cannot give: the discrete
brackets need `d` up to `3`, and stacking `Derivative` that deep does not compose.

```jldoctest
julia> b = BSplineBasis(UniformMesh(8, 0 .. 1), 3);

julia> evaluate(b, 1, 0.1) ≈ b[0.1, 1]
true

julia> evaluate(b, nbasis(b), 1.0)         # interpolatory at the right endpoint
1.0
```

On a [`PeriodicBSplineBasis`](@ref) any real `x` is accepted and reduced onto the domain
first, so the result is the periodic extension. On the other two, `x` outside the domain
gives zero — which is the mathematically correct value for a compactly supported basis, and
is worth being aware of in a particle method, where a particle that leaves the domain
silently stops contributing rather than raising an error.

This is the reference path, one basis function at a time, and it runs the recursion as written
rather than through a faster equivalent: one value costs ``O(2^p)``. A loop that needs every
nonzero function at a point — a particle deposition, or a matrix assembly — should use
[`evaluate_all`](@ref) instead, which is ``O(p^2)`` for the whole block of `p+1` together. The
two are within a factor of two at ``p = 3`` and a factor of 150 apart at ``p = 12``.
"""
function evaluate(b::BSplineBasis{T}, j::Integer, x::Number, d::Integer = 0) where {T}
    @boundscheck (1 ≤ j ≤ nbasis(b)) || throw(BoundsError(b, j))
    d ≥ 0 || throw(ArgumentError("the derivative order must be non-negative, got d = $(d)"))
    _bspline(b.knots, Int(j), b.p, x, Int(d), _closeat(b))
end

function evaluate(b::PeriodicBSplineBasis{T}, j::Integer, x::Number,
        d::Integer = 0) where {T}
    @boundscheck (1 ≤ j ≤ nbasis(b)) || throw(BoundsError(b, j))
    d ≥ 0 || throw(ArgumentError("the derivative order must be non-negative, got d = $(d)"))

    a = leftendpoint(domain(b))
    L = domainlength(b)
    R = _evaltype(T, typeof(x))

    # Reduce onto [a,a+L) first, so that three images always suffice: a basis function is
    # supported inside a window of width 2L before wrapping, and x̃ reaches it through the
    # shifts 0 and +L. The shift -L is kept because it costs nothing and makes the sum the
    # one written in the definition rather than a truncation of it that happens to be exact.
    x̃ = a + mod(x - a, L)
    k = b.offset + j

    v = zero(R)
    for shift in (-L, zero(L), L)
        v += _bspline(b.knots, k, b.p, x̃ + shift, Int(d), nothing)
    end
    return v
end

function evaluate(b::AbstractBSplineBasis, j::Integer, X::AbstractVector, d::Integer = 0)
    [evaluate(b, j, x, d) for x in X]
end

function evaluate(b::AbstractBSplineBasis{T}, û::AbstractVector{S}, x::Number,
        d::Integer = 0) where {T, S}
    length(û) == nbasis(b) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the basis has $(nbasis(b))"))
    R = _evaltype(promote_type(T, S), typeof(x))
    _evaluate_block!(Vector{R}(undef, local_width(b)), b, û, x, d)
end

# The local-block sum, against a caller-supplied buffer. Only the block contributes, so this
# is O(p) per point where summing `evaluate` over `eachindex(û)` is O(N).
#
# Split out so that a sweep over many points allocates one buffer for the sweep rather than
# one per point. The buffer is the whole of the difference between this and BSplineKit at
# small `n`, which reaches the block through the spline order carried in its *type* and can
# therefore keep it on the stack; here the degree is a field, so the length is a run-time
# value. Passing a `Val`-sized `MVector` instead was measured and does not help -- it halves
# the allocation and returns the saving in dispatch, because `evaluate_all!` takes an
# `AbstractVector` and the buffer escapes into it either way.
function _evaluate_block!(values::AbstractVector{R}, b::AbstractBSplineBasis,
        û::AbstractVector, x::Number, d::Integer) where {R}
    j₀ = evaluate_all!(values, b, x, d)

    v = zero(R)
    for t in eachindex(values)
        j = basis_index(b, j₀ + t - 1)
        # A periodic axis wraps, so every index is in range; a bounded one does not, and an
        # index outside 1:N means the block of this cell reaches past the end of the basis.
        # Such a term is genuinely absent rather than zero-valued, so it is skipped.
        1 ≤ j ≤ nbasis(b) || continue
        v += values[t] * û[j]
    end
    return v
end

function evaluate(b::AbstractBSplineBasis{T}, û::AbstractVector{S}, X::AbstractVector,
        d::Integer = 0) where {T, S}
    length(û) == nbasis(b) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the basis has $(nbasis(b))"))
    R = _evaltype(promote_type(T, S), eltype(X))

    # One buffer for the whole sweep. Evaluating point by point through the scalar method
    # takes one per point, which for a plot or a diagnostic over a fine grid is the entire
    # allocation of the call.
    values = Vector{R}(undef, local_width(b))
    out = Vector{R}(undef, length(X))
    # `enumerate`, not `pairs`: the counter has to index `out`, whose axes are `1:length(X)`
    # whatever the axes of `X` are. `pairs` yields the keys of `X`, which for an
    # offset-axis vector are not indices of `out` at all.
    for (i, x) in enumerate(X)
        out[i] = _evaluate_block!(values, b, û, x, d)
    end
    return out
end

(b::AbstractBSplineBasis)(x::Number, j::Integer) = evaluate(b, j, x, 0)

Base.getindex(b::AbstractBSplineBasis, x::Number, j::Integer) = evaluate(b, j, x, 0)
function Base.getindex(b::AbstractBSplineBasis, x::Number, ::Colon)
    [evaluate(b, j, x, 0) for j in eachindex(b)]
end
function Base.getindex(b::AbstractBSplineBasis, X::AbstractVector, j::Integer)
    [evaluate(b, j, x, 0) for x in X]
end
function Base.getindex(b::AbstractBSplineBasis, X::AbstractVector, ::Colon)
    [evaluate(b, j, x, 0) for x in X, j in eachindex(b)]
end

## ---------------------------------------------------------------------------------------
## Local evaluation
## ---------------------------------------------------------------------------------------

# The shared implementation. `findcell` itself carries the docstring, below: a `@doc` block
# here would document *this* function instead, leaving the exported name undocumented.
function _findcell(y::AbstractVector, n::Integer, x::Number)
    x ≤ y[begin] && return 1
    x ≥ y[end] && return n
    # searchsortedlast gives the k with y[k] ≤ x < y[k+1]; both ends are handled above, so
    # the result is already in 1:n and needs no clamping.
    return searchsortedlast(y, x)
end

@doc raw"""
    findcell(b::AbstractBSplineBasis, x)
    findcell(m::Mesh, x)

The index of the cell containing `x`, in `1:ncells`.

The right endpoint `b` belongs to the last cell, not to a cell of its own — the cells are
half-open except for the last, which is closed. A point outside the domain is clamped to the
nearest cell for a bounded mesh; on a [`PeriodicBSplineBasis`](@ref) reduce `x` onto the
domain first.

Called on a basis rather than on a mesh it allocates nothing, the breakpoints being cached in
the basis; on a [`UniformMesh`](@ref) it is a division that does not read them at all.
"""
findcell(m::Mesh, x::Number) = _findcell(breakpoints(m), ncells(m), x)

# On a uniform mesh the cell index is a division, with no need for the breakpoint vector at
# all. Kept as its own method because it is the one a particle loop actually takes.
function findcell(m::UniformMesh, x::Number)
    n = ncells(m)
    t = (x - m.a) / (m.b - m.a)
    t ≤ 0 && return 1
    t ≥ 1 && return n
    return min(floor(Int, t * n) + 1, n)
end

# Reads the breakpoints cached in the basis, so this allocates nothing.
findcell(b::AbstractBSplineBasis, x::Number) = _findcell(breakpoints(b), ncells(b), x)
findcell(b::BSplineBasis{T, <:UniformMesh}, x::Number) where {T} = findcell(mesh(b), x)
function findcell(b::PeriodicBSplineBasis{T, <:UniformMesh}, x::Number) where {T}
    findcell(mesh(b), x)
end

@doc raw"""
    evaluate_all(b::AbstractBSplineBasis, x, d = 0)
    evaluate_all!(values, b::AbstractBSplineBasis, x, d = 0)

The `d`-th derivatives at `x` of the `p+1` basis functions that do not vanish there,
returned as `(j₀, values)` — or, for the in-place form, written into `values` with `j₀`
returned.

`values` must have `p+1` entries; `values[t]` is the derivative of basis function
`basis_index(b, j₀ + t - 1)`.

```jldoctest
julia> b = BSplineBasis(UniformMesh(8, 0 .. 1), 3);

julia> j₀, v = evaluate_all(b, 0.3);

julia> j₀, length(v)
(3, 4)

julia> all(v[t] ≈ evaluate(b, j₀ + t - 1, 0.3) for t in eachindex(v))
true

julia> sum(v) ≈ 1                     # the other N - p - 1 functions vanish at 0.3
true
```

This is the routine a particle method needs. Depositing ``N_p`` particles onto the basis costs
``O(N_p \, p^2)`` through this and ``O(N_p \, N \, 2^p)`` through [`evaluate`](@ref) one index
at a time — the difference between a loop over the `p+1` functions that actually overlap the
particle and a loop over the whole basis, and, within each term of it, between the triangular
scheme and the unmemoised recursion.

!!! note "The index may need wrapping"
    `j₀ + t - 1` is the index *before* wrapping, and the block is contiguous only in that
    form. On a [`PeriodicBSplineBasis`](@ref) it can fall outside `1:N` at either end; pass
    it through [`basis_index`](@ref), which wraps where the basis is periodic and is the
    identity where it is not.

Outside the domain of a bounded basis `values` is filled with zeros, matching [`evaluate`](@ref)
one index at a time, so the deposition above adds nothing for a particle that has left the
domain rather than depositing an extrapolated polynomial. On a
[`PeriodicBSplineBasis`](@ref) every real `x` is reduced onto the domain first and no point is
outside.

# Method

De Boor's triangular scheme evaluates the `q+1` nonzero splines of degree ``q = p - d`` on
the span containing `x` in ``O(q^2)``, and the derivative recursion

```math
D^{m+1} \phi_j^r = r \left( \frac{D^m \phi_j^{r-1}}{x_{j+r} - x_j}
                          - \frac{D^m \phi_{j+1}^{r-1}}{x_{j+r+1} - x_{j+1}} \right)
```

then lifts the block from degree ``p-d`` to degree ``p``, one order at a time, the block
growing by one function at each step. The internal `_bspline` computes the same numbers from the
recursion written out as it stands, and the test suite checks the two against each other at
every degree and derivative order.
"""
function evaluate_all! end

function evaluate_all!(values::AbstractVector{R}, b::AbstractBSplineBasis, x::Number,
        d::Integer = 0) where {R}
    p = degree(b)
    length(values) == p + 1 || throw(DimensionMismatch(
        "evaluate_all! needs a buffer of $(p + 1) entries for a degree-$(p) basis, got " *
        "$(length(values))"))
    d ≥ 0 || throw(ArgumentError("the derivative order must be non-negative, got d = $(d)"))

    x̃ = _reduce_argument(b, x)
    c = findcell(b, x̃)
    k = _spanindex(b, c)
    kv = knotvector(b)

    if !_inside(b, x)
        # Outside the domain of a bounded basis every basis function vanishes, so the whole
        # block is zero. The local path cannot discover that from `c` alone: `findcell`
        # clamps to the nearest cell, and de Boor's scheme below would then evaluate that
        # cell's polynomial at a point outside it -- an extrapolation, not the value. A
        # periodic basis reduces its argument, so for it every real `x` is inside.
        #
        # The cell index is still the clamped one, so a deposition loop indexes the same
        # block it would for a point just inside and adds zero to it.
        fill!(values, zero(R))
        return _firstindex(b, c)
    end

    if d > p
        # Every derivative above the degree vanishes identically. Returning zeros rather
        # than erroring keeps a loop over derivative orders from needing a special case.
        fill!(values, zero(R))
        return _firstindex(b, c)
    end

    q = p - Int(d)

    # De Boor's triangular scheme for the q+1 nonzero splines of degree q on span k.
    # values[1:q+1] holds φ_{k-q}^q .. φ_k^q afterwards.
    values[1] = one(R)
    for j in 1:q
        saved = zero(R)
        for r in 1:j
            # right = kv[k+r] - x, left = kv[k+1-(j-r+1)] - ... written out below to keep
            # the index arithmetic of the standard form visible.
            right = kv[k + r] - x̃
            left = x̃ - kv[k + 1 - j + r - 1]
            temp = values[r] / (right + left)
            values[r] = saved + right * temp
            saved = left * temp
        end
        values[j + 1] = saved
    end

    # Lift from degree q to degree p by applying the derivative recursion d times. At the
    # start of each step `count` entries are valid, holding D^m φ_j^r for
    # j = k-count+1 .. k; afterwards there is one more, for j = k-count .. k.
    count = q + 1
    for _ in 1:Int(d)
        r = count            # the degree being stepped up to, p-d+m+1
        # Descending, so that values[t-1] and values[t] are still the previous step's when
        # they are read. values[count+1] is the new bottom entry and reads a zero above it.
        for t in (count + 1):-1:1
            below = t ≥ 2 ? values[t - 1] : zero(R)
            above = t ≤ count ? values[t] : zero(R)
            j = k - count - 1 + t
            da = kv[j + r] - kv[j]
            db = kv[j + r + 1] - kv[j + 1]
            out = zero(R)
            da > 0 && (out += below / da)
            db > 0 && (out -= above / db)
            values[t] = r * out
        end
        count += 1
    end

    return _firstindex(b, c)
end

"""
    evaluate_all(b::AbstractBSplineBasis, x, d = 0)

The allocating form of [`evaluate_all!`](@ref): returns `(j₀, values)` with a freshly allocated
buffer of [`local_width`](@ref) entries.

Convenient at a call site that runs once. A particle loop should hold its own buffer and call
[`evaluate_all!`](@ref), which allocates nothing.
"""
function evaluate_all(b::AbstractBSplineBasis{T}, x::Number, d::Integer = 0) where {T}
    R = _evaltype(T, typeof(x))
    # `local_width`, not `degree(b) + 1`: a recombined basis has a wider block at the two
    # ends, because a recombined function there spans the union of two parent supports.
    values = Vector{R}(undef, local_width(b))
    j₀ = evaluate_all!(values, b, x, d)
    return (j₀, values)
end

# A periodic basis accepts any real argument and reduces it; a bounded one does not, and
# clamping here would be wrong -- a point outside the domain must evaluate to zero, which
# `findcell` plus the half-open knot spans already gives.
_reduce_argument(::BSplineBasis, x) = x
function _reduce_argument(b::PeriodicBSplineBasis, x)
    a = leftendpoint(domain(b))
    a + mod(x - a, domainlength(b))
end

# Whether a point carries any of the basis at all. A periodic basis reduces its argument, so
# it always does; a bounded one is zero outside its closed domain.
_inside(::PeriodicBSplineBasis, x) = true
function _inside(b::AbstractBSplineBasis, x)
    leftendpoint(domain(b)) ≤ x ≤ rightendpoint(domain(b))
end

## ---------------------------------------------------------------------------------------
## Comparison
## ---------------------------------------------------------------------------------------

function Base.hash(b::AbstractBSplineBasis, h::UInt)
    hash(mesh(b), hash(degree(b), hash(boundary(b), h)))
end
function Base.:(==)(b1::AbstractBSplineBasis, b2::AbstractBSplineBasis)
    (typeof(b1) == typeof(b2) && degree(b1) == degree(b2) && mesh(b1) == mesh(b2) &&
     boundary(b1) == boundary(b2))
end
function Base.isequal(b1::AbstractBSplineBasis{T1}, b2::AbstractBSplineBasis{T2}) where {
        T1, T2}
    (T1 == T2 && b1 == b2)
end
function Base.isapprox(b1::AbstractBSplineBasis, b2::AbstractBSplineBasis; kwargs...)
    (typeof(b1) == typeof(b2) && degree(b1) == degree(b2) &&
     boundary(b1) == boundary(b2) && isapprox(mesh(b1), mesh(b2); kwargs...))
end

## ---------------------------------------------------------------------------------------
## Derivative
## ---------------------------------------------------------------------------------------

@simplify *(D::Derivative, b::AbstractBSplineBasis) = Mul(D, b)

"""
    BSplineDerivative

The type of `Derivative(axes(b,1)) * b` for an [`AbstractBSplineBasis`](@ref) `b`,
equivalently of `b'`.

A lazy product: it stores the basis and runs the derivative recursion on indexing. For
derivatives of order higher than one use [`evaluate`](@ref) with an explicit `d`.
"""
const BSplineDerivative = QMul2{<:Derivative, <:AbstractBSplineBasis}

Base.getindex(D::BSplineDerivative, x::Number, j::Integer) = evaluate(D.B, j, x, 1)
function Base.getindex(D::BSplineDerivative, x::Number, ::Colon)
    [evaluate(D.B, j, x, 1) for j in eachindex(D.B)]
end
function Base.getindex(D::BSplineDerivative, X::AbstractVector, j::Integer)
    [evaluate(D.B, j, x, 1) for x in X]
end
function Base.getindex(D::BSplineDerivative, X::AbstractVector, ::Colon)
    [evaluate(D.B, j, x, 1) for x in X, j in eachindex(D.B)]
end

Base.adjoint(b::AbstractBSplineBasis) = Derivative(axes(b, 1)) * b

"""
    PeriodicBSplineDerivative

The [`BSplineDerivative`](@ref) of a [`PeriodicBSplineBasis`](@ref) specifically.

Retained so that code written against the periodic-only version of this package keeps
resolving; new code should dispatch on [`BSplineDerivative`](@ref), which covers all three
bases.
"""
const PeriodicBSplineDerivative = QMul2{<:Derivative, <:PeriodicBSplineBasis}
