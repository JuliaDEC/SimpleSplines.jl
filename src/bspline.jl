
@inline _evaltype(::Type{T}, ::Type{S}) where {T, S} = promote_type(T, S)

@doc raw"""
    _bspline(kv, k, p, x, d)

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
built on them doubles as a check of the formulae themselves.

A knot span of zero length — a repeated knot — contributes nothing, which is what makes the
recursion well defined at a knot of multiplicity greater than one. The guard is on the span
being positive rather than on the numerator, because it is the division that fails.
"""
function _bspline(kv::AbstractVector{T}, k::Int, p::Int, x::S, d::Int) where {T, S}
    R = _evaltype(T, S)

    if d > 0
        p == 0 && return zero(R)
        a = kv[k + p] - kv[k]
        b = kv[k + p + 1] - kv[k + 1]
        out = zero(R)
        a > 0 && (out += _bspline(kv, k, p - 1, x, d - 1) / a)
        b > 0 && (out -= _bspline(kv, k + 1, p - 1, x, d - 1) / b)
        return p * out
    end

    if p == 0
        return (kv[k] ≤ x < kv[k + 1]) ? one(R) : zero(R)
    end

    a = kv[k + p] - kv[k]
    b = kv[k + p + 1] - kv[k + 1]
    out = zero(R)
    a > 0 && (out += (x - kv[k]) / a * _bspline(kv, k, p - 1, x, 0))
    b > 0 && (out += (kv[k + p + 1] - x) / b * _bspline(kv, k + 1, p - 1, x, 0))
    return out
end

@doc raw"""
    PeriodicBSplineBasis(mesh, p)
    PeriodicBSplineBasis{T}(mesh, p)
    PeriodicBSplineBasis(n, p; L = 2π)

The periodic B-spline basis of degree `p` on the [`Mesh`](@ref) `mesh`, spanning the spline
space ``\mathcal{S}^p_n`` of ``\mathcal{C}^{p-1}`` piecewise polynomials of degree `p` on the
torus ``\Omega = [0,L)``.

```jldoctest
julia> b = PeriodicBSplineBasis(UniformMesh(8, 1.0), 3);

julia> nbasis(b), degree(b), order(b)
(8, 3, 4)

julia> sum(b[0.3, j] for j in eachindex(b)) ≈ 1      # partition of unity
true
```

# Dimension

The number of degrees of freedom is ``N = n``, the number of *cells* — not ``n + p``, as it
would be on a bounded interval. The construction differs from the bounded case only in how
the knot vector is closed up: instead of repeating the end knots to clamp the basis at the
two ends, the breakpoints are continued periodically, ``y_{i+n} = y_i + L``, and the
recursion is applied to the resulting bi-infinite sequence. The splines it produces satisfy
``\phi_{j+n}^p (x) = \phi_j^p (x - L)``, so only `n` of them are distinct on ``\Omega``, and
the periodic basis is obtained by wrapping these onto the torus,

```math
\phi_j^p \big\vert_\Omega (x) = \sum_{r \in \mathbb{Z}} \phi_j^p (x + rL) ,
\qquad 1 \le j \le n ,
```

a finite sum, since ``\phi_j^p`` is supported on the `p+1` cells
``[y_j, y_{j+p+1}]``. The `p` extra functions of the bounded case are precisely those the
clamping introduces at the two ends, and the wrapping identifies them in pairs.

# Why this matters

There are no boundary functions at all: every basis function spans `p+1` cells, and the
basis is ``\mathcal{C}^{p-1}`` across the seam ``a \equiv b`` as well as inside. Integration by
parts over ``\Omega`` therefore leaves no boundary terms, which is what makes the discrete
brackets assembled from this basis exactly antisymmetric.

On a [`UniformMesh`](@ref) the basis functions are in addition translates of a single
cardinal B-spline, and the mass, stiffness and derivative matrices are circulant.

# Requirements

`p ≥ 0` and `n > p`. The second is what makes the wrapping well defined: a basis function
spans `p+1` cells, so with `n ≤ p` it would wrap onto itself and the sum above would not
terminate.

See also [`SplineQuadrature`](@ref) for the assembly built on this basis, and
[`evaluate`](@ref) for derivatives of arbitrary order.
"""
struct PeriodicBSplineBasis{T, MT <: Mesh{T}} <: Basis{T}
    mesh::MT
    p::Int
    knots::Vector{T}
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
        y = breakpoints(mesh)

        # Five periodic images of the breakpoints. The recursion for basis function j
        # reaches from knot 2n+j to knot 2n+j+p+1 ≤ 3n+p+1, and evaluation shifts the
        # argument by ±L, so the two outer blocks are what keeps every index in range.
        knots = vcat(y .- 2L, y .- L, y, y .+ L, y .+ 2L)

        new{T, MT}(mesh, p, knots, 2n)
    end
end

PeriodicBSplineBasis(mesh::Mesh{T}, p::Integer) where {T} = PeriodicBSplineBasis{T}(mesh, p)
function PeriodicBSplineBasis(n::Integer, p::Integer; L = 2π)
    PeriodicBSplineBasis(UniformMesh(n, L), p)
end

"""
    knotvector(b::PeriodicBSplineBasis)

The periodically extended knot sequence the recursion runs on.

Five images of the breakpoints, `[y .- 2L; y .- L; y; y .+ L; y .+ 2L]`. Basis function `j`
is supported on `knotvector(b)[2n+j] .. knotvector(b)[2n+j+p+1]` before wrapping.
"""
knotvector(b::PeriodicBSplineBasis) = b.knots

mesh(b::PeriodicBSplineBasis) = b.mesh
ncells(b::PeriodicBSplineBasis) = ncells(b.mesh)
domainlength(b::PeriodicBSplineBasis) = domainlength(b.mesh)
breakpoints(b::PeriodicBSplineBasis) = breakpoints(b.mesh)
cellbounds(b::PeriodicBSplineBasis) = cellbounds(b.mesh)
meshwidth(b::PeriodicBSplineBasis) = meshwidth(b.mesh)

nbasis(b::PeriodicBSplineBasis) = ncells(b.mesh)
degree(b::PeriodicBSplineBasis) = b.p

@doc raw"""
    order(b::PeriodicBSplineBasis)

The order ``k = p + 1`` of the spline basis `b`.

Note that this is the *spline* meaning of the word — a B-spline of order `k` is piecewise of
degree `k-1`, and the knot vector of a basis of `N` functions has `N + k` entries. It is not
the meaning `order` carries for the bases of `CompactBasisFunctions`, where it is the number
of basis functions. For a spline basis the two differ: `order(b) == degree(b) + 1`, while
`nbasis(b)` is the number of cells.
"""
order(b::PeriodicBSplineBasis) = b.p + 1

@doc raw"""
    nodes(b::PeriodicBSplineBasis)

The Greville abscissae of `b`, reduced onto ``[0,L)``,

```math
\xi_j = \frac{1}{p} \sum_{i=1}^{p} x_{j+i} ,
```

the averages of the `p` interior knots of each basis function.

These are the points a spline basis is naturally interpolated at: ``\xi_j`` lies in the
support of ``\phi_j`` and the collocation matrix ``\phi_j(\xi_i)`` is invertible
(Schoenberg-Whitney). For `p = 1` they are the breakpoints themselves.
"""
function nodes(b::PeriodicBSplineBasis{T}) where {T}
    L = domainlength(b)
    p = b.p
    p == 0 && return T[mod(_cellcentre(b, j), L) for j in 1:nbasis(b)]
    T[mod(sum(b.knots[b.offset + j + i] for i in 1:p) / p, L) for j in 1:nbasis(b)]
end

function _cellcentre(b::PeriodicBSplineBasis, j::Integer)
    (b.knots[b.offset + j] + b.knots[b.offset + j + 1]) / 2
end

nnodes(b::PeriodicBSplineBasis) = nbasis(b)

@doc raw"""
    evaluate(b::PeriodicBSplineBasis, j, x, d = 0)
    evaluate(b::PeriodicBSplineBasis, û::AbstractVector, x, d = 0)

The `d`-th derivative of the `j`-th basis function of `b` at `x`, or of the spline
``u_h = \sum_j \hat{u}_j \phi_j`` with coefficients `û`.

`x` may be any real number: it is reduced onto ``[0,L)`` first, so the result is the
periodic extension. `x` may also be a vector, in which case a vector is returned.

The explicit derivative order is what the lazy `b'` products cannot give: the discrete
brackets need `d` up to `3`, and stacking `Derivative` that deep does not compose.

```jldoctest
julia> b = PeriodicBSplineBasis(UniformMesh(8, 1.0), 3);

julia> evaluate(b, 1, 0.1) ≈ b[0.1, 1]
true

julia> abs(evaluate(b, 1, 1.1) - evaluate(b, 1, 0.1)) < 1e-14     # periodic
true
```
"""
function evaluate(b::PeriodicBSplineBasis{T}, j::Integer, x::Number, d::Integer = 0) where {T}
    @boundscheck (1 ≤ j ≤ nbasis(b)) || throw(BoundsError(b, j))
    d ≥ 0 || throw(ArgumentError("the derivative order must be non-negative, got d = $(d)"))

    L = domainlength(b)
    R = _evaltype(T, typeof(x))

    # Reduce onto [0,L) first, so that three images always suffice: a basis function is
    # supported inside [0,2L) before wrapping, and x̃ ∈ [0,L) reaches it through the shifts
    # 0 and +L. The shift -L is kept because it costs nothing and makes the sum the one
    # written in the definition rather than a truncation of it that happens to be exact.
    x̃ = mod(x, L)
    k = b.offset + j

    v = zero(R)
    for shift in (-L, zero(L), L)
        v += _bspline(b.knots, k, b.p, x̃ + shift, Int(d))
    end
    return v
end

function evaluate(b::PeriodicBSplineBasis, j::Integer, X::AbstractVector, d::Integer = 0)
    [evaluate(b, j, x, d) for x in X]
end

function evaluate(b::PeriodicBSplineBasis{T}, û::AbstractVector, x::Number,
        d::Integer = 0) where {T}
    length(û) == nbasis(b) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the basis has $(nbasis(b))"))
    sum(û[j] * evaluate(b, j, x, d) for j in eachindex(û))
end

function evaluate(b::PeriodicBSplineBasis, û::AbstractVector, X::AbstractVector, d::Integer = 0)
    [evaluate(b, û, x, d) for x in X]
end

(b::PeriodicBSplineBasis)(x::Number, j::Integer) = evaluate(b, j, x, 0)

Base.eltype(::PeriodicBSplineBasis{T}) where {T} = T
Base.eachindex(b::PeriodicBSplineBasis) = Base.OneTo(nbasis(b))
Base.axes(b::PeriodicBSplineBasis) = (Inclusion(0 .. domainlength(b)), eachindex(b))
ContinuumArrays.grid(b::PeriodicBSplineBasis) = nodes(b)

Base.hash(b::PeriodicBSplineBasis, h::UInt) = hash(b.mesh, hash(b.p, h))
function Base.:(==)(b1::PeriodicBSplineBasis, b2::PeriodicBSplineBasis)
    (b1.p == b2.p && b1.mesh == b2.mesh)
end
function Base.isequal(b1::PeriodicBSplineBasis{T1}, b2::PeriodicBSplineBasis{T2}) where {
        T1, T2}
    (T1 == T2 && b1 == b2)
end
function Base.isapprox(b1::PeriodicBSplineBasis, b2::PeriodicBSplineBasis; kwargs...)
    (b1.p == b2.p && isapprox(b1.mesh, b2.mesh; kwargs...))
end

Base.getindex(b::PeriodicBSplineBasis, x::Number, j::Integer) = evaluate(b, j, x, 0)
function Base.getindex(b::PeriodicBSplineBasis, x::Number, ::Colon)
    [evaluate(b, j, x, 0) for j in eachindex(b)]
end
function Base.getindex(b::PeriodicBSplineBasis, X::AbstractVector, j::Integer)
    [evaluate(b, j, x, 0) for x in X]
end
function Base.getindex(b::PeriodicBSplineBasis, X::AbstractVector, ::Colon)
    [evaluate(b, j, x, 0) for x in X, j in eachindex(b)]
end

## Derivative

@simplify *(D::Derivative, b::PeriodicBSplineBasis) = Mul(D, b)

"""
    PeriodicBSplineDerivative

The type of `Derivative(axes(b,1)) * b` for a [`PeriodicBSplineBasis`](@ref) `b`,
equivalently of `b'`.

A lazy product: it stores the basis and runs the derivative recursion on indexing. For
derivatives of order higher than one use [`evaluate`](@ref) with an explicit `d`.
"""
const PeriodicBSplineDerivative = QMul2{<:Derivative, <:PeriodicBSplineBasis}

Base.getindex(D::PeriodicBSplineDerivative, x::Number, j::Integer) = evaluate(D.B, j, x, 1)
function Base.getindex(D::PeriodicBSplineDerivative, x::Number, ::Colon)
    [evaluate(D.B, j, x, 1) for j in eachindex(D.B)]
end
function Base.getindex(D::PeriodicBSplineDerivative, X::AbstractVector, j::Integer)
    [evaluate(D.B, j, x, 1) for x in X]
end
function Base.getindex(D::PeriodicBSplineDerivative, X::AbstractVector, ::Colon)
    [evaluate(D.B, j, x, 1) for x in X, j in eachindex(D.B)]
end

Base.adjoint(b::PeriodicBSplineBasis) = Derivative(axes(b, 1)) * b
