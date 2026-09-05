
@doc raw"""
    Spline(basis, coefficients)

The spline function ``u_h = \sum_I \hat{u}_I \, \phi_I``: a basis together with the
coefficients of one element of its span, callable at a point.

The basis may be one-dimensional — any [`AbstractBSplineBasis`](@ref), with a vector of
coefficients — or a [`TensorProductBasis`](@ref), with a `D`-dimensional array:

```jldoctest
julia> b = BSplineBasis(UniformMesh(8, 0 .. 1), 3);

julia> s = Spline(b, l2_projection(SplineQuadrature(b), x -> x^2));

julia> s(0.3) ≈ 0.09, s(0.3, 1) ≈ 0.6            # value and first derivative
(true, true)

julia> B = b ⊗ BSplineBasis(UniformMesh(6, 0 .. 2), 2);

julia> S = Spline(B, l2_projection(TensorProductQuadrature(B), x -> x[1]^2 * x[2]));

julia> S((0.3, 1.4)) ≈ 0.09 * 1.4
true
```

# Why this exists separately from the basis

A basis answers "what is ``\phi_j`` here"; a `Spline` answers "what is ``u_h`` here". Keeping
them apart is what lets a coefficient array be updated in place — a particle deposition
rewrites the coefficients every step while the basis, the quadrature and the mass
factorisation stay put — and it is why [`coefficients`](@ref) returns the array itself rather
than a copy. A `Spline` built on an array shares it, so a projection written into that array
is visible through the spline with no rebuild.

The derivative is reached either as a second argument, `s(x, d)`, or as a callable of its own
through [`derivative`](@ref). The second form is what a right-hand side wants when the
derivative appears in a broadcast alongside the value.
"""
struct Spline{T, BT, CT <: AbstractArray}
    basis::BT
    coefficients::CT

    function Spline(basis, coefficients::AbstractArray)
        _check_coefficients(basis, coefficients)
        T = promote_type(eltype(basis), eltype(coefficients))
        new{T, typeof(basis), typeof(coefficients)}(basis, coefficients)
    end
end

function _check_coefficients(basis::AbstractBSplineBasis, c::AbstractArray)
    (ndims(c) == 1 && length(c) == nbasis(basis)) || throw(DimensionMismatch(
        "a one-dimensional basis of $(nbasis(basis)) functions needs a vector of " *
        "$(nbasis(basis)) coefficients, got an array of size $(size(c))"))
end

function _check_coefficients(basis::TensorProductBasis, c::AbstractArray)
    size(c) == size(basis) || throw(DimensionMismatch(
        "a tensor-product basis of size $(size(basis)) needs a coefficient array of that " *
        "size, got $(size(c))"))
end

"""
    Spline(basis)

The zero spline on `basis`, with a freshly allocated coefficient array of the right shape.
"""
function Spline(basis::AbstractBSplineBasis{T}) where {T}
    Spline(basis, zeros(T, nbasis(basis)))
end
Spline(basis::TensorProductBasis{T}) where {T} = Spline(basis, zeros(T, size(basis)...))

"""
    basis(s::Spline)

The basis the spline is expanded in.
"""
basis(s::Spline) = s.basis

"""
    coefficients(s::Spline)

The coefficient array **itself**, not a copy: writing into it changes the spline. That is the
point — a projection writes the coefficients of a spline that is already wired into a
right-hand side.
"""
coefficients(s::Spline) = s.coefficients

Base.eltype(::Spline{T}) where {T} = T
Base.ndims(s::Spline) = ndims(s.coefficients)
Base.size(s::Spline) = size(s.coefficients)
Base.length(s::Spline) = length(s.coefficients)

nbasis(s::Spline) = nbasis(s.basis)
degree(s::Spline) = degree(s.basis)
order(s::Spline) = order(s.basis)
domain(s::Spline) = domain(s.basis)

Base.similar(s::Spline) = Spline(s.basis, similar(s.coefficients))
function Base.similar(s::Spline, ::Type{T}) where {T}
    Spline(s.basis, similar(s.coefficients, T))
end
Base.copy(s::Spline) = Spline(s.basis, copy(s.coefficients))

function Base.show(io::IO, s::Spline)
    print(io, "Spline(", nameof(typeof(s.basis)), ", ", size(s.coefficients), ")")
end

## Evaluation

(s::Spline)(x) = evaluate(s.basis, s.coefficients, x)
(s::Spline)(x, d) = evaluate(s.basis, s.coefficients, x, d)

"""
    evaluate(s::Spline, x, d = 0)

The `d`-th derivative of `s` at `x`. For a tensor product `d` is a per-axis tuple.
"""
evaluate(s::Spline, x) = evaluate(s.basis, s.coefficients, x)
evaluate(s::Spline, x, d) = evaluate(s.basis, s.coefficients, x, d)

@doc raw"""
    SplineDerivative

A derivative of a [`Spline`](@ref), as returned by [`derivative`](@ref). Callable, so that
`derivative(s).(v)` broadcasts over a vector of points the way `s.(v)` does.

```jldoctest
julia> b = BSplineBasis(UniformMesh(16, 0 .. 1), 3);

julia> s = Spline(b, l2_projection(SplineQuadrature(b), sin));

julia> ds = derivative(s);

julia> maximum(abs, ds.([0.2, 0.5, 0.8]) - cos.([0.2, 0.5, 0.8])) < 1e-5
true
```

This is the shape a collision operator's right-hand side wants. An expression such as
``\nu \, ( f_s'(v_\alpha) / f_s(v_\alpha) + v_\alpha )`` reads as one broadcast over the
particle velocities with `derivative(fs).(v) ./ fs.(v)`, and the derivative object holds no
state of its own — it shares the coefficient array, so a reprojection is visible through it
without rebuilding anything.
"""
struct SplineDerivative{T, ST <: Spline{T}, DT}
    spline::ST
    d::DT
end

@doc raw"""
    derivative(s::Spline, d = 1)
    derivative(s::Spline, k::Integer)

The `d`-th derivative of `s` as a [`SplineDerivative`](@ref) — a callable, so that
`derivative(s).(v)` broadcasts over a vector of points the way `s.(v)` does.

On a [`TensorProductBasis`](@ref) an integer `k` selects the ``k``-th partial derivative,
`derivative(s, 2)` being ``\partial_2 s``; a tuple gives a mixed derivative directly.
"""
derivative(s::Spline, d = 1) = SplineDerivative(s, d)

function derivative(s::Spline{<:Any, <:TensorProductBasis{<:Any, D}}, k::Integer) where {D}
    # `Spline`'s first parameter is `promote_type(eltype(basis), eltype(coefficients))`, so
    # binding it to the basis element type as well would silently miss this method whenever
    # the coefficients are wider than the basis, leaving the bare integer in the tuple slot.
    1 ≤ k ≤ D || throw(ArgumentError(
        "a $(D)-dimensional spline has no axis $(k); `derivative(s, k)` selects the " *
        "k-th partial derivative and needs `1 ≤ k ≤ $(D)`"))
    SplineDerivative(s, ntuple(i -> i == k ? 1 : 0, D))
end

(ds::SplineDerivative)(x) = evaluate(ds.spline, x, ds.d)

Base.eltype(::SplineDerivative{T}) where {T} = T
basis(ds::SplineDerivative) = basis(ds.spline)
coefficients(ds::SplineDerivative) = coefficients(ds.spline)

function Base.show(io::IO, ds::SplineDerivative)
    print(io, "SplineDerivative(", ds.d, ", ", nameof(typeof(basis(ds))), ")")
end

## Projection into an existing spline

"""
    l2_projection!(s::Spline, q, f)

Project `f` onto the space of `s`, writing the coefficients into the array `s` already holds
so that every reference to `s` sees the new function.
"""
function l2_projection!(s::Spline, q, f)
    l2_projection!(s.coefficients, q, f)
    return s
end
