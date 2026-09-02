
@doc raw"""
    Mesh{T}

A subdivision of a closed interval ``\Omega = [a,b]`` into `n` cells by the `n+1`
breakpoints

```math
a = y_1 < y_2 < \dots < y_{n+1} = b ,
```

so that cell `k` is ``[y_k, y_{k+1}]``.

A mesh carries no boundary condition. It is the geometry alone, and the same mesh serves a
clamped basis, a recombined one and a periodic one — what differs between them is how the
knot vector is closed at the two ends, which is the basis's business and not the mesh's. See
[`BoundaryCondition`](@ref).

!!! note "Both endpoints are breakpoints"
    There are `n+1` breakpoints for `n` cells, including *both* endpoints. A
    [`PeriodicBSplineBasis`](@ref) identifies ``y_{n+1} \equiv y_1`` and therefore has only
    `n` distinct breakpoints, but that identification belongs to the basis; `breakpoints`
    always returns the geometric list of `n+1`.

Four families are provided. The difference between the first three matters for testing
rather than for use:

  - [`UniformMesh`](@ref) — equally spaced. The assembled matrices of a periodic basis are
    circulant, which makes several quantities vanish identically that do not vanish in
    general.
  - [`GradedMesh`](@ref) — the image of a uniform mesh under a fixed smooth map. Refining
    `n` therefore gives a genuine *family* of meshes, and rates of convergence measured
    across it are meaningful.
  - [`RandomMesh`](@ref) — randomly perturbed cell widths. A different mesh for every `n`,
    so convergence rates across it mean nothing; its use is to check the properties that
    must hold exactly on *any* mesh.
  - [`GeneralMesh`](@ref) — an explicit list of breakpoints, for a subdivision that none of
    the three families describes.
"""
abstract type Mesh{T} end

@doc raw"""
    breakpoints(m::Mesh)

The `n+1` breakpoints of `m`, in increasing order, running from ``a`` to ``b`` inclusive.

These are the cell boundaries: cell `k` is `breakpoints(m)[k] .. breakpoints(m)[k+1]`. They
are to be distinguished from the knot vector of a basis built on `m`, whose entries may
repeat.
"""
breakpoints(m::Mesh) = error("breakpoints is not implemented for $(typeof(m)).")

"""
    domain(m::Mesh)

The closed interval ``[a,b]`` the mesh subdivides, as a `ClosedInterval`.
"""
domain(m::Mesh) = ClosedInterval(m.a, m.b)

"""
    ncells(m::Mesh)

The number of cells `n`, one fewer than the number of [`breakpoints`](@ref).
"""
ncells(m::Mesh) = m.n

"""
    domainlength(m::Mesh)

The width ``b - a`` of the domain.

For a [`PeriodicBSplineBasis`](@ref) built on `m` this is the period ``L``, which is why the
name is not `width`.
"""
domainlength(m::Mesh) = rightendpoint(domain(m)) - leftendpoint(domain(m))

Base.eltype(::Mesh{T}) where {T} = T
Base.length(m::Mesh) = ncells(m)
Base.first(m::Mesh) = leftendpoint(domain(m))
Base.last(m::Mesh) = rightendpoint(domain(m))

"""
    meshwidth(m::Mesh)

The largest cell width, the ``h`` that convergence rates are measured against.
"""
function meshwidth(m::Mesh)
    y = breakpoints(m)
    h = zero(eltype(y))
    for i in firstindex(y):(lastindex(y) - 1)
        h = max(h, y[i + 1] - y[i])
    end
    return h
end

Base.hash(m::Mesh, h::UInt) = hash(breakpoints(m), hash(domain(m), h))
function Base.:(==)(m1::Mesh, m2::Mesh)
    (domain(m1) == domain(m2) && breakpoints(m1) == breakpoints(m2))
end
Base.isequal(m1::Mesh{T1}, m2::Mesh{T2}) where {T1, T2} = (T1 == T2 && m1 == m2)
function Base.isapprox(m1::Mesh, m2::Mesh; kwargs...)
    (isapprox(leftendpoint(domain(m1)), leftendpoint(domain(m2)); kwargs...) &&
     isapprox(rightendpoint(domain(m1)), rightendpoint(domain(m2)); kwargs...) &&
     isapprox(breakpoints(m1), breakpoints(m2); kwargs...))
end

function _check_ncells(n::Integer)
    n ≥ 1 || throw(ArgumentError(
        "a mesh needs at least one cell, got n = $(n)"))
end

function _check_domain(a, b)
    (isfinite(a) && isfinite(b)) || throw(ArgumentError(
        "the domain must be finite, got [$(a), $(b)]"))
    b > a || throw(ArgumentError(
        "the domain must have positive width, got [$(a), $(b)]"))
end

# The domain may be given as an interval, as a two-tuple, or as a single number standing for
# [0, L]. The last is what the periodic constructors took before there was a domain at all,
# and it stays because `UniformMesh(n, 2π)` is the common case in a periodic setting.
_interval(d::ClosedInterval) = d
_interval(d::Tuple{Any, Any}) = ClosedInterval(d[1], d[2])
_interval(L::Number) = ClosedInterval(zero(L), L)

_rawtype(d::ClosedInterval{T}) where {T} = T
_rawtype(d::Tuple{Any, Any}) = promote_type(typeof(d[1]), typeof(d[2]))
_rawtype(L::Number) = typeof(L)

# An integer domain is promoted, because the breakpoints of a subdivision of it are not
# integers -- `UniformMesh(4, 0 .. 1)` would otherwise try to store 0.25 in an `Int` and throw
# an `InexactError` from inside `breakpoints`, a long way from the constructor that chose the
# type. Rational and extended-precision types are left alone: they are closed under the
# division `breakpoints` performs, and narrowing them to `Float64` would silently discard the
# exactness that is the reason for using them.
_domaintype(d) = _floattype(_rawtype(d))
_floattype(::Type{T}) where {T <: Integer} = float(T)
_floattype(::Type{T}) where {T} = T

@doc raw"""
    UniformMesh(n, domain)
    UniformMesh{T}(n, domain)

The uniform mesh of `n` cells on `domain`, ``y_i = a + (i-1) (b-a) / n``.

`domain` may be a `ClosedInterval` such as `-1 .. 1`, a tuple `(a, b)`, or a single number
`L` standing for ``[0, L]``. `UniformMesh(n)` is ``[0, 2\pi]``.

```jldoctest
julia> breakpoints(UniformMesh(4, 0 .. 1))
5-element Vector{Float64}:
 0.0
 0.25
 0.5
 0.75
 1.0

julia> breakpoints(UniformMesh(2, -1 .. 1))
3-element Vector{Float64}:
 -1.0
  0.0
  1.0
```

On a uniform mesh the B-spline basis functions of a periodic basis are translates of a
single cardinal spline, so the mass, stiffness and derivative matrices assembled from it are
*circulant*. Several identities of the discrete brackets hold on a uniform mesh and nowhere
else; a test that means to check a property valid on any mesh should use
[`RandomMesh`](@ref) instead.
"""
struct UniformMesh{T} <: Mesh{T}
    n::Int
    a::T
    b::T

    function UniformMesh{T}(n::Integer, domain) where {T}
        d = _interval(domain)
        a, b = convert(T, leftendpoint(d)), convert(T, rightendpoint(d))
        _check_ncells(n)
        _check_domain(a, b)
        new{T}(n, a, b)
    end
end

UniformMesh(n::Integer, domain) = UniformMesh{_domaintype(domain)}(n, domain)
UniformMesh(n::Integer) = UniformMesh(n, 2convert(Float64, π))

function breakpoints(m::UniformMesh{T}) where {T}
    T[m.a + (m.b - m.a) * (i - 1) / m.n for i in 1:(m.n + 1)]
end

meshwidth(m::UniformMesh) = (m.b - m.a) / m.n

@doc raw"""
    GradedMesh(n, domain; amplitude = 0.12)
    GradedMesh{T}(n, domain; amplitude = 0.12)

The image of a uniform mesh under the fixed smooth map

```math
s \mapsto s + \frac{a}{2\pi} \sin 2 \pi s ,
\qquad s_i = \frac{i-1}{n} ,
```

with `amplitude` ``= a``, affinely rescaled onto `domain`. The map does not depend on `n`,
so refining `n` gives a genuine family of meshes and the rates measured across it are
meaningful — which is exactly what [`RandomMesh`](@ref) does not give.

The map is a bijection of ``[0,1]`` with positive derivative for ``|a| < 1``, and fixes both
endpoints, so the breakpoints still run from ``a`` to ``b``. The cell widths vary by a factor
of ``(1+a)/(1-a)`` across the domain.

```jldoctest
julia> m = GradedMesh(8, 0 .. 1);

julia> issorted(breakpoints(m)) && breakpoints(m)[begin] == 0 && breakpoints(m)[end] == 1
true
```
"""
struct GradedMesh{T} <: Mesh{T}
    n::Int
    a::T
    b::T
    amplitude::T
    y::Vector{T}

    function GradedMesh{T}(n::Integer, domain; amplitude = 0.12) where {T}
        d = _interval(domain)
        a, b = convert(T, leftendpoint(d)), convert(T, rightendpoint(d))
        _check_ncells(n)
        _check_domain(a, b)
        abs(amplitude) < 1 || throw(ArgumentError(
            "the grading amplitude must satisfy |a| < 1 for the map to stay increasing, " *
            "got amplitude = $(amplitude)"))
        α = convert(T, amplitude)
        new{T}(n, a, b, α, _graded_breakpoints(T, n, a, b, α))
    end
end

function GradedMesh(n::Integer, domain; kwargs...)
    GradedMesh{_domaintype(domain)}(n, domain; kwargs...)
end
GradedMesh(n::Integer; kwargs...) = GradedMesh(n, 2convert(Float64, π); kwargs...)

function _graded_breakpoints(::Type{T}, n::Integer, a::T, b::T, α::T) where {T}
    L = b - a
    [a + L * (s + α * sinpi(2s) / (2convert(T, π)))
     for s in (T(i - 1) / n for i in 1:(n + 1))]
end

breakpoints(m::GradedMesh) = m.y

@doc raw"""
    RandomMesh(n, domain; seed = 1, spread = 0.6)
    RandomMesh{T}(n, domain; seed = 1, spread = 0.6)

A mesh of `n` cells whose widths are drawn as ``1 + \sigma \, r_i`` with ``r_i`` uniform on
``[0,1)`` and `spread` ``= \sigma``, then rescaled to sum to the width of `domain`. The
stream is seeded, so the mesh is reproducible.

The widths vary by up to a factor of ``1 + \sigma``, and a different `n` gives an unrelated
mesh rather than a refinement of this one. Convergence rates measured across a family of
`RandomMesh`es are therefore meaningless; use [`GradedMesh`](@ref) for those. What this mesh
is for is the properties that must hold exactly on *any* mesh — the antisymmetry of the
discrete brackets, for instance, which a uniform mesh would confirm for the wrong reason,
its assemblies being circulant.
"""
struct RandomMesh{T} <: Mesh{T}
    n::Int
    a::T
    b::T
    seed::UInt
    spread::T
    y::Vector{T}

    function RandomMesh{T}(n::Integer, domain; seed = 1, spread = 0.6) where {T}
        d = _interval(domain)
        a, b = convert(T, leftendpoint(d)), convert(T, rightendpoint(d))
        _check_ncells(n)
        _check_domain(a, b)
        spread ≥ 0 || throw(ArgumentError(
            "the width spread must be non-negative, got spread = $(spread)"))
        s, σ = convert(UInt, seed), convert(T, spread)
        new{T}(n, a, b, s, σ, _random_breakpoints(T, n, a, b, s, σ))
    end
end

function RandomMesh(n::Integer, domain; kwargs...)
    RandomMesh{_domaintype(domain)}(n, domain; kwargs...)
end
RandomMesh(n::Integer; kwargs...) = RandomMesh(n, 2convert(Float64, π); kwargs...)

function _random_breakpoints(::Type{T}, n::Integer, a::T, b::T, seed::UInt,
        spread::T) where {T}
    rng = Xoshiro(seed)
    w = [one(T) + spread * rand(rng, T) for _ in 1:n]
    w .*= (b - a) / sum(w)
    y = Vector{T}(undef, n + 1)
    y[1] = a
    for i in 2:n
        y[i] = y[i - 1] + w[i - 1]
    end
    # The last breakpoint is set rather than accumulated, so that it is the right endpoint
    # exactly and not to within the rounding of n additions. A basis whose knot vector ends
    # a few eps short of b evaluates to zero at b, which is a wrong answer at exactly the
    # point a clamped basis is meant to be interpolatory.
    y[end] = b
    return y
end

breakpoints(m::RandomMesh) = m.y

@doc raw"""
    GeneralMesh(breakpoints)
    GeneralMesh{T}(breakpoints)

The mesh with the given explicit breakpoints, which must be strictly increasing. The domain
is `first(breakpoints) .. last(breakpoints)`.

This is the escape hatch for a subdivision none of the three families describes. The case it
exists for is a mesh that is uniform in the interior but carries one oversized cell at each
end — a device for keeping particles that stray outside the resolved region inside the
support of the basis, used in particle discretisations of kinetic equations:

```jldoctest
julia> m = GeneralMesh([-20.0; range(-10, 10; length = 5); 20.0]);

julia> ncells(m), domain(m)
(6, -20.0 .. 20.0)

julia> diff(breakpoints(m))
6-element Vector{Float64}:
 10.0
  5.0
  5.0
  5.0
  5.0
 10.0
```

A `GeneralMesh` is never circulant even when its breakpoints happen to be equally spaced, in
the sense that no assembly built on it takes the [`CirculantMass`](@ref) path — that is
decided by the mesh *type*, so an equally spaced `GeneralMesh` is a legitimate way to force
the general path in a test.
"""
struct GeneralMesh{T} <: Mesh{T}
    y::Vector{T}

    function GeneralMesh{T}(y::AbstractVector) where {T}
        length(y) ≥ 2 || throw(ArgumentError(
            "a mesh needs at least one cell, hence at least two breakpoints, got " *
            "$(length(y))"))
        yy = collect(T, y)
        all(isfinite, yy) || throw(ArgumentError("the breakpoints must all be finite"))
        # Strictly increasing, not merely sorted: a repeated breakpoint is a cell of zero
        # width, which the Cox-de Boor recursion would silently skip rather than reject, and
        # the resulting basis would have the wrong dimension for its knot vector.
        all(>(0), diff(yy)) || throw(ArgumentError(
            "the breakpoints must be strictly increasing"))
        new{T}(yy)
    end
end

GeneralMesh(y::AbstractVector{T}) where {T} = GeneralMesh{float(T)}(y)

breakpoints(m::GeneralMesh) = m.y
ncells(m::GeneralMesh) = length(m.y) - 1
domain(m::GeneralMesh) = ClosedInterval(m.y[begin], m.y[end])
