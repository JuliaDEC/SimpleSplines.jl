
@doc raw"""
    Mesh{T}

A subdivision of the periodic domain ``\Omega = [0,L)`` into `n` cells by the breakpoints

```math
0 = y_1 < y_2 < \dots < y_n < L ,
```

which are continued periodically, ``y_{i+n} = y_i + L``, so that the last cell is
``[y_n, y_1 + L)``. There are `n` breakpoints and `n` cells, not `n+1`: on a torus the
right endpoint of the last cell *is* the left endpoint of the first.

The breakpoints are to be distinguished from the knot vector of
[`PeriodicBSplineBasis`](@ref), whose entries may repeat.

Three families are provided, and the difference between them matters for testing rather
than for use:

  - [`UniformMesh`](@ref) — equally spaced. The assembled matrices are circulant, which
    makes several quantities vanish identically that do not vanish in general.
  - [`GradedMesh`](@ref) — the image of a uniform mesh under a fixed smooth map. Refining
    `n` therefore gives a genuine *family* of meshes, and rates of convergence measured
    across it are meaningful.
  - [`RandomMesh`](@ref) — randomly perturbed cell widths. A different mesh for every `n`,
    so convergence rates across it mean nothing; its use is to check the properties that
    must hold exactly on *any* mesh.
"""
abstract type Mesh{T} end

"""
    ncells(m::Mesh)

The number of cells `n`, which on a periodic mesh equals the number of breakpoints.
"""
ncells(m::Mesh) = m.n

"""
    domainlength(m::Mesh)

The period ``L`` of the domain ``\\Omega = [0,L)``.
"""
domainlength(m::Mesh) = m.L

Base.eltype(::Mesh{T}) where {T} = T
Base.length(m::Mesh) = ncells(m)

"""
    cellbounds(m::Mesh)

The `n+1` cell boundaries `[y_1, …, y_n, y_1 + L]`, i.e. [`breakpoints`](@ref) with the
periodic image of the first breakpoint appended, so that cell `k` is
`cellbounds(m)[k] .. cellbounds(m)[k+1]`.
"""
cellbounds(m::Mesh) = (y = breakpoints(m); vcat(y, y[begin] + domainlength(m)))

"""
    meshwidth(m::Mesh)

The largest cell width, the ``h`` that convergence rates are measured against.
"""
meshwidth(m::Mesh) = maximum(diff(cellbounds(m)))

Base.hash(m::Mesh, h::UInt) = hash(breakpoints(m), hash(domainlength(m), h))
Base.:(==)(m1::Mesh, m2::Mesh) =
    (domainlength(m1) == domainlength(m2) && breakpoints(m1) == breakpoints(m2))
Base.isequal(m1::Mesh{T1}, m2::Mesh{T2}) where {T1,T2} = (T1 == T2 && m1 == m2)
Base.isapprox(m1::Mesh, m2::Mesh; kwargs...) =
    (isapprox(domainlength(m1), domainlength(m2); kwargs...) &&
     isapprox(breakpoints(m1), breakpoints(m2); kwargs...))

_check_ncells(n::Integer) = n ≥ 1 || throw(ArgumentError(
    "a mesh needs at least one cell, got n = $(n)"))
_check_length(L) = (L > 0 && isfinite(L)) || throw(ArgumentError(
    "the domain length must be positive and finite, got L = $(L)"))


@doc raw"""
    UniformMesh(n, L)
    UniformMesh{T}(n, L)

The uniform mesh of `n` cells on ``[0,L)``, ``y_i = (i-1) L / n``.

```jldoctest
julia> breakpoints(UniformMesh(4, 1.0))
4-element Vector{Float64}:
 0.0
 0.25
 0.5
 0.75
```

On a uniform mesh the B-spline basis functions are translates of a single cardinal spline,
so the mass, stiffness and derivative matrices assembled from it are *circulant*. Several
identities of the discrete brackets hold on a uniform mesh and nowhere else; a test that
means to check a property valid on any mesh should use [`RandomMesh`](@ref) instead.
"""
struct UniformMesh{T} <: Mesh{T}
    n::Int
    L::T

    function UniformMesh{T}(n::Integer, L) where {T}
        _check_ncells(n)
        _check_length(L)
        new{T}(n, convert(T, L))
    end
end

UniformMesh(n::Integer, L::T) where {T <: Number} = UniformMesh{T}(n, L)
UniformMesh(n::Integer) = UniformMesh(n, 2convert(Float64, π))

breakpoints(m::UniformMesh{T}) where {T} =
    T[m.L * (i - 1) / m.n for i in 1:m.n]

meshwidth(m::UniformMesh) = m.L / m.n


@doc raw"""
    GradedMesh(n, L; amplitude = 0.12)
    GradedMesh{T}(n, L; amplitude = 0.12)

The image of a uniform mesh under the fixed smooth map

```math
s \mapsto L \left( s + \frac{a}{2\pi} \sin 2 \pi s \right) ,
\qquad s_i = \frac{i-1}{n} ,
```

with `amplitude` ``= a``. The map does not depend on `n`, so refining `n` gives a genuine
family of meshes and the rates measured across it are meaningful — which is exactly what
[`RandomMesh`](@ref) does not give.

The map is a bijection of ``[0,1]`` with positive derivative for ``|a| < 1``, and the
cell widths vary by a factor of ``(1+a)/(1-a)`` across the domain.

```jldoctest
julia> m = GradedMesh(8, 1.0);

julia> issorted(breakpoints(m)) && breakpoints(m)[1] == 0
true
```
"""
struct GradedMesh{T} <: Mesh{T}
    n::Int
    L::T
    amplitude::T

    function GradedMesh{T}(n::Integer, L; amplitude = 0.12) where {T}
        _check_ncells(n)
        _check_length(L)
        abs(amplitude) < 1 || throw(ArgumentError(
            "the grading amplitude must satisfy |a| < 1 for the map to stay increasing, " *
            "got amplitude = $(amplitude)"))
        new{T}(n, convert(T, L), convert(T, amplitude))
    end
end

GradedMesh(n::Integer, L::T; kwargs...) where {T <: Number} = GradedMesh{T}(n, L; kwargs...)
GradedMesh(n::Integer; kwargs...) = GradedMesh(n, 2convert(Float64, π); kwargs...)

function breakpoints(m::GradedMesh{T}) where {T}
    a = m.amplitude
    [m.L * (s + a * sinpi(2s) / (2convert(T, π))) for s in (T(i - 1) / m.n for i in 1:m.n)]
end


@doc raw"""
    RandomMesh(n, L; seed = 1, spread = 0.6)
    RandomMesh{T}(n, L; seed = 1, spread = 0.6)

A mesh of `n` cells whose widths are drawn as ``1 + \sigma \, r_i`` with ``r_i`` uniform on
``[0,1)`` and `spread` ``= \sigma``, then rescaled to sum to `L`. The stream is seeded, so
the mesh is reproducible.

The widths vary by up to a factor of ``1 + \sigma``, and a different `n` gives an unrelated
mesh rather than a refinement of this one. Convergence rates measured across a family of
`RandomMesh`es are therefore meaningless; use [`GradedMesh`](@ref) for those. What this
mesh is for is the properties that must hold exactly on *any* mesh — the antisymmetry of
the discrete brackets, for instance, which a uniform mesh would confirm for the wrong
reason, its assemblies being circulant.
"""
struct RandomMesh{T} <: Mesh{T}
    n::Int
    L::T
    seed::UInt
    spread::T

    function RandomMesh{T}(n::Integer, L; seed = 1, spread = 0.6) where {T}
        _check_ncells(n)
        _check_length(L)
        spread ≥ 0 || throw(ArgumentError(
            "the width spread must be non-negative, got spread = $(spread)"))
        new{T}(n, convert(T, L), convert(UInt, seed), convert(T, spread))
    end
end

RandomMesh(n::Integer, L::T; kwargs...) where {T <: Number} = RandomMesh{T}(n, L; kwargs...)
RandomMesh(n::Integer; kwargs...) = RandomMesh(n, 2convert(Float64, π); kwargs...)

function breakpoints(m::RandomMesh{T}) where {T}
    rng = Xoshiro(m.seed)
    w = [one(T) + m.spread * rand(rng, T) for _ in 1:m.n]
    w .*= m.L / sum(w)
    y = similar(w)
    y[1] = zero(T)
    for i in 2:m.n
        y[i] = y[i-1] + w[i-1]
    end
    return y
end


@doc raw"""
    breakpoints(m::Mesh)

The `n` breakpoints of `m` in ``[0,L)``, in increasing order and starting at zero.

The cell boundaries, which repeat the first breakpoint at ``L``, are [`cellbounds`](@ref).
"""
breakpoints(m::Mesh) = error("breakpoints is not implemented for $(typeof(m)).")
