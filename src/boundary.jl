
@doc raw"""
    BoundaryCondition

How a B-spline basis is closed at the ends of its [`Mesh`](@ref).

Two quite different things go by this name, and separating them is what makes the general
case tractable:

**How the knot vector is closed.** This fixes the spline *space* and its dimension.
[`Periodic`](@ref) continues the breakpoints periodically and gives ``N = n``;
[`Free`](@ref) repeats each end knot ``p+1`` times — the *clamped* or *open* knot vector —
and gives ``N = n + p``. These are different constructions rather than variants of one, and
periodicity is not a per-end setting: it couples the two ends, so it applies to a whole axis
or not at all.

**Homogeneous linear constraints at one end.** [`Dirichlet`](@ref) ``u = 0``,
[`Neumann`](@ref) ``u' = 0``, [`Robin`](@ref) ``\alpha u + \beta u' = 0``,
[`Natural`](@ref) ``u'' = 0``, or a [`Constraint`](@ref) of arbitrary order. These are
constraints *on* the clamped space, imposed by recombination — see
[`RecombinedBSplineBasis`](@ref) — and each one costs one degree of freedom at the end it
applies to.

# Specifying them

A single condition applies to both ends; a two-tuple gives the left and the right:

```jldoctest
julia> b = BSplineBasis(UniformMesh(8, 0 .. 1), 3);                       # Free, both ends

julia> nbasis(b)
11

julia> nbasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3, Dirichlet()))       # one per end
9

julia> nbasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3, (Dirichlet(), Free())))
10

julia> nbasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3, Periodic()))
8
```

Lowercase symbols are accepted as sugar and normalised on construction — `:periodic`,
`:dirichlet`, `:neumann`, `:natural`, `:free`.

# Which one conserves what

For a particle or finite-element discretisation the choice is not free: a conservation law
survives the discretisation only if the conserved density lies in the span of the basis. The
clamped basis reproduces every polynomial of degree ``\le p`` exactly, a periodic basis
reproduces only the constants, and a Dirichlet-recombined basis reproduces none — not even
the constants, since every one of its functions vanishes at the ends.

| condition | polynomials reproduced | ``\int f`` | ``\int v f`` | ``\int v^2 f`` |
|:--|:--|:--|:--|:--|
| [`Free`](@ref) | degree ``\le p`` | ✓ | ``p \ge 1`` | ``p \ge 2`` |
| [`Periodic`](@ref) | constants | ✓ | ✗ | ✗ |
| [`Dirichlet`](@ref) | none | ✗ | ✗ | ✗ |

[`polynomial_reproduction`](@ref) reports this as a number, so that a scheme whose
conservation proof needs ``1, v, v^2`` in the span can assert it rather than assume it.

See also [`Free`](@ref), [`Periodic`](@ref), [`Dirichlet`](@ref), [`Neumann`](@ref),
[`Robin`](@ref), [`Natural`](@ref), [`Constraint`](@ref).
"""
abstract type BoundaryCondition end

@doc raw"""
    Periodic()

The periodic closure: the breakpoints are continued periodically, ``y_{i+n} = y_i + L``, and
the basis is wrapped onto the torus.

Applies to a whole axis rather than to one end, since it identifies the two. Gives
``N = n``, and there are no boundary functions at all: every basis function spans ``p+1``
cells and the basis is ``\mathcal{C}^{p-1}`` across the seam as well as inside, so
integration by parts leaves no boundary terms. See [`PeriodicBSplineBasis`](@ref).
"""
struct Periodic <: BoundaryCondition end

@doc raw"""
    Free()

No constraint: the plain clamped basis, of dimension ``n + p`` on the end it applies to.

The knot vector repeats the end knot ``p+1`` times, so the basis is interpolatory there —
``\varphi_1(a) = 1`` and every other function vanishes at ``a``. The full spline space
``\mathcal{S}^p`` of ``\mathcal{C}^{p-1}`` piecewise polynomials is represented, and every
polynomial of degree ``\le p`` is reproduced exactly.

!!! note "This is not the natural boundary condition"
    `Free` is the *absence* of a condition. The natural boundary condition is
    ``u'' = 0``, which is [`Natural`](@ref). The two are easy to confuse because a clamped
    basis is sometimes loosely called a "natural" spline basis; they span different spaces
    and have different dimensions.
"""
struct Free <: BoundaryCondition end

@doc raw"""
    Dirichlet()

The homogeneous Dirichlet condition ``u = 0``.

On a clamped basis ``\varphi_1`` is the only function that does not vanish at the end, so
the recombination reduces to dropping it — the textbook elimination — and the dimension
falls by one per end it applies to.
"""
struct Dirichlet <: BoundaryCondition end

@doc raw"""
    Neumann()

The homogeneous Neumann condition ``u' = 0``.

Unlike [`Dirichlet`](@ref) this is a genuine recombination: both ``\varphi_1`` and
``\varphi_2`` have nonzero derivative at the end, and the surviving function is the
combination of the two that the derivative annihilates.
"""
struct Neumann <: BoundaryCondition end

@doc raw"""
    Natural()

The natural boundary condition ``u'' = 0``.

Requires `p ≥ 2`; for a lower degree the second derivative of every basis function vanishes
identically and the condition constrains nothing, which is reported as an error rather than
silently accepted.
"""
struct Natural <: BoundaryCondition end

@doc raw"""
    Robin(α, β)

The homogeneous Robin condition ``\alpha u + \beta u' = 0``.

Reduces to [`Dirichlet`](@ref) at ``\beta = 0`` and to [`Neumann`](@ref) at ``\alpha = 0``,
though those types are preferable where they apply — they say what is meant, and `Dirichlet`
takes the cheaper elimination path. At least one of the two coefficients must be nonzero.

The sign convention is the one written above, with both terms on the same side and the
outward direction playing no role: `Robin(1, 1)` is ``u + u' = 0`` at *both* ends if given as
a single condition, not ``u \pm u' = 0``. Pass a two-tuple where the ends differ.
"""
struct Robin{T <: Number} <: BoundaryCondition
    α::T
    β::T

    function Robin(α::T, β::T) where {T <: Number}
        (iszero(α) && iszero(β)) && throw(ArgumentError(
            "a Robin condition needs at least one nonzero coefficient, got α = β = 0"))
        new{T}(α, β)
    end
end

Robin(α::Number, β::Number) = Robin(promote(α, β)...)

@doc raw"""
    Constraint(c...)

The general homogeneous local condition ``\sum_{k} c_{k+1} \, D^k u = 0``, with `c[1]` the
coefficient of ``u``, `c[2]` of ``u'``, and so on.

`Constraint(1)` is [`Dirichlet`](@ref), `Constraint(0, 1)` is [`Neumann`](@ref),
`Constraint(α, β)` is [`Robin`](@ref) and `Constraint(0, 0, 1)` is [`Natural`](@ref); the
named types are preferable where they apply. What this adds is the conditions with no
standard name — ``u''' = 0``, or ``u - 2u'' = 0``.

The highest derivative appearing must be at most `p`, since the ``p``-th derivative of a
degree-``p`` spline is piecewise constant and everything above it vanishes identically.

!!! note "Local constraints only"
    Every condition here is *local*: it involves the solution at one endpoint only, which is
    what keeps the recombination sparse and the mass matrix banded. A nonlocal or multi-point
    constraint — ``u(a) = u(b)`` other than through [`Periodic`](@ref), or an integral
    condition — is deliberately out of scope. Imposing one requires a nullspace of a dense
    constraint matrix, which destroys the banding every assembly in this package relies on.
"""
struct Constraint{N, T <: Number} <: BoundaryCondition
    c::NTuple{N, T}

    function Constraint(c::NTuple{N, T}) where {N, T <: Number}
        N ≥ 1 || throw(ArgumentError("a constraint needs at least one coefficient"))
        any(!iszero, c) || throw(ArgumentError(
            "a constraint needs at least one nonzero coefficient, got all zeros"))
        new{N, T}(c)
    end
end

Constraint(c::Number...) = Constraint(promote(c...))

@doc raw"""
    constraint_coefficients(bc::BoundaryCondition)

The coefficients ``(c_0, c_1, \dots)`` of the local condition
``\sum_k c_k \, D^k u = 0`` that `bc` imposes, as a tuple, or `nothing` for the conditions
that impose none — [`Free`](@ref) and [`Periodic`](@ref).

This is the one place the named conditions are given their meaning, so that the recombination
has a single implementation rather than one method per condition.
"""
constraint_coefficients(::Free) = nothing
constraint_coefficients(::Periodic) = nothing
constraint_coefficients(::Dirichlet) = (1,)
constraint_coefficients(::Neumann) = (0, 1)
constraint_coefficients(::Natural) = (0, 0, 1)
constraint_coefficients(bc::Robin) = (bc.α, bc.β)
constraint_coefficients(bc::Constraint) = bc.c

"""
    nconstraints(bc::BoundaryCondition)

The number of degrees of freedom the condition removes at the end it applies to: `0` for
[`Free`](@ref) and [`Periodic`](@ref), `1` for every local condition.
"""
nconstraints(bc::BoundaryCondition) = constraint_coefficients(bc) === nothing ? 0 : 1

"""
    constraint_order(bc::BoundaryCondition)

The highest derivative order appearing in the condition, or `-1` if it imposes none.

Used to check the condition against the degree of the basis: a condition involving ``D^k``
with `k > p` constrains nothing, because the `k`-th derivative of every degree-`p` spline
vanishes identically.
"""
function constraint_order(bc::BoundaryCondition)
    c = constraint_coefficients(bc)
    c === nothing && return -1
    k = findlast(!iszero, c)
    return k - 1
end

Base.:(==)(b1::Robin, b2::Robin) = (b1.α == b2.α && b1.β == b2.β)
Base.:(==)(b1::Constraint, b2::Constraint) = (b1.c == b2.c)
Base.hash(bc::Robin, h::UInt) = hash(bc.α, hash(bc.β, hash(:Robin, h)))
Base.hash(bc::Constraint, h::UInt) = hash(bc.c, hash(:Constraint, h))

Base.show(io::IO, ::Free) = print(io, "Free()")
Base.show(io::IO, ::Periodic) = print(io, "Periodic()")
Base.show(io::IO, ::Dirichlet) = print(io, "Dirichlet()")
Base.show(io::IO, ::Neumann) = print(io, "Neumann()")
Base.show(io::IO, ::Natural) = print(io, "Natural()")
Base.show(io::IO, bc::Robin) = print(io, "Robin(", bc.α, ", ", bc.β, ")")
Base.show(io::IO, bc::Constraint) = print(io, "Constraint(", join(bc.c, ", "), ")")

# The symbol sugar. Lowercase only, and deliberately not case-insensitive: `:Natural` meant
# the *unconstrained* clamped basis in the code this package replaces, while `:natural` here
# means u'' = 0. Accepting the capitalised form would silently hand back a different function
# space than the caller's old code had, so the names that changed meaning throw instead.
const _BC_SYMBOLS = Dict{Symbol, BoundaryCondition}(
    :periodic => Periodic(),
    :dirichlet => Dirichlet(),
    :neumann => Neumann(),
    :natural => Natural(),
    :free => Free()
)

const _BC_RENAMED = Dict{Symbol, String}(
    :Natural =>
        "`:Natural` used to mean the unconstrained clamped basis, which is now " *
        "`Free()`. `:natural` now means the natural condition u'' = 0, i.e. " *
        "`Natural()`. Say which you mean.",
    :nothing =>
        "`:nothing` used to select the unconstrained clamped basis by falling " *
        "through a branch. Write `Free()`.",
    :Dirichlet =>
        "boundary conditions are given in lower case or as types: `:dirichlet` " *
        "or `Dirichlet()`.",
    :Periodic =>
        "boundary conditions are given in lower case or as types: `:periodic` " *
        "or `Periodic()`."
)

"""
    BoundaryCondition(s::Symbol)

Normalise a symbol to a [`BoundaryCondition`](@ref). Accepts `:periodic`, `:dirichlet`,
`:neumann`, `:natural` and `:free`.
"""
function BoundaryCondition(s::Symbol)
    haskey(_BC_SYMBOLS, s) && return _BC_SYMBOLS[s]
    if haskey(_BC_RENAMED, s)
        throw(ArgumentError("boundary condition :$(s) is not accepted: " * _BC_RENAMED[s]))
    end
    throw(ArgumentError(
        "unknown boundary condition :$(s); the accepted symbols are " *
        join((":$(k)" for k in sort(collect(keys(_BC_SYMBOLS)))), ", ") *
        ", or pass a BoundaryCondition directly"))
end

BoundaryCondition(bc::BoundaryCondition) = bc

@doc raw"""
    boundary_conditions(bc)

Normalise a user-supplied boundary specification to either [`Periodic`](@ref) or a pair
`(left, right)` of [`BoundaryCondition`](@ref)s.

Accepts a single condition, meaning both ends; a two-tuple, giving left and right; and a
symbol or pair of symbols in place of either.

```jldoctest
julia> boundary_conditions(Dirichlet())
(Dirichlet(), Dirichlet())

julia> boundary_conditions((:dirichlet, :neumann))
(Dirichlet(), Neumann())

julia> boundary_conditions(:periodic)
Periodic()
```

`Periodic` is returned bare rather than as a pair, because it is a condition on the axis and
not on its ends: there is no such thing as being periodic at the left end only, and pairing
it with anything else is rejected.
"""
boundary_conditions(bc::Periodic) = bc
boundary_conditions(bc::BoundaryCondition) = (bc, bc)
boundary_conditions(s::Symbol) = boundary_conditions(BoundaryCondition(s))

function boundary_conditions(bcs::Tuple{Any, Any})
    left, right = BoundaryCondition(bcs[1]), BoundaryCondition(bcs[2])
    (left isa Periodic || right isa Periodic) && throw(ArgumentError(
        "Periodic identifies the two ends of the domain, so it cannot be given for one " *
        "end only; pass `Periodic()` for the whole axis"))
    return (left, right)
end
