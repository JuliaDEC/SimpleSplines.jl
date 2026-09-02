
@doc raw"""
    RecombinedBSplineBasis(parent, bc)

A clamped [`BSplineBasis`](@ref) with a homogeneous boundary condition imposed at one or both
ends by *recombination*: the basis functions that violate the condition are replaced by the
combinations of them that satisfy it.

Usually reached through [`BSplineBasis`](@ref) rather than constructed by name:

```jldoctest
julia> b = BSplineBasis(UniformMesh(8, 0 .. 1), 3, Dirichlet());

julia> b isa RecombinedBSplineBasis, nbasis(b)
(true, 9)

julia> û = randn(nbasis(b));

julia> abs(evaluate(b, û, 0.0)) < 1e-14, abs(evaluate(b, û, 1.0)) < 1e-14
(true, true)

julia> c = BSplineBasis(UniformMesh(8, 0 .. 1), 3, Neumann());

julia> v̂ = randn(nbasis(c));

julia> abs(evaluate(c, v̂, 0.0, 1)) < 1e-12          # u'(a) = 0 for any coefficients
true
```

# How the recombination works

Write the condition at the left end as ``L u (a) = 0`` with
``L = \sum_k c_k D^k`` of order ``m``, and put ``a_i = (L \phi_i)(a)``. A clamped knot vector
has ``D^k \phi_i (a) = 0`` for ``i > k+1``, so only the first ``m+1`` functions violate the
condition, and only ``\phi_{m+1}`` contributes to ``a_{m+1} = c_m D^m \phi_{m+1}(a)``, which
is nonzero whenever the leading coefficient ``c_m`` is. The ``m`` combinations

```math
\psi_t = \phi_t - \frac{a_t}{a_{m+1}} \, \phi_{m+1} ,
\qquad t = 1, \dots, m ,
```

are then annihilated by ``L`` by construction and span the constrained part of the space, and
``\phi_{m+2}, \dots`` pass through unchanged. Counting: ``m+1`` functions in, ``m`` out, so
each constrained end costs exactly one degree of freedom whatever the order of its condition.
The right end is the mirror image, anchored on ``\phi_{N_p - m}``.

For [`Dirichlet`](@ref), ``m = 0``: there are no combinations and ``\phi_1`` is simply
dropped, which is the textbook elimination. For [`Neumann`](@ref), ``m = 1``: one genuine
combination of ``\phi_1`` and ``\phi_2``. For [`Natural`](@ref), ``m = 2``: ``\phi_1`` and
``\phi_2`` already have vanishing second derivative at ``a`` and pass through, and ``\phi_3``
is the one that goes.

# What is lost

The recombined basis is **not** a partition of unity and its functions are **not**
non-negative — ``\psi_t`` is a difference. The mass matrix stays symmetric positive definite
and banded, so every assembly and every solve is unaffected, but a quantity whose
conservation rested on ``\sum_j \phi_j \equiv 1`` no longer has it. See
[`polynomial_reproduction`](@ref), which is `-1` for a Dirichlet-recombined basis: not even
the constants survive.

# Requirements

The two end blocks must not overlap: `nbasis(parent) > m_left + m_right + 1`. The order of
each condition must be at most the degree.
"""
struct RecombinedBSplineBasis{T, PT <: BSplineBasis{T}, BCL, BCR} <: AbstractBSplineBasis{T}
    parent::PT
    left::BCL
    right::BCR
    R::SparseMatrixCSC{T, Int}
    firstcol::Vector{Int}
    lastcol::Vector{Int}
    width::Int

    function RecombinedBSplineBasis(parent::PT,
            left::BCL,
            right::BCR) where {
            T, PT <: BSplineBasis{T}, BCL <: BoundaryCondition, BCR <: BoundaryCondition}
        p = degree(parent)
        Np = nbasis(parent)
        mL = constraint_order(left)
        mR = constraint_order(right)

        mL ≤ p || throw(ArgumentError(
            "the left boundary condition involves D^$(mL), but every derivative above " *
            "order $(p) of a degree-$(p) spline vanishes identically, so the condition " *
            "constrains nothing"))
        mR ≤ p || throw(ArgumentError(
            "the right boundary condition involves D^$(mR), but every derivative above " *
            "order $(p) of a degree-$(p) spline vanishes identically, so the condition " *
            "constrains nothing"))

        # The two end blocks are [1, mL+1] and [Np-mR, Np]. They must be disjoint, or a
        # single basis function would be recombined by both ends at once and the count below
        # would be wrong.
        Np > mL + mR + 1 || throw(ArgumentError(
            "a degree-$(p) basis on $(ncells(parent)) cells has only $(Np) functions, too " *
            "few to impose $(left) on the left and $(right) on the right; refine the mesh"))

        R, firstcol, lastcol = _recombination(parent, left, right)

        # The local block width is a constant of the basis and `evaluate_all!` needs it on
        # every call, so it is computed here rather than by a maximum over all cells each
        # time -- which made the recombined particle path O(ncells) per point.
        width = maximum(lastcol[c] - firstcol[c] + 1 for c in eachindex(firstcol))

        new{T, PT, BCL, BCR}(parent, left, right, R, firstcol, lastcol, width)
    end
end

# a_i = (L φ_i)(x) for the block of parent functions the condition can reach.
function _constraint_values(parent::BSplineBasis{T}, bc::BoundaryCondition, x,
        block::AbstractVector{<:Integer}) where {T}
    c = constraint_coefficients(bc)
    [sum(c[k + 1] * evaluate(parent, i, x, k) for k in 0:(length(c) - 1); init = zero(T))
     for i in block]
end

function _recombination(parent::BSplineBasis{T}, left::BoundaryCondition,
        right::BoundaryCondition) where {T}
    Np = nbasis(parent)
    p = degree(parent)
    n = ncells(parent)
    y = breakpoints(parent)
    a, b = y[begin], y[end]

    mL = constraint_order(left)
    mR = constraint_order(right)
    ncL = nconstraints(left)
    ncR = nconstraints(right)

    # Parent rows that pass through unchanged, and the columns produced at each end.
    firstpass = ncL == 0 ? 1 : mL + 2
    lastpass = ncR == 0 ? Np : Np - mR - 1
    npass = lastpass - firstpass + 1
    N = mL * ncL + npass + mR * ncR

    Is = Int[]
    Js = Int[]
    Vs = T[]
    col = 0

    if ncL > 0
        aL = _constraint_values(parent, left, a, 1:(mL + 1))
        anchor = aL[end]
        for t in 1:mL
            col += 1
            push!(Is, t)
            push!(Js, col)
            push!(Vs, one(T))
            push!(Is, mL + 1)
            push!(Js, col)
            push!(Vs, -aL[t] / anchor)
        end
    end

    for i in firstpass:lastpass
        col += 1
        push!(Is, i)
        push!(Js, col)
        push!(Vs, one(T))
    end

    if ncR > 0
        aR = _constraint_values(parent, right, b, (Np - mR):Np)
        anchor = aR[begin]
        for s in 2:(mR + 1)
            col += 1
            push!(Is, Np - mR - 1 + s)
            push!(Js, col)
            push!(Vs, one(T))
            push!(Is, Np - mR)
            push!(Js, col)
            push!(Vs, -aR[s] / anchor)
        end
    end

    R = sparse(Is, Js, Vs, Np, N)

    # For each cell, the contiguous range of columns that are nonzero on it. The parent
    # functions nonzero on cell c are c .. c+p; a column is nonzero on the cell if any of its
    # rows is in that range. Precomputed rather than derived at each call, because the end
    # blocks make the relation between cell and column irregular and getting it wrong would
    # silently drop a contribution. The result really is contiguous — the columns are ordered
    # left block, pass-through, right block, which is monotone in parent-row position — and
    # the test suite checks it against `evaluate` at every degree and condition.
    #
    # Inverted: rather than scanning every nonzero once per cell, which is O(n · nnz), each
    # nonzero is visited once and the cells it can reach are updated. Parent row i is nonzero
    # on cells i-p .. i, clipped to 1:n.
    rows = rowvals(R)
    firstcol = fill(N + 1, n)
    lastcol = fill(0, n)
    for j in 1:N, t in nzrange(R, j)

        i = rows[t]
        for c in max(1, i - p):min(n, i)
            firstcol[c] = min(firstcol[c], j)
            lastcol[c] = max(lastcol[c], j)
        end
    end

    return R, firstcol, lastcol
end

Base.parent(b::RecombinedBSplineBasis) = b.parent

mesh(b::RecombinedBSplineBasis) = mesh(b.parent)
knotvector(b::RecombinedBSplineBasis) = knotvector(b.parent)
breakpoints(b::RecombinedBSplineBasis) = breakpoints(b.parent)
findcell(b::RecombinedBSplineBasis, x::Number) = findcell(b.parent, x)
degree(b::RecombinedBSplineBasis) = degree(b.parent)
nbasis(b::RecombinedBSplineBasis) = size(b.R, 2)
boundary(b::RecombinedBSplineBasis) = (b.left, b.right)

"""
    recombination_matrix(b::RecombinedBSplineBasis)

The sparse matrix `R` with `ψ_j = Σ_i R[i,j] φ_i`, of size `nbasis(parent) × nbasis(b)`.

Every assembly of a recombined basis is the parent's assembly conjugated by this matrix —
`M̃ = R' * M * R` for the mass matrix, `Φ̃ = R' * Φ` for a tabulation — which is why the
quadrature needs no separate implementation for the recombined case.
"""
recombination_matrix(b::RecombinedBSplineBasis) = b.R

# Not even the constants survive a Dirichlet condition, so nothing above degree -1 is
# reproduced. Any other local condition leaves a space that contains no polynomial either,
# unless it happens to be satisfied by one -- Neumann admits the constants, since a constant
# has vanishing derivative at both ends. Reported by testing the two cases that occur rather
# than by a general argument, because a general argument here would be a guess.
function polynomial_reproduction(b::RecombinedBSplineBasis)
    _reproduces_constants(b.left) && _reproduces_constants(b.right) ? 0 : -1
end

_reproduces_constants(::Free) = true
_reproduces_constants(::Neumann) = true
_reproduces_constants(::Natural) = true
_reproduces_constants(::Dirichlet) = false
_reproduces_constants(bc::Robin) = iszero(bc.α)
_reproduces_constants(bc::Constraint) = iszero(bc.c[1])

function evaluate(b::RecombinedBSplineBasis{T}, j::Integer, x::Number,
        d::Integer = 0) where {T}
    @boundscheck (1 ≤ j ≤ nbasis(b)) || throw(BoundsError(b, j))
    R = b.R
    rows = rowvals(R)
    vals = nonzeros(R)
    S = _evaltype(T, typeof(x))
    v = zero(S)
    for t in nzrange(R, Int(j))
        v += vals[t] * evaluate(b.parent, rows[t], x, d)
    end
    return v
end

nodes(b::RecombinedBSplineBasis) = _greville_from_columns(b)

# The Greville abscissa of a recombined function is taken to be that of the parent function
# it is anchored on -- the one with unit coefficient -- which for a pass-through column is the
# parent's own. That keeps `nodes` a set of N points inside the domain, in increasing order,
# and interlaced with the supports, which is what a collocation or a plotting grid wants. It
# is not a Schoenberg-Whitney set for the recombined basis, and is not claimed to be.
function _greville_from_columns(b::RecombinedBSplineBasis{T}) where {T}
    ξ = nodes(b.parent)
    R = b.R
    rows = rowvals(R)
    vals = nonzeros(R)
    out = Vector{T}(undef, nbasis(b))
    for j in 1:nbasis(b)
        rng = nzrange(R, j)
        k = rng[argmax(abs(vals[t]) for t in rng)]
        out[j] = ξ[rows[k]]
    end
    return out
end

_local_width(b::AbstractBSplineBasis) = degree(b) + 1
_local_width(b::RecombinedBSplineBasis) = b.width

"""
    local_width(b::AbstractBSplineBasis)

The number of entries [`evaluate_all!`](@ref) writes, i.e. the largest number of basis
functions that are nonzero at a single point.

`p+1` for a [`BSplineBasis`](@ref) or a [`PeriodicBSplineBasis`](@ref). For a
[`RecombinedBSplineBasis`](@ref) it can be larger, because a recombined function near an end
spans the union of two parent supports; the block is still contiguous, and entries beyond the
nonzeros of a particular cell are filled with zeros.
"""
local_width(b::AbstractBSplineBasis) = _local_width(b)

## ---------------------------------------------------------------------------------------
## The dispatching constructor
## ---------------------------------------------------------------------------------------

# `BSplineBasis(mesh, p, bc)` returns whichever of the three types the boundary condition
# calls for. Placed here because it names all three, and documented on `BSplineBasis` itself,
# where a reader looking for it will be.
function BSplineBasis(mesh::Mesh, p::Integer, bc)
    bcs = boundary_conditions(bc)
    _basis_for(mesh, p, bcs)
end

_basis_for(mesh::Mesh, p::Integer, ::Periodic) = PeriodicBSplineBasis(mesh, p)

function _basis_for(mesh::Mesh, p::Integer, bcs::Tuple{
        BoundaryCondition, BoundaryCondition})
    parent = BSplineBasis(mesh, p)
    # Two Free ends impose nothing, so there is nothing to recombine and the clamped basis is
    # returned as it stands rather than wrapped in an identity recombination.
    (bcs[1] isa Free && bcs[2] isa Free) && return parent
    RecombinedBSplineBasis(parent, bcs[1], bcs[2])
end

function evaluate_all!(values::AbstractVector{R}, b::RecombinedBSplineBasis, x::Number,
        d::Integer = 0) where {R}
    w = local_width(b)
    length(values) == w || throw(DimensionMismatch(
        "evaluate_all! needs a buffer of $(w) entries for this basis, got " *
        "$(length(values)); see `local_width`"))

    c = findcell(b, x)
    j₀ = b.firstcol[c]
    jn = b.lastcol[c]

    fill!(values, zero(R))
    Rm = b.R
    rows = rowvals(Rm)
    vals = nonzeros(Rm)
    for j in j₀:jn, t in nzrange(Rm, j)

        values[j - j₀ + 1] += vals[t] * evaluate(b.parent, rows[t], x, d)
    end
    return j₀
end
