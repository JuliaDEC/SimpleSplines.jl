
@doc raw"""
    TensorProductBasis(bases...)
    b₁ ⊗ b₂ ⊗ …

The tensor-product basis
``\Phi_{i_1 \dots i_D}(x) = \prod_{d=1}^{D} \phi^{(d)}_{i_d}(x_d)``
built from `D` one-dimensional bases.

Each factor is an independent [`AbstractBSplineBasis`](@ref), so **degree, mesh, domain and
boundary condition are per-axis**. Nothing is shared between the axes, and nothing needs to
be:

```jldoctest
julia> B = TensorProductBasis(
           BSplineBasis(UniformMesh(32, 0 .. 2π), 3, Periodic()),      # x: cubic, periodic
           BSplineBasis(UniformMesh(41, -10 .. 10), 4, Free()),        # v: quartic, clamped
       );

julia> ndims(B), size(B), nbasis(B)
(2, (32, 45), 1440)

julia> degree(B)
(3, 4)

julia> domain(B)
(0.0 .. 6.28319) × (-10.0 .. 10.0)

julia> leftendpoint(domain(bases(B)[2])), rightendpoint(domain(bases(B)[2]))
(-10.0, 10.0)
```

# Coefficients are an array, not a vector

A spline in this basis is ``u_h = \sum_I \hat{u}_I \Phi_I`` with `û` a `D`-dimensional array
of size `size(B)`. That is the natural shape: the Kronecker structure of every operator is
visible in it, and `LinearIndices`/`CartesianIndices` do the index arithmetic that would
otherwise be written by hand — one of the places a hand-rolled tensor-product spline
reliably goes wrong, since the flattening convention has to agree between the evaluation, the
mass matrix and the projection.

# The Kronecker structure is used, not just noted

The mass matrix of a tensor-product basis is
``\mathbb{M} = \mathbb{M}^{(D)} \otimes \dots \otimes \mathbb{M}^{(1)}``, and it is never
formed. A solve is `D` one-dimensional solves applied along each axis in turn — see
[`KroneckerMass`](@ref) — which is ``O(N \sum_d p_d)`` against the ``O(N^3)`` of a dense
factorisation of the Kronecker product, and needs ``O(\sum_d N_d^2)`` storage rather than
``O(N^2)``. At the 41-element cubic basis of a two-dimensional velocity space that is a
``1681 \times 1681`` dense Cholesky avoided; in three dimensions it is a ``68921^2`` one, which
does not fit.

The load vector of an ``L^2`` projection factorises the same way, as `D` successive sparse
contractions — see [`l2_projection`](@ref). The integrand itself need not be separable; only
the *basis* is, and that is enough.

See also [`TensorProductQuadrature`](@ref) for the assembly, [`evaluate_all!`](@ref) for the
local evaluation a particle loop needs, and [`polynomial_reproduction`](@ref), which is the
minimum over the axes.
"""
struct TensorProductBasis{T, D, BS <: Tuple}
    bases::BS

    function TensorProductBasis(bases::AbstractBSplineBasis...)
        D = length(bases)
        D ≥ 1 || throw(ArgumentError("a tensor product needs at least one factor"))
        T = promote_type(map(eltype, bases)...)
        new{T, D, typeof(bases)}(bases)
    end
end

TensorProductBasis(bases::Tuple) = TensorProductBasis(bases...)

"""
    bases(B::TensorProductBasis)

The tuple of one-dimensional bases the product is built from.
"""
bases(B::TensorProductBasis) = B.bases

"""
    ⊗(b₁, b₂)

Tensor product of two bases, or of a tensor product and a basis — `b₁ ⊗ b₂ ⊗ b₃` associates
to a single flat `TensorProductBasis` of three factors rather than a nest of two.
"""
⊗(b1::AbstractBSplineBasis, b2::AbstractBSplineBasis) = TensorProductBasis(b1, b2)
⊗(B::TensorProductBasis, b::AbstractBSplineBasis) = TensorProductBasis(bases(B)..., b)
⊗(b::AbstractBSplineBasis, B::TensorProductBasis) = TensorProductBasis(b, bases(B)...)
function ⊗(B1::TensorProductBasis, B2::TensorProductBasis)
    TensorProductBasis(bases(B1)...,
        bases(B2)...)
end

Base.eltype(::TensorProductBasis{T}) where {T} = T
Base.eltype(::Type{<:TensorProductBasis{T}}) where {T} = T
Base.ndims(::TensorProductBasis{T, D}) where {T, D} = D
Base.ndims(::Type{<:TensorProductBasis{T, D}}) where {T, D} = D
Base.size(B::TensorProductBasis) = map(nbasis, B.bases)
Base.size(B::TensorProductBasis, d::Integer) = nbasis(B.bases[d])
Base.length(B::TensorProductBasis) = prod(size(B))
Base.axes(B::TensorProductBasis) = map(Base.OneTo, size(B))
Base.CartesianIndices(B::TensorProductBasis) = CartesianIndices(axes(B))
Base.LinearIndices(B::TensorProductBasis) = LinearIndices(axes(B))
Base.eachindex(B::TensorProductBasis) = CartesianIndices(B)

nbasis(B::TensorProductBasis) = length(B)
degree(B::TensorProductBasis) = map(degree, B.bases)
order(B::TensorProductBasis) = map(order, B.bases)
ncells(B::TensorProductBasis) = map(ncells, B.bases)
meshwidth(B::TensorProductBasis) = maximum(map(meshwidth, B.bases))
mesh(B::TensorProductBasis) = map(mesh, B.bases)
boundary(B::TensorProductBasis) = map(boundary, B.bases)
local_width(B::TensorProductBasis) = map(local_width, B.bases)

"""
    domain(B::TensorProductBasis)

The product domain ``\\Omega_1 \\times \\dots \\times \\Omega_D``, as a `DomainSets`
`ProductDomain`, so that `x ∈ domain(B)` answers for a `D`-vector `x`.
"""
domain(B::TensorProductBasis) = ProductDomain(map(domain, B.bases)...)

"""
    nodes(B::TensorProductBasis)

The per-axis node vectors, as a tuple. The tensor-product grid is their `Iterators.product`;
it is not materialised, since at three dimensions it is the whole coefficient array.
"""
nodes(B::TensorProductBasis) = map(nodes, B.bases)

"""
    nnodes(B::TensorProductBasis)

The number of points of the tensor-product node grid, ``\\prod_d N_d`` — a scalar, unlike
[`nodes`](@ref)`(B)`, which returns the per-axis vectors whose product that grid is.
"""
nnodes(B::TensorProductBasis) = length(B)

"""
    polynomial_reproduction(B::TensorProductBasis)

The minimum over the axes: a polynomial is in the span of the product only if its restriction
to each axis is in the span of that axis's basis.

For the conservation of ``\\int v_k f`` and ``\\int v_k^2 f`` in a velocity space this must be
at least `1` and `2` respectively **on every axis** — one periodic or Dirichlet axis is enough
to lose it.
"""
function polynomial_reproduction(B::TensorProductBasis)
    minimum(map(polynomial_reproduction, B.bases))
end

function Base.show(io::IO, B::TensorProductBasis{T, D}) where {T, D}
    print(io, "TensorProductBasis{", T, ", ", D, "}(")
    for (i, b) in enumerate(B.bases)
        i > 1 && print(io, " ⊗ ")
        print(io, nameof(typeof(b)), "(p=", degree(b), ", n=", ncells(b), ")")
    end
    print(io, ")")
end

Base.:(==)(B1::TensorProductBasis, B2::TensorProductBasis) = (B1.bases == B2.bases)
Base.hash(B::TensorProductBasis, h::UInt) = hash(B.bases, hash(:TensorProductBasis, h))

## ---------------------------------------------------------------------------------------
## Evaluation
## ---------------------------------------------------------------------------------------

@doc raw"""
    evaluate(B::TensorProductBasis, I, x, d = ntuple(_ -> 0, D))
    evaluate(B::TensorProductBasis, û::AbstractArray, x, d = ntuple(_ -> 0, D))

The mixed derivative ``\partial_1^{d_1} \cdots \partial_D^{d_D}`` at `x` of the basis function
indexed by `I`, or of the spline with coefficient array `û`.

`I` may be a `CartesianIndex`, a tuple, or a linear index into `LinearIndices(B)`. `x` is any
indexable `D`-vector. `d` is a per-axis tuple of derivative orders; the gradient component
``\partial_k`` is `d = ntuple(i -> i == k ? 1 : 0, D)`.

```jldoctest
julia> B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(8, 0 .. 1), 2);

julia> evaluate(B, (3, 4), (0.3, 0.4)) ≈
       evaluate(bases(B)[1], 3, 0.3) * evaluate(bases(B)[2], 4, 0.4)
true

julia> û = zeros(size(B)...); û[3, 4] = 1.0;

julia> evaluate(B, û, (0.3, 0.4)) ≈ evaluate(B, (3, 4), (0.3, 0.4))
true
```

Evaluating the spline this way costs ``O(N)``; evaluating it at a point where only
``\prod_d (p_d+1)`` terms are nonzero costs that much instead, and is what
[`evaluate_all!`](@ref) is for.
"""
function evaluate(B::TensorProductBasis{T, D}, I::CartesianIndex{D}, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    prod(evaluate(B.bases[k], I[k], x[k], d[k]) for k in 1:D)
end

function evaluate(B::TensorProductBasis{T, D}, I::NTuple{D, <:Integer}, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    evaluate(B, CartesianIndex(I), x, d)
end

function evaluate(B::TensorProductBasis{T, D}, i::Integer, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    evaluate(B, CartesianIndices(B)[i], x, d)
end

function evaluate(B::TensorProductBasis{T, D}, û::AbstractArray{S, D}, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D, S}
    R = _evaltype(promote_type(T, S), eltype(x))
    bufs = ntuple(k -> Vector{R}(undef, local_width(B.bases[k])), D)
    evaluate!(bufs, B, û, x, d)
end

@doc raw"""
    evaluate!(bufs::NTuple{D,AbstractVector}, B::TensorProductBasis, û::AbstractArray, x,
              d = ntuple(_ -> 0, D))

The value of [`evaluate`](@ref)`(B, û, x, d)`, computed in the caller's per-axis buffers,
which allocates nothing.

`bufs[k]` must have `local_width(bases(B)[k])` entries, or a `DimensionMismatch` is raised;
their contents on entry do not matter, and on return they are the blocks of
[`evaluate_all!`](@ref). Their element type is the type the sum is accumulated in, so it must
hold the promoted type of the basis, the coefficients and `x`.

The buffers belong to one task. Two tasks that evaluate at once need a tuple each.
"""
function evaluate!(bufs::NTuple{D, <:AbstractVector{R}}, B::TensorProductBasis{T, D},
        û::AbstractArray{S, D}, x, d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {R, T, D, S}
    size(û) == size(B) || throw(DimensionMismatch(
        "the coefficient array is $(size(û)) but the basis is $(size(B))"))
    widths = local_width(B)
    map(length, bufs) == widths || throw(DimensionMismatch(
        "evaluate! needs buffers of $(widths) entries on the axes, got $(map(length, bufs))"))

    # Only the local block contributes, so this is the fast path rather than a sum over the
    # whole array. `evaluate_all!` gives the p_d+1 nonzero factors on each axis; their outer
    # product against the corresponding block of û is the value.
    j₀ = evaluate_all!(bufs, B, x, d)

    # `size(B, k)` reads `B.bases[k]`, and the axes need not have the same basis type, so
    # indexing that tuple with a loop variable dispatches dynamically. Taken once as a tuple
    # of integers instead, which `map` over the axes builds unrolled.
    sz = size(B)

    v = zero(R)
    for t in CartesianIndices(ntuple(k -> length(bufs[k]), D))
        idx = ntuple(k -> basis_index(B.bases[k], j₀[k] + t[k] - 1), D)
        # A periodic axis wraps, so every index is in range; a bounded one does not, and an
        # index outside 1:N_d means the point lies in a cell whose block reaches past the end
        # of that axis. Such a term is genuinely absent rather than zero-valued, so it is
        # skipped rather than indexed.
        all(k -> 1 ≤ idx[k] ≤ sz[k], 1:D) || continue
        # The weight is formed here rather than accumulated inside the `ntuple` above: a
        # variable assigned from within that closure is boxed, which infers as `Any` and makes
        # every element access below a dynamic dispatch.
        v += prod(ntuple(k -> bufs[k][t[k]], D)) * û[idx...]
    end
    return v
end

(B::TensorProductBasis)(x, I) = evaluate(B, I, x)

Base.getindex(B::TensorProductBasis, x, I) = evaluate(B, I, x)

@doc raw"""
    evaluate_all!(bufs::NTuple{D,AbstractVector}, B::TensorProductBasis, x,
                  d = ntuple(_ -> 0, D))

The per-axis local blocks at `x`: `bufs[k]` is filled with the `local_width(bases(B)[k])`
values of the `d[k]`-th derivatives of the nonzero one-dimensional basis functions on axis
`k`, and the tuple of their first indices is returned.

The `D`-dimensional block is the outer product of the factors, and is deliberately **not**
materialised: at ``D = 3`` and cubic bases that is 64 numbers per particle, and the loop that
consumes them can form each product as it goes. A deposition therefore reads

```julia
sz = size(B)
bufs = ntuple(k -> zeros(local_width(bases(B)[k])), ndims(B))
j₀ = evaluate_all!(bufs, B, v, (0, 0))
for t in CartesianIndices(map(length, bufs))
    I = ntuple(k -> basis_index(bases(B)[k], j₀[k] + t[k] - 1), ndims(B))
    all(k -> 1 ≤ I[k] ≤ sz[k], 1:ndims(B)) || continue
    coeffs[I...] += w * prod(ntuple(k -> bufs[k][t[k]], ndims(B)))
end
```

which is ``O(\prod_d (p_d + 1))`` per particle rather than ``O(N)``. As on one axis, the
returned indices are the ones *before* wrapping; put them through
[`basis_index`](@ref).

The extent tuple is taken once outside the loop on purpose: `size(B, k)` reaches into the
tuple of bases, whose axes need not share a basis type, so a loop variable there costs a
dynamic dispatch per term — the one place this pattern is easy to get wrong.
"""
function evaluate_all!(bufs::NTuple{D, <:AbstractVector}, B::TensorProductBasis{T, D}, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    ntuple(k -> evaluate_all!(bufs[k], B.bases[k], x[k], d[k]), D)
end

function evaluate_all(B::TensorProductBasis{T, D}, x,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D}
    R = _evaltype(T, eltype(x))
    bufs = ntuple(k -> Vector{R}(undef, local_width(B.bases[k])), D)
    j₀ = evaluate_all!(bufs, B, x, d)
    return (j₀, bufs)
end

## ---------------------------------------------------------------------------------------
## The Kronecker mass operator
## ---------------------------------------------------------------------------------------

@doc raw"""
    KroneckerMass(ops...)

The mass operator of a [`TensorProductBasis`](@ref),
``\mathbb{M} = \mathbb{M}^{(D)} \otimes \dots \otimes \mathbb{M}^{(1)}``, represented by its
`D` one-dimensional factors and never assembled.

A solve applies each factor's inverse along its own axis:

```math
\mathbb{M}^{-1} = \left( \mathbb{M}^{(D)} \right)^{-1} \otimes \dots \otimes
                  \left( \mathbb{M}^{(1)} \right)^{-1} ,
```

which is exact — this is an identity, not an approximate splitting — and costs
``\sum_d (N/N_d)`` one-dimensional solves. Each factor keeps whatever representation it had:
a [`CirculantMass`](@ref) on a periodic uniform axis, a [`FactorizedMass`](@ref) otherwise.

`\` and `ldiv!` take and return arrays of size `size(B)`. A vector of length `length(B)` is
also accepted and is reshaped, on the convention that the first axis varies fastest — the
same convention `LinearIndices(B)` uses, and the one that makes `Matrix(op)` equal
`kron(M_D, …, M_1)` rather than the reverse.

```jldoctest
julia> B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(6, 0 .. 1), 2);

julia> q = TensorProductQuadrature(B);

julia> op = mass_operator(q);

julia> b = randn(size(B)...);

julia> maximum(abs, Matrix(op) * vec(b) - vec(op * b)) < 1e-10
true

julia> maximum(abs, op \ (op * b) - b) < 1e-10
true
```
"""
struct KroneckerMass{T, D, OT <: Tuple} <: MassOperator{T}
    ops::OT
    dims::NTuple{D, Int}

    function KroneckerMass(ops::MassOperator...)
        D = length(ops)
        T = promote_type(map(eltype, ops)...)
        dims = ntuple(k -> size(ops[k], 1), D)
        new{T, D, typeof(ops)}(ops, dims)
    end
end

Base.ndims(::KroneckerMass{T, D}) where {T, D} = D
Base.size(op::KroneckerMass) = (prod(op.dims), prod(op.dims))
function Base.size(op::KroneckerMass, d::Integer)
    d ≥ 1 || throw(BoundsError(size(op), d))
    d ≤ 2 ? prod(op.dims) : 1
end

"""
    mass_matrix(op::KroneckerMass)

The assembled Kronecker product, `kron(M_D, …, M_1)`.

Formed on demand and **not** stored: it is what the representation exists to avoid, and at
three dimensions it will not fit. Provided so that a test can check the factored solve
against the dense one at a size where both are possible.
"""
function mass_matrix(op::KroneckerMass)
    # A fold rather than a splat into `kron`, which has no one-argument method and so cannot
    # assemble the one-factor product that `TensorProductBasis` allows. At `D = 1` the fold
    # is the identity and would hand back the factor's own storage, hence the `copy`: the
    # docstring promises a matrix formed on demand, which a caller may mutate. The length is
    # a property of the tuple type, so the branch is resolved when the method is compiled.
    Ms = reverse(map(mass_matrix, op.ops))
    length(Ms) == 1 ? copy(only(Ms)) : foldl(kron, Ms)
end
Base.Matrix(op::KroneckerMass) = Matrix(mass_matrix(op))

"""
    mass_factors(op::KroneckerMass)

The tuple of one-dimensional [`MassOperator`](@ref)s, axis order.
"""
mass_factors(op::KroneckerMass) = op.ops

# The extents of a D-dimensional array before and after axis k, for the three-way
# (before, N_k, after) reading of the array that the per-axis loops below work on.
#
# Written as a loop rather than as `prod(dims[1:(k-1)]; init = 1)`. Slicing a tuple with a
# runtime `k` gives a tuple whose *length* is not known to inference, so the product infers as
# `Any`, the linear indices built from it infer as `Any`, and every element access in the
# loops that follow becomes a dynamic dispatch. Indexing the tuple one element at a time keeps
# `k` a run-time value and the result an `Int`.
@inline function _splitdims(dims::NTuple{D, Int}, k::Integer) where {D}
    before = 1
    after = 1
    for i in 1:D
        i < k && (before *= dims[i])
        i > k && (after *= dims[i])
    end
    return before, after
end

# Apply a one-dimensional operator along axis k of the D-dimensional array of size `dims`
# that `A` holds in its linear order, in place. The array is read as (before, N_k, after) and
# each fibre along the middle index is solved on its own. The linear index is computed rather
# than taken from a `reshape`, because a `reshape` of an `Array` allocates its header.
#
# The fibre is copied into a contiguous buffer rather than passed as a view. A view into the
# middle index of a three-way reading has stride `before`, and an FFTW plan or a LAPACK banded
# solve addresses only contiguous memory -- so the one-dimensional solve would stage such a
# fibre through its own buffer in any case. The two buffers `src` and `dst` have N_k entries
# and are reused across the fibres of the axis. They can come from a caller's workspace, so
# the copies keep their bounds checks.
#
# `f!` receives (out, op, in) with two *distinct* buffers, so a method that cannot alias its
# arguments -- `mul!` -- needs no defensive copy of its own.
function _apply_along!(f!, A::AbstractArray, dims::NTuple{D, Int}, op, k::Integer,
        src::AbstractVector, dst::AbstractVector) where {D}
    before, after = _splitdims(dims, k)
    nk = dims[k]
    o = firstindex(A) - 1
    for j in 1:after, i in 1:before

        base = o + i + before * nk * (j - 1)
        for r in 1:nk
            src[r] = A[base + before * (r - 1)]
        end
        f!(dst, op, src)
        for r in 1:nk
            A[base + before * (r - 1)] = dst[r]
        end
    end
    return A
end

# The form with buffers of its own, allocated once for the axis.
function _apply_along!(f!, A::AbstractArray{T}, op, k::Integer) where {T}
    nk = size(A, k)
    _apply_along!(f!, A, size(A), op, k, Vector{T}(undef, nk), Vector{T}(undef, nk))
end

# The solves along the axes, one axis per call. Recursion on the tuples rather than a loop
# over `1:D`: the axes need not have the same operator type, and indexing that tuple with a
# loop variable dispatches dynamically, which allocates.
_solve_axes!(y, dims, ::Tuple{}, ::Tuple{}, ::Tuple{}, k) = y
function _solve_axes!(y, dims, ops::Tuple, src::Tuple, dst::Tuple, k)
    _apply_along!(mass_solve!, y, dims, first(ops), k, first(src), first(dst))
    _solve_axes!(y, dims, Base.tail(ops), Base.tail(src), Base.tail(dst), k + 1)
end

struct KroneckerMassWorkspace{T, D}
    src::NTuple{D, Vector{T}}
    dst::NTuple{D, Vector{T}}
end

Base.eltype(::KroneckerMassWorkspace{T}) where {T} = T

@doc raw"""
    mass_workspace(op::KroneckerMass, T)
    mass_solve!(y, op::KroneckerMass, x, work)

A workspace of element type `T` for [`mass_solve!`](@ref) on `op`, and the solve that uses it,
which allocates nothing.

The workspace holds two buffers of `size(mass_factors(op)[k], 1)` entries for each axis `k`.
The solve takes `y` and `x` as arrays of the size of the tensor-product basis, or as vectors of
`size(op, 1)` entries, and `y === x` is allowed. A `y` of another length raises a
`DimensionMismatch`; a workspace whose element type is not `eltype(y)`, or whose buffer
lengths do not fit `op`, raises an `ArgumentError`. `mass_solve!(y, op, x)` builds a
workspace of `eltype(y)` on each call.

A workspace belongs to one task. It does not make the operator safe for two tasks: a
[`CirculantMass`](@ref) factor holds a scratch vector of its own, which every solve writes.

```jldoctest
julia> B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(6, 0 .. 1), 2);

julia> op = mass_operator(TensorProductQuadrature(B)); x = randn(size(B)...);

julia> work = mass_workspace(op, Float64);

julia> mass_solve!(similar(x), op, x, work) == op \ x
true
```
"""
function mass_workspace(op::KroneckerMass{T, D}, ::Type{S}) where {T, D, S}
    KroneckerMassWorkspace{S, D}(map(n -> Vector{S}(undef, n), op.dims),
        map(n -> Vector{S}(undef, n), op.dims))
end

function mass_solve!(y::AbstractArray, op::KroneckerMass{T, D}, x::AbstractArray,
        work::KroneckerMassWorkspace) where {T, D}
    eltype(work) === eltype(y) || throw(ArgumentError(
        "the workspace has element type $(eltype(work)) but the array has element type " *
        "$(eltype(y))"))
    (map(length, work.src) == op.dims && map(length, work.dst) == op.dims) ||
        throw(ArgumentError(
            "the workspace has buffers of lengths $(map(length, work.src)) and " *
            "$(map(length, work.dst)) but the operator needs $(op.dims)"))
    length(y) == prod(op.dims) || throw(DimensionMismatch(
        "the array has $(length(y)) entries but the operator acts on $(op.dims)"))
    y === x || copyto!(y, x)
    _solve_axes!(y, op.dims, op.ops, work.src, work.dst, 1)
    return y
end

function mass_solve!(y::AbstractArray, op::KroneckerMass, x::AbstractArray)
    mass_solve!(y, op, x, mass_workspace(op, eltype(y)))
end

function LinearAlgebra.mul!(y::AbstractArray, op::KroneckerMass{T, D},
        x::AbstractArray) where {T, D}
    y === x || copyto!(y, x)
    Y = reshape(y, op.dims)
    for k in 1:D
        _apply_along!((o, A, i) -> mul!(o, mass_matrix(A), i), Y, op.ops[k], k)
    end
    return y
end

_masstype(op::MassOperator, x) = promote_type(eltype(op), eltype(x))

_kron_mul(op::KroneckerMass, x) = mul!(similar(x, _masstype(op, x)), op, x)
_kron_solve(op::KroneckerMass, x) = mass_solve!(similar(x, _masstype(op, x)), op, x)

# `AbstractMatrix` and `AbstractVector` are spelled out alongside `AbstractArray` purely to
# disambiguate against the one-dimensional methods in mass.jl, which are declared for
# `AbstractVector` and `AbstractMatrix`. The distinction is not cosmetic: the matrix method
# there solves column by column, which for a two-axis KroneckerMass would apply only the
# first factor and silently return a wrong answer for the natural argument -- a coefficient
# array of a two-dimensional basis, which is exactly an `AbstractMatrix`.
Base.:*(op::KroneckerMass, x::AbstractArray) = _kron_mul(op, x)
Base.:*(op::KroneckerMass, x::AbstractVector) = _kron_mul(op, x)
Base.:*(op::KroneckerMass, x::AbstractMatrix) = _kron_mul(op, x)

Base.:\(op::KroneckerMass, x::AbstractArray) = _kron_solve(op, x)
Base.:\(op::KroneckerMass, x::AbstractVector) = _kron_solve(op, x)
Base.:\(op::KroneckerMass, x::AbstractMatrix) = _kron_solve(op, x)

function LinearAlgebra.ldiv!(y::AbstractArray, op::KroneckerMass, x::AbstractArray)
    mass_solve!(y, op, x)
end
function LinearAlgebra.ldiv!(y::AbstractVector, op::KroneckerMass, x::AbstractVector)
    mass_solve!(y, op, x)
end
LinearAlgebra.ldiv!(op::KroneckerMass, x::AbstractArray) = mass_solve!(x, op, x)
LinearAlgebra.ldiv!(op::KroneckerMass, x::AbstractVector) = mass_solve!(x, op, x)

## ---------------------------------------------------------------------------------------
## The tensor-product quadrature
## ---------------------------------------------------------------------------------------

@doc raw"""
    TensorProductQuadrature(B; nq = quadrature_order.(degree(B)), dmax = 3)

The assembly table of a [`TensorProductBasis`](@ref): `D` one-dimensional
[`SplineQuadrature`](@ref)s, one per axis, together with the [`KroneckerMass`](@ref) built
from their mass operators.

`nq` and `dmax` may be given per axis as a tuple, or once for all axes.

There is no `D`-dimensional tabulation. Everything a Galerkin assembly on a tensor-product
basis needs is a sequence of contractions with the one-dimensional tabulations, and holding
those is ``O(\sum_d N_d n_d n_{q,d})`` rather than the ``O(N \prod_d n_d n_{q,d})`` a
`D`-dimensional table would cost — which at three dimensions is the difference between
kilobytes and not fitting.

```jldoctest
julia> B = BSplineBasis(UniformMesh(16, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(12, 0 .. 2), 2);

julia> q = TensorProductQuadrature(B);

julia> size(quadrature_nodes(q)[1]), size(quadrature_nodes(q)[2])
((80,), (36,))

julia> û = l2_projection(q, x -> sin(π * x[1]) * x[2]);

julia> abs(evaluate(B, û, (0.3, 1.1)) - sin(π * 0.3) * 1.1) < 1e-3
true
```
"""
struct TensorProductQuadrature{T, D, BT <: TensorProductBasis{T, D}, QS <: Tuple,
    MO <: KroneckerMass{T, D}}
    basis::BT
    quadratures::QS
    mass::MO

    function TensorProductQuadrature(B::TensorProductBasis{T, D};
            nq = map(quadrature_order, degree(B)), dmax = 3) where {T, D}
        nqs = nq isa Tuple ? nq : ntuple(_ -> nq, D)
        dmaxs = dmax isa Tuple ? dmax : ntuple(_ -> dmax, D)
        length(nqs) == D || throw(DimensionMismatch(
            "nq was given for $(length(nqs)) axes but the basis has $(D)"))
        length(dmaxs) == D || throw(DimensionMismatch(
            "dmax was given for $(length(dmaxs)) axes but the basis has $(D)"))

        qs = ntuple(k -> SplineQuadrature(B.bases[k]; nq = nqs[k], dmax = dmaxs[k]), D)
        mass = KroneckerMass(map(mass_operator, qs)...)
        new{T, D, typeof(B), typeof(qs), typeof(mass)}(B, qs, mass)
    end
end

"""
    basis(q::TensorProductQuadrature)

The [`TensorProductBasis`](@ref) the quadrature was built for.
"""
basis(q::TensorProductQuadrature) = q.basis

"""
    quadratures(q::TensorProductQuadrature)

The tuple of one-dimensional [`SplineQuadrature`](@ref)s, one per axis, in axis order.

This is how the per-axis tables are reached: `basis_values(quadratures(q)[k], d)` is the
tabulation of axis `k`, and there is no `D`-dimensional table to ask for.
"""
quadratures(q::TensorProductQuadrature) = q.quadratures

"""
    mass_operator(q::TensorProductQuadrature)

The [`KroneckerMass`](@ref) built from the per-axis mass operators, never assembled.
"""
mass_operator(q::TensorProductQuadrature) = q.mass

"""
    mass_matrix(q::TensorProductQuadrature)

The assembled Kronecker product `kron(M_D, …, M_1)`, formed on demand — see
[`mass_matrix`](@ref)`(op::KroneckerMass)` for why it is not stored.
"""
mass_matrix(q::TensorProductQuadrature) = mass_matrix(q.mass)
nbasis(q::TensorProductQuadrature) = nbasis(q.basis)
degree(q::TensorProductQuadrature) = degree(q.basis)
Base.ndims(::TensorProductQuadrature{T, D}) where {T, D} = D
Base.size(q::TensorProductQuadrature) = size(q.basis)
Base.eltype(::TensorProductQuadrature{T}) where {T} = T

"""
    quadrature_nodes(q::TensorProductQuadrature)

The per-axis node vectors, as a tuple. The `D`-dimensional grid is their product; a point of
it is `(x[1][q1], x[2][q2], …)`.
"""
quadrature_nodes(q::TensorProductQuadrature) = map(quadrature_nodes, q.quadratures)

"""
    quadrature_weights(q::TensorProductQuadrature)

The per-axis weight vectors, as a tuple. The weight of a `D`-dimensional point is the product
of the per-axis weights, which is what [`quadrature_sample`](@ref) applies.
"""
quadrature_weights(q::TensorProductQuadrature) = map(quadrature_weights, q.quadratures)

"""
    quadrature_grid_size(q::TensorProductQuadrature)

The size of the `D`-dimensional quadrature grid, `(n_d * nq_d)` per axis.
"""
function quadrature_grid_size(q::TensorProductQuadrature)
    map(qq -> length(quadrature_nodes(qq)), q.quadratures)
end

@doc raw"""
    quadrature_sample(q::TensorProductQuadrature, f)

Sample `f` on the `D`-dimensional quadrature grid, returning an array of size
[`quadrature_grid_size`](@ref).

`f` is called with a `D`-tuple of coordinates. This is the array
[`l2_projection`](@ref) contracts, and it is exposed because the integrand of interest is
often not a plain function of position — the ``\mathbb{L}_k`` of a metriplectic collision
operator is ``\int \varphi_i \, (1 + \log f_s)``, whose integrand is built from a spline that
is already in hand:

```julia
fs = [evaluate(B, f̂, x) for x in Iterators.product(quadrature_nodes(q)...)]
L̂  = l2_projection(q, 1 .+ log.(fs))
```
"""
function quadrature_sample(q::TensorProductQuadrature{T, D}, f) where {T, D}
    x = quadrature_nodes(q)
    [f(pt) for pt in Iterators.product(x...)]
end

## ---------------------------------------------------------------------------------------
## Contraction and projection
## ---------------------------------------------------------------------------------------

@doc raw"""
    contract(q::TensorProductQuadrature, F, d = ntuple(_ -> 0, D))

Contract the array `F`, given on the quadrature grid, against the one-dimensional basis
tabulations, giving the load array

```math
L_{i_1 \dots i_D} = \sum_{q_1 \dots q_D} F_{q_1 \dots q_D}
    \prod_{k=1}^{D} D^{d_k} \phi^{(k)}_{i_k}(x_{q_k}) \, w_{q_k} .
```

`F` must already include whatever the integrand is; the quadrature weights are applied here.

# Method

One axis at a time. With ``\Phi_k`` the sparse ``N_k \times Q_k`` tabulation of axis `k`,

```math
L = \Phi_1 \times_1 \Phi_2 \times_2 \dots \times_D \, (F \odot w) ,
```

evaluated by reshaping the array so that the axis being contracted is first, multiplying by
``\Phi_k``, and cycling that axis to the back. After `D` steps the axes are back in order.
Each step is one sparse matrix-matrix product, so the whole contraction costs
``O\!\left(\sum_k N_k p_k \prod_{j \ne k} \cdot\right)`` — linear in the grid size, against
the ``O(N \prod_k n_k n_{q,k})`` of forming each entry by its own quadrature loop.
"""
function contract(q::TensorProductQuadrature{T, D}, F::AbstractArray{S, D},
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D, S}
    R = promote_type(T, S)
    L = Array{R}(undef, size(q.basis))
    n = _contract_length(q)
    # No mass workspace: `contract!` does not read it.
    contract!(
        L, q, F, ContractWorkspace(Vector{R}(undef, n), Vector{R}(undef, n), nothing), d)
end

struct ContractWorkspace{T, MW <: Union{KroneckerMassWorkspace{T}, Nothing}}
    a::Vector{T}
    b::Vector{T}
    mass::MW
end

Base.eltype(::ContractWorkspace{T}) where {T} = T

# The largest intermediate array of the contraction. Step k reads an array of
# N_1 ⋯ N_(k-1) Q_k ⋯ Q_D entries and writes one of N_1 ⋯ N_k Q_(k+1) ⋯ Q_D.
function _contract_length(q::TensorProductQuadrature{T, D}) where {T, D}
    Qs = quadrature_grid_size(q)
    Ns = size(q.basis)
    n = m = prod(Qs)
    for k in 1:D
        m = m ÷ Qs[k] * Ns[k]
        n = max(n, m)
    end
    return n
end

"""
    contract_workspace(q::TensorProductQuadrature, T)

A workspace of element type `T` for [`contract!`](@ref) and the four-argument
[`l2_projection!`](@ref) on `q`. A workspace belongs to one task.
"""
function contract_workspace(q::TensorProductQuadrature, ::Type{S}) where {S}
    n = _contract_length(q)
    ContractWorkspace(Vector{S}(undef, n), Vector{S}(undef, n),
        mass_workspace(q.mass, S))
end

@doc raw"""
    contract!(L, q::TensorProductQuadrature, F, work, d = ntuple(_ -> 0, D))
    l2_projection!(û, q::TensorProductQuadrature, F::AbstractArray, work)

The forms of [`contract`](@ref) and [`l2_projection!`](@ref) that take a workspace from
[`contract_workspace`](@ref), and allocate nothing.

`contract!` writes [`contract`](@ref)`(q, F, d)` into `L`, of size `size(basis(q))`;
`l2_projection!` writes `l2_projection(q, F)` into `û`. `F` is the sample on the quadrature
grid and is not changed. The workspace holds two buffers for the intermediate arrays of the
contraction and a [`mass_workspace`](@ref) of `mass_operator(q)` for the solve. Wrong sizes
of `L`, `û` or `F` raise a `DimensionMismatch`, for `û` and `F` that of the allocating
forms; a workspace whose element type is not `eltype(L)`, or `eltype(û)`, or whose sizes do
not fit `q`, raises an `ArgumentError`. `contract` builds the two buffers on each call, and
the three-argument `l2_projection!` a whole workspace.

A workspace belongs to one task. It does not make the quadrature safe for two tasks: a
[`CirculantMass`](@ref) factor of its mass operator holds a scratch vector of its own.

```jldoctest
julia> B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(6, 0 .. 1), 2);

julia> q = TensorProductQuadrature(B); F = quadrature_sample(q, x -> x[1] * x[2]);

julia> work = contract_workspace(q, Float64); L = zeros(size(B)...);

julia> contract!(L, q, F, work) == contract(q, F)
true
```
"""
function contract!(L::AbstractArray, q::TensorProductQuadrature{T, D},
        F::AbstractArray{S, D}, work::ContractWorkspace,
        d::NTuple{D, Int} = ntuple(_ -> 0, D)) where {T, D, S}
    size(F) == quadrature_grid_size(q) || throw(DimensionMismatch(
        "the sample is $(size(F)) but the quadrature grid is $(quadrature_grid_size(q))"))
    size(L) == size(q.basis) || throw(DimensionMismatch(
        "the load array is $(size(L)) but the basis is $(size(q.basis))"))
    eltype(work) === eltype(L) || throw(ArgumentError(
        "the workspace has element type $(eltype(work)) but the load array has element " *
        "type $(eltype(L))"))
    n = _contract_length(q)
    (length(work.a) == n && length(work.b) == n) || throw(ArgumentError(
        "the workspace holds buffers of $(length(work.a)) and $(length(work.b)) entries but " *
        "this quadrature needs $(n)"))

    # A copy, always: the weighting below is in place, and `F` is the caller's sample.
    copyto!(work.a, firstindex(work.a), F, firstindex(F), length(F))

    # The weights of every axis, applied once, before any contraction. Doing it here rather
    # than inside the loop keeps the weighting a single elementwise pass and makes the
    # per-axis step a plain matrix product. The weights of all axes are `Vector{T}`, so the
    # loop indexes a tuple of one type.
    w = quadrature_weights(q)
    for k in 1:D
        _scale_along!(work.a, size(F), w[k], k)
    end

    C = _contract_axes!(work.a, work.b, length(F), q.quadratures, d)
    copyto!(L, firstindex(L), C, firstindex(C), length(L))
    return L
end

# The contraction, one axis per call. Recursion on the tuple rather than a loop over `1:D`,
# for the reason of `_solve_axes!`: the axes need not have the same quadrature type.
# `src` holds an array of `n` entries whose first axis is the quadrature axis of `first(qs)`.
# Each step writes `Φ * M` into `dst` transposed, as a (rest, N_k) array, which cycles the
# contracted axis to the back; after the last step the axes are back in order, and the
# buffer that holds the result is returned.
_contract_axes!(src, dst, n, ::Tuple{}, ::Tuple{}) = src
function _contract_axes!(src, dst, n, qs::Tuple, d::Tuple)
    Φ = basis_values(first(qs), first(d))
    N, Q = size(Φ)
    rest = n ÷ Q
    _contract_axis!(dst, Φ, src, rest)
    _contract_axes!(dst, src, rest * N, Base.tail(qs), Base.tail(d))
end

# dst[r, i] = Σ_q Φ[i, q] src[q, r], with `src` read as (Q, rest) and `dst` written as
# (rest, N). The sum runs over the stored entries of each column of Φ, with `muladd`, in the
# order of SparseArrays' sparse-times-dense product, so the result is the one that
# `permutedims(Φ * M)` gives.
function _contract_axis!(dst::AbstractVector, Φ::SparseMatrixCSC, src::AbstractVector,
        rest::Integer)
    N, Q = size(Φ)
    rows = rowvals(Φ)
    vals = nonzeros(Φ)
    @inbounds for r in 1:rest
        for i in 1:N
            dst[r + rest * (i - 1)] = zero(eltype(dst))
        end
        for c in 1:Q
            s = src[c + Q * (r - 1)]
            for t in nzrange(Φ, c)
                i = r + rest * (rows[t] - 1)
                dst[i] = muladd(vals[t], s, dst[i])
            end
        end
    end
    return dst
end

# Scale axis k of the array of size `dims` that `A` holds in its linear order by `w`.
function _scale_along!(A::AbstractVector, dims::NTuple{D, Int}, w::AbstractVector,
        k::Integer) where {D}
    before, after = _splitdims(dims, k)
    nk = dims[k]
    @inbounds for j in 1:after, r in 1:nk, i in 1:before
        A[i + before * ((r - 1) + nk * (j - 1))] *= w[r]
    end
    return A
end

@doc raw"""
    l2_projection(q::TensorProductQuadrature, f)
    l2_projection!(û, q::TensorProductQuadrature, f)
    l2_projection!(û, q::TensorProductQuadrature, F::AbstractArray, work)

The coefficient array of the ``L^2`` projection of `f` onto the tensor-product spline space,
``\hat{u} = \mathbb{M}^{-1} \int_\Omega f \, \Phi_I \, dx``.

`f` may be a function of a `D`-tuple of coordinates, or an array already sampled on the
quadrature grid — see [`quadrature_sample`](@ref). The result has size `size(basis(q))`. The
form with a workspace from [`contract_workspace`](@ref) takes the sample and allocates
nothing.

```jldoctest
julia> B = BSplineBasis(UniformMesh(16, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(16, 0 .. 1), 3);

julia> q = TensorProductQuadrature(B);

julia> û = l2_projection(q, x -> x[1]^2 * x[2]);

julia> abs(evaluate(B, û, (0.37, 0.62)) - 0.37^2 * 0.62) < 1e-12
true
```

The last check is exact to round-off rather than approximate, because a clamped basis of
degree 3 reproduces ``x^2 y`` exactly — which is [`polynomial_reproduction`](@ref), and the
property the conservation of mass, momentum and energy rests on.

Both the load assembly and the solve exploit the Kronecker structure: the first is `D` sparse
contractions ([`contract`](@ref)), the second `D` one-dimensional solves
([`KroneckerMass`](@ref)). Neither forms the ``N \times N`` mass matrix.
"""
function l2_projection(q::TensorProductQuadrature{T, D}, f) where {T, D}
    F = f isa AbstractArray ? f : quadrature_sample(q, f)
    L = contract(q, F)
    return mass_operator(q) \ L
end

function l2_projection!(û::AbstractArray, q::TensorProductQuadrature{T, D}, f) where {T, D}
    # The size of `û` is checked before `f` is sampled on the whole grid.
    _check_coefficients(û, q)
    F = f isa AbstractArray ? f : quadrature_sample(q, f)
    l2_projection!(û, q, F, contract_workspace(q, eltype(û)))
end

function l2_projection!(û::AbstractArray, q::TensorProductQuadrature{T, D},
        F::AbstractArray, work::ContractWorkspace) where {T, D}
    _check_coefficients(û, q)
    contract!(û, q, F, work)
    mass_solve!(û, mass_operator(q), û, work.mass)
    return û
end

function _check_coefficients(û::AbstractArray, q::TensorProductQuadrature)
    size(û) == size(q.basis) || throw(DimensionMismatch(
        "the coefficient array is $(size(û)) but the basis is $(size(q.basis))"))
    return nothing
end
