
@doc raw"""
    quadrature_order(p)

The number of Gauß-Legendre points per cell that integrates degree ``3p-1`` exactly,
``n_q = \lceil 3p/2 \rceil``.

An `nq`-point Gauß-Legendre rule is exact to degree ``2 n_q - 1``, so this is the smallest
`nq` with ``2 n_q - 1 \ge 3p - 1``. Degree ``3p-1`` is what the *consistency* of a Galerkin
discretisation of a quadratic nonlinearity needs — an identity such as
``\int 6 u_h u_{h,x} \phi_i = -3 \int u_h^2 \phi_i'`` pairs three basis functions and one
derivative. The mass matrix alone would need only ``2p-1``.

What this rule is *not* needed for is antisymmetry. A bracket assembled in explicitly
skew-symmetrised form is antisymmetric to the last bit at any `nq` and on any mesh; it is
accuracy, not structure, that the quadrature buys.

```jldoctest
julia> quadrature_order.(1:4)
4-element Vector{Int64}:
 2
 3
 5
 6
```
"""
quadrature_order(p::Integer) = cld(3p, 2)

"""
    local_indices(b::AbstractBSplineBasis, cell)

The indices of the basis functions that are nonzero on `cell`, as an iterable of `Int`.

`cell:(cell+p)` for a clamped [`BSplineBasis`](@ref); the same block wrapped onto `1:N` for a
[`PeriodicBSplineBasis`](@ref); and the precomputed column range for a
[`RecombinedBSplineBasis`](@ref), which may be wider than `p+1` near an end.

The indices are distinct, which is what lets an assembly push one entry per pair without
having to worry about a duplicate being summed into place.
"""
function local_indices(b::BSplineBasis, cell::Integer)
    cell:(cell + degree(b))
end

function local_indices(b::PeriodicBSplineBasis, cell::Integer)
    p = degree(b)
    (basis_index(b, cell - p + t - 1) for t in 1:(p + 1))
end

function local_indices(b::RecombinedBSplineBasis, cell::Integer)
    b.firstcol[cell]:b.lastcol[cell]
end

@doc raw"""
    SplineQuadrature(basis; nq = quadrature_order(degree(basis)), dmax = 3)

The assembly table of an [`AbstractBSplineBasis`](@ref): its basis functions and their
derivatives tabulated at the global Gauß-Legendre quadrature points, together with the
quadrature weights and the mass matrix.

Any of the three bases serves — clamped, periodic or recombined. What differs between them is
which functions are nonzero on a cell ([`local_indices`](@ref)) and which representation the
mass matrix takes ([`mass_operator`](@ref)); the contractions below are the same in all three
cases.

This is the one data structure every assembly here is built from. With `Φ[d+1][i,q]` the
`d`-th derivative of ``\phi_i`` at the `q`-th quadrature point and `w` the weight vector,
every matrix of the form ``\int_\Omega f(x) \, D^a \phi_k \, D^b \phi_l \, dx`` is a single
weighted contraction,

```math
A = \Phi_a \, \mathrm{diag}(f \odot w) \, \Phi_b^T ,
```

which is what [`mixed_matrix`](@ref) and [`weighted_matrix`](@ref) do. Writing the assembly
this way rather than element-by-element is what makes a variable coefficient — the
``\int u_h \phi_k \phi_l'`` of the second KdV bracket, say — cost no more than a constant
one.

# Arguments

  - `nq`: quadrature points per cell. The default integrates degree ``3p-1`` exactly; see
    [`quadrature_order`](@ref).
  - `dmax`: the highest derivative order tabulated. Three is what a third-order operator
    needs after one integration by parts.

```jldoctest
julia> q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), 3));

julia> S = derivative_matrix(q);

julia> maximum(abs, S + S') < 1e-14        # antisymmetric on a periodic mesh
true
```

# Storage

`Φ` is a `SparseMatrixCSC`, `N` by `n * nq`, one per derivative order. Only the entries inside
each basis function's support of `p+1` cells are ever nonzero, and only those are stored: the
structurally nonzero count per row is `(p+1) * nq` rather than `n * nq`, which is what keeps
the contraction ``\Phi \, \mathrm{diag}(f \odot w) \, \Phi^T`` proportional to `N p` instead
of `N²`.

!!! warning "One quadrature per thread"
    A `SplineQuadrature` carries mutable state behind an otherwise read-only interface:
    [`mixed_matrix`](@ref) memoises its results into a `Dict`, and [`l2_projection!`](@ref)
    forms its ``f \odot w`` product in a shared buffer. Neither is synchronised, so one
    quadrature must not be used from two threads at once. Give each thread its own, or stay
    with the allocating [`l2_projection`](@ref), which forms its own product.
"""
struct SplineQuadrature{T, BT <: AbstractBSplineBasis{T}, MO <: MassOperator{T}}
    basis::BT
    nq::Int
    x::Vector{T}
    w::Vector{T}
    Φ::Vector{SparseMatrixCSC{T, Int}}
    mass::MO
    integrals::Vector{T}
    scratch::Vector{T}
    cache::Dict{Tuple{Int, Int}, SparseMatrixCSC{T, Int}}

    function SplineQuadrature(basis::BT;
            nq::Integer = quadrature_order(degree(basis)),
            dmax::Integer = 3) where {T, BT <: AbstractBSplineBasis{T}}
        nq ≥ 1 || throw(ArgumentError(
            "at least one quadrature point per cell is needed, got nq = $(nq)"))
        dmax ≥ 0 || throw(ArgumentError(
            "the highest derivative order must be non-negative, got dmax = $(dmax)"))

        n = ncells(basis)
        p = degree(basis)
        N = nbasis(basis)

        ξ = gauss_legendre_nodes(T, nq)
        ω = gauss_legendre_weights(T, nq)
        bounds = breakpoints(basis)

        x = Vector{T}(undef, n * nq)
        w = Vector{T}(undef, n * nq)
        for k in 1:n
            a, h = bounds[k], bounds[k + 1] - bounds[k]
            for r in 1:nq
                x[(k - 1) * nq + r] = a + h * ξ[r]
                w[(k - 1) * nq + r] = h * ω[r]
            end
        end

        # Only the basis functions supported on a cell are evaluated there, and only those
        # entries are stored. That is the whole difference between an O(N p² n_q) assembly
        # and an O(N² n n_q) one: a dense tabulation makes every contraction
        # Φ diag(f w) Φᵀ cost N² times the number of quadrature points, when the number of
        # structurally nonzero entries per row is only (p+1) n_q.
        #
        # The entry count is not n (p+1) nq exactly: a recombined basis has a wider block at
        # the two ends. The vectors are grown rather than presized for that reason.
        Is = Int[]
        Js = Int[]
        Vs = [T[] for _ in 0:dmax]

        for k in 1:n, i in local_indices(basis, k)

            for r in 1:nq
                q = (k - 1) * nq + r
                push!(Is, i)
                push!(Js, q)
                for d in 0:dmax
                    push!(Vs[d + 1], evaluate(basis, i, x[q], d))
                end
            end
        end

        Φ = [sparse(Is, Js, Vs[d + 1], N, n * nq) for d in 0:dmax]

        M = Φ[1] * Diagonal(w) * Φ[1]'
        M = (M + M') / 2                    # symmetric by construction; enforce it exactly

        # The representation of the mass matrix is chosen by the basis: circulant, hence
        # diagonalised by the Fourier transform, for a periodic basis on a uniform mesh; a
        # sparse Cholesky factorisation otherwise. Too coarse a quadrature makes M singular
        # rather than merely inexact, and `mass_operator` says so by name.
        mass = try
            mass_operator(M, basis)
        catch err
            err isa ArgumentError && throw(ArgumentError(
                "the mass matrix assembled with nq = $(nq) points per cell is not usable " *
                "for a degree-$(p) basis on $(n) cells: $(err.msg). Use nq ≥ $(p + 1) for " *
                "an exact mass matrix, or at least nq ≥ 2"))
            rethrow()
        end

        # ∫ φ_i dx is a constant of the discretisation like the mass matrix, so it is
        # assembled here rather than recomputed on every call. `scratch` is the f ⊙ w
        # buffer that keeps `l2_projection!` from allocating one per call.
        integrals = Φ[1] * w
        scratch = Vector{T}(undef, n * nq)

        new{T, BT, typeof(mass)}(basis, Int(nq), x, w, Φ, mass, integrals, scratch,
            Dict{Tuple{Int, Int}, SparseMatrixCSC{T, Int}}())
    end
end

"""
    quadrature_nodes(q::SplineQuadrature)

The global quadrature points, `nq` Gauß-Legendre points in each of the `n` cells,
concatenated in cell order.
"""
quadrature_nodes(q::SplineQuadrature) = q.x

"""
    quadrature_weights(q::SplineQuadrature)

The global quadrature weights, scaled by the cell widths, so that
`sum(quadrature_weights(q)) == L`.
"""
quadrature_weights(q::SplineQuadrature) = q.w

@doc raw"""
    basis_values(q::SplineQuadrature, d = 0)

The table `Φ[i,r]` of the `d`-th derivative of ``\phi_i`` at the `r`-th quadrature point.
"""
function basis_values(q::SplineQuadrature, d::Integer = 0)
    0 ≤ d ≤ length(q.Φ) - 1 || throw(ArgumentError(
        "derivatives up to order $(length(q.Φ) - 1) were tabulated, but order $(d) was " *
        "requested; rebuild the quadrature with dmax = $(d)"))
    q.Φ[d + 1]
end

"""
    mass_operator(q::SplineQuadrature)

The [`MassOperator`](@ref) of the quadrature — a [`CirculantMass`](@ref) on a uniform mesh,
a [`FactorizedMass`](@ref) otherwise.
"""
mass_operator(q::SplineQuadrature) = q.mass

basis(q::SplineQuadrature) = q.basis
nbasis(q::SplineQuadrature) = nbasis(q.basis)
degree(q::SplineQuadrature) = degree(q.basis)
order(q::SplineQuadrature) = order(q.basis)
ncells(q::SplineQuadrature) = ncells(q.basis)
domainlength(q::SplineQuadrature) = domainlength(q.basis)
meshwidth(q::SplineQuadrature) = meshwidth(q.basis)

Base.eltype(::SplineQuadrature{T}) where {T} = T

@doc raw"""
    mixed_matrix(q::SplineQuadrature, a, b)

The matrix ``\int_\Omega D^a \phi_k \, D^b \phi_l \, dx``.

```jldoctest
julia> q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), 3));

julia> K0 = mixed_matrix(q, 1, 2);      # ∫ φ_k' φ_l''  =  -∫ φ_k φ_l'''

julia> maximum(abs, K0 + mixed_matrix(q, 0, 3)) < 1e-10
true
```
"""
function mixed_matrix(q::SplineQuadrature, a::Integer, b::Integer)
    # Memoised: these are constants of the discretisation, and the stiffness matrix in
    # particular is otherwise reassembled on every Newton iteration of every step.
    get!(q.cache, (Int(a), Int(b))) do
        A = basis_values(q, a) * Diagonal(q.w) * basis_values(q, b)'
        SparseMatrixCSC{eltype(q), Int}(A)
    end
end

@doc raw"""
    mass_matrix(q::SplineQuadrature)

The mass matrix ``\mathbb{M}_{kl} = \int_\Omega \phi_k \phi_l \, dx``, symmetric positive definite.

Assembled once when the [`SplineQuadrature`](@ref) is built and returned by reference; see
[`mass_factorization`](@ref) for the Cholesky factor that goes with it.
"""
mass_matrix(q::SplineQuadrature) = mass_matrix(q.mass)

"""
    mass_factorization(q::SplineQuadrature)

The Cholesky factorization of [`mass_matrix`](@ref), for solving with the mass matrix
without refactorizing.
"""
mass_factorization(q::SplineQuadrature) = q.mass

@doc raw"""
    stiffness_matrix(q::SplineQuadrature)

The stiffness matrix ``\mathbb{K}^1_{kl} = \int_\Omega \phi_k' \phi_l' \, dx``, symmetric
positive semi-definite with the constants in its kernel.
"""
stiffness_matrix(q::SplineQuadrature) = mixed_matrix(q, 1, 1)

@doc raw"""
    derivative_matrix(q::SplineQuadrature)

The matrix ``S_{kl} = \int_\Omega \phi_k \phi_l' \, dx``.

On a periodic mesh this is *already* antisymmetric, with no skew-symmetrisation needed:
``\int_\Omega \partial_x (\phi_k \phi_l) \, dx = 0`` because the basis is
``\mathcal{C}^{p-1}`` across the seam and there are no boundary terms. Its skew part is
therefore `S` itself and not `S/2`, which is where the factor of one half in the first
discrete bracket comes from.

The proviso is that the quadrature integrate that total derivative, of degree ``2p-1``,
exactly — which means `nq ≥ p`, one point per cell more than
[`quadrature_order`](@ref) would suggest is needed at low degree. Below it the identity
fails outright rather than gracefully: at `p = 3` and `nq = 2` the defect is ``9 \times
10^{-3}`` on a [`RandomMesh`](@ref). On a [`UniformMesh`](@ref) it holds at any `nq`, the
assemblies being circulant, so a check run only there confirms it for the wrong reason.
"""
derivative_matrix(q::SplineQuadrature) = mixed_matrix(q, 0, 1)

@doc raw"""
    weighted_matrix(q::SplineQuadrature, f, a, b)

The matrix ``\int_\Omega f(x) \, D^a \phi_k \, D^b \phi_l \, dx``.

`f` may be a function of the coordinate, or a vector already sampled at
[`quadrature_nodes`](@ref) — the second form is what a variable coefficient given as a
spline expansion becomes, and it avoids resampling a field that is already in hand.

The element type is the promotion of the quadrature's with `f`'s, so a complex coefficient
field gives a complex matrix — unlike [`mixed_matrix`](@ref), which carries no weight and
is always `eltype(q)`.

```jldoctest
julia> q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16, 2π), 3));

julia> A = weighted_matrix(q, sin, 0, 1);   # ∫ sin(x) φ_k φ_l'

julia> size(A)
(16, 16)
```
"""
weighted_matrix(q::SplineQuadrature, f, a::Integer, b::Integer) = weighted_matrix(q, f.(q.x), a, b)

function weighted_matrix(q::SplineQuadrature, f::AbstractVector, a::Integer, b::Integer)
    length(f) == length(q.x) || throw(DimensionMismatch(
        "the coefficient was sampled at $(length(f)) points but the quadrature has " *
        "$(length(q.x))"))
    # `mixed_matrix` narrows to `eltype(q)`; this cannot. With a user-supplied weight the
    # element type is the promotion of the quadrature's with the sample's, so a complex
    # coefficient field gives a complex matrix rather than an `InexactError`. `q.Φ` is
    # concretely typed, so the product already is that `SparseMatrixCSC`.
    basis_values(q, a) * Diagonal(f .* q.w) * basis_values(q, b)'
end

@doc raw"""
    basis_integrals(q::SplineQuadrature)

The vector ``\int_\Omega \phi_i \, dx``.

This is the gradient of the total mass ``C_0 = \int_\Omega u \, dx`` with respect to the
degrees of freedom, and it equals ``\mathbb{M} \mathbf{1}`` because the basis is a partition of
unity. It spans the kernel of the first discrete bracket, which is why the mass is a
Casimir there.

Assembled once when the [`SplineQuadrature`](@ref) is built and returned by reference, as
[`mass_matrix`](@ref) is — do not mutate the result.
"""
basis_integrals(q::SplineQuadrature) = q.integrals

@doc raw"""
    l2_projection(q::SplineQuadrature, f)

The coefficients of the ``L^2`` projection of `f` onto the spline space,
``\hat{u} = \mathbb{M}^{-1} \int_\Omega f \phi_i \, dx``.

`f` may be a function or a vector of values at [`quadrature_nodes`](@ref).

```jldoctest
julia> q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(32, 2π), 3));

julia> û = l2_projection(q, sin);

julia> abs(evaluate(basis(q), û, 1.0) - sin(1.0)) < 1e-5
true
```
"""
l2_projection(q::SplineQuadrature, f) = l2_projection(q, f.(q.x))

function l2_projection(q::SplineQuadrature, f::AbstractVector)
    length(f) == length(q.x) || throw(DimensionMismatch(
        "the function was sampled at $(length(f)) points but the quadrature has " *
        "$(length(q.x))"))
    q.mass \ (basis_values(q, 0) * (q.w .* f))
end

@doc raw"""
    l2_projection!(û, q::SplineQuadrature, f)

In-place [`l2_projection`](@ref), writing the coefficients into `û`.

This allocates nothing on every basis but one: the ``f \odot w`` product goes into a buffer
held by the quadrature, the load vector is formed with `mul!` straight into `û`, and both the
[`CirculantMass`](@ref) and the [`BandedMass`](@ref) solve are themselves allocation-free.
The exception is a periodic basis on a non-uniform mesh, where the [`FactorizedMass`](@ref)
solve still allocates a CHOLMOD temporary, as [`mass_solve!`](@ref) notes.

A sample whose element type is wider than the quadrature's — a complex `f` — gets its own
product instead of being narrowed into that buffer, so this method accepts exactly what
[`l2_projection`](@ref) accepts and merely stops being allocation-free there.

The shared buffer is one of the two reasons a [`SplineQuadrature`](@ref) may not be used from
two threads at once; see the warning there.
"""
function l2_projection!(û::AbstractVector, q::SplineQuadrature, f::AbstractVector)
    length(û) == nbasis(q) || throw(DimensionMismatch(
        "the coefficient vector has $(length(û)) entries but the basis has $(nbasis(q))"))
    length(f) == length(q.x) || throw(DimensionMismatch(
        "the function was sampled at $(length(f)) points but the quadrature has " *
        "$(length(q.x))"))
    # The buffer has the quadrature's element type, so it can only take the product when the
    # product lands in that type; a complex or extended-precision sample gets its own array
    # rather than an `InexactError` or a silent narrowing. The test is on types alone, hence
    # resolved when the method is compiled, so the ordinary path still allocates nothing.
    S = promote_type(eltype(q.w), eltype(f))
    fw = S === eltype(q.scratch) ? q.scratch : similar(f, S)
    fw .= q.w .* f
    mul!(û, basis_values(q, 0), fw)
    mass_solve!(û, q.mass, û)
    return û
end

l2_projection!(û::AbstractVector, q::SplineQuadrature, f) = l2_projection!(û, q, f.(q.x))
