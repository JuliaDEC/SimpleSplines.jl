
@doc raw"""
    MassOperator{T}

The mass matrix in the form the solves actually want: something that can be applied and,
above all, inverted, without forming ``\mathbb{M}^{-1}``.

Two representations are provided, and which one is built is decided by the mesh:

  - [`CirculantMass`](@ref) on a [`UniformMesh`](@ref), where the basis functions are
    translates of a single cardinal spline and ``\mathbb{M}`` is *circulant*, hence
    diagonalised by the discrete Fourier transform. A solve is two transforms and a
    pointwise division, ``O(N \log N)``, with the transforms planned once.
  - [`FactorizedMass`](@ref) otherwise. On a [`GradedMesh`](@ref) or a
    [`RandomMesh`](@ref) the matrix is banded modulo ``N`` but *not* circulant — the basis
    functions are no longer translates of one another — so there is nothing for a Fourier
    transform to diagonalise, and a sparse Cholesky factorisation is what is left.

Both answer `\`, `ldiv!` and `Matrix`.

!!! note "Where this does and does not matter"
    A mass solve is a small part of the cost of these discretisations — at ``N = 384`` it is
    around ``0.06`` ms against a ``2`` ms implicit step. The transform is used because it is
    the right representation of a circulant operator and because it costs nothing to plan,
    not because it is where the time goes; that is the assembly, and the answer there is the
    sparsity of the basis tabulation.
"""
abstract type MassOperator{T} end

Base.eltype(::MassOperator{T}) where {T} = T

@doc raw"""
    FactorizedMass(M)

The sparse Cholesky factorisation of the mass matrix, for meshes on which it is not
circulant.

``\mathbb{M}`` is symmetric positive definite and banded modulo ``N`` — a basis function
overlaps only the ``2p+1`` others whose supports meet its own — so the factorisation is
cheap and the solve is ``O(Np)``.
"""
struct FactorizedMass{T, MT, FT} <: MassOperator{T}
    M::MT
    fact::FT

    function FactorizedMass(M::MT) where {T, MT <: AbstractMatrix{T}}
        F = cholesky(Symmetric(M); check = false)
        issuccess(F) || throw(ArgumentError(
            "the mass matrix is not positive definite; the quadrature is too coarse to " *
            "resolve the basis"))
        new{T, MT, typeof(F)}(M, F)
    end
end

@doc raw"""
    CirculantMass(M, n)

The mass matrix of a uniform periodic mesh, represented by the eigenvalues of its circulant
structure and a pair of planned real transforms.

On a uniform mesh every basis function is a translate of one cardinal spline, so
``\mathbb{M}_{kl}`` depends only on ``k - l \bmod N`` and

```math
\mathbb{M} = F^{*} \operatorname{diag}(\hat{c}) F ,
\qquad \hat{c} = \mathcal{F}(\mathbb{M}_{:,1}) ,
```

with ``F`` the discrete Fourier transform. A solve is therefore a forward transform, a
pointwise division and an inverse transform.

The plans are created once, at construction, and the spectral buffer is preallocated, so a
solve allocates nothing beyond its result. `\` allocates the result; [`mass_solve!`](@ref)
does not.

The first column is read from the assembled matrix rather than recomputed, and the
construction checks that the matrix really is circulant — a silent mismatch here would give
wrong answers on every mesh that is uniform by accident rather than by construction.
"""
struct CirculantMass{T, MT, PT, IT} <: MassOperator{T}
    M::MT
    ĉ::Vector{Complex{T}}
    plan::PT
    iplan::IT
    buf::Vector{Complex{T}}
    n::Int

    function CirculantMass(M::MT, n::Integer; atol = 1e-10) where {
            T, MT <: AbstractMatrix{T}}
        size(M, 1) == n || throw(DimensionMismatch(
            "the mass matrix is $(size(M, 1))×$(size(M, 2)) but n = $(n)"))
        c = Vector{T}(M[:, 1])

        # Verify rather than assume. A matrix that is banded but not circulant would still
        # produce plausible numbers through the transform, and the failure would show up
        # much later as a wrong conservation law.
        for j in 2:n, i in 1:n

            abs(M[i, j] - c[mod1(i - j + 1, n)]) ≤ atol * max(one(T), maximum(abs, c)) ||
                throw(ArgumentError(
                    "the mass matrix is not circulant to within $(atol); a CirculantMass " *
                    "is only valid on a uniform mesh"))
        end

        # The plans are made UNALIGNED so that they accept any strided argument -- a view
        # into a column of a matrix, in particular, whose alignment an aligned plan would
        # reject at run time. At these sizes the difference is not measurable, and the
        # alternative is a plan that works everywhere except where it is passed a view.
        #
        # ESTIMATE has to be given explicitly alongside it. Passing UNALIGNED alone replaces
        # the flags rather than adding to them, and FFTW's default rigor then measures --
        # which OVERWRITES the array being planned for. That destroys `c` before the line
        # below reads it, and the symptom is a mass matrix that appears singular at some
        # sizes and not others.
        buf = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        plan = plan_rfft(c; flags = FFTW.ESTIMATE | FFTW.UNALIGNED)
        iplan = plan_irfft(buf, n; flags = FFTW.ESTIMATE | FFTW.UNALIGNED)
        ĉ = plan * c

        all(x -> abs(x) > eps(T), ĉ) || throw(ArgumentError(
            "the circulant mass matrix has a zero eigenvalue and is not invertible"))

        new{T, MT, typeof(plan), typeof(iplan)}(M, ĉ, plan, iplan, buf, Int(n))
    end
end

"""
    mass_matrix(op::MassOperator)

The assembled mass matrix behind the operator.
"""
mass_matrix(op::MassOperator) = op.M

Base.size(op::MassOperator, args...) = size(op.M, args...)
Base.Matrix(op::MassOperator) = Matrix(op.M)

@doc raw"""
    mass_solve!(y, op::MassOperator, x)

Solve ``\mathbb{M} y = x`` in place. `y` and `x` may alias.

Attached to the function rather than to either method, so that the two representations of
[`MassOperator`](@ref) share one piece of documentation and a cross-reference to the name
resolves.
"""
function mass_solve! end

# CHOLMOD factors do not implement an in-place `ldiv!`, so this one allocates a temporary.
# That is the non-uniform path, taken by the graded and random meshes the test suite uses to
# check mesh-independent properties; the runs that matter for time are on uniform meshes and
# go through `CirculantMass`, which allocates nothing.
function mass_solve!(y::AbstractVector, op::FactorizedMass, x::AbstractVector)
    copyto!(y, op.fact \ x)
    return y
end

function mass_solve!(y::AbstractVector, op::CirculantMass, x::AbstractVector)
    mul!(op.buf, op.plan, x)
    op.buf ./= op.ĉ
    mul!(y, op.iplan, op.buf)
    return y
end

function LinearAlgebra.ldiv!(y::AbstractVector, op::MassOperator, x::AbstractVector)
    mass_solve!(y, op, x)
end
LinearAlgebra.ldiv!(op::MassOperator, x::AbstractVector) = mass_solve!(x, op, x)

function Base.:\(op::MassOperator, x::AbstractVector)
    mass_solve!(similar(x, promote_type(eltype(op), eltype(x))), op, x)
end

function Base.:\(op::MassOperator, X::AbstractMatrix)
    Y = similar(X, promote_type(eltype(op), eltype(X)))
    for j in axes(X, 2)
        mass_solve!(view(Y, :, j), op, view(X, :, j))
    end
    return Y
end

@doc raw"""
    mass_operator(M, basis)

Build the [`MassOperator`](@ref) appropriate to `basis`: a [`CirculantMass`](@ref) for a
periodic basis on a [`UniformMesh`](@ref), a [`FactorizedMass`](@ref) otherwise.

!!! note "Circulance is a property of the basis, not of the mesh"
    A uniform mesh is necessary but not sufficient. The mass matrix is circulant only when
    every basis function is a translate of one cardinal spline, which needs the *periodic*
    closure as well: a clamped basis on a uniform mesh has ``p`` boundary functions at each
    end that are not translates of anything, and its mass matrix is banded but not circulant.
    Dispatching on the mesh alone — as an earlier version of this function did, when the
    periodic basis was the only one — would take the Fourier path for a clamped basis and get
    wrong answers everywhere except in [`CirculantMass`](@ref)'s own verification, which
    would reject it.
"""
mass_operator(M::AbstractMatrix, ::AbstractBSplineBasis) = FactorizedMass(M)

function mass_operator(M::AbstractMatrix, b::PeriodicBSplineBasis)
    mesh(b) isa UniformMesh ? CirculantMass(M, nbasis(b)) : FactorizedMass(M)
end
