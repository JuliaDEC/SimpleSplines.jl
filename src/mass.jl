
@doc raw"""
    MassOperator{T}

The mass matrix in the form the solves actually want: something that can be applied and,
above all, inverted, without forming ``\mathbb{M}^{-1}``.

Three representations are provided, and which one is built is decided by the basis:

  - [`CirculantMass`](@ref) for a periodic basis on a [`UniformMesh`](@ref), where the basis
    functions are translates of a single cardinal spline and ``\mathbb{M}`` is *circulant*,
    hence diagonalised by the discrete Fourier transform. A solve is two transforms and a
    pointwise multiplication, ``O(N \log N)``, with the transforms planned once.
  - [`BandedMass`](@ref) for a bounded basis. The overlaps are contiguous — there is no seam —
    so the matrix is banded outright and a banded Cholesky solves it in ``O(Np)`` with no
    allocation at all.
  - [`FactorizedMass`](@ref) for a periodic basis on a [`GradedMesh`](@ref) or a
    [`RandomMesh`](@ref), where the matrix is banded modulo ``N`` but *not* circulant — the
    basis functions are no longer translates of one another — so there is nothing for a
    Fourier transform to diagonalise, and the wrap-around entries put it outside the banded
    representation too. A sparse Cholesky factorisation is what is left.

All three answer `\`, `ldiv!` and `Matrix`.

A mass matrix is positive definite, but the same three representations carry the other
assemblies of a [`SplineQuadrature`](@ref), and a stiffness matrix on a periodic basis is
*singular*: the constants the basis represents lie in its kernel. The `kernel` keyword says
what to do about that. `:reject` is the default and throws, since a singular assembly is
usually too coarse a quadrature rather than an intended one. `:project` states that the
kernel is the constants and asks for the solution that has no constant component — the
mean-free solution — which is what ``-\phi'' = \rho`` on a periodic domain asks for. Both
periodic representations implement it, each in the way its own structure allows, and both
verify the assertion rather than trusting it. Neither forms the rank-one shift
``\mathbb{M} + \mathbb{1}\mathbb{1}^T/N`` that would remove the singularity by filling the
matrix in completely.

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
    FactorizedMass(M; kernel = :reject)

The sparse Cholesky factorisation of the mass matrix, for meshes on which it is not
circulant.

``\mathbb{M}`` is symmetric positive definite and banded modulo ``N`` — a basis function
overlaps only the ``2p+1`` others whose supports meet its own — so the factorisation is
cheap and the solve is ``O(Np)``.

With `kernel = :project` the matrix may be singular instead, with the constants in its
kernel. What is factorised is then the minor that drops the first degree of freedom, which is
positive definite exactly when the kernel is the constants and nothing more: a vector
supported away from the first index lies in the span of ``\mathbb{1}`` only if it is zero.
The dropped degree of freedom is the gauge, and [`mass_solve!`](@ref) fixes it afterwards by
taking the mean out. Deleting a row and a column preserves the sparsity; the rank-one shift
``\mathbb{M} + \mathbb{1}\mathbb{1}^T/N``, which also removes the singularity, makes the
matrix structurally full and the factorisation ``O(N^2)``.
"""
struct FactorizedMass{T, MT, FT, K} <: MassOperator{T}
    M::MT
    fact::FT

    Base.@constprop :aggressive function FactorizedMass(M::MT;
            kernel = :reject) where {T, MT <: AbstractMatrix{T}}
        if kernel === :reject
            F = cholesky(Symmetric(M); check = false)
            issuccess(F) || throw(ArgumentError(
                "the mass matrix is not positive definite; the quadrature is too coarse to " *
                "resolve the basis"))
            return new{T, MT, typeof(F), :reject}(M, F)
        elseif kernel === :project
            _check_constant_kernel(M)
            F = cholesky(Symmetric(M[2:end, 2:end]); check = false)
            _check_definite_minor(F)
            return new{T, MT, typeof(F), :project}(M, F)
        end
        throw(_not_a_kernel_mode(kernel))
    end
end

function _not_a_kernel_mode(kernel)
    ArgumentError("kernel must be :reject or :project, not $(repr(kernel))")
end

function _singular_beyond_the_constants()
    ArgumentError(
        "the matrix is singular beyond the constants: the minor that drops the first degree " *
        "of freedom is not positive definite either")
end

# CHOLMOD calls the factorisation of a positive *semi*definite matrix a success -- a zero pivot
# is not a breakdown to it -- so `issuccess` alone does not say whether the minor is
# invertible, and a kernel one dimension larger than the constants passes it. The whole
# argument for `:project` rests on that minor, so the pivots are read rather than assumed.
#
# `diag` of a CHOLMOD factor is the diagonal of L, so the pivots are its squares, and the minor
# is positive definite to working precision exactly when the smallest is above `n * eps` of the
# largest -- the same relative standard `_check_constant_kernel` and `_reciprocal_eigenvalues`
# hold their own assertions to. A matrix whose kernel is `span{𝟙, v}` puts that ratio at 2e-16,
# where a genuine assembly holds it above 0.18 -- `scripts/mass_tolerance_margins.jl`.
#
# A dense minor takes the same path, and there `diag` of a `Cholesky` is the diagonal of the
# reconstructed matrix rather than of L, so the ratio says nothing about the pivots. It does not
# have to: LAPACK refuses a matrix that is not positive definite, so `issuccess` is already the
# whole answer for a dense factor.
function _check_definite_minor(F)
    issuccess(F) || throw(_singular_beyond_the_constants())
    d = diag(F)
    lo, hi = extrema(abs, d)
    lo > sqrt(length(d) * eps(eltype(d))) * hi || throw(_singular_beyond_the_constants())
    return nothing
end

# `:project` deflates the constants specifically, so it is correct only where that is what the
# kernel holds. Verify rather than assume: a matrix singular for some other reason would take
# the same path, and both the positive-definite minor and the mean-free gauge would then be
# answering a question nobody asked.
#
# The scale is `n * eps(T)` relative to ‖M‖∞, the row-sum scale `M𝟙` is measured on. It has to
# be *relative* and it has to have no floor: a bound that cannot fall below one is absolute for
# every matrix smaller than that, and accepts any invertible matrix scaled down far enough.
#
# `M𝟙` of a genuine singular assembly is rounding, and it saturates about half of that scale in
# the worst corner -- n barely above 2p+1, where every row is nearly full. The gate therefore
# sits a decade above the scale rather than on it. It can afford to: the two sides are not in
# competition here, since the smallest violation an invertible assembly produces is 10⁴ (single
# precision) to 10¹² (double) above the scale. `scripts/mass_tolerance_margins.jl` measures
# both sides.
function _check_constant_kernel(M::AbstractMatrix{T}) where {T}
    n = size(M, 2)
    residual = norm(M * ones(T, n), Inf)
    tol = 10 * n * eps(T) * norm(M, Inf)
    residual ≤ tol || throw(ArgumentError(
        "kernel = :project deflates the constants, but M𝟙 has norm $(residual) against a " *
        "tolerance of $(tol); the kernel of this matrix is not what the projection assumes"))
    return nothing
end

@doc raw"""
    BandedMass(M)

The banded Cholesky factorisation of the mass matrix, for the bases whose mass matrix is
genuinely banded rather than banded modulo ``N``.

A basis function overlaps only the ``2p+1`` others whose supports meet its own, and on a
*bounded* basis those are contiguous — there is no seam, so no corner entries. The
factorisation is therefore ``O(Np^2)`` with no fill-in analysis and no reordering, and, unlike
CHOLMOD, the factor answers `ldiv!` **in place**: a solve allocates nothing at all. That is
what makes [`l2_projection!`](@ref) allocation-free on every bounded basis rather than only on
the uniform periodic one.

[`FactorizedMass`](@ref) remains for the periodic non-uniform case, where the matrix wraps.
"""
struct BandedMass{T, MT, FT} <: MassOperator{T}
    M::MT
    fact::FT

    function BandedMass(M::MT) where {T, MT <: AbstractMatrix{T}}
        kd = _bandwidth(M)
        B = BandedMatrix(Symmetric(M), (kd, kd))
        F = cholesky(Symmetric(B); check = false)
        issuccess(F) || throw(ArgumentError(
            "the mass matrix is not positive definite; the quadrature is too coarse to " *
            "resolve the basis"))
        new{T, MT, typeof(F)}(M, F)
    end
end

# The half-bandwidth: the largest |i - j| carrying a nonzero. Read from the stored entries
# where there are any, since that is the representation every assembly here produces.
function _bandwidth(M::SparseMatrixCSC)
    rows = rowvals(M)
    kd = 0
    for j in axes(M, 2), t in nzrange(M, j)

        kd = max(kd, abs(rows[t] - j))
    end
    return kd
end

function _bandwidth(M::AbstractMatrix)
    kd = 0
    for j in axes(M, 2), i in axes(M, 1)

        iszero(M[i, j]) || (kd = max(kd, abs(i - j)))
    end
    return kd
end

@doc raw"""
    CirculantMass(M, n; kernel = :reject, rtol = sqrt(eps(T)))

The mass matrix of a uniform periodic mesh, represented by the eigenvalues of its circulant
structure and a pair of planned real transforms.

On a uniform mesh every basis function is a translate of one cardinal spline, so
``\mathbb{M}_{kl}`` depends only on ``k - l \bmod N`` and

```math
\mathbb{M} = F^{*} \operatorname{diag}(\hat{c}) F ,
\qquad \hat{c} = \mathcal{F}(\mathbb{M}_{:,1}) ,
```

with ``F`` the discrete Fourier transform. A solve is therefore a forward transform, a
pointwise multiplication by the reciprocals of ``\hat{c}``, and an inverse transform.

The plans are created once, at construction, and the spectral buffer is preallocated, so a
solve allocates nothing beyond its result. `\` allocates the result; [`mass_solve!`](@ref)
does not.

The first column is read from the assembled matrix rather than recomputed, and the
construction checks that the matrix really is circulant — a silent mismatch here would give
wrong answers on every mesh that is uniform by accident rather than by construction. `rtol`
is the tolerance of that check, *relative* to the largest entry of the first column, so that
the verdict survives a rescaling of the assembly and holds in every element type.

`kernel = :project` accepts a matrix that is singular with the constants in its kernel, and
gives the constant mode a zero factor instead of an infinite one. That is the Moore–Penrose
pseudoinverse: the solve drops the constant component of the right-hand side and returns the
solution that has none of it. The transform does the whole of the work, so nothing is added
to the matrix and nothing is taken out of it. This is what makes a periodic stiffness matrix
usable here, and it is the rule an FFT Poisson solver applies when it sets the ``k = 0``
factor to zero rather than dividing by it.

The deflation lives entirely in the stored reciprocal eigenvalues, so the type carries no
kernel-mode parameter and there is one [`mass_solve!`](@ref) for both modes.
[`FactorizedMass`](@ref) carries such a parameter because there the two modes are two
different solves, and the parameter is what selects between them.
"""
struct CirculantMass{T, MT, PT, IT} <: MassOperator{T}
    M::MT
    ĉ⁻¹::Vector{Complex{T}}
    plan::PT
    iplan::IT
    buf::Vector{Complex{T}}
    n::Int

    function CirculantMass(M::MT, n::Integer; rtol = sqrt(eps(T)),
            kernel = :reject) where {
            T, MT <: AbstractMatrix{T}}
        size(M, 1) == n || throw(DimensionMismatch(
            "the mass matrix is $(size(M, 1))×$(size(M, 2)) but n = $(n)"))
        c = Vector{T}(M[:, 1])

        # Verify rather than assume. A matrix that is banded but not circulant would still
        # produce plausible numbers through the transform, and the failure would show up
        # much later as a wrong conservation law.
        _check_circulant(M, c, Int(n), rtol)

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
        ĉ⁻¹ = _reciprocal_eigenvalues(plan * c, Int(n), kernel)

        new{T, MT, typeof(plan), typeof(iplan)}(M, ĉ⁻¹, plan, iplan, buf, Int(n))
    end
end

# The solve multiplies by these rather than dividing by the eigenvalues themselves. It moves
# n ÷ 2 + 1 divisions out of every solve and into the construction, and it is what lets the
# kernel carry a zero factor rather than a branch in the inner loop.
#
# One tolerance decides both modes, and it is *relative*: `n * eps` below the largest
# eigenvalue is where `pinv` puts the same cutoff, and an eigenvalue there is not
# distinguishable from zero by a transform that accumulated its own rounding on the way. An
# absolute bound instead calls a singular matrix invertible whenever its entries are large
# enough -- a periodic stiffness matrix has `|ĉ₀| ≈ 10⁻¹⁴` rather than zero, and the solve
# that follows divides by it.
function _reciprocal_eigenvalues(ĉ::Vector{Complex{T}}, n::Int, kernel) where {T}
    tol = n * eps(T) * maximum(abs, ĉ)
    if kernel === :reject
        all(x -> abs(x) > tol, ĉ) || throw(ArgumentError(
            "the circulant mass matrix has a zero eigenvalue and is not invertible; pass " *
            "kernel = :project to solve in the complement of the kernel instead"))
        return inv.(ĉ)
    elseif kernel === :project
        # The constant mode is ĉ[1] and nothing else, so this deflates the same subspace the
        # factorised representation deflates, and it verifies that the subspace is the whole
        # of the kernel rather than zeroing whichever mode happens to come out small.
        abs(ĉ[1]) ≤ tol || throw(ArgumentError(
            "kernel = :project deflates the constants, but the constant mode of this " *
            "matrix has eigenvalue $(abs(ĉ[1])); its kernel is not what the projection " *
            "assumes"))
        all(x -> abs(x) > tol, @view ĉ[2:end]) || throw(ArgumentError(
            "the matrix is singular beyond the constants: a second mode of the transform " *
            "vanishes as well"))
        return [i == 1 ? zero(z) : inv(z) for (i, z) in enumerate(ĉ)]
    end
    throw(_not_a_kernel_mode(kernel))
end

function _not_circulant(rtol)
    ArgumentError(
        "the mass matrix is not circulant to a relative tolerance of $(rtol); a " *
        "CirculantMass is only valid on a uniform mesh")
end

# `rtol` is relative to the largest entry of the first column, so the verdict does not change
# when the assembly is scaled. It defaults to `sqrt(eps(T))`, which is what makes the question
# answerable in every element type: the residual of a genuinely circulant assembly is
# rounding, at most 180·eps of that scale, while a graded mesh misses by 6e-2 of it
# (`scripts/mass_tolerance_margins.jl`). An absolute default instead fixes the answer to one
# element type -- at 1e-10 no `Float32` assembly is circulant at all, since `Float32` rounding
# alone exceeds it.
#
# The generic check: every entry against the first column. Quadratic, which for a dense
# matrix is what the question costs.
function _check_circulant(M::AbstractMatrix{T}, c::Vector{T}, n::Int, rtol) where {T}
    tol = rtol * maximum(abs, c)
    for j in 2:n, i in 1:n

        abs(M[i, j] - c[mod1(i - j + 1, n)]) ≤ tol || throw(_not_circulant(rtol))
    end
    return nothing
end

# The sparse check visits only the stored entries. Within a column the row indices are
# distinct and r = mod1(i-j+1, n) is a bijection on 1:n, so counting the stored entries whose
# predicted value is above the tolerance and comparing that count with the number of such
# entries in `c` covers the *unstored* positions as well: a column missing one of them cannot
# reach the count. Both directions in O(nnz), which at n = 512 is 0.004 ms against the 9.1 ms
# of probing all n^2 positions of a sparse matrix one `getindex` at a time.
function _check_circulant(M::SparseMatrixCSC{T}, c::Vector{T}, n::Int, rtol) where {T}
    tol = rtol * maximum(abs, c)
    rows = rowvals(M)
    vals = nonzeros(M)
    expected = count(x -> abs(x) > tol, c)

    for j in 1:n
        hits = 0
        for t in nzrange(M, j)
            cᵣ = c[mod1(rows[t] - j + 1, n)]
            abs(vals[t] - cᵣ) ≤ tol || throw(_not_circulant(rtol))
            abs(cᵣ) > tol && (hits += 1)
        end
        hits == expected || throw(_not_circulant(rtol))
    end
    return nothing
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

# CHOLMOD factors do not implement an in-place `ldiv!`, so this one allocates a temporary. It
# is the only representation here that does, and it is reached only by a periodic basis on a
# non-uniform mesh -- the one case that is neither circulant nor banded. A bounded basis on
# the same mesh goes through `BandedMass`, which allocates nothing.
function mass_solve!(y::AbstractVector, op::FactorizedMass{T, MT, FT, :reject},
        x::AbstractVector) where {T, MT, FT}
    copyto!(y, op.fact \ x)
    return y
end

# The constants are in the kernel, so the right-hand side determines a solution only once its
# own constant part is out of the way, and determines it only up to a constant. The first
# degree of freedom is the one the factorisation dropped: it is set to zero, which picks one
# solution, and the mean is taken out afterwards, which picks the one in the complement of
# the kernel. Both steps are O(N). Two temporaries are allocated: the mean-free right-hand
# side, which is what the factor is given, and the one the CHOLMOD solve produces in any
# case.
function mass_solve!(y::AbstractVector, op::FactorizedMass{T, MT, FT, :project},
        x::AbstractVector) where {T, MT, FT}
    n = length(x)
    ŷ = op.fact \ @views(x[2:n] .- sum(x) / n)
    y[1] = zero(eltype(y))
    copyto!(view(y, 2:n), ŷ)
    y .-= sum(y) / n
    return y
end

# The banded factor does implement an in-place `ldiv!`, so this path allocates nothing.
function mass_solve!(y::AbstractVector, op::BandedMass, x::AbstractVector)
    y === x || copyto!(y, x)
    ldiv!(op.fact, y)
    return y
end

function mass_solve!(y::AbstractVector, op::CirculantMass, x::AbstractVector)
    mul!(op.buf, op.plan, x)
    op.buf .*= op.ĉ⁻¹
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
    mass_operator(M, basis; kernel = :reject)

Build the [`MassOperator`](@ref) appropriate to `basis`: a [`CirculantMass`](@ref) for a
periodic basis on a [`UniformMesh`](@ref), a [`FactorizedMass`](@ref) for a periodic basis on
any other mesh, and a [`BandedMass`](@ref) for a bounded basis — clamped or recombined —
whose mass matrix has no seam to wrap across.

`kernel = :project` passes the deflation on to whichever of the two periodic representations
is chosen, so that a caller with a singular assembly says what it wants rather than which
representation implements it. See [`MassOperator`](@ref) for what the deflation means and
each representation for how it is done.

!!! note "Circulance is a property of the basis, not of the mesh"
    A uniform mesh is necessary but not sufficient. The mass matrix is circulant only when
    every basis function is a translate of one cardinal spline, which needs the *periodic*
    closure as well: a clamped basis on a uniform mesh has ``p`` boundary functions at each
    end that are not translates of anything, and its mass matrix is banded but not circulant.
    Dispatching on the mesh alone would take the Fourier path for a clamped basis and get
    wrong answers everywhere except in [`CirculantMass`](@ref)'s own verification, which
    would reject it.
"""
function mass_operator(M::AbstractMatrix, ::AbstractBSplineBasis; kernel = :reject)
    # A name that is not a mode at all is reported as such. Reaching for the basis first would
    # say that `kernel = :ignore` is implemented for a periodic basis, which it is not.
    kernel === :project && throw(ArgumentError(
        "kernel = :project is implemented for a periodic basis only; a bounded basis takes " *
        "the banded representation, which carries no deflation"))
    kernel === :reject || throw(_not_a_kernel_mode(kernel))
    return BandedMass(M)
end

function mass_operator(M::AbstractMatrix, b::PeriodicBSplineBasis; kernel = :reject)
    mesh(b) isa UniformMesh ? CirculantMass(M, nbasis(b); kernel) :
    FactorizedMass(M; kernel)
end
