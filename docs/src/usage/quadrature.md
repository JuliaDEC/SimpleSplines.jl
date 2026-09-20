```@meta
CurrentModule = SimpleSplines
```

# [Assembly](@id usage-assembly)

Everything this package assembles comes out of one data structure. A
[`SplineQuadrature`](@ref) tabulates a basis and its derivatives at the Gauß-Legendre points
of every cell, and every matrix of the form

```math
A_{kl} = \int_\Omega f(x) \, D^a \phi_k \, D^b \phi_l \, \mathrm{d}x
```

is then a single weighted contraction of that table,

```math
A = \Phi_a \, \mathrm{diag}(f \odot w) \, \Phi_b^{\mathsf T} .
```

```@example asm
using SimpleSplines
using LinearAlgebra
using SparseArrays
```

## Construction

```julia
SplineQuadrature(basis; nq = quadrature_order(degree(basis)), dmax = 3)
```

Any of the three bases serves. What differs between them is which functions are nonzero on a
cell ([`local_indices`](@ref)) and which representation the mass matrix takes
([`mass_operator`](@ref)); the contractions are identical.

```@example asm
b = BSplineBasis(UniformMesh(16, 0 .. 2π), 3, Periodic())
q = SplineQuadrature(b)
nbasis(q), degree(q), ncells(q), length(quadrature_nodes(q))
```

  - **`nq`** is the number of quadrature points per cell. The default,
    [`quadrature_order`](@ref)`(p) = ⌈3p/2⌉`, integrates degree ``3p-1`` exactly — see
    [Choosing `nq`](@ref choosing-nq) below.
  - **`dmax`** is the highest derivative order tabulated. Three is what a third-order operator
    needs after one integration by parts. Asking [`basis_values`](@ref) for more is an error
    naming the fix.

```@example asm
quadrature_order.(0:5)
```

```@example asm
try
    basis_values(SplineQuadrature(b; dmax = 1), 2)
catch err
    println(err.msg)
end
```

## The table

| call | returns |
|:--|:--|
| [`quadrature_nodes`](@ref)`(q)` | the `n*nq` global points, in cell order |
| [`quadrature_weights`](@ref)`(q)` | the matching weights, scaled by the cell widths, so `sum(w) == L` |
| [`basis_values`](@ref)`(q, d = 0)` | the sparse `N × n*nq` table ``\Phi_d[i,r] = D^d\phi_i(x_r)`` |
| [`basis`](@ref)`(q)` | the basis |
| [`mass_operator`](@ref)`(q)` | the [`MassOperator`](@ref) |
| `nbasis`, `degree`, `order`, `ncells`, `domainlength`, `meshwidth`, `eltype` | forwarded to the basis |

```@example asm
sum(quadrature_weights(q)) ≈ domainlength(b)
```

The global index of the `r`-th point of cell `k` is `(k-1)*nq + r`, so a hand-written assembly
loop reads

```@example asm
Φ = basis_values(q, 0)
w = quadrature_weights(q)
nq = length(quadrature_nodes(q)) ÷ ncells(q)
total = sum(w[(k - 1) * nq + r] * Φ[i, (k - 1) * nq + r]
for k in 1:ncells(q) for i in local_indices(basis(q), k) for r in 1:nq)
total ≈ domainlength(b)              # ∑ᵢ ∫ φᵢ = ∫ 1 = L
```

### The table is sparse, and that is the point

A basis function is supported on `p+1` cells, so only `(p+1)*nq` of the `n*nq` entries in its
row can be nonzero, and only those are stored.

```@example asm
nnz(Φ), ncells(q) * (degree(q) + 1) * nq, nnz(Φ) / length(Φ)
```

Storing ``\Phi`` densely would make every contraction
``\Phi \, \mathrm{diag}(f \odot w) \, \Phi^{\mathsf T}`` cost ``O(N^2 n n_q)`` instead of
``O(N p^2 n_q)``.

## The matrices

| call | matrix |
|:--|:--|
| [`mass_matrix`](@ref)`(q)` | ``\mathbb{M}_{kl} = \int \phi_k \phi_l`` |
| [`stiffness_matrix`](@ref)`(q)` | ``\mathbb{K}_{kl} = \int \phi_k' \phi_l'`` |
| [`derivative_matrix`](@ref)`(q)` | ``S_{kl} = \int \phi_k \phi_l'`` |
| [`mixed_matrix`](@ref)`(q, a, b)` | ``\int D^a\phi_k \, D^b\phi_l`` |
| [`weighted_matrix`](@ref)`(q, f, a, b)` | ``\int f \, D^a\phi_k \, D^b\phi_l`` |
| [`basis_integrals`](@ref)`(q)` | the vector ``\int \phi_i`` |

`stiffness_matrix` and `derivative_matrix` are `mixed_matrix(q, 1, 1)` and
`mixed_matrix(q, 0, 1)` — the same objects, not merely equal ones:

```@example asm
stiffness_matrix(q) === mixed_matrix(q, 1, 1),
derivative_matrix(q) === mixed_matrix(q, 0, 1)
```

!!! note "`mixed_matrix` is memoised, and returns the cached object"
    These are constants of the discretisation and are asked for inside every Newton iteration
    of every step of a downstream integrator, so each `(a, b)` is assembled once and kept.
    The consequence is that the result is **shared**: mutating it changes what every later
    call returns. Take a `copy` before writing into one.

```@example asm
mixed_matrix(q, 0, 2) === mixed_matrix(q, 0, 2)
```

The same holds for [`mass_matrix`](@ref) and [`basis_integrals`](@ref), which are assembled
when the quadrature is built and returned by reference.

### Structural identities

On a periodic mesh `derivative_matrix` is *already* antisymmetric, with no
skew-symmetrisation needed, because integration by parts over a torus leaves no boundary term:

```@example asm
S = derivative_matrix(q)
maximum(abs, S + S')
```

The stiffness matrix is symmetric positive semi-definite with the constants in its kernel:

```@example asm
K = stiffness_matrix(q)
(maximum(abs, K - K'),
    minimum(eigvals(Symmetric(Matrix(K)))),
    maximum(abs, K * ones(nbasis(q))))
```

And integration by parts holds between the mixed matrices, which is what lets a third-order
operator be carried by a ``\mathcal{C}^2`` cubic basis:

```@example asm
maximum(abs, mixed_matrix(q, 1, 2) + mixed_matrix(q, 0, 3))
```

### A variable coefficient costs no more than a constant one

[`weighted_matrix`](@ref) takes either a function of the coordinate or a vector already
sampled at [`quadrature_nodes`](@ref) — the second form is what a coefficient that is itself a
spline expansion becomes, and it avoids resampling a field already in hand.

```@example asm
A1 = weighted_matrix(q, sin, 0, 1)
A2 = weighted_matrix(q, sin.(quadrature_nodes(q)), 0, 1)
A1 == A2, maximum(abs, weighted_matrix(q, one, 0, 0) - mass_matrix(q))
```

Its element type is the promotion of the quadrature's with the sample's, so a complex
coefficient field gives a complex matrix — unlike [`mixed_matrix`](@ref), which carries no
weight and is always `eltype(q)`:

```@example asm
eltype(weighted_matrix(q, cis, 0, 0)), eltype(mixed_matrix(q, 0, 0))
```

!!! warning "`weighted_matrix` allocates on every call, and there is no in-place form"
    It is the one assembly that cannot be memoised — it depends on the field, so a time
    integrator asks for a *different* one inside every Newton iteration. Measured at
    ``N = 128``, ``p = 3``, ``n_q = 5`` that is 260 kB per call, and almost all of it is the
    intermediate ``\Phi_a D`` and scratch inside the sparse-sparse product — the returned
    matrix is 15 kB of it. `weighted_matrix!` does not exist yet; a caller in that position
    should hold the matrix across steps and accept the allocation, or assemble the contraction
    by hand against a cached sparsity pattern. `scripts/weighted_matrix_allocation.jl`
    measures the breakdown.

## Projection

```julia
l2_projection(q, f)            # f a function, or a vector sampled at quadrature_nodes(q)
l2_projection!(û, q, f)        # in place
l2_projection!(s::Spline, q, f)
```

``\hat{u} = \mathbb{M}^{-1} \int_\Omega f \phi_i``, which is the best approximation of `f` in
the space in the ``L^2`` norm.

```@example asm
û = l2_projection(q, sin)
maximum(abs(evaluate(b, û, x) - sin(x)) for x in range(0, 2π; length = 401))
```

```@example asm
v̂ = similar(û)
l2_projection!(v̂, q, sin)
v̂ == û
```

The in-place form allocates nothing on every representation but one: the ``f \odot w`` product
goes into a buffer the quadrature holds, the load vector is formed with `mul!` straight into
`û`, and both the [`CirculantMass`](@ref) and the [`BandedMass`](@ref) solve are themselves
allocation-free. The exception is a periodic basis on a non-uniform mesh — the
[`FactorizedMass`](@ref) case — where CHOLMOD has no in-place solve.

```@example asm
fq = sin.(quadrature_nodes(q))
l2_projection!(v̂, q, fq)                       # warm up
@allocated l2_projection!(v̂, q, fq)
```

A sample whose element type is wider than the quadrature's — a complex `f` — gets its own
product rather than being narrowed into that buffer, so the in-place form accepts exactly what
the allocating form accepts and merely stops being allocation-free there.

## Mass operators

Solves against the mass matrix go through a [`MassOperator`](@ref), and **which representation
is built is decided by the basis, not by the mesh**. A uniform mesh is necessary for
circulance but not sufficient: a clamped basis on one has ``p`` boundary functions at each end
that are not translates of anything.

| basis | mesh | representation | a solve is |
|:--|:--|:--|:--|
| [`PeriodicBSplineBasis`](@ref) | [`UniformMesh`](@ref) | [`CirculantMass`](@ref) | two planned transforms and a pointwise multiplication, ``O(N\log N)``, no allocation |
| [`PeriodicBSplineBasis`](@ref) | any other | [`FactorizedMass`](@ref) | a sparse Cholesky; the only representation whose solve allocates |
| bounded — clamped or recombined | any | [`BandedMass`](@ref) | a banded Cholesky, ``O(Np)``, no allocation |

```@example asm
for (name, bb) in (("periodic, uniform", BSplineBasis(UniformMesh(16, 0 .. 2π), 3, Periodic())),
    ("periodic, graded", BSplineBasis(GradedMesh(16, 0 .. 2π), 3, Periodic())),
    ("periodic, random", BSplineBasis(RandomMesh(16, 0 .. 2π), 3, Periodic())),
    ("clamped, uniform", BSplineBasis(UniformMesh(16, 0 .. 2π), 3)),
    ("Dirichlet, random", BSplineBasis(RandomMesh(16, 0 .. 2π), 3, Dirichlet())))
    println(rpad(name, 20), nameof(typeof(mass_operator(SplineQuadrature(bb)))))
end
```

The circulant construction *verifies* circulance rather than assuming it, to `rtol =
sqrt(eps(T))` by default — *relative* to the largest entry of the first column, so that the
verdict survives a rescaling of the assembly and holds in every element type. A matrix that is
banded but not circulant would still produce plausible numbers through the transform, and the
failure would surface much later as a wrong conservation law.

```@example asm
qg = SplineQuadrature(BSplineBasis(GradedMesh(16, 0 .. 2π), 3, Periodic()))
try
    CirculantMass(mass_matrix(qg), nbasis(qg))
catch err
    println(err.msg)
end
```

### A singular assembly

The mass matrix is positive definite, but the same three representations carry the other
assemblies of a quadrature, and [`stiffness_matrix`](@ref) on a periodic basis is *singular*:
``-\phi'' = \rho`` says nothing about the constants, and a periodic basis represents them.

```@example asm
bp = BSplineBasis(UniformMesh(64, 0 .. 2π), 3, Periodic())
qp = SplineQuadrature(bp)
Sp = stiffness_matrix(qp)
N = nbasis(bp)
maximum(abs, Sp * ones(N))
```

By default [`mass_operator`](@ref) refuses it, since a singular assembly is usually too coarse
a quadrature rather than an intended one:

```@example asm
try
    mass_operator(Sp, bp)
catch err
    println(err.msg)
end
```

`kernel = :project` asserts that the kernel is the constants, and asks for the solution that
has no constant component. Each representation deflates that subspace in the way its own
structure allows — [`CirculantMass`](@ref) gives the constant mode a zero factor, which is the
Moore–Penrose pseudoinverse, and [`FactorizedMass`](@ref) factorises the minor that drops the
first degree of freedom and takes the mean out afterwards. Both verify the assertion, and
neither changes the matrix:

```@example asm
op = mass_operator(Sp, bp; kernel = :project)
rhs = sin.(2π .* (0:(N - 1)) ./ N)     # one whole period, so its mean is already zero
φ = op \ rhs
maximum(abs, Sp * φ - rhs), sum(φ)     # it solves the system, in the mean-free gauge
```

The textbook cure is the rank-one shift by the mean projector,
``\mathbb{S} + \mathbb{1}\mathbb{1}^T/N``, which is invertible and gives the same answer. It
is also structurally full, so it costs ``O(N^2)`` to store and to factorise where the
assembly itself costs ``O(N)``:

```@example asm
shifted = mass_operator(Matrix(Sp) .+ inv(N), bp)
maximum(abs, φ - shifted \ rhs),
Base.summarysize(Sp), Base.summarysize(Matrix(Sp) .+ inv(N))
```

### Solving

```julia
mass_solve!(y, op, x)          # in place; y and x may alias
ldiv!(y, op, x)   ldiv!(op, x)
op \ x                         # x a vector or a matrix of right-hand sides
mass_matrix(op)   Matrix(op)   size(op)   eltype(op)
```

```@example asm
op = mass_operator(q)
rhs = basis_integrals(q)
x = op \ rhs
maximum(abs, x .- 1)             # M⁻¹ ∫φᵢ = 1, by the partition of unity
```

```@example asm
y = similar(rhs)
mass_solve!(y, op, rhs)
y == x, (@allocated mass_solve!(y, op, rhs))
```

[`mass_factorization`](@ref)`(q)` is the same operator, named for the case where the
factorisation rather than the matrix is what is wanted.

## [Choosing `nq`](@id choosing-nq)

An `nq`-point Gauß-Legendre rule is exact to degree ``2n_q - 1``. Three thresholds matter:

| what needs to be exact | degree | requires |
|:--|:--|:--|
| the mass matrix ``\int \phi_k \phi_l`` | ``2p`` | ``n_q \ge p + 1`` |
| the antisymmetry of [`derivative_matrix`](@ref), i.e. ``\int (\phi_k\phi_l)'`` | ``2p - 1`` | ``n_q \ge p`` |
| a quadratic nonlinearity, ``\int u_h u_{h,x} \phi_i`` | ``3p - 1`` | ``n_q \ge \lceil 3p/2 \rceil``, the default |

Below the second threshold the antisymmetry does not degrade gracefully — it fails outright,
and only on a non-uniform mesh, because a uniform periodic mesh gives circulant assemblies for
which it holds at any `nq`. That is exactly the case where a check run on a uniform mesh
confirms the property for the wrong reason:

```@example asm
p = 3
for nq in 2:6
    qu = SplineQuadrature(BSplineBasis(UniformMesh(16, 0 .. 2π), p, Periodic()); nq = nq)
    qr = SplineQuadrature(BSplineBasis(RandomMesh(16, 0 .. 2π), p, Periodic()); nq = nq)
    du = maximum(abs, derivative_matrix(qu) + derivative_matrix(qu)')
    dr = maximum(abs, derivative_matrix(qr) + derivative_matrix(qr)')
    println("nq = ", nq, "   uniform ", rpad(du, 24), "   random ", dr)
end
```

Too coarse an `nq` makes the mass matrix **singular** rather than merely inexact, and the
constructor says so by name:

```@example asm
try
    SplineQuadrature(BSplineBasis(UniformMesh(8, 0 .. 1), 3); nq = 1)
catch err
    println(err.msg)
end
```

## Two traps

!!! warning "One quadrature per thread"
    A `SplineQuadrature` carries mutable state behind an otherwise read-only interface:
    [`mixed_matrix`](@ref) memoises into a `Dict`, and [`l2_projection!`](@ref) forms its
    ``f \odot w`` product in a shared buffer. Neither is synchronised, so **one quadrature must
    not be used from two threads at once.** Give each thread its own, or stay with the
    allocating [`l2_projection`](@ref), which forms its own product.

!!! warning "Several results are returned by reference"
    [`mass_matrix`](@ref), [`basis_integrals`](@ref), [`basis_values`](@ref) and every
    [`mixed_matrix`](@ref) hand back the quadrature's own storage. `copy` before mutating.
