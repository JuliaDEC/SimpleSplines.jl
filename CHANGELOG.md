# Changelog

All notable changes to SimpleSplines.jl are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first release with any implementation in it. Before this the package was a
`PkgTemplates` skeleton: a registered UUID, CI, docs scaffolding and a declared dependency
set, with `src/SimpleSplines.jl` a bare `module … end`.

### General boundary conditions and tensor products

The package was periodic-only: one basis, on `[0,L)`, on a torus. It now covers bounded
intervals with arbitrary homogeneous boundary conditions, and tensor products of any number
of such bases with independent degree, mesh, domain and boundary condition on each axis.

Nothing here was released before, so the reshaping below is not a breaking change to anything
outside the package. Within it, two things moved: `Mesh` now describes a partition of a closed
interval rather than a periodic one, and `breakpoints` accordingly returns `n+1` points instead
of `n`. `cellbounds` is gone — it was `breakpoints` plus the periodic image of the first point,
which is now what `breakpoints` itself returns.

#### The three bases

| basis | closure | dimension | reproduces |
|:--|:--|:--|:--|
| `BSplineBasis` | clamped — end knots repeated `p+1` times | `n + p` | polynomials of degree `≤ p` |
| `PeriodicBSplineBasis` | breakpoints continued periodically | `n` | constants |
| `RecombinedBSplineBasis` | clamped, plus one constraint per constrained end | `n + p − #constraints` | see below |

`BSplineBasis(mesh, p, bc)` returns whichever of the three the boundary condition calls for.
That is deliberate rather than an accident of naming: it lets one call site select any of them,
which is what a tensor product with a different condition on each axis needs.

#### Boundary conditions are types, not symbols

`Periodic`, `Free`, `Dirichlet`, `Neumann`, `Natural`, `Robin(α, β)` and the general local
`Constraint(c...)`. A single condition applies to both ends, a two-tuple gives left and right,
and lowercase symbols are accepted as sugar and normalised on construction.

Three reasons this is not a symbol:

- **`Robin` carries coefficients.** `:robin` has nowhere to put `α` and `β`.
- **The ends can differ.** `(Dirichlet(), Neumann())` is the natural spelling; `Periodic` is
  rejected in a pair, because it identifies the two ends rather than constraining one.
- **A typo fails at the call site.** In the code this replaces, an unrecognised symbol fell
  through an `else` branch to the unconstrained basis, so `:nothing` selected the clamped basis
  *by accident* and `:Perodic` would have done the same silently. `Free()` says it, and
  `BoundaryCondition(:perodic)` throws and names the accepted set.

Two names changed meaning and therefore throw rather than being accepted case-insensitively:
`:Natural` used to mean the *unconstrained* clamped basis, which is now `Free()`, while
`:natural` now means the actual natural condition `u'' = 0`. Silently handing back a different
function space than a caller's old code had is exactly the failure this API exists to prevent,
so the error spells the rename out.

Constraints are imposed by **recombination**: for a condition `L u(a) = 0` of order `m`, the
`m+1` functions that violate it are replaced by the `m` combinations
`ψ_t = φ_t − (Lφ_t(a) / Lφ_{m+1}(a)) φ_{m+1}` that `L` annihilates identically. Each
constrained end costs exactly one degree of freedom whatever the order of its condition, and
for `Dirichlet` the construction degenerates to dropping `φ_1` — the textbook elimination.
The mass matrix stays symmetric positive definite and banded, and every assembly is the
parent's conjugated by the sparse recombination matrix, `M̃ = Rᵀ M R`, which is why
`SplineQuadrature` needed no recombined special case.

Deliberately out of scope: **nonlocal** constraints. Imposing `u(a) = u(b)` other than through
`Periodic`, or an integral condition, needs the nullspace of a dense constraint matrix, which
destroys the banding every assembly here relies on.

#### `polynomial_reproduction`

Reports the largest `m` with every polynomial of degree `≤ m` in the span: `p` for a clamped
basis, `0` for a periodic one, `-1` for a Dirichlet-recombined one, and the minimum over the
axes for a tensor product.

This exists because the boundary condition is **not** a free choice in a conservative scheme.
A Galerkin or particle discretisation conserves `∫ π(v) f dv` for a polynomial `π` exactly when
`π` is in the span of the basis, so a scheme whose conservation of mass, momentum and energy
rests on `1, v, v²` needs the value to be at least `2` — which a periodic axis (`v` is not
periodic) and a Dirichlet condition (nothing survives, not even the constants) both fail.
Turning that from a property a reader has to know into a number a test can assert is the whole
point.

#### `evaluate_all` — local evaluation

`evaluate_all!(values, b, x, d)` gives the `local_width(b)` basis functions that do not vanish
at `x`, and the index of the first. Depositing `N_p` particles costs `O(N_p p²)` through it and
`O(N_p N)` through `evaluate` one index at a time, which is the difference between a loop over
the `p+1` functions overlapping a particle and a loop over the whole basis. It is
allocation-free, and the suite asserts that: the breakpoints are cached in the basis rather than
rebuilt from the mesh on every `findcell`, which was a 320-byte allocation per call in the
innermost loop of a deposition.

`evaluate_all` runs de Boor's triangular scheme and then lifts the block from degree `p−d` to
degree `p` by the derivative recursion. `_bspline` — the Cox-de Boor recursion written out as it
stands — remains the reference, and the suite checks the two against each other at every degree,
every derivative order, all four mesh families and every boundary condition, and separately that
nothing outside the reported block is nonzero.

#### `GeneralMesh`

A mesh from explicit breakpoints, for a subdivision none of the three families describes. The
case it exists for is a mesh that is uniform inside but carries one oversized cell at each end —
the device particle discretisations of kinetic equations use to keep a particle that strays
outside the resolved region inside the support of the basis.

#### Domains via `DomainSets`

`domain(mesh)` and `domain(basis)` return a `ClosedInterval`; `domain` of a tensor product
returns a `ProductDomain`, so `x ∈ domain(B)` answers for a vector. `DomainSets` re-exports the
`IntervalSets` `..` that `ContinuumArrays` already used, so `0 .. 1` is the same binding reached
through two re-exporters rather than two competing ones, and `..` is re-exported here so that
`using SimpleSplines` is enough to write a domain.

Mesh constructors take a domain as an interval, a tuple, or a single number `L` standing for
`[0,L]`. An integer domain is promoted: `UniformMesh(4, 0 .. 1)` would otherwise try to store
`0.25` in an `Int` and throw from inside `breakpoints`, a long way from the constructor that
chose the type. `Rational` and extended-precision types are left alone, being closed under the
division `breakpoints` performs.

#### `TensorProductBasis` and `TensorProductQuadrature`

`TensorProductBasis(b₁, …, b_D)`, or `b₁ ⊗ b₂`. Coefficients are a `D`-dimensional array of
size `size(B)`, which is the shape that makes the Kronecker structure of every operator visible
and lets `CartesianIndices` do the index arithmetic — one of the places a hand-rolled tensor
product reliably goes wrong, since the flattening convention has to agree between the
evaluation, the mass matrix and the projection.

The Kronecker structure is **used**, not merely noted:

- `KroneckerMass` never forms `M = M⁽ᴰ⁾ ⊗ … ⊗ M⁽¹⁾`. A solve applies each factor's inverse
  along its own axis, which is an identity and not an approximate splitting. Each factor keeps
  its own representation, so a periodic uniform axis still takes the Fourier path while a
  clamped one takes the Cholesky. At the 41-element cubic basis of a two-dimensional velocity
  space this avoids a 1681×1681 dense Cholesky; in three dimensions it avoids a 68921² one,
  which does not fit.
- `contract` assembles a load array as `D` successive sparse contractions with the
  one-dimensional tabulations, cycling each contracted axis to the back so that every step is
  one sparse matrix product. The integrand need not be separable — only the basis is, and that
  is enough. This is what `l2_projection` uses, and `quadrature_sample` exposes the grid for an
  integrand that is built from a spline already in hand rather than given as a function of
  position.
- There is no `D`-dimensional tabulation. Holding the per-axis ones is `O(Σ_d N_d n_d n_{q,d})`
  rather than the `O(N Π_d n_d n_{q,d})` a `D`-dimensional table would cost.

`mass_operator` now dispatches on the **basis** rather than the mesh. A uniform mesh is
necessary but not sufficient for circulance: it also needs the periodic closure, since a clamped
basis has `p` boundary functions at each end that are not translates of anything. Dispatching on
the mesh alone, as the periodic-only version did, would take the Fourier path for a clamped
basis.

One trap is recorded in the source alongside the FFTW planner one below, for the same reason.
`_apply_along!` copies each fibre into a contiguous buffer rather than passing a view: a view
into the middle index of a three-way reshape has a stride, and an FFTW plan encodes the strides
of the array it was planned for, not merely its alignment — so the `CirculantMass` plan rejects
such a fibre outright with "plan applied to wrong-strides array". Creating the plans `UNALIGNED`,
which is what lets them accept a *contiguous* column view, does not help.

### Periodic B-spline finite elements

The package now provides one basis and one assembly table built on it.

`PeriodicBSplineBasis` is the B-spline basis of arbitrary degree on the torus, built from the
Cox-de Boor recursion written out as it stands rather than through a faster equivalent, so
that the assembly doubles as a check of the formulae. Derivatives of arbitrary order come
from the same recursion; the discrete brackets downstream need up to third order, and the
lazy `Derivative` products of `ContinuumArrays` do not compose that deep.

The dimension is `N = n`, the number of *cells* — not `n + p`. On a torus the `p` extra
functions of the bounded case are the ones the clamping introduces at the two ends, and the
periodic wrap identifies them in pairs. There are no boundary functions at all: every basis
function spans `p+1` cells and the basis is `C^{p-1}` across the seam as well as inside, so
integration by parts over the domain leaves no boundary terms.

Three mesh families are provided, and the difference between them is about testing rather
than about the discretisation:

| mesh | refining `n` gives | use it for |
|:--|:--|:--|
| `UniformMesh` | a uniform refinement | ordinary computation — but its assemblies are *circulant*, so it confirms some identities for the wrong reason |
| `GradedMesh` | a genuine mesh family | convergence rates |
| `RandomMesh` | an unrelated mesh | properties that must hold on *any* mesh |

The bases join the `Basis` hierarchy of `CompactBasisFunctions` and extend the accessor
generics of `GeometricBase` rather than redefining them, so that the packages of the
ecosystem share one function per accessor.

### `SplineQuadrature`

The assembly table: the basis and its derivatives tabulated at the global Gauß-Legendre
points, with the weights and the mass matrix. Every matrix of the form
`∫ f(x) Dᵃφ_k Dᵇφ_l dx` is then one weighted contraction, `Φₐ diag(f ⊙ w) Φᵦᵀ`, which is
what makes a variable coefficient cost no more than a constant one.

`quadrature_order(p) = ⌈3p/2⌉` by default, which integrates degree `3p-1` exactly — what the
consistency of a Galerkin discretisation of a quadratic nonlinearity needs. Two lower
thresholds are worth knowing and are documented: the mass matrix needs `nq ≥ p+1`, and the
antisymmetry of `derivative_matrix` needs `nq ≥ p` and fails outright below it on a
non-uniform mesh.

### `MassOperator`

Solves against the mass matrix go through a representation chosen by the mesh:

- `CirculantMass` on a `UniformMesh`, where the basis functions are translates of a single
  cardinal spline and the matrix is circulant, hence diagonalised by the discrete Fourier
  transform. A solve is two planned transforms and a pointwise division, and allocates
  nothing beyond its result.
- `FactorizedMass` otherwise. On a graded or random mesh the matrix is banded modulo `N` but
  *not* circulant — the basis functions are no longer translates of one another — so there is
  nothing for a transform to diagonalise and a sparse Cholesky is what is left.

The construction *verifies* circulance rather than assuming it. A matrix that is banded but
not circulant would still produce plausible numbers through the transform, and the failure
would surface much later as a wrong conservation law downstream.

One trap is recorded in the source because it cost real time to find:
`plan_rfft(c; flags = FFTW.UNALIGNED)` *replaces* the planner flags rather than adding to
them, and FFTW's default rigor then measures — which overwrites the array being planned for.
That destroys the first column before its transform is taken, and the symptom is a mass
matrix that appears singular at some sizes and not others.
`FFTW.ESTIMATE | FFTW.UNALIGNED` is the correct spelling.

### Storage and cost

The basis tabulation is **sparse**. A basis function is supported on `p+1` cells, so only
`(p+1)·nq` of the `n·nq` entries in its row are structurally nonzero, and storing `Φ` densely
makes every contraction cost `O(N² n nq)` where it should cost `O(N p² nq)`. Measured from
[PoissonBrackets.jl](https://github.com/JuliaGNI/PoissonBrackets.jl) at `N = 384`, that alone
was the difference between 6.0 ms and 0.4 ms for assembling one Hessian.

The constant assemblies — `mass_matrix`, `stiffness_matrix`, `derivative_matrix` and any
`mixed_matrix` — are memoised on first use. They do not depend on the field, but a downstream
time integrator asks for them inside every Newton iteration of every step. `basis_integrals`
is one of these constants too, and is now assembled with the quadrature and returned by
reference rather than recomputed per call.

The paths a time integrator actually runs per step allocate nothing. Evaluation is
allocation-free at any degree and derivative order on any mesh; a `CirculantMass` solve is
allocation-free; and `l2_projection!` is now allocation-free end to end on a uniform mesh,
the `f ⊙ w` product going into a buffer held by the quadrature and the load vector being
formed with `mul!` straight into the output. The buffer is taken only when the product lands
in the quadrature's element type, so a wider sample — a complex `f` — is still projected,
through a product of its own, rather than narrowed into it. On a non-uniform mesh a CHOLMOD
temporary remains, CHOLMOD having no in-place `ldiv!`.

That buffer is mutable state on a struct that reads as immutable, as the memoising `cache`
behind `mixed_matrix` already was, so the `SplineQuadrature` docstring now warns that one
quadrature must not be shared between threads.

### Dependencies

`BSplineKit` was dropped: the basis is implemented here, and its variable-coefficient
assemblies do not map onto that package's Galerkin interface. `FFTW` and `SparseArrays` were
added for the above, and `CompactBasisFunctions` for the shared `Basis` hierarchy.

### Repository and CI

The documentation build moved out of `CI.yml` into its own `Documenter.yml` workflow, so that
a docs failure and a test failure are separate signals and the docs job is not queued behind
the test matrix. The README gained a badge for it.

The workflows were brought up to current action versions — `actions/checkout@v7`,
`julia-actions/setup-julia@v3`, `julia-actions/cache@v3`, `codecov/codecov-action@v7` — and
the test matrix now names Julia versions by alias rather than by number:

- `min` resolves the lower bound of the `julia` compat entry, so the matrix tracks the
  declared support window instead of having to be edited alongside it;
- `lts` and `1` cover the long-term-support and current stable releases;
- `pre` and `nightly` run on Linux only and are `continue-on-error`, since an upcoming-release
  failure is information rather than a broken build.

`arch` is now `default` rather than `x64`, which is what tests Julia natively on the ARM64
macOS runners instead of under Rosetta.

`AUTHORS.md` was added and `LICENSE` renamed to `LICENSE.md`, whose copyright line now names
"The SimpleSplines Authors" and points at it.

The `pre-push` hook the README asks the reader to enable now exists in `.githooks/`. It runs
the test suite and refuses the push if it fails; `SIMPLESPLINES_SKIP_TESTS=1` overrides it.

### Fixed

- `QuadratureRules` compat was `"0.1"`, which could not co-resolve with
  `CompactBasisFunctions`; it is now `"0.2"`.
- `LinearAlgebra` compat was `"1.12.0"` alongside `julia = "1.10"`, which contradicted the
  1.10 row of the CI matrix; it is now `"1"`.
- `docs/Project.toml` pinned `CompactBasisFunctions` to an absolute path on a developer's
  machine through a `[sources]` entry. That path does not exist on a CI runner, so the
  documentation build could only ever have succeeded locally; the entry is removed and the
  dependency now resolves from the registry.
- `CompatHelper.yml` invoked `julia` without installing it. The runner images no longer ship
  a Julia, so the workflow failed before CompatHelper started; it now sets Julia up first.

## Open Issues

### `weighted_matrix` allocates a fresh matrix per call

`weighted_matrix` is the one assembly that cannot be memoised — it depends on the field, so a
downstream time integrator asks for a *different* one inside every Newton iteration of every
step, which is exactly the call pattern under which allocation matters most. Measured at
`N = 128`, `p = 3`, `nq = 5`:

| | bytes |
|:--|--:|
| `f .* q.w` temporary | 5 kB |
| `Φₐ * Diagonal(f ⊙ w)` | 47 kB |
| `(Φₐ D) * Φᵦᵀ` sparse-sparse product | 208 kB |
| **total per call** | **260 kB** |

The `f ⊙ w` temporary is the same one `l2_projection!` no longer pays and could be removed the
same way, with the buffer the quadrature already holds. The other 255 kB are not a temporary
at all: they are the result, a freshly built `SparseMatrixCSC` with its `colptr`, `rowval` and
`nzval` allocated and its structure recomputed from scratch.

That structure does not depend on `f`. For a fixed `(a, b)` the sparsity pattern of
`Φₐ diag(f ⊙ w) Φᵦᵀ` is the same for every coefficient — a basis function overlaps only the
`2p+1` others whose supports meet its own — so the pattern could be assembled once per
`(a, b)`, cached beside the `mixed_matrix` results, and only `nzval` refilled per
call. That turns 260 kB into zero.

What it needs is an in-place entry point, `weighted_matrix!(A, q, f, a, b)`, since the present
signature has nowhere to write. Callers holding a matrix across steps would use it and callers
wanting a value would keep the allocating form. Deferred rather than done because it widens
the API, and because the sparse triple product would have to be written out by hand against
the cached pattern instead of delegating to `SparseArrays`, which is the part that needs to be
got right rather than merely written.
