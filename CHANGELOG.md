# Changelog

All notable changes to SimpleSplines.jl are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first release with any implementation in it. Before this the package was a
`PkgTemplates` skeleton: a registered UUID, CI, docs scaffolding and a declared dependency
set, with `src/SimpleSplines.jl` a bare `module … end`.

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
formed with `mul!` straight into the output. That buffer makes `l2_projection!`
non-reentrant, which its docstring says. On a non-uniform mesh a CHOLMOD temporary remains,
CHOLMOD having no in-place `ldiv!`.

### Dependencies

`BSplineKit` was dropped: the basis is implemented here, and its variable-coefficient
assemblies do not map onto that package's Galerkin interface. `FFTW` and `SparseArrays` were
added for the above, and `CompactBasisFunctions` for the shared `Basis` hierarchy.

### Fixed

- `QuadratureRules` compat was `"0.1"`, which could not co-resolve with
  `CompactBasisFunctions`; it is now `"0.2"`.
- `LinearAlgebra` compat was `"1.12.0"` alongside `julia = "1.10"`, which contradicted the
  1.10 row of the CI matrix; it is now `"1"`.
