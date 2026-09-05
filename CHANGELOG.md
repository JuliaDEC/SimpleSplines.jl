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
basis, `0` for a periodic one, and the minimum over the axes for a tensor product. For a
recombined basis it is one less than the order of the lowest derivative either condition
involves, minimised over the two ends — `-1` for `Dirichlet`, `0` for `Neumann`, `1` for
`Natural`, since a linear polynomial has vanishing second derivative everywhere.

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

#### Argument and dispatch holes closed after review

Five defects found by review rather than by the suite, all in exported API and none of them
visible to the 63 875 assertions that were passing at the time.

`mass_matrix` and `Matrix` of a `KroneckerMass` threw `MethodError` on a one-factor product.
The assembly splatted into `kron`, which has no one-argument method, so a `D = 1`
`TensorProductBasis` — explicitly legal, and constructed by the suite — could be solved with
but not assembled. Nothing else built a `TensorProductQuadrature` on one axis. The assembled
matrix is a copy at `D = 1` as well, where the fold is the identity and would otherwise hand
back the factor's own storage rather than the matrix formed on demand that is documented.

`derivative(s, k)` on a tensor-product `Spline` silently did the wrong thing whenever the
coefficients were wider than the basis. `Spline`'s element type is the promotion of the two,
so binding it to the basis type as well made the axis-selecting method unreachable for, say,
complex coefficients on a real basis: the fallback stored the bare integer `k` in the
derivative slot and the result raised `MethodError` when called. An out-of-range `k` now
throws instead of returning the zeroth derivative.

`RecombinedBSplineBasis` built a basis with **no functions at all** rather than refusing.
The existing guard enforced that the two end blocks stay disjoint, which is a weaker
requirement than the result spanning something: each constrained end costs one degree of
freedom, so `BSplineBasis(UniformMesh(1, 0 .. 1), 1, Dirichlet())` produced a basis of
dimension zero that answered the whole interface and reproduced nothing. Both conditions are
now checked, and the docstring states them separately.

`isequal` on a `Mesh` is now type-aware. `==` remains geometric — the domain and the
breakpoints, not the family that produced them — but the mesh *type* selects the assembly
path, an equally spaced `GeneralMesh` deliberately taking the general route where the
`UniformMesh` with identical breakpoints takes the circulant one. The two therefore no longer
collide as dictionary keys. `hash` stays geometric, which the `isequal ⟹ hash` contract
permits.

One smaller one: `size(::KroneckerMass, d)` answered for `d ≤ 0` where `Base` throws a
`BoundsError`, and now throws it too.

`weighted_matrix` is unchanged, but its element type is now stated and tested: the promotion of
the quadrature's with the sample's, so that a complex coefficient field gives a complex matrix.
It is the one assembly with a user-supplied weight, and therefore the one that cannot narrow to
`eltype(q)` the way `mixed_matrix` does — a narrowing that would `InexactError` on a complex
field and silently round a `Float64` one sampled against a `Float32` basis, which is what
`l2_projection!` takes its own care to avoid.

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

Solves against the mass matrix go through a representation chosen by the **basis**:

- `CirculantMass` for a periodic basis on a `UniformMesh`, where the basis functions are
  translates of a single cardinal spline and the matrix is circulant, hence diagonalised by
  the discrete Fourier transform. A solve is two planned transforms and a pointwise division,
  and allocates nothing beyond its result.
- `BandedMass` for a bounded basis, clamped or recombined. There is no seam, so the overlaps
  are contiguous and the matrix is banded outright rather than banded modulo `N`. A banded
  Cholesky solves it in `O(Np)` and — unlike CHOLMOD, which has no in-place `ldiv!` — solves
  it with **no allocation at all**.
- `FactorizedMass` for a periodic basis on a graded or random mesh. There the matrix is banded
  modulo `N` but *not* circulant — the basis functions are no longer translates of one
  another — so there is nothing for a transform to diagonalise, and the wrap-around entries
  put it outside the banded representation too. A sparse Cholesky is what is left.

Dispatching on the basis rather than the mesh is what makes this correct: a uniform mesh is
necessary for circulance but not sufficient, since a clamped basis on one has `p` boundary
functions at each end that are not translates of anything.

The construction *verifies* circulance rather than assuming it. A matrix that is banded but
not circulant would still produce plausible numbers through the transform, and the failure
would surface much later as a wrong conservation law downstream. The check walks the stored
entries rather than probing all `N²` positions: within a column the row indices are distinct
and `r = mod1(i-j+1, N)` is a bijection, so comparing the count of above-tolerance hits with
the count in the first column covers the unstored positions as well. At `N = 512` that is
0.004 ms against 9.1 ms, and it is what keeps `SplineQuadrature` construction linear in `n`
on a periodic uniform mesh instead of growing as roughly `n^2.7`.

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
allocation-free at any degree and derivative order on any mesh; a `CirculantMass` and a
`BandedMass` solve are both allocation-free; and `l2_projection!` is allocation-free end to
end on every basis except the periodic non-uniform one, the `f ⊙ w` product going into a
buffer held by the quadrature and the load vector being formed with `mul!` straight into the
output. The buffer is taken only when the product lands in the quadrature's element type, so
a wider sample — a complex `f` — is still projected, through a product of its own, rather
than narrowed into it. Only the periodic non-uniform case keeps a CHOLMOD temporary, that
being the one representation with no in-place solve.

Evaluating a spline — `evaluate(b, û, x)`, and therefore a callable `Spline` — sums the local
block of `p+1` functions rather than the whole basis, which it did until this was measured:
at `N = 1027` a single evaluation cost 24 µs and scaled linearly with `N`. It is now flat in
`N`, 21.8 ns per point at every size from 64 to 4096 cells. Outside a bounded domain the
result is still zero, which the local path has to be told, `findcell` clamping to the nearest
cell where de Boor's recursion would otherwise extrapolate its polynomial. That guard sits in
`evaluate_all!` itself, so every caller of the local block inherits it — the one-dimensional
sum, the tensor-product one, and a hand-written particle deposition alike.

Evaluating at a *vector* of points shares one block buffer across the sweep, so
`evaluate(b, û, xs)` — and `s(xs)`, which routes to it — costs 13.6 ns and 8 bytes per point
against the 22.1 ns and 190 bytes of going through the scalar method once per point. The
scalar method still takes a buffer per call: the degree is a field here rather than a type
parameter, so the block length is a run-time value and cannot live on the stack. A
`Val`-sized `MVector` was measured and rejected — it halves the allocation and gives the
saving straight back in dispatch, `evaluate_all!` taking an `AbstractVector` into which the
buffer escapes either way.

That buffer is mutable state on a struct that reads as immutable, as the memoising `cache`
behind `mixed_matrix` already was, so the `SplineQuadrature` docstring now warns that one
quadrature must not be shared between threads.

### Dependencies

`BSplineKit` was dropped: the basis is implemented here, and its variable-coefficient
assemblies do not map onto that package's Galerkin interface. `FFTW` and `SparseArrays` were
added for the above, and `CompactBasisFunctions` for the shared `Basis` hierarchy.

`BandedMatrices` was added for `BandedMass`. It adds nothing to the dependency tree: it is
already a hard dependency of `ContinuumArrays`, so this only makes an existing transitive
dependency explicit.

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

Found in review of the branch, not by the suite, and each now has a regression test:

- **`evaluate_all!` extrapolated outside the domain, and so did every tensor-product
  evaluation.** The guard was on the one-dimensional sum rather than on the block it sums:
  `findcell` clamps to the nearest cell, so de Boor's scheme evaluated that cell's polynomial
  at a point outside it. On a degree-3 basis on `0 .. 1` the block at `x = -0.5` came out as
  `[125.0, -196.0, 82.67, -10.67]` instead of zero, and `evaluate(B, û, x)` on a tensor
  product reached `-418` where the answer is `0` — in exactly the straying-particle case the
  `evaluate` docstring names, and with the identity in its own doctest failing off-domain.
  The guard now sits in `evaluate_all!`, which is the routine that had the fault and the one
  every other path goes through. It costs nothing measurable: the out-of-domain call is
  allocation-free and, short-circuiting the recursion, faster than an interior one.
- **`polynomial_reproduction` understated what a recombined basis reproduces.** It reported
  `0` whenever both conditions admitted the constants and `-1` otherwise, which is right only
  for conditions on `u` and `u'`. A `Natural` basis (`u'' = 0`) reproduces the linears as well,
  and `Constraint(0, 0, 0, 1)` (`u''' = 0`) the quadratics, so both were reported one or two
  degrees low. Understating this is not the safe direction: the number is the predicate a
  conservative scheme tests, so `polynomial_reproduction(b) ≥ 2` refused a `u''' = 0` basis at
  `p = 3` that does preserve all three moments. The value is now derived rather than
  enumerated — one less than the order of the lowest derivative either condition involves,
  minimised over the ends — and the suite checks it against an L² projection at every degree
  for eight conditions, monomials up to the reported degree being exact and the next one not.
- **`nodes` on a recombined basis returned repeated points.** The abscissa of a recombined
  function is meant to be that of the parent function it carries with unit coefficient; the code
  took the one of largest magnitude instead, which for an end block is the anchor row every one
  of that block's columns shares. On `Natural()` at `p = 3`, `n = 8` that gave 7 distinct points
  out of 9 and on `Constraint(0, 0, 0, 1)` five, so a collocation matrix built on them was
  singular — rank 7 and 5 against 9 on the correct abscissae. `nodes` is
  `ContinuumArrays.grid`, so this reached anything that asked the basis for a grid. The rows are
  now recorded by the recombination that creates them rather than recovered afterwards by
  searching `R`, which also removes a tie that is not hypothetical: for `Neumann` the anchor
  coefficient is exactly `1.0`, the two end derivatives being `∓p/h`, so searching a column for
  the value `1` cannot tell the two rows apart.
- **`TensorProductQuadrature` left its basis field abstract.** `basis::TensorProductBasis{T, D}`
  does not fix the tuple of bases, so `size`, `contract` and `l2_projection` inferred `Any`
  through it; a caller differing only in that took 19 936 bytes and 9.67 µs against 12 912 bytes
  and 5.83 µs. It now carries the basis type as a parameter, as `SplineQuadrature` already did.
- **The scalar `evaluate` of a tensor product allocated, and lost inference**, through two
  independent faults in one loop: a weight accumulated from inside the index `ntuple` is
  boxed, and `size(B, k)` reads the tuple of bases with a loop variable, which dispatches
  dynamically once per term whenever the axes differ in basis type. On a clamped ⊗ periodic ⊗
  Dirichlet basis that was 72 832 bytes and 34× the runtime; it now allocates its `D` per-axis
  buffers and nothing else — 256 bytes — for any mix of bases. `evaluate_all!` on the same
  bases was already allocation-free, which is why the suite could not see it. The deposition
  loop the `evaluate_all!` docstring recommends had the same `size(B, k)` in it, and now
  hoists the extents as the implementation does.
- **`RecombinedBSplineBasis` accepted `Periodic` at an end.** It imposes no local condition, so
  it passed through the constructor as `Free` does and produced an identity recombination of
  the clamped basis — a basis that is not periodic, under a name saying it is. Only the
  exported type was reachable this way; `boundary_conditions` already rejected `Periodic` in a
  pair, and it is now rejected in the constructor too. The same docstring advertised a
  two-argument form that does not exist, and now names the real one.
- **`?SplineDerivative` printed `derivative`'s signature.** The type's docstring was headed
  with the function's, so the type had no description of its own and the two overlapped.
- **`_apply_along!` claimed the operator was usable from several threads at once.** Its
  buffers are local to the call, which is not the same thing: a `CirculantMass` holds a
  scratch vector that `mass_solve!` writes, so two threads applying one operator would
  collide inside it. The comment now says what the buffers do buy — `O(D)` of them per call
  rather than one pair per fibre — and nothing more.
- **`breakpoints` had no documented aliasing contract**, while returning the mesh's own array
  for `GradedMesh`, `RandomMesh` and `GeneralMesh` and a fresh one for `UniformMesh`. Writing
  to the result therefore corrupted the mesh on three families out of four, silently: the
  domain and cell count still agree, so nothing rejects it, but `hash` and `==` then report a
  different mesh. Returning a copy is not the fix — sharing the array is what keeps `findcell`
  off the allocator — so the contract is now stated on both `breakpoints` methods, and the
  uniform case is documented as an implementation detail rather than a licence.
- **`evaluate(b, û, X)` over a vector of points assumed `X` was 1-based.** It indexed its
  output with the keys of `X`, so a vector with offset axes raised a `BoundsError` instead of
  being evaluated. It counts its own output now.
- **`Constraint`'s inner constructor left a type parameter unbound.** `NTuple{N, T}` admits
  `N = 0`, where there is nothing in `()` from which to infer `T`; the empty tuple is its own
  method now, and the general one requires a coefficient. Behaviour is unchanged — both
  spellings threw the intended `ArgumentError` already, by the order of the checks rather than
  by construction — but the signature was one Aqua's `unbound_args` check reports, and the
  suite now runs Aqua.
- **`contract` consumed the array it was given.** The quadrature weights are applied in
  place, and the copy that was meant to protect the caller was `convert(Array{R}, F)` —
  the *identity* when `F` already has the working element type, which is exactly what
  `quadrature_sample` returns. The caller's sample came back scaled by the weights, so a
  second `l2_projection` of the same array gave a different answer. It now copies.
- **The per-axis loops of a tensor product were type-unstable**, and it cost 248×.
  `prod(dims[1:(k-1)])` slices a tuple to a length inference cannot know, so the reshape
  built from it inferred as `Any` and every element access in `contract` and in every
  `KroneckerMass` solve became a dynamic dispatch. Multiplying the extents out in a plain
  loop fixes it: `_scale_along!` on a 160×160 array went from 1862 µs and 2.9 MB to 14.8 µs
  and no allocation, `contract` from 5.3 MB to 0.33 MB per call.
- **`KroneckerMass` multiplication copied every fibre.** `_apply_along!` now hands `f!` two
  distinct buffers, so `mul!` needs no defensive copy of its own.
- **`local_width` was recomputed on every `evaluate_all!` of a recombined basis**, making the
  particle path `O(ncells)` per point rather than `O(p²)`. It is a constant of the basis and
  is now stored with `firstcol`/`lastcol`.
- **Building a `RecombinedBSplineBasis` was `O(n²)`**: the cell-to-column table scanned every
  nonzero of `R` once per cell. Each nonzero is now visited once and the cells it reaches are
  updated, which at `n = 1024` is 0.01 ms against 1.24 ms.
- **`GradedMesh` and `RandomMesh` rebuilt their breakpoints on every call**, `RandomMesh`
  re-running its `Xoshiro` stream each time — 19 kB per `findcell`. Both now hold the vector,
  as `GeneralMesh` always did. `meshwidth` no longer materialises a `diff`.
- A stale paragraph in the `SplineQuadrature` docstring described `Φ` as stored densely "so
  the contractions run as one BLAS call". It is a `SparseMatrixCSC`, and the comment beside
  the assembly argues correctly for the opposite.
- The `mass_operator(::SplineQuadrature)` docstring still offered the choice as `CirculantMass`
  on a uniform mesh and `FactorizedMass` otherwise, which adding `BandedMass` made wrong, and
  `MassOperator`'s said "Both answer `\`, `ldiv!` and `Matrix`" after the same change took its
  list to three.
- `StaticArrays` was a dependency no source file used: the module's `using` was its only
  occurrence in `src/`. It is a test dependency now rather than a package one.
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

### An ill-scaled `Robin` condition degrades silently

`_recombination` anchors each end block on the last of the `m+1` functions the condition
reaches, because only that one contributes to `a_{m+1} = c_m D^m φ_{m+1}(a)`. That anchor is
nonzero whenever the leading coefficient `c_m` is — but *how* nonzero is the caller's, not the
construction's. `Robin(1.0, 1e-20)` puts a small coefficient on the highest derivative, the
anchor is then numerically zero, and the recombination coefficients `-a_t/a_{m+1}` overflow:
the mass matrix comes out finite with a condition number already `Inf`, `cholesky(…;
check = false)` reports `issuccess` on a matrix containing `Inf`, and the projection that
follows looks plausible.

A finiteness guard would catch only the most extreme case and is a symptom patch — the
condition number is already useless well before anything becomes `Inf`. The cause-level fix is
to **pivot**: anchor on whichever of the `m+1` candidates has the largest `|a_i|`, which bounds
every entry of `R` by 1 by construction. That is deferred rather than done because it changes
which basis functions the recombination produces, and with them `nodes` and every assembly
conjugated by `R` — a design change to accept or decline, not a follow-up commit.

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
