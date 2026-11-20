# Known issues

## Upstream

### K1 · Revise prints EMFILE errors in the test log

- **location:** `test/quality/jet.jl`
- **evidence:** JET 0.12 loads Revise, and its file watcher runs out of file handles.
  `grep -c 'UNHANDLED TASK ERROR'` on a `run-tests.jl full` log of the branch that adds
  `test/quality/jet.jl` counts 5 blocks, each an
  `IOError: FolderMonitor: too many open files (EMFILE)` stack trace. The same count on a
  `run-tests.jl full` log of `origin/main` gives 0. The test totals do not change.
- **kind:** upstream
- **found:** 2026-09-28

## This package

### K5 · `SplineQuadrature` refuses a `Float32` periodic basis on a narrow uniform mesh far from zero

- **location:** `src/mass.jl`, the circulant check that raises `_not_circulant`
- **evidence:** the rounded breakpoints of such a mesh are not equally spaced to the check's
  tolerance `sqrt(eps(T))`, so the assembly is not circulant to it, and the constructor throws.
  The cells of `UniformMesh(28, -46.660625f0 .. -46.447563f0)` vary in width by `5.0e-4` of
  `meshwidth`, against `sqrt(eps(Float32)) = 3.5e-4`; there `eps(Float32) |a| / h = 7.3e-4`.
  `SplineQuadrature(PeriodicBSplineBasis(UniformMesh(28, -46.660625f0 .. -46.447563f0), 1))` throws,
  on `a204d86` and on #39 alike: "the mass matrix assembled with nq = 2 points per cell is not
  usable for a degree-1 basis on 28 cells: the mass matrix is not circulant to a relative tolerance
  of 0.00034526698; a CirculantMass is only valid on a uniform mesh. Use nq ≥ 2 for an exact mass
  matrix, or at least nq ≥ 2". Its advice names the `nq` already in use. Over 1000 spans drawn by
  `random_span` of `test/helpers/meshes.jl` (`Xoshiro(20261031)`, `n ∈ 4:32`, `p ∈ 1:3`), 17
  `Float32` spans throw on #39 and no `Float64` span does.
- **kind:** defect
- **found:** 2026-10-31; older than #39

### K7 · `l2_projection!` into a `Float32` array fails on a `Float64` periodic quadrature

- **location:** `src/tensorproduct.jl`, `l2_projection!`, and the FFT solve of `CirculantMass`
- **evidence:** with `B = BSplineBasis(UniformMesh(8, 0.0 .. 1.0), 3, Periodic()) ⊗
  BSplineBasis(UniformMesh(8, -1.0 .. 2.0), 3)` and `q = TensorProductQuadrature(B)`,
  `l2_projection!(zeros(Float32, size(B)...), q, ones(quadrature_grid_size(q)...))` raises
  `MethodError: no method matching mul!(::Vector{ComplexF64}, ::FFTW.rFFTWPlan{Float64, -1,
  false, 1, Tuple{Int64}}, ::Vector{Float32}, ::Bool, …`, on `cb82e6f` and on the branch that
  adds caller-held workspaces alike: the `Float64` plan of the periodic factor does not take a
  `Float32` vector.
- **kind:** defect
- **found:** 2026-11-02; older than `cb82e6f`

### K9 · `evaluate!` on a `TensorProductBasis` takes buffers of one type only

- **location:** `src/tensorproduct.jl`, `evaluate!(bufs, B::TensorProductBasis, û, x, d)`
- **evidence:** the method takes `bufs` as an `NTuple{D, <:AbstractVector}`, so a tuple
  of a `Vector` and a `SubArray`, or of vectors of different element types, raises a
  `MethodError`. `evaluate_all!(bufs, B, x, d)`, which it calls, takes `bufs` as the same
  type, as on `cb82e6f`, and refuses the same tuples.
- **kind:** found late
- **found:** 2026-11-02

### K10 · `evaluate_all` on a domain of subnormal length returns `Inf`

- **location:** `src/bspline.jl:878`
- **evidence:** with `b = PeriodicBSplineBasis(UniformMesh(8, (0.0f0, 1.0f-39)), 3)`,
  `evaluate_all(b, 3.7f-40)` gives `Float32[Inf, Inf, Inf, Inf]`, while `evaluate(b, j, 3.7f-40)`
  for the same four functions gives `[1.07f-5, 0.187, 0.665, 0.147]`, which sum to one. The
  `Float64` domain `(0.0, 1e-309)` at `3.7e-310` does the same, on `72a3e17` and on the branch
  that removes K6 alike. The de Boor step at `src/bspline.jl:932`,
  `temp = values[r] / (right + left)`, divides by a knot span of a few subnormal cell widths,
  whose reciprocal already overflows, so `temp` and every value become `Inf`. Forming the
  ratios `right / s` and `left / s` where the cell of `x` is subnormal gives the values of
  `evaluate`, but in its cheapest form tried, a test of the cell width once per call, it makes
  `evaluate_all!` about 5 % slower at degree 3 and for points outside the domain at degree 1
  (`scripts/periodic_reduction.jl`, two rounds against `72a3e17`), so the fix stays open.
- **kind:** defect
- **found:** 2026-11-20; older than `72a3e17`

### K11 · A `Float16` `UniformMesh` refuses ordinary domains, because its closed form overflows

- **location:** `src/mesh.jl`, `_uniform_breakpoint`, which `breakpoints(::UniformMesh)` and the
  `UniformMesh` constructor share
- **evidence:** `_uniform_breakpoint` computes `a + (b - a) * (i - 1) / n`, and in `Float16` the
  product `(b - a) * (i - 1)` exceeds `floatmax(Float16) = 65504` on ordinary meshes.
  `UniformMesh{Float16}(100, 0 .. 1000)` throws "ArgumentError: the breakpoints of 100 cells on
  [0.0, 1.0e3] overflow at this precision"; on `72a3e17` its breakpoints were `Inf`.
  `UniformMesh{Float16}(10, 0 .. 1000)` builds, and `GradedMesh{Float16}` and
  `RandomMesh{Float16}` on `0 .. 1000` with 100 cells give finite breakpoints. The form
  `(b - a) * ((i - 1) / n)` avoids the overflow, but it changes the rounding of every `Float64`
  uniform mesh, and with it `==`, `hash` and the circulant path.
- **kind:** defect
- **found:** 2026-11-20; older than `72a3e17`
