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

### K2 · `evaluate` on a degree-0 periodic basis counts two functions just below `b`

- **location:** `src/bspline.jl:616`
- **evidence:** `evaluate` sums the images `x̃ - L`, `x̃` and `x̃ + L`, and `x̃ - L` can round
  onto `a`, so the first basis function is counted beside the last one.
  `B = PeriodicBSplineBasis(UniformMesh(14, -2.210389847221002 .. -1.67460502203573), 0)`
  gives `sum(B[prevfloat(b), :]) == 2.0`, nonzero at `j = 1` and `j = 14`, while
  `evaluate_all(B, prevfloat(b))` gives `(14, [1.0])`. Over 690 random degree-0 spans, 40
  `Float64` and 19 `Float32` bases fail this way at `prevfloat(b)`. Degree 1 and higher are
  continuous across the seam, so the same rounding there moves a value by rounding only. Setting
  the knot `y₁ + L` to `b` does not fix it: the shifted argument `a + L` then lands below that
  knot, and 35 of the `Float32` bases count two functions at `a` instead.
- **kind:** defect
- **found:** 2026-10-31

### K3 · A domain whose length overflows gives infinite breakpoints, which `nodes` clamps out of sight

- **location:** `src/mesh.jl`, `breakpoints(::UniformMesh)` and `_graded_breakpoints`
- **evidence:** `b - a` overflows, so every interior breakpoint is `Inf`, and no constructor
  refuses the domain. `breakpoints(UniformMesh(4, -1e308 .. 1e308))` and
  `breakpoints(GradedMesh(4, -1e308 .. 1e308))` both give `[-1.0e308, Inf, Inf, Inf, 1.0e308]`.
  `nodes(BSplineBasis(UniformMesh(4, -1e308 .. 1e308), 2))` gives
  `[-1.0e308, 1.0e308, 1.0e308, 1.0e308, 1.0e308, 1.0e308]`: the clamp of `nodes` to `[a, b]`
  turns the overflowed Greville averages into finite, wrong nodes instead of an error. Computing the
  averages in a difference form does not help, because the breakpoints are already infinite.
- **kind:** defect
- **found:** #39 (its review and CodeRabbit's merge-risk note); the clamp stays in #39
  (user, 2026-10-31)

### K4 · A span a few ulps wide gives repeated breakpoints, so some cells have zero width

- **location:** `src/mesh.jl`, `breakpoints(::UniformMesh)`
- **evidence:** no constructor refuses a span narrower than its cell count in ulps.
  `breakpoints(UniformMesh(8, 1.0 .. nextfloat(1.0, 3)))` gives
  `[1.0, 1.0, 1.0000000000000002, 1.0000000000000002, 1.0000000000000004, 1.0000000000000004,
  1.0000000000000004, 1.0000000000000007, 1.0000000000000007]`, so `any(iszero, diff(...))` is
  `true`, and a basis on that mesh has cells of zero width.
- **kind:** defect
- **found:** #39 (its review); older than #39

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
