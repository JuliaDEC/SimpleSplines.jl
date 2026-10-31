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
