# Known issues

## Upstream

### K1 · Revise prints EMFILE errors in the test log

- **location:** `test/quality/jet.jl`
- **evidence:** JET 0.12 loads Revise, and its file watcher runs out of file handles.
  `grep -c 'UNHANDLED TASK ERROR'` on a `run-tests.jl full` log of the branch that adds
  `test/quality/jet.jl` counts 5 blocks, each an
  `IOError: FolderMonitor: too many open files (EMFILE)` stack trace. The same count on a log of
  every other test file run on the base's `src/` gives 0. The test totals do not change.
- **kind:** upstream
- **found:** 2026-09-28

## Test file

### K2 · A comment in `jet.jl` calls a periodic basis bounded

- **location:** `test/quality/jet.jl:13`
- **evidence:** the comment reads "a free, a Dirichlet and a periodic bounded basis". A periodic
  basis has no boundary. The fix: "a free and a Dirichlet bounded basis and a periodic basis".
- **kind:** docs
- **found:** 2026-09-28

### K3 · Two `jet.jl` checks repeat one analysis

- **location:** `test/quality/jet.jl`, the `mass_solve!` lines
- **evidence:** the deflated and the undeflated circulant operators have one type
  (`typeof(opd) == typeof(opc)` is `true` on 1.10 and 1.13), so the second line adds no check,
  and the comment "deflated and not" names a difference that JET cannot see. The banded loop
  builds 48 quadratures for one unique type; one line does the same check.
- **kind:** dead code
- **found:** 2026-09-28

### K4 · The base evidence of K1 is not a run of `origin/main`

- **location:** `KNOWN_ISSUES.md`, K1
- **evidence:** K1 gives the base count from the other test files on the base `src/`, not from a
  full run of `origin/main`. A `run-tests.jl full` log of `origin/main` also gives 0
  (`grep -c 'UNHANDLED TASK ERROR'`), so the fact holds; the wording does not.
- **kind:** docs
- **found:** 2026-09-28
