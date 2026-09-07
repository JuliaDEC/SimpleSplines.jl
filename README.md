# SimpleSplines

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDEC.github.io/SimpleSplines.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDEC.github.io/SimpleSplines.jl/dev/)
[![Build Status](https://github.com/JuliaDEC/SimpleSplines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDEC/SimpleSplines.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://github.com/JuliaDEC/SimpleSplines.jl/actions/workflows/Documenter.yml/badge.svg?branch=main)](https://github.com/JuliaDEC/SimpleSplines.jl/actions/workflows/Documenter.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDEC/SimpleSplines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDEC/SimpleSplines.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSplines.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSplines.html)

B-spline finite elements on an interval, built from the Cox-de Boor recursion, with the
quadrature and assembly a Galerkin discretisation needs.

The package provides the clamped, periodic and recombined B-spline bases of arbitrary degree
on uniform, graded, random or arbitrary meshes; homogeneous Dirichlet, Neumann, Robin, natural
and general local boundary conditions per end; tensor products in any number of dimensions,
with degree, mesh, domain and boundary condition per axis; and an assembly table from which the
mass, stiffness, derivative and variable-coefficient matrices follow as single weighted
contractions. Mass solves go through a representation chosen by the basis — an FFT for a
periodic uniform basis, a banded Cholesky for a bounded one, and a factored Kronecker product
in several dimensions.

The [manual](https://JuliaDEC.github.io/SimpleSplines.jl/dev/) has a tutorial, the spline
theory the package rests on, a gallery of solved problems and the full API.

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
