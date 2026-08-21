# SimpleSplines

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDEC.github.io/SimpleSplines.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDEC.github.io/SimpleSplines.jl/dev/)
[![Build Status](https://github.com/JuliaDEC/SimpleSplines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDEC/SimpleSplines.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDEC/SimpleSplines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDEC/SimpleSplines.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSplines.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSplines.html)

Periodic B-spline finite elements on an interval, built from the Cox-de Boor recursion,
with the quadrature and assembly a Galerkin discretisation needs. The package provides the
periodic B-spline basis of arbitrary degree on uniform, graded and random meshes, together
with an assembly table from which the mass, stiffness, derivative and variable-coefficient
matrices follow as single weighted contractions.

## Development

To run the test suite before every push, enable the repository's git hooks:

```sh
git config core.hooksPath .githooks
```
