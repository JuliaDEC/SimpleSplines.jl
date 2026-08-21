```@meta
CurrentModule = SimpleSplines
```

# SimpleSplines.jl

Periodic B-spline finite elements on an interval, built from the Cox-de Boor recursion, with
the quadrature and assembly a Galerkin discretisation needs.

The package is deliberately small. It provides one basis — the periodic B-spline basis of
arbitrary degree on a uniform or non-uniform mesh — and one assembly table built on it, from
which the mass, stiffness, derivative and variable-coefficient matrices all follow as single
weighted contractions.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaDEC/SimpleSplines.jl")
```

## Basic usage

A basis is a [`Mesh`](@ref) and a degree:

```@example intro
using SimpleSplines

b = PeriodicBSplineBasis(UniformMesh(16, 2π), 3)
nbasis(b), degree(b), order(b)
```

Note that the number of degrees of freedom is the number of *cells*, `N = n`, not `n + p`:
on a torus the `p` extra functions of the bounded case are the ones the clamping introduces
at the two ends, and the periodic wrap identifies them in pairs.

Basis functions are indexed as `b[x, j]`, and derivatives of arbitrary order come from
[`evaluate`](@ref):

```@example intro
b[0.7, 3], evaluate(b, 3, 0.7, 1), evaluate(b, 3, 0.7, 2)
```

Assembly goes through a [`SplineQuadrature`](@ref), which tabulates the basis and its
derivatives at the global Gauß-Legendre points:

```@example intro
q = SplineQuadrature(b)
M = mass_matrix(q)
S = derivative_matrix(q)
maximum(abs, S + S')      # antisymmetric on a periodic mesh
```

Projecting a function onto the spline space:

```@example intro
û = l2_projection(q, sin)
abs(evaluate(b, û, 1.0) - sin(1.0))
```

## Meshes

Three families are provided, and which one to use is a question about what is being tested
rather than about the discretisation:

| mesh | refining `n` gives | use it for |
|:--|:--|:--|
| [`UniformMesh`](@ref) | a uniform refinement | ordinary computation; but its assemblies are *circulant*, so it confirms some identities for the wrong reason |
| [`GradedMesh`](@ref) | a genuine mesh family | convergence rates |
| [`RandomMesh`](@ref) | an unrelated mesh | properties that must hold on *any* mesh |

## The mass matrix

Solves against the mass matrix go through a [`MassOperator`](@ref), and which representation
is built is decided by the mesh:

  - on a [`UniformMesh`](@ref) the basis functions are translates of a single cardinal
    spline, so ``\mathbb{M}`` is **circulant** and diagonalised by the discrete Fourier
    transform. A solve is two planned transforms and a pointwise division, and allocates
    nothing beyond its result.
  - on a [`GradedMesh`](@ref) or [`RandomMesh`](@ref) it is banded modulo ``N`` but *not*
    circulant — the basis functions are no longer translates of one another — so there is
    nothing for a transform to diagonalise and a sparse Cholesky factorisation is used.

```@example intro
mass_operator(q)
```

The construction *verifies* circulance rather than assuming it: a matrix that is banded but
not circulant would still produce plausible numbers through the transform, and the failure
would surface much later as a wrong conservation law.

## Storage

The basis tabulation is **sparse**. A basis function is supported on `p+1` cells, so only
`(p+1)·nq` of the `n·nq` entries in its row are structurally nonzero, and storing ``\Phi``
densely would make every contraction ``\Phi \, \mathrm{diag}(fw) \, \Phi^T`` cost
``O(N^2 n n_q)`` instead of ``O(N p^2 n_q)``.

The constant assemblies — [`mass_matrix`](@ref), [`stiffness_matrix`](@ref),
[`derivative_matrix`](@ref) and any [`mixed_matrix`](@ref) — are memoised on first use, since
they do not depend on the field but are asked for inside every Newton iteration of a
downstream time integrator.

## Quadrature

The default rule, [`quadrature_order`](@ref), integrates degree ``3p-1`` exactly — what the
consistency of a Galerkin discretisation of a quadratic nonlinearity needs. Two lower
thresholds are worth knowing:

  - the mass matrix needs degree ``2p``, i.e. `nq ≥ p+1`;
  - the antisymmetry of [`derivative_matrix`](@ref) needs degree ``2p-1``, i.e. `nq ≥ p`,
    and fails outright below it on a non-uniform mesh.

## Library

```@index
```

```@autodocs
Modules = [SimpleSplines]
```
