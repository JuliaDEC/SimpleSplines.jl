```@meta
CurrentModule = SimpleSplines
```

# SimpleSplines.jl

B-spline finite elements on an interval, built from the Cox-de Boor recursion, with the
quadrature and assembly a Galerkin discretisation needs.

The package is deliberately small: one family of bases, one assembly table, and the mass,
stiffness, derivative and variable-coefficient matrices as single weighted contractions of
that table. What it is *not* is a curve- and surface-modelling library — there are no NURBS,
no knot insertion, no degree elevation and no least-squares fitting of data. What it is for
is discretising a differential equation and then solving with the result.

## What it provides

| | |
|:--|:--|
| **Bases** | clamped ([`BSplineBasis`](@ref)), periodic ([`PeriodicBSplineBasis`](@ref)), and recombined ([`RecombinedBSplineBasis`](@ref)) — of any degree ``p \ge 0`` |
| **Meshes** | [`UniformMesh`](@ref), [`GradedMesh`](@ref), [`RandomMesh`](@ref), [`GeneralMesh`](@ref) |
| **Boundary conditions** | [`Free`](@ref), [`Periodic`](@ref), [`Dirichlet`](@ref), [`Neumann`](@ref), [`Natural`](@ref), [`Robin`](@ref), and the general [`Constraint`](@ref) — per end |
| **Assembly** | [`SplineQuadrature`](@ref) and, from it, [`mass_matrix`](@ref), [`stiffness_matrix`](@ref), [`derivative_matrix`](@ref), [`mixed_matrix`](@ref), [`weighted_matrix`](@ref), [`basis_integrals`](@ref) |
| **Mass solves** | [`CirculantMass`](@ref) (FFT), [`BandedMass`](@ref) (banded Cholesky), [`FactorizedMass`](@ref) (sparse Cholesky), [`KroneckerMass`](@ref) (factored, ``D``-dimensional) |
| **Tensor products** | [`TensorProductBasis`](@ref) and [`TensorProductQuadrature`](@ref) in any number of dimensions, with degree, mesh, domain and boundary condition **per axis** |
| **Functions** | [`Spline`](@ref), [`SplineDerivative`](@ref), [`l2_projection`](@ref) |
| **Particle path** | [`findcell`](@ref) and [`evaluate_all!`](@ref), allocation-free, for depositing onto the basis |

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaDEC/SimpleSplines.jl")
```

## In thirty seconds

A basis is a [`Mesh`](@ref), a degree, and a [`BoundaryCondition`](@ref):

```@example intro
using SimpleSplines

b = BSplineBasis(UniformMesh(16, 0 .. 1), 3, Dirichlet())
nbasis(b), degree(b), order(b), polynomial_reproduction(b)
```

Assembly goes through a [`SplineQuadrature`](@ref), which tabulates the basis and its
derivatives at the global Gauß-Legendre points and hands back every matrix built on them:

```@example intro
q = SplineQuadrature(b)
M = mass_matrix(q)
K = stiffness_matrix(q)
size(M), size(K)
```

Fitting a function to the space is an ``L^2`` projection, and the result is callable:

```@example intro
u = Spline(b, l2_projection(q, x -> sin(π * x)))
u(0.5), u(0.5, 1), u(0.0)        # value, first derivative, and the imposed u(0) = 0
```

Solving ``-u'' = f`` with ``u(0) = u(1) = 0`` is then two lines, because the recombined basis
has already taken care of the boundary condition:

```@example intro
f(x) = π^2 * sin(π * x)
rhs = basis_values(q, 0) * (quadrature_weights(q) .* f.(quadrature_nodes(q)))
û = Matrix(K) \ rhs
maximum(abs(evaluate(b, û, x) - sin(π * x)) for x in range(0, 1; length = 101))
```

## How the pieces fit

```
Mesh + BoundaryCondition
        │
        ▼
AbstractBSplineBasis  ───▶  Spline        ───▶  s(x),  derivative(s)
        │
        ▼
SplineQuadrature      ───▶  mass_matrix, stiffness_matrix, derivative_matrix,
        │                   mixed_matrix, weighted_matrix, basis_integrals,
        │                   l2_projection
        ▼
MassOperator          ───▶  mass_solve!,  op \ x
```

The mesh is geometry alone and carries no boundary condition; the boundary condition decides
which of the three bases the one constructor [`BSplineBasis`](@ref)`(mesh, p, bc)` returns;
the basis decides which representation the mass matrix takes. A tensor product is the same
picture with a tuple in each box.

## Where to go next

  - **[Tutorial](@ref)** — one worked example from a bare `using` to a solved problem.
  - **Theory** — [B-Splines](@ref) for the recursion, the derivative formula and the dimension
    counts; [Boundary Conditions](@ref theory-boundary) for recombination; [Tensor
    Products](@ref theory-tensorproduct) for the Kronecker structure.
  - **Usage** — [Meshes](@ref usage-meshes), [Bases](@ref usage-bases), [Boundary
    Conditions](@ref usage-boundary), [Assembly](@ref usage-assembly), [Tensor Products](@ref
    usage-tensorproduct) and [Splines](@ref usage-splines) give the constructors, the
    accessors and the traps.
  - **[Gallery](@ref)** — seven solved problems with their measured errors.
  - **[Library](@ref)** — every exported name.

## Related packages

[CompactBasisFunctions.jl](https://github.com/JuliaGNI/CompactBasisFunctions.jl) provides the
`Basis` hierarchy these bases join, along with the Lagrange, Chebyshev, Legendre and Bernstein
families; [QuadratureRules.jl](https://github.com/JuliaGNI/QuadratureRules.jl) provides the
Gauß-Legendre nodes and weights; and `GeometricBase` owns the shared accessors
[`basis`](@ref), [`degree`](@ref), [`nodes`](@ref), `nnodes` and [`order`](@ref), so that one
generic function per accessor is extended across the ecosystem rather than one defined per
package.
