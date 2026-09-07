```@meta
CurrentModule = SimpleSplines
```

# Library

The complete API of *SimpleSplines.jl*. See the [Tutorial](@ref) for how the pieces fit
together, the Theory pages for the mathematics, and the Usage pages for the details of each
object.

Seven accessors are **shared with the rest of the ecosystem** rather than defined here.
[`basis`](@ref), [`degree`](@ref), [`nodes`](@ref), `nnodes` and [`order`](@ref) belong to
`GeometricBase`, `grid` to `ContinuumArrays`, and [`nbasis`](@ref) to
[CompactBasisFunctions.jl](https://github.com/JuliaGNI/CompactBasisFunctions.jl), so that one
generic function per accessor is extended across the packages instead of one being defined in
each. Four further names are **re-exported unchanged** so that `using SimpleSplines` is enough
to write a domain: `..`, `leftendpoint` and `rightendpoint` come from `IntervalSets` and
`DomainSets`, and `Basis` from `CompactBasisFunctions`; their documentation lives in those
packages.

```@index
```

## Meshes

```@docs
Mesh
UniformMesh
GradedMesh
RandomMesh
GeneralMesh
breakpoints
ncells
domain
domainlength
meshwidth
findcell
```

## Boundary conditions

```@docs
BoundaryCondition
Free
Periodic
Dirichlet
Neumann
Natural
Robin
Constraint
boundary_conditions
constraint_coefficients
constraint_order
nconstraints
```

## Bases

```@docs
AbstractBSplineBasis
BSplineBasis
PeriodicBSplineBasis
RecombinedBSplineBasis
recombination_matrix
```

### Accessors

```@docs
nbasis
degree
order
mesh
knotvector
nodes
nnodes
grid
boundary
basis_index
local_width
local_indices
polynomial_reproduction
```

### Evaluation

```@docs
evaluate
evaluate_all
evaluate_all!
BSplineDerivative
PeriodicBSplineDerivative
```

## Assembly

```@docs
SplineQuadrature
quadrature_order
quadrature_nodes
quadrature_weights
basis_values
basis
mass_matrix
mass_factorization
stiffness_matrix
derivative_matrix
mixed_matrix
weighted_matrix
basis_integrals
l2_projection
l2_projection!
```

## Mass operators

```@docs
MassOperator
CirculantMass
BandedMass
FactorizedMass
KroneckerMass
mass_operator
mass_factors
mass_solve!
```

## Tensor products

```@docs
TensorProductBasis
⊗
bases
TensorProductQuadrature
quadratures
quadrature_grid_size
quadrature_sample
contract
```

## Splines

```@docs
Spline
SplineDerivative
coefficients
derivative
```
