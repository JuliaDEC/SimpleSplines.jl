```@meta
CurrentModule = SimpleSplines
```

# [Polar Splines](@id usage-polar)

A [`PolarSplineBasis`](@ref) is a spline space on a parameter rectangle whose left radial edge
is a **pole** — one point of the physical domain, reached from every angle. It is a subspace of
a two-factor [`TensorProductBasis`](@ref), and it is ``C^1`` at the pole by construction. For
why a plain tensor product is not even ``C^0`` there, and for the mathematics of the pole
triangle, see [Polar Splines](@ref theory-polar).

```@example upolar
using SimpleSplines
using LinearAlgebra
using SparseArrays
using Random
Random.seed!(1234)
```

## Construction

```julia
PolarSplineBasis(radial, angular)      PolarSplineBasis(radial ⊗ angular)
```

The radial axis is clamped and the angular axis periodic. The pole triangle replaces the first
two radial rows — ``2 N_\theta`` parent functions — by three, so the space has
``3 + (N_s - 2) N_\theta`` functions.

```@example upolar
radial = BSplineBasis(UniformMesh(8, 0 .. 1), 3)
angular = PeriodicBSplineBasis(UniformMesh(16, 0 .. 2π), 3)

B = PolarSplineBasis(radial, angular)
```

```@example upolar
nbasis(B), nbasis(parent(B)), 3 + (nbasis(radial) - 2) * nbasis(angular)
```

### The four guards

Each rejects a space the construction cannot build, rather than building a wrong one.

```@example upolar
try
    PolarSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 1), angular)
catch err
    println(first(split(err.msg, ':')))
end
```

```@example upolar
try
    PolarSplineBasis(radial, BSplineBasis(UniformMesh(16, 0 .. 2π), 3))
catch err
    println(first(split(err.msg, ':')))
end
```

The other two are counts: a radial basis of fewer than three functions leaves nothing behind
the triangle, and an angular basis of fewer than three cannot hold the constants, ``C`` and
``S`` independently. The fourth guard is on the rim, below.

## Accessors

Most forward to the parent, so they report the **parent's** per-axis numbers. Three do not.

```@example upolar
nbasis(B), ndims(B), degree(B), ncells(B), pole(B)
```

| call | returns |
|:--|:--|
| [`nbasis`](@ref)`(B)`, `length(B)`, `eachindex(B)` | ``3 + (N_s-2)N_\theta``, the same, and `1:` that |
| `ndims(B)`, `eltype(B)` | always `2`, and the element type |
| [`degree`](@ref), [`order`](@ref), [`ncells`](@ref), [`mesh`](@ref), [`bases`](@ref) | per-axis tuples, from the parent |
| [`meshwidth`](@ref)`(B)` | the maximum over the axes, a scalar |
| [`domain`](@ref)`(B)` | the parent's `ProductDomain`, the **parameter** rectangle |
| [`pole`](@ref)`(B)` | the radial coordinate of the pole |
| `parent(B)` | the [`TensorProductBasis`](@ref) the space sits inside |
| [`recombination_matrix`](@ref)`(B)` | the sparse `R` with ``\Psi_k = \sum_I R_{Ik} \Phi_I`` |
| [`pole_triangle`](@ref)`(B)` | the ``3 \times 2`` vertex coordinates in the chart |

```@example upolar
size(recombination_matrix(B)), nnz(recombination_matrix(B))
```

The three pole columns carry ``2N_\theta`` nonzeros each; every other column carries one.

```@example upolar
R = recombination_matrix(B)
length(nzrange(R, 1)), length(nzrange(R, 4)), 2 * nbasis(angular)
```

**`nbasis(B)` is the only count.** There is no `size(B)`, because the index set is not a
product — see *What is not provided*.

## Coefficients are a vector

This is the first thing to get right, and it is where a habit from
[Tensor Products](@ref usage-tensorproduct) misleads. A tensor-product spline carries a
coefficient **array** of size `size(B)`. A polar spline carries a **vector** of length
`nbasis(B)`, because the index set is not a product: the three pole functions sit at the front,
and the outer rows follow, the radial index fastest.

```@example upolar
û = randn(nbasis(B))
û[1:3]            # the pole triangle
```

[`parent_coefficients`](@ref) is the bridge back to the parent's array shape, and it is where
every evaluation goes.

```@example upolar
ĉ = parent_coefficients(B, û)
size(ĉ) == size(parent(B)), ĉ ≈ reshape(R * û, size(parent(B)))
```

Use it to reach the parent's own machinery — plotting on the parameter grid, a tensor-product
routine — without rewriting the index arithmetic.

## Evaluating

```julia
evaluate(B, k, x, d = (0, 0))        # the k-th basis function
evaluate(B, û, x, d = (0, 0))        # the spline
evaluate(B, û, X, d = (0, 0))        # a vector of points, in one pass
B[x, k]        B(x, k)
```

`x` is the coordinate pair ``(s, \theta)`` and `d` is a **per-axis multi-index**:
`(1, 0)` is ``\partial_s``, `(0, 1)` is ``\partial_\theta``. There is no scalar derivative
order on a two-dimensional space.

```@example upolar
evaluate(B, û, (0.3, 1.1)), evaluate(B, û, (0.3, 1.1), (1, 0))
```

The space is a partition of unity, at the pole included:

```@example upolar
maximum(abs(sum(evaluate(B, k, (s, θ)) for k in eachindex(B)) - 1)
        for s in range(0, 1; length = 9), θ in range(0, 2π; length = 9))
```

Each pole function takes the value ``1/3`` at the pole, so a spline is single-valued there:

```@example upolar
vals = [evaluate(B, û, (0.0, θ)) for θ in range(0, 2π; length = 33)]
maximum(vals) - minimum(vals)
```

**Pass the whole vector of points when you have one.** `evaluate(B, û, x)` forms `R * û` on
every call; the vector method forms it once.

```@example upolar
pts = [(0.1 * i, 0.2 * j) for i in 1:9, j in 1:9]
vs = evaluate(B, û, vec(pts))
maximum(abs, vs - [evaluate(B, û, p) for p in vec(pts)])
```

### `evaluate_all` returns indices, not an offset

```@example upolar
idx, vals = evaluate_all(B, (0.02, 1.1))
idx[1:3], length(idx), sum(vals)
```

In the first two radial cells the three pole functions are nonzero alongside the outer rows,
and they sit at the **front** of the index set rather than beside them. The block is therefore
not contiguous, and an offset cannot describe it. Away from the pole it is an ordinary
``(p_s+1)(p_\theta+1)`` block, but the indices are still given rather than an offset, so that
one code path serves both:

```@example upolar
idx2, vals2 = evaluate_all(B, (0.9, 1.1))
length(idx2), sum(vals2), 3 ∈ idx2
```

## [The rim](@id usage-polar-rim)

The pole is not a boundary condition — it couples the two axes. The **rim**, the outer radial
end, is an ordinary one, and it is imposed on the radial axis before the polar basis is built.

```@example upolar
rim = RecombinedBSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3), Free(), Dirichlet())
BD = PolarSplineBasis(rim, angular)
nbasis(BD), nbasis(B)
```

`Free()` at the pole end is required, and it is the fourth guard:

```@example upolar
try
    PolarSplineBasis(
        RecombinedBSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3), Dirichlet(), Free()),
        angular)
catch err
    println(first(split(err.msg, ':')))
end
```

**A Dirichlet rim removes the constant, so the partition of unity is gone** — in the last
radial cell, and only there. That is the point of such a space, not a defect in it; see
[The rim](@ref theory-polar-rim).

```@example upolar
[(s, round(abs(sum(evaluate(BD, k, (s, 0.7)) for k in eachindex(BD)) - 1); digits = 12))
 for s in (0.1, 0.5, 0.95)]
```

## Assembly

```julia
PolarSplineQuadrature(B; nq = map(quadrature_order, degree(B)), dmax = 3)
```

```@example upolar
q = PolarSplineQuadrature(B)
nbasis(q), quadrature_grid_size(q), length(quadrature_weights(q))
```

| call | returns |
|:--|:--|
| [`basis`](@ref)`(q)`, `parent(q)` | the [`PolarSplineBasis`](@ref), and the parent [`TensorProductQuadrature`](@ref) |
| [`quadrature_nodes`](@ref)`(q)` | per-axis **tuple** of vectors, `(s, θ)` |
| [`quadrature_weights`](@ref)`(q)` | the **flattened** weight vector, `kron(w_θ, w_s)` |
| [`quadrature_grid_size`](@ref)`(q)` | `(n_d * nq_d)` per axis |
| [`basis_values`](@ref)`(q, d)` | the sparse table ``\Phi_d[k,r]``, memoised |
| [`mixed_matrix`](@ref), [`stiffness_matrix`](@ref), [`basis_integrals`](@ref) | assemblies against the **parameter** measure |
| [`weighted_matrix`](@ref)`(q, f, a, b)` | the same with a coefficient, not memoised |
| [`mass_matrix`](@ref)`(q)`, [`mass_operator`](@ref)`(q)` | the assembled matrix, and its [`FactorizedMass`](@ref) |

Note the two shapes that differ from the parent. `quadrature_nodes` is a per-axis tuple, as it
is there; `quadrature_weights` is **one number per grid point**, because the polar tabulation
has one column per grid point rather than a per-axis factor.

```@example upolar
map(length, quadrature_nodes(q)), length(quadrature_weights(q))
```

### The tabulation

```@example upolar
Φ = basis_values(q, (0, 0))
size(Φ), size(Φ) == (nbasis(B), prod(quadrature_grid_size(q)))
```

`d` is a per-axis multi-index here too. The scalar form is accepted only for `d = 0`, where it
is unambiguous:

```@example upolar
try
    basis_values(q, 1)
catch err
    println(first(split(err.msg, ':')))
end
```

The table is formed on first use and **memoised**, which is what makes
[`weighted_matrix`](@ref) affordable inside a Newton loop: the coefficient changes at every
iteration and the tabulation does not.

### Matrices and projection

Every assembly is against the **parameter** measure ``ds \, d\theta``. A polar space is used
for a *mapped* domain, and the Jacobian of that map belongs in the weight — it is the caller's
to supply, because the map is.

```@example upolar
M = mass_matrix(q)
K = stiffness_matrix(q)
size(M), issymmetric(M), size(K)
```

```@example upolar
A = weighted_matrix(q, x -> x[1], (1, 0), (1, 0))     # ∫ s ∂ₛΨₖ ∂ₛΨₗ ds dθ
size(A), maximum(abs, A - A')
```

On a free space the basis integrals are ``\mathbb{M}\mathbf{1}``, because the space is a
partition of unity, and they sum to the area of the parameter rectangle:

```@example upolar
maximum(abs, basis_integrals(q) - M * ones(nbasis(B))), sum(basis_integrals(q)) - 2π
```

[`l2_projection`](@ref) takes a function of the pair ``(s, \theta)``, or a vector already
sampled on the flattened grid. The constants are in the free space exactly:

```@example upolar
v̂ = l2_projection(q, x -> 1.0)
maximum(abs(evaluate(B, v̂, (s, θ)) - 1)
        for s in range(0, 1; length = 9), θ in range(0, 2π; length = 9))
```

```@example upolar
f(x) = exp(-4 * x[1]^2) * (1 + 0.3 * cos(x[2]))
ŵ = l2_projection(q, f)
ŵ2 = similar(ŵ)
l2_projection!(ŵ2, q, f)
maximum(abs, ŵ2 - ŵ)
```

A sample already on the grid gives the same answer. The flattening runs the **radial axis
fastest**, which is the convention of `vec` of a [`quadrature_sample`](@ref) array, of
[`parent_coefficients`](@ref), and of the index set itself.

```@example upolar
sθ = quadrature_nodes(q)
F = vec([f(pt) for pt in Iterators.product(sθ...)])
maximum(abs, l2_projection(q, F) - ŵ)
```

For a worked mapped problem — the disc's operator as two [`weighted_matrix`](@ref) calls with
weights ``s`` and ``1/s``, and a rim condition imposed by hand — see the gallery's *Poisson on
a disc*.

## What is not provided

**There is no [`KroneckerMass`](@ref).** A pole function is a sum over the whole angular axis,
so the mass matrix does not factor per axis. It is assembled and factorised instead — a sparse
Cholesky with only a ``3 \times 3`` dense block.

```@example upolar
nameof(typeof(mass_operator(q)))
```

That is the cost of the pole, and it is paid on the solve rather than on the assembly. At
``64 \times 128`` cubic cells one polar mass solve measures 0.74 ms against 0.136 ms for the
[`KroneckerMass`](@ref) solve of the tensor-product space at the same mesh — a factor of 5.4,
and still under a millisecond. `scripts/polar_mass_cost.jl` measures it.

**Several one-dimensional and tensor-product accessors have no polar method**, and each
absence is a statement rather than a gap to be filled later:

| call | why not |
|:--|:--|
| `size(B)`, `nodes(B)`, `LinearIndices(B)` | the index set is not a product, so there is no per-axis shape and no grid of nodes |
| [`boundary`](@ref)`(B)`, [`local_width`](@ref)`(B)` | the pole is not a boundary condition, and a pole function is not local in ``\theta`` |
| [`polynomial_reproduction`](@ref)`(B)` | not defined — ask the **radial axis** instead; see below |
| [`Spline`](@ref)`(B, û)` | a `Spline` holds an array of `size(basis)`; use `evaluate(B, û, x)` |

### Which regime a space is in

[`polynomial_reproduction`](@ref) has no polar method, and the per-axis numbers have to be
read with care. **Ask the radial axis**: it carries the rim condition, and it is what
separates a free space from a constrained one.

```@example upolar
polynomial_reproduction(bases(B)[1]), polynomial_reproduction(bases(BD)[1])
```

A free radial axis gives `p`; a homogeneous-Dirichlet rim gives `-1`, meaning not even the
constants survive — which is exactly the partition of unity the rim removed. The **angular**
axis is always `0`, because ``\theta`` is not periodic, and it says nothing about the polar
space at all:

```@example upolar
polynomial_reproduction(bases(B)[2])
```

Neither number is a statement about the disk. What the free space reproduces *there* is ``1``,
``\tilde{x}`` and ``\tilde{y}`` — the chart's constants and linears, to round-off — and that
is a property of the pole triangle rather than of either axis. `scripts/polar_continuity.jl`
and `scripts/polar_approximation_order.jl` measure it.

**There is no `weighted_matrix` with a non-separable coefficient on the parent**, for the same
reason there is none for a [`TensorProductQuadrature`](@ref) — but here it does not matter: the
polar tabulation is already one flat sparse table over the whole grid, so
[`weighted_matrix`](@ref) takes any coefficient at all.
