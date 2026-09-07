```@meta
CurrentModule = SimpleSplines
```

# [Tensor Products](@id usage-tensorproduct)

A [`TensorProductBasis`](@ref) is ``D`` one-dimensional bases, one per axis, in any number of
dimensions. For the mathematics see [Tensor Products](@ref theory-tensorproduct).

```@example utp
using SimpleSplines
using LinearAlgebra
using Random
Random.seed!(1234)
```

## Construction

```julia
TensorProductBasis(b₁, b₂, …)      TensorProductBasis((b₁, b₂, …))
b₁ ⊗ b₂ ⊗ b₃
```

`⊗` associates **flat**: three factors give one three-factor basis, not a nest of two.

```@example utp
bx = BSplineBasis(UniformMesh(10, 0 .. 2π), 3, Periodic())
bv = BSplineBasis(UniformMesh(8, -10 .. 10), 4)
bz = BSplineBasis(UniformMesh(6, 0 .. 1), 2, Dirichlet())

B = bx ⊗ bv ⊗ bz
ndims(B), size(B), nbasis(B)
```

```@example utp
bases(B) === (bx, bv, bz), (bx ⊗ bv) ⊗ bz == bx ⊗ (bv ⊗ bz)
```

A one-factor product is allowed, and is occasionally the point — it lets code written for `D`
dimensions run unchanged at ``D = 1``. Zero factors are rejected.

```@example utp
B1 = TensorProductBasis(BSplineBasis(UniformMesh(7, 0 .. 1), 3))
ndims(B1), size(B1)
```

## Accessors return per-axis tuples

```@example utp
degree(B), order(B), ncells(B), size(B), local_width(B)
```

| call | returns |
|:--|:--|
| `ndims(B)`, `size(B)`, `size(B, d)`, `length(B)` | ``D``, the per-axis dimensions, one of them, and ``\prod_d N_d`` |
| [`nbasis`](@ref)`(B)` | `length(B)` — the total, **not** a tuple |
| [`degree`](@ref), [`order`](@ref), [`ncells`](@ref), [`mesh`](@ref), [`boundary`](@ref), [`local_width`](@ref) | per-axis tuples |
| [`meshwidth`](@ref)`(B)` | the **maximum** over the axes, a scalar |
| [`domain`](@ref)`(B)` | a `DomainSets` `ProductDomain`, so `x ∈ domain(B)` answers for a ``D``-vector |
| [`nodes`](@ref)`(B)` | the per-axis node vectors; the grid is their product and is not materialised |
| [`polynomial_reproduction`](@ref)`(B)` | the **minimum** over the axes |
| `CartesianIndices(B)`, `LinearIndices(B)`, `eachindex(B)` | index sets; `eachindex` is the Cartesian one |

```@example utp
[1.0, 2.0, 0.5] ∈ domain(B), meshwidth(B), polynomial_reproduction(B)
```

Two of those are easy to misread. `nbasis(B)` is the *total* dimension, because that is the
length a flattened coefficient vector has; `size(B)` is the per-axis tuple. And
`boundary(B)` is a tuple of whatever each axis reports, so a periodic axis contributes a bare
`Periodic()` while a bounded one contributes a pair:

```@example utp
boundary(B)
```

## Coefficients

A coefficient array has size `size(B)`, one axis per factor.

```@example utp
û = zeros(size(B)...)
û[3, 4, 2] = 1.0
size(û) == size(B)
```

When it has to be flattened, the convention is that the **first axis varies fastest** — `vec`,
`LinearIndices(B)`, Julia's column-major order — and the assembled mass matrix is
`kron(M_D, …, M_1)` to match.

```@example utp
LinearIndices(B)[3, 4, 2], findfirst(!iszero, vec(û))
```

## Evaluating

```julia
evaluate(B, I, x, d = ntuple(_ -> 0, D))       # I a CartesianIndex, a tuple, or a linear index
evaluate(B, û, x, d = ntuple(_ -> 0, D))       # the spline
B[x, I]        B(x, I)
```

`x` is any indexable ``D``-vector: a tuple, an `SVector`, a `Vector`. As on one axis,
[`evaluate`](@ref) takes the **index first** and `getindex` the **point first**.

```@example utp
Bp = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(8, 0 .. 1), 2)
b1, b2 = bases(Bp)
pt = (0.3, 0.4)
(evaluate(Bp, (3, 4), pt),
    evaluate(Bp, CartesianIndex(3, 4), pt),
    evaluate(Bp, LinearIndices(Bp)[3, 4], pt),
    Bp[pt, (3, 4)],
    evaluate(b1, 3, 0.3) * evaluate(b2, 4, 0.4))
```

### Derivatives are a per-axis tuple

There is no scalar derivative order. `d` is an `NTuple{D,Int}` of orders, giving
``\partial_1^{d_1}\cdots\partial_D^{d_D}``; the gradient component ``\partial_k`` is
`ntuple(i -> i == k ? 1 : 0, D)`.

```@example utp
evaluate(Bp, (3, 4), pt, (1, 2)) ≈
    evaluate(b1, 3, 0.3, 1) * evaluate(b2, 4, 0.4, 2)
```

For a [`Spline`](@ref) on a product basis, `derivative(s, k::Integer)` builds that tuple for
you — see [Splines](@ref usage-splines).

### The local path

[`evaluate_all!`](@ref) takes a **tuple of buffers**, one per axis, each of length
`local_width(bases(B)[k])`, and returns the tuple of first indices. The ``D``-dimensional block
is deliberately not materialised: at ``D = 3`` with cubics that is 64 numbers per particle,
and the loop that consumes them can form each product as it goes.

```@example utp
bufs = ntuple(k -> zeros(local_width(bases(Bp)[k])), ndims(Bp))
j₀ = evaluate_all!(bufs, Bp, pt)
j₀, map(length, bufs)
```

```@example utp
all(bufs[k] ≈ evaluate_all(bases(Bp)[k], pt[k])[2] for k in 1:ndims(Bp))
```

A deposition then reads:

```@example utp
sz = size(Bp)
coeffs = zeros(sz)
for (pos, wt) in ((0.13, 0.22) => 1.0, (0.71, 0.44) => 2.5)
    jj = evaluate_all!(bufs, Bp, pos)
    for t in CartesianIndices(map(length, bufs))
        I = ntuple(k -> basis_index(bases(Bp)[k], jj[k] + t[k] - 1), ndims(Bp))
        all(k -> 1 ≤ I[k] ≤ sz[k], 1:ndims(Bp)) || continue
        coeffs[I...] += wt * prod(ntuple(k -> bufs[k][t[k]], ndims(Bp)))
    end
end
sum(coeffs)
```

The sum is ``3.5``, each particle having deposited its whole weight. As on one axis, the
returned indices are the ones *before* wrapping, so they go through [`basis_index`](@ref); and
an index outside `1:N_d` on a bounded axis is genuinely absent rather than zero, so it is
skipped.

A point outside the box evaluates to exactly zero:

```@example utp
BD = BSplineBasis(UniformMesh(8, 0 .. 1), 3, Dirichlet()) ⊗
     BSplineBasis(UniformMesh(8, 0 .. 1), 3)
v̂ = randn(size(BD)...)
evaluate(BD, v̂, (1.5, 0.5)), evaluate(BD, v̂, (0.5, -0.2))
```

## Assembly

```julia
TensorProductQuadrature(B; nq = map(quadrature_order, degree(B)), dmax = 3)
```

`nq` and `dmax` may each be a ``D``-tuple or a scalar broadcast to every axis. There is no
``D``-dimensional tabulation: the quadrature is ``D`` one-dimensional
[`SplineQuadrature`](@ref)s plus the [`KroneckerMass`](@ref) built from their mass operators.

```@example utp
Q = TensorProductQuadrature(Bp; nq = (5, 7), dmax = (2, 3))
map(length, quadrature_nodes(Q)), quadrature_grid_size(Q)
```

| call | returns |
|:--|:--|
| [`quadratures`](@ref)`(Q)` | the tuple of one-dimensional [`SplineQuadrature`](@ref)s |
| [`basis`](@ref)`(Q)` | the [`TensorProductBasis`](@ref) |
| [`quadrature_nodes`](@ref)`(Q)`, [`quadrature_weights`](@ref)`(Q)` | per-axis **tuples** of vectors |
| [`quadrature_grid_size`](@ref)`(Q)` | `(n_d * nq_d)` per axis |
| [`quadrature_sample`](@ref)`(Q, f)` | `f` evaluated on the grid, as an array of that size |
| [`mass_operator`](@ref)`(Q)`, [`mass_matrix`](@ref)`(Q)` | the [`KroneckerMass`](@ref), and its assembled Kronecker product |
| `ndims(Q)`, `size(Q)`, `nbasis(Q)`, `degree(Q)`, `eltype(Q)` | forwarded to the basis |

The per-axis structure is why the one-dimensional accessors are reached through
`quadratures(Q)[k]`:

```@example utp
q1, q2 = quadratures(Q)
size(basis_values(q1, 0)), size(basis_values(q2, 3))
```

```@example utp
try
    basis_values(q1, 3)               # axis 1 was built with dmax = 2
catch err
    println(err.msg)
end
```

Note that a grid point is `(x[1][r₁], x[2][r₂], …)` and its weight the *product* of the
per-axis weights — [`quadrature_sample`](@ref) and [`contract`](@ref) apply that for you.

```@example utp
X = quadrature_nodes(Q)
F = quadrature_sample(Q, x -> x[1]^2 * x[2])
size(F) == quadrature_grid_size(Q), F[3, 4] ≈ X[1][3]^2 * X[2][4]
```

### Contraction and projection

[`contract`](@ref)`(Q, F, d)` turns an array given on the quadrature grid into the load array

```math
L_{i_1 \dots i_D} = \sum_{q_1 \dots q_D} F_{q_1 \dots q_D}
    \prod_{k} D^{d_k} \phi^{(k)}_{i_k}(x_{q_k}) \, w_{q_k} ,
```

as ``D`` sparse matrix products. `F` must already contain the whole integrand; the weights are
applied by `contract` itself, and always on a *copy*, so the caller's sample is not scaled.

```@example utp
L = contract(Q, F)
size(L) == size(Bp), F[3, 4] ≈ X[1][3]^2 * X[2][4]     # F is unchanged
```

[`l2_projection`](@ref) is then that contraction followed by the factored mass solve. `f` may
be a function of a ``D``-tuple or an array already sampled on the grid.

```@example utp
f(x) = x[1]^2 * x[2] + 3x[1] - x[2]^2 + 1
û2 = l2_projection(Q, f)
ŵ = similar(û2)
l2_projection!(ŵ, Q, f)
(size(û2),
    maximum(abs, ŵ - û2),
    maximum(abs, l2_projection(Q, quadrature_sample(Q, f)) - û2))
```

```@example utp
maximum(abs(evaluate(Bp, û2, (x, y)) - f((x, y)))
        for x in range(0, 1; length = 21), y in range(0, 1; length = 21))
```

That is round-off rather than an approximation: `Bp` is cubic × quadratic, so
[`polynomial_reproduction`](@ref) is `2` and ``x^2 y`` is reproduced exactly.

## The Kronecker mass operator

[`KroneckerMass`](@ref) holds the ``D`` one-dimensional operators and is never assembled. Each
factor keeps whatever representation its own axis earned:

```@example utp
op = mass_operator(TensorProductQuadrature(bx ⊗ bv))
map(fac -> nameof(typeof(fac)), mass_factors(op)), size(op), ndims(op)
```

```julia
op * X      op \ X      ldiv!(Y, op, X)      ldiv!(op, X)      mass_solve!(Y, op, X)
mass_matrix(op)         Matrix(op)           mass_factors(op)
```

Arrays of size `size(B)` **and** flat vectors of length `length(B)` are both accepted; a
vector is reshaped on the first-axis-fastest convention.

```@example utp
opp = mass_operator(Q)
Û = randn(size(Bp)...)
(maximum(abs, opp \ vec(Û) - vec(opp \ Û)),
    maximum(abs, opp \ (opp * Û) - Û))
```

[`mass_matrix`](@ref)`(op)` forms `kron(M_D, …, M_1)` on demand and does **not** store it — it
is what the representation exists to avoid, and at three dimensions it will not fit. It is
provided so that a test can check the factored solve against the dense one at a size where
both are possible.

```@example utp
M1, M2 = map(mass_matrix, mass_factors(opp))
maximum(abs, Matrix(opp) - kron(Matrix(M2), Matrix(M1)))
```

## What is not provided

A matrix ``\int_\Omega f \, D^a\Phi_I \, D^b\Phi_J`` with a **non-separable** ``f`` does not
factorise, and there is no `weighted_matrix` for a tensor-product quadrature. Building one
means forming the Kronecker product of the one-dimensional tabulations explicitly and paying
the storage — at ``64^2`` cubic cells that is some 26 MB per derivative multi-index. The
pieces are exposed if you need it:

```@example utp
Φ1, Φ2 = basis_values(quadratures(Q)[1], 0), basis_values(quadratures(Q)[2], 0)
size(Φ1), size(Φ2), size(kron(Φ2, Φ1))
```

Note the factor order, `kron(Φ2, Φ1)`: the same first-axis-fastest convention as everywhere
else.
