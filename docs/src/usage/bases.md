```@meta
CurrentModule = SimpleSplines
```

# [Bases](@id usage-bases)

Three concrete bases implement [`AbstractBSplineBasis`](@ref), and which one you get is
decided by the boundary condition rather than chosen by name.

```@example bases
using SimpleSplines
```

## Construction

One constructor covers all three:

```julia
BSplineBasis(mesh, p)                 # clamped, the default
BSplineBasis(mesh, p, bc)             # dispatches on bc
```

```@example bases
m = UniformMesh(8, 0 .. 1)
[(bc, nameof(typeof(BSplineBasis(m, 3, bc))), nbasis(BSplineBasis(m, 3, bc)))
 for bc in (Free(), (Free(), Free()), Periodic(), :periodic,
     Dirichlet(), (Dirichlet(), Free()), Neumann(), Natural())]
```

`Periodic()` gives a [`PeriodicBSplineBasis`](@ref), two `Free` ends give the plain clamped
[`BSplineBasis`](@ref) unwrapped, and anything else gives a
[`RecombinedBSplineBasis`](@ref). This is deliberate: it lets one call site select any of the
three, which is exactly what a tensor product with a different condition per axis needs.

The named constructors are also available directly, and `PeriodicBSplineBasis` has a
convenience form that builds its own uniform mesh:

```julia
BSplineBasis(mesh, p)                     BSplineBasis{T}(mesh, p)
PeriodicBSplineBasis(mesh, p)             PeriodicBSplineBasis{T}(mesh, p)
PeriodicBSplineBasis(n, p; L = 2π)        # = PeriodicBSplineBasis(UniformMesh(n, L), p)
RecombinedBSplineBasis(parent, left, right)
```

```@example bases
PeriodicBSplineBasis(16, 3) == BSplineBasis(UniformMesh(16, 2π), 3, Periodic())
```

### What is required, and what is rejected

| basis | requires |
|:--|:--|
| [`BSplineBasis`](@ref) | ``p \ge 0``, ``n \ge 1`` — a single cell carries the whole of ``\mathbb{P}_p`` |
| [`PeriodicBSplineBasis`](@ref) | ``p \ge 0`` and ``n > p``, so that a basis function does not wrap onto itself |
| [`RecombinedBSplineBasis`](@ref) | a **clamped** parent; each condition of order ``\le p``; the two end blocks disjoint; at least one function left |

```@example bases
for bad in (() -> BSplineBasis(UniformMesh(8, 0 .. 1), -1),
    () -> PeriodicBSplineBasis(UniformMesh(3, 0 .. 1), 3),
    () -> BSplineBasis(UniformMesh(8, 0 .. 1), 1, Natural()),
    () -> BSplineBasis(UniformMesh(1, 0 .. 1), 0, Dirichlet()),
    () -> RecombinedBSplineBasis(BSplineBasis(UniformMesh(8, 0 .. 1), 3),
        Periodic(), Dirichlet()))
    try
        bad()
    catch err
        println(err.msg, "\n")
    end
end
```

## Accessors

```@example bases
b = BSplineBasis(UniformMesh(8, 0 .. 1), 3)
(nbasis(b), degree(b), order(b), ncells(b), local_width(b),
    polynomial_reproduction(b), boundary(b))
```

| call | returns |
|:--|:--|
| [`nbasis`](@ref)`(b)` | the dimension ``N``, and the length a coefficient vector must have |
| [`degree`](@ref)`(b)` | the polynomial degree ``p`` |
| [`order`](@ref)`(b)` | ``p + 1`` |
| [`ncells`](@ref)`(b)`, [`mesh`](@ref)`(b)`, [`domain`](@ref)`(b)`, [`domainlength`](@ref)`(b)`, [`meshwidth`](@ref)`(b)` | forwarded to the mesh |
| [`breakpoints`](@ref)`(b)` | the `n+1` cell boundaries, **cached in the basis** |
| [`knotvector`](@ref)`(b)` | the knot sequence the recursion runs on |
| [`boundary`](@ref)`(b)` | `Periodic()`, or the pair `(left, right)` |
| [`nodes`](@ref)`(b)` | the Greville abscissae; `ContinuumArrays`' `grid` is the same |
| `nnodes(b)` | `nbasis(b)` |
| [`local_width`](@ref)`(b)` | how many entries [`evaluate_all!`](@ref) writes: ``p+1``, or more near a recombined end |
| [`polynomial_reproduction`](@ref)`(b)` | the largest degree the span reproduces exactly, or `-1` |
| [`basis_index`](@ref)`(b, j)` | `j`, wrapped onto `1:N` where the basis is periodic |
| `eachindex(b)`, `axes(b)`, `eltype(b)` | `1:N`, `(Inclusion(domain), 1:N)`, the element type |

!!! note "`order` here means `p + 1`"
    That is the *spline* convention: a B-spline of order ``k`` is piecewise of degree
    ``k - 1``. It is **not** the meaning `order` carries for the bases of
    `CompactBasisFunctions`, where it is the number of basis functions. For a spline basis the
    two differ, and `nbasis` is the one you want for a dimension.

## Evaluating

```julia
evaluate(b, j, x, d = 0)                # basis function j
evaluate(b, û, x, d = 0)                # the spline with coefficients û
b[x, j]     b[x, :]     b[X, j]     b[X, :]        b(x, j)
b'          # a BSplineDerivative: first order only
```

`x` may be a number or a vector; with a vector the result is a vector (or, for `b[X, :]`, a
matrix of points × functions).

```@example bases
xs = [0.1, 0.5, 0.9]
evaluate(b, 3, 0.3), evaluate(b, 3, 0.3, 2), size(b[xs, :]), b'[0.3, 3]
```

!!! warning "The argument order is not the same in the two spellings"
    [`evaluate`](@ref) takes the **index first**, `evaluate(b, j, x, d)`, because the
    derivative order comes last. `getindex` and the callable form take the **point first**,
    `b[x, j]` and `b(x, j)`, to match a matrix of samples whose rows are points. Both spellings
    are correct; mixing them up is silent whenever `j` and `x` are both plausible numbers.

### Derivatives of arbitrary order

The fourth argument of [`evaluate`](@ref) is the derivative order, any non-negative integer.
`b'` is a [`BSplineDerivative`](@ref), which is a lazy product and **first order only** — the
products do not compose, so a second derivative is not `b''`.

```@example bases
[evaluate(b, 3, 0.3, d) for d in 0:4]
```

Orders above `p` are identically zero and are returned as such rather than raised, so a loop
over derivative orders needs no special case. A negative order is an error.

### Outside the domain

A bounded basis evaluates to **zero** outside its closed domain — not an error, and not an
extrapolated polynomial. That is the mathematically correct value for a compactly supported
function, and it is the behaviour a particle method wants; it is also silent, so it is worth
knowing.

```@example bases
sum(b[1.7, :]), evaluate(b, 4, -0.2)
```

A [`PeriodicBSplineBasis`](@ref) instead accepts any real argument and reduces it onto the
domain, so the result is the periodic extension:

```@example bases
c = BSplineBasis(UniformMesh(8, 0 .. 1), 3, Periodic())
sum(c[1.7, :]), evaluate(c, 4, 0.7) ≈ evaluate(c, 4, 0.7 + 3.0)
```

### At the right endpoint

Knot spans are half-open, except the topmost, which is closed on the right so that the
clamped basis is interpolatory at ``b`` as it should be. The value there is exactly the limit
from the left.

```@example bases
b[1.0, nbasis(b)], sum(b[1.0, :])
```

## The local path

A loop that needs *every* nonzero function at a point — a matrix assembly, a particle
deposition — should not call [`evaluate`](@ref) once per index. That path runs the recursion
literally, at ``O(2^p)`` per value; [`evaluate_all!`](@ref) fills a caller-supplied buffer with
the whole local block by de Boor's triangular scheme, in ``O(p^2)`` for all `p+1` together, and
allocates nothing:

```@example bases
buf = zeros(local_width(b))
j₀ = evaluate_all!(buf, b, 0.3)
j₀, buf, sum(buf)
```

```@example bases
all(buf[t] ≈ evaluate(b, basis_index(b, j₀ + t - 1), 0.3) for t in eachindex(buf))
```

Three rules for using it:

1. **The buffer must have exactly [`local_width`](@ref)`(b)` entries.** That is `p+1` for a
   clamped or periodic basis, and possibly more for a recombined one, where a function near an
   end spans the union of two parent supports. A wrong length is a `DimensionMismatch`.
2. **`j₀` is the first index *before* wrapping**, because the block is contiguous only in that
   form. On a periodic basis it can fall outside `1:N`; put every index through
   [`basis_index`](@ref) rather than writing `mod1` at the call site, since whether the wrap is
   needed is a property of the basis and getting it wrong on a bounded basis silently folds the
   two ends of the domain together.
3. **Outside a bounded domain the buffer is zeroed** and the *clamped* cell's `j₀` is
   returned, so a deposition loop touches the same block it would for a point just inside and
   adds nothing to it.

```@example bases
buf2 = zeros(local_width(c))
jc = evaluate_all!(buf2, c, 0.05)             # near the periodic seam
jc, [basis_index(c, jc + t - 1) for t in eachindex(buf2)]
```

The allocating form [`evaluate_all`](@ref) returns `(j₀, values)` with a fresh buffer, which
is convenient at a call site that runs once and wasteful in a loop.

A deposition therefore reads:

```@example bases
coeffs = zeros(nbasis(b))
particles = [(0.13, 1.0), (0.42, 2.0), (0.97, 0.5), (1.4, 3.0)]   # the last is outside
scratch = zeros(local_width(b))
for (x, w) in particles
    j = evaluate_all!(scratch, b, x)
    for t in eachindex(scratch)
        i = basis_index(b, j + t - 1)
        1 ≤ i ≤ nbasis(b) || continue
        coeffs[i] += w * scratch[t]
    end
end
coeffs, sum(coeffs)
```

The sum is ``1.0 + 2.0 + 0.5 = 3.5`` and not ``6.5``: the partition of unity means each
particle deposits its whole weight, and the one outside the domain deposited nothing.

## Cell lookup

[`findcell`](@ref)`(b, x)` gives the cell index in `1:ncells`, reading the breakpoints cached
in the basis so that it allocates nothing; on a uniform mesh it is a division that does not
touch them at all. [`local_indices`](@ref)`(b, cell)` gives the indices of the functions
nonzero on a cell — `cell:(cell+p)` for a clamped basis, the same block wrapped for a periodic
one, and a precomputed range for a recombined one.

```@example bases
findcell(b, 0.3), collect(local_indices(b, 3)), collect(local_indices(c, 1))
```

Note the periodic case: cell `1` is reached by functions `6, 7, 8, 1`, wrapped.

## Comparison

`==` compares type, degree, mesh and boundary condition; `isequal` additionally requires the
same element type; `isapprox` compares the meshes approximately.

```@example bases
b == BSplineBasis(UniformMesh(8, 0 .. 1), 3),
b == BSplineBasis(UniformMesh(8, 0 .. 1), 3, Dirichlet()),
b == c
```
