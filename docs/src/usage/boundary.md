```@meta
CurrentModule = SimpleSplines
```

# [Boundary Conditions](@id usage-boundary)

The seven conditions, how to write them, and what the recombined basis they produce does.
For why recombination works and what it costs, see [Boundary Conditions](@ref theory-boundary)
in the Theory section.

```@example ubnd
using SimpleSplines
using LinearAlgebra
using SparseArrays
```

## The seven types

| type | condition | order ``m`` | constructor |
|:--|:--|:--|:--|
| [`Free`](@ref) | none — the plain clamped basis | ``-1`` | `Free()` |
| [`Periodic`](@ref) | the periodic closure of the whole axis | ``-1`` | `Periodic()` |
| [`Dirichlet`](@ref) | ``u = 0`` | ``0`` | `Dirichlet()` |
| [`Neumann`](@ref) | ``u' = 0`` | ``1`` | `Neumann()` |
| [`Natural`](@ref) | ``u'' = 0`` | ``2`` | `Natural()` |
| [`Robin`](@ref) | ``\alpha u + \beta u' = 0`` | ``1``, or ``0`` if ``\beta = 0`` | `Robin(α, β)` |
| [`Constraint`](@ref) | ``\sum_k c_{k+1} D^k u = 0`` | `findlast(!iszero, c) - 1` | `Constraint(c...)` |

`Robin` promotes its arguments, so `Robin(1, 2.0)` works, and rejects ``\alpha = \beta = 0``.
`Constraint` takes the coefficient of ``u`` first, then of ``u'``, and so on; it rejects an
all-zero and an empty coefficient list. The named types are preferable where they apply:
they say what is meant, and `Dirichlet` in addition takes the cheaper elimination path.

```@example ubnd
Robin(1, 2.0), Constraint(0, 0, 0, 1), constraint_order(Constraint(1, 0))
```

## Three traits, one meaning

Every condition is given its meaning in exactly one place, so that the recombination has a
single implementation:

```@example ubnd
for bc in (Free(), Periodic(), Dirichlet(), Neumann(), Natural(),
    Robin(2.0, 3.0), Robin(1.0, 0.0), Constraint(1, 0, -2))
    println(rpad(repr(bc), 20),
        "  coefficients = ", rpad(repr(constraint_coefficients(bc)), 12),
        "  order = ", rpad(constraint_order(bc), 3),
        "  costs = ", nconstraints(bc))
end
```

  - [`constraint_coefficients`](@ref) — the ``(c_0, c_1, \dots)``, or `nothing` for the two
    conditions that impose none.
  - [`constraint_order`](@ref) — the highest derivative appearing, or `-1`. Note
    `constraint_order(Robin(1.0, 0.0)) == 0`: the trailing zero does not count.
  - [`nconstraints`](@ref) — the degrees of freedom removed: `0` or `1`, never more, whatever
    the order.

## Writing a specification

[`boundary_conditions`](@ref) normalises whatever you pass:

| you write | you get |
|:--|:--|
| a single condition | `(bc, bc)` — both ends |
| a two-tuple | `(left, right)` |
| `Periodic()` | `Periodic()`, **bare** — not a pair |
| a `Symbol`, or a tuple of them | the same, with each symbol resolved |

```@example ubnd
(boundary_conditions(Dirichlet()),
    boundary_conditions((Dirichlet(), Neumann())),
    boundary_conditions(:periodic),
    boundary_conditions((:free, Natural())))
```

`Periodic` comes back unpaired because it is a condition on the *axis*, not on its ends: there
is no such thing as being periodic at the left end only, so pairing it with anything is
rejected.

```@example ubnd
try
    boundary_conditions((Periodic(), Dirichlet()))
catch err
    println(err.msg)
end
```

### Symbol sugar

`:periodic`, `:dirichlet`, `:neumann`, `:natural`, `:free`. Lowercase only, and deliberately
**not** case-insensitive: `:Natural`, `:nothing`, `:Dirichlet` and `:Periodic` are rejected
with a message saying what to write instead, and any other symbol with the list of the five
that are accepted.

```@example ubnd
for s in (:Natural, :nothing, :Dirichlet, :quasiperiodic)
    try
        BoundaryCondition(s)
    catch err
        println(err.msg, "\n")
    end
end
```

## Which basis a condition selects

```@example ubnd
m = UniformMesh(8, 0 .. 1)
for bc in (Free(), Periodic(), Dirichlet(), Neumann(), Natural(),
    Robin(1.0, 2.0), (Dirichlet(), Free()), (Natural(), Neumann()))
    bb = BSplineBasis(m, 3, bc)
    println(rpad(repr(bc), 24), rpad(nameof(typeof(bb)), 24),
        "N = ", rpad(nbasis(bb), 4), "reproduces ", polynomial_reproduction(bb))
end
```

Read it back with [`boundary`](@ref):

```@example ubnd
(boundary(BSplineBasis(m, 3)),
    boundary(BSplineBasis(m, 3, Periodic())),
    boundary(BSplineBasis(m, 3, (Dirichlet(), Neumann()))))
```

## Working with a recombined basis

A [`RecombinedBSplineBasis`](@ref) answers the whole [`AbstractBSplineBasis`](@ref) interface,
so most code needs to know nothing about it. Four things are specific to it.

**Its parent is reachable.** `parent(b)` is the clamped basis the recombination was built
from, and [`recombination_matrix`](@ref) is the sparse ``R`` with
``\psi_j = \sum_i R_{ij}\varphi_i``.

```@example ubnd
b = BSplineBasis(m, 3, Natural())
bp = parent(b)
R = recombination_matrix(b)
nbasis(bp), nbasis(b), size(R), nnz(R)
```

Every assembly is the parent's conjugated by ``R``, which is worth knowing when comparing
against a hand-built reference:

```@example ubnd
q, qp = SplineQuadrature(b), SplineQuadrature(bp)
maximum(abs, Matrix(mass_matrix(q)) - Matrix(R' * mass_matrix(qp) * R))
```

**[`local_width`](@ref) may exceed ``p+1``.** A recombined function near an end spans the
union of two parent supports, so the local block is wider there. Any buffer for
[`evaluate_all!`](@ref) must have exactly that many entries, and how many depends on the
condition rather than on the degree alone:

```@example ubnd
[(bc, degree(BSplineBasis(m, 3, bc)) + 1, local_width(BSplineBasis(m, 3, bc)))
 for bc in (Free(), Dirichlet(), Neumann(), Natural(), Constraint(0, 0, 0, 1))]
```

```@example ubnd
degree(b) + 1, local_width(b),
[length(collect(local_indices(b, k))) for k in 1:ncells(b)]
```

```@example ubnd
try
    evaluate_all!(zeros(degree(b) + 1), b, 0.3)
catch err
    println(err.msg)
end
```

So `local_width(b)`, never `degree(b) + 1`, is what a buffer should be sized by.

**[`nodes`](@ref) is not a Schoenberg-Whitney set.** For a recombined basis `nodes` returns
the Greville abscissa of the parent function each column carries with unit coefficient. Those
are `N` distinct increasing points inside the domain, interlaced with the supports — which is
what a plotting or collocation grid wants — but the collocation matrix ``\psi_j(\xi_i)`` is
not guaranteed invertible, and that is not claimed.

```@example ubnd
nodes(b)
```

**The dimension is `n + p` less one per constrained end**, whatever the order of the
condition. `Natural()` reaches three parent functions and removes one, exactly as `Dirichlet()`
reaches one and removes one.

```@example ubnd
[(bc, nbasis(BSplineBasis(m, 4, bc)))
 for bc in (Dirichlet(), Neumann(), Natural(), Constraint(0, 0, 0, 1))]
```

## Traps

  - **A condition is imposed on the space, not on a solution.** There is no "apply the
    boundary condition" step after assembling: the basis has one function fewer per constrained
    end, so a stiffness matrix built on it is already the matrix of the constrained problem.
    Do not also delete rows.
  - **Homogeneous only.** ``u(a) = g \ne 0`` is not a boundary condition here. Write
    ``u = u_0 + w`` with ``u_0`` any function taking the required values and ``w`` in the
    homogeneous space, and solve for ``w``.
  - **`Free` is not `Natural`.** `Free()` is the *absence* of a condition; the natural
    boundary condition is ``u'' = 0``, which is [`Natural`](@ref). They span different spaces
    and have different dimensions. The confusion is easy because a clamped basis is sometimes
    loosely called a "natural" spline basis, and it is why `:Natural` is rejected as a symbol.
  - **A [`Dirichlet`](@ref) basis does not reproduce the constants**, so
    ``\int_\Omega u_h \, \mathrm{d}x`` is not conserved by anything built on it and
    [`basis_integrals`](@ref)` != mass_matrix(q) * ones(N)`. Check
    [`polynomial_reproduction`](@ref) before assuming a conservation law survives.
  - **An ill-scaled [`Robin`](@ref) degrades silently.** `Robin(1.0, 1e-20)` makes the
    recombination anchor numerically zero; the mass matrix comes out finite with condition
    number already `Inf`, and the Cholesky reports success. Keep the two coefficients within a
    few orders of magnitude of each other, and use the named types where they apply.

```@example ubnd
bd = BSplineBasis(m, 3, Dirichlet())
qd = SplineQuadrature(bd)
maximum(abs, basis_integrals(qd) - mass_matrix(qd) * ones(nbasis(bd)))
```

That is the size of the discrepancy the partition of unity would have made zero.
