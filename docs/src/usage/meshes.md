```@meta
CurrentModule = SimpleSplines
```

# [Meshes](@id usage-meshes)

A [`Mesh`](@ref) is a subdivision of a closed interval ``[a,b]`` into `n` cells by `n+1`
breakpoints. It is geometry alone: it carries no degree and no boundary condition, and the
same mesh serves a clamped, a recombined and a periodic basis.

```@example mesh
using SimpleSplines
```

## The four families

| mesh | breakpoints | refining `n` gives | use it for |
|:--|:--|:--|:--|
| [`UniformMesh`](@ref) | ``y_i = a + (i-1)h`` | a uniform refinement | ordinary computation — but its periodic assemblies are *circulant*, so it confirms some identities for the wrong reason |
| [`GradedMesh`](@ref) | the image of a uniform mesh under one fixed smooth map | a genuine mesh family | measuring convergence rates |
| [`RandomMesh`](@ref) | perturbed cell widths from a seeded stream | an *unrelated* mesh | properties that must hold on **any** mesh |
| [`GeneralMesh`](@ref) | whatever you pass | whatever you pass | a subdivision none of the three describes |

The distinction between the first three is about what is being *tested*, not about the
discretisation. A convergence rate measured across a family of `RandomMesh`es means nothing,
because each `n` gives a different mesh rather than a refinement of the previous one; that is
what `GradedMesh` is for. Conversely, an identity checked only on a `UniformMesh` may hold
there because the assemblies are circulant and fail everywhere else, which is what
`RandomMesh` catches.

```@example mesh
[(nameof(typeof(m)), ncells(m), round(meshwidth(m); digits = 4))
 for m in (UniformMesh(8, 0 .. 1), GradedMesh(8, 0 .. 1), RandomMesh(8, 0 .. 1))]
```

## Constructors

```julia
UniformMesh(n, domain)                                  # UniformMesh(n) is [0, 2π]
GradedMesh(n, domain;  amplitude = 0.12)
RandomMesh(n, domain;  seed = 1, spread = 0.6)
GeneralMesh(breakpoints)
```

Each also has a `{T}` form — `UniformMesh{Float32}(8, 0 .. 1)` — for fixing the breakpoint
element type explicitly.

### Three ways to write the domain

A `ClosedInterval`, a two-tuple, or a bare number standing for ``[0, L]``. The last is kept
because `UniformMesh(16, 2π)` is the common periodic case.

```@example mesh
UniformMesh(8, 0 .. 1) == UniformMesh(8, (0, 1)) == UniformMesh(8, 1)
```

```@example mesh
domain(UniformMesh(8)), domain(UniformMesh(4, -10 .. 10))
```

`..` is re-exported, so `using SimpleSplines` is enough to write an interval; so are
`leftendpoint` and `rightendpoint`.

```@example mesh
leftendpoint(domain(UniformMesh(4, -10 .. 10))),
rightendpoint(domain(UniformMesh(4, -10 .. 10)))
```

### Element types

An **integer** domain is promoted to a float, because the breakpoints of a subdivision of it
are not integers — otherwise `UniformMesh(4, 0 .. 1)` would try to store `0.25` in an `Int`
and throw an `InexactError` from a long way away. `Rational` and extended-precision types are
left alone, since they are closed under the division `breakpoints` performs and narrowing them
would discard the exactness that is the reason for using them.

```@example mesh
eltype(UniformMesh(4, 0 .. 1)),
eltype(UniformMesh(4, 0 .. 1//1)),
eltype(UniformMesh(4, big(0.0) .. big(1.0)))
```

## Accessors

| call | returns |
|:--|:--|
| [`breakpoints`](@ref)`(m)` | the `n+1` cell boundaries, increasing, both endpoints included |
| [`ncells`](@ref)`(m)` | `n` |
| [`domain`](@ref)`(m)` | the `ClosedInterval` ``[a,b]`` |
| [`domainlength`](@ref)`(m)` | ``b - a``; named this way because it is the period ``L`` of a periodic basis |
| [`meshwidth`](@ref)`(m)` | the **largest** cell width, the ``h`` a convergence rate is measured against |
| `eltype(m)`, `length(m)`, `first(m)`, `last(m)` | element type, `ncells`, and the two endpoints |

```@example mesh
m = GeneralMesh([-20.0; range(-10, 10; length = 5); 20.0])
ncells(m), domain(m), meshwidth(m), diff(breakpoints(m))
```

`meshwidth` is the largest cell, not the average: for that mesh it is `10.0`, not `40/6`.

!!! warning "`breakpoints` must not be mutated"
    For [`GradedMesh`](@ref), [`RandomMesh`](@ref) and [`GeneralMesh`](@ref) this returns the
    mesh's *own* array rather than a copy, which is what keeps [`findcell`](@ref) off the
    allocator on the innermost loop of a particle deposition. Writing to it corrupts the mesh,
    and silently: the domain and cell count are unchanged so nothing rejects the result, while
    `hash` and `==` now report a different mesh. [`UniformMesh`](@ref) computes its
    breakpoints from a closed form and hands back a fresh vector, but that is an
    implementation detail, not a licence. Take a `copy` if you need to modify it.

## Equality is geometric, `isequal` is not

`==` and `hash` compare the domain and the breakpoints. `isequal` also compares the *type*,
because the mesh type selects the assembly path — an equally spaced [`GeneralMesh`](@ref)
deliberately does not take the [`CirculantMass`](@ref) route that a [`UniformMesh`](@ref) with
the same breakpoints does.

```@example mesh
u = UniformMesh(4, 0 .. 1)
g = GeneralMesh(collect(range(0, 1; length = 5)))
u == g, u ≈ g, isequal(u, g)
```

So the two are interchangeable as values and distinct as dictionary keys, and an equally
spaced `GeneralMesh` is a legitimate way to force the general assembly path in a test.

## Keyword arguments worth knowing

`GradedMesh`'s `amplitude` is the ``a`` in the map ``s \mapsto s + (a/2\pi)\sin 2\pi s``. It
must satisfy ``|a| < 1`` for the map to stay increasing. The derivative of the map varies
between ``1-a`` and ``1+a``, so the ratio of the widest to the narrowest cell approaches
``(1+a)/(1-a)`` as the mesh is refined — the two below differ because a cell width is a finite
difference of the map, not its derivative.

```@example mesh
w = diff(breakpoints(GradedMesh(12, 0 .. 1; amplitude = 0.4)))
maximum(w) / minimum(w), (1 + 0.4) / (1 - 0.4)
```

`RandomMesh`'s widths are ``1 + \sigma r_i`` with ``r_i`` uniform on ``[0,1)`` and `spread`
``= \sigma``, rescaled to fill the domain. The stream is seeded, so the mesh is reproducible;
a different `seed` gives a different mesh.

```@example mesh
breakpoints(RandomMesh(4, 0 .. 1; seed = 7)) ==
    breakpoints(RandomMesh(4, 0 .. 1; seed = 7)),
breakpoints(RandomMesh(4, 0 .. 1; seed = 7)) ==
    breakpoints(RandomMesh(4, 0 .. 1; seed = 8))
```

## What is rejected

```@example mesh
for bad in (() -> UniformMesh(0, 0 .. 1),
    () -> UniformMesh(4, 1 .. 0),
    () -> UniformMesh(4, 0 .. Inf),
    () -> GradedMesh(4, 0 .. 1; amplitude = 1.5),
    () -> RandomMesh(4, 0 .. 1; spread = -1),
    () -> GeneralMesh([0.0]),
    () -> GeneralMesh([0.0, 0.5, 0.5, 1.0]),
    () -> GeneralMesh([0.0, 0.7, 0.3]))
    try
        bad()
    catch err
        println(err.msg)
    end
end
```

A repeated breakpoint is rejected rather than tolerated: it is a cell of zero width, which the
Cox-de Boor recursion would silently skip, leaving a basis whose dimension does not match its
knot vector. Multiplicity belongs in the *knot vector*, where the basis puts it, not in the
mesh.

## Finding a cell

[`findcell`](@ref) returns the index of the cell containing a point, in `1:ncells`. The cells
are half-open except for the last, which is closed, so the right endpoint belongs to cell `n`.
A point outside the domain is clamped to the nearest cell.

```@example mesh
u8 = UniformMesh(8, 0 .. 1)
findcell.(Ref(u8), [0.0, 0.06, 0.125, 0.99, 1.0, 5.0, -3.0])
```

On a [`UniformMesh`](@ref) this is a division and needs no breakpoint vector at all; on the
others it is a `searchsortedlast`. Both are allocation-free when called on a *basis*, which
caches the breakpoints — see [Bases](@ref usage-bases).
