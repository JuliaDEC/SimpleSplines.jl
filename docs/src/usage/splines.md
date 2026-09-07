```@meta
CurrentModule = SimpleSplines
```

# [Splines](@id usage-splines)

A basis answers "what is ``\phi_j`` here". A [`Spline`](@ref) answers "what is ``u_h`` here":
it is a basis together with the coefficients of one element of its span, and it is callable.

```@example spl
using SimpleSplines
```

## Construction

```julia
Spline(basis, coefficients)      # coefficients: a vector, or a D-array for a product basis
Spline(basis)                    # the zero spline, freshly allocated in the right shape
```

```@example spl
b = BSplineBasis(UniformMesh(12, 0 .. 1), 3)
q = SplineQuadrature(b)
s = Spline(b, l2_projection(q, x -> sinpi(2x)))
s, s(0.125), s(0.125, 1)
```

```@example spl
z = Spline(b)
size(z), all(iszero, coefficients(z))
```

The shape is checked: a one-dimensional basis needs a vector of `nbasis(basis)` entries, a
product basis an array of `size(basis)`.

```@example spl
try
    Spline(b, zeros(nbasis(b) - 1))
catch err
    println(err.msg)
end
```

## Evaluation

```julia
s(x)          s(x, d)
evaluate(s, x)              evaluate(s, x, d)
```

`d` is the derivative order — an integer on one axis, a per-axis tuple on a product basis.

```@example spl
xs = range(0, 1; length = 5)
s.(xs), s.(xs, 1)
```

Evaluating a spline is ``O(p^2)`` and not ``O(N)``: only the local block of `p+1` functions
contributes, and the value is that block's weighted sum. Summing `evaluate(b, j, x)` over the
whole index range would give the same answer and cost ``O(N)``.

```@example spl
s(0.3) ≈ sum(coefficients(s)[j] * b[0.3, j] for j in eachindex(b))
```

Outside a bounded domain the value is zero, since every basis function vanishes there.

```@example spl
s(-0.1), s(1.4)
```

## Derivatives as objects

[`derivative`](@ref)`(s, d = 1)` gives a [`SplineDerivative`](@ref), which is callable and so
broadcasts the way `s` does. It holds no state of its own — it shares the coefficient array —
so a reprojection is visible through it with nothing rebuilt.

```@example spl
ds = derivative(s)
ds, ds.(xs), maximum(abs(ds(x) - 2π * cospi(2x)) for x in range(0, 1; length = 201))
```

```@example spl
derivative(s, 2)(0.3) ≈ s(0.3, 2)
```

This is the shape a right-hand side wants when the value and the derivative appear together in
one broadcast: `derivative(fs).(v) ./ fs.(v)` reads as it should.

```@example spl
maximum(abs, derivative(s).(xs) ./ (1 .+ s.(xs) .^ 2))
```

## Accessors

```@example spl
(basis(s) === b, eltype(s), ndims(s), size(s), length(s),
    nbasis(s), degree(s), order(s), domain(s))
```

| call | returns |
|:--|:--|
| [`basis`](@ref)`(s)` | the basis |
| [`coefficients`](@ref)`(s)` | the coefficient array **itself**, not a copy |
| `eltype`, `ndims`, `size`, `length` | of the coefficient array |
| `nbasis`, `degree`, `order`, `domain` | forwarded to the basis |
| `similar(s)`, `similar(s, T)`, `copy(s)` | a spline on the same basis with fresh coefficients |

!!! note "`coefficients` aliases, and that is the point"
    It hands back the array the spline holds, so writing into it changes the spline. That is
    what lets a projection be written into a spline already wired into a right-hand side, and
    what lets a particle deposition rewrite the coefficients every step while the basis, the
    quadrature and the mass factorisation stay put.

```@example spl
t = Spline(b, copy(coefficients(s)))
coefficients(t) .= 0
t(0.3)
```

For the same reason, a `Spline` built on an array you already hold shares it:

```@example spl
û = zeros(nbasis(b))
u = Spline(b, û)
û[5] = 1.0
u(0.3) == b[0.3, 5]
```

Use `copy(s)` when you want an independent spline, and `similar(s)` when you want the same
shape with undefined contents.

## Projecting into an existing spline

[`l2_projection!`](@ref)`(s, q, f)` writes into the array `s` already holds and returns `s`,
so every reference to `s` sees the new function.

```@example spl
l2_projection!(u, q, cospi)
u(0.25), coefficients(u) === û
```

## On a product basis

Everything above carries over, with a coefficient array and tuple-valued derivative orders.
The integer form `derivative(s, k)` is the shorthand for ``\partial_k``.

```@example spl
B = BSplineBasis(UniformMesh(8, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(8, 0 .. 2), 3)
Q = TensorProductQuadrature(B)
g(x) = x[1]^2 * x[2]
S = Spline(B, l2_projection(Q, g))
S, ndims(S), size(S)
```

```@example spl
(S((0.3, 1.4)),
    S((0.3, 1.4), (1, 0)),
    derivative(S, 1)((0.3, 1.4)),
    derivative(S, 2)((0.3, 1.4)))
```

``\partial_1 g = 2x_1x_2`` and ``\partial_2 g = x_1^2``, so at ``(0.3, 1.4)`` those are
``0.84`` and ``0.09``.

```@example spl
try
    derivative(S, 3)
catch err
    println(err.msg)
end
```

A mixed derivative is given directly as a tuple:

```@example spl
S((0.3, 1.4), (1, 1))          # ∂₁∂₂ g = 2x₁
```
