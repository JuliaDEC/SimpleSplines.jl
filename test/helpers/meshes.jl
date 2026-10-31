# The mesh families that `bspline.jl` and `quadrature.jl` sweep over.
const MESHES = ((:uniform, n -> UniformMesh(n, 2π)),
    (:graded, n -> GradedMesh(n, 2π)),
    (:random, n -> RandomMesh(n, 2π)))

# A span [a, b] with independently drawn endpoints, so that b - a is in general inexact and
# a + (b - a) need not round back to b. Drawing b as a + w would not do: for |w| ≤ |a| the
# difference of the rounded sum and a is exact, and the end would always come back.
function random_span(rng, ::Type{T}) where {T}
    c = ldexp(2rand(rng) - 1, rand(rng, -3:6))
    w = ldexp(0.1 + rand(rng), rand(rng, -3:4))
    return T(c - w * rand(rng) / 2), T(c + w * (1 + rand(rng)) / 2)
end
