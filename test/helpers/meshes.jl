# The mesh families that `bspline.jl` and `quadrature.jl` sweep over.
const MESHES = ((:uniform, n -> UniformMesh(n, 2π)),
    (:graded, n -> GradedMesh(n, 2π)),
    (:random, n -> RandomMesh(n, 2π)))
