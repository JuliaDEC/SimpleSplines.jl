using SimpleSplines
using Random
using Test

# The mesh families and the projection tests draw pseudorandom data. The seed is fixed so
# that a failure is reproducible: a spline assembly that is wrong only for some meshes is
# exactly the kind of fault a fresh stream each run would turn into an intermittent one.
Random.seed!(0x2f7a91c4)

include("aqua_tests.jl")
include("mesh_tests.jl")
include("boundary_tests.jl")
include("bspline_tests.jl")
include("basis_tests.jl")
include("mass_tests.jl")
include("quadrature_tests.jl")
include("tensorproduct_tests.jl")
