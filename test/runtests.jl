using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Meshes" include("mesh.jl")
    @safetestset "Boundary conditions" include("boundary.jl")
    @safetestset "Periodic B-spline bases" include("bspline.jl")
    @safetestset "Clamped and recombined bases" include("basis.jl")
    @safetestset "Mass operators" include("mass.jl")
    @safetestset "Spline quadrature" include("quadrature.jl")
    @safetestset "Tensor products" include("tensorproduct.jl")
    @safetestset "Polar splines" include("polar.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
