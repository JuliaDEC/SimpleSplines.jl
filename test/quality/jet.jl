using JET
using SimpleSplines
using Test

# Static optimisation analysis of the hot paths: every function of `src/` that a test file
# asserts with `@allocated`, at the concrete argument types those tests pass. A runtime
# dispatch on one of these paths is what the allocation tests measure only indirectly.
# Each further element type that a test outside `test/quality/` passes directly to the same
# method gets one line too; an element type that reaches it only through another function
# (`op \ b32` on a `CirculantMass{Float32}`, test/mass.jl) gets none.

if isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
    # test/basis.jl: evaluate_all! on a free and a Dirichlet bounded basis and a periodic basis
    b = BSplineBasis(UniformMesh(32, -10 .. 10), 3)
    br = BSplineBasis(UniformMesh(32, -10 .. 10), 3, Dirichlet())
    bp = BSplineBasis(UniformMesh(32, 0 .. 2π), 3, Periodic())
    @test isempty(JET.get_reports(JET.report_opt(evaluate_all!,
        (Vector{Float64}, typeof(b), Float64, Int); target_modules = (SimpleSplines,))))
    @test isempty(JET.get_reports(JET.report_opt(evaluate_all!,
        (Vector{Float64}, typeof(br), Float64, Int); target_modules = (SimpleSplines,))))
    @test isempty(JET.get_reports(JET.report_opt(evaluate_all!,
        (Vector{Float64}, typeof(bp), Float64, Int); target_modules = (SimpleSplines,))))

    # test/tensorproduct.jl: evaluate on a tensor product of mixed axis types
    B = TensorProductBasis(
        BSplineBasis(UniformMesh(5, 0 .. 1), 1),
        BSplineBasis(UniformMesh(4, -1 .. 1), 2, Periodic()),
        BSplineBasis(UniformMesh(6, 2 .. 5), 3, Dirichlet()))
    @test isempty(JET.get_reports(JET.report_opt(evaluate,
        (typeof(B), Array{Float64, 3}, NTuple{3, Float64});
        target_modules = (SimpleSplines,))))
    # test/basis.jl: complex coefficients on a 2-D tensor product, with a derivative order
    B2 = BSplineBasis(UniformMesh(6, 0 .. 1), 3) ⊗ BSplineBasis(UniformMesh(5, 0 .. 1), 3)
    @test isempty(JET.get_reports(JET.report_opt(evaluate,
        (typeof(B2), Matrix{ComplexF64}, NTuple{2, Float64}, NTuple{2, Int});
        target_modules = (SimpleSplines,))))

    # test/quadrature.jl: basis_integrals and l2_projection! on a uniform periodic basis
    q = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(16), 3))
    @test isempty(JET.get_reports(JET.report_opt(basis_integrals, (typeof(q),);
        target_modules = (SimpleSplines,))))
    q2 = SplineQuadrature(PeriodicBSplineBasis(UniformMesh(32, 2π), 3))
    @test isempty(JET.get_reports(JET.report_opt(l2_projection!,
        (Vector{Float64}, typeof(q2), Vector{Float64}); target_modules = (SimpleSplines,))))
    # test/quadrature.jl: complex samples on a graded periodic quadrature
    qc = SplineQuadrature(PeriodicBSplineBasis(GradedMesh(16, 2π), 3))
    @test isempty(JET.get_reports(JET.report_opt(l2_projection!,
        (Vector{ComplexF64}, typeof(qc), Vector{ComplexF64});
        target_modules = (SimpleSplines,))))

    # test/mass.jl: mass_solve! on a circulant operator; the deflated one of the `:project`
    # kernel has the same type, since the deflation is a stored zero factor, not a type
    opc = mass_operator(q2)
    @test isempty(JET.get_reports(JET.report_opt(mass_solve!,
        (Vector{Float64}, typeof(opc), Vector{Float64}); target_modules = (SimpleSplines,))))

    # test/mass.jl: mass_solve! on each banded operator type that the banded testset builds
    banded = unique(typeof(mass_operator(SplineQuadrature(BSplineBasis(m, p, bc))))
    for p in 1:4, bc in (Free(), Dirichlet(), Neumann(), Robin(1.0, 2.0)),
    m in (UniformMesh(24, 0 .. 1), GradedMesh(24, 0 .. 1), RandomMesh(24, 0 .. 1)))
    @testset "$T" for T in banded
        @test isempty(JET.get_reports(JET.report_opt(mass_solve!,
            (Vector{Float64}, T, Vector{Float64}); target_modules = (SimpleSplines,))))
    end

    # test/mass.jl: mass_solve! into a stride-2 view, on a banded and a circulant operator
    S2 = typeof(view(zeros(4), 1:2:4))
    opb = mass_operator(SplineQuadrature(BSplineBasis(UniformMesh(34, 0 .. 1), 3, Dirichlet())))
    @test isempty(JET.get_reports(JET.report_opt(mass_solve!,
        (S2, typeof(opb), Vector{Float64}); target_modules = (SimpleSplines,))))
    @test isempty(JET.get_reports(JET.report_opt(mass_solve!,
        (S2, typeof(opc), Vector{Float64}); target_modules = (SimpleSplines,))))
else
    @test_skip "JET does not work on this Julia version"  # aviatesk/JET.jl#681
end
