using SimpleSplines
using Test

@testset "$(rpad("Boundary Condition Tests",80))" begin
    @testset "$(rpad("coefficients and order",76))" begin
        @test constraint_coefficients(Free()) === nothing
        @test constraint_coefficients(Periodic()) === nothing
        @test constraint_coefficients(Dirichlet()) == (1,)
        @test constraint_coefficients(Neumann()) == (0, 1)
        @test constraint_coefficients(Natural()) == (0, 0, 1)
        @test constraint_coefficients(Robin(2.0, 3.0)) == (2.0, 3.0)
        @test constraint_coefficients(Constraint(1, 0, -2)) == (1, 0, -2)

        @test nconstraints(Free()) == 0
        @test nconstraints(Periodic()) == 0
        @test nconstraints(Dirichlet()) == 1
        @test nconstraints(Neumann()) == 1
        @test nconstraints(Natural()) == 1
        @test nconstraints(Robin(1.0, 1.0)) == 1

        # the order is the highest derivative that actually appears, so a Robin condition
        # with a zero derivative coefficient is a Dirichlet condition and costs the same
        @test constraint_order(Free()) == -1
        @test constraint_order(Periodic()) == -1
        @test constraint_order(Dirichlet()) == 0
        @test constraint_order(Neumann()) == 1
        @test constraint_order(Natural()) == 2
        @test constraint_order(Robin(1.0, 2.0)) == 1
        @test constraint_order(Robin(1.0, 0.0)) == 0
        @test constraint_order(Constraint(0, 0, 0, 1)) == 3
    end

    @testset "$(rpad("construction errors",76))" begin
        @test_throws ArgumentError Robin(0, 0)
        @test_throws ArgumentError Constraint(0, 0)
        @test_throws ArgumentError Constraint(0.0)
    end

    @testset "$(rpad("specification",76))" begin
        @test boundary_conditions(Dirichlet()) == (Dirichlet(), Dirichlet())
        @test boundary_conditions(Free()) == (Free(), Free())
        @test boundary_conditions(Periodic()) === Periodic()
        @test boundary_conditions((Dirichlet(), Neumann())) == (Dirichlet(), Neumann())

        # symbol sugar
        @test boundary_conditions(:dirichlet) == (Dirichlet(), Dirichlet())
        @test boundary_conditions(:periodic) === Periodic()
        @test boundary_conditions((:dirichlet, :neumann)) == (Dirichlet(), Neumann())
        @test boundary_conditions((:free, Natural())) == (Free(), Natural())

        # Periodic identifies the two ends, so it is not a per-end condition
        @test_throws ArgumentError boundary_conditions((Periodic(), Dirichlet()))
        @test_throws ArgumentError boundary_conditions((:periodic, :periodic))
    end

    @testset "$(rpad("renamed and unknown symbols fail loudly",76))" begin
        # :Natural used to mean the *unconstrained* clamped basis; :natural now means u''=0.
        # Accepting the capitalised form would silently hand back a different function space
        # than the caller's old code had, so it is rejected with the rename spelled out.
        e = try
            BoundaryCondition(:Natural)
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("Free()", e.msg)

        @test_throws ArgumentError BoundaryCondition(:nothing)
        @test_throws ArgumentError BoundaryCondition(:Dirichlet)
        @test_throws ArgumentError BoundaryCondition(:Periodic)
        @test_throws ArgumentError BoundaryCondition(:bogus)

        # the error names the accepted set
        e2 = try
            BoundaryCondition(:bogus)
        catch err
            err
        end
        @test occursin(":dirichlet", e2.msg)
    end

    @testset "$(rpad("which basis a condition selects",76))" begin
        m = UniformMesh(8, 0 .. 1)
        @test BSplineBasis(m, 3) isa BSplineBasis
        @test BSplineBasis(m, 3, Free()) isa BSplineBasis
        @test BSplineBasis(m, 3, (Free(), Free())) isa BSplineBasis
        @test BSplineBasis(m, 3, Periodic()) isa PeriodicBSplineBasis
        @test BSplineBasis(m, 3, :periodic) isa PeriodicBSplineBasis
        @test BSplineBasis(m, 3, Dirichlet()) isa RecombinedBSplineBasis
        @test BSplineBasis(m, 3, (Dirichlet(), Free())) isa RecombinedBSplineBasis
        @test BSplineBasis(m, 3, :dirichlet) isa RecombinedBSplineBasis

        @test boundary(BSplineBasis(m, 3)) == (Free(), Free())
        @test boundary(BSplineBasis(m, 3, Periodic())) === Periodic()
        @test boundary(BSplineBasis(m, 3, (Dirichlet(), Neumann()))) ==
              (Dirichlet(), Neumann())
    end

    @testset "$(rpad("rejected combinations",76))" begin
        # a condition of order above the degree constrains nothing
        @test_throws ArgumentError BSplineBasis(UniformMesh(8, 0 .. 1), 1, Natural())
        @test_throws ArgumentError BSplineBasis(UniformMesh(8, 0 .. 1), 0, Neumann())
        # too few functions for both ends
        @test_throws ArgumentError BSplineBasis(UniformMesh(1, 0 .. 1), 0, Dirichlet())
    end

    @testset "$(rpad("show",76))" begin
        @test string(Free()) == "Free()"
        @test string(Dirichlet()) == "Dirichlet()"
        @test string(Robin(1.0, 2.0)) == "Robin(1.0, 2.0)"
        @test string(Constraint(1, 0, 2)) == "Constraint(1, 0, 2)"
    end
end
