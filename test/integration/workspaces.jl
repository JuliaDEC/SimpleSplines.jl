using ForwardDiff: Dual
using LinearAlgebra
using Random
using SimpleSplines
using Test

Random.seed!(1234)

# Fixed-arity barriers: the arguments arrive with concrete types, and the call runs once
# before it is measured.
allocs(f::F, a::A, b::B, c::C) where {F, A, B, C} = (f(a, b, c); @allocated f(a, b, c))
function allocs(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D}
    f(a, b, c, d)
    @allocated f(a, b, c, d)
end
function allocs(f::F, a::A, b::B, c::C, d::D, e::E) where {F, A, B, C, D, E}
    f(a, b, c, d, e)
    @allocated f(a, b, c, d, e)
end

# Coverage instrumentation allocates on every line, so the readings mean nothing under it.
const MEASURE = Base.JLOptions().code_coverage == 0

# The outcome of a call: its value, or the type of the exception it raised.
outcome(f) =
    try
        f()
    catch e
        typeof(e)
    end

# The message of the exception that `f` raises.
message(f) =
    try
        f()
        ""
    catch e
        sprint(showerror, e)
    end

const Dual12 = Dual{Nothing, Float64, 12}
dual(x) = Dual{Nothing}(Float64(x), ntuple(i -> Float64(i) / 7, 12)...)

function bases_1d(::Type{T}) where {T}
    m = UniformMesh(8, T(0) .. T(1))
    (BSplineBasis(m, 1), BSplineBasis(m, 3), BSplineBasis(m, 3, Periodic()),
        BSplineBasis(m, 3, Dirichlet()), BSplineBasis(m, 3, Natural()))
end

function basis_2d(::Type{T}, n = 8) where {T}
    BSplineBasis(UniformMesh(n, T(0) .. T(1)), 3, Periodic()) ⊗
    BSplineBasis(UniformMesh(n, T(-1) .. T(2)), 3)
end

function basis_3d(::Type{T}, n = 4) where {T}
    BSplineBasis(UniformMesh(n, T(0) .. T(1)), 3) ⊗
    BSplineBasis(UniformMesh(n, T(0) .. T(2)), 3, Periodic()) ⊗
    BSplineBasis(UniformMesh(n, T(-1) .. T(1)), 3, Dirichlet())
end

# The same basis on more cells: a workspace built for it has longer buffers, so a missing
# size check shows as a wrong result or an error rather than as a write out of bounds.
larger(B) = ndims(B) == 2 ? basis_2d(eltype(B), 9) : basis_3d(eltype(B), 5)

@testset "$(rpad("evaluate! on a one-dimensional basis",76))" begin
    for T in (Float32, Float64), b in bases_1d(T)

        û = randn(T, nbasis(b))
        buf = zeros(T, local_width(b))
        s = Spline(b, û)
        # interior points, a breakpoint, both ends, and points outside the domain
        for x in T[0.37, 0.5, 0, 1, prevfloat(T(1)), -0.25, 1.75], d in 0:(degree(b) + 1)

            @test (@inferred evaluate!(buf, b, û, x, d)) == evaluate(b, û, x, d)
            @test (@inferred evaluate!(buf, s, x, d)) == evaluate(s, x, d)
        end
        @test evaluate!(buf, b, û, T(0.37)) == evaluate(b, û, T(0.37))
        @test evaluate!(buf, s, T(0.37)) == s(T(0.37))

        # A Dual argument with a Dual buffer. `isequal` rather than `==`: a periodic basis at
        # the ends of its domain gives NaN partials on both sides (K6 of KNOWN_ISSUES.md), and
        # `==` is false for them.
        bufd = zeros(Dual12, local_width(b))
        for x in (0.37, 0.5, 1.0, 1.75), d in 0:1

            @test isequal((@inferred evaluate!(bufd, b, û, dual(x), d)),
                evaluate(b, û, dual(x), d))
            @test isequal((@inferred evaluate!(bufd, s, dual(x), d)), evaluate(s, dual(x), d))
        end
    end
end

@testset "$(rpad("evaluate! allocates nothing",76))" begin
    if MEASURE
        for T in (Float32, Float64), b in bases_1d(T), d in 0:1
            û = randn(T, nbasis(b))
            s = Spline(b, û)
            x = T(0.37)
            buf = zeros(T, local_width(b))
            @test allocs(evaluate!, buf, b, û, x, d) == 0
            @test allocs(evaluate!, buf, s, x, d) == 0
            @test allocs(evaluate, b, û, x, d) > 0
            @test allocs(evaluate, s, x, d) > 0

            bufd = zeros(Dual12, local_width(b))
            @test allocs(evaluate!, bufd, b, û, dual(0.37), d) == 0
            @test allocs(evaluate!, bufd, s, dual(0.37), d) == 0
            @test allocs(evaluate, b, û, dual(0.37), d) > 0
        end
    end
end

@testset "$(rpad("evaluate! refuses a buffer of the wrong length",76))" begin
    for b in bases_1d(Float64)
        û = randn(nbasis(b))
        w = local_width(b)
        for n in (w - 1, w + 1)
            @test_throws DimensionMismatch evaluate!(zeros(n), b, û, 0.37)
            msg = message(() -> evaluate!(zeros(n), b, û, 0.37))
            @test startswith(msg,
                "DimensionMismatch: evaluate! needs a buffer of $(w) entries for this " *
                "basis, got $(n)")
            @test_throws DimensionMismatch evaluate!(zeros(n), Spline(b, û), 0.37)
        end
        # the coefficient check of `evaluate`
        @test_throws DimensionMismatch evaluate!(zeros(w), b, zeros(nbasis(b) + 1), 0.37)
        @test startswith(message(() -> evaluate!(zeros(w), b, zeros(nbasis(b) + 1), 0.37)),
            "DimensionMismatch: the coefficient vector has $(nbasis(b) + 1) entries")
        # a buffer that cannot hold the value, of a narrower type or for a Dual argument
        @test_throws ArgumentError evaluate!(zeros(Float32, w), b, û, 0.37)
        @test startswith(message(() -> evaluate!(zeros(Float32, w), b, û, 0.37)),
            "ArgumentError: evaluate! needs a buffer that holds Float64, got a buffer of " *
            "Float32")
        @test_throws ArgumentError evaluate!(zeros(Float32, w), Spline(b, û), 0.37)
        @test_throws ArgumentError evaluate!(zeros(w), b, û, dual(0.37))
    end
end

@testset "$(rpad("evaluate! on hostile arguments matches evaluate",76))" begin
    for T in (Float32, Float64), b in bases_1d(T)

        û = randn(T, nbasis(b))
        buf = zeros(T, local_width(b))
        for x in T[0, -0.0, nextfloat(T(0)), floatmin(T), eps(T), 1, floatmax(T) / 2,
                floatmax(T), Inf, -Inf, NaN, 1e-8, 1e-4, 1e4, 1e8],
            d in (0, 1, 4, -1)

            @test isequal(outcome(() -> evaluate!(buf, b, û, x, d)),
                outcome(() -> evaluate(b, û, x, d)))
        end
        # `d < 0` is the `ArgumentError` of `evaluate_all!`
        @test_throws ArgumentError evaluate!(buf, b, û, T(0.5), -1)
    end
    # BigFloat: the values agree
    for b in bases_1d(BigFloat)
        û = randn(nbasis(b))
        buf = zeros(BigFloat, local_width(b))
        for x in big.([0.37, 1.0, 1.5]), d in 0:2

            @test evaluate!(buf, b, û, x, d) == evaluate(b, û, x, d)
        end
    end
    # degree 0 on one cell
    for b in (BSplineBasis(UniformMesh(1, 0 .. 1), 0),
        BSplineBasis(UniformMesh(1, 0 .. 1), 0, Periodic()))
        û = [2.5]
        for x in (0.0, 0.5, 1.0, 1.5, -0.5), d in 0:1

            @test evaluate!(zeros(1), b, û, x, d) == evaluate(b, û, x, d)
        end
    end
end

@testset "$(rpad("evaluate! on a tensor-product basis",76))" begin
    for T in (Float32, Float64), B in (basis_2d(T), basis_3d(T))

        D = ndims(B)
        û = randn(T, size(B)...)
        bufs = map(w -> zeros(T, w), local_width(B))
        bufsd = map(w -> zeros(Dual12, w), local_width(B))
        points = D == 2 ? (T.((0.37, 0.4)), T.((0.0, 2.0)), T.((1.5, 3.0))) :
                 (T.((0.37, 1.2, -0.3)), T.((1.0, 0.0, 1.0)), T.((-0.5, 2.5, 0.2)))
        orders = (ntuple(_ -> 0, D), ntuple(k -> k == 1 ? 1 : 0, D),
            ntuple(k -> k == D ? 1 : 0, D))
        for x in points, d in orders

            @test (@inferred evaluate!(bufs, B, û, x, d)) == evaluate(B, û, x, d)
            xd = map(dual, x)
            @test isequal((@inferred evaluate!(bufsd, B, û, xd, d)), evaluate(B, û, xd, d))
        end
        @test evaluate!(bufs, B, û, first(points)) == evaluate(B, û, first(points))

        if MEASURE
            x = first(points)
            d = ntuple(k -> k == 1 ? 1 : 0, D)
            @test allocs(evaluate!, bufs, B, û, x, d) == 0
            @test allocs(evaluate, B, û, x, d) > 0
            @test allocs(evaluate!, bufsd, B, û, map(dual, x), d) == 0
            @test allocs(evaluate, B, û, map(dual, x), d) > 0
        end

        # a wrong length on any one axis
        for k in 1:D
            w = local_width(B)
            bad = ntuple(i -> zeros(T, i == k ? w[i] + 1 : w[i]), D)
            @test_throws DimensionMismatch evaluate!(bad, B, û, first(points))
            @test startswith(message(() -> evaluate!(bad, B, û, first(points))),
                "DimensionMismatch: evaluate! needs buffers of $(w) entries on the axes, " *
                "got $(map(length, bad))")
        end
        # a coefficient array of the wrong size, as `evaluate` refuses it
        ûbad = zeros(T, (size(B) .+ 1)...)
        @test_throws DimensionMismatch evaluate!(bufs, B, ûbad, first(points))
        @test startswith(message(() -> evaluate!(bufs, B, ûbad, first(points))),
            "DimensionMismatch: the coefficient array is $(size(ûbad)) but the basis is " *
            "$(size(B))")
        # buffers that cannot hold the value, for Dual arguments or of a narrower type
        @test_throws ArgumentError evaluate!(bufs, B, û, map(dual, first(points)))
        if T === Float64
            narrow = map(w -> zeros(Float32, w), local_width(B))
            @test_throws ArgumentError evaluate!(narrow, B, û, first(points))
            @test startswith(message(() -> evaluate!(narrow, B, û, first(points))),
                "ArgumentError: evaluate! needs buffers that hold Float64, got buffers of " *
                "Float32")
        end
    end
end

@testset "$(rpad("mass_solve! with a workspace",76))" begin
    for T in (Float32, Float64), B in (basis_2d(T), basis_3d(T))

        op = mass_operator(TensorProductQuadrature(B))
        x = randn(T, size(B)...)
        work = mass_workspace(op, T)
        y = similar(x)
        @test (@inferred mass_solve!(y, op, x, work)) === y
        @test y == mass_solve!(similar(x), op, x)
        # an independent reference: the assembled Kronecker product
        @test vec(y) ≈ Matrix(op) \ vec(x) rtol = 100 * eps(T) * cond(Matrix(op))
        # y and x may alias, and a vector of length `length(B)` is accepted
        z = copy(x)
        @test mass_solve!(z, op, z, work) == y
        v = vec(copy(x))
        @test mass_solve!(v, op, v, work) == vec(y)

        if MEASURE
            @test allocs(mass_solve!, y, op, x, work) == 0
            @test allocs(mass_solve!, y, op, x) > 0
        end

        # a workspace of the wrong element type names both types
        S = T === Float64 ? Float32 : Float64
        bad = mass_workspace(op, S)
        @test_throws ArgumentError mass_solve!(y, op, x, bad)
        @test startswith(message(() -> mass_solve!(y, op, x, bad)),
            "ArgumentError: the workspace has element type $(S) but the array has element " *
            "type $(T)")
        # a workspace of a larger operator names both sizes
        other = mass_operator(TensorProductQuadrature(larger(B)))
        badsize = mass_workspace(other, T)
        @test_throws ArgumentError mass_solve!(y, op, x, badsize)
        @test startswith(message(() -> mass_solve!(y, op, x, badsize)),
            "ArgumentError: the workspace has buffers of lengths " *
            "$(map(n -> size(n, 1), mass_factors(other))) and " *
            "$(map(n -> size(n, 1), mass_factors(other))) but the operator needs " *
            "$(map(n -> size(n, 1), mass_factors(op)))")
        # a workspace of another dimension
        # (of 19 functions per axis, longer than every axis of `B`)
        b19 = BSplineBasis(UniformMesh(16, T(0) .. T(1)), 3)
        flat = mass_workspace(mass_operator(TensorProductQuadrature(b19 ⊗ b19 ⊗ b19 ⊗ b19)), T)
        @test_throws ArgumentError mass_solve!(y, op, x, flat)
        # a long `src` buffer, a long `dst` buffer, and both too long by different amounts,
        # which the message names each at its own place
        dims = map(n -> size(n, 1), mass_factors(op))
        grow(vs, m) = map(v -> [v; zeros(T, m)], vs)
        longsrc = typeof(work)(grow(work.src, 1), work.dst)
        @test_throws ArgumentError mass_solve!(y, op, x, longsrc)
        @test startswith(message(() -> mass_solve!(y, op, x, longsrc)),
            "ArgumentError: the workspace has buffers of lengths $(dims .+ 1) and $(dims) " *
            "but the operator needs $(dims)")
        longdst = typeof(work)(work.src, grow(work.dst, 1))
        @test_throws ArgumentError mass_solve!(y, op, x, longdst)
        @test startswith(message(() -> mass_solve!(y, op, x, longdst)),
            "ArgumentError: the workspace has buffers of lengths $(dims) and $(dims .+ 1) " *
            "but the operator needs $(dims)")
        longboth = typeof(work)(grow(work.src, 2), grow(work.dst, 1))
        @test startswith(message(() -> mass_solve!(y, op, x, longboth)),
            "ArgumentError: the workspace has buffers of lengths $(dims .+ 2) and " *
            "$(dims .+ 1) but the operator needs $(dims)")
        # an array of the wrong length
        long = zeros(T, length(x) + 1)
        @test_throws DimensionMismatch mass_solve!(long, op, vec(x), work)
        @test startswith(message(() -> mass_solve!(long, op, vec(x), work)),
            "DimensionMismatch: the array has $(length(x) + 1) entries but the operator " *
            "acts on $(size(x))")
        # an argument shorter or longer than `y`, through both forms
        for n in (length(x) - 5, length(x) + 1)
            xbad = ones(T, n)
            @test_throws DimensionMismatch mass_solve!(y, op, xbad, work)
            @test_throws DimensionMismatch mass_solve!(y, op, xbad)
            @test startswith(message(() -> mass_solve!(y, op, xbad, work)),
                "DimensionMismatch: the argument has $(n) entries but the operator acts on " *
                "$(size(x))")
        end
    end
end

# The load array by its definition, from the dense per-axis tables.
function reference_contract(q, F, d)
    A = [Matrix(basis_values(quadratures(q)[k], d[k])) *
         Diagonal(quadrature_weights(quadratures(q)[k])) for k in 1:ndims(q)]
    reshape(foldl(kron, reverse(A)) * vec(F), size(q)...)
end

@testset "$(rpad("contract! and l2_projection! with a workspace",76))" begin
    for T in (Float32, Float64), B in (basis_2d(T), basis_3d(T))

        D = ndims(B)
        q = TensorProductQuadrature(B)
        F = randn(T, quadrature_grid_size(q)...)
        F₀ = copy(F)
        work = contract_workspace(q, T)
        L = zeros(T, size(B)...)
        for d in (ntuple(_ -> 0, D), ntuple(k -> k == D ? 1 : 0, D))
            @test (@inferred contract!(L, q, F, work, d)) === L
            @test L == contract(q, F, d)
            @test L ≈ reference_contract(q, F, d) rtol = 100 * eps(T)
        end
        @test contract!(L, q, F, work) == contract(q, F)
        @test F == F₀                                   # the sample is not consumed

        û = zeros(T, size(B)...)
        @test (@inferred l2_projection!(û, q, F, work)) === û
        @test û == l2_projection!(zeros(T, size(B)...), q, F)
        @test û == l2_projection(q, F)
        @test F == F₀

        if MEASURE
            d = ntuple(k -> k == D ? 1 : 0, D)
            @test allocs(contract!, L, q, F, work, d) == 0
            @test allocs(contract, q, F, d) > 0
            @test allocs(l2_projection!, û, q, F, work) == 0
            @test allocs(l2_projection!, û, q, F) > 0
        end

        # wrong sizes, as the allocating forms raise them
        Fbad = zeros(T, (quadrature_grid_size(q) .+ 1)...)
        @test_throws DimensionMismatch contract!(L, q, Fbad, work)
        @test message(() -> contract!(L, q, Fbad, work)) ==
              message(() -> contract(q, Fbad))
        Lbad = zeros(T, (size(B) .+ 1)...)
        @test_throws DimensionMismatch contract!(Lbad, q, F, work)
        @test startswith(message(() -> contract!(Lbad, q, F, work)),
            "DimensionMismatch: the load array is $(size(Lbad)) but the basis is $(size(B))")
        @test_throws DimensionMismatch l2_projection!(Lbad, q, F, work)
        @test startswith(message(() -> l2_projection!(Lbad, q, F, work)),
            "DimensionMismatch: the coefficient array is $(size(Lbad)) but the basis is " *
            "$(size(B))")
        # the three-argument form checks `û` before it samples `f`
        calls = Ref(0)
        counted = x -> (calls[] += 1; one(T))
        @test_throws DimensionMismatch l2_projection!(Lbad, q, counted)
        @test calls[] == 0
        @test startswith(message(() -> l2_projection!(Lbad, q, counted)),
            "DimensionMismatch: the coefficient array is $(size(Lbad)) but the basis is " *
            "$(size(B))")

        # a workspace of the wrong element type names both types
        S = T === Float64 ? Float32 : Float64
        bad = contract_workspace(q, S)
        @test_throws ArgumentError contract!(L, q, F, bad)
        @test startswith(message(() -> contract!(L, q, F, bad)),
            "ArgumentError: the workspace has element type $(S) but the load array has " *
            "element type $(T)")
        @test_throws ArgumentError l2_projection!(û, q, F, bad)
        # a workspace of a larger quadrature names both lengths
        badsize = contract_workspace(TensorProductQuadrature(larger(B)), T)
        @test_throws ArgumentError contract!(L, q, F, badsize)
        @test startswith(message(() -> contract!(L, q, F, badsize)),
            "ArgumentError: the workspace holds buffers of $(length(badsize.a)) and " *
            "$(length(badsize.b)) entries but this quadrature needs $(length(work.a))")
        @test length(badsize.a) != length(work.a)
        @test_throws ArgumentError l2_projection!(û, q, F, badsize)
        # a long first buffer, a long second buffer, and both too long by different
        # amounts, which the message names each at its own place
        n = length(work.a)
        longa = SimpleSplines.ContractWorkspace([work.a; zero(T)], work.b, work.mass)
        @test_throws ArgumentError contract!(L, q, F, longa)
        @test startswith(message(() -> contract!(L, q, F, longa)),
            "ArgumentError: the workspace holds buffers of $(n + 1) and $(n) entries but " *
            "this quadrature needs $(n)")
        longb = SimpleSplines.ContractWorkspace(work.a, [work.b; zero(T)], work.mass)
        @test_throws ArgumentError contract!(L, q, F, longb)
        @test startswith(message(() -> contract!(L, q, F, longb)),
            "ArgumentError: the workspace holds buffers of $(n) and $(n + 1) entries but " *
            "this quadrature needs $(n)")
        longab = SimpleSplines.ContractWorkspace([work.a; zeros(T, 2)], [work.b; zero(T)],
            work.mass)
        @test startswith(message(() -> contract!(L, q, F, longab)),
            "ArgumentError: the workspace holds buffers of $(n + 2) and $(n + 1) entries " *
            "but this quadrature needs $(n)")
        # a mass workspace of a larger operator
        wrongmass = SimpleSplines.ContractWorkspace(work.a, work.b, badsize.mass)
        @test_throws ArgumentError l2_projection!(û, q, F, wrongmass)
        @test startswith(message(() -> l2_projection!(û, q, F, wrongmass)),
            "ArgumentError: the workspace has buffers of lengths")
        # a workspace without a mass workspace, as `contract` builds it, leaves `û` alone
        nomass = SimpleSplines.ContractWorkspace(work.a, work.b, nothing)
        û₀ = copy(û)
        @test_throws MethodError l2_projection!(û, q, F, nomass)
        @test û == û₀
    end
end
