using Ferrite
using LinearAlgebra
using SparseArrays
#using GLMakie
using ProtoStructs

#=
-∇⋅(α(y)∇y) = βu    on Ω
∂y/∂n = 0           on Γ = ∂Ω
u ∈ [u⁻,u⁺]         on Ω

α - Thermal diffusivity
β - Actuator shape function
=#

@proto struct SolverCache{Tfact}
    K::Tfact
    f::Vector{Float64}
    y::Vector{Float64}
    dh::DofHandler
end

function assemble_element!(Ke::Matrix{Float64}, fe::Vector{Float64}, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    fill!(Ke, 0)
    fill!(fe, 0)

    # Quadrature
    for q_point in 1:getnquadpoints(cellvalues)
        Jdx = getdetJdV(cellvalues, q_point)

        # Test
        for i in 1:n_basefuncs
            v = shape_value(cellvalues, q_point, i)
            ∇v = shape_gradient(cellvalues, q_point, i)
            fe[i] += v * Jdx

            # Trial
            for j in 1:n_basefuncs
                ∇y = shape_gradient(cellvalues, q_point, j)
                Ke[i, j] += (∇v ⋅ ∇y) * Jdx
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues::CellValues, K::SparseMatrixCSC, f::Vector{Float64}, dh::DofHandler)
    # Allocate local
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    # Assemble
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        assemble_element!(Ke, fe, cellvalues)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

function sol!(cache)
    ldiv!(cache.u, cache.K, cache.f)
end

function write_vtk(FileName::String, cache)
    VTKGridFile(FileName, cache.dh) do vtk
        write_solution(vtk, cache.dh, cache.u)
    end
end

function setup_cache()
    grid = generate_grid(Quadrilateral, (20, 20))

    ip = Lagrange{RefQuadrilateral,1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :y, ip)
    close!(dh)

    K = allocate_matrix(dh)
    f = zeros(ndofs(dh));

    ch = ConstraintHandler(dh)
    ∂Ω = union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right"),
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
    )
    dbc = Dirichlet(:y, ∂Ω, (x, t) -> 0)
    add!(ch, dbc)
    close!(ch)

    K, f = assemble_global(cellvalues, K, f, dh);
    apply!(K, f, ch)
    K_fact = lu(K)

    y = zeros(ndofs(dh))

    cache = SolverCache(K_fact, f, y, dh)

    return cache
end

cache = setup_cache()
sol!(cache)
write_vtk("poisson_test", cache)

