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

@proto struct SolverCache{}
    y::Vector{Float64}
    u::Vector{Float64}

    y_des::Vector{Float64}
    grad::Vector{Float64}

    temp_n::Vector{Float64}
    temp_m::Vector{Float64}

    n::Int64
    m::Int64
    α::Float64
    β::Float64
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

function setup_cache()
    grid = generate_grid(Quadrilateral, (20, 20))

    ip = Lagrange{RefQuadrilateral,1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :y, ip)
    close!(dh)

    K = allocate_matrix(dh)
    display(K)

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


end
