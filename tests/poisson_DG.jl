using Ferrite
using SparseArrays

# Volume integrals Ke, fe over element
function assemble_element!(Ke::Matrix{Float64}, fe::Vector{Float64}, cellvalues::CellValues)
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

# Surface intgerals Ki over interface
function assemble_interface!(Ki::Matrix{Float64}, iv::InterfaceValues, μ::Float64)
    fill!(Ki, 0)

    # Quadrature
    for q_point in 1:getnquadpoints(fv)
        ν = getnormal(iv, q_point)
        Jds = getdetJdV(iv, q_point)
        
        # Test
        for i in 1:getnbasefunctions(iv)
            v_jump = shape_value_jump(iv, q_point, i) * (-ν)
            ∇v_avg = shape_gradient_average(iv, q_point, i)
            
            # Trial
            for  j in 1:getnbasefunctions(iv) 
                y_jump = shape_value_jump(iv, q_point, j) * (-ν)
                ∇y_avg = shape_gradient_average(iv, q_point, j)

                Ki[i, j] += -(v_jump ⋅ ∇y_avg + ∇v_avg ⋅ y_jump) * Jds + μ * (v_jump ⋅ y_jump) * Jds
            end
        end
    end
    return Ki
end

# Surface integrals fe over boundary facet
function assemble_boundary!(fe::Vector{Float64}, fv::FacetValues)
    fill!(fe, 0)

    # Quadrature
    for q_point in 1:getnquadpoints(fv)
        ν = getnormal(fv, q_point)
        ∂Ω
    end
end

function setup_cache()
    dim = 2
    grid = generate_grid(Quadrilateral, (20, 20))
    topology = ExclusiveTopology(grid)

    order = 1
    ip = DiscontinuousLagrange{RefQuadrilateral, oder}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    facet_qr = FacetQuadratureRule{RefQuadrilateral, oder}()
    
    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(facet_qr, ip)
    interfacevalues = InterfaceValues(facet_qr, ip)
    
    dh = DofHandler(grid)
    add!(dh, :y, ip)
    close!(dh)

    K = allocate_matrix(dh, topology=topology, interface_coupling=trues(1,1))

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:y, getfacetset(grid, "right"), (x,t) -> 1.0))
    add!(ch, Dirichlet(:y, getfacetset(grid, "left"), (x,t) -> 1.0))
    close!(ch)

    ∂Ωₙ = union(getfacetset(grid, "top"), getfacetset(grid, "bottom"))
end