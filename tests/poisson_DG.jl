using Ferrite
using LinearAlgebra
using SparseArrays
using ProtoStructs

struct SolverCache{Tfact}
    K::Tfact
    f::Vector{Float64}
    y::Vector{Float64}
    dh::DofHandler
end

# Volume integrals Ke, fe over element
function assemble_element!(Ke::Matrix{Float64}, fe::Vector{Float64}, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)
    fill!(fe, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(cv)
        detJdx = getdetJdV(cv, q_point)

        # Test
        for i = 1:n_basefuncs
            v = shape_value(cv, q_point, i)
            ∇v = shape_gradient(cv, q_point, i)
            fe[i] += v * detJdx

            # Trial
            for j = 1:n_basefuncs
                ∇y = shape_gradient(cv, q_point, j)
                Ke[i, j] += (∇v ⋅ ∇y) * detJdx
            end
        end
    end
    return Ke, fe
end

# Surface intgerals Ki over interface
function assemble_interface!(Ki::Matrix{Float64}, iv::InterfaceValues, μ::Float64)
    fill!(Ki, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(iv)
        ν = getnormal(iv, q_point)
        detJds = getdetJdV(iv, q_point)
        
        # Test
        for i = 1:getnbasefunctions(iv)
            v_jump = shape_value_jump(iv, q_point, i) * (-ν)
            ∇v_avg = shape_gradient_average(iv, q_point, i)
            
            # Trial
            for  j = 1:getnbasefunctions(iv) 
                y_jump = shape_value_jump(iv, q_point, j) * (-ν)
                ∇y_avg = shape_gradient_average(iv, q_point, j)

                Ki[i, j] += -(v_jump ⋅ ∇y_avg + ∇v_avg ⋅ y_jump) * detJds + μ * (v_jump ⋅ y_jump) * detJds
            end
        end
    end
    return Ki
end

# Surface integrals fe over boundary facet
function assemble_boundary!(fe::Vector{Float64}, fv::FacetValues)
    fill!(fe, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(fv)
        ν = getnormal(fv, q_point)
        detJds = getdetJdV(fv, q_point)

        # Test
        for i = 1:getnbasefunctions(fv)
            v = shape_value(fv, q_point, i)
            fe[i] += ν[2] * v * detJds
        end 
    end
    return fe
end

function assemble_global(cv::CellValues, facetvalues::FacetValues, interfacevalues::InterfaceValues, 
    K::SparseMatrixCSC, dh::DofHandler, ∂Ωₙ::AbstractSet{FacetIndex}, order::Int, dim::Int)

    # Allocate element
    n_basefuncs = getnbasefunctions(cv)
    Me = zeros(n_basefuncs, n_basefuncs)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    Ki = zeros(n_basefuncs * 2, n_basefuncs * 2)
    
    # Allocate global 
    f = zeros(ndofs(dh))

    assembler = start_assemble(K, f)
    # Assemble veolume terms
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        assemble_element!(Ke, fe, cv)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end

    # Assemble interface terms
    for ic = InterfaceIterator(dh)
        reinit!(interfacevalues, ic)
        interfacecoords = ∩(getcoordinates(ic)...)
        hₑ = getdiameter(interfacecoords)
        μ = (1 + order)^dim / hₑ
        assemble_interface!(Ki, interfacevalues, μ)
        assemble!(assembler, interfacedofs(ic), Ki)
    end

    # Assemble boundary terms
    for fc = FacetIterator(dh, ∂Ωₙ)
        reinit!(facetvalues, fc)
        assemble_boundary!(fe, facetvalues)
        assemble!(f, celldofs(fc), fe)
    end
    return K, f
end

function sol!(cache)
    ldiv!(cache.y, cache.K, cache.f)
end

function write_vtk(FilePath::String, cache)
    VTKGridFile(FilePath, cache.dh) do vtk
        write_solution(vtk, cache.dh, cache.y)
    end
end

getdistance(p1::Vec{N, T}, p2::Vec{N, T}) where {N, T} = norm(p1 - p2);
getdiameter(cell_coords::Vector{Vec{N, T}}) where {N, T} = maximum(getdistance.(cell_coords, reshape(cell_coords, (1, :))));

function setup_cache()
    dim = 2
    grid = generate_grid(Quadrilateral, (20, 20))
    
    topology = ExclusiveTopology(grid)

    order = 1
    ip = DiscontinuousLagrange{RefQuadrilateral, order}()
    qr = QuadratureRule{RefQuadrilateral}(2*order)
    facet_qr = FacetQuadratureRule{RefQuadrilateral}(2*order)
    
    cv = CellValues(qr, ip)
    facetvalues = FacetValues(facet_qr, ip)
    interfacevalues = InterfaceValues(facet_qr, ip)
    
    dh = DofHandler(grid)
    add!(dh, :y, ip)
    close!(dh)

    K = allocate_matrix(dh, topology=topology, interface_coupling=trues(1,1))

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:y, getfacetset(grid, "right"), (x,t) -> 1.0))
    add!(ch, Dirichlet(:y, getfacetset(grid, "left"), (x,t) -> -1.0))
    close!(ch)

    ∂Ωₙ = union(getfacetset(grid, "top"), getfacetset(grid, "bottom"))

    K, f = assemble_global(cv, facetvalues, interfacevalues, K, dh, ∂Ωₙ, order, dim);
    apply!(K, f, ch)

    K_fact = lu(K)

    y = zeros(ndofs(dh))

    cache = SolverCache(K_fact, f, y, dh)
    return cache
end

cache = setup_cache()
sol!(cache)
write_vtk("tests/poisson_DG_test", cache)