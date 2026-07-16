module AssembleDG

using Ferrite
using LinearAlgebra
import ..assemble_mass!   # shared generic owned by PDEOpt; this adds the element-local method

export assemble_mass!, assemble_diffusion!, assemble_interface!, assemble_advection!,
    assemble_advection_interface!, assemble_advection_boundary!, assemble_inflow!,
    assemble_reaction!, interface_field_dofs, getdistance, getdiameter

# Mass martix over element
function assemble_mass!(Me::Matrix{Float64}, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Me, 0)

    for q_point in 1:getnquadpoints(cv)
        detJdx = getdetJdV(cv, q_point)

        #Test
        for i = 1:n_basefuncs
            v = shape_value(cv, q_point, i)

            # Trial
            for j = 1:n_basefuncs
                y = shape_value(cv, q_point, j)
                Me[i, j] += (y * v) * detJdx
            end
        end
    end
    return Me
end

# Diffusion stiffness Ke over element (linear)
function assemble_diffusion!(Ke::Matrix{Float64}, cv::CellValues, D::SymmetricTensor{2,2,Float64,3})
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(cv)
        detJdx = getdetJdV(cv, q_point)

        # Test
        for i = 1:n_basefuncs
            ∇v = shape_gradient(cv, q_point, i)

            # Trial
            for j = 1:n_basefuncs
                ∇y = shape_gradient(cv, q_point, j)
                Ke[i, j] += (∇v ⋅ (D ⋅ ∇y)) * detJdx
            end
        end
    end
    return Ke
end

# Same as assemble_diffusion! but nonlinear
function assemble_diffusion_jacobian!(Ke::Matrix{Float64}, de::Vector{Float64}, cv::CellValues)
end

# Surface intgerals Ki over interface (linear)
function assemble_interface!(Ki::Matrix{Float64}, iv::InterfaceValues, μ::Float64, 
    D::SymmetricTensor{2,2,Float64,3})
    fill!(Ki, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(iv)
        ν = getnormal(iv, q_point)
        detJds = getdetJdV(iv, q_point)
        δ = ν ⋅ (D ⋅ ν) # δ=D_ax for ν=(1,0) (axial) ---- δ=D_rad for ν=(0,1) (radial)

        # Test
        for i = 1:getnbasefunctions(iv)
            v_jump = shape_value_jump(iv, q_point, i) * (-ν)
            ∇v_avg = shape_gradient_average(iv, q_point, i)

            # Trial
            for  j = 1:getnbasefunctions(iv)
                y_jump = shape_value_jump(iv, q_point, j) * (-ν)
                ∇y_avg = shape_gradient_average(iv, q_point, j)

                Ki[i, j] += (-(v_jump ⋅ (D ⋅ ∇y_avg) + (D ⋅ ∇v_avg) ⋅ y_jump) * detJds # use D ⋅ for single contraction 
                            + δ * μ * (v_jump ⋅ y_jump) * detJds)
            end
        end
    end
    return Ki
end

# Same as assemble_interface! but nonlinear
function assemble_interface_jacobian!()
end

# Volume advection Ce over element -∫ (β y)·∇v (conservative)
function assemble_advection!(Ce::Matrix{Float64}, cv::CellValues, β::Vec)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ce, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(cv)
        detJdx = getdetJdV(cv, q_point)

        # Test
        for i = 1:n_basefuncs
            ∇v = shape_gradient(cv, q_point, i)

            # Trial
            for j = 1:n_basefuncs
                y = shape_value(cv, q_point, j)
                Ce[i, j] += -(β ⋅ ∇v) * y * detJdx
            end
        end
    end
    return Ce
end

# Upwind interface flux Ci over interface ∫ (β·ν) ŷ ⟦v⟧, ŷ = upwind trace (here if β·ν ≥ 0)
function assemble_advection_interface!(Ci::Matrix{Float64}, iv::InterfaceValues, β::Vec)
    fill!(Ci, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(iv)
        ν = getnormal(iv, q_point) 
        βν = β ⋅ ν
        detJds = getdetJdV(iv, q_point)
        up = (βν ≥ 0)

        # Test
        for i = 1:getnbasefunctions(iv)
            v_jump = (shape_value(iv, q_point, i; here=true) 
                    - shape_value(iv, q_point, i; here=false))

            # Trial
            for j = 1:getnbasefunctions(iv)
                ŷ = shape_value(iv, q_point, j; here=up)
                Ci[i, j] += βν * ŷ * v_jump * detJds
            end
        end
    end
    return Ci
end

# Advective boundary flux Cb over a boundary facet.
# Outflow (β·ν ≥ 0): interior trace ∫ (β·ν) y v ds  -> matrix.
# Inflow  (β·ν < 0): skipped; the inlet value enters the load via assemble_inflow! (RHS).
function assemble_advection_boundary!(Cb::Matrix{Float64}, fv::FacetValues, β::Vec)
    n_basefuncs = getnbasefunctions(fv)
    fill!(Cb, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(fv)
        ν = getnormal(fv, q_point)
        βν = β ⋅ ν
        βν ≥ 0 || continue # only outflow contributes to the matrix
        detJds = getdetJdV(fv, q_point)

        # Test
        for i = 1:n_basefuncs
            v = shape_value(fv, q_point, i)

            # Trial
            for j = 1:n_basefuncs
                y = shape_value(fv, q_point, j)
                Cb[i, j] += βν * y * v * detJds
            end
        end
    end
    return Cb
end

# Inflow advective load fe over a boundary facet (RHS contribution, weak inlet BC).
# Inflow  (β·ν < 0): exterior trace is the known inlet g(x); the incoming advective
#   flux enters as a source  -∫ (β·ν) g v ds  (positive injection, since β·ν < 0).
# Outflow (β·ν ≥ 0): skipped; handled by assemble_advection_boundary! (matrix).
# `g(x)` is the spatial inlet profile; a separable time factor τ(t) is applied by the
# driver when the load is used, so the inlet value is g(x)·τ(t).
function assemble_inflow!(fe::Vector{Float64}, fv::FacetValues, β::Vec, g,
    coords::AbstractVector{<:Vec})
    n_basefuncs = getnbasefunctions(fv)
    fill!(fe, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(fv)
        ν = getnormal(fv, q_point)
        βν = β ⋅ ν
        βν < 0 || continue # only inflow contributes to the load
        detJds = getdetJdV(fv, q_point)
        g_q = g(spatial_coordinate(fv, q_point, coords))

        # Test
        for i = 1:n_basefuncs
            v = shape_value(fv, q_point, i)
            fe[i] += -βν * g_q * v * detJds
        end
    end
    return fe
end

# Reaction residual re, tangent Jre over element (state-dependent, nonlinear)
function assemble_reaction!(Jre::Matrix{Float64}, re::Vector{Float64}, cv::CellValues,
    ye::AbstractVector{Float64}, r, r′)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Jre, 0)
    fill!(re, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(cv)
        detJdx = getdetJdV(cv, q_point)
        y_q = function_value(cv, q_point, ye) 
        rq, r′q = r(y_q), r′(y_q)

        # Test
        for i = 1:n_basefuncs
            v = shape_value(cv, q_point, i)
            re[i] += rq * v * detJdx

            # Trial
            for j = 1:n_basefuncs
                y = shape_value(cv, q_point, j)
                Jre[i, j] += r′q * y * v * detJdx
            end
        end
    end
    return Jre, re
end

# Coupled pointwise reaction for two fields (C, T) over an element. Sources
# s(C,T) = (sC, sT) and Jacobian ds(C,T) = (∂sC/∂C, ∂sC/∂T, ∂sT/∂C, ∂sT/∂T).
# Element block order is [C(1:nbf), T(nbf+1:2nbf)] (matches a 2-field cell's dofs).
function assemble_reaction!(Ke::Matrix{Float64}, re::Vector{Float64}, cv::CellValues,
    Ce::AbstractVector{Float64}, Te::AbstractVector{Float64}, s, ds)
    nbf = getnbasefunctions(cv)
    fill!(Ke, 0)
    fill!(re, 0)

    # Quadrature
    for q_point = 1:getnquadpoints(cv)
        detJdx = getdetJdV(cv, q_point)
        C_q = function_value(cv, q_point, Ce)
        T_q = function_value(cv, q_point, Te)
        sC, sT = s(C_q, T_q)
        ∂CC, ∂CT, ∂TC, ∂TT = ds(C_q, T_q)

        # Test
        for i = 1:nbf
            v = shape_value(cv, q_point, i)
            re[i] += sC * v * detJdx
            re[nbf+i] += sT * v * detJdx

            # Trial
            for j = 1:nbf
                φ = shape_value(cv, q_point, j)
                w = φ * v * detJdx
                Ke[i, j] += ∂CC * w
                Ke[i, nbf+j] += ∂CT * w
                Ke[nbf+i, j] += ∂TC * w
                Ke[nbf+i, nbf+j] += ∂TT * w
            end
        end
    end
    return Ke, re
end

getdistance(p1::Vec{N, T}, p2::Vec{N, T}) where {N, T} = norm(p1 - p2);
getdiameter(cell_coords::Vector{Vec{N, T}}) where {N, T} = maximum(getdistance.(cell_coords, reshape(cell_coords, (1, :))));

# TODO: interface_field_dofs

end
