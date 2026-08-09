module AssembleFVM

using SparseArrays
using ..StructuredMesh
using ..Models
using ..Problem
import ..assemble_mass!   

export advection_flux, diffusion_flux,
    assemble_mass!, assemble_transport!, assemble_inlet_outlet!, assemble_wall!,
    assemble_react!, assemble_boundary!, StateAssembly,
    face_flux, bnd_energy_kernel,
    vanalbada, antidiff_eps, assemble_antidiffusion!

function advection_flux(βn::Real, area::Real)
    F = βn * area
    #return F ≥ 0 ? (F, 0.0) : (0.0, F)
    return (max(F, zero(F)), min(F, zero(F)))
end

diffusion_flux(Df::Real, area::Real, dist::Real) = Df * area / dist

# Constant mass matrix assembly
function assemble_mass!(M::AbstractSparseMatrix, grid::AbstractGrid, geom::AbstractGeometry, dm::DofMap)
    for fi in eachindex(dm.fields)
        for c = 1:ncells(grid)
            cdof = dof(dm, c, fi)
            i, j = cellij(grid, c)
            M[cdof, cdof] = cellvolume(geom, i, j)
        end
    end
end

# State-dependent mass matrix C(y) assembly (Methanation.jl)
function assemble_mass!(M::AbstractSparseMatrix, props::MethanationProps,
    grid::AbstractGrid, geom::AbstractGeometry, dm::DofMap, model::AbstractModel)
    for fi in eachindex(dm.fields)
        for c = 1:ncells(grid)
            i, j = cellij(grid, c)
            cdof = dof(dm, c, fi)
            M[cdof, cdof] = cellvolume(geom, i, j) * capacity(model, props, fi, c)
        end
    end
    return M
end

# Constant transport assembly
function assemble_transport!(K::AbstractSparseMatrix, dirs::NTuple{N,Tuple{FaceSet,FaceGeometry}}, 
    dm::DofMap, model::AbstractModel) where {N}
    for fi in eachindex(dm.fields)
        transported(model, fi) || continue
        for (fs, fg) in dirs # axial/radial (FaceSet, FaceGeometry)
            for e = 1:length(fs.owner)
                o, n = fs.owner[e], fs.neighbor[e]
                go, gn = dof(dm, o, fi), dof(dm, n, fi)

                # Flux coeffs
                k = diffusion_flux(diffusivity(model, fi, fs.axis),
                    fg.area[e], fg.dist[e]) # flux = k(yₒ - yₙ)
                (co, cn) = advection_flux(velocity(model, fs.axis), fg.area[e]) # flux = cₒyₒ + cₙyₙ

                # Diffusion assembly
                K[go, go] += k
                K[go, gn] -= k
                K[gn, go] -= k
                K[gn, gn] += k

                # Advection assembly
                K[go, go] += co
                K[go, gn] += cn
                K[gn, go] -= co
                K[gn, gn] -= cn
            end
        end
    end
end

# State-dependent transport K(y) assembly (Methanation.jl)
function assemble_transport!(K::AbstractSparseMatrix, props::MethanationProps,
    dirs::NTuple{N,Tuple{FaceSet,FaceGeometry}}, dm::DofMap, model::AbstractModel) where {N}
    fill!(K.nzval, 0.0)
    for fi in eachindex(dm.fields)
        transported(model, fi) || continue
        for (fs, fg) in dirs
            for e = 1:length(fs.owner)
                o, n = fs.owner[e], fs.neighbor[e]
                go, gn = dof(dm, o, fi), dof(dm, n, fi)

                # Face coeffs
                Df = face_diffusivity(model, props, fi, fs.axis, o, n, fg.wf[e])
                k = diffusion_flux(Df, fg.area[e], fg.dist[e])
                βf = face_velocity(model, props, fi, fs.axis, o, n, fg.wf[e])
                (co, cn) = advection_flux(βf, fg.area[e])

                # Diffusion
                K[go, go] += k
                K[go, gn] -= k
                K[gn, go] -= k
                K[gn, gn] += k

                # Advection
                K[go, go] += co
                K[go, gn] += cn
                K[gn, go] -= co
                K[gn, gn] -= cn
            end
        end
    end
    return K
end

# Kernel assemble_transport! (local)
@inline function face_flux(m::AbstractModel, yo, yn, axis::Int, area, dist, wf)
    nsp = nspecies(m)
    Po = props_cell(m, yo)
    Pn = props_cell(m, yn)
    vz = m.vz
    Z = zero(Po.ρcp_flow)
    tr = ntuple(fi -> transported(m, fi), nfields_val(m))
    return ntuple(nfields_val(m)) do fi
        tr[fi] || return Z
        if axis == 1
            Df = Z
            βf = fi == nsp + 1 ? (wf * Po.ρcp_flow + (1 - wf) * Pn.ρcp_flow) * vz : vz + Z
        else
            Df = fi == nsp + 1 ? wf * Po.λeff + (1 - wf) * Pn.λeff :
                 wf * Po.Deff[fi] + (1 - wf) * Pn.Deff[fi]
            βf = Z
        end
        k = diffusion_flux(Df, area, dist)
        (co, cn) = advection_flux(βf, area)
        k * (yo[fi] - yn[fi]) + co * yo[fi] + cn * yn[fi]
    end
end

# Kernel assemble_energy_advection! (local)
@inline function bnd_energy_kernel(m::AbstractModel, yc, βn, area, rc)
    P = props_cell(m, yc)
    coeff = βn * P.ρcp_flow
    Z = zero(coeff)
    βn > 0 && return (coeff * area, Z)
    βn < 0 && return (Z, -coeff * area * inlet_value(m, :T, rc))
    return (Z, Z)
end

# State-dependent reassembly (Methanation.jl)
struct StateAssembly{TP,TProps,TDirs,TBnd}
    prob::TP
    props::TProps
    K_bc::SparseMatrixCSC{Float64,Int} # constant boundary contribs
    dirs::TDirs
    ebnd::TBnd 
end

function (sa::StateAssembly)(y::AbstractVector)
    p = sa.prob
    properties!(p.model, sa.props, y)
    assemble_mass!(p.M, sa.props, p.grid, p.geom, p.dm, p.model)
    assemble_transport!(p.K, sa.props, sa.dirs, p.dm, p.model)
    @. p.K.nzval += sa.K_bc.nzval # re-add constant boundary contribs
    assemble_energy_advection!(p.K, p.f_in, sa.props, p.dm, p.geom, p.grid,
        p.model, sa.ebnd...) 
    return sa
end

# State-dependent axial advection of T field at the inlet/outlet
# coeff is ρ·cp,gas·βn
# - outlet (coeff > 0): K[dof,dof] += ρcp_flow·βn·A
# - inlet (coeff < 0): f_in[dof]  = ρcp_flow·βn·A·g(r)
function assemble_energy_advection!(K::SparseMatrixCSC, f_in::AbstractVector,
    props::MethanationProps, dm::DofMap, geom::AbstractGeometry, grid::AbstractGrid,
    model::AbstractModel, bcs::Vararg{Tuple{BoundaryFaceSet,BoundaryFaceGeometry}})
    fi = nspecies(model) + 1
    field = dm.fields[fi]
    for (bfs, bfg) in bcs
        βn = velocity(model, bfs.axis) * bfs.side
        for k = 1:length(bfs.cells)
            c = bfs.cells[k]
            gc = dof(dm, c, fi)
            coeff = βn * props.ρcp_flow[c]
            if βn > 0
                K[gc, gc] += coeff * bfg.area[k]
            elseif βn < 0
                _, j = cellij(grid, c)
                f_in[gc] = -coeff * bfg.area[k] * inlet_value(model, field, geom.rc[j])
            end
        end
    end
    return K
end

# Weak Dirichlet or Danckwerts
# Inlet (β·n < 0): (β⋅C(0) - D_ax⋅∂_zC(0)) = β⋅C_in <=> J_ax(0)⋅A = β⋅A⋅C_in
# => f_in[dof] += -β·n·A·g
# Outlet (β·n > 0): ∂_zC(L) = 0 <=> J_ax(L)⋅A = β⋅A⋅C_out
# => K[dof,dof] += β·n·A
function assemble_inlet_outlet!(K::SparseMatrixCSC, f_in::AbstractVector,
    dm::DofMap, geom::AbstractGeometry, grid::AbstractGrid, model::AbstractModel,
    bcs::Vararg{Tuple{BoundaryFaceSet, BoundaryFaceGeometry}}; fields=dm.fields)
    for field in fields
        for (bfs, bfg) in bcs
            βn = velocity(model, bfs.axis) * bfs.side
            for k = 1:length(bfs.cells)
                A = bfg.area[k]
                c = bfs.cells[k]
                gc = dof(dm, c, field)
                if βn > 0
                    K[gc, gc] += βn * A
                elseif βn < 0
                    _, j = cellij(grid, c)
                    f_in[gc] += -βn * A * inlet_value(model, field, geom.rc[j])
                end
            end
        end
    end
end

# Flux: -λ⋅∂ᵣT(r=R)⋅A = U⋅A⋅(T - Tw)
# K[dof,dof] += U⋅A, f_wall[dof] += U⋅A
function assemble_wall!(K::AbstractSparseMatrix, f_wall::AbstractVector, dm::DofMap,
    model::AbstractModel, bfs::BoundaryFaceSet, bfg::BoundaryFaceGeometry)
    for k = 1:length(bfs.cells)
        A = bfg.area[k]
        c = bfs.cells[k]
        gc = dof(dm, c, wall_field(model))
        K[gc, gc] += wall_coeff(model) * A
        f_wall[gc] = wall_coeff(model) * A
    end
end

# Re-assemble global reaction (r, Jr) at current y
# Fill local source/jac into (re, Jre) and add into global (r, Jr)
function assemble_react!(prob::ProblemCache, y::AbstractVector, 
    ::Val{WithJac}) where {WithJac}
    re, Jre = prob.re, prob.Jre
    fill!(prob.r, 0.0)
    WithJac && fill!(nonzeros(prob.Jr), 0.0)
    for c in 1:ncells(prob.grid)
        i, j = cellij(prob.grid, c)
        vol = cellvolume(prob.geom, i, j)
        cd = celldof(prob.dm, c) # (C dof, T dof)
        yc = view(y, cd)
        reaction!(prob.model, re, yc) # local source per field
        for a in eachindex(cd)
            prob.r[cd[a]] += re[a] * vol
        end
        if WithJac
            # local finite diff jac
            reaction_jac!(prob.model, Jre, yc, prob.re0, prob.re, prob.yscr)
            for b in eachindex(cd), a in eachindex(cd)
                prob.Jr[cd[a], cd[b]] += Jre[a, b] * vol
            end
        end
    end
    return prob
end

# Van Albada limiter, averaged-slope form
# ψ(a,b) = ab(a+b)/(a² + b² + ε)
@inline vanalbada(a, b, ε) = a * b * (a + b) / (a * a + b * b + ε)

# ε for vanalbada (dependent)
antidiff_eps(yref::AbstractVector, href::Real; rel::Real=1e-3) =
    [(rel * yr / href)^2 for yr in yref]

# 2nd-order axial advection correction (face value y_f = y_o  -> y_f = y_o + ψ(a,b)·df)
# Flux F = βf·Δy·A -> add to r(y), r[owner] -= F, r[neighbor] += F
# Jr is kept as is -> inexact newton
function assemble_antidiffusion!(prob::ProblemCache, props, st::UpwindStencil,
    fs::FaceSet, fg::FaceGeometry, y::AbstractVector, εf::AbstractVector)
    dm, model = prob.dm, prob.model
    for fi in eachindex(dm.fields)
        transported(model, fi) || continue
        ε = εf[fi]
        for e in eachindex(fs.owner)
            o, n, uu = fs.owner[e], fs.neighbor[e], st.upup[e]
            go, gn, guu = dof(dm, o, fi), dof(dm, n, fi), dof(dm, uu, fi)

            # uu = o on first axial face
            a = (y[go] - y[guu]) / st.hu[e]
            b = (y[gn] - y[go]) / st.hd[e]
            Δy = vanalbada(a, b, ε) * st.df[e]

            βf = face_velocity(model, props, fi, fs.axis, o, n, fg.wf[e])
            F = βf * Δy * fg.area[e]
            prob.r[go] -= F
            prob.r[gn] += F
        end
    end
    return prob
end

# Boundary forcing vector
# b(t) = f_in·y_in(t) + f_wall⋅T_wall(t)
function assemble_boundary!(prob::ProblemCache, b::AbstractVector, t::Real)
    for field in prob.dm.fields
        fd = fielddof(prob.dm, field)
        yin = inlet_mod(prob.model, field, t) # inlet/outlet (:C, :T)
        @views @. b[fd] = prob.f_in[fd] * yin
    end
    tw = wall_mod(prob.model, t) # wall (only :T)
    @. b += prob.f_wall * tw
end

end
