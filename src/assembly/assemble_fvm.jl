module AssembleFVM

using SparseArrays
using ..StructuredMesh
using ..Models

export advection_flux, diffusion_flux,
    assemble_global, assemble_transport!, assemble_inlet_outlet!, assemble_wall!,
    assemble_react!, assemble_boundary!

function advection_flux(βn::Real, area::Real)
    F = βn * area
    return F ≥ 0 ? (F, 0.0) : (0.0, F)
end

diffusion_flux(Df::Real, area::Real, dist::Real) = Df * area / dist

# Global transport assembly: 
# Mass M, stiffness K (diffusion + advection)
function assemble_global(M::AbstractSparseMatrix, K::AbstractSparseMatrix,
    dirs::NTuple{N,Tuple{FaceSet,FaceGeometry}}, grid::StructuredGrid, geom::Geometry,
    dm::DofMap, model::AbstractModel) where {N}
    # Cell loop
    for fi in eachindex(dm.fields)
        for c = 1:ncells(grid)
            cdof = dof(dm, c, fi)
            i, j = cellij(grid, c)
            M[cdof, cdof] = cellvolume(geom, i, j)
        end
    end

    # For constant transport proceed below
    state_dependent(model) && return M, K

    # Face loop (use fieldindex here?)
    for fi in eachindex(dm.fields)
        transported(model, fi) || continue # skip cell-local fields (e.g. PSA loadings)
        for (fs, fg) in dirs # axial/radial (FaceSet, FaceGeometry)
            for e = 1:length(fs.owner)
                o, n = fs.owner[e], fs.neighbor[e] # cell indices
                go, gn = dof(dm, o, fi), dof(dm, n, fi) # global dofs

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
    return M, K
end

# Per-step transport reassembly for state-dependent coefficients 
# Similar to assemble_react! 
function assemble_transport!(prob::ProblemCache, y::AbstractVector, 
    dirs::NTuple{N,Tuple{FaceSet,FaceGeometry}}) where {N}
    pressure!(prob.model, prob.pcell, y, prob.grid, prob.geom)
    fill!(K.nzval, 0.0)
    for fi in eachindex(dm.fields)
        transported(model, fi) || continue
        for (fs, fg) in dirs
            for e = 1:length(fs.owner)
                o, n = fs.owner[e], fs.neighbor[e] 
                go, gn = dof(dm, o, fi), dof(dm, n, fi) 

                T_face = fg.wf[e] * y[dof(prob.dm,o,fi)] + (1-fg.wf[e]) * y[dof(prob.dm,n,fi)]
                p_face = fg.wf[e] * pcell[o] + (1-fg.wf[e]) * pcell[n]

                k = diffusion_flux(diffusivity(model, fi, fs.axis, T_face, p_face),
                    fg.area[e], fg.dist[e]) 
                (co, cn) = advection_flux(velocity(model, fs.axis), fg.area[e]) 

                K[go, go] += k
                K[go, gn] -= k
                K[gn, go] -= k
                K[gn, gn] += k

                K[go, go] += co
                K[go, gn] += cn
                K[gn, go] -= co
                K[gn, gn] -= cn
            end
        end
    end               

end

# Weak Dirichlet or Danckwerts: Axial flux J_ax = βC - D_ax⋅∂_zC
# Inlet (β·n < 0): (β⋅C(0) - D_ax⋅∂_zC(0)) = β⋅C_in <=> J_ax(0)⋅A = β⋅A⋅C_in
# => f_in[dof] += -β·n·A·g
# Outlet  (β·n > 0): ∂_zC(L) = 0 <=> J_ax(L)⋅A = β⋅A⋅C_out
# => K[dof,dof] += β·n·A
function assemble_inlet_outlet!(K::SparseMatrixCSC, f_in::AbstractVector,
    dm::DofMap, geom::Geometry, grid::StructuredGrid, model::AbstractModel,
    bcs::Vararg{Tuple{BoundaryFaceSet, BoundaryFaceGeometry}})
    for field in dm.fields
        for (bfs, bfg) in bcs
            βn = velocity(model, bfs.axis) * bfs.side # outward normal advection
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
            reaction_jac!(prob.model, Jre, yc) # local nf×nf reaction jac
            for b in eachindex(cd), a in eachindex(cd)
                prob.Jr[cd[a], cd[b]] += Jre[a, b] * vol
            end
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
