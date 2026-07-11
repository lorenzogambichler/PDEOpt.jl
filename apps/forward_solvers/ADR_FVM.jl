using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt

#=
TODO:
- add f-evals and jacs counter to cn_solve and newton
- T-/p-dependence of D, λ, U, etc. (move diffusion assembly out of global, use properties dir)
- axially varying wall BC
- validation
- higher-order (MUSCL/TVD) axial advection 
=#

struct ProblemCache{Tgrid, Tgeom, Tdm, F1, F2, F3, F4}
    M::SparseMatrixCSC{Float64,Int}
    K::SparseMatrixCSC{Float64,Int}
    Jr::SparseMatrixCSC{Float64,Int}
    r::Vector{Float64} # reaction 
    Jre::Matrix{Float64}
    re::Vector{Float64} # element reaction
    f_in::Vector{Float64} # inlet forcing
    f_wall::Vector{Float64} # wall forcing     
    grid::Tgrid
    geom::Tgeom
    dm::Tdm
    src::F1 # reaction source
    Jsrc::F2
    y_in::F3 # inlet time modulation
    T_wall::F4 # wall time modulation
end

struct Coefficients{TD, Tλ}
    D::TD 
    λ::Tλ 
    β::Vector{Float64}
    k0::Float64
    Ea::Float64 
    ΔHr::Float64
    T_floor::Float64
    U::Float64
end

function assemble_global(M::SparseMatrixCSC, K::SparseMatrixCSC, 
    dirs::NTuple{N,Tuple{FaceSet,FaceGeometry}}, grid::StructuredGrid, geom::Geometry, 
    dm::DofMap, coeffs::Coefficients) where {N}
    # Cell loop
    for field in dm.fields
        for c = 1:ncells(grid)
            cdof = dof(dm, c, field)
            i, j = cellij(grid, c)
            M[cdof, cdof] = cellvolume(geom, i, j)
        end
    end

    # Face loop
    for field in dm.fields
        for (fs, fg) in dirs # axial/radial (FaceSet, FaceGeometry)
            for e = 1:length(fs.owner)
                o, n = fs.owner[e], fs.neighbor[e] # cell indices
                go, gn = dof(dm, o, field), dof(dm, n, field) # global dofs

                # Flux coeffs
                k = diffusion_flux(field==:C ? coeffs.D[fs.axis, fs.axis] : coeffs.λ[fs.axis, fs.axis],
                    fg.area[e], fg.dist[e]) # flux = k(yₒ - yₙ)
                (co, cn) = advection_flux(coeffs.β[fs.axis], fg.area[e]) # flux = cₒyₒ + cₙyₙ

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

# Weak Dirichlet or Danckwerts: Axial flux J_ax = βC - D_ax⋅∂_zC
# Inlet (β·n < 0): (β⋅C(0) - D_ax⋅∂_zC(0)) = β⋅C_in <=> J_ax(0)⋅A = β⋅A⋅C_in 
# => f_in[dof] += -β·n·A·g 
# Outlet  (β·n > 0): ∂_zC(L) = 0 <=> J_ax(L)⋅A = β⋅A⋅C_out
# => K[dof,dof] += β·n·A 
function assemble_inlet_outlet!(K::SparseMatrixCSC, f_in::Vector{Float64},
    dm::DofMap, geom::Geometry, grid::StructuredGrid, coeffs::Coefficients, 
    g, bcs::Vararg{Tuple{BoundaryFaceSet, BoundaryFaceGeometry}}) 

    for field in dm.fields
        for (bfs, bfg) in bcs
            βn = coeffs.β[bfs.axis] * bfs.side # outward normal advection
            for k = 1:length(bfs.cells)
                A = bfg.area[k]
                c = bfs.cells[k]
                gc = dof(dm, c, field)
                if βn > 0
                    K[gc, gc] += βn * A 
                elseif βn < 0
                    _, j = cellij(grid, c)
                    f_in[gc] += -βn * A * g[field](geom.rc[j])
                end
            end
        end
    end
end

# FLux: -λ⋅∂ᵣT(r=R)⋅A = U⋅A⋅(T - Tw)
# K[dof,dof] += U⋅A, f_wall[dof] += U⋅A
function assemble_wall!(K::SparseMatrixCSC, f_wall::Vector{Float64}, dm::DofMap,
    coeffs::Coefficients, bfs::BoundaryFaceSet, bfg::BoundaryFaceGeometry)
    for k = 1:length(bfs.cells)
        A = bfg.area[k]
        c = bfs.cells[k]
        gc = dof(dm, c, :T)
        K[gc, gc] += coeffs.U * A
        f_wall[gc] = coeffs.U * A
    end
end

# Re-assemble global reaction (r, Jr) from local (re, Jre) at every time step
function assemble_react_global!(prob::ProblemCache, y::AbstractVector)
    fill!(nonzeros(prob.Jr), 0.0)
    fill!(prob.r, 0.0)
    for c = 1:ncells(prob.grid)
        i, j = cellij(prob.grid, c)
        vol = cellvolume(prob.geom, i, j)
        cd = celldof(prob.dm, c) # (C dof, T dof)
        reaction!(prob.Jre, prob.re, y[cd[1]], y[cd[2]], vol, prob.src, prob.Jsrc) # (re, Jre)
        prob.r[cd[1]] += prob.re[1]
        prob.r[cd[2]] += prob.re[2]
        prob.Jr[cd[1], cd[1]] += prob.Jre[1, 1]
        prob.Jr[cd[1], cd[2]] += prob.Jre[1, 2]
        prob.Jr[cd[2], cd[1]] += prob.Jre[2, 1]
        prob.Jr[cd[2], cd[2]] += prob.Jre[2, 2]
    end
    return prob.Jr, prob.r
end

# Boundary forcing vector b(t) = f_in·y_in(t) + f_wall⋅T_wall(t)
function assemble_boundary_global!(prob::ProblemCache, b::AbstractVector, t::Float64)
    for field in prob.dm.fields
        fd = fielddof(prob.dm, field)
        @views @. b[fd] = prob.f_in[fd] * prob.y_in[field](t) # inlet/outlet (:C, :T)
    end
    @. b += prob.f_wall * prob.T_wall(t) # wall (only :T)
end

# (C,T) -> (sC, sT)
function Arrhenius(C, T, coeffs)
    R = coeffs.k0 * exp(-coeffs.Ea / (8.314 * max(T, coeffs.T_floor))) * C
    return (-R, -coeffs.ΔHr * R)
end

# (C,T) -> (∂sC/∂C, ∂sC/∂T, ∂sT/∂C, ∂sT/∂T)
function ArrheniusJac(C, T, coeffs) 
    e = coeffs.k0 * exp(-coeffs.Ea / (8.314 * max(T, coeffs.T_floor)))
    R = e * C
    RT = T > coeffs.T_floor ? R * coeffs.Ea / (8.314 * T^2) : 0.0
    return (-e, -RT, -coeffs.ΔHr * e, -coeffs.ΔHr * RT)
end

function write_vtk(path::String, cache::CNCache)
    prob = cache.prob
    grid = prob.grid
    nz, nr = grid.nz, grid.nr
    Cdofs, Tdofs = fielddof(prob.dm, :C), fielddof(prob.dm, :T)
    mkpath(dirname(path))
    paraview_collection(path) do pvd
        for n = 1:cache.N
            t = (n - 1) * cache.Δt
            # cell-index order, reshape (nr,nz) then transpose -> [i,j]
            Cgrid = permutedims(reshape(cache.y[Cdofs, n], nr, nz))
            Tgrid = permutedims(reshape(cache.y[Tdofs, n], nr, nz))
            vtk_grid("$(path)_$(n)", grid.z, grid.r) do vtk
                vtk["C", VTKCellData()] = Cgrid
                vtk["T", VTKCellData()] = Tgrid
                pvd[t] = vtk
            end
        end
    end
end

# Initial condition: each field constant at its inlet value at t = 0 (weak-BC consistent).
function set_ic!(cache::CNCache)
    prob = cache.prob
    for field in prob.dm.fields
        @views cache.y[fielddof(prob.dm, field), 1] .= prob.y_in[field](0.0)
    end
    return cache
end

function setup_problem(tf::Float64)
    # StructuredGrid, Geometry
    L, R = 2.0, 0.5
    nz, nr = 100, 25 
    grid = StructuredGrid(L, R, nz, nr)
    geom = Geometry(grid)

    # FaceSets, FaceGeometries
    fs_axial, fs_radial = interior_faces(grid)
    fg_axial = FaceGeometry(grid, geom, fs_axial)
    fg_radial = FaceGeometry(grid, geom, fs_radial)

    # BoundaryFaceSets, BoundaryFaceGeometries
    bfs = boundary_faces(grid)
    bfg_inlet = BoundaryFaceGeometry(grid, geom, bfs.inlet)
    bfg_outlet = BoundaryFaceGeometry(grid, geom, bfs.outlet)
    bfg_wall = BoundaryFaceGeometry(grid, geom, bfs.wall)

    # DofMap
    dm = DofMap(ncells(grid), :C, :T)

    # Coefficients
    D = [0.0 0.0; 0.0 1e-2] # W/(mK), radial diffusion/conduction
    λ = [0.0 0.0; 0.0 1e-2]
    β = [2.0, 0.0] # m/s, axial advection
    k0 = 1e7 # 1/s
    Ea = 40e3 # J/mol
    ΔHr = -120.0 # J/mol
    T_floor = 0.1 # K
    U = 50.0 # W/(m^2K), 1/(1/α_inner + d_w/λ_w + 1/α_outer)

    coeffs = Coefficients(D, λ, β, k0, Ea, ΔHr, T_floor, U)

    # Matrices, shared pattern
    K = sparsity_pattern(grid, dm)
    M = copy(K)
    Jr = copy(K)

    # Boundary conditions:
    # Inlet
    g = (C = r -> 1.0, T = r -> 1.0)
    t_ramp = 0.5 * tf
    C_in(t) = 1.0
    T_in(t) = 100.0 + (500.0 - 100.0) * clamp(t / t_ramp, 0.0, 1.0) # 100K -> 500K
    y_in = (C = C_in, T = T_in) 
    # Wall
    T_wall(t) = 100.0 # time-varying later

    # Assembly
    f_in = zeros(ndof(dm))
    f_wall = zeros(ndof(dm))
    assemble_global(M, K, ((fs_axial,fg_axial),(fs_radial,fg_radial)), grid, geom, dm, coeffs)
    assemble_inlet_outlet!(K, f_in, dm, geom, grid, coeffs, g,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet))
    assemble_wall!(K, f_wall, dm, coeffs, bfs.wall, bfg_wall)

    # Arrhenius source term and jac
    Jre = zeros(2, 2)
    re = zeros(2)
    r = zeros(ndof(dm))
    src(C, T) = Arrhenius(C, T, coeffs) 
    Jsrc(C, T) = ArrheniusJac(C, T, coeffs)

    return ProblemCache(M, K, Jr, r, Jre, re, f_in, f_wall, grid, geom, dm, src, Jsrc, y_in, T_wall)
end

function main()
    # Time-stepping
    tf = 2.5
    Δt = 0.01
    N = round(Int64, tf / Δt) + 1

    # Caches
    prob = setup_problem(tf)
    cache = CNCache(prob, Δt, N)

    # IC
    set_ic!(cache)

    cn_solve!(cache, y -> assemble_react_global!(prob, y), 
        (b, t) -> assemble_boundary_global!(prob, b, t), quasi_newton!) # newton! or quasi_newton!
    write_vtk("apps/forward_solvers/results/ADR_FVM/ADR_FVM", cache)
end
