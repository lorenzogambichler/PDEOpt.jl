using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt
using Printf

const SPECIES = (:CH4, :CO, :CO2, :H2O, :H2, :N2)

function write_vtk(path::String, cache::CNCache)
    prob = cache.prob
    grid = prob.grid
    nz, nr = grid.nz, grid.nr
    mkpath(dirname(path))
    paraview_collection(path) do pvd
        for n = 1:cache.N
            t = (n - 1) * cache.Δt
            vtk_grid("$(path)_$(n)", grid.z, grid.r) do vtk
                for field in prob.dm.fields
                    fd = fielddof(prob.dm, field)
                    # cell-index order
                    vtk[string(field), VTKCellData()] = permutedims(reshape(cache.y[fd, n], nr, nz))
                end
                pvd[t] = vtk
            end
        end
    end
end

function setup_problem()
    # StructGrid2D, Geometry
    L, R = 5.0, 0.01 
    nz, nr = 150, 7
    grid = StructGrid2D(L, R, nz, nr)
    geom = Geometry2D(grid)

    # FaceSets, FaceGeometries
    (fs_axial, fs_radial) = interior_faces(grid)
    fg_axial = FaceGeometry(grid, geom, fs_axial)
    fg_radial = FaceGeometry(grid, geom, fs_radial)

    # BoundaryFaceSets, BoundaryFaceGeometries
    bfs = boundary_faces(grid)
    bfg_inlet = BoundaryFaceGeometry(grid, geom, bfs.inlet)
    bfg_outlet = BoundaryFaceGeometry(grid, geom, bfs.outlet)
    bfg_wall = BoundaryFaceGeometry(grid, geom, bfs.wall)

    # Fields + DofMap
    fields = (SPECIES..., :T)
    dm = DofMap(ncells(grid), fields)

    # Inlet
    T0, p0, Rgas = 400.0, 5.0e5, 8.314
    Mα = (CH4=0.016043, CO=0.028010, CO2=0.044010, H2O=0.018015, H2=0.0020159, N2=0.0280134)
    xin = (CH4=1e-5, CO=1e-5, CO2=0.2, H2O=1e-5, H2=0.8, N2=1e-5)
    cmol = p0 / (Rgas * T0) # total molar density, mol/m³
    ρin = NamedTuple{SPECIES}(ntuple(i -> xin[SPECIES[i]] * cmol * Mα[SPECIES[i]], length(SPECIES)))

    # Inlet BC (g(r)·y_in(t))
    #t_ramp = 0.5
    #T_in(t) = 300.0 + 250.0 * clamp(t / t_ramp, 0.0, 1.0) # 300K -> 550K
    T_in(t) = 400.0
    g = (CH4=r->1.0, CO=r->1.0, CO2=r->1.0, H2O=r->1.0, H2=r->1.0, N2=r->1.0, T=r->1.0)
    y_in = (CH4=t->ρin.CH4, CO=t->ρin.CO, CO2=t->ρin.CO2, H2O=t->ρin.H2O,
        H2=t->ρin.H2, N2=t->ρin.N2, T=T_in)
    T_wall(t) = 650.0

    # Model
    model = MethanationModel(fields; g=g, y_in=y_in, T_wall=T_wall,
        vz=1.0, kw=120.0, Rrad=R)

    # Alloc matrices
    K = sparsity_pattern(grid, dm)
    M = copy(K)
    Jr = copy(K)

    # Boundaries, exclude state-dependent T (assembled in sa)
    f_in = zeros(ndof(dm))
    f_wall = zeros(ndof(dm))
    K_bc = copy(K) # constant boundary transport contrib
    fill!(K_bc.nzval, 0.0)
    assemble_inlet_outlet!(K_bc, f_in, dm, geom, grid, model,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet); fields=SPECIES)
    assemble_wall!(K_bc, f_wall, dm, model, bfs.wall, bfg_wall)

    # Reaction, global + local
    nf = length(fields)
    r = zeros(ndof(dm))
    re = zeros(nf)
    Jre = zeros(nf, nf)

    prob = ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model)

    # Re-assembly
    props = MethanationProps(nspecies(model), ncells(grid))
    dirs = ((fs_axial, fg_axial), (fs_radial, fg_radial)) # int faces
    ebnd = ((bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet)) # inlet/outlet faces
    sa = StateAssembly(prob, props, K_bc, dirs, ebnd)

    # Evaluate M, K at inlet for first CN factorization
    y0 = zeros(ndof(dm))
    for field in fields
        @views y0[fielddof(dm, field)] .= inlet_mod(model, field, 0.0)
    end
    sa(y0)

    return prob, sa
end

function main()
    # Time-stepping
    tf = 1000.0
    Δt = 0.5
    N = round(Int, tf / Δt) + 1

    # Caches
    prob, sa = setup_problem()
    cache = CNCache(prob, Δt, N)

    # IC
    for field in prob.dm.fields
        set_ic!(cache, field, inlet_mod(model, field, 0.0))
    end

    # Solve (state-dependent -> reassemble!) and write results
    @time cn_solve!(cache,
        (y, WithJac) -> assemble_react!(prob, y, WithJac),
        (b, t) -> assemble_boundary!(prob, b, t),
        shamanskii(refactor_every = 20); reassemble! = sa) # newton!, chord!, shamanskii()
    write_vtk("apps/forward_solvers/results/Methanation/Methanation", cache)
    @printf("%9s %7s %9s\n", "f-evals", "jacs", "steps")
    @printf("%9d %7d %9d\n", cache.stats.nf, cache.stats.nfact, N-1)
    #@printf("%9s %7f\n", "Tmax at t=400s: ", max(cache.y[:,800]))
end
