module Problem

using SparseArrays
using ..Models

export ProblemCache

# Discretization cache
struct ProblemCache{Tgrid, Tgeom, Tdm, Tmodel<:AbstractModel}
    M::SparseMatrixCSC{Float64,Int}
    K::SparseMatrixCSC{Float64,Int}
    Jr::SparseMatrixCSC{Float64,Int}
    r::Vector{Float64} # global reaction source
    re::Vector{Float64} # local reaction source
    Jre::Matrix{Float64} # local reaction Jacobian
    f_in::Vector{Float64} # inlet forcing
    f_wall::Vector{Float64} # wall forcing
    grid::Tgrid
    geom::Tgeom
    dm::Tdm
    model::Tmodel # AbstractModel
    re0::Vector{Float64} # reaction jac finite diff
    yscr::Vector{Float64} # scratch
end

ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model) =
    ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model,
        similar(re), similar(re))

end