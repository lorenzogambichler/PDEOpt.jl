module Problem

using SparseArrays
using ..Models

export ProblemCache

# Discretization cache
struct ProblemCache{Tv, Tgrid, Tgeom, Tdm, Tmodel<:AbstractModel}
    M::SparseMatrixCSC{Tv,Int}
    K::SparseMatrixCSC{Tv,Int}
    Jr::SparseMatrixCSC{Tv,Int}
    r::Vector{Tv} # global reaction source
    re::Vector{Tv} # local reaction source
    Jre::Matrix{Tv} # local reaction Jacobian
    f_in::Vector{Tv} # inlet forcing
    f_wall::Vector{Tv} # wall forcing
    grid::Tgrid
    geom::Tgeom
    dm::Tdm
    model::Tmodel # AbstractModel
    re0::Vector{Tv} # reaction jac finite diff
    yscr::Vector{Tv} # scratch
end

ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model) =
    ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model,
        similar(re), similar(re))
end