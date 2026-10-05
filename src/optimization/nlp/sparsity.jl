# diagonal nzval positions CSC matrix
function _diagindex(A::SparseMatrixCSC)
    idx = Vector{Int}(undef, size(A, 1))
    rv, cp = rowvals(A), A.colptr
    for d in eachindex(idx)
        p = searchsortedfirst(view(rv, cp[d]:(cp[d+1]-1)), d)
        idx[d] = cp[d] + p - 1
    end
    return idx
end