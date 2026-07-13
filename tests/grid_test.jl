using SparseArrays
using PDEOpt

grid = StructuredGrid(1.0, 1.0, 10, 10)
geom = Geometry(grid)

# 
nc = ncells(grid)
dm = DofMap(nc, :C, :T)

# Sparsity pattern
K = sparsity_pattern(grid, dm)
display(K)

# Faces
f_axial, f_radial = interior_faces(grid)
#@show size(f_axial.owner)


