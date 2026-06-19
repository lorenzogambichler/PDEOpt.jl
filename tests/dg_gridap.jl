using Gridap
using GLMakie
using GridapMakie
using OrdinaryDiffEq
using LinearAlgebra

# Domain
domain = (0.0, 1.0)
partition = (200,)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "left", [1])
add_tag_from_tags!(labels, "right", [2])
h = (domain[2] - domain[1]) / partition[1]

# FE spaces
order = 1
reffe = ReferenceFE(lagrangian, Float64, order) 
V = TestFESpace(model, reffe, conformity = :L2) 
U  = TrialFESpace(V)

# Collocation
Ω = Triangulation(model)
Γ_left = BoundaryTriangulation(model, tags = "left")
Γ_right = BoundaryTriangulation(model, tags = "right")
Λ = SkeletonTriangulation(model)

degree = 2 * order
dΩ = Measure(Ω, degree)
dΓ_left = Measure(Γ_left, degree)
dΓ_right = Measure(Γ_right, degree)
dΛ = Measure(Λ, degree)

n_Γ_left = get_normal_vector(Γ_left)
n_Γ_right = get_normal_vector(Γ_right)
n_Λ = get_normal_vector(Λ)

# Weak Formulation
D = 1e-6
b = VectorValue(0.1)

# Bulk terms
m(u, v) = ∫( v * u )dΩ
a_Ω(u, v) = ∫( ∇(v) ⋅ (D * ∇(u)) - ∇(v) ⋅ (b * u) )dΩ

# Boundary terms
γ = order * (order + 1)
a_Γ_left(u,v) = ∫( - v * (D * ∇(u) ⋅ n_Γ_left) 
                   - (D * ∇(v) ⋅ n_Γ_left) * u 
                   + (γ * D / h) * v * u )dΓ_left
a_Γ_right(u,v) = ∫( v * (b ⋅ n_Γ_right) * u )dΓ_right  
a_Γ(u, v) = a_Γ_left(u, v) + a_Γ_right(u, v)

l_Γ_left(v) = ∫(- (D * ∇(v) ⋅ n_Γ_left) - v * (b ⋅ n_Γ_left) + (γ * D / h) * v )dΓ_left 

# Interior facet terms (SIPG)
a_Λ_diff(u,v) = ∫( - jump(v * n_Λ) ⋅ mean(D * ∇(u)) 
                   - mean(D * ∇(v)) ⋅ jump(u * n_Λ) 
                   + (γ * D / h) * jump(v * n_Λ) ⋅ jump(u * n_Λ) )dΛ
a_Λ_adv(u, v) = ∫( jump(v) * mean(b * u) + 0.5 * abs(b ⋅ n_Λ.⁺) * jump(v) * jump(u) )dΛ

a_Λ(u, v) = a_Λ_diff(u, v) + a_Λ_adv(u, v)

# Total forms
a(u,v) = a_Ω(u,v) + a_Γ(u, v) + a_Λ(u,v)
l(v) = l_Γ_left(v)

# IC and BC
g(t) = 1.0 + 0.25 * sin(0.25 * π * t)
uh0 = interpolate_everywhere(x -> 0.5 * (1.0 - tanh(20.0 * (x[1] - 1.0))), U)
u0_vec = get_free_dof_values(uh0)

# Matrix assembly
M = assemble_matrix(m, U, V)
K = assemble_matrix(a, U, V)
F = assemble_vector(l, V)

# ODE solver
function f_rhs!(du, u, p, t)
    g_t = g(t)    
    mul!(du, K, u)
    @. du = F * g_t - du - 0.001 * u^2 # Scale F with g(t)
end

f_ode = ODEFunction(f_rhs!, mass_matrix=M)

#prob = SplitODEProblem(f_imp, f_exp, u0_vec, (0.0, 10.0))
prob = ODEProblem(f_ode, u0_vec, (0.0, 10.0))
sol = @btime OrdinaryDiffEq.solve(prob, KenCarp4());

# Plot
time_obs = Observable(sol.t[1])

uh_obs = lift(time_obs) do t
    u_vec = sol(t)                
    return FEFunction(U, u_vec)   
end

title_obs = lift(t -> "t = $(round(t, digits=2)) s", time_obs)

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], title=title_obs)

plt = lines!(ax, Ω, uh_obs)
ylims!(0.0, 2.0)

fps = 30
dt_video = 0.1  
video_times = sol.t[1] : dt_video : sol.t[end]

record(fig, "results/dg_sol.mp4", video_times; framerate=fps) do t
    time_obs[] = t
end