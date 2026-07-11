module NonlinearSolvers

using LinearAlgebra

export newton!, quasi_newton!, frozen_newton

# Newton (solve F(x) = 0)
function newton!(x, residual!, jacobian!, linsolve!, F, s; maxiter::Int=30, tol::Float64=1e-8)
    for it = 1:maxiter
        residual!(F, x) # assembles F
        norm(F) < tol && break
        jacobian!(x) # reassemble and factorize Jacobian
        linsolve!(s, F) # solve Jδ = F
        @. x -= s
    end
    return x
end

# Quasi-Newton
# Factorize jac once at the initial guess and reuse it over the time step
# Refactorize jac every time step
function quasi_newton!(x, residual!, jacobian!, linsolve!, F, s; maxiter::Int=30, tol::Float64=1e-8)
    for it = 1:maxiter
        residual!(F, x)
        norm(F) < tol && break
        it == 1 && jacobian!(x) # factorize once, reuse
        linsolve!(s, F)
        @. x -= s
    end
    return x
end

# Frozen-Jacobian Newton
# Reuse one factorization across time-steps 
# Refactorize jac every refactor_every steps or after convergence failure (no accuracy loss)
# Use functor for persistent state
mutable struct frozen_newton
    refactor_every::Int 
    since::Int 
end
frozen_newton(; refactor_every::Int=10) = frozen_newton(refactor_every, refactor_every)

function (fn::frozen_newton)(x, residual!, jacobian!, linsolve!, F, s;
    maxiter::Int=30, tol::Float64=1e-8)
    refresh = fn.since ≥ fn.refactor_every # refactorize if true
    converged = false
    for it = 1:maxiter
        residual!(F, x)
        norm(F) < tol && (converged = true; break)
        if it == 1 && refresh
            jacobian!(x) # refactorize
            fn.since = 0
        end
        linsolve!(s, F)
        @. x -= s
    end
    fn.since = converged ? fn.since + 1 : fn.refactor_every # stalled -> force refactorization next step
    return x
end

end
