module NonlinearSolvers

using LinearAlgebra

export newton!, chord!, shamanskii

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
function chord!(x, residual!, jacobian!, linsolve!, F, s; maxiter::Int=30, tol::Float64=1e-8)
    for it = 1:maxiter
        residual!(F, x)
        norm(F) < tol && break
        it == 1 && jacobian!(x) # factorize once, reuse
        linsolve!(s, F)
        @. x -= s
    end
    return x
end

# Frozen-Jacobian Newton (Shamanskii)
# Reuse one factorization across time-steps
# Refactorize jac every refactor_every steps OR mid-step if r/rprev ≥ slow
# Use functor for persistent state
mutable struct shamanskii
    refactor_every::Int 
    slow::Float64 # contractoin rate threshold
    since::Int 
end
shamanskii(; refactor_every::Int=10, slow::Float64=0.5) =
    shamanskii(refactor_every, slow, refactor_every)

function (sh::shamanskii)(x, residual!, jacobian!, linsolve!, F, s;
    maxiter::Int=30, tol::Float64=1e-8)
    refresh = sh.since ≥ sh.refactor_every # scheduled refactor due at step entry?
    refreshed = false # already refactored during this step?
    converged = false
    rprev = Inf
    for _ = 1:maxiter
        residual!(F, x)
        r = norm(F)
        r < tol && (converged = true; break)
        # Refactor if scheduled, or if frozen J is stalling (slow contraction)
        if !refreshed && (refresh || r > sh.slow * rprev)
            jacobian!(x) # refactorize and retry 
            sh.since = 0
            refreshed = true
        end
        rprev = r
        linsolve!(s, F)
        @. x -= s
    end
    sh.since = converged ? sh.since + 1 : sh.refactor_every # stalled -> force refactor next step
    return x
end

end
