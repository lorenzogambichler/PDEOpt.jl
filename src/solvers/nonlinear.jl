module NonlinearSolvers

using LinearAlgebra

export newton!, quasi_newton!

# Newton (solve F(x) = 0)
function newton!(x, residual!, jacobian!, linsolve!, F, δ; maxiter::Int=30, tol::Float64=1e-8)
    for _ = 1:maxiter
        residual!(F, x) # assembles F
        norm(F) < tol && break
        jacobian!(x) # reassemble and factorize Jacobian
        linsolve!(δ, F) # solve Jδ = F
        @. x -= δ
    end
    return x
end

# Quasi-Newton, factorize Jacobian once at the initial guess and reuse it
function quasi_newton!(x, residual!, jacobian!, linsolve!, F, δ; maxiter::Int=30, tol::Float64=1e-8)
    for it = 1:maxiter
        residual!(F, x)
        norm(F) < tol && break
        it == 1 && jacobian!(x) # factorize once, reuse
        linsolve!(δ, F)
        @. x -= δ
    end
    return x
end

end
