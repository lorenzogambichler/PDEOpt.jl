module AssembleFVM

export advection_flux, diffusion_flux, reaction!

function advection_flux(βn::Float64, area::Float64)
    F = βn * area
    return F ≥ 0 ? (F, 0.0) : (0.0, F)
end

diffusion_flux(Df::Float64, area::Float64, dist::Float64) = Df * area / dist

function reaction!(Jre::AbstractMatrix{Float64}, re::AbstractVector{Float64},
    C::Float64, T::Float64, V::Float64, s, ds)
    sC, sT = s(C, T)
    ∂CC, ∂CT, ∂TC, ∂TT = ds(C, T)
    re[1] = sC * V
    re[2] = sT * V
    Jre[1, 1] = ∂CC * V
    Jre[1, 2] = ∂CT * V
    Jre[2, 1] = ∂TC * V
    Jre[2, 2] = ∂TT * V
    return Jre, re
end

end