struct MethanationOCP{TSA, TAD} <: AbstractOCP
    lay::CollocationLayout
    sc::Scaling
    sa::TSA
    adiff::TAD
    y0::Vector{Float64}
    # TODO
    # e.g. 
    # obj::Objective
    # resd::Residual
end

