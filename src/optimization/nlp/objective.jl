struct Objective{TC, TG}
    cost::TC # x -> J(x)
    cost_grad!::TG # x -> ∇J(x)
end
