#src/regulation.jl
module SimReg

using Statistics

function regulation(
    transposonMatrix::Matrix{Float64}, 
    genomeMatrix::Matrix{Float64},     
    v1::Vector{Int},                   
    v2::Vector{Int};

    kz1::Bool = false,
    kz2::Bool = false,
    f_kz::Float64 = 0.0,
    
    eta::Float64 = 0.0,            
    regulationStr::Float64 = 0.0   
)

    te_ids = vcat(v1, v2)
    num_te = length(te_ids)
    
    if num_te == 0
        return Float64[], Dict{Int, Float64}()
    end


    te_families  = view(transposonMatrix, te_ids, 1)
    te_sites     = view(transposonMatrix, te_ids, 2)
    te_excisions = view(transposonMatrix, te_ids, 4)


    te_taus = Vector{Float64}(undef, num_te)
    
    for i in 1:num_te
        site_idx = round(Int, te_sites[i])
        te_taus[i] = genomeMatrix[site_idx, 4]
    end


    unique_families = unique(te_families)
    regulation_set = Dict{Int, Float64}()
    
    for fam_id in unique_families
        fam_indices = (te_families .== fam_id)
        net_tau = sum(te_taus[fam_indices])

        if net_tau > 1.0
            net_tau = 1.0
        end
        
        regulation_set[round(Int, fam_id)] = net_tau
    end

    if eta > 0.0
        keys_list = collect(keys(regulation_set))
        current_values = copy(regulation_set)
        
        for fam_id in keys_list
            other_sum = 0.0
            for other_id in keys_list
                if other_id != fam_id
                    other_sum += current_values[other_id]
                end
            end
            new_val = current_values[fam_id] + (other_sum * eta)
            if new_val > 1.0; new_val = 1.0; end
            regulation_set[fam_id] = new_val
        end
    end


    effective_rates = zeros(Float64, num_te)
    denominator = 1.0 + (num_te * regulationStr)
    
    has_kz = kz1 || kz2
    
    for i in 1:num_te
        fam_id = round(Int, te_families[i])
        base_u = te_excisions[i]
    
        R_pi = get(regulation_set, fam_id, 0.0)
        
        u_eff = (base_u - (base_u * R_pi)) / denominator

        if has_kz && f_kz > 0.0
            u_eff = u_eff * (1.0 - f_kz)
        end
        
        effective_rates[i] = u_eff
    end

    return effective_rates, regulation_set
end

end # module