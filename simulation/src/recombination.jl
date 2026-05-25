#src/recombination.jl
module SimRec

using Random
using Statistics


function recombination(
    rate2Map::Vector{Float64},
    transposonMatrix::Matrix{Float64}, 
    v1::Vector{Int},
    v2::Vector{Int},
    kz1::Bool,
    kz2::Bool,
    kz_locus::Int
)
    
    # 1. Extraction (TE sites + KZFP locus)
    sites_v1 = isempty(v1) ? Int[] : [round(Int, s) for s in view(transposonMatrix, v1, 2)]
    sites_v2 = isempty(v2) ? Int[] : [round(Int, s) for s in view(transposonMatrix, v2, 2)]
    
    map_v1 = Dict{Int, Vector{Int}}()
    for (i, s) in enumerate(sites_v1); push!(get!(map_v1, s, Int[]), v1[i]); end
    map_v2 = Dict{Int, Vector{Int}}()
    for (i, s) in enumerate(sites_v2); push!(get!(map_v2, s, Int[]), v2[i]); end

    unique_sites = vcat(sites_v1, sites_v2)
    if kz_locus > 0
        push!(unique_sites, kz_locus)
    end
    
    unique_sites = sort(unique(unique_sites))
    
    if isempty(unique_sites)
        return Int[], false
    end

    
    # 2. Recombination
    gamete_te_ids = Int[]
    gamete_kz = false
    
    current_hap = rand(1:2) 
    prev_map_pos = 0.0
    
    for site in unique_sites
        curr_map_pos = rate2Map[site]
        dist = max(0.0, curr_map_pos - prev_map_pos)
        
        # Haldane mapping function
        r = 0.5 * (1.0 - exp(-2.0 * dist))
        if rand() < r
            current_hap = 3 - current_hap
        end
        
        # --- KZFP locus processing ---
        if site == kz_locus
            gamete_kz = (current_hap == 1) ? kz1 : kz2
        end
        
        # --- TE processing ---
        if current_hap == 1
            if haskey(map_v1, site)
                append!(gamete_te_ids, map_v1[site])
            end
        else
            if haskey(map_v2, site)
                append!(gamete_te_ids, map_v2[site])
            end
        end
        
        prev_map_pos = curr_map_pos
    end

    return gamete_te_ids, gamete_kz
end

end # module