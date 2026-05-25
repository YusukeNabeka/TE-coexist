#src/transposition.jl
module SimTransOpt

include("regulation.jl")
using .SimReg 
using Random
using Statistics
using StatsBase


# Helper (Pre-allocation)
function ensure_capacity!(matrix::Matrix{Float64}, current_count::Int, needed::Int)
    rows, cols = size(matrix)
    if current_count + needed > rows
        new_size = max(rows * 2, current_count + needed + 1000)
        new_matrix = zeros(Float64, new_size, cols)
        new_matrix[1:rows, :] = matrix
        return new_matrix
    end
    return matrix
end


# Helper (Swap and Pop)
function swap_delete!(vec::Vector{Int}, item::Int, item_to_index_map::Dict{Int, Int})
    if !haskey(item_to_index_map, item)
        return
    end
    
    idx_to_remove = item_to_index_map[item]
    last_idx = length(vec)
    
    if idx_to_remove == last_idx
        pop!(vec)
    else
        last_item = vec[last_idx]
        vec[idx_to_remove] = last_item
        item_to_index_map[last_item] = idx_to_remove
        pop!(vec)
    end
    delete!(item_to_index_map, item)
end


# Transposition
function transposition!(
    transposonMatrix::Matrix{Float64}, 
    genomeMatrix::Matrix{Float64},
    v1::Vector{Int},
    v2::Vector{Int},
    u::Float64, v_rate::Float64, i_rate::Float64,
    kz1::Bool, kz2::Bool, f_kz::Float64, kz_locus::Int,
    current_te_count::Int,
    seed::Int
)
    rng = MersenneTwister(seed)
    
    # 1. Regulation
    effective_excisions, _ = SimReg.regulation(
        transposonMatrix, genomeMatrix, v1, v2; 
        eta=0.0, regulationStr=0.0,
        kz1=kz1, kz2=kz2, f_kz=f_kz
    )
    
    all_te_ids = vcat(v1, v2)
    num_te = length(all_te_ids)
    if num_te == 0
        return v1, v2, current_te_count, transposonMatrix
    end

    
    # 2. Judge (BitVector)
    check_exc = rand(rng, num_te) .< effective_excisions
    check_rep = rand(rng, num_te) .< v_rate
    check_ins = rand(rng, num_te) .< i_rate
    
    moving_indices = findall(check_exc .& check_rep .& check_ins)
    num_events = length(moving_indices)
    
    if num_events == 0
        return v1, v2, current_te_count, transposonMatrix
    end
    

    transposonMatrix = ensure_capacity!(transposonMatrix, current_te_count, num_events)
    
    # 3. Allocation
    L = size(genomeMatrix, 1)
    
    new_sites = Vector{Int}(undef, num_events)
    
    if kz_locus > 0
        if num_events > L - 1
            error("Too many transposition events ($num_events) for available sites ($(L-1))")
        end
        
        raw_sites = sample(rng, 1:(L-1), num_events, replace=false)
        for i in 1:num_events
            s = raw_sites[i]
            if s >= kz_locus
                new_sites[i] = s + 1
            else
                new_sites[i] = s
            end
        end
    else

        new_sites = sample(rng, 1:L, num_events, replace=false)
    end
    
    dest_chrom = rand(rng, 1:2, num_events)
    
   

    site_map_v1 = Dict{Int, Int}()
    site_map_v2 = Dict{Int, Int}()
    idx_map_v1 = Dict{Int, Int}()
    idx_map_v2 = Dict{Int, Int}()

    sites_v1_view = view(transposonMatrix, v1, 2)
    for (i, id) in enumerate(v1)
        s = round(Int, sites_v1_view[i])
        site_map_v1[s] = id
        idx_map_v1[id] = i
    end
    
    sites_v2_view = view(transposonMatrix, v2, 2)
    for (i, id) in enumerate(v2)
        s = round(Int, sites_v2_view[i])
        site_map_v2[s] = id
        idx_map_v2[id] = i
    end


    final_te_count = current_te_count
    
    for k in 1:num_events
        orig_idx_in_all = moving_indices[k]
        orig_te_id = all_te_ids[orig_idx_in_all]
        
        final_te_count += 1
        new_id = final_te_count
        new_site = new_sites[k]
        target_v = dest_chrom[k]
        

        transposonMatrix[new_id, 1] = transposonMatrix[orig_te_id, 1]
        transposonMatrix[new_id, 4] = transposonMatrix[orig_te_id, 4]
        transposonMatrix[new_id, 5] = transposonMatrix[orig_te_id, 5]
        transposonMatrix[new_id, 6] = transposonMatrix[orig_te_id, 6]
        
        transposonMatrix[new_id, 2] = Float64(new_site)
        transposonMatrix[new_id, 3] = genomeMatrix[new_site, 1]
        

        if target_v == 1
            if haskey(site_map_v1, new_site)
                old_id = site_map_v1[new_site]
                swap_delete!(v1, old_id, idx_map_v1)
            end
            push!(v1, new_id)
            site_map_v1[new_site] = new_id
            idx_map_v1[new_id] = length(v1)
            
        else # target_v == 2
            if haskey(site_map_v2, new_site)
                old_id = site_map_v2[new_site]
                swap_delete!(v2, old_id, idx_map_v2)
            end
            push!(v2, new_id)
            site_map_v2[new_site] = new_id
            idx_map_v2[new_id] = length(v2)
        end
    end
    
    return v1, v2, final_te_count, transposonMatrix
end

end # module