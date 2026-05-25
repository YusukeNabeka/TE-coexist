#src/generatePoplation.jl
module SimPop

using Random
using Statistics
using StatsBase

# Individual genotype and fitness
mutable struct Individual
    v1::Vector{Int}  # TE IDs on haplotype 1
    v2::Vector{Int}  # TE IDs on haplotype 2

    kz1::Bool        # KZFP allele on haplotype 1
    kz2::Bool        # KZFP allele on haplotype 2
    
    w::Float64       # fitness
end


# Initialize a population with TEs and, optionally, KZFP alleles
function generatePopulation(
    genomeMatrix::Matrix{Float64}, 
    piRNAindices::Vector{Int}
    
    # KZFP settings
    kz_locus::Int = 0,               # KZFP locus; 0 means absent
    initial_kz_freq::Float64 = 0.0,  # initial frequency of KZFP carriers
    
    # Population settings
    NumberOfIndividual::Int = 1000,
    NumberOfTransposonTypes::Int = 1,
    
    # Initial TE settings
    NumberOfInsertionsPerType::Vector{Int} = [1],
    FrequencyOfInsertions::Vector{Float64} = [0.1],
    ExcisionRates::Vector{Float64} = [0.1],
    RepairRates::Vector{Float64} = [1.0],
    InsertionRates::Vector{Float64} = [1.0],
    
    seed::Int = 123
)
    rng = MersenneTwister(seed)
    
    # Identify valid TE insertion sites
    L = size(genomeMatrix, 1)

    all_sites = Set(1:L)
    pi_sites = Set(piRNAindices)
    
    valid_sites_set = setdiff(all_sites, pi_sites)
    
    # Exclude the KZFP locus from possible TE insertion sites
    if kz_locus > 0
        delete!(valid_sites_set, kz_locus)
    end
    
    valid_sites = collect(valid_sites_set)
    
    if isempty(valid_sites)
        error("No valid insertion sites available (Genome is full of piRNA/KZFP?)")
    end

    # Temporary arrays for TE attributes
    temp_te_family    = Int[]
    temp_te_site      = Int[]
    temp_te_selection = Float64[]
    temp_te_excision  = Float64[]
    temp_te_repair    = Float64[]
    temp_te_insertion = Float64[]
    
    TEset = Dict{Int, Vector{Int}}()
    for k in 1:NumberOfTransposonTypes
        TEset[k] = Int[]
    end

    # Initialize empty individuals
    population = [Individual(Int[], Int[], false, false, 1.0) for _ in 1:NumberOfIndividual]
    
    # Introduce KZFP alleles as heterozygotes
    if kz_locus > 0 && initial_kz_freq > 0.0
        num_kz_carriers = floor(Int, NumberOfIndividual * initial_kz_freq)
        carrier_indices = sample(rng, 1:NumberOfIndividual, num_kz_carriers, replace=false)
        
        for idx in carrier_indices
            if rand(rng) < 0.5
                population[idx].kz1 = true
            else
                population[idx].kz2 = true
            end
        end
    end

    # Distribute initial TE insertions
    current_te_id = 1
    ind_indices = collect(1:NumberOfIndividual)

    for fam_idx in 1:NumberOfTransposonTypes
        shuffle!(rng, ind_indices)
        limit = floor(Int, NumberOfIndividual * FrequencyOfInsertions[fam_idx])
        target_individuals = ind_indices[1:limit]
        
        insertions_per_ind = NumberOfInsertionsPerType[fam_idx]
        
        for ind_idx in target_individuals
            for _ in 1:insertions_per_ind
                site = rand(rng, valid_sites)
                sel_coef = genomeMatrix[site, 1] 
                
                push!(temp_te_family, fam_idx)
                push!(temp_te_site, site)
                push!(temp_te_selection, sel_coef)
                push!(temp_te_excision, ExcisionRates[fam_idx])
                push!(temp_te_repair, RepairRates[fam_idx])
                push!(temp_te_insertion, InsertionRates[fam_idx])
                
                push!(TEset[fam_idx], current_te_id)
                
                target_ind = population[ind_idx]
                if rand(rng) < 0.5
                    push!(target_ind.v1, current_te_id)
                else
                    push!(target_ind.v2, current_te_id)
                end
                
                current_te_id += 1
            end
        end
    end
    
    # Build transposon matrix:
    # columns = family, site, selection coefficient, excision rate, repair rate, insertion rate
    num_te = length(temp_te_family)
    transposonMatrix = zeros(Float64, num_te, 6)
    
    if num_te > 0
        transposonMatrix[:, 1] = temp_te_family
        transposonMatrix[:, 2] = temp_te_site
        transposonMatrix[:, 3] = temp_te_selection
        transposonMatrix[:, 4] = temp_te_excision
        transposonMatrix[:, 5] = temp_te_repair
        transposonMatrix[:, 6] = temp_te_insertion
    end
    

    return population, transposonMatrix, TEset
end

end # module