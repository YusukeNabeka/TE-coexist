#src/generateGenome.jl
module SimGen

using Random
using Statistics
using StatsBase

function generateGenome(;
    baseSelection::Union{Float64, String, Nothing} = nothing,
    baseInsertionProb::Union{Float64, String} = 1.0,
    numberOfInsertionSites::Int = 10000,
    numberOfChromosomes::Int = 30,
    baseRecombinationRate::Float64 = 0.01,
    baseTau::Float64 = 1.0,
    numberOfPiRNA::Int = 1,
    piPercentage::Float64 = 3.0,
    disablePiRecombination::Union{Bool, Float64} = true,
    enablePiSelection::Union{Bool, Float64} = true,
    clusterLoc::Bool = true,
    kz_chrom::Int = 0,
    seed::Int = 123
)

    rng = MersenneTwister(seed)

    # 1. Initialize genome columns
    # selection coefficient
    if isnothing(baseSelection)
        SelectionCoef = zeros(Float64, numberOfInsertionSites)
    elseif baseSelection == "Random"
        SelectionCoef = randn(rng, numberOfInsertionSites) .* 0.01 .- 0.02
    else
        SelectionCoef = fill(Float64(baseSelection), numberOfInsertionSites)
    end

    # insertion probability
    if baseInsertionProb == "Random"
        insertionProbability = rand(rng, numberOfInsertionSites) .* (0.99 - 0.01) .+ 0.01
    else
        insertionProbability = fill(Float64(baseInsertionProb), numberOfInsertionSites)
    end

    # recombination rate
    RecombinationRates = fill(baseRecombinationRate, numberOfInsertionSites)

    # piRNA activity
    piRNArray = zeros(Float64, numberOfInsertionSites)
    
    piRNAcoord = Dict{Int, Tuple{Int, Int}}() 



    # 2. Determine piRNA cluster lengths
    if numberOfPiRNA > 0
        totalPiRNALength = floor(Int, numberOfInsertionSites * (piPercentage / 100))
        individualPiRNALength = floor(Int, totalPiRNALength / numberOfPiRNA)
    else
        totalPiRNALength = 0
        individualPiRNALength = 0
    end

    leftOverLength = totalPiRNALength - (individualPiRNALength * numberOfPiRNA)
    individualPiRNALengthArray = fill(individualPiRNALength, numberOfPiRNA)
    for _ in 1:leftOverLength
        idx = rand(rng, 1:numberOfPiRNA)
        individualPiRNALengthArray[idx] += 1
    end


    # 3. Define chromosome boundaries
    splits = [round(Int, x) for x in range(0, numberOfInsertionSites, length=numberOfChromosomes+1)]
    
    chrom_ranges = []
    for i in 1:numberOfChromosomes
        start_idx = splits[i] + 1
        end_idx   = splits[i+1]
        push!(chrom_ranges, (start_idx, end_idx))
    end


    if numberOfChromosomes > 1
        for i in 1:(numberOfChromosomes - 1)
            boundary_site = chrom_ranges[i][2]
            if boundary_site < numberOfInsertionSites
                RecombinationRates[boundary_site] = 0.499
            end
        end
    end


    # 4. Reserve KZFP locus
    kz_locus = 0
    
    if kz_chrom > 0 && kz_chrom <= numberOfChromosomes
        (_, chr_end) = chrom_ranges[kz_chrom]
        kz_locus = chr_end
        
        insertionProbability[kz_locus] = 0.0
    end


    # 5. Place piRNA clusters
    counter = 1

    if clusterLoc
        # Place clusters at chromosome starts
        for i in 1:numberOfChromosomes
            if counter > numberOfPiRNA; break; end
            
            (chr_start, chr_end) = chrom_ranges[i]
            current_len = individualPiRNALengthArray[counter]
            
            pi_start = chr_start
            pi_end   = pi_start + current_len - 1
            
            if kz_locus > 0 && (pi_start <= kz_locus <= pi_end)
                 error("KZFP locus at $kz_locus overlaps with piRNA cluster on chromosome $i")
            end
            
            if pi_end > chr_end
                 error("Chromosome $i is too small for piRNA cluster")
            end
            
            piRNArray[pi_start:pi_end] .= baseTau
            piRNAcoord[counter] = (pi_start, pi_end)
            counter += 1
        end
        
    else
        # Place clusters randomly
        while counter <= numberOfPiRNA
            for (chr_start, chr_end) in chrom_ranges
                if counter > numberOfPiRNA; break; end
                
                current_len = individualPiRNALengthArray[counter]
                valid_start_max = chr_end - current_len + 1
                
                if valid_start_max < chr_start; continue; end
                
                possible_starts = chr_start:valid_start_max
                
                pi_start = rand(rng, possible_starts)
                pi_end   = pi_start + current_len - 1
                

                retry = 0
                is_overlapping = true
                while is_overlapping
                    hits_kz = (kz_locus > 0) && (pi_start <= kz_locus <= pi_end)
                    
                    if hits_kz || any(x -> x > 0, piRNArray[pi_start:pi_end])
                        pi_start = rand(rng, possible_starts)
                        pi_end   = pi_start + current_len - 1
                        retry += 1
                        if retry > 10; break; end
                    else
                        is_overlapping = false
                    end
                end
                
                if !is_overlapping
                    piRNArray[pi_start:pi_end] .= baseTau
                    piRNAcoord[counter] = (pi_start, pi_end)
                    counter += 1
                end
            end
            if counter <= numberOfPiRNA
                error("Could not place all piRNA clusters.")
            end
        end
    end



    piRNAindices = findall(!iszero, piRNArray)

    if enablePiSelection == false
        SelectionCoef[piRNAindices] .= 0.0
    elseif isa(enablePiSelection, Float64)
        SelectionCoef[piRNAindices] .= enablePiSelection
    end

    if disablePiRecombination != false
        val = isa(disablePiRecombination, Float64) ? disablePiRecombination : 0.0
        RecombinationRates[piRNAindices] .= val
    end


    # Convert recombination rates to map positions
    clamped_rates = min.(RecombinationRates, 0.499999)
    dists = -0.5 .* log.(1.0 .- (2.0 .* clamped_rates))
    rate2Map = cumsum(dists)


    # Genome columns:
    # selection coefficient, insertion probability, recombination rate, piRNA activity
    genome = hcat(SelectionCoef, insertionProbability, RecombinationRates, piRNArray)

    return genome, piRNAcoord, piRNAindices, rate2Map, kz_locus
end

end # module