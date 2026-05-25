#src/simKZFP.jl
module SimKZFP

using Random
using StatsBase
# It depends on the structure of SimPop.Individual.

function inject_kzfp!(population, freq::Float64, rng::AbstractRNG)
    N = length(population)
    if freq <= 0.0
        return
    end
    
    num_carriers = round(Int, N * freq)
    if num_carriers == 0 && freq > 0
        num_carriers = 1
    end
    
    indices = sample(rng, 1:N, num_carriers, replace=false)
    
    for idx in indices
        ind = population[idx]
        
        
        if rand(rng) < 0.5
            ind.kz1 = true
        else
            ind.kz2 = true
        end
    end
    
    return num_carriers
end

end # module