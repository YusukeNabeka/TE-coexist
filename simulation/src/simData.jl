#src/simData.jl
module SimData

using JLD2

function save_checkpoint(
    filepath::String, 
    population, 
    te_matrix, 
    genome, 
    rate2Map, 
    pi_coords,
    kz_locus,
    params, 
    current_te_count, 
    rng, 
    stats
)
    dir = dirname(filepath)
    if !isdir(dir) && !isempty(dir)
        mkpath(dir)
    end

    jldsave(filepath; 
        population, 
        te_matrix, 
        genome, 
        rate2Map, 
        pi_coords,
        kz_locus,
        params, 
        current_te_count,
        rng, 
        stats
    )
end

function load_checkpoint(filepath::String)
    d = load(filepath)
    return (
        population       = d["population"],
        te_matrix        = d["te_matrix"],
        genome           = d["genome"],
        rate2Map         = d["rate2Map"],
        pi_coords        = d["pi_coords"],
        kz_locus         = d["kz_locus"],
        params           = d["params"],
        current_te_count = d["current_te_count"],
        rng              = d["rng"],
        stats            = d["stats"]
    )
end

end # module