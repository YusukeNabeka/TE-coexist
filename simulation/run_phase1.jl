# run_phase1.jl
# Phase 1: burn-in without KZFP.
# Generates TE-piRNA equilibrium populations and saves valid checkpoints.

# Run this script from the repository root.

include("src/setup.jl")

output_dir = "./data/phase1"
if !isdir(output_dir)
    mkpath(output_dir)
end

# Parameter grid
grid_u        = [0.01]                              # basic transposition rate
grid_s        = [0.005, 0.0025, 0.001, 0.00075]     # selection coefficient
grid_pi_ratio = [0.01]                              # cluster size
grid_r        = [0.499]                             # recombination rate

replicates    = 10
generations   = 10000

base_N = 10000
base_L = 10000
base_chroms = 30
kz_chrom_setting = 2 




println(">>> Starting Phase 1: Equilibration...")
println("    Target Directory: $output_dir")

total_combinations = length(grid_u) * length(grid_s) * length(grid_pi_ratio) * length(grid_r) * replicates
counter = 0
saved_count = 0

for u in grid_u, s in grid_s, pi_r in grid_pi_ratio, r in grid_r
    for rep in 1:replicates
        global counter += 1
        
        seed = 10000 + counter
        rng = MersenneTwister(seed)
        
        fname = @sprintf("N%d_p1_u%.3f_s%.3f_pi%.2f_r%.3f_rep%d", base_N, u, s, pi_r, r, rep)
        save_path = joinpath(output_dir, fname*".jld2")
        log_path  = joinpath(output_dir, fname*".tsv")
        
        @printf("[%d/%d] Running: %s ... \n", counter, total_combinations, fname)
        
        # Set parameters
        local params = Main.Params(
            N=base_N, generations=generations, L=base_L, chroms=base_chroms, r=r,
            init_te_num=10, init_te_freq=0.1, u=u, s=s,
            pi_ratio=pi_r, num_clusters=1, pi_s=true, pi_rec=false, f_pi=1.0,
            kz_chrom=kz_chrom_setting, f_kz=0.0, c_kz=0.0, initial_kz_freq=0.0, kz_cost_mode=:dominant,
            seed=seed, output_file=log_path, log_interval=10
        )
        
        # Initialize genome and population
        genome, pi_coords, pi_indices, rate2Map, kz_locus = SimGen.generateGenome(
            numberOfInsertionSites=params.L, numberOfChromosomes=params.chroms, 
            baseRecombinationRate=params.r, baseSelection=-abs(params.s), baseTau=params.f_pi, 
            numberOfPiRNA=params.num_clusters, piPercentage=params.pi_ratio*100, clusterLoc=true,
            enablePiSelection=(params.pi_s ? -abs(params.s) : 0.0), disablePiRecombination=!params.pi_rec,
            kz_chrom=params.kz_chrom, seed=params.seed
        )

        population, te_matrix, _ = SimPop.generatePopulation(
            genome, pi_indices, kz_locus=kz_locus, initial_kz_freq=params.initial_kz_freq,
            NumberOfIndividual=params.N, NumberOfInsertionsPerType=[params.init_te_num],
            FrequencyOfInsertions=[params.init_te_freq], ExcisionRates=[params.u], seed=params.seed
        )
        
        current_te_count = size(te_matrix, 1)


        local final_pop, final_te_matrix, final_te_count
        
        open(log_path, "w") do io
            @printf(io, "gen\tte\tpi_freq\tkz_freq\n")
            
            stats0 = SimStats.calculate_statistics(population, te_matrix, genome, pi_coords)
            @printf(io, "%d\t%.4f\t%.4f\t%.4f\n", 0, stats0.mean_te_copy_number, stats0.pi_allele_frequency,0.0)
            @printf("Gen %d: TE=%.2f pi=%.3f kz=%.3f\n", 0, stats0.mean_te_copy_number, stats0.pi_allele_frequency,0.0)
            
            # Run burn-in generations
            final_pop, final_te_matrix, final_te_count = SimCore.run_generations!(
                population, te_matrix, genome, pi_coords, rate2Map,
                params, current_te_count, kz_locus, rng, io;
                start_gen=1, phase=1
            )
        end
        
        # Save only non-absorbing TE-piRNA equilibria
        stats = SimStats.calculate_statistics(final_pop, final_te_matrix, genome, pi_coords)
        
        te_num = stats.mean_te_copy_number
        pi_freq = stats.pi_allele_frequency
        
        is_valid = (te_num > 0.5) && (pi_freq < 1.0)
        
        if is_valid
            SimData.save_checkpoint(
                save_path, 
                final_pop, 
                final_te_matrix, 
                genome, 
                rate2Map,
                pi_coords,
                kz_locus,
                params, 
                final_te_count,
                rng, 
                stats
            )
            println("-> Saved (TE: $(round(te_num, digits=2)), pi: $(round(pi_freq, digits=3)))")
            global saved_count += 1
        else
            println("-> Skipped (Absorbing State: TE=$(round(te_num, digits=2)), pi=$(round(pi_freq, digits=3)))")
        end
        
        # Free memory between replicates
        final_pop = nothing
        final_te_matrix = nothing
        genome = nothing
        rate2Map = nothing
        population = nothing
        te_matrix = nothing
        GC.gc()
    end
end

println("\n>>> Phase 1 Finished.")
println("    Total Simulations: $counter")
println("    Saved Checkpoints: $saved_count")