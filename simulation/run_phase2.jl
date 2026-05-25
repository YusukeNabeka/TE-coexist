# run_phase2.jl
# Phase 2: KZFP invasion from TE-piRNA burn-in checkpoints.

# Run this script from the repository root.
# julia run_phase2.jl <global_setting> <phase1_rep> <f_kz> <c_kz> <rep>

include("src/setup.jl")

# Input arguments
global_setting = ARGS[1]
p1_rep = ARGS[2]
p1_file_name = global_setting * "_rep" * p1_rep * ".jld2"

phase1_dir = "./data/phase1/" * global_setting
phase2_dir = "./data/phase2/" * global_setting
if !isdir(phase2_dir)
    mkpath(phase2_dir)
end

f_val = parse(Float64, ARGS[3])
c_val = parse(Float64, ARGS[4])
rep_val = parse(Int, ARGS[5])

p1_file = joinpath(phase1_dir, p1_file_name)

# Phase 2 parameters
f_kz = f_val
c_kz = c_val
rep = rep_val

invasion_generations = 200000
init_kz_freq = 0.2
log_interval_p2 = 10

# Load Phase 1 checkpoint
base_name = splitext(basename(p1_file))[1]
println(">>> Processing Source: $base_name")
@printf("Invading: fkz=%.2f, ckz=%.1e, rep=%d\n", f_kz, c_kz, rep)

data = SimData.load_checkpoint(p1_file)

population       = data.population
te_matrix        = data.te_matrix
genome           = data.genome
rate2Map         = data.rate2Map
pi_coords        = data.pi_coords
kz_locus         = data.kz_locus
current_te_count = data.current_te_count
old_params       = data.params

# Update parameters for Phase 2
params = deepcopy(old_params)

params.generations     = invasion_generations
params.f_kz            = f_kz
params.c_kz            = c_kz
params.initial_kz_freq = init_kz_freq
params.log_interval    = log_interval_p2

# Use a reproducible seed for each parameter combination
hash_val = hash((base_name, f_kz, c_kz, rep))
safe_seed = abs(Int(hash_val % typemax(Int)))

rng = MersenneTwister(safe_seed)
params.seed = safe_seed

# Introduce KZFP alleles
SimKZFP.inject_kzfp!(population, params.initial_kz_freq, rng)

# Prepare log file
log_name = @sprintf("%s-p2_fkz%.2f_ckz%.1e_rep%d.tsv", base_name, f_kz, c_kz, rep)
log_dir_name = @sprintf("f%.2f_c%.1e", f_kz, c_kz)
log_dir = joinpath(phase2_dir, log_dir_name)
if !isdir(log_dir)
    mkpath(log_dir)
end

log_path = joinpath(log_dir, log_name)

open(log_path, "w") do io
    # Log header
    @printf(io, "gen\tte\tpi_freq\tkz_freq\tvar_te\tfano_te\tcov_pi_te\tTE_pi+\tTE_pi-\tdelta_te_pi\tcov_kz_te\tTE_kz+\tTE_kz-\tdelta_te_kz\n")

    # Initial state
    stats0 = SimStats.calculate_statistics(population, te_matrix, genome, pi_coords)

    @printf(io,
        "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
        0,
        stats0.mean_te_copy_number,
        stats0.pi_allele_frequency,
        stats0.kzfp_allele_frequency,
        stats0.var_te_copy_number,
        stats0.fano_te,
        stats0.cov_pi_te,
        stats0.mean_te_pi_present,
        stats0.mean_te_pi_absent,
        stats0.delta_te_pi_present_absent,
        stats0.cov_kz_te,
        stats0.mean_te_kz_present,
        stats0.mean_te_kz_absent,
        stats0.delta_te_kz_present_absent,
    )

    @printf(
        "Gen %d: TE=%.3f pi=%.3f kz=%.3f VarTE=%.3f Fano=%.3f Cov(pi,TE)=%.3f TE[PI>0]=%.3f TE[PI=0]=%.3f dTE[PI>0]-[PI=0]=%.3f Cov(KZ,TE)=%.3f TE[KZ>0]=%.3f TE[KZ=0]=%.3f dTE[KZ>0]-[KZ=0]=%.3f\n",
        0,
        stats0.mean_te_copy_number,
        stats0.pi_allele_frequency,
        stats0.kzfp_allele_frequency,
        stats0.var_te_copy_number,
        stats0.fano_te,
        stats0.cov_pi_te,
        stats0.mean_te_pi_present,
        stats0.mean_te_pi_absent,
        stats0.delta_te_pi_present_absent,
        stats0.cov_kz_te,
        stats0.mean_te_kz_present,
        stats0.mean_te_kz_absent,
        stats0.delta_te_kz_present_absent,
    )

    # Run invasion dynamics
    SimCore.run_generations!(
        population,
        te_matrix,
        genome,
        pi_coords,
        rate2Map,
        params,
        current_te_count,
        kz_locus,
        rng,
        io;
        start_gen=1,
        phase=2
    )
end

# Free memory
population = nothing
te_matrix = nothing
genome = nothing
rate2Map = nothing
pi_coords = nothing
GC.gc()

println("\n>>> Phase 2 Finished.")