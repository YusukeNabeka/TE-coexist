#src/simCore.jl
module SimCore

using Random, Statistics, Printf, StatsBase
import ..SimTransOpt, ..SimFit, ..SimRec, ..SimStats, ..SimPop

function run_generations!(
    population, 
    te_matrix, 
    genome, 
    pi_coords, 
    rate2Map,
    params, 
    current_te_count, 
    kz_locus, 
    rng, 
    io_log = nothing; 
    start_gen = 1,
    phase=1
)
    N = params.N
    end_gen = start_gen + params.generations - 1

    fix_count=0
    extinct_count=0

    for gen in start_gen:end_gen
        
        # 1. Transposition
        te_matrix = SimTransOpt.ensure_capacity!(te_matrix, current_te_count, N * 5)
        for i in 1:N
            ind = population[i]
            step_seed = rand(rng, Int)
            ind.v1, ind.v2, current_te_count, te_matrix = SimTransOpt.transposition!(
                te_matrix, genome, ind.v1, ind.v2,
                params.u, 1.0, 1.0,
                ind.kz1, ind.kz2, params.f_kz, kz_locus,
                current_te_count, step_seed
            )
        end
        
        # 2. Fitness calculation
        weights = Vector{Float64}(undef, N)
        for i in 1:N
            ind = population[i]
            w = SimFit.calculateFitness(te_matrix, ind.v1, ind.v2;
                kz1=ind.kz1, kz2=ind.kz2,
                c_kz=params.c_kz, kz_cost_mode=params.kz_cost_mode,
                fitnessFunction=params.fitnessFunction,
                epistasisCoefficient=params.epistasisCoefficient)
            ind.w = w
            weights[i] = w
        end
        
        # 3. Gamete selection / Recombination
        total_w = sum(weights)
        probs = (total_w > 0) ? weights ./ total_w : fill(1.0/N, N)
        parents = sample(rng, 1:N, Weights(probs), 2 * N)
        
        next_pop = Vector{eltype(population)}(undef, N)
        for i in 1:N
            p1, p2 = population[parents[2i-1]], population[parents[2i]]
            
            gv1, gkz1 = SimRec.recombination(rate2Map, te_matrix, p1.v1, p1.v2, p1.kz1, p1.kz2, kz_locus)
            gv2, gkz2 = SimRec.recombination(rate2Map, te_matrix, p2.v1, p2.v2, p2.kz1, p2.kz2, kz_locus)
            
            next_pop[i] = SimPop.Individual(gv1, gv2, gkz1, gkz2, 1.0)
        end
        population = next_pop
        
        # 4. Summary Statistics
        if !isnothing(io_log) && (gen % params.log_interval == 0)

            s = SimStats.calculate_statistics(population, te_matrix, genome, pi_coords)
            
            if phase==1
                @printf(io_log, "%d\t%.4f\t%.4f\t%.4f\n", gen, s.mean_te_copy_number, s.pi_allele_frequency, s.kzfp_allele_frequency)
                flush(io_log)
                @printf("Gen %d: TE=%.2f pi=%.3f kz=%.3f\n", gen, s.mean_te_copy_number, s.pi_allele_frequency, s.kzfp_allele_frequency)
            
            
            elseif phase==2
                @printf(io_log,
                    "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                    gen,
                    s.mean_te_copy_number,
                    s.pi_allele_frequency,
                    s.kzfp_allele_frequency,
                    s.var_te_copy_number,
                    s.fano_te,
                    s.cov_pi_te,
                    s.mean_te_pi_present,
                    s.mean_te_pi_absent,
                    s.delta_te_pi_present_absent,
                    s.cov_kz_te,
                    s.mean_te_kz_present,
                    s.mean_te_kz_absent,
                    s.delta_te_kz_present_absent,
                )
                flush(io_log)
                @printf(
                    "Gen %d: TE=%.3f pi=%.3f kz=%.3f VarTE=%.3f Fano=%.3f Cov(pi,TE)=%.3f TE[PI>0]=%.3f TE[PI=0]=%.3f dTE[PI>0]-[PI=0]=%.3f Cov(KZ,TE)=%.3f TE[KZ>0]=%.3f TE[KZ=0]=%.3f dTE[KZ>0]-[KZ=0]=%.3f\n",
                    gen,
                    s.mean_te_copy_number,
                    s.pi_allele_frequency,
                    s.kzfp_allele_frequency,
                    s.var_te_copy_number,
                    s.fano_te,
                    s.cov_pi_te,
                    s.mean_te_pi_present,
                    s.mean_te_pi_absent,
                    s.delta_te_pi_present_absent,
                    s.cov_kz_te,
                    s.mean_te_kz_present,
                    s.mean_te_kz_absent,
                    s.delta_te_kz_present_absent,
                )
            end
            
            if s.kzfp_allele_frequency == 1.0 && phase==2
                fix_count+=1
                if fix_count ==100
                    break
                end
            end

            if s.kzfp_allele_frequency == 0.0 && phase==2
                extinct_count+=1
                if extinct_count ==5
                    break
                end
            end

            if s.mean_te_copy_number == 0.0 && phase == 2
                break
            end

        end
        
        if gen % 100 == 0; GC.gc(); end
    end
    
    return population, te_matrix, current_te_count
end

end