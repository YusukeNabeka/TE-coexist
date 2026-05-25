#src/stats.jl
module SimStats

using Statistics

export PopulationStats, calculate_statistics

struct PopulationStats
    mean_te_copy_number::Float64
    mean_pi_te_copy_number::Float64
    pi_allele_frequency::Float64
    kzfp_allele_frequency::Float64
    var_te_copy_number::Float64
    fano_te::Float64
    cov_kz_te::Float64
    cov_pi_te::Float64
    delta_te_pi_present_absent::Float64
    mean_te_pi_present::Float64
    mean_te_pi_absent::Float64
    delta_te_kz_present_absent::Float64
    mean_te_kz_present::Float64
    mean_te_kz_absent::Float64
end

function calculate_statistics(population, transposonMatrix, genomeMatrix, piRNAcoord)
    N = length(population)
    if N == 0
        return PopulationStats(
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            NaN, NaN, NaN,
            NaN, NaN, NaN
        )
    end

    total_te_count = 0
    total_pi_te_count = 0
    total_occupied_slots = 0
    total_kz_alleles = 0

    te_counts = Float64[]              # total TE
    te_noncluster_counts = Float64[]   # non-cluster TE

    kz_dosage = Float64[]
    pi_dosage = Float64[]

    te_when_pi_present = Float64[]     # non-cluster TE
    te_when_pi_absent = Float64[]      # non-cluster TE

    te_when_kz_present = Float64[]     # total TE
    te_when_kz_absent = Float64[]      # total TE

    num_clusters = length(piRNAcoord)
    clusters = collect(values(piRNAcoord))

    for ind in population
        kz_i = (ind.kz1 ? 1 : 0) + (ind.kz2 ? 1 : 0)

        te_i_total = length(ind.v1) + length(ind.v2)
        te_i_cluster = 0
        pi_i = 0

        if ind.kz1
            total_kz_alleles += 1
        end
        if ind.kz2
            total_kz_alleles += 1
        end

        for hap_te_ids in (ind.v1, ind.v2)
            total_te_count += length(hap_te_ids)

            if isempty(hap_te_ids)
                continue
            end

            sites = view(transposonMatrix, hap_te_ids, 2)
            hap_sites = [round(Int, s) for s in sites]

            # Count cluster TE copies
            for s in hap_sites
                if genomeMatrix[s, 4] > 0.0
                    total_pi_te_count += 1
                    te_i_cluster += 1
                end
            end

            # Count occupied piRNA clusters as piRNA dosage
            if num_clusters > 0
                for (c_start, c_end) in clusters
                    is_occupied = false

                    for s in hap_sites
                        if c_start <= s <= c_end
                            is_occupied = true
                            break
                        end
                    end

                    if is_occupied
                        total_occupied_slots += 1
                        pi_i += 1
                    end
                end
            end
        end

        te_i_noncluster = te_i_total - te_i_cluster

        push!(te_counts, Float64(te_i_total))
        push!(te_noncluster_counts, Float64(te_i_noncluster))

        push!(kz_dosage, Float64(kz_i))
        push!(pi_dosage, Float64(pi_i))

        
        if pi_i > 0
            push!(te_when_pi_present, Float64(te_i_noncluster))
        else
            push!(te_when_pi_absent, Float64(te_i_noncluster))
        end


        if kz_i > 0
            push!(te_when_kz_present, Float64(te_i_noncluster))
        else
            push!(te_when_kz_absent, Float64(te_i_noncluster))
        end
    end

    mean_te = mean(te_counts)
    mean_pi_copy = total_pi_te_count / N

    pi_freq = num_clusters > 0 ? total_occupied_slots / (2 * N * num_clusters) : 0.0
    kz_freq = total_kz_alleles / (2 * N)

    var_te = var(te_counts; corrected=false)
    fano_te = mean_te > 0 ? var_te / mean_te : 0.0

    cov_kz_te = cov(kz_dosage, te_counts; corrected=false)
    cov_pi_te = cov(pi_dosage, te_noncluster_counts; corrected=false)

    mean_te_pi_present = isempty(te_when_pi_present) ? NaN : mean(te_when_pi_present)
    mean_te_pi_absent = isempty(te_when_pi_absent) ? NaN : mean(te_when_pi_absent)
    delta_te_pi = (isnan(mean_te_pi_present) || isnan(mean_te_pi_absent)) ? NaN :
                  (mean_te_pi_present - mean_te_pi_absent)

    mean_te_kz_present = isempty(te_when_kz_present) ? NaN : mean(te_when_kz_present)
    mean_te_kz_absent = isempty(te_when_kz_absent) ? NaN : mean(te_when_kz_absent)
    delta_te_kz = (isnan(mean_te_kz_present) || isnan(mean_te_kz_absent)) ? NaN :
                  (mean_te_kz_present - mean_te_kz_absent)

    return PopulationStats(
        mean_te,
        mean_pi_copy,
        pi_freq,
        kz_freq,
        var_te,
        fano_te,
        cov_kz_te,
        cov_pi_te,
        delta_te_pi,
        mean_te_pi_present,
        mean_te_pi_absent,
        delta_te_kz,
        mean_te_kz_present,
        mean_te_kz_absent
    )
end

end # module
