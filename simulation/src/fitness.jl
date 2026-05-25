#src/fitness.jl
module SimFit

using Statistics

function calculateFitness(
    transposonMatrix::Matrix{Float64}, 
    v1::Vector{Int}, 
    v2::Vector{Int};
    kz1::Bool = false,
    kz2::Bool = false,
    c_kz::Float64 = 0.0,
    kz_cost_mode::Symbol = :dominant, # :dominant, :additive, :multiplicative
    
    fitnessFunction::Int = 1,
    epistasisCoefficient::Float64 = 0.0
)
    
    # 1. TE (w_te)
    te_ids = vcat(v1, v2)
    w_te = 1.0
    
    if !isempty(te_ids)
        penalties = view(transposonMatrix, te_ids, 3)
        
        if fitnessFunction == 1
            # Multiplicative model: w = exp(sum(s))
            w_te = exp(sum(penalties))
            
        elseif fitnessFunction == 2
            # Log-quadratic model
            s_representative = abs(penalties[1])
            n = length(penalties)
        
            # linear term:
            term_linear = sum(penalties)
        
            # quadratic term:
            term_quad = - 0.5 * epistasisCoefficient * (n^2)
        
            w_te = exp(term_linear + term_quad)
        else
            error("Unknown fitnessFunction ID: $fitnessFunction")
        end
    end

    # 2. KZFP (w_kz)
    w_kz = 1.0
    
    if c_kz > 0.0
        # KZFP allele dosage k (0, 1, 2)
        k_count = (kz1 ? 1 : 0) + (kz2 ? 1 : 0)
        
        if k_count > 0
            if kz_cost_mode == :dominant
                w_kz = 1.0 - c_kz
                
            elseif kz_cost_mode == :additive
                w_kz = 1.0 - (k_count * c_kz)
                
            elseif kz_cost_mode == :multiplicative
                w_kz = (1.0 - c_kz) ^ k_count
            else
                w_kz = 1.0 - c_kz
            end
            
            if w_kz < 0.0
                w_kz = 0.0
            end
        end
    end


    return w_te * w_kz
end

end # module