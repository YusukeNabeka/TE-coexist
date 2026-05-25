#src/setup_LD.jl

using Random
using Statistics
using StatsBase
using Printf
using Dates
using JLD2
using DelimitedFiles

include("generateGenome.jl")
include("generatePopulation.jl")
include("regulation.jl")
include("fitness.jl")
include("transposition.jl")
include("recombination.jl")
include("stats.jl")
include("simData.jl")
include("simKZFP.jl")
include("simCore.jl")

using .SimGen
using .SimPop
using .SimReg
using .SimFit
using .SimTransOpt
using .SimRec
using .SimStats
using .SimData
using .SimKZFP
using .SimCore


Base.@kwdef mutable struct Params
    # --- base settings ---
    N::Int                  = 1000
    generations::Int        = 5000
    L::Int                  = 10000
    chroms::Int             = 30
    r::Float64              = 0.499
    init_te_num::Int        = 10
    init_te_freq::Float64   = 0.1
    
    # --- TE parameter ---
    u::Float64              = 0.1
    s::Float64              = 0.01
    
    # --- piRNA parameter ---
    pi_ratio::Float64       = 0.03
    num_clusters::Int       = 1
    pi_s::Bool              = true
    pi_rec::Bool            = false
    f_pi::Float64           = 1.0
    
    # --- KZFP parameter ---
    kz_chrom::Int           = 2
    f_kz::Float64           = 0.0
    c_kz::Float64           = 0.0
    initial_kz_freq::Float64= 0.0
    kz_cost_mode::Symbol    = :dominant
    
    # --- system ---
    seed::Int               = abs(rand(Int))
    output_file::String     = ""
    log_interval::Int       = 10

    # --- fitness function ---
    fitnessFunction::Int = 1
    epistasisCoefficient::Float64 = 0.0
end