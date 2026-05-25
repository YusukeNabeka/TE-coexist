# Forward simulations

This directory contains Julia code for forward individual-based simulations of TE, piRNA, and KZFP dynamics.

The simulations are divided into two phases:

1. **Phase 1: TE-piRNA burn-in**  
   Populations are evolved without KZFP until a TE-piRNA equilibrium is reached.

2. **Phase 2: KZFP invasion**  
   KZFP alleles are introduced into the burn-in populations, and their invasion dynamics are simulated.

## Directory structure

```text
.
├── src/
│   ├── setup.jl
│   ├── generateGenome.jl
│   ├── generatePopulation.jl
│   ├── ...
├── run_phase1.jl
├── run_phase2.jl
├── data/
│   ├── phase1/
│   └── phase2/
└── README.md
```

## Requirements

The simulations are written in Julia.

Recommended Julia version:

```text
Julia v1.12.4 or later
```

Required Julia packages include:

```julia
Random
Distributions
DataFrames
CSV
Statistics
StatsBase
JLD2
Printf
```

Please install any missing packages using Julia's package manager.


## Usage

The simulation consists of two phases: reaching the pre-invasion TE-piRNA equilibrium and simulating the post-invasion dynamics after introducing the KZFP allele.

All scripts should be run from the repository root.

### 1. Pre-invasion equilibrium: Phase 1

Run this script to simulate the population up to the TE-piRNA equilibrium before KZFP invasion:

```bash
julia run_phase1.jl
```

This generates the initial equilibrium population data required for Phase 2.

The output files are written to:

```text
data/phase1/
```

### 2. Post-invasion dynamics: Phase 2

Run this script to introduce the KZFP allele and simulate the evolutionary dynamics while tracking genetic drift and linkage disequilibrium:

```bash
julia run_phase2.jl <global_setting> <phase1_rep> <f_kz> <c_kz> <rep>
```

Example:

```bash
julia run_phase2.jl N10000_p1_u0.01_s0.005_pi0.01_r0.499 1 0.75 1e-5 1
```

Arguments:

```text
<global_setting>   Base name of the Phase 1 parameter setting
<phase1_rep>       Phase 1 replicate ID
<f_kz>             KZFP suppression efficacy
<c_kz>             KZFP maintenance cost
<rep>              Phase 2 replicate ID
```

The output files are written to:

```text
data/phase2/
```

## Output

The simulations output trajectory files containing:

- TE copy number
- piRNA allele frequency
- KZFP allele frequency
- variance in TE copy number
- Fano factor of TE copy number
- covariance between piRNA allele dosage and TE copy number
- covariance between KZFP allele dosage and TE copy number

These output files are subsequently read by `figA5.py` and `figA6.py` in the parent directory for visualization.

## Source code

The core simulation logic is modularized in the `src/` directory.

### `setup.jl`

Loads required packages, defines shared parameters, and imports simulation modules.

### `generateGenome.jl`

Generates the genome, including explicit chromosomes, potential TE insertion sites, piRNA cluster locations, recombination rates, and the KZFP locus.

### `generatePopulation.jl`

Defines the Individual struct and initializes the starting host population. It distributes initial TE insertions across valid genomic sites and introduces the KZFP alleles at a specified initial frequency.

### `simCore.jl`

Runs the main life cycle of the simulated population across generations. It sequentially coordinates TE transposition, individual fitness calculations, gamete selection, recombination, and the logging of summary statistics.

### `simKZFP.jl`
Handles the stochastic introduction (injection) of the KZFP suppressor allele into a target fraction of the population at the beginning of Phase 2.

### `transposition.jl`

Implements stochastic TE transposition events.

### `recombination.jl`

Implements chromosomal crossover, recombination, and gamete formation.

### `regulation.jl`

Implements piRNA- and KZFP-mediated TE suppression.

### `fitness.jl`

Calculates individual fitness.

### `stats.jl`

Calculates population-level summary statistics, including TE copy number, allele frequencies, variance, Fano factor, and covariance.

### `simData.jl`

Handles checkpoint saving and loading.
