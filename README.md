This repository contains the Python codes used for numerical analysis and figure generation for the manuscript:

**"Coexistence of piRNA and KZFP defense systems: Evolutionary dynamics of layered defense against transposable elements"**
by Y. Nabeka, and H. Innan.

## Requirements

The codes are written in Python 3. The following external libraries are required:

* `numpy`
* `scipy`
* `matplotlib`


## Usage

To reproduce the figures, please follow the steps below. Some figures require a **calculation step** (solving ODEs and saving data to `.pkl` files) before plotting.

### 1. Data Generation (Calculation)

Run the following scripts first to perform numerical calculations. These will generate the `.pkl` (pickle) files required for the phase diagrams of the post-invasion equilibrium states.

| Script | Output File (`.pkl`) | Used for |
| :--- | :--- | :--- |
| `f-vs-c.py` | `f-vs-c_*.pkl` | **Figure 1**, **Figure A1** & **Figure A5** |
| `u-vs-s.py` | `u-vs-s_*.pkl` | **Figure A3** |
| `pi-vs-s.py` | `pi-vs-s_*.pkl` | **Figure A4** |

### 2. Figure Generation

After generating the necessary data, run the following scripts to produce the figures.

**Main & Appendix Figures**

| Figure | Description | Command | Pre-requisite |
| :--- | :--- | :--- | :--- |
| **Fig 1** | Post-invasion equilibrium states in parameter space $(f, c)$| `python fig1.py` | Run `f-vs-c.py` |
| **Fig 2** | Invasion threshold analysis | `python fig2.py` | - |
| **Fig 3** | Growth rate of KZFP ($\sigma$) as a function of $p_{qs}$ | `python fig3.py` | - |
| **Fig 4** | TE copy number $(n_{eq})$ vs. piRNA frequency $(p_{eq})$ | `python fig4.py` | - |
| **Fig 5** | Evolutionary dynamics (Time-course) | `python fig5.py` | - |
| **Fig A1** | Stability map | `python figA1.py` | Run `f-vs-c.py` |
| **Fig A2** | Validation of quasi-equilibrium approximation | `python figA2.py` | - |
| **Fig A3** | Post-invasion equilibrium states in parameter space $(u, s)$ | `python figA3.py` | Run `u-vs-s.py` |
| **Fig A4** | Post-invasion equilibrium states in parameter space $(\pi, s)$ | `python figA4.py` | Run `pi-vs-s.py` |
| **Fig A5** | Forward simulation results overlaid on $(f, c)$ space | `python figA5.py` | Run `f-vs-c.py` & Julia simulations |
| **Fig A6** | Trajectories from the forward simulations | `python figA6.py` | Run Julia simulations |

*(Note: For **Fig A5** and **Fig A6**, you must run the forward simulations in the `simulation/` directory to generate the required simulation data before plotting. See `simulation/README.md` for details.)*
