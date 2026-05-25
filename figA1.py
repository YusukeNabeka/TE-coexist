#figA1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pickle
import math

u_fixed  = 0.01
pi_fixed = 0.01
grid = 50

s_fixed  = 0.005
s_fixed1  = 0.0025
s_fixed2  = 0.001
s_fixed3  = 0.00075

filename  = f'f-vs-c_u{u_fixed:.1e}_s{s_fixed:.1e}_pi{pi_fixed:.1e}_newjacobian.pkl'
filename1 = f'f-vs-c_u{u_fixed:.1e}_s{s_fixed1:.1e}_pi{pi_fixed:.1e}_newjacobian.pkl'
filename2 = f'f-vs-c_u{u_fixed:.1e}_s{s_fixed2:.1e}_pi{pi_fixed:.1e}_newjacobian.pkl'
filename3 = f'f-vs-c_u{u_fixed:.1e}_s{s_fixed3:.1e}_pi{pi_fixed:.1e}_newjacobian.pkl'


with open(filename, "rb") as f:
    res = pickle.load(f)
    
with open(filename1, "rb") as f:
    res1 = pickle.load(f)

with open(filename2, "rb") as f:
    res2 = pickle.load(f)

with open(filename3, "rb") as f:
    res3 = pickle.load(f)




def p_star_func(s, u):
    return 1.0 - np.sqrt(s / u)


def n_star_func(s, u, pi):
    ps = p_star_func(s, u)
    denom = pi + 2.0 * s * ps
    n_star = 2.0 * ps * (1.0 - ps) / denom
    return n_star

    
def check_stability_full(s, u):
    return (0 < s < u)

def yaxis_range(s, u, pi):
    ps = p_star_func(s, u)
    ns = n_star_func(s, u, pi)
    if np.isnan(ps) or np.isnan(ns):
        return False

    c_thr_max = s * u * ns * 1.0 * (1.0 - ps)**2
    upper = 0.8634642194720046 + math.log10(c_thr_max)
    lower = upper - 4
    return lower, upper

def check_kzfp_invasion(s, u, pi, f, c):

    ps = p_star_func(s, u)
    ns = n_star_func(s, u, pi)
    if np.isnan(ps) or np.isnan(ns):
        return False

    sigma_ben = s * u * ns * f * (1.0 - ps)**2
    return sigma_ben > c

def sci_label(name, x, digits=2):
    if x == 0:
        return rf'${name}=0$'
    s = f"{x:.{digits}e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    return rf'${name}={mant}\times10^{{{int(exp)}}}$'


fs        = [r["f"] for r in res]
cs         = [r["c"] for r in res]
stab_final = [r["stability"] for r in res]
LOW, UP    = yaxis_range(s_fixed,  u_fixed, pi_fixed)

fs1        = [r["f"] for r in res1]
cs1         = [r["c"] for r in res1]
stab_final1 = [r["stability"] for r in res1]
LOW1, UP1   = yaxis_range(s_fixed1, u_fixed, pi_fixed)

fs2        = [r["f"] for r in res2]
cs2         = [r["c"] for r in res2]
stab_final2 = [r["stability"] for r in res2]
LOW2, UP2   = yaxis_range(s_fixed2, u_fixed, pi_fixed)

fs3        = [r["f"] for r in res3]
cs3         = [r["c"] for r in res3]
stab_final3 = [r["stability"] for r in res3]
LOW3, UP3   = yaxis_range(s_fixed3, u_fixed, pi_fixed)

f_bg  = np.linspace(0, 1, 500)
c_bg   = np.logspace(LOW,  UP,  500)
F_bg,  C_bg  = np.meshgrid(f_bg,  c_bg)
inv_map  = np.vectorize(check_kzfp_invasion)(s_fixed,  u_fixed, pi_fixed, F_bg,  C_bg)

f_bg1 = np.linspace(0, 1, 500)
c_bg1  = np.logspace(LOW1, UP1, 500)
F_bg1, C_bg1 = np.meshgrid(f_bg1, c_bg1)
inv_map1 = np.vectorize(check_kzfp_invasion)(s_fixed1, u_fixed, pi_fixed, F_bg1, C_bg1)

f_bg2 = np.linspace(0, 1, 500)
c_bg2  = np.logspace(LOW2, UP2, 500)
F_bg2, C_bg2 = np.meshgrid(f_bg2, c_bg2)
inv_map2 = np.vectorize(check_kzfp_invasion)(s_fixed2, u_fixed, pi_fixed, F_bg2, C_bg2)

f_bg3 = np.linspace(0, 1, 500)
c_bg3  = np.logspace(LOW3, UP3, 500)
F_bg3, C_bg3 = np.meshgrid(f_bg3, c_bg3)
inv_map3 = np.vectorize(check_kzfp_invasion)(s_fixed3, u_fixed, pi_fixed, F_bg3, C_bg3)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), constrained_layout=True)

cmap_stability = mcolors.ListedColormap(
    ["blue", "cyan", "green", "orange", "red", "dimgray", "magenta"]
)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm   = mcolors.BoundaryNorm(bounds, cmap_stability.N)

ssL        = [s_fixed,  s_fixed1,  s_fixed2,  s_fixed3]
fsL       = [fs,      fs1,      fs2,      fs3]
csL        = [cs,       cs1,       cs2,       cs3]
FL        = [F_bg,    F_bg1,    F_bg2,    F_bg3]
CL         = [C_bg,     C_bg1,     C_bg2,     C_bg3]
invL       = [inv_map,  inv_map1,  inv_map2,  inv_map3]
stabilityL = [stab_final, stab_final1, stab_final2, stab_final3]

for i in [0, 1, 2, 3]:
    axes[i].scatter(
        fsL[i],
        csL[i],
        c=stabilityL[i],
        cmap=cmap_stability,
        norm=norm,
        edgecolor="none",
        marker="s",
        s=35,
    )

    handles = [
        mpatches.Patch(color="blue", label="Stable node"),
        mpatches.Patch(color="cyan",  label="Stable focus"),
    ]

    axes[i].grid(True, which="both", ls="--", alpha=0.4)
    axes[i].contour(
        FL[i],
        CL[i],
        invL[i].astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    axes[i].set_yscale("log")
    axes[i].set_xlabel(r"$f$ (suppression efficacy)", fontsize=13)
    axes[i].set_ylabel(r"$c$ (maintenance cost)", fontsize=13)
    axes[i].set_title(
        sci_label("s", ssL[i], digits=1),
        loc="left",
        fontsize=16,
        pad=6,
    )

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    handles=handles,
    fontsize=13,
)
plt.savefig('figA1.pdf', format="pdf", bbox_inches='tight')
plt.show()