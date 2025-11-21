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

filename  = f'f1-vs-c_u{u_fixed:.1e}_s{s_fixed:.1e}_pi{pi_fixed:.1e}.pkl'
filename1 = f'f1-vs-c_u{u_fixed:.1e}_s{s_fixed1:.1e}_pi{pi_fixed:.1e}.pkl'
filename2 = f'f1-vs-c_u{u_fixed:.1e}_s{s_fixed2:.1e}_pi{pi_fixed:.1e}.pkl'
filename3 = f'f1-vs-c_u{u_fixed:.1e}_s{s_fixed3:.1e}_pi{pi_fixed:.1e}.pkl'


with open(filename, "rb") as f:
    res = pickle.load(f)
    
with open(filename1, "rb") as f:
    res1 = pickle.load(f)

with open(filename2, "rb") as f:
    res2 = pickle.load(f)

with open(filename3, "rb") as f:
    res3 = pickle.load(f)




def p_star_func(s, u):
    if not (0 < s < u):
        return np.nan
    return 1.0 - np.sqrt(s / u)


def check_stability_full(s, u):
    if not (0 < s < u):
        return False
    ps = p_star_func(s, u)
    if np.isnan(ps):
        return False
    return (1.0 - 2.0 * s * ps > 1e-9) and (1.0 - 2.0 * s * (ps**2) > 1e-9)


def yaxis_range(s, u, pi):
    ps = p_star_func(s, u)
    denom = pi * np.sqrt(u * s + 1e-22) * (1.0 - 2.0 * s * ps)
    if abs(denom) < 1e-22:
        return False
    upper = 0.8634642194720046 + math.log10((2.0 * (s**3) * ps) / denom)
    lower = upper - 4.0
    return lower, upper


def check_kzfp_invasion(s, u, pi, fm, c):
    if not check_stability_full(s, u):
        return False
    ps = p_star_func(s, u)
    denom = pi * np.sqrt(u * s + 1e-22) * (1.0 - 2.0 * s * ps)
    if abs(denom) < 1e-22:
        return False
    return (2.0 * (s**3) * ps * fm) / denom > c


def sci_label(name, x, digits=2):
    if x == 0:
        return rf'${name}=0$'
    s = f"{x:.{digits}e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    return rf'${name}={mant}\times10^{{{int(exp)}}}$'


fms        = [r["fm"] for r in res]
cs         = [r["c"] for r in res]
stab_final = [r["stability"] for r in res]
LOW, UP    = yaxis_range(s_fixed,  u_fixed, pi_fixed)

fms1        = [r["fm"] for r in res1]
cs1         = [r["c"] for r in res1]
stab_final1 = [r["stability"] for r in res1]
LOW1, UP1   = yaxis_range(s_fixed1, u_fixed, pi_fixed)

fms2        = [r["fm"] for r in res2]
cs2         = [r["c"] for r in res2]
stab_final2 = [r["stability"] for r in res2]
LOW2, UP2   = yaxis_range(s_fixed2, u_fixed, pi_fixed)

fms3        = [r["fm"] for r in res3]
cs3         = [r["c"] for r in res3]
stab_final3 = [r["stability"] for r in res3]
LOW3, UP3   = yaxis_range(s_fixed3, u_fixed, pi_fixed)

fm_bg  = np.linspace(0, 1, 500)
c_bg   = np.logspace(LOW,  UP,  500)
FM_bg,  C_bg  = np.meshgrid(fm_bg,  c_bg)
inv_map  = np.vectorize(check_kzfp_invasion)(s_fixed,  u_fixed, pi_fixed, FM_bg,  C_bg)

fm_bg1 = np.linspace(0, 1, 500)
c_bg1  = np.logspace(LOW1, UP1, 500)
FM_bg1, C_bg1 = np.meshgrid(fm_bg1, c_bg1)
inv_map1 = np.vectorize(check_kzfp_invasion)(s_fixed1, u_fixed, pi_fixed, FM_bg1, C_bg1)

fm_bg2 = np.linspace(0, 1, 500)
c_bg2  = np.logspace(LOW2, UP2, 500)
FM_bg2, C_bg2 = np.meshgrid(fm_bg2, c_bg2)
inv_map2 = np.vectorize(check_kzfp_invasion)(s_fixed2, u_fixed, pi_fixed, FM_bg2, C_bg2)

fm_bg3 = np.linspace(0, 1, 500)
c_bg3  = np.logspace(LOW3, UP3, 500)
FM_bg3, C_bg3 = np.meshgrid(fm_bg3, c_bg3)
inv_map3 = np.vectorize(check_kzfp_invasion)(s_fixed3, u_fixed, pi_fixed, FM_bg3, C_bg3)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), constrained_layout=True)

cmap_stability = mcolors.ListedColormap(
    ["blue", "cyan", "green", "orange", "red", "dimgray", "magenta"]
)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm   = mcolors.BoundaryNorm(bounds, cmap_stability.N)

ssL        = [s_fixed,  s_fixed1,  s_fixed2,  s_fixed3]
fmsL       = [fms,      fms1,      fms2,      fms3]
csL        = [cs,       cs1,       cs2,       cs3]
FML        = [FM_bg,    FM_bg1,    FM_bg2,    FM_bg3]
CL         = [C_bg,     C_bg1,     C_bg2,     C_bg3]
invL       = [inv_map,  inv_map1,  inv_map2,  inv_map3]
stabilityL = [stab_final, stab_final1, stab_final2, stab_final3]

for i in [0, 1, 2, 3]:
    axes[i].scatter(
        fmsL[i],
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
        mpatches.Patch(color="cyan",  label="Stable spiral"),
    ]

    axes[i].grid(True, which="both", ls="--", alpha=0.4)
    axes[i].contour(
        FML[i],
        CL[i],
        invL[i].astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    axes[i].set_yscale("log")
    axes[i].set_xlabel(r"$f_1$ (suppression efficacy)", fontsize=13)
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

plt.show()
