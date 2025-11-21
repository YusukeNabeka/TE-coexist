import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.transforms import blended_transform_factory
import math

u_fixed = 0.01
pi_fixed = 0.01

s_fixed = 0.005
s_fixed1 = 0.0025
s_fixed2 = 0.001
s_fixed3 = 0.00075

filename = f"f1-vs-c_u{u_fixed:.1e}_s{s_fixed:.1e}_pi{pi_fixed:.1e}.pkl"
filename1 = f"f1-vs-c_u{u_fixed:.1e}_s{s_fixed1:.1e}_pi{pi_fixed:.1e}.pkl"
filename2 = f"f1-vs-c_u{u_fixed:.1e}_s{s_fixed2:.1e}_pi{pi_fixed:.1e}.pkl"
filename3 = f"f1-vs-c_u{u_fixed:.1e}_s{s_fixed3:.1e}_pi{pi_fixed:.1e}.pkl"

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
    return 1 - np.sqrt(s / u)


def n_star_func(s, u, pi):
    ps = p_star_func(s, u)
    if np.isnan(ps):
        return np.nan
    denom = 1 - 2 * s * ps
    if denom <= 1e-9:
        return np.nan
    n_star = 2 * s * ps / (pi * np.sqrt(u * s + 1e-22) * denom)
    return n_star if n_star > 0 else np.nan


def check_stability_full(s, u):
    if not (0 < s < u):
        return False
    ps = p_star_func(s, u)
    if np.isnan(ps):
        return False
    return (1 - 2 * s * ps > 1e-9) and (1 - 2 * s * (ps**2) > 1e-9)


def yaxis_range(s, u, pi):
    ps = p_star_func(s, u)
    denom = pi * np.sqrt(u * s + 1e-22) * (1 - 2 * s * ps)
    if abs(denom) < 1e-22:
        return False
    upper = 0.8634642194720046 + math.log10((2 * (s**3) * ps * 1) / denom)
    lower = upper - 4
    return lower, upper


def check_kzfp_invasion(s, u, pi, fm, c):
    if not check_stability_full(s, u):
        return False
    ps = p_star_func(s, u)
    denom = pi * np.sqrt(u * s + 1e-22) * (1 - 2 * s * ps)
    if abs(denom) < 1e-22:
        return False
    return (2 * (s**3) * ps * fm) / denom > c


def sigma_y1_zero_curve(
    s, u, pi, low_log10=None, up_log10=None, npts=600, mask_invasion=True
):
    f1_max = max(1e-8, 1.0 - s / u)
    if f1_max <= 1e-6:
        return np.array([]), np.array([])

    f1_line = np.linspace(1e-6, f1_max - 1e-6, npts)

    y1 = np.sqrt(s / (u * (1.0 - f1_line)))
    denom = 1.0 - 2.0 * s + 2.0 * s * y1
    num = y1**3 * (1.0 - y1)

    c_line = (2.0 * s * u * f1_line / pi) * (num / denom)

    if mask_invasion and f1_line.size:
        valid = ~np.isnan(c_line)
        if np.any(valid):
            inv_ok = np.zeros_like(c_line, dtype=bool)
            idx = np.where(valid)[0]
            for i in idx:
                inv_ok[i] = check_kzfp_invasion(
                    s, u, pi, float(f1_line[i]), float(c_line[i])
                )
            c_line[~inv_ok] = np.nan

    return f1_line, c_line


def pc_and_nmax_levels(s, pi):
    if 2.0 * s >= 1.0:
        return np.nan, np.nan
    p_c = (1.0 - np.sqrt(1.0 - 2.0 * s)) / (2.0 * s)
    den = 1.0 - 2.0 * s * p_c
    if den <= 0:
        return p_c, np.nan
    n_max = (2.0 / pi) * (p_c * (1.0 - p_c) / den)
    return p_c, n_max


def overlay_pc_and_nmax(
    ax_p,
    ax_n,
    fms,
    cs,
    p_vals,
    n_vals,
    s,
    pi,
    color_pc="cyan",
    color_nmax="cyan",
):
    p_c, n_max = pc_and_nmax_levels(s, pi)

    if np.isfinite(p_c):
        mask_p = np.isfinite(fms) & np.isfinite(cs) & np.isfinite(p_vals)
        if np.count_nonzero(mask_p) > 3:
            ax_p.tricontour(
                np.array(fms)[mask_p],
                np.array(cs)[mask_p],
                np.array(p_vals)[mask_p],
                levels=[p_c],
                colors=color_pc,
                linewidths=3,
                linestyles=":",
            )

    if np.isfinite(n_max):
        mask_n = np.isfinite(fms) & np.isfinite(cs) & np.isfinite(n_vals)
        if np.count_nonzero(mask_n) > 3:
            ax_n.tricontour(
                np.array(fms)[mask_p],
                np.array(cs)[mask_p],
                np.array(p_vals)[mask_p],
                levels=[p_c],
                colors=color_pc,
                linewidths=3,
                linestyles=":",
            )


fms = [r["fm"] for r in res]
cs = [r["c"] for r in res]
xm_final = [r["xm"] for r in res]
p_final = [r["p"] for r in res]
n_final = [r["n"] for r in res]

LOW, UP = yaxis_range(s_fixed, u_fixed, pi_fixed)

fms1 = [r["fm"] for r in res1]
cs1 = [r["c"] for r in res1]
xm_final1 = [r["xm"] for r in res1]
p_final1 = [r["p"] for r in res1]
n_final1 = [r["n"] for r in res1]

LOW1, UP1 = yaxis_range(s_fixed1, u_fixed, pi_fixed)

fms2 = [r["fm"] for r in res2]
cs2 = [r["c"] for r in res2]
xm_final2 = [r["xm"] for r in res2]
p_final2 = [r["p"] for r in res2]
n_final2 = [r["n"] for r in res2]

LOW2, UP2 = yaxis_range(s_fixed2, u_fixed, pi_fixed)

fms3 = [r["fm"] for r in res3]
cs3 = [r["c"] for r in res3]
xm_final3 = [r["xm"] for r in res3]
p_final3 = [r["p"] for r in res3]
n_final3 = [r["n"] for r in res3]

LOW3, UP3 = yaxis_range(s_fixed3, u_fixed, pi_fixed)

fig = plt.figure(figsize=(15, 13.5), constrained_layout=False)
mosaic = """
    AABBCC
    AABBCC
    AABBCC
    AABBCC
    AABBCC
    112233
    112233
    112233
    112233
    112233
    445566
    445566
    445566
    445566
    445566
    778899
    778899
    778899
    778899
    778899
"""
ax_dict = fig.subplot_mosaic(mosaic)
ax_main = [ax_dict["A"], ax_dict["B"], ax_dict["C"]]
ax123 = [ax_dict["1"], ax_dict["2"], ax_dict["3"]]
ax456 = [ax_dict["4"], ax_dict["5"], ax_dict["6"]]
ax789 = [ax_dict["7"], ax_dict["8"], ax_dict["9"]]

fm_bg = np.linspace(0, 1, 500)
c_bg = np.logspace(LOW, UP, 500)
FM_bg, C_bg = np.meshgrid(fm_bg, c_bg)
inv_map = np.vectorize(check_kzfp_invasion)(s_fixed, u_fixed, pi_fixed, FM_bg, C_bg)

fm_bg1 = np.linspace(0, 1, 500)
c_bg1 = np.logspace(LOW1, UP1, 500)
FM_bg1, C_bg1 = np.meshgrid(fm_bg1, c_bg1)
inv_map1 = np.vectorize(check_kzfp_invasion)(s_fixed1, u_fixed, pi_fixed, FM_bg1, C_bg1)

fm_bg2 = np.linspace(0, 1, 500)
c_bg2 = np.logspace(LOW2, UP2, 500)
FM_bg2, C_bg2 = np.meshgrid(fm_bg2, c_bg2)
inv_map2 = np.vectorize(check_kzfp_invasion)(s_fixed2, u_fixed, pi_fixed, FM_bg2, C_bg2)

fm_bg3 = np.linspace(0, 1, 500)
c_bg3 = np.logspace(LOW3, UP3, 500)
FM_bg3, C_bg3 = np.meshgrid(fm_bg3, c_bg3)
inv_map3 = np.vectorize(check_kzfp_invasion)(s_fixed3, u_fixed, pi_fixed, FM_bg3, C_bg3)

for ax in ax123:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg1,
        C_bg1,
        inv_map1.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")

sc1 = ax_dict["1"].scatter(
    fms1,
    cs1,
    c=xm_final1,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["1"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["2"].scatter(
    fms1,
    cs1,
    c=p_final1,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["2"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

p_star = p_star_func(s_fixed1, u_fixed)
pc, nm = pc_and_nmax_levels(s_fixed1, pi_fixed)
if np.isfinite(p_star) and 0 <= p_star <= 1:
    pcax = cbar2.ax
    trans = blended_transform_factory(pcax.transAxes, pcax.transData)
    pcax.annotate(
        "$p^*$",
        color="r",
        xy=(1.1, p_star),
        xycoords=trans,
        xytext=(1.8, p_star + 0.32),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    pcax.annotate(
        "$p_{n\\max}$",
        color="c",
        xy=(1.1, pc),
        xycoords=trans,
        xytext=(2.5, pc + 0.17),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

sc3 = ax_dict["3"].scatter(
    fms1,
    cs1,
    c=n_final1,
    cmap="magma",
    vmin=0,
    vmax=50,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar3 = fig.colorbar(
    sc3,
    ax=ax_dict["3"],
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)

n_star = n_star_func(s_fixed1, u_fixed, pi_fixed)
if np.isfinite(n_star) and 0 <= n_star:
    ncax = cbar3.ax
    trans = blended_transform_factory(ncax.transAxes, ncax.transData)
    ncax.annotate(
        "$n^*$",
        color="r",
        xy=(4, n_star),
        xycoords=trans,
        xytext=(6.5, n_star - 7),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    ncax.annotate(
        "$n_{\\max}$",
        color="c",
        xy=(4, nm),
        xycoords=trans,
        xytext=(6.5, nm),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

for ax in ax456:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg2,
        C_bg2,
        inv_map2.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")

sc1 = ax_dict["4"].scatter(
    fms2,
    cs2,
    c=xm_final2,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["4"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["5"].scatter(
    fms2,
    cs2,
    c=p_final2,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["5"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

p_star = p_star_func(s_fixed2, u_fixed)
pc, nm = pc_and_nmax_levels(s_fixed2, pi_fixed)
if np.isfinite(p_star) and 0 <= p_star <= 1:
    pcax = cbar2.ax
    trans = blended_transform_factory(pcax.transAxes, pcax.transData)
    pcax.annotate(
        "$p^*$",
        color="r",
        xy=(1.1, p_star),
        xycoords=trans,
        xytext=(3.2, p_star + 0.19),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    pcax.annotate(
        "$p_{n\\max}$",
        color="c",
        xy=(1.1, pc),
        xycoords=trans,
        xytext=(2.2, pc + 0.19),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

sc3 = ax_dict["6"].scatter(
    fms2,
    cs2,
    c=n_final2,
    cmap="magma",
    vmin=0,
    vmax=50,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar3 = fig.colorbar(
    sc3,
    ax=ax_dict["6"],
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)

n_star = n_star_func(s_fixed2, u_fixed, pi_fixed)
if np.isfinite(n_star) and 0 <= n_star:
    ncax = cbar3.ax
    trans = blended_transform_factory(ncax.transAxes, ncax.transData)
    ncax.annotate(
        "$n^*$",
        color="r",
        xy=(4, n_star),
        xycoords=trans,
        xytext=(6.5, n_star),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    ncax.annotate(
        "$n_{\\max}$",
        color="c",
        xy=(4, nm),
        xycoords=trans,
        xytext=(6.5, nm),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

for ax in ax789:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg3,
        C_bg3,
        inv_map3.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")

sc1 = ax_dict["7"].scatter(
    fms3,
    cs3,
    c=xm_final3,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["7"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["8"].scatter(
    fms3,
    cs3,
    c=p_final3,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["8"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

p_star = p_star_func(s_fixed3, u_fixed)
pc, nm = pc_and_nmax_levels(s_fixed3, pi_fixed)
if np.isfinite(p_star) and 0 <= p_star <= 1:
    pcax = cbar2.ax
    trans = blended_transform_factory(pcax.transAxes, pcax.transData)
    pcax.annotate(
        "$p^*$",
        color="r",
        xy=(1.1, p_star),
        xycoords=trans,
        xytext=(3.5, p_star + 0.15),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    pcax.annotate(
        "$p_{n\\max}$",
        color="c",
        xy=(1.1, pc),
        xycoords=trans,
        xytext=(2.2, pc + 0.19),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

sc3 = ax_dict["9"].scatter(
    fms3,
    cs3,
    c=n_final3,
    cmap="magma",
    vmin=0,
    vmax=50,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar3 = fig.colorbar(
    sc3,
    ax=ax_dict["9"],
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)

n_star = n_star_func(s_fixed3, u_fixed, pi_fixed)
if np.isfinite(n_star) and 0 <= n_star:
    ncax = cbar3.ax
    trans = blended_transform_factory(ncax.transAxes, ncax.transData)
    ncax.annotate(
        "$n^*$",
        color="r",
        xy=(4, n_star),
        xycoords=trans,
        xytext=(6.5, n_star),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    ncax.annotate(
        "$n_{\\max}$",
        color="c",
        xy=(4, nm),
        xycoords=trans,
        xytext=(6.5, nm),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

sc1 = ax_dict["A"].scatter(
    fms,
    cs,
    c=xm_final,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["A"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)
ax_dict["A"].set_title(
    "$x_{1eq}$ \n(Equilibrium KZFP Frequency)",
    fontsize=18,
    y=1.15,
)

sc2 = ax_dict["B"].scatter(
    fms,
    cs,
    c=p_final,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["B"], ticks=[0, 0.5, 1], pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)
ax_dict["B"].set_title(
    "$p_{eq}$ \n(Equilibrium piRNA Frequency)",
    fontsize=18,
    y=1.15,
)

p_star = p_star_func(s_fixed, u_fixed)
pc, nm = pc_and_nmax_levels(s_fixed, pi_fixed)
if np.isfinite(p_star) and 0 <= p_star <= 1:
    pcax = cbar2.ax
    trans = blended_transform_factory(pcax.transAxes, pcax.transData)
    pcax.annotate(
        "$p^*$",
        color="r",
        xy=(1.02, p_star),
        xycoords=trans,
        xytext=(3.5, p_star),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    pcax.annotate(
        "$p_{n\\max}$",
        color="c",
        xy=(1.1, pc),
        xycoords=trans,
        xytext=(2.2, pc + 0.19),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

sc3 = ax_dict["C"].scatter(
    fms,
    cs,
    c=n_final,
    cmap="magma",
    vmin=0,
    vmax=nm,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar3 = fig.colorbar(
    sc3,
    ax=ax_dict["C"],
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)
ax_dict["C"].set_title(
    "$n_{eq}$ \n(Equilibrium TE Copy Number)",
    fontsize=18,
    y=1.15,
)

n_star = n_star_func(s_fixed, u_fixed, pi_fixed)
if np.isfinite(n_star) and 0 <= n_star:
    ncax = cbar3.ax
    trans = blended_transform_factory(ncax.transAxes, ncax.transData)
    ncax.annotate(
        "$n^*$",
        color="r",
        xy=(4, n_star),
        xycoords=trans,
        xytext=(6.5, n_star),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="r", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )
    ncax.annotate(
        "$n_{\\max}$",
        color="c",
        xy=(4, nm),
        xycoords=trans,
        xytext=(6.5, nm),
        textcoords=trans,
        arrowprops=dict(arrowstyle="->", color="c", lw=2.2),
        va="center",
        size=16,
        annotation_clip=False,
    )

for ax in ax_main:
    ax.set_yscale("log")
    ax.tick_params(labelsize=15)
    ax.set_ylim(10**LOW, 10**UP)

f1_line, c_line = sigma_y1_zero_curve(
    s_fixed,
    u_fixed,
    pi_fixed,
    low_log10=LOW,
    up_log10=UP,
    mask_invasion=True,
)
for i, ax in enumerate(ax_main):
    if f1_line.size and not np.all(np.isnan(c_line)):
        lbl = r"$\sigma(y_1)=0$" if i == 0 else None
        ax.plot(
            f1_line,
            c_line,
            ls="--",
            lw=2.5,
            color="w",
            alpha=0.9,
            label=lbl,
        )

for ax in ax123:
    ax.set_yscale("log")
    ax.tick_params(labelsize=15)
    ax.set_ylim(10**LOW1, 10**UP1)

f1_line1, c_line1 = sigma_y1_zero_curve(
    s_fixed1,
    u_fixed,
    pi_fixed,
    low_log10=LOW1,
    up_log10=UP1,
    mask_invasion=True,
)
for ax in ax123:
    if f1_line1.size and not np.all(np.isnan(c_line1)):
        ax.plot(
            f1_line1,
            c_line1,
            ls="--",
            lw=2.5,
            color="w",
            alpha=0.9,
        )

for ax in ax456:
    ax.set_yscale("log")
    ax.tick_params(labelsize=15)
    ax.set_ylim(10**LOW2, 10**UP2)

f1_line2, c_line2 = sigma_y1_zero_curve(
    s_fixed2,
    u_fixed,
    pi_fixed,
    low_log10=LOW2,
    up_log10=UP2,
    mask_invasion=True,
)
for ax in ax456:
    if f1_line2.size and not np.all(np.isnan(c_line2)):
        ax.plot(
            f1_line2,
            c_line2,
            ls="--",
            lw=2.5,
            color="w",
            alpha=0.9,
        )

for ax in ax789:
    ax.set_yscale("log")
    ax.tick_params(labelsize=15)
    ax.set_ylim(10**LOW3, 10**UP3)

f1_line3, c_line3 = sigma_y1_zero_curve(
    s_fixed3,
    u_fixed,
    pi_fixed,
    low_log10=LOW3,
    up_log10=UP3,
    mask_invasion=True,
)
for ax in ax789:
    if f1_line3.size and not np.all(np.isnan(c_line3)):
        ax.plot(
            f1_line3,
            c_line3,
            ls="--",
            lw=2.5,
            color="w",
            alpha=0.9,
        )


def sci_label(name, x, digits=2):
    if x == 0:
        return rf"${name}=0$"
    s = f"{x:.{digits}e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    return rf"${name}={mant}\times10^{{{int(exp)}}}$"


bboxA = ax_dict["A"].get_position()
fig.text(
    0.02,
    bboxA.y0 + 0.05,
    sci_label("s", s_fixed, digits=1),
    fontsize=20,
    color="k",
    rotation=90,
)

bbox1 = ax_dict["1"].get_position()
fig.text(
    0.02,
    bbox1.y0 + 0.03,
    sci_label("s", s_fixed1, digits=2),
    fontsize=20,
    color="k",
    rotation=90,
)

bbox2 = ax_dict["4"].get_position()
fig.text(
    0.02,
    bbox2.y0 + 0.04,
    sci_label("s", s_fixed2, digits=1),
    fontsize=20,
    color="k",
    rotation=90,
)

bbox3 = ax_dict["7"].get_position()
fig.text(
    0.02,
    bbox3.y0 + 0.015,
    sci_label("s", s_fixed3, digits=2),
    fontsize=20,
    color="k",
    rotation=90,
)

overlay_pc_and_nmax(
    ax_dict["B"],
    ax_dict["C"],
    np.array(fms),
    np.array(cs),
    np.array(p_final),
    np.array(n_final),
    s_fixed,
    pi_fixed,
)

overlay_pc_and_nmax(
    ax_dict["2"],
    ax_dict["3"],
    np.array(fms1),
    np.array(cs1),
    np.array(p_final1),
    np.array(n_final1),
    s_fixed1,
    pi_fixed,
)

overlay_pc_and_nmax(
    ax_dict["5"],
    ax_dict["6"],
    np.array(fms2),
    np.array(cs2),
    np.array(p_final2),
    np.array(n_final2),
    s_fixed2,
    pi_fixed,
)

overlay_pc_and_nmax(
    ax_dict["8"],
    ax_dict["9"],
    np.array(fms3),
    np.array(cs3),
    np.array(p_final3),
    np.array(n_final3),
    s_fixed3,
    pi_fixed,
)

fm_A, c_A = 0.25, 1e-5
fm_B, c_B = 0.75, 3e-4
fm_C, c_C = 0.75, 3e-6
fm_D, c_D = 0.85, 1e-6
fm_E, c_E = 0.6, 1e-6

markerA = "o"
markerB = "^"
markerC = "s"
markerD = "X"
markerE = "*"

ms = 13
mfc = "w"
mec = "k"
mew = 2

for ax in ax_main:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg,
        C_bg,
        inv_map.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)

    ax.plot(
        fm_A,
        c_A,
        marker=markerA,
        markersize=ms,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
        color="w",
        clip_on=False,
    )
    ax.plot(
        fm_B,
        c_B,
        marker=markerB,
        markersize=ms,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
        color="w",
        clip_on=False,
    )
    ax.plot(
        fm_C,
        c_C,
        marker=markerC,
        markersize=12,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
        color="w",
        clip_on=False,
    )

for ax in ax123:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg1,
        C_bg1,
        inv_map1.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)

for ax in ax456:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg2,
        C_bg2,
        inv_map2.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)

    ax.plot(
        fm_D,
        c_D,
        marker=markerD,
        markersize=ms,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
        color="w",
        clip_on=False,
    )
    ax.plot(
        fm_E,
        c_E,
        marker=markerE,
        markersize=17,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
        color="w",
        clip_on=False,
    )

for ax in ax789:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        FM_bg3,
        C_bg3,
        inv_map3.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("$f_1$ (suppression efficacy)", fontsize=16)
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)

fig.text(0.06, 0.96, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.36, 0.96, "B", fontsize=25, color="k", fontweight="bold")
fig.text(0.66, 0.96, "C", fontsize=25, color="k", fontweight="bold")

plt.subplots_adjust(wspace=3, hspace=10)
plt.show()
