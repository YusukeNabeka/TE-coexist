import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

pi_fixed = 0.01
c_fixed = 0.00001

fm_fixed = 0.25
fm_fixed1 = 0.5
fm_fixed2 = 0.75
fm_fixed3 = 0.9


filename = f"u-vs-s_pi{pi_fixed:.1e}_f{fm_fixed:.1e}_c{c_fixed:.1e}.pkl"
filename1 = f"u-vs-s_pi{pi_fixed:.1e}_f{fm_fixed1:.1e}_c{c_fixed:.1e}.pkl"
filename2 = f"u-vs-s_pi{pi_fixed:.1e}_f{fm_fixed2:.1e}_c{c_fixed:.1e}.pkl"
filename3 = f"u-vs-s_pi{pi_fixed:.1e}_f{fm_fixed3:.1e}_c{c_fixed:.1e}.pkl"

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


def system(t, y, pi, s, u, fm, c):
    n, p, xm = y
    p = np.clip(p, 0, 1)
    xm = np.clip(xm, 0, 1)
    n = max(n, 0)

    f_bar = xm * (2 - xm) * fm
    repress = (1 - f_bar) * (1 - p) ** 2
    dn_dt = u * repress * n - s * n

    sel_den = 1 - 2 * s * p
    sel_term = (
        s * p * (1 - p) / sel_den
        if abs(sel_den) > 1e-12
        else np.sign(s * p * (1 - p)) * 1e12
    )

    dp_dt = (pi / 2) * u * repress * n - sel_term
    sigma_ben = s * u * n * fm * (1 - p) ** 2
    dxm_dt = (sigma_ben - c) * xm * (1 - xm) ** 2

    return [dn_dt, dp_dt, dxm_dt]


def sigma_y1_zero_curve_us(
    pi,
    fm,
    c,
    npts=1200,
    y_eps=1e-6,
    mask_invasion=True,
    u_min=1e-4,
    u_max=1e-1,
    s_min=1e-4,
    s_max=1e-1,
):

    y = np.linspace(y_eps, 1.0 - y_eps, npts)
    A = (2.0 * fm / ((1.0 - fm) * c * pi)) * y * (1.0 - y)

    disc = (1.0 - y) ** 2 + A
    s = ((1.0 - y) + np.sqrt(disc)) / A

    u = s / ((1.0 - fm) * y**2)

    good = np.isfinite(u) & np.isfinite(s) & (u > 0) & (s > 0)

    if mask_invasion:
        inv = np.vectorize(check_kzfp_invasion)(s, u, pi, fm, c)
        good &= inv

    u_line = np.where(good, u, np.nan)
    s_line = np.where(good, s, np.nan)
    return u_line, s_line


us = [r["u"] for r in res]
ss = [r["s"] for r in res]
xm_final = [r["xm"] for r in res]
p_final = [r["p"] for r in res]
n_final = [r["n"] for r in res]


us1 = [r["u"] for r in res1]
ss1 = [r["s"] for r in res1]
xm_final1 = [r["xm"] for r in res1]
p_final1 = [r["p"] for r in res1]
n_final1 = [r["n"] for r in res1]

us2 = [r["u"] for r in res2]
ss2 = [r["s"] for r in res2]
xm_final2 = [r["xm"] for r in res2]
p_final2 = [r["p"] for r in res2]
n_final2 = [r["n"] for r in res2]


us3 = [r["u"] for r in res3]
ss3 = [r["s"] for r in res3]
xm_final3 = [r["xm"] for r in res3]
p_final3 = [r["p"] for r in res3]
n_final3 = [r["n"] for r in res3]



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

u_bg = np.logspace(-4, -1, 500)
s_bg = np.logspace(-4, -1, 500)
U_bg, S_bg = np.meshgrid(u_bg, s_bg)
inv_map = np.vectorize(check_kzfp_invasion)(S_bg, U_bg, pi_fixed, fm_fixed, c_fixed)

u_bg1 = np.logspace(-4, -1, 500)
s_bg1 = np.logspace(-4, -1, 500)
U_bg1, S_bg1 = np.meshgrid(u_bg1, s_bg1)
inv_map1 = np.vectorize(check_kzfp_invasion)(
    S_bg1,
    U_bg1,
    pi_fixed,
    fm_fixed1,
    c_fixed,
)

u_bg2 = np.logspace(-4, -1, 500)
s_bg2 = np.logspace(-4, -1, 500)
U_bg2, S_bg2 = np.meshgrid(u_bg2, s_bg2)
inv_map2 = np.vectorize(check_kzfp_invasion)(
    S_bg2,
    U_bg2,
    pi_fixed,
    fm_fixed2,
    c_fixed,
)

u_bg3 = np.logspace(-4, -1, 500)
s_bg3 = np.logspace(-4, -1, 500)
U_bg3, S_bg3 = np.meshgrid(u_bg3, s_bg3)
inv_map3 = np.vectorize(check_kzfp_invasion)(
    S_bg3,
    U_bg3,
    pi_fixed,
    fm_fixed3,
    c_fixed,
)

for ax in ax123:
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.contour(
        U_bg1,
        S_bg1,
        inv_map1.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("$s$ (selection coefficient)", fontsize=15)
    ax.tick_params(labelsize=15)

sc1 = ax_dict["1"].scatter(
    us1,
    ss1,
    c=xm_final1,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["1"], label="$x_{1eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["2"].scatter(
    us1,
    ss1,
    c=p_final1,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["2"], label="$p_{eq}$", pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

sc3 = ax_dict["3"].scatter(
    us1,
    ss1,
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
    label="$n_{eq}$",
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)

for ax in ax456:
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.contour(
        U_bg2,
        S_bg2,
        inv_map2.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("$s$ (selection coefficient)", fontsize=15)
    ax.tick_params(labelsize=15)

sc1 = ax_dict["4"].scatter(
    us2,
    ss2,
    c=xm_final2,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["4"], label="$x_{1eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["5"].scatter(
    us2,
    ss2,
    c=p_final2,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["5"], label="$p_{eq}$", pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

sc3 = ax_dict["6"].scatter(
    us2,
    ss2,
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
    label="$n_{eq}$",
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)

for ax in ax789:
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.contour(
        U_bg3,
        S_bg3,
        inv_map3.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$u$ (transposition rate)", fontsize=15)
    ax.set_ylabel("$s$ (selection coefficient)", fontsize=15)
    ax.tick_params(labelsize=15)

sc1 = ax_dict["7"].scatter(
    us3,
    ss3,
    c=xm_final3,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["7"], label="$x_{1eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["8"].scatter(
    us3,
    ss3,
    c=p_final3,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["8"], label="$p_{eq}$", pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

sc3 = ax_dict["9"].scatter(
    us3,
    ss3,
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
    label="$n_{eq}$",
    ticks=[0, 10, 20, 30, 40, 50],
    pad=0.05,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)

sc1 = ax_dict["A"].scatter(
    us,
    ss,
    c=xm_final,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["A"], label="$x_{1eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{1eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)
ax_dict["A"].set_title(
    "$x_{1eq}$ \n(Equilibrium KZFP Frequency)",
    fontsize=18,
    y=1.15,
)

sc2 = ax_dict["B"].scatter(
    us,
    ss,
    c=p_final,
    cmap="plasma",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["B"], label="$p_{eq}$", pad=0.05, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)
ax_dict["B"].set_title(
    "$p_{eq}$ \n(Equilibrium piRNA Frequency)",
    fontsize=18,
    y=1.15,
)

sc3 = ax_dict["C"].scatter(
    us,
    ss,
    c=n_final,
    cmap="magma",
    vmin=0,
    vmax=50,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar3 = fig.colorbar(
    sc3,
    ax=ax_dict["C"],
    label="$n_{eq}$",
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

for ax in ax_main:
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.contour(
        U_bg,
        S_bg,
        inv_map.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("$s$ (selection coefficient)", fontsize=15)
    ax.tick_params(labelsize=15)

fig.text(0.06, 0.96, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.36, 0.96, "B", fontsize=25, color="k", fontweight="bold")
fig.text(0.66, 0.96, "C", fontsize=25, color="k", fontweight="bold")

bboxA = ax_dict["A"].get_position()
fig.text(
    0.02,
    bboxA.y0 + 0.08,
    f"$f_1$={fm_fixed}",
    fontsize=20,
    color="k",
    rotation=90,
)

bbox1 = ax_dict["1"].get_position()
fig.text(
    0.02,
    bbox1.y0 + 0.07,
    f"$f_1$={fm_fixed1}",
    fontsize=20,
    color="k",
    rotation=90,
)

bbox2 = ax_dict["4"].get_position()
fig.text(
    0.02,
    bbox2.y0 + 0.06,
    f"$f_1$={fm_fixed2}",
    fontsize=20,
    color="k",
    rotation=90,
)

bbox3 = ax_dict["7"].get_position()
fig.text(
    0.02,
    bbox3.y0 + 0.05,
    f"$f_1$={fm_fixed3}",
    fontsize=20,
    color="k",
    rotation=90,
)

u_line, s_line = sigma_y1_zero_curve_us(pi_fixed, fm_fixed, c_fixed, mask_invasion=True)
for ax in ax_main:
    ax.plot(u_line, s_line, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

u_line1, s_line1 = sigma_y1_zero_curve_us(
    pi_fixed,
    fm_fixed1,
    c_fixed,
    mask_invasion=True,
)
for ax in ax123:
    ax.plot(u_line1, s_line1, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

u_line2, s_line2 = sigma_y1_zero_curve_us(
    pi_fixed,
    fm_fixed2,
    c_fixed,
    mask_invasion=True,
)
for ax in ax456:
    ax.plot(u_line2, s_line2, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

u_line3, s_line3 = sigma_y1_zero_curve_us(
    pi_fixed,
    fm_fixed3,
    c_fixed,
    mask_invasion=True,
)
for ax in ax789:
    ax.plot(u_line3, s_line3, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

plt.subplots_adjust(wspace=3, hspace=10)
plt.show()
