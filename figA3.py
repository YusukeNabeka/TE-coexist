#figA3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

pi_fixed = 0.01
c_fixed = 0.00001

f_fixed = 0.0
f_fixed1 = 0.25
f_fixed2 = 0.5
f_fixed3 = 0.75


filename = f"u-vs-s_pi{pi_fixed:.1e}_f{f_fixed:.1e}_c{c_fixed:.1e}.pkl"
filename1 = f"u-vs-s_pi{pi_fixed:.1e}_f{f_fixed1:.1e}_c{c_fixed:.1e}.pkl"
filename2 = f"u-vs-s_pi{pi_fixed:.1e}_f{f_fixed2:.1e}_c{c_fixed:.1e}.pkl"
filename3 = f"u-vs-s_pi{pi_fixed:.1e}_f{f_fixed3:.1e}_c{c_fixed:.1e}.pkl"

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
    denom = pi + 2.0 * s * ps
    n_star = 2.0 * ps * (1.0 - ps) / denom
    return n_star

def check_stability_full(s, u):
    return (0 < s < u)


def check_kzfp_invasion(s, u, pi, f, c):
    if not check_stability_full(s, u):
        return False
    ps = p_star_func(s, u)
    ns = n_star_func(s, u, pi)
    if np.isnan(ps) or np.isnan(ns):
        return False
    sigma_ben = s * u * ns * f * (1 - ps) ** 2
    return sigma_ben > c

def sigma_y1_zero_curve_us(
    pi,
    f,
    c,
    npts=1200,
    y_eps=1e-6,
    mask_invasion=True,
    u_min=1e-4,
    u_max=1e-1,
    s_min=1e-4,
    s_max=1e-1,
):


    if f <= 0.0 or f >= 1.0:
        return np.array([]), np.array([])

    y = np.linspace(y_eps, 1.0 - y_eps, npts)

    A = 2.0 * f * y * (1.0 - y) / (1.0 - f)

    with np.errstate(divide="ignore", invalid="ignore"):
        disc = c**2 * (1.0 - y)**2 + A * c * pi
        s = (c * (1.0 - y) + np.sqrt(disc)) / A
        u = s / ((1.0 - f) * y**2)

    good = (
        np.isfinite(u)
        & np.isfinite(s)
    )

    if mask_invasion:
        inv = np.vectorize(check_kzfp_invasion)(s, u, pi, f, c)
        good &= inv

    u_line = np.where(good, u, np.nan)
    s_line = np.where(good, s, np.nan)

    return u_line, s_line

us = [r["u"] for r in res]
ss = [r["s"] for r in res]
x_final = [r["x"] for r in res]
p_final = [r["p"] for r in res]
n_final = [r["n"] for r in res]


us1 = [r["u"] for r in res1]
ss1 = [r["s"] for r in res1]
x_final1 = [r["x"] for r in res1]
p_final1 = [r["p"] for r in res1]
n_final1 = [r["n"] for r in res1]

us2 = [r["u"] for r in res2]
ss2 = [r["s"] for r in res2]
x_final2 = [r["x"] for r in res2]
p_final2 = [r["p"] for r in res2]
n_final2 = [r["n"] for r in res2]


us3 = [r["u"] for r in res3]
ss3 = [r["s"] for r in res3]
x_final3 = [r["x"] for r in res3]
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
inv_map = np.vectorize(check_kzfp_invasion)(S_bg, U_bg, pi_fixed, f_fixed, c_fixed)

u_bg1 = np.logspace(-4, -1, 500)
s_bg1 = np.logspace(-4, -1, 500)
U_bg1, S_bg1 = np.meshgrid(u_bg1, s_bg1)
inv_map1 = np.vectorize(check_kzfp_invasion)(
    S_bg1,
    U_bg1,
    pi_fixed,
    f_fixed1,
    c_fixed,
)

u_bg2 = np.logspace(-4, -1, 500)
s_bg2 = np.logspace(-4, -1, 500)
U_bg2, S_bg2 = np.meshgrid(u_bg2, s_bg2)
inv_map2 = np.vectorize(check_kzfp_invasion)(
    S_bg2,
    U_bg2,
    pi_fixed,
    f_fixed2,
    c_fixed,
)

u_bg3 = np.logspace(-4, -1, 500)
s_bg3 = np.logspace(-4, -1, 500)
U_bg3, S_bg3 = np.meshgrid(u_bg3, s_bg3)
inv_map3 = np.vectorize(check_kzfp_invasion)(
    S_bg3,
    U_bg3,
    pi_fixed,
    f_fixed3,
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
    c=x_final1,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["1"], label="$x_{eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["2"].scatter(
    us1,
    ss1,
    c=p_final1,
    cmap="viridis",
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
    c=x_final2,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["4"], label="$x_{eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["5"].scatter(
    us2,
    ss2,
    c=p_final2,
    cmap="viridis",
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
    c=x_final3,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["7"], label="$x_{eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["8"].scatter(
    us3,
    ss3,
    c=p_final3,
    cmap="viridis",
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
    c=x_final,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["A"], label="$x_{eq}$", pad=0.05, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)
ax_dict["A"].set_title(
    "$x_{eq}$ \n(Equilibrium KZFP Frequency)",
    fontsize=18,
    y=1.15,
)

sc2 = ax_dict["B"].scatter(
    us,
    ss,
    c=p_final,
    cmap="viridis",
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
    f"$f$={f_fixed}",
    fontsize=20,
    color="k",
    rotation=90,
)

bbox1 = ax_dict["1"].get_position()
fig.text(
    0.02,
    bbox1.y0 + 0.07,
    f"$f$={f_fixed1}",
    fontsize=20,
    color="k",
    rotation=90,
)

bbox2 = ax_dict["4"].get_position()
fig.text(
    0.02,
    bbox2.y0 + 0.06,
    f"$f$={f_fixed2}",
    fontsize=20,
    color="k",
    rotation=90,
)

bbox3 = ax_dict["7"].get_position()
fig.text(
    0.02,
    bbox3.y0 + 0.05,
    f"$f$={f_fixed3}",
    fontsize=20,
    color="k",
    rotation=90,
)

u_line, s_line = sigma_y1_zero_curve_us(pi_fixed, f_fixed, c_fixed, mask_invasion=True)
for ax in ax_main:
    ax.plot(u_line, s_line, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

u_line1, s_line1 = sigma_y1_zero_curve_us(
    pi_fixed,
    f_fixed1,
    c_fixed,
    mask_invasion=True,
)
for ax in ax123:
    ax.plot(u_line1, s_line1, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

u_line2, s_line2 = sigma_y1_zero_curve_us(
    pi_fixed,
    f_fixed2,
    c_fixed,
    mask_invasion=True,
)
for ax in ax456:
    ax.plot(u_line2, s_line2, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

u_line3, s_line3 = sigma_y1_zero_curve_us(
    pi_fixed,
    f_fixed3,
    c_fixed,
    mask_invasion=True,
)
for ax in ax789:
    ax.plot(u_line3, s_line3, ls="--", lw=2.5, color="w", alpha=0.9)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-4, 1e-1)

plt.subplots_adjust(wspace=3, hspace=10)
plt.savefig('figA3.pdf', format="pdf", bbox_inches='tight')
plt.show()
