#figA5
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

filename = f"f-vs-c_u{u_fixed:.1e}_s{s_fixed:.1e}_pi{pi_fixed:.1e}.pkl"
filename1 = f"f-vs-c_u{u_fixed:.1e}_s{s_fixed1:.1e}_pi{pi_fixed:.1e}.pkl"
filename2 = f"f-vs-c_u{u_fixed:.1e}_s{s_fixed2:.1e}_pi{pi_fixed:.1e}.pkl"
filename3 = f"f-vs-c_u{u_fixed:.1e}_s{s_fixed3:.1e}_pi{pi_fixed:.1e}.pkl"

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



def sigma_y1_zero_curve(
    s, u, pi, low_log10=None, up_log10=None, npts=600, mask_invasion=True
):
    f_max = max(1e-8, 1.0 - s / u)
    if f_max <= 1e-6:
        return np.array([]), np.array([])

    f_line = np.linspace(1e-6, f_max - 1e-6, npts)

    y1 = np.sqrt(s / (u * (1.0 - f_line)))
    denom = 1.0 #- 2.0 * s + 2.0 * s * y1
    num = y1**3 * (1.0 - y1)

    c_line = (2.0 * s * u * f_line / pi) * (num / denom)

    if mask_invasion and f_line.size:
        valid = ~np.isnan(c_line)
        if np.any(valid):
            inv_ok = np.zeros_like(c_line, dtype=bool)
            idx = np.where(valid)[0]
            for i in idx:
                inv_ok[i] = check_kzfp_invasion(
                    s, u, pi, float(f_line[i]), float(c_line[i])
                )
            c_line[~inv_ok] = np.nan

    return f_line, c_line



fs = [r["f"] for r in res]
cs = [r["c"] for r in res]
x_final = [r["x"] for r in res]
p_final = [r["p"] for r in res]
n_final = [r["n"] for r in res]

LOW, UP = yaxis_range(s_fixed, u_fixed, pi_fixed)

fs1 = [r["f"] for r in res1]
cs1 = [r["c"] for r in res1]
x_final1 = [r["x"] for r in res1]
p_final1 = [r["p"] for r in res1]
n_final1 = [r["n"] for r in res1]

LOW1, UP1 = yaxis_range(s_fixed1, u_fixed, pi_fixed)

fs2 = [r["f"] for r in res2]
cs2 = [r["c"] for r in res2]
x_final2 = [r["x"] for r in res2]
p_final2 = [r["p"] for r in res2]
n_final2 = [r["n"] for r in res2]

LOW2, UP2 = yaxis_range(s_fixed2, u_fixed, pi_fixed)

fs3 = [r["f"] for r in res3]
cs3 = [r["c"] for r in res3]
x_final3 = [r["x"] for r in res3]
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

f_bg = np.linspace(0, 1, 500)
c_bg = np.logspace(LOW, UP, 500)
F_bg, C_bg = np.meshgrid(f_bg, c_bg)
inv_map = np.vectorize(check_kzfp_invasion)(s_fixed, u_fixed, pi_fixed, F_bg, C_bg)

f_bg1 = np.linspace(0, 1, 500)
c_bg1 = np.logspace(LOW1, UP1, 500)
F_bg1, C_bg1 = np.meshgrid(f_bg1, c_bg1)
inv_map1 = np.vectorize(check_kzfp_invasion)(s_fixed1, u_fixed, pi_fixed, F_bg1, C_bg1)

f_bg2 = np.linspace(0, 1, 500)
c_bg2 = np.logspace(LOW2, UP2, 500)
F_bg2, C_bg2 = np.meshgrid(f_bg2, c_bg2)
inv_map2 = np.vectorize(check_kzfp_invasion)(s_fixed2, u_fixed, pi_fixed, F_bg2, C_bg2)

f_bg3 = np.linspace(0, 1, 500)
c_bg3 = np.logspace(LOW3, UP3, 500)
F_bg3, C_bg3 = np.meshgrid(f_bg3, c_bg3)
inv_map3 = np.vectorize(check_kzfp_invasion)(s_fixed3, u_fixed, pi_fixed, F_bg3, C_bg3)

for ax in ax123:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        F_bg1,
        C_bg1,
        inv_map1.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")

sc1 = ax_dict["1"].scatter(
    fs1,
    cs1,
    c=x_final1,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["1"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["2"].scatter(
    fs1,
    cs1,
    c=p_final1,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["2"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)


sc3 = ax_dict["3"].scatter(
    fs1,
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
    pad=0.08,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)


for ax in ax456:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        F_bg2,
        C_bg2,
        inv_map2.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")

sc1 = ax_dict["4"].scatter(
    fs2,
    cs2,
    c=x_final2,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["4"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["5"].scatter(
    fs2,
    cs2,
    c=p_final2,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["5"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)


sc3 = ax_dict["6"].scatter(
    fs2,
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
    pad=0.08,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)


for ax in ax789:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        F_bg3,
        C_bg3,
        inv_map3.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")

sc1 = ax_dict["7"].scatter(
    fs3,
    cs3,
    c=x_final3,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["7"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)

sc2 = ax_dict["8"].scatter(
    fs3,
    cs3,
    c=p_final3,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["8"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)

sc3 = ax_dict["9"].scatter(
    fs3,
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
    pad=0.08,
    aspect=14,
)
cbar3.set_label("$n_{eq}$", size=15)
cbar3.ax.tick_params(labelsize=15)


sc1 = ax_dict["A"].scatter(
    fs,
    cs,
    c=x_final,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar1 = fig.colorbar(sc1, ax=ax_dict["A"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar1.set_label("$x_{eq}$", size=15)
cbar1.ax.tick_params(labelsize=15)
ax_dict["A"].set_title(
    "$x_{eq}$ \n(Equilibrium KZFP Frequency)",
    fontsize=18,
    y=1.15,
)

sc2 = ax_dict["B"].scatter(
    fs,
    cs,
    c=p_final,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolor="none",
    marker="s",
    s=35,
)
cbar2 = fig.colorbar(sc2, ax=ax_dict["B"], ticks=[0, 0.5, 1], pad=0.08, aspect=14)
cbar2.set_label("$p_{eq}$", size=15)
cbar2.ax.tick_params(labelsize=15)
ax_dict["B"].set_title(
    "$p_{eq}$ \n(Equilibrium piRNA Frequency)",
    fontsize=18,
    y=1.15,
)


sc3 = ax_dict["C"].scatter(
    fs,
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
    pad=0.08,
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
    ax.set_yscale("log")
    ax.tick_params(labelsize=15)
    ax.set_ylim(10**LOW, 10**UP)

f_line, c_line = sigma_y1_zero_curve(
    s_fixed,
    u_fixed,
    pi_fixed,
    low_log10=LOW,
    up_log10=UP,
    mask_invasion=True,
)
for i, ax in enumerate(ax_main):
    if f_line.size and not np.all(np.isnan(c_line)):
        lbl = r"$\sigma(y_1)=0$" if i == 0 else None
        ax.plot(
            f_line,
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

f_line1, c_line1 = sigma_y1_zero_curve(
    s_fixed1,
    u_fixed,
    pi_fixed,
    low_log10=LOW1,
    up_log10=UP1,
    mask_invasion=True,
)
for ax in ax123:
    if f_line1.size and not np.all(np.isnan(c_line1)):
        ax.plot(
            f_line1,
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

f_line2, c_line2 = sigma_y1_zero_curve(
    s_fixed2,
    u_fixed,
    pi_fixed,
    low_log10=LOW2,
    up_log10=UP2,
    mask_invasion=True,
)
for ax in ax456:
    if f_line2.size and not np.all(np.isnan(c_line2)):
        ax.plot(
            f_line2,
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

f_line3, c_line3 = sigma_y1_zero_curve(
    s_fixed3,
    u_fixed,
    pi_fixed,
    low_log10=LOW3,
    up_log10=UP3,
    mask_invasion=True,
)
for ax in ax789:
    if f_line3.size and not np.all(np.isnan(c_line3)):
        ax.plot(
            f_line3,
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



f_A, c_A = 0.25, 1e-5
f_B, c_B = 0.75, 3e-4
f_C, c_C = 0.75, 3e-6
f_D, c_D = 0.85, 1e-6
f_E, c_E = 0.6, 1e-6

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
        F_bg,
        C_bg,
        inv_map.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)


for ax in ax123:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        F_bg1,
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
        F_bg2,
        C_bg2,
        inv_map2.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)


for ax in ax789:
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.contour(
        F_bg3,
        C_bg3,
        inv_map3.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("$f$ (suppression efficacy)", fontsize=16)
    ax.set_ylabel("$c$ (maintenance cost)", fontsize=16)

fig.text(0.06, 0.96, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.36, 0.96, "B", fontsize=25, color="k", fontweight="bold")
fig.text(0.66, 0.96, "C", fontsize=25, color="k", fontweight="bold")




## =========== simulations ===========
import pandas as pd
import numpy as np
import glob
import os

def extract_equilibrium_state(df, smooth_window=100):
    max_kz = df['kz_freq'].max()
    final_kz = df['kz_freq'].iloc[-1]
    final_te = df['te'].iloc[-1]
    

    is_loss = (final_te > 0) and (final_kz == 0)    
    is_established = (final_te == 0) or (final_kz > 0)
    
    state_dict = {
        'is_loss': is_loss,
        'is_established': is_established,
        'te': np.nan,
        'pi_freq': np.nan,
        'kz_freq': np.nan
    }


    if final_te == 0:
        first_zero_idx = df[df['te'] == 0].index[0]
        state = df.loc[first_zero_idx]
        state_dict['te'] = 0.0
        state_dict['pi_freq'] = state['pi_freq']
        state_dict['kz_freq'] = state['kz_freq']
    else:
        tail_df = df.tail(smooth_window)
        state_dict['te'] = tail_df['te'].mean()
        state_dict['pi_freq'] = tail_df['pi_freq'].mean()
        state_dict['kz_freq'] = tail_df['kz_freq'].mean()
        
    return state_dict



masterdir="./simulation/data/phase2/"
est_thres=0.0
f_list=[0.0,0.2,0.4,0.6,0.8,1.0]
c_list_main=[3e-3,3e-4,3e-5,3e-6]
for f in f_list:
    for c in c_list_main:
        data_dir = masterdir+f"N10000_p1_u0.010_s0.005_pi0.01_r0.499/f{f:.2f}_c{c:.1e}"
        
        files = glob.glob(os.path.join(data_dir, "*.tsv"))

        summary_data = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            if not df.empty:
                state = extract_equilibrium_state(df)
                summary_data.append(state)

        result_df = pd.DataFrame(summary_data)

        final_mean=result_df[['te', 'pi_freq', 'kz_freq']].mean()

        if not result_df.empty:
            total_reps = len(result_df)
            loss_df = result_df[result_df['is_loss']]
            established_df = result_df[result_df['is_established']]

            p_loss = len(loss_df) / total_reps
            p_est = len(established_df) / total_reps
            ms=7.5+15*p_est

            if p_est > est_thres:
                final_mean = established_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Established Reps"

            else:
                final_mean = loss_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Not Established Reps"

        mec="gray"
        ax_main[0].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[2]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax_main[1].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[1]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax_main[2].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('magma')(final_mean[0]/50),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )

c_list_main=[7e-4,7e-5,7e-6,7e-7]
for f in f_list:
    for c in c_list_main:
        data_dir = masterdir+f"N10000_p1_u0.010_s0.0025_pi0.01_r0.499/f{f:.2f}_c{c:.1e}"
        
        files = glob.glob(os.path.join(data_dir, "*.tsv"))

        summary_data = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            if not df.empty:
                state = extract_equilibrium_state(df)
                summary_data.append(state)

        result_df = pd.DataFrame(summary_data)

        if not result_df.empty:
            total_reps = len(result_df)
            loss_df = result_df[result_df['is_loss']]
            established_df = result_df[result_df['is_established']]

            p_loss = len(loss_df) / total_reps
            p_est = len(established_df) / total_reps
            ms=7.5+15*p_est

            if p_est > est_thres:
                final_mean = established_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Established Reps"

            else:
                final_mean = loss_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Not Established Reps"

        
        ax123[0].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[2]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax123[1].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[1]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax123[2].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('magma')(final_mean[0]/50),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )


c_list_main=[1e-4,1e-5,1e-6,1e-7]
for f in f_list:
    for c in c_list_main:
        data_dir = masterdir+f"N10000_p1_u0.010_s0.001_pi0.01_r0.499/f{f:.2f}_c{c:.1e}"
                
        files = glob.glob(os.path.join(data_dir, "*.tsv"))

        summary_data = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            if not df.empty:
                state = extract_equilibrium_state(df)
                summary_data.append(state)

        result_df = pd.DataFrame(summary_data)

        if not result_df.empty:
            total_reps = len(result_df)
            loss_df = result_df[result_df['is_loss']]
            established_df = result_df[result_df['is_established']]

            p_loss = len(loss_df) / total_reps
            p_est = len(established_df) / total_reps
            ms=7.5+15*p_est

            if p_est > est_thres:
                final_mean = established_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Established Reps"

            else:
                final_mean = loss_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Not Established Reps"

        
        ax456[0].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[2]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax456[1].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[1]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax456[2].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('magma')(final_mean[0]/50),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )


c_list_main=[5e-5,5e-6,5e-7,5e-8]
for f in f_list:
    for c in c_list_main:
        data_dir = masterdir+f"N10000_p1_u0.010_s0.00075_pi0.01_r0.499/f{f:.2f}_c{c:.1e}"
        
        files = glob.glob(os.path.join(data_dir, "*.tsv"))

        summary_data = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            if not df.empty:
                state = extract_equilibrium_state(df)
                summary_data.append(state)

        result_df = pd.DataFrame(summary_data)
        if not result_df.empty:
            total_reps = len(result_df)
            loss_df = result_df[result_df['is_loss']]
            established_df = result_df[result_df['is_established']]

            p_loss = len(loss_df) / total_reps
            p_est = len(established_df) / total_reps
            ms=7.5+15*p_est

            if p_est > est_thres:
                final_mean = established_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Established Reps"
            else:
                final_mean = loss_df[['te', 'pi_freq', 'kz_freq']].mean()
                adoption_type = "Mean of Not Established Reps"

        ax789[0].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[2]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax789[1].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('viridis')(final_mean[1]),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )
        ax789[2].plot(
            f,
            c,
            marker="o",
            markersize=ms,
            markerfacecolor=plt.get_cmap('magma')(final_mean[0]/50),
            markeredgecolor=mec,
            markeredgewidth=1,
            color="w",
            clip_on=False,
            )

import matplotlib.lines as mlines
legend_probs = [0.0, 0.5, 1.0]
legend_handles = []

for p in legend_probs:
    legend_ms = 7.5 + 15 * p

    handle = mlines.Line2D(
        [], [], color='none', marker='o',
        markersize=legend_ms,
        markerfacecolor='lightgray',
        markeredgecolor='dimgray',
        markeredgewidth=1,
        label=f'{p * 100:.0f}%'
    )
    legend_handles.append(handle)

plt.legend(
    handles=legend_handles, 
    title="Fraction of replicates in which the KZFP allele was established",
    loc='center',
    bbox_to_anchor=(-1.6, -0.6),
    ncol=3,
    frameon=True,
    columnspacing=2.0,
    fontsize=16,
    title_fontsize=16
)


plt.subplots_adjust(wspace=2.5, hspace=10)
plt.savefig('figA5.pdf', format="pdf", bbox_inches='tight')
plt.show()

