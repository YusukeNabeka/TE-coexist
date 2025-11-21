import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.ticker import ScalarFormatter


def sigma_of_y(y, s, u, pi, f1, c):
    num = (2 * s * u * f1 / pi) * (y**3) * (1 - y)
    den = 1 - 2 * s + 2 * s * y
    return num / den - c


def y_interval(s, u, f1):
    y0 = np.sqrt(s / u)
    y1 = np.sqrt(s / (u * (1 - f1)))
    yup = min(1.0, y1)
    return y0, y1, yup


def plot_sigma_y(ax, *, s, u, pi, f1, c):
    y0, y1, yup = y_interval(s, u, f1)
    y = np.linspace(-1, 1.2, 600)
    sig = sigma_of_y(y, s, u, pi, f1, c)

    ax.plot(y, sig, color="k", lw=3, zorder=2)
    ax.axhline(0, color="k", lw=1, ls="-", zorder=0)
    ax.axvspan(y0, yup, facecolor="#FFC080", alpha=0.35, zorder=0)

    s0 = sigma_of_y(y0, s, u, pi, f1, c)
    s1 = sigma_of_y(y1, s, u, pi, f1, c)
    su = sigma_of_y(yup, s, u, pi, f1, c)

    ax.axvline(y0, color="blue", ls=":", lw=2.5)
    ax.axvline(y1, color="blue", ls=":", lw=2.5)
    ax.axvline(0.0, color="gray", lw=2, ls="--")
    ax.axvline(1.0, color="gray", lw=2, ls="--")

    ax.axhline(-0.00025, color="gray", lw=2, ls=":")
    y_target = -c
    trans = axes[0].get_yaxis_transform()

    axes[0].annotate(
        r"$\bf{-c}$",
        xy=(0, y_target),
        xycoords=trans,
        xytext=(-3, 32),
        textcoords="offset points",
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=2.2, color="gray"),
        color="gray",
        annotation_clip=False,
        fontsize=20,
    )

    axes[1].annotate(
        r"$\bf{-c}$",
        xy=(1.4, y_target),
        xycoords=trans,
        xytext=(-3, 32),
        textcoords="offset points",
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=2.2, color="gray"),
        color="gray",
        annotation_clip=False,
        fontsize=20,
    )

    ax.plot([y0], [s0], "o", mfc="w", mec="blue", ms=10, mew=2, zorder=3)
    ax.plot([y1], [s1], "o", mfc="w", mec="blue", ms=10, mew=2, zorder=3)

    ax.annotate(
        r"$\bf{y_0}$",
        xy=(y0, ax.get_ylim()[0]),
        xycoords=("data", "axes fraction"),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color="blue", lw=2.2),
        color="blue",
        size=20,
    )

    ax.annotate(
        r"$\bf{y_1}$",
        xy=(y1, ax.get_ylim()[0]),
        xycoords=("data", "axes fraction"),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color="blue", lw=2.2),
        color="blue",
        size=20,
    )

    sig = sigma_of_y(y, s, u, pi, f1, c)
    if np.any(sig > 0) and np.any(sig < 0):
        idx = np.where(np.signbit(sig[:-1]) != np.signbit(sig[1:]))[0]
        if len(idx):
            try:
                y_star = brentq(
                    lambda yy: sigma_of_y(yy, s, u, pi, f1, c),
                    y[idx[1]],
                    y[idx[1] + 1],
                )
                if y0 < y_star < yup:
                    ax.plot([y_star], [0], "x", ms=12, mew=3, color="r")
                    ax.text(
                        y_star - 0.14,
                        -0.0001,
                        r"$\bf{y^*}$",
                        color="r",
                        ha="left",
                        va="bottom",
                        size=20,
                    )
            except ValueError:
                pass

    ax.set_xlabel(r"$y$", fontsize=20)
    ax.margins(x=0.02)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-0.0003, 0.0005)
    ax.set_yticks([-0.00025, 0, 0.00025, 0.0005])
    ax.tick_params(labelsize=18, width=1.5, length=6)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-4, -4))
    ax.yaxis.offsetText.set_fontsize(16)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)


fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=False, sharey=False)

plot_sigma_y(
    axes[0],
    s=5e-3,
    u=1e-2,
    pi=1e-2,
    f1=0.35,
    c=2.5e-4,
)
axes[0].set_title(
    "KZFP fixation zone ($f_1={0.35}$)",
    fontsize=21,
    x=0.45,
    y=1.13,
)
axes[0].set_ylabel(r"$\sigma$ (Growth rate of KZFP)", size=20)

plot_sigma_y(
    axes[1],
    s=5e-3,
    u=1e-2,
    pi=1e-2,
    f1=0.55,
    c=2.5e-4,
)
axes[1].set_title(
    "KZFP polymorphic zone ($f_1={0.55}$)",
    fontsize=21,
    x=0.47,
    y=1.13,
)
axes[1].set_ylabel(r"$\sigma$ (Growth rate of KZFP)", size=20)

fig.text(0.04, 1, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.49, 1, "B", fontsize=25, color="k", fontweight="bold")
plt.subplots_adjust(wspace=0.4)
plt.show()
