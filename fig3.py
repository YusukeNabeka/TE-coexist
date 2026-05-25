#fig3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.ticker import ScalarFormatter


def sigma_of_pqs(pqs, s, u, pi, f, c):
    num = (2 * s * u * f) * (1-pqs)**3 * pqs
    den = pi + 2 * s * pqs
    return num / den - c


def pqs_interval(s, u, f):
    x0 = 1-np.sqrt(s / u)
    x1 = 1-np.sqrt(s / (u * (1 - f)))
    xlow = max(0.0,x1)
    return x0, x1, xlow


def plot_sigma_pqs(ax, *, s, u, pi, f, c):
    x0, x1, xlow = pqs_interval(s, u, f)
    pqs = np.linspace(-1, 1.2, 600)
    sig = sigma_of_pqs(pqs, s, u, pi, f, c)

    ax.plot(pqs, sig, color="k", lw=3, zorder=2)
    ax.axhline(0, color="k", lw=1, ls="-", zorder=0)
    ax.axvspan(xlow, x0, facecolor="#FFC080", alpha=0.35, zorder=0)

    s0 = sigma_of_pqs(x1, s, u, pi, f, c)
    s1 = sigma_of_pqs(x0, s, u, pi, f, c)
    su = sigma_of_pqs(xlow, s, u, pi, f, c)

    ax.axvline(x0, color="blue", ls=":", lw=2.5)
    ax.axvline(x1, color="blue", ls=":", lw=2.5)
    ax.axvline(0.0, color="gray", lw=2, ls="--")
    ax.axvline(1.0, color="gray", lw=2, ls="--")

    ax.axhline(-0.00025, color="gray", lw=2, ls=":")
    pqs_target = -c
    trans = axes[0].get_yaxis_transform()

    axes[0].annotate(
        r"$\bf{-c}$",
        xy=(0, pqs_target),
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
        xy=(1.4, pqs_target),
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

    ax.plot([x0], [s1], "o", mfc="w", mec="blue", ms=10, mew=2, zorder=3)
    ax.plot([x1], [s0], "o", mfc="w", mec="blue", ms=10, mew=2, zorder=3)
    axes[1].plot(0.0, -c, "o", mfc="w", mec="gray", ms=10, mew=2, zorder=3)

    ax.annotate(
        r"$\bf{p_{x0}}$",
        xy=(x0, -0.01),
        xycoords=("data", "axes fraction"),
        xytext=(0, -25),
        textcoords="offset points",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color="blue", lw=2.2),
        color="blue",
        size=20,
        annotation_clip=False,
    )

    ax.annotate(
        r"$\bf{p_{x1}}$",
        xy=(x1, -0.01),
        xycoords=("data", "axes fraction"),
        xytext=(0, -25),
        textcoords="offset points",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color="blue", lw=2.2),
        color="blue",
        size=20,
        annotation_clip=False,
    )
    
    
    pqs_feas = np.linspace(xlow, x0, 600)
    sig_feas = sigma_of_pqs(pqs_feas, s, u, pi, f, c)

    idx = np.where(np.signbit(sig_feas[:-1]) != np.signbit(sig_feas[1:]))[0]

    if len(idx) > 0:
        i = idx[0]

        try:
            pqs_star = brentq(
                lambda yy: sigma_of_pqs(yy, s, u, pi, f, c),
                pqs_feas[i],
                pqs_feas[i + 1],
            )

            ax.plot([pqs_star], [0], "x", ms=15, mew=4, color="r", zorder=4)

            ax.text(
                pqs_star + 0.025,
                -0.000082,
                r"$\bf{p_{\mathrm{qs}}^*}$",
                color="r",
                ha="left",
                va="bottom",
                size=20,
            )

        except ValueError:
            pass

    ax.set_xlabel(r"$p_\mathrm{qs}$", fontsize=20)
    ax.margins(x=0.02)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-0.0003, 0.0003)
    ax.set_yticks([-0.00025, 0, 0.00025])
    ax.tick_params(labelsize=18, width=1.5, length=6)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-4, -4))
    ax.yaxis.offsetText.set_fontsize(16)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)


fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=False, sharey=False)

plot_sigma_pqs(
    axes[0],
    s=5e-3,
    u=1e-2,
    pi=1e-2,
    f=0.35,
    c=2.5e-4,
)
axes[0].set_title(
    "KZFP fixation zone ($f={0.35}$)",
    fontsize=21,
    x=0.45,
    y=1.13,
)
axes[0].set_ylabel(r"$\sigma$ (net selective advantage)", size=20)

plot_sigma_pqs(
    axes[1],
    s=5e-3,
    u=1e-2,
    pi=1e-2,
    f=0.55,
    c=2.5e-4,
)
axes[1].set_title(
    "KZFP polymorphic zone ($f={0.55}$)",
    fontsize=21,
    x=0.47,
    y=1.13,
)
axes[1].set_ylabel(r"$\sigma$ (net selective advantage)", size=20)

fig.text(0.04, 1, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.49, 1, "B", fontsize=25, color="k", fontweight="bold")
plt.subplots_adjust(wspace=0.4)
plt.savefig('fig3.pdf', format="pdf", bbox_inches='tight')
plt.show()