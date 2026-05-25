#fig5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import ScalarFormatter

u_fixed  = 0.01
pi_fixed = 0.01
x0_small = 0.01

s_fixed   = 0.005
s_fixed1  = 0.001


def p_star_func(s, u):
    return 1.0 - np.sqrt(s / u)


def n_star_func(s, u, pi):
    ps = p_star_func(s, u)
    denom = pi + 2.0 * s * ps
    n_star = 2.0 * ps * (1.0 - ps) / denom
    return n_star


def system_log(t, y, pi, s, u, f, c):
    n, p, x = y


    f_bar   = x * (2.0 - x) * f
    repress = (1.0 - f_bar) * (1.0 - p)**2

    dn_dt = n * (u*repress - s)

    direct_gain = (pi / 2.0) * u * repress * n
    
    cost = s * p * (1.0 - p)
    
    benefit = s * u * repress * n * p

    dp_dt = direct_gain - cost + benefit


    sigma_ben = s * u * n * f * (1.0 - p)**2
    dx_dt = (sigma_ben - c) * x * (1.0 - x)**2

    return [dn_dt, dp_dt, dx_dt]


def run_case(*, s, u, pi, f, c, tmax, x0=x0_small):
    n0 = n_star_func(s, u, pi)
    p0 = p_star_func(s, u)

    sol = solve_ivp(
        system_log,
        (0.0, float(tmax)),
        [n0, p0, x0],
        args=(pi, s, u, f, c),
        rtol=1e-9,
        atol=1e-12,
        method="RK45",
    )

    t = sol.t
    nT, pT, xT = sol.y

    return dict(t=t, x=xT, p=pT, n=nT)

scenarios = [
    dict(
        tags=("A", "B"),
        s=s_fixed, u=u_fixed, f=0.25, c=1e-5, tmax=2e5,
        title="$f=0.25$, $c=1\\times10^{-5}$ (●)",
    ),
    dict(
        tags=("C", "D"),
        s=s_fixed, u=u_fixed, f=0.4, c=3e-5, tmax=3e5,
        title="$f=0.75$, $c=3\\times10^{-4}$ (▲)",
    ),
    dict(
        tags=("E", "F"),
        s=s_fixed, u=u_fixed, f=0.75, c=3e-6, tmax=1.5e6,
        title="$f=0.75$, $c=3\\times10^{-6}$ (■)",
    ),

    dict(
        tags=("G", "H"),
        s=s_fixed1, u=u_fixed, f=0.85, c=1e-6, tmax=1e6,
        title="$f=0.85$, $c=1\\times10^{-6}$ (×)",
    ),
    dict(
        tags=("I", "J"),
        s=s_fixed1, u=u_fixed, f=0.60, c=1e-6, tmax=1e6,
        title="$f=0.60$, $c=1\\times10^{-6}$ (★)",
    ),
]


series = {}
global_nmax = 0.0

for sc in scenarios:
    res = run_case(
        s=sc["s"],
        u=sc["u"],
        pi=pi_fixed,
        f=sc["f"],
        c=sc["c"],
        tmax=sc["tmax"],
    )
    series[sc["tags"]] = res
    global_nmax = max(global_nmax, float(np.nanmax(res["n"])))


fig = plt.figure(figsize=(14, 13), constrained_layout=False)
mosaic = """
    AAGGII
    AAGGII
    BBHHJJ
    BBHHJJ
    ......
    CCEE..
    CCEE..
    DDFF..
    DDFF..
"""
ax = fig.subplot_mosaic(mosaic)


def style_freq(axf):
    axf.set_ylim(-0.05, 1.05)
    axf.grid(True, ls=":")
    axf.set_ylabel("Frequency ($x,p$)", fontsize=16)
    axf.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axf.xaxis.offsetText.set_fontsize(14)
    axf.ticklabel_format(style="sci", axis="x", scilimits=(5, 5))
    axf.tick_params("y", labelsize=15)
    axf.tick_params("x", labelsize=15)


def style_n(axn):
    axn.set_ylim(-5, global_nmax * 1.15)
    axn.grid(True, ls=":", alpha=0.9)
    axn.set_ylabel("TE copy number (n)", fontsize=15)
    axn.set_xlabel("Time", fontsize=15)
    axn.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axn.xaxis.offsetText.set_fontsize(14)
    axn.ticklabel_format(style="sci", axis="x", scilimits=(5, 5))
    axn.tick_params(labelsize=15)
    axn.tick_params("x", labelsize=15)


handles_demo = []

for sc in scenarios:
    tag_f, tag_n = sc["tags"]
    dat = series[(tag_f, tag_n)]
    t, x, p, n = dat["t"], dat["x"], dat["p"], dat["n"]

    ln1, = ax[tag_f].plot(t, x, "g", lw=3, label="$x$")
    ln2, = ax[tag_f].plot(t, p, "b", lw=3, label="$p$")
    style_freq(ax[tag_f])
    ax[tag_f].set_title(sc["title"], loc="left", fontsize=16, pad=6, y=1.05)

    ln3, = ax[tag_n].plot(t, n, "r-", lw=3, alpha=0.7, label="$n$")
    style_n(ax[tag_n])

    if not handles_demo:
        handles_demo = [ln1, ln2, ln3]


fig.legend(
    handles_demo,
    ["$x$ (KZFP frequency)", "$p$ (piRNA frequency)", "$n$ (TE copy number)"],
    loc="lower center",
    ncol=1,
    bbox_to_anchor=(0.78, 0.2),
    fontsize=16,
)

fig.text(0.07, 0.92, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.352, 0.92, "B", fontsize=25, color="k", fontweight="bold")
fig.text(0.637, 0.92, "C", fontsize=25, color="k", fontweight="bold")
fig.text(0.07, 0.46, "D", fontsize=25, color="k", fontweight="bold")
fig.text(0.352, 0.46, "E", fontsize=25, color="k", fontweight="bold")

plt.subplots_adjust(wspace=1.2, hspace=1.2)
plt.savefig('fig5.pdf', format="pdf", bbox_inches='tight')
plt.show()
