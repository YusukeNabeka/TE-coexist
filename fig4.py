#fig4
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm


u_val = 0.01
pi_val = 0.01
c_val = 1e-5

t_end = 5




def p_star_func_resident(s, u):
    r = s / (u + 1e-22)
    return np.nan if not (0 <= r < 1) else 1 - np.sqrt(r)

def n_star_func_resident(s, u, pi):
    ps = p_star_func_resident(s, u)
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



def calculate_trajectory(s_val, u_val, pi_val, f_range):
    traj = []


    n0 = n_star_func_resident(s_val, u_val, pi_val)
    p0 = p_star_func_resident(s_val, u_val)
    if np.isnan(n0) or np.isnan(p0) or n0 <= 0:
        return traj


    x0 = 0.01

    for f in tqdm(f_range, leave=False):
        sol = solve_ivp(
            system_log,
            (0, t_end),
            [n0, p0, x0],
            args=(pi_val, s_val, u_val, f, c_val),
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            continue

        n_eq, p_eq, x_eq = sol.y[:, -1]

        traj.append({"f": f, "n": n_eq, "p": p_eq})

    return traj


def plot_trajectory(ax, s_val, u_val, pi_val):
    f_range = np.linspace(0, 1, 1000)
    trajectory = calculate_trajectory(s_val, u_val, pi_val, f_range)

    if len(trajectory) == 0:
        ax.text(
            0.5,
            0.5,
            "No stable equilibrium found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return None

    n_points = np.array([r["n"] for r in trajectory])
    p_points = np.array([r["p"] for r in trajectory])
    f_points = np.array([r["f"] for r in trajectory])

    valid = ~np.isnan(n_points) & ~np.isnan(p_points)
    n_plot = n_points[valid]
    p_plot = p_points[valid]
    f_plot = f_points[valid]


    sc = ax.scatter(
        p_plot,
        n_plot,
        c=f_plot,
        cmap="viridis",
        s=45,
        zorder=3,
    )
    ax.plot(p_plot, n_plot, lw=1.5, color="k", alpha=0.9, zorder=2)


    p_grid = np.linspace(0, 1, 2000)
    den = pi_val + 2.0 * s_val * p_grid
    n_curve = (2.0 ) * (p_grid * (1.0 - p_grid) / den)
    ax.plot(
        p_grid,
        n_curve,
        color="gray",
        lw=1.6,
        alpha=0.7,
        label=r"$n_{eq}(p_{eq})$",
    )


    ax.plot(
        p_plot[0],
        n_plot[0],
        "o",
        color="cyan",
        markersize=12,
        markeredgecolor="black",
        label=r"$f=0$",
        zorder=5,
    )
    ax.plot(
        p_plot[-1],
        n_plot[-1],
        "o",
        color="red",
        markersize=12,
        markeredgecolor="black",
        label=r"$f=1$",
        zorder=5,
    )

    p_star0 = p_star_func_resident(s_val, u_val)
    
    has_pc = (-pi_val + np.sqrt(pi_val**2 + 2.0 * s_val * pi_val)) < (2.0 * s_val)
    if has_pc:
        p_c = (-pi_val + np.sqrt(pi_val**2 + 2.0 * s_val * pi_val)) / (2.0 * s_val)
    else:
        p_c = np.nan
    non_monotonic = has_pc and (p_star0 > p_c)


    if has_pc and (p_star0 > p_c):
        ax.text(
            p_plot[0] * 1.15,
            n_plot[0] * 1.05,
            r"$(p^*,n^*)$",
            color="k",
            ha="center",
            va="bottom",
            fontsize=13,
        )
    else:
        ax.text(
            p_plot[0] * 0.65,
            n_plot[0] * 1.05,
            r"$(p^*,n^*)$",
            color="k",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    n_star = n_star_func_resident(s_val, u_val, pi_val)
    if not np.isnan(n_star) and np.isfinite(n_star):
        ax.axhline(
            n_star,
            color="tab:orange",
            linestyle="--",
            lw=1.8,
            alpha=0.95,
        )

    den_pc = pi_val + 2.0 * s_val * p_c
    if den_pc > 0:
        n_at_pc = (2.0) * (p_c * (1.0 - p_c) / den_pc)
        ax.plot(
            p_c,
            n_at_pc,
            "*",
            color="yellow",
            markersize=18,
            markeredgecolor="k",
            zorder=6,
        )
        ax.text(
            p_c*1.2,
            n_at_pc * 1.07,
            r"$(p_{n\max},n_{\max})$",
            color="k",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    if non_monotonic and (n_plot.max() > n_star):
        max_idx = int(np.argmax(n_plot))
        post_n = n_plot[max_idx:]
        post_p = p_plot[max_idx:]
        below = np.where(post_n <= n_star)[0]
        if len(below) > 0 and below[0] > 0:
            j = below[0]
            n1, n2 = post_n[j - 1], post_n[j]
            p1, p2 = post_p[j - 1], post_p[j]
            alpha = (n_star - n1) / (n2 - n1 + 1e-16)
            p_cross = p1 + alpha * (p2 - p1)
            ax.plot(
                p_cross,
                n_star,
                "o",
                color="orange",
                markersize=12,
                markeredgecolor="k",
                zorder=6,
            )
            ax.text(
                p_cross * 0.65,
                n_star * 1.05,
                r"$(p^{**},n^{*})$",
                color="k",
                ha="center",
                va="bottom",
                fontsize=13,
            )


    ax.set_ylim(-3, 53)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)

    return sc



fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.7, 4),
    sharey=True,
    constrained_layout=True,
)
axA, axB = axes

sc1 = plot_trajectory(axA, s_val=0.005, u_val=u_val, pi_val=pi_val)
axA.set_title("Under strong selection ($s = 0.005$)", fontsize=15)
axA.set_xlabel("$p_{eq}$ (Equilibrium piRNA frequency)", fontsize=13)
axA.set_ylabel("$n_{eq}$ (Equilibrium TE copy number)", fontsize=13)
axA.tick_params(labelsize=13)

sc2 = plot_trajectory(axB, s_val=0.001, u_val=u_val, pi_val=pi_val)
axB.set_title("Under weak selection ($s = 0.001$)", fontsize=15)
axB.set_xlabel("$p_{eq}$ (Equilibrium piRNA frequency)", fontsize=13)
axB.tick_params(labelsize=13)

mappable = sc1 if sc1 is not None else sc2
if mappable is not None:
    cbar = fig.colorbar(mappable, ax=axes, pad=0.03, aspect=15)
    cbar.set_label("$f$ (suppression efficacy)", fontsize=13)
    cbar.ax.tick_params(labelsize=13)


fig.text(0.06, 1.03, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.48, 1.03, "B", fontsize=25, color="k", fontweight="bold")
plt.subplots_adjust(wspace=0.4)

plt.savefig('fig4.pdf', format="pdf", bbox_inches='tight')
plt.show()
