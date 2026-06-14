#fig1
import numpy as np
import matplotlib.pyplot as plt


def K_slope(s, u, pi):
    if not (0 < s < u):
        return np.nan
    num = 2*s**2 * (1 - np.sqrt(s / u)) * np.sqrt(s / u)
    den = pi + 2*s*(1-np.sqrt(s * u))
    K = num / den
    return K

def plot_line(ax, K, f1_grid, label, **kw):
    if not np.isfinite(K) or K <= 0:
        return
    ax.plot(f1_grid, K * f1_grid, label=label, **kw)


u0 = 1e-2
s0 = 1e-3
pi0 = 1e-2

s_list = [7.5e-4, 1e-3, 2.5e-3, 5e-3]

u_list = [1e-2, 2.5e-2, 5e-2, 1e-1]
u_list2 = [1.05e-3]

pi_list = [5e-3, 1e-2, 2.5e-2, 5e-2]

f1 = np.linspace(0, 1.0, 500)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10, 3.5),
    constrained_layout=True,
    sharey=True,
)

ax = axes[0]
for s in s_list:
    K = K_slope(s, u0, pi0)
    plot_line(ax, K, f1, label=fr"$s={s:.2g}$", lw=2)

ax.set_title("Effect of $s$", size=18)
ax.set_xlabel(r"$f$ (suppression efficacy)", size=15)
ax.set_ylabel(r"$c$ (maintenance cost)", size=15)
ax.set_yscale("log")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(1e-8, 2e-3)
ax.tick_params(labelsize=13)
ax.legend(frameon=True, fontsize=10, loc="lower right")

ax = axes[1]
for u in u_list2:
    K = K_slope(s0, u, pi0)
    plot_line(
        ax,
        K,
        f1,
        label=f"$u={u}$",
        lw=2,
        linestyle="--",
        color="tab:purple",
    )

for u in u_list:
    K = K_slope(s0, u, pi0)
    plot_line(ax, K, f1, label=fr"$u={u:.2g}$", lw=2)

ax.set_title("Effect of $u$", size=18)
ax.set_xlabel(r"$f$ (suppression efficacy)", size=15)
ax.set_yscale("log")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(1e-8, 2e-3)
ax.tick_params(labelsize=13)
ax.legend(frameon=True, fontsize=10, loc="lower right")

ax = axes[2]
for pi in pi_list:
    K = K_slope(s0, u0, pi)
    plot_line(ax, K, f1, label=fr"$\pi={pi:.2g}$", lw=2)

ax.set_title("Effect of $\pi$", size=18)
ax.set_xlabel(r"$f$ (suppression efficacy)", size=15)
ax.set_yscale("log")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(1e-8, 2e-3)
ax.tick_params(labelsize=13)
ax.legend(frameon=True, fontsize=10, loc="lower right")

for ax in axes:
    ax.grid(True, which="both", ls="--", alpha=0.35)

fig.text(0.08, 0.96, "A", fontsize=25, color="k", fontweight="bold")
fig.text(0.39, 0.96, "B", fontsize=25, color="k", fontweight="bold")
fig.text(0.70, 0.96, "C", fontsize=25, color="k", fontweight="bold")

plt.savefig('fig1.pdf', format="pdf", bbox_inches='tight')
plt.show()

