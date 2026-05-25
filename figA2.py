#figA2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

u  = 1e-2
s  = 1e-3
pi = 1e-2
c  = 1e-5

def fbar_two_allele(x, f):
    return x * (2.0 - x) * f

def fast_equilibrium(x, f, u, s, pi, strict=False):

    fb   = fbar_two_allele(x, f)
    Ueff = u * (1.0 - fb)

    if Ueff <= 0.0 or Ueff <= s:
        msg = (f"No internal TE equilibrium: "
               f"U_eff={Ueff:.3e} <= s={s:.3e} (x={x:.3f}, f={f:.3f})")
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    y = np.sqrt(s / Ueff)

    if not (0.0 < y < 1.0):
        msg = (f"No internal equilibrium: "
               f"y={y:.3e} not in (0,1) (x={x:.3f}, f={f:.3f})")
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    p_star = 1.0 - y

    denom = pi + 2.0 * s * (1.0 - y)

    if denom <= 0.0:
        msg = (f"No stable internal equilibrium: "
               f"denom={denom:.3e} <=0 (x={x:.3f}, f={f:.3f})")
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    n_star = 2.0 * y * (1.0 - y) / denom

    if (not np.isfinite(n_star)) or (n_star <= 0.0):
        msg = f"Invalid n*: n*={n_star} (x={x:.3f}, f={f:.3f})"
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    return n_star, p_star


def ode_full(t, y, u, s, pi, c, f):

    n, p, x = y

    fbar = fbar_two_allele(x, f)
    Ueff = u * (1.0 - fbar)

    repress = (1.0 - p)**2
    dn = Ueff * repress * n - s * n

    dp = Ueff * repress * n * (pi / 2.0 + s * p) - s * p * (1.0 - p)

    sigma = s * u * n * (1.0 - p)**2 * f - c
    dx   = sigma * x * (1.0 - x)**2

    return [dn, dp, dx]



def ode_reduced(t, x, u, s, pi, c, f):
    n_qs, p_qs = fast_equilibrium(x, f, u, s, pi, strict=True)
    sigma      = s * u * n_qs * (1.0 - p_qs)**2 * f - c
    return sigma * x * (1.0 - x)**2



f_list    = [0.01, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.0]
color_list = ["r", "b", "g", "y", "m", "c", "gray", "k"]

x_init = 0.01
t_span  = (0.0, 5e5)

plt.figure(figsize=(8, 5))

for f_i, clr in zip(f_list, color_list):
    try:
        n_star, p_star = fast_equilibrium(x_init, f_i, u, s, pi, strict=True)
    except RuntimeError as e:
        print(f"[skip  f={f_i:.2f}] {e}")
        continue

    y0 = [n_star, p_star, x_init]

    sol_full = solve_ivp(
        ode_full, t_span, y0,
        args=(u, s, pi, c, f_i),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    x_series = sol_full.y[2]

    try:
        sol_red = solve_ivp(
            lambda t, x: ode_reduced(t, x, u, s, pi, c, f_i),
            t_span,
            [x_init],
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
    except RuntimeError as e:
        print(f"[halt 1D f={f_i:.2f}] {e}")
        sol_red = None

    if sol_red is not None:
        plt.plot(
            sol_red.t,
            sol_red.y[0],
            color=clr,
            label=f"$f={f_i}$ (1D)",
            ls="-",
            lw=2,
        )

    plt.plot(
        sol_full.t,
        x_series,
        color=clr,
        label=f"$f={f_i}$ (3D)",
        ls=":",
        lw=4.5,
    )

plt.xlabel("Time", size=15)
plt.ylabel("$x$ (KZFP frequency)", size=15)
plt.tick_params(labelsize=13, width=1.5, length=6)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.04),
    fontsize=12,
    ncol=1,
    handlelength=1.5,
    frameon=False,
)
plt.tight_layout()
plt.savefig('figA2.pdf', format="pdf", bbox_inches='tight')
plt.show()
