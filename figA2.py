import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


u  = 1e-2
s  = 1e-3
pi = 1e-2
c  = 1e-5



def fbar_two_allele(x1, f1):
    return x1 * (2.0 - x1) * f1


def fast_equilibrium(x1, f1, u, s, pi, strict=False):

    fb   = fbar_two_allele(x1, f1)
    Ueff = u * (1.0 - fb)

    # 0 < n^*
    if Ueff <= 0.0 or Ueff <= s:
        msg = (f"No internal TE equilibrium: "
               f"U_eff={Ueff:.3e} <= s={s:.3e} (x1={x1:.3f}, f1={f1:.3f})")
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    y = np.sqrt(s / Ueff)

    # 0 < p^* < 1
    if not (0.0 < y < 1.0):
        msg = (f"No internal equilibrium: "
               f"y={y:.3e} not in (0,1) (x1={x1:.3f}, f1={f1:.3f})")
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    p_star = 1.0 - y

    denom = 1.0 - 2.0 * s + 2.0 * s * y
    if denom <= 0.0:
        msg = (f"No stable internal equilibrium: "
               f"denom=1-2s+2sy={denom:.3e} <=0 (x1={x1:.3f}, f1={f1:.3f})")
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    n_star = (2.0 / pi) * (y * (1.0 - y)) / denom
    if (not np.isfinite(n_star)) or (n_star <= 0.0):
        msg = f"Invalid n*: n*={n_star} (x1={x1:.3f}, f1={f1:.3f})"
        if strict:
            raise RuntimeError(msg)
        return np.nan, np.nan

    return n_star, p_star


def ode_full(t, y, u, s, pi, c, f1):

    n, p, x1 = y

    fbar = fbar_two_allele(x1, f1)
    Ueff = u * (1.0 - fbar)

    # dn/dt
    dn = Ueff * (1.0 - p)**2 * n - s * n

    # dp/dt
    den_sel = 1.0 - 2.0 * s * p
    if abs(den_sel) < 1e-12:
        den_sel = np.sign(den_sel) * 1e-12
    dp = (pi / 2.0) * Ueff * (1.0 - p)**2 * n - (s * p * (1.0 - p)) / den_sel

    # dx1/dt
    sigma = s * u * n * (1.0 - p)**2 * f1 - c
    dx1   = sigma * x1 * (1.0 - x1)**2

    return [dn, dp, dx1]



def ode_reduced(t, x1, u, s, pi, c, f1):
    n_qs, p_qs = fast_equilibrium(x1, f1, u, s, pi, strict=True)
    sigma      = s * u * n_qs * (1.0 - p_qs)**2 * f1 - c
    return sigma * x1 * (1.0 - x1)**2



f1_list    = [0.01, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1.0]
color_list = ["r", "b", "g", "y", "m", "c", "gray", "k"]

x1_init = 0.01
t_span  = (0.0, 5e5)

plt.figure(figsize=(8, 5))

for f1_i, clr in zip(f1_list, color_list):
    try:
        n_star, p_star = fast_equilibrium(x1_init, f1_i, u, s, pi, strict=True)
    except RuntimeError as e:
        print(f"[skip  f1={f1_i:.2f}] {e}")
        continue

    y0 = [n_star, p_star, x1_init]

    sol_full = solve_ivp(
        ode_full, t_span, y0,
        args=(u, s, pi, c, f1_i),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    x1_series = sol_full.y[2]

    try:
        sol_red = solve_ivp(
            lambda t, x: ode_reduced(t, x, u, s, pi, c, f1_i),
            t_span,
            [x1_init],
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
    except RuntimeError as e:
        print(f"[halt 1D f1={f1_i:.2f}] {e}")
        sol_red = None

    if sol_red is not None:
        plt.plot(
            sol_red.t,
            sol_red.y[0],
            color=clr,
            label=f"$f_1={f1_i}$ (1D)",
            ls="-",
            lw=2,
        )

    plt.plot(
        sol_full.t,
        x1_series,
        color=clr,
        label=f"$f_1={f1_i}$ (3D)",
        ls=":",
        lw=4.5,
    )

plt.xlabel("Time", size=15)
plt.ylabel("$x_1$ (KZFP frequency)", size=15)
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
plt.show()
