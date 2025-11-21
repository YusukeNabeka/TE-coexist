import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
from numpy.linalg import eigvals
from matplotlib.transforms import blended_transform_factory
from tqdm import tqdm
import math
import sys
import pickle

# --- 1. Fixed parameters ---
pi_fixed  = 0.01
fm_fixed  = 0.6
c_fixed   = 0.00001
x0_small  = 0.01
t_end, tol_speed, min_time = 1000000000, 1e-10, 50
grid = 50
tol_freq = 0.1

# --- 2. Helper functions ---
def p_star_func_resident(s, u):
    r = s / (u + 1e-22)
    return np.nan if not (0 <= r < 1) else 1 - np.sqrt(r)

def n_star_func_resident(s, u, pi):
    ps = p_star_func_resident(s, u)
    if np.isnan(ps) or pi <= 1e-22:
        return np.nan
    denom = 1 - 2 * s * ps
    if abs(denom) < 1e-12:
        return np.nan
    n_star = 2 * s * ps / (pi * np.sqrt(u * s + 1e-22) * denom)
    return np.nan if (n_star < 0 or np.isnan(n_star)) else n_star

def check_boundary_stability(s, u):
    if not (0 < s < u):
        return False
    ps = p_star_func_resident(s, u)
    if np.isnan(ps):
        return False
    return (1 - 2 * s * ps > 1e-9) and (1 - 2 * s * ps**2 > 1e-9)

def check_kzfp_invasion(s, u, pi, fm, c):
    if not check_boundary_stability(s, u):
        return False
    ps = p_star_func_resident(s, u)
    denom = pi * np.sqrt(u * s + 1e-22) * (1 - 2 * s * ps)
    if abs(denom) < 1e-22:
        return False
    return (2 * (s**3) * ps * fm) / denom > c

# --- 2.5: Safe exp/log ---
Z_MIN_FOR_EXP = -700.0   # exp(-700) ~ 5e-305
Z_MAX_FOR_EXP =  50.0
def safe_exp(z):
    return np.exp(np.clip(z, Z_MIN_FOR_EXP, Z_MAX_FOR_EXP))

# --- 3. ODE system in log-n and event function ---
def system_log(t, y, pi, s, u, fm, c):
    """
    State y = [z, p, x], where z = log n.
    """
    z, p, xm = y
    p  = np.clip(p, 0.0, 1.0)
    xm = np.clip(xm, 0.0, 1.0)
    n  = safe_exp(z)

    f_bar  = xm * (2 - xm) * fm
    repress = (1 - f_bar) * (1 - p)**2

    dz_dt = u * repress - s

    sel_den = 1 - 2 * s * p
    if abs(sel_den) > 1e-12:
        sel_term = s * p * (1 - p) / sel_den
    else:
        sel_term = np.sign(s * p * (1 - p)) * 1e12
    dp_dt = (pi/2) * u * repress * n - sel_term

    sigma_ben = s * u * n * fm * (1 - p)**2
    dxm_dt = (sigma_ben - c) * xm * (1 - xm)**2

    return [dz_dt, dp_dt, dxm_dt]

def steady_event_log(t, y, pi, s, u, fm, c):
    if t < min_time:
        return 1.0
    dz, dp, dx = system_log(t, y, pi, s, u, fm, c)
    return max(abs(dz), abs(dp), abs(dx)) - tol_speed

steady_event_log.terminal = True
steady_event_log.direction = -1

# --- 3.5: Jacobians ---
def jacobian_3d(vars, pi, s, u, fm, c):
    n, p, x_m = vars
    p_calc   = np.clip(p, 0, 1)
    x_m_calc = np.clip(x_m, 0, 1)
    n_calc   = max(0.0, n)
    f_bar = x_m_calc * (2 - x_m_calc) * fm

    J11 = u * (1 - f_bar) * (1 - p_calc)**2 - s
    J12 = -2 * u * n_calc * (1 - f_bar) * (1 - p_calc)
    J13 = -2 * u * n_calc * fm * (1 - x_m_calc) * (1 - p_calc)**2

    J21 = (pi/2) * u * (1 - f_bar) * (1 - p_calc)**2
    term_p_deriv_num = s * (1 - 2 * p_calc + 2 * s * p_calc**2)
    term_p_deriv_den = (1 - 2 * s * p_calc)**2
    if abs(term_p_deriv_den) < 1e-12:
        term_p_deriv_den = np.sign(term_p_deriv_den) * 1e-12 + 1e-13
    J22 = -pi*u*n_calc*(1 - f_bar)*(1 - p_calc) - term_p_deriv_num/term_p_deriv_den
    J23 = -pi*u*n_calc*fm*(1 - x_m_calc)*(1 - p_calc)**2

    sigma_benefit_val = s*u*n_calc*fm*(1 - p_calc)**2
    J31 = (s*u*fm*(1 - p_calc)**2)*x_m_calc*((1 - x_m_calc)**2)
    J32 = (-2*s*u*n_calc*fm*(1 - p_calc))*x_m_calc*((1 - x_m_calc)**2)
    J33 = (sigma_benefit_val - c)*(1 - x_m_calc)*(1 - 3*x_m_calc)

    return np.array([[J11, J12, J13],
                     [J21, J22, J23],
                     [J31, J32, J33]])

def jacobian_2d_at_x0(n, p, pi, s, u):
    """
    Jacobian of the (n, p) subsystem at x=0 (no KZFP), evaluated at (n_eq, p_eq) with f_bar = 0.
    """
    p = np.clip(p, 0.0, 1.0)
    n = max(0.0, n)

    J11 = u * (1 - p)**2 - s
    J12 = -2 * u * n * (1 - p)

    J21 = (pi/2) * u * (1 - p)**2
    term_p_deriv_num = s * (1 - 2 * p + 2 * s * p**2)
    term_p_deriv_den = (1 - 2 * s * p)**2
    if abs(term_p_deriv_den) < 1e-12:
        term_p_deriv_den = np.sign(term_p_deriv_den) * 1e-12 + 1e-13
    J22 = -pi*u*n*(1 - p) - term_p_deriv_num/term_p_deriv_den

    return np.array([[J11, J12],
                     [J21, J22]])

# --- 4. Main computation over (u, s) plane ---
u_vals = np.logspace(-4, 0, grid)
s_vals = np.logspace(-4, 0, grid)
points = [(u, s) for u in u_vals for s in s_vals]

results = []
for u, s in tqdm(points, desc="Analyzing (u, s) plane"):
    n0 = n_star_func_resident(s, u, pi_fixed)
    p0 = p_star_func_resident(s, u)
    if np.isnan(n0) or np.isnan(p0):
        continue

    z0 = np.log(max(n0, 1e-300))

    sol = solve_ivp(
        system_log,
        (0, t_end),
        [z0, p0, x0_small],
        args=(pi_fixed, s, u, fm_fixed, c_fixed),
        events=steady_event_log,
        rtol=1e-10,
        atol=1e-12
    )

    zT, pT, xT = sol.y[:, -1]
    nT = float(safe_exp(zT))

    p_eq  = 0.0 if pT < 0 else (1.0 if pT > 1 else pT)
    xm_eq = 0.0 if xT < 0 else (1.0 if xT > 1 else xT)
    n_eq  = max(0.0, nT)

    invasion = check_kzfp_invasion(s, u, pi_fixed, fm_fixed, c_fixed)
    if not invasion:
        xm_eq = 0.0

    if sol.success:
        try:
            if invasion and xm_eq > 0:
                J = jacobian_3d((n_eq, p_eq, xm_eq), pi_fixed, s, u, fm_fixed, c_fixed)
                eigenvalues = eigvals(J)
            else:
                J2 = jacobian_2d_at_x0(n_eq, p_eq, pi_fixed, s, u)
                eigenvalues = eigvals(J2)

            real_parts = np.real(eigenvalues)
            max_real = np.max(real_parts)
            if max_real < 0:
                stability_code = 1 if np.any(np.abs(np.imag(eigenvalues)) > 0) else 0
            elif abs(max_real) == 0:
                stability_code = 2
            else:
                stability_code = 3 if np.any(real_parts < 0) else 4
        except np.linalg.LinAlgError:
            stability_code = 5
    else:
        stability_code = 5

    results.append({
        'u': u, 's': s,
        'xm': xm_eq, 'p': p_eq, 'n': n_eq,
        'stability': stability_code, 'time': sol.t[-1]
    })

# --- Save results with pickle ---
filename = f'u-vs-s_pi{pi_fixed:.1e}_f{fm_fixed:.1e}_c{c_fixed:.1e}.pkl'
print(f"\nSaving results to file {filename} ...")
try:
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print("Saving finished successfully.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
