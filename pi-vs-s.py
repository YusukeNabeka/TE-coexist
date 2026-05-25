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

u_fixed  = 0.01
f_fixed = 0.75
c_fixed  = 0.00001

x0_small = 0.01
t_end = 2e8
tol_speed = 1e-10
min_time = 50
grid = 50


def p_star_func_resident(s, u):
    r = s / (u + 1e-22)
    return np.nan if not (0 <= r < 1) else 1 - np.sqrt(r)

def n_star_func_resident(s, u, pi):
    ps = p_star_func_resident(s, u)
    denom = pi + 2.0 * s * ps
    n_star = 2.0 * ps * (1.0 - ps) / denom
    return n_star

def check_boundary_stability(s, u):
    return (0 < s < u)

def check_kzfp_invasion(s, u, pi, f, c):
    if not check_boundary_stability(s, u):
        return False
    ps = p_star_func_resident(s, u)
    ns = n_star_func_resident(s, u, pi)
    if np.isnan(ps) or np.isnan(ns):
        return False
    sigma_ben = s * u * ns * f * (1.0 - ps)**2
    return sigma_ben > c



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


def steady_event_log(t, y, pi, s, u, f, c):
    if t < min_time:
        return 1.0
    dn, dp, dx = system_log(t, y, pi, s, u, f, c)
    return max(abs(dn), abs(dp), abs(dx)) - tol_speed

steady_event_log.terminal = True
steady_event_log.direction = -1


def jacobian(vars, pi, s, u, f, c):
    n, p, x_m = vars

    f_bar = x_m * (2.0 - x_m) * f

    one_minus_f = 1.0 - f_bar
    one_minus_p = 1.0 - p
    one_minus_x = 1.0 - x_m

    R = one_minus_f * one_minus_p**2


    dR_dp = -2.0 * one_minus_f * one_minus_p
    dR_dx = -2.0 * f * one_minus_x * one_minus_p**2

    A = pi / 2.0 + s * p

    J11 = u * R - s
    J12 = n * u * dR_dp
    J13 = n * u * dR_dx

    J21 = u * R * A
    J22 = u * n * (dR_dp * A + R * s) - s * (1.0 - 2.0 * p)
    J23 = u * n * dR_dx * A

    sigma_benefit_val = s * u * n * f * one_minus_p**2
    sigma = sigma_benefit_val - c

    J31 = s * u * f * one_minus_p**2 * x_m * one_minus_x**2
    J32 = -2.0 * s * u * n * f * one_minus_p * x_m * one_minus_x**2
    J33 = sigma * one_minus_x * (1.0 - 3.0 * x_m)

    return np.array([
        [J11, J12, J13],
        [J21, J22, J23],
        [J31, J32, J33]
    ])



pi_vals = np.logspace(-4, 0, grid)
s_vals  = np.logspace(-4, -1.8, grid)

points = [(pi, s) for pi in pi_vals for s in s_vals]

results = []
for pi, s in tqdm(points, desc="Analyzing (pi, s) plane"):
    n0 = n_star_func_resident(s, u_fixed, pi)
    p0 = p_star_func_resident(s, u_fixed)
    if np.isnan(n0) or np.isnan(p0):
        continue


    sol = solve_ivp(
        system_log,
        (0, t_end),
        [n0, p0, x0_small],
        args=(pi, s, u_fixed, f_fixed, c_fixed),
        events=steady_event_log,
        rtol=1e-10,
        atol=1e-12
    )

    nT, pT, xT = sol.y[:, -1]

   
    p_eq  = pT
    x_eq = xT
    n_eq  = nT

    invasion = check_kzfp_invasion(s, u_fixed, pi, f_fixed, c_fixed)


    if sol.success:
        try:
            J = jacobian((n_eq, p_eq, x_eq), pi, s, u_fixed, f_fixed, c_fixed)
            eigenvalues = eigvals(J)

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
        'pi': pi, 's': s,
        'x': x_eq, 'p': p_eq, 'n': n_eq,
        'stability': stability_code, 'time': sol.t[-1]
    })

filename = f'pi-vs-s_u{u_fixed:.1e}_f{f_fixed:.1e}_c{c_fixed:.1e}.pkl'

print(f"\nSaving results to file {filename} ...")
try:
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print("Saving finished successfully.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")