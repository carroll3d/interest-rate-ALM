
import numpy as np

def cir_path(r0: float, kappa: float, theta: float, sigma: float, T: float, dt: float, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = int(np.ceil(T / dt))
    t = np.linspace(0.0, dt*(N-1), N)
    r = np.empty(N)
    r[0] = max(r0, 0.0)
    sqdt = np.sqrt(dt)
    for i in range(1, N):
        r_prev = max(r[i-1], 0.0)
        dr = kappa * (theta - r_prev) * dt + sigma * np.sqrt(max(r_prev, 0.0)) * sqdt * rng.normal()
        r[i] = max(r_prev + dr, 0.0)
    return t, r

def cir_paths(r0: float, kappa: float, theta: float, sigma: float, T: float, dt: float, n_paths: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    N = int(np.ceil(T / dt))
    t = np.linspace(0.0, dt*(N-1), N)
    paths = np.empty((N, n_paths))
    for j in range(n_paths):
        _, r = cir_path(r0, kappa, theta, sigma, T, dt, rng=rng)
        paths[:, j] = r
    return t, paths

def cir_analytic_mean_var(r0: float, kappa: float, theta: float, sigma: float, t: np.ndarray):
    expk = np.exp(-kappa * t)
    mean = theta + (r0 - theta) * expk
    if kappa > 0:
        var = (sigma**2) * (r0 * expk * (1 - expk) / kappa + theta * (1 - expk)**2 / (2 * kappa))
    else:
        var = np.full_like(t, np.nan, dtype=float)
    return mean, var
