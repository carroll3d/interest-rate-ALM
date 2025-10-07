
import numpy as np

def vasicek_path(r0: float, a: float, b: float, sigma: float, T: float, dt: float, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = int(np.ceil(T / dt))
    t = np.linspace(0.0, dt * (N-1), N)
    r = np.empty(N)
    r[0] = r0
    sqdt = np.sqrt(dt)
    for i in range(1, N):
        dr = a * (b - r[i-1]) * dt + sigma * sqdt * rng.normal()
        r[i] = r[i-1] + dr
    return t, r

def vasicek_paths(r0: float, a: float, b: float, sigma: float, T: float, dt: float, n_paths: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    N = int(np.ceil(T / dt))
    t = np.linspace(0.0, dt * (N-1), N)
    paths = np.empty((N, n_paths))
    for j in range(n_paths):
        _, r = vasicek_path(r0, a, b, sigma, T, dt, rng=rng)
        paths[:, j] = r
    return t, paths

def vasicek_analytic_mean_var(r0: float, a: float, b: float, sigma: float, t: np.ndarray):
    mean = b + (r0 - b) * np.exp(-a * t)
    var = (sigma**2) / (2 * a) * (1 - np.exp(-2 * a * t)) if a > 0 else np.full_like(t, np.nan, dtype=float)
    return mean, var
