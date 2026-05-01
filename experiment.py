"""Interior-point methods for Markowitz portfolio optimization.

This script reproduces the numerical results reported in the project. Three
methods are applied to the same constrained quadratic program:

    1. Projected gradient descent (first-order baseline).
    2. Logarithmic barrier method (sequential centering with Newton's method).
    3. Primal-dual interior-point method (single Newton step on the relaxed
       KKT system).

Two experiments are run:

    A. A simulated 5-asset problem with a fixed random seed.
    B. A real-data problem built from daily adjusted closing prices of
       Apple (AAPL), Alphabet (GOOGL), Meta (META), Amazon (AMZN), and
       Microsoft (MSFT) over the fixed window 2023-01-03 to 2025-12-31.

Reported wall-clock times were measured on a MacBook Air 13-inch (Apple M3,
2024). For each method the median over five repeated runs is reported in
order to reduce timing noise.

Dependencies:
    NumPy, SciPy. The real-data experiment also requires yfinance; the
    optional curl_cffi package is used to avoid Yahoo rate-limiting.

Run:
    python3 experiment.py
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize as sp_minimize


# Fixed window used by the real-data experiment.
REAL_DATA_START = "2023-01-03"
REAL_DATA_END = "2025-12-31"
TRADING_DAYS_PER_YEAR = 252
RESULTS_PATH = Path(__file__).with_name("results.json")


# =========================================================================
# Problem container
# =========================================================================

@dataclass
class Portfolio:
    """Data that defines a mean-variance portfolio problem."""
    mu: np.ndarray           # expected returns, shape (n,)
    Sigma: np.ndarray        # covariance matrix, shape (n, n)
    r_min: float             # target return on the same horizon as mu
    names: Sequence[str]     # asset names, used only for printing

    @property
    def n(self) -> int:
        return len(self.mu)


def objective(p: Portfolio, w: np.ndarray) -> float:
    """Half-variance objective f(w) = (1/2) w^T Sigma w."""
    return 0.5 * w @ p.Sigma @ w


# =========================================================================
# Method 1: Logarithmic barrier method
# =========================================================================

def _barrier_value(p, w, t):
    s = p.mu @ w - p.r_min
    if np.any(w <= 0) or s <= 0:
        return np.inf
    return t * 0.5 * w @ p.Sigma @ w - np.sum(np.log(w)) - np.log(s)


def _barrier_grad(p, w, t):
    s = p.mu @ w - p.r_min
    return t * p.Sigma @ w - 1.0 / w - p.mu / s


def _barrier_hess(p, w, t):
    s = p.mu @ w - p.r_min
    return t * p.Sigma + np.diag(1.0 / w**2) + np.outer(p.mu, p.mu) / s**2


def _newton_step(p, w, t):
    """Newton direction for the barrier subproblem with equality 1^T w = 1."""
    n = p.n
    g = _barrier_grad(p, w, t)
    H = _barrier_hess(p, w, t)
    ones = np.ones(n)
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, :n] = H
    kkt[:n, n] = ones
    kkt[n, :n] = ones
    rhs = np.zeros(n + 1)
    rhs[:n] = -g
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n]


def barrier_method(p: Portfolio, w0, t0=1.0, gamma=10.0, eps=1e-8,
                   newton_tol=1e-10, max_outer=100, max_newton=50):
    """Solve the portfolio QP with a logarithmic barrier method.

    Each centering step uses a damped Newton update with a backtracking line
    search that preserves strict feasibility and enforces sufficient decrease
    of the barrier objective.
    """
    w = w0.copy()
    t = t0
    outer = 0
    total_newton = 0
    log = []

    while True:
        gap = (p.n + 1) / t
        steps = 0

        for _ in range(max_newton):
            dw = _newton_step(p, w, t)

            step = 1.0
            phi_w = _barrier_value(p, w, t)
            slope = _barrier_grad(p, w, t) @ dw
            for _ in range(60):
                w_try = w + step * dw
                s_try = p.mu @ w_try - p.r_min
                if np.all(w_try > 0) and s_try > 0:
                    if _barrier_value(p, w_try, t) <= phi_w + 0.01 * step * slope:
                        break
                step *= 0.5

            w = w + step * dw
            steps += 1
            if np.linalg.norm(step * dw) < newton_tol:
                break

        total_newton += steps
        log.append({"outer": outer, "t": t, "newton": steps,
                    "obj": objective(p, w), "gap": gap})

        if gap < eps:
            break
        t *= gamma
        outer += 1
        if outer >= max_outer:
            break

    return w, log, total_newton


# =========================================================================
# Method 2: Primal-Dual Interior-Point Method
# =========================================================================
#
# We work with primal variables w and dual variables (nu, eta, lambda) at
# once and take a single Newton step on the relaxed KKT system at every
# iteration. The relaxed complementarity conditions are
#     eta * (mu^T w - r_min) = tau,
#     w_i * lambda_i = tau,    for i = 1, ..., n,
# where tau > 0 is driven to zero. The duality measure d_k is defined below.

def primal_dual_ipm(p: Portfolio, w0, sigma=0.1, tol=1e-10,
                    max_iter=100, safety=0.995):
    """Path-following primal-dual interior-point method for the portfolio QP."""
    n = p.n
    w = w0.copy()
    nu = 0.0
    eta = 1.0
    lam = np.ones(n)
    log = []

    for k in range(max_iter):
        s_ret = p.mu @ w - p.r_min
        d_k = (eta * s_ret + w @ lam) / (n + 1)
        log.append({"iter": k, "d_k": d_k, "obj": objective(p, w)})
        if d_k < tol:
            break

        tau = sigma * d_k

        r_dual = p.Sigma @ w + nu - eta * p.mu - lam
        r_eq = np.sum(w) - 1.0
        r_ret = eta * s_ret - tau
        r_comp = w * lam - tau

        size = 2 * n + 2
        J = np.zeros((size, size))
        J[:n, :n] = p.Sigma
        J[:n, n] = 1.0
        J[:n, n + 1] = -p.mu
        J[:n, n + 2:] = -np.eye(n)
        J[n, :n] = 1.0
        J[n + 1, :n] = eta * p.mu
        J[n + 1, n + 1] = s_ret
        J[n + 2:, :n] = np.diag(lam)
        J[n + 2:, n + 2:] = np.diag(w)

        rhs = -np.concatenate([r_dual, [r_eq, r_ret], r_comp])
        delta = np.linalg.solve(J, rhs)

        dw = delta[:n]
        dnu = delta[n]
        deta = delta[n + 1]
        dlam = delta[n + 2:]

        # Fraction-to-boundary rule keeps w, lambda, eta, and the return
        # slack strictly positive at every iteration.
        alpha = 1.0
        for val, step in [(w, dw), (lam, dlam)]:
            neg = step < 0
            if np.any(neg):
                alpha = min(alpha, safety * np.min(-val[neg] / step[neg]))
        if deta < 0:
            alpha = min(alpha, safety * (-eta / deta))
        ds_ret = p.mu @ dw
        if ds_ret < 0:
            alpha = min(alpha, safety * (-s_ret / ds_ret))

        w = w + alpha * dw
        nu = nu + alpha * dnu
        eta = eta + alpha * deta
        lam = lam + alpha * dlam

    return w, log


# =========================================================================
# Method 3: Projected gradient descent
# =========================================================================

def _project(p: Portfolio, v):
    """Euclidean projection of v onto the feasible set, computed by SLSQP."""
    res = sp_minimize(
        lambda w: 0.5 * np.sum((w - v) ** 2),
        x0=np.ones(p.n) / p.n,
        jac=lambda w: w - v,
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: p.mu @ w - p.r_min},
        ],
        bounds=[(0, None)] * p.n,
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    return res.x


def projected_gradient(p: Portfolio, w0, tol=1e-8, max_iter=10000):
    """Projected gradient descent with step size 1 / lambda_max(Sigma)."""
    eigvals = np.linalg.eigvalsh(p.Sigma)
    alpha = 1.0 / eigvals[-1]

    w = w0.copy()
    iters = 0
    for _ in range(max_iter):
        grad = p.Sigma @ w
        w_new = _project(p, w - alpha * grad)
        iters += 1
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w, iters


# =========================================================================
# Runner
# =========================================================================

def time_it(fn: Callable, *args, trials: int = 5, **kwargs):
    """Run a function several times and return (result, median_seconds).

    Each method is deterministic, so iteration counts and final iterates do
    not change between trials. Only wall-clock time varies; reporting the
    median reduces timing noise.
    """
    times = []
    out = None
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def run_all(p: Portfolio, label: str):
    """Run all three methods on the same problem and return a summary dict."""
    print("\n" + "=" * 72)
    print(f"EXPERIMENT: {label}")
    print("=" * 72)

    eigvals = np.linalg.eigvalsh(p.Sigma)
    cond = float(eigvals[-1] / eigvals[0])
    w_unif = np.ones(p.n) / p.n

    print(f"Assets            : {list(p.names)}")
    print(f"Expected returns  : {np.round(p.mu, 4)}")
    print(f"Condition number  : {cond:.2f}")
    print(f"Target return     : {p.r_min:.4f}")
    print(f"Uniform return    : {p.mu @ w_unif:.4f}")
    print(f"Slater satisfied  : {bool(p.mu @ w_unif > p.r_min)}")

    w0 = np.ones(p.n) / p.n

    (w_bar, _, bar_newton), t_bar = time_it(
        barrier_method, p, w0, t0=1.0, gamma=10.0, eps=1e-8
    )
    (w_pd, pd_log), t_pd = time_it(
        primal_dual_ipm, p, w0, sigma=0.1, tol=1e-10
    )
    (w_pg, pg_iters), t_pg = time_it(
        projected_gradient, p, w0, tol=1e-8, max_iter=10000
    )

    print("\nMethod                        Iters    Time (s)     Objective")
    print("-" * 68)
    print(f"Projected gradient descent    {pg_iters:>5}    "
          f"{t_pg:>8.4f}    {objective(p, w_pg):.6e}")
    print(f"Barrier method                {bar_newton:>5}    "
          f"{t_bar:>8.4f}    {objective(p, w_bar):.6e}")
    print(f"Primal-dual IPM               {len(pd_log) - 1:>5}    "
          f"{t_pd:>8.4f}    {objective(p, w_pd):.6e}")

    print("\nWeights (barrier)             : "
          f"{np.array2string(w_bar, precision=4)}")
    print(f"||w_pd - w_bar||              : {np.linalg.norm(w_pd - w_bar):.2e}")
    print(f"||w_pg - w_bar||              : {np.linalg.norm(w_pg - w_bar):.2e}")

    return {
        "label": label,
        "cond_Sigma": cond,
        "mu": p.mu.tolist(),
        "r_min": p.r_min,
        "names": list(p.names),
        "barrier": {
            "iters": int(bar_newton),
            "time_s": t_bar,
            "obj": objective(p, w_bar),
            "w": w_bar.tolist(),
        },
        "primal_dual": {
            "iters": len(pd_log) - 1,
            "time_s": t_pd,
            "obj": objective(p, w_pd),
            "w": w_pd.tolist(),
            "dist_to_barrier": float(np.linalg.norm(w_pd - w_bar)),
        },
        "projected_gradient": {
            "iters": int(pg_iters),
            "time_s": t_pg,
            "obj": objective(p, w_pg),
            "w": w_pg.tolist(),
            "dist_to_barrier": float(np.linalg.norm(w_pg - w_bar)),
        },
    }


# =========================================================================
# Experiment A: simulated data
# =========================================================================

def make_simulated() -> Portfolio:
    """Return the simulated 5-asset problem used in Section 5.1.

    The legacy NumPy RandomState API is used so that the seed reproduces the
    exact covariance matrix reported in the project (condition number 13.6).
    """
    np.random.seed(42)
    n = 5
    mu = np.array([0.12, 0.10, 0.07, 0.03, 0.05])
    A = np.random.randn(n, n)
    Sigma = A.T @ A / 100 + 0.01 * np.eye(n)
    return Portfolio(mu=mu, Sigma=Sigma, r_min=0.06,
                     names=[f"A{i+1}" for i in range(n)])


# =========================================================================
# Experiment B: real market data
# =========================================================================

def _download_prices(tickers, start, end):
    """Download daily adjusted closing prices over a fixed [start, end] window.

    The yfinance import is performed inside the function so that the
    simulated experiment can run without yfinance installed. The optional
    curl_cffi session is used when available to avoid rate limiting.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "The real-data experiment requires the 'yfinance' package. "
            "Install it with `pip install yfinance` (and optionally "
            "`pip install curl_cffi` to avoid Yahoo rate limiting)."
        ) from exc

    try:
        from curl_cffi import requests as cr
        session = cr.Session(impersonate="chrome")
    except ImportError:
        session = None

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        session=session,
    )
    closes = raw["Close"] if "Close" in raw.columns.levels[0] else raw
    return closes[list(tickers)].dropna()


def make_real(tickers, start=REAL_DATA_START, end=REAL_DATA_END,
              r_min_frac=0.5):
    """Build an annualized mean-variance problem from real daily prices.

    Daily simple returns are scaled by TRADING_DAYS_PER_YEAR so that mu and
    Sigma are on the same annualized horizon.
    """
    prices = _download_prices(tickers, start, end)
    daily = prices.pct_change().dropna()
    mu = daily.mean().to_numpy() * TRADING_DAYS_PER_YEAR
    Sigma = daily.cov().to_numpy() * TRADING_DAYS_PER_YEAR

    r_min = r_min_frac * float(np.max(mu))
    return Portfolio(mu=mu, Sigma=Sigma, r_min=r_min,
                     names=list(tickers)), prices.index


# =========================================================================
# Entry point
# =========================================================================

def main():
    print("Interior-point methods: simulated and real-data experiments.")
    print("Hardware: MacBook Air 13-inch, M3, 2024.\n")

    sim_problem = make_simulated()
    sim_result = run_all(sim_problem, "Simulated 5-asset problem")

    tickers = ("AAPL", "GOOGL", "META", "AMZN", "MSFT")
    real_problem, dates = make_real(tickers,
                                    start=REAL_DATA_START,
                                    end=REAL_DATA_END,
                                    r_min_frac=0.5)
    print(f"\nReal data window  : {dates.min().date()} to {dates.max().date()}")
    real_result = run_all(
        real_problem,
        f"Real data: {', '.join(tickers)} ({REAL_DATA_START} to {REAL_DATA_END})",
    )
    real_result["window"] = {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
    }

    payload = {
        "hardware": "MacBook Air 13-inch, M3, 2024",
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "simulated": sim_result,
        "real": real_result,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
