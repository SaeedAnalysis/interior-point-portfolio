# Interior-Point Methods for Markowitz Portfolio Optimization

Final project for **MATH 707 — Nonlinear Optimization** (Spring 2026).

This repository compares three numerical methods for solving the constrained
mean–variance portfolio problem

$$
\begin{aligned}
\min_{w \in \mathbb{R}^n}\quad & \tfrac{1}{2}\, w^{\top} \Sigma\, w \\
\text{s.t.}\quad & \mathbf{1}^{\top} w = 1, \\
& \mu^{\top} w \ge r_{\min}, \\
& w \ge 0.
\end{aligned}
$$

## Methods

1. **Projected gradient descent** — first-order baseline with Euclidean
   projection onto the feasible set computed by SLSQP.
2. **Logarithmic barrier method** — sequential centering with damped Newton
   steps and a backtracking line search that preserves strict feasibility.
3. **Primal–dual interior-point method** — single Newton step on the relaxed
   KKT system at every iteration, with a fraction-to-boundary rule.

## Experiments

| Experiment | Problem | Notes |
|---|---|---|
| **A — Simulated** | 5 synthetic assets with a fixed seed | Condition number ≈ 13.6, reproducible from `np.random.seed(42)`. |
| **B — Real data** | AAPL, GOOGL, META, AMZN, MSFT | Daily adjusted close, fixed window `2023-01-03` to `2025-12-31`, annualized with 252 trading days. |

Wall-clock times are measured on a MacBook Air 13-inch (Apple M3, 2024); the
median over five runs is reported to reduce timing noise. Iteration counts and
final iterates are deterministic.

## Repository contents

```
.
├── experiment.py   # all three solvers + simulated and real-data experiments
├── results.json    # cached output of `python experiment.py`
├── Report.pdf      # written project report
└── README.md
```

## Reproducing the results

### Requirements

- Python ≥ 3.10
- `numpy`, `scipy`
- `yfinance` (real-data experiment only); optionally `curl_cffi` to avoid
  Yahoo Finance rate limiting

Install everything with:

```bash
pip install -r requirements.txt
```

### Run

```bash
python experiment.py
```

The script prints a summary for each experiment and writes a machine-readable
copy of the results to `results.json`. A representative output is committed in
this repo for reference.

If the real-data experiment fails because Yahoo Finance is unreachable, the
simulated experiment still runs to completion — the real-data section is
guarded behind a separate `_download_prices` call.

## Sample results

From `results.json` (median of 5 runs, MacBook Air M3):

**Simulated 5-asset problem** — all three methods converge to the same
optimum $f^\star \approx 5.302 \times 10^{-3}$:

| Method | Iters | Time (s) |
|---|---:|---:|
| Projected gradient | 175 | 0.068 |
| Barrier (Newton steps) | 38 | 0.0036 |
| Primal–dual IPM | 12 | 0.00041 |

**Real-data problem (AAPL, GOOGL, META, AMZN, MSFT)** — $f^\star \approx 2.142 \times 10^{-2}$:

| Method | Iters | Time (s) |
|---|---:|---:|
| Projected gradient | 129 | 0.084 |
| Barrier (Newton steps) | 68 | 0.0028 |
| Primal–dual IPM | 13 | 0.00045 |

The primal–dual iterate matches the barrier iterate to roughly $4 \times 10^{-6}$
in both experiments, consistent with the analysis in the report.

## Report

See [`Report.pdf`](./Report.pdf) for the full write-up, including the KKT
derivation, convergence discussion, and a self-contained presentation of the
numerical results.

## License

Released under the MIT License — see [LICENSE](./LICENSE).
