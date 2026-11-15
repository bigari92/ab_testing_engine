#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUPED must remove rho-squared of the variance, and must not move the answer.

Two claims, and the second matters more than the first.

1. THE PAYOFF IS WHAT THEORY SAYS. Variance after adjustment should be
   (1 - rho^2) times variance before, for any rho. Checked across a grid
   from 0.1 to 0.95, because a method that works at one correlation and
   not another has a bug that one test would miss.

2. THE ESTIMATE IS STILL UNBIASED, AND THE INTERVAL STILL COVERS. This is
   the claim that stops CUPED being dangerous. A technique that narrows
   intervals is trivial to write and catastrophic if it narrows them
   around the wrong number. So: simulate many experiments with a known
   effect, run the adjusted test, and check that the mean estimate equals
   the truth and that 95% of the intervals contain it. If the adjustment
   were absorbing part of the treatment effect, the bias check would fail
   and the coverage check would fail with it.

3. AND THE FAILURE MODE IS REPRODUCED ON PURPOSE. Hand CUPED a covariate
   measured AFTER the treatment started — one the treatment itself moves —
   and the estimate must go visibly wrong. A proof suite that only shows
   the happy path does not tell you what the guard rails are for. The
   claim here is that the bias appears and is large, which is the reason
   cuped.py refuses to look clever about covariate choice and makes it the
   caller's declared responsibility.
"""
from __future__ import annotations

import numpy as np

from _harness import header

from explab.cuped import cuped_adjust, variance_reduction
from explab.inference import welch_t

CONF = 0.95


def main() -> int:
    proof = header(
        "PROOF: CUPED variance reduction",
        "the adjustment removes rho^2 of the variance and leaves the effect "
        "estimate unbiased",
    )
    rng = np.random.default_rng(5150)

    # ---- the payoff matches rho^2 across a grid ---------------------------
    print("     rho   predicted   measured   effective n gain")
    print("  " + "-" * 50)
    for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
        n = 40_000
        # Build (x, y) with exactly this correlation.
        x = rng.standard_normal(n)
        y = rho * x + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n)
        result = cuped_adjust(y[: n // 2], x[: n // 2], y[n // 2:], x[n // 2:])
        print(f"  {rho:6.2f}   {variance_reduction(rho):9.4f}   "
              f"{result.reduction:8.4f}   {result.effective_n_gain:16.2f}x")
        proof.check_close(
            f"variance reduction at rho={rho}", result.reduction,
            variance_reduction(rho), tolerance=0.01,
            detail=f"theory says 1 - rho^2 = {variance_reduction(rho):.4f}",
        )
    print()

    # ---- unbiased, and the interval still covers --------------------------
    true_effect = 0.25
    rho = 0.75
    n_sims, n_per_arm = 2000, 1200
    estimates_raw, estimates_adj = [], []
    covered_raw = covered_adj = 0
    width_raw = width_adj = 0.0
    for _ in range(n_sims):
        x = rng.standard_normal(2 * n_per_arm)
        y = rho * x + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(2 * n_per_arm)
        # The treatment effect lands on y only, never on x.
        y[n_per_arm:] += true_effect
        raw = welch_t(y[:n_per_arm], y[n_per_arm:], confidence=CONF)
        adj_data = cuped_adjust(
            y[:n_per_arm], x[:n_per_arm], y[n_per_arm:], x[n_per_arm:]
        )
        adj = welch_t(adj_data.adjusted_control, adj_data.adjusted_treatment,
                      confidence=CONF)
        estimates_raw.append(raw.estimate)
        estimates_adj.append(adj.estimate)
        covered_raw += raw.ci[0] <= true_effect <= raw.ci[1]
        covered_adj += adj.ci[0] <= true_effect <= adj.ci[1]
        width_raw += raw.ci[1] - raw.ci[0]
        width_adj += adj.ci[1] - adj.ci[0]

    proof.check_mean(
        "the CUPED-adjusted estimate is unbiased", estimates_adj, true_effect,
        detail=f"mean of {n_sims:,} simulated experiments against a truth of "
               f"{true_effect}; the unadjusted mean was "
               f"{np.mean(estimates_raw):.6f}",
    )
    proof.check_rate(
        "the CUPED-adjusted 95% interval still covers", covered_adj, n_sims,
        CONF,
        detail="this is the claim that makes the narrowing safe rather than "
               "just flattering",
    )
    narrowing = 1.0 - (width_adj / width_raw)
    proof.check_close(
        "the interval narrows by 1 - sqrt(1 - rho^2)", narrowing,
        1.0 - np.sqrt(1.0 - rho ** 2), tolerance=0.01,
        detail=f"intervals shrank {narrowing:.1%} on average; width scales "
               f"with the standard deviation, so the gain is the square root "
               f"of the variance gain",
    )

    # ---- and the power gain is real ---------------------------------------
    hits_raw = hits_adj = 0
    small_effect = 0.08
    power_sims = 1500
    for _ in range(power_sims):
        x = rng.standard_normal(2 * n_per_arm)
        y = rho * x + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(2 * n_per_arm)
        y[n_per_arm:] += small_effect
        hits_raw += welch_t(y[:n_per_arm], y[n_per_arm:]).p_value < 0.05
        a = cuped_adjust(y[:n_per_arm], x[:n_per_arm], y[n_per_arm:], x[n_per_arm:])
        hits_adj += welch_t(a.adjusted_control, a.adjusted_treatment).p_value < 0.05
    proof.check_true(
        "CUPED finds a small effect the raw test misses",
        hits_adj > hits_raw * 1.4,
        detail=f"on an effect of {small_effect} at {n_per_arm} per arm: raw "
               f"detected {hits_raw}/{power_sims} ({hits_raw / power_sims:.1%}), "
               f"adjusted detected {hits_adj}/{power_sims} "
               f"({hits_adj / power_sims:.1%}) - same users, same effect",
    )

    # ---- the failure mode, reproduced deliberately ------------------------
    # A covariate the treatment moves. CUPED then subtracts part of the real
    # effect, and the estimate is biased toward zero.
    bad_estimates = []
    for _ in range(600):
        x = rng.standard_normal(2 * n_per_arm)
        y = rho * x + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(2 * n_per_arm)
        y[n_per_arm:] += true_effect
        # The treatment also moves the "covariate" - which makes it not one.
        x_contaminated = x.copy()
        x_contaminated[n_per_arm:] += true_effect / rho
        a = cuped_adjust(
            y[:n_per_arm], x_contaminated[:n_per_arm],
            y[n_per_arm:], x_contaminated[n_per_arm:],
        )
        bad_estimates.append(
            welch_t(a.adjusted_control, a.adjusted_treatment).estimate
        )
    bias = float(np.mean(bad_estimates)) - true_effect
    proof.check_true(
        "a post-treatment covariate biases the estimate toward zero",
        abs(bias) > 0.5 * true_effect,
        detail=f"mean estimate {np.mean(bad_estimates):.6f} against a truth "
               f"of {true_effect}: bias {bias:+.6f}, which is "
               f"{abs(bias) / true_effect:.0%} of the real effect. The "
               f"interval was narrow and the number was wrong, and nothing "
               f"in the output said so. This is why the covariate being "
               f"pre-treatment is the caller's declared responsibility.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
