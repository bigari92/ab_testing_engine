#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A 95% interval must contain the true effect 95% of the time. Measured.

COVERAGE IS THE PROPERTY THAT MATTERS
-------------------------------------
A confidence interval is not "a range the effect is probably in". It is a
procedure with a frequency guarantee: repeat the experiment many times and
95% of the intervals the procedure produces will contain the true value.
That is checkable, and almost nobody checks it.

Here it is checked for every interval the library produces, against known
truths, with the effect deliberately non-zero so that coverage is tested
where it is used rather than only at the null.

UNDER-COVERAGE AND OVER-COVERAGE ARE DIFFERENT FAILURES
    under  (say 90% instead of 95%) the interval is too narrow, decisions
           are made on evidence that is not there
    over   (say 99% instead of 95%) the interval is too wide, real effects
           are called inconclusive, sample size is wasted

The bootstrap is expected to be very slightly under-covering at small n,
and is tested at a sample size where that is visible, with the claim
stated as a bound rather than an equality.
"""
from __future__ import annotations

import numpy as np

from _harness import header

from explab.inference import (bootstrap_diff, ratio_delta,
                              two_proportion_from_arrays, welch_t)

CONF = 0.95
N_SIMS = 4000


def main() -> int:
    proof = header(
        "PROOF: confidence interval coverage",
        f"a {CONF:.0%} interval contains the true effect {CONF:.0%} of the time",
    )
    rng = np.random.default_rng(31337)

    # ---- difference of means, real effect --------------------------------
    true_effect = 0.4
    covered = 0
    for _ in range(N_SIMS):
        c = rng.normal(10.0, 3.0, 800)
        t = rng.normal(10.0 + true_effect, 3.0, 800)
        lo, hi = welch_t(c, t, confidence=CONF).ci
        covered += lo <= true_effect <= hi
    proof.check_rate(
        "welch_t interval, equal variances, true effect +0.4",
        covered, N_SIMS, CONF,
    )

    # Unequal variances and unequal sizes, where the pooled interval fails.
    covered = 0
    for _ in range(N_SIMS):
        c = rng.normal(10.0, 1.0, 300)
        t = rng.normal(10.0 + true_effect, 4.0, 1200)
        lo, hi = welch_t(c, t, confidence=CONF).ci
        covered += lo <= true_effect <= hi
    proof.check_rate(
        "welch_t interval, 4x variance ratio, 4x size ratio",
        covered, N_SIMS, CONF,
    )

    # ---- difference of proportions ---------------------------------------
    p_c, effect = 0.08, 0.012
    covered = 0
    for _ in range(N_SIMS):
        c = (rng.random(4000) < p_c).astype(float)
        t = (rng.random(4000) < p_c + effect).astype(float)
        lo, hi = two_proportion_from_arrays(c, t, confidence=CONF).ci
        covered += lo <= effect <= hi
    proof.check_rate(
        "two_proportion unpooled interval, 8% base, +1.2pp effect",
        covered, N_SIMS, CONF,
        detail="the interval uses the unpooled SE while the p-value uses the "
               "pooled one; this is the check that the unpooled choice is "
               "right for the interval",
    )

    # ---- ratio metric ------------------------------------------------------
    # True CTR effect built into per-user rates, so the true difference in
    # the ratio of sums is known exactly.
    covered = 0
    ratio_sims = 3000
    base, bump = 0.11, 0.01
    for _ in range(ratio_sims):
        n = 1500
        q = rng.standard_normal(2 * n)
        imp = (rng.poisson(np.exp(1.6 + 0.5 * q)) + 1).astype(float)
        ctr = rng.beta(base * 8, (1 - base) * 8, size=2 * n)
        ctr_t = np.clip(ctr[n:] + bump, 1e-6, 1 - 1e-6)
        clicks_c = rng.binomial(imp[:n].astype(int), ctr[:n]).astype(float)
        clicks_t = rng.binomial(imp[n:].astype(int), ctr_t).astype(float)
        lo, hi = ratio_delta(
            clicks_c, imp[:n], clicks_t, imp[n:], confidence=CONF).ci
        covered += lo <= bump <= hi
    proof.check_rate(
        "ratio_delta interval, +1pp CTR effect applied per user",
        covered, ratio_sims, CONF,
        detail="the estimand is the ratio of sums; the per-user bump makes "
               "the true difference exactly +0.01",
    )

    # ---- bootstrap ---------------------------------------------------------
    # Heavy-tailed revenue where the normal approximation is still arriving.
    covered = 0
    boot_sims = 600
    sigma = 1.2
    mu_c, mu_t = 1.0, 1.15
    mean_c = np.exp(mu_c + sigma ** 2 / 2)
    mean_t = np.exp(mu_t + sigma ** 2 / 2)
    true_diff = mean_t - mean_c
    for _ in range(boot_sims):
        c = rng.lognormal(mu_c, sigma, 500)
        t = rng.lognormal(mu_t, sigma, 500)
        lo, hi = bootstrap_diff(
            c, t, n_boot=2000, confidence=CONF,
            seed=int(rng.integers(0, 2**31)),
        ).ci
        covered += lo <= true_diff <= hi
    rate = covered / boot_sims
    se = np.sqrt(CONF * (1 - CONF) / boot_sims)
    proof.check_true(
        "percentile bootstrap on lognormal revenue does not badly under-cover",
        rate >= CONF - 4 * se,
        detail=f"measured {rate:.4f} against {CONF} (Monte Carlo se "
               f"{se:.4f}). The percentile bootstrap is known to under-cover "
               f"slightly for a mean at n=500 with this much skew; the claim "
               f"is a bound, not an equality, and the number is printed so "
               f"the size of the shortfall is visible.",
    )

    # The same data through welch_t, for comparison rather than judgement.
    covered = 0
    for _ in range(boot_sims):
        c = rng.lognormal(mu_c, sigma, 500)
        t = rng.lognormal(mu_t, sigma, 500)
        lo, hi = welch_t(c, t, confidence=CONF).ci
        covered += lo <= true_diff <= hi
    proof.note(
        f"for contrast, welch_t on the same lognormal data covers "
        f"{covered / boot_sims:.4f} of the time at n=500 per arm"
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
