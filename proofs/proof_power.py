#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The sample size formula must deliver the power it promises.

`power.sample_size_means` says: run this many units and you will detect an
effect of this size 80% of the time. That is a falsifiable prediction, and
this proof falsifies or confirms it by simulating experiments at exactly
the recommended size and counting how often they come out significant.

It also checks the two inverse directions, because all three functions
must describe one relationship and an inconsistency between them is the
easiest bug to ship:

    sample_size -> simulate -> measured power == target power
    mde          -> simulate at that n and effect -> power == target
    power_at     -> simulate at that n and effect -> matches the formula

and across a grid of effect sizes rather than one point, because a formula
can be right at 80% power and wrong at 50%.
"""
from __future__ import annotations

import numpy as np

from _harness import header

from explab.inference import two_proportion_from_arrays, welch_t
from explab.power import (mde_means, mde_proportions, power_at,
                          power_at_proportions, sample_size_means,
                          sample_size_proportions)

ALPHA = 0.05
N_SIMS = 3000


def main() -> int:
    proof = header(
        "PROOF: power and sample size",
        "an experiment run at the recommended size detects the stated effect "
        "at the stated rate",
    )
    rng = np.random.default_rng(606060)

    # ---- means: the recommended n delivers the promised power -------------
    sd, mde = 3.0, 0.45
    for target in (0.50, 0.80, 0.95):
        plan = sample_size_means(sd, mde, alpha=ALPHA, power=target)
        n = plan.n_per_group
        hits = 0
        for _ in range(N_SIMS):
            c = rng.normal(0.0, sd, n)
            t = rng.normal(mde, sd, n)
            hits += welch_t(c, t).p_value < ALPHA
        proof.check_rate(
            f"sample_size_means at {target:.0%} power (n={n:,} per arm)",
            hits, N_SIMS, target,
        )

    # ---- power_at agrees with simulation across a grid --------------------
    n = 900
    for effect in (0.0, 0.15, 0.3, 0.45, 0.6):
        predicted = power_at(sd, n, effect, alpha=ALPHA)
        hits = 0
        for _ in range(N_SIMS):
            c = rng.normal(0.0, sd, n)
            t = rng.normal(effect, sd, n)
            hits += welch_t(c, t).p_value < ALPHA
        proof.check_rate(
            f"power_at(sd={sd}, n={n}, effect={effect})", hits, N_SIMS,
            predicted,
            detail=f"formula says {predicted:.4f}; at effect 0 this is just "
                   f"the false positive rate again",
        )

    # ---- mde_means inverts sample_size_means ------------------------------
    recovered = mde_means(sd, sample_size_means(sd, mde).n_per_group)
    proof.check_close(
        "mde_means inverts sample_size_means", recovered, mde,
        tolerance=0.01 * mde,
        detail="within rounding: sample_size_means ceilings n, so the "
               "detectable effect at that n is a shade smaller than asked for",
    )

    # ---- proportions -------------------------------------------------------
    base, effect = 0.08, 0.01
    plan = sample_size_proportions(base, effect, alpha=ALPHA, power=0.80)
    n = plan.n_per_group
    hits = 0
    prop_sims = 2500
    for _ in range(prop_sims):
        c = (rng.random(n) < base).astype(float)
        t = (rng.random(n) < base + effect).astype(float)
        hits += two_proportion_from_arrays(c, t).p_value < ALPHA
    proof.check_rate(
        f"sample_size_proportions, 8% base, +1pp, 80% power (n={n:,})",
        hits, prop_sims, 0.80,
    )

    predicted = power_at_proportions(base, 6000, effect, alpha=ALPHA)
    hits = 0
    for _ in range(prop_sims):
        c = (rng.random(6000) < base).astype(float)
        t = (rng.random(6000) < base + effect).astype(float)
        hits += two_proportion_from_arrays(c, t).p_value < ALPHA
    proof.check_rate(
        "power_at_proportions at n=6000", hits, prop_sims, predicted,
        detail=f"formula says {predicted:.4f}",
    )

    recovered = mde_proportions(base, plan.n_per_group)
    proof.check_close(
        "mde_proportions inverts sample_size_proportions", recovered, effect,
        tolerance=0.01 * effect,
    )

    # ---- the shape of the relationship -------------------------------------
    n_small = sample_size_means(sd, 0.4).n_per_group
    n_half = sample_size_means(sd, 0.2).n_per_group
    proof.check_close(
        "halving the detectable effect quadruples the sample",
        n_half / n_small, 4.0, tolerance=0.02,
        detail="n scales with 1/MDE^2; this is why variance reduction pays",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
