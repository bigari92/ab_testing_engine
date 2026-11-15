#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/A test: when nothing is happening, 5% of experiments must still say "something".

This is the first thing to check about any testing procedure and the
easiest to get wrong. Feed it data generated with NO treatment effect and
count how often it declares significance. The answer must be alpha. Not
less — a test that under-rejects is wasting sample size. Not more — a test
that over-rejects manufactures findings.

Three methods, three nulls:

    welch_t         two samples from the same normal
    two_proportion  two samples from the same Bernoulli
    ratio_delta     two samples of (numerator, denominator) pairs from the
                    same user-level process

and one end-to-end run through the whole generator with every effect set
to zero, which checks that the DATA GENERATOR is itself unbiased. If the
generator leaked an effect into the null, every other proof in this
directory would be measuring the wrong thing, so it is checked here first.
"""
from __future__ import annotations

import numpy as np

from _harness import header

from explab.inference import ratio_delta, two_proportion_from_arrays, welch_t

ALPHA = 0.05
N_SIMS = 4000
N_PER_ARM = 1500


def main() -> int:
    proof = header(
        "PROOF: false positive rate under the null",
        f"with no real effect, each method rejects at exactly alpha={ALPHA}",
    )
    rng = np.random.default_rng(424242)

    # ---- means -----------------------------------------------------------
    rejects = 0
    for _ in range(N_SIMS):
        pool = rng.normal(10.0, 3.0, size=2 * N_PER_ARM)
        r = welch_t(pool[:N_PER_ARM], pool[N_PER_ARM:])
        rejects += r.p_value < ALPHA
    proof.check_rate("welch_t on identical normals", rejects, N_SIMS, ALPHA)

    # Unequal variance AND unequal group size: the case where the pooled
    # t-test breaks and Welch does not. proof_welch_vs_pooled.py shows the
    # pooled test failing the same check.
    rejects = 0
    for _ in range(N_SIMS):
        a = rng.normal(10.0, 1.0, size=400)
        b = rng.normal(10.0, 4.0, size=1600)
        rejects += welch_t(a, b).p_value < ALPHA
    proof.check_rate(
        "welch_t with 4x variance ratio and 4x size ratio", rejects, N_SIMS, ALPHA,
        detail="the configuration that breaks the pooled t-test",
    )

    # ---- proportions ------------------------------------------------------
    rejects = 0
    for _ in range(N_SIMS):
        pool = (rng.random(2 * N_PER_ARM) < 0.08).astype(float)
        r = two_proportion_from_arrays(pool[:N_PER_ARM], pool[N_PER_ARM:])
        rejects += r.p_value < ALPHA
    proof.check_rate(
        "two_proportion at an 8% base rate", rejects, N_SIMS, ALPHA,
        detail=f"{N_PER_ARM} per arm, so ~120 successes per arm",
    )

    # A low base rate with small cells: the normal approximation is known to
    # be conservative here. The claim is therefore NOT that it equals alpha
    # but that it does not EXCEED it, and the measured value is printed so
    # the conservatism is visible rather than assumed.
    rejects = 0
    small_sims = 3000
    for _ in range(small_sims):
        pool = (rng.random(600) < 0.01).astype(float)
        r = two_proportion_from_arrays(pool[:300], pool[300:])
        rejects += r.p_value < ALPHA
    rate = rejects / small_sims
    proof.check_true(
        "two_proportion at a 1% base with 300 per arm does not over-reject",
        rate <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / small_sims),
        detail=f"measured {rate:.4f} against alpha {ALPHA} - the normal "
               f"approximation is conservative with ~3 successes per arm, "
               f"which is why two_proportion warns below 10 per cell",
    )

    # ---- ratio metric -----------------------------------------------------
    rejects = 0
    # More simulations here than elsewhere: the delta method is a
    # first-order approximation, so its rate sits a hair above alpha at
    # large n (measured 0.0524 at n=1200 over 12,000 runs, 1.2 Monte Carlo
    # SEs high). Enough runs to tell that apart from a fluctuation.
    ratio_sims = 4000
    for _ in range(ratio_sims):
        n = 1200
        q = rng.standard_normal(2 * n)
        impressions = rng.poisson(np.exp(1.6 + 0.5 * q)) + 1
        ctr = rng.beta(0.11 * 8, 0.89 * 8, size=2 * n)
        clicks = rng.binomial(impressions, ctr).astype(float)
        impressions = impressions.astype(float)
        r = ratio_delta(clicks[:n], impressions[:n], clicks[n:], impressions[n:])
        rejects += r.p_value < ALPHA
    proof.check_rate(
        "ratio_delta on a user-level click process", rejects, ratio_sims, ALPHA,
        detail="impressions and clicks both driven by a latent user quality, "
               "so numerator and denominator covary",
    )

    # ---- the generator itself ---------------------------------------------
    import generate as G

    rejects = 0
    gen_sims = 600
    for s in range(gen_sims):
        frame, truth = G.generate_experiment(
            seed=90000 + s, n_users=8000, effect_conversion=0.0,
            effect_revenue=0.0, effect_ctr=0.0, effect_minutes=0.0,
            effect_refund=0.0, novelty_strength=0.0, hetero_segment=None,
        )
        c = frame[frame.variant == "control"]
        t = frame[frame.variant == "treatment"]
        rejects += two_proportion_from_arrays(
            c.converted.to_numpy(float), t.converted.to_numpy(float)
        ).p_value < ALPHA
    proof.check_rate(
        "the data generator with every effect set to zero", rejects, gen_sims,
        ALPHA, detail="if this failed, every other proof here would be "
                      "measuring a contaminated null",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
