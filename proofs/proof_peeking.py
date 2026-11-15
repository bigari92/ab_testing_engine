#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much peeking actually costs, in numbers you can put in front of someone.

The argument "you cannot stop the experiment just because it crossed 0.05
yesterday" loses to "but it crossed 0.05" unless you can say what the
error rate becomes. This proof produces that table.

TWO SIMULATIONS, BECAUSE TWO THINGS ARE BEING CLAIMED
-----------------------------------------------------
1. The abstract one: a Brownian path observed at K equally spaced times,
   which is what a cumulative z-statistic under the null is. Fast, and it
   is the model `sequential.naive_peeking_alpha` uses.

2. The concrete one: actual users arriving over actual days, with an
   actual two-proportion test re-run on the accumulated data at each look.
   Slower, and it is the one that proves the abstract model describes the
   thing people really do.

If the two disagree, the abstract model is wrong and the headline number
is not trustworthy. They are checked against each other.

THE STOPPING RULE BEING SIMULATED is the honest version of what happens in
practice: look every few days, stop and declare a winner the first time
p < 0.05, otherwise run to the end and use the final p. Nobody writes this
rule down. Everybody follows it.
"""
from __future__ import annotations

import numpy as np

from _harness import header

from explab.inference import two_proportion_from_arrays
from explab.sequential import naive_peeking_alpha

ALPHA = 0.05


def main() -> int:
    proof = header(
        "PROOF: the cost of peeking",
        "stopping at the first look where p < 0.05 inflates the false "
        "positive rate far above 0.05",
    )

    # ---- the table -------------------------------------------------------
    print("  looks   false positive rate   inflation")
    print("  " + "-" * 44)
    rates = {}
    for k in (1, 2, 3, 5, 7, 10, 14, 20, 30):
        rate = naive_peeking_alpha(k, alpha=ALPHA, n_sims=200_000, seed=4242 + k)
        rates[k] = rate
        print(f"  {k:5d}   {rate:19.4f}   {rate / ALPHA:7.2f}x")
    print()

    proof.check_rate(
        "one look is the honest 5%", int(rates[1] * 200_000), 200_000, ALPHA,
    )
    proof.check_true(
        "five looks roughly triples the error rate",
        0.12 <= rates[5] <= 0.16,
        detail=f"measured {rates[5]:.4f}; the textbook figure for 5 equally "
               f"spaced looks at alpha 0.05 is about 0.14",
    )
    proof.check_true(
        "twenty looks takes it past 1 in 4",
        rates[20] >= 0.22,
        detail=f"measured {rates[20]:.4f}",
    )
    proof.check_true(
        "the inflation is monotone in the number of looks",
        all(rates[a] <= rates[b] + 1e-3
            for a, b in zip(sorted(rates)[:-1], sorted(rates)[1:])),
        detail="more looks can only make it worse",
    )

    # ---- the concrete simulation -----------------------------------------
    # Real users, real tests, no treatment effect, stop at the first
    # significant look.
    rng = np.random.default_rng(80808)
    n_sims, n_final, n_looks, base = 4000, 6000, 5, 0.10
    per_look = n_final // n_looks
    stopped_early = 0
    rejected = 0
    for _ in range(n_sims):
        c_all = (rng.random(n_final) < base).astype(float)
        t_all = (rng.random(n_final) < base).astype(float)
        for look in range(1, n_looks + 1):
            n = per_look * look
            p = two_proportion_from_arrays(c_all[:n], t_all[:n]).p_value
            if p < ALPHA:
                rejected += 1
                stopped_early += look < n_looks
                break
    measured = rejected / n_sims
    proof.check_rate(
        "real two-proportion tests, 5 looks, stop at the first p < 0.05",
        rejected, n_sims, rates[5],
        detail=f"the Brownian model predicted {rates[5]:.4f}; "
               f"{stopped_early:,} of {rejected:,} false positives were "
               f"declared before the planned end, which is exactly the "
               f"experiment somebody would have called a win",
    )

    # ---- what it costs you in the other direction -------------------------
    # Peeking does not only invent winners. Stopping early at the first
    # crossing also biases the EFFECT SIZE upward, because you stop on the
    # draws where noise happened to be generous.
    rng = np.random.default_rng(171717)
    true_effect = 0.01
    stopped_estimates, full_estimates = [], []
    for _ in range(2500):
        c_all = (rng.random(n_final) < base).astype(float)
        t_all = (rng.random(n_final) < base + true_effect).astype(float)
        stopped_at = None
        for look in range(1, n_looks + 1):
            n = per_look * look
            r = two_proportion_from_arrays(c_all[:n], t_all[:n])
            if r.p_value < ALPHA and stopped_at is None:
                stopped_at = r.estimate
        full = two_proportion_from_arrays(c_all, t_all)
        full_estimates.append(full.estimate)
        if stopped_at is not None:
            stopped_estimates.append(stopped_at)
    inflation = float(np.mean(stopped_estimates) / true_effect)
    proof.check_mean(
        "running to the end gives an unbiased effect estimate",
        full_estimates, true_effect,
    )
    proof.check_true(
        "stopping at the first significant look overstates the effect",
        inflation > 1.15,
        detail=f"mean estimate among experiments that stopped early was "
               f"{np.mean(stopped_estimates):.5f} against a truth of "
               f"{true_effect}, i.e. {inflation:.2f}x. This is the winner's "
               f"curse: you stop on the draws where noise was helping, so "
               f"the effect you ship is not the effect you measured.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
