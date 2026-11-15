#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The calibrated boundaries must hold alpha — checked on data they never saw.

`sequential.calibrate_boundary` finds its constant by simulation. A
boundary validated on the same simulation that produced it proves nothing:
that is fitting, and it would pass even if the method were nonsense.

So every check here uses an INDEPENDENT seed, and the last one abandons
the Brownian model entirely and runs real two-proportion tests on real
simulated users. If the boundaries hold there, they hold.

AND A CHECK AGAINST THE PUBLISHED NUMBERS
-----------------------------------------
Group-sequential boundaries are tabulated in the clinical-trials
literature. For five equally spaced looks at two-sided alpha 0.05:

    Pocock           2.413 at every look
    O'Brien-Fleming  4.56, 3.23, 2.63, 2.28, 2.04

Those were computed by numerical integration, decades ago, by people with
no connection to this code. Matching them is the strongest single piece of
evidence in this repository: an independent derivation, an independent
implementation, same answer.
"""
from __future__ import annotations

import numpy as np

from _harness import header

from explab.inference import two_proportion_from_arrays
from explab.sequential import calibrate_boundary

ALPHA = 0.05


def main() -> int:
    proof = header(
        "PROOF: group-sequential boundaries",
        "boundaries calibrated by simulation hold the false positive rate at "
        "alpha on data they were not fitted to",
    )

    # ---- against the published tables -------------------------------------
    pocock = calibrate_boundary(5, alpha=ALPHA, shape="pocock", n_sims=400_000,
                                seed=11)
    obf = calibrate_boundary(5, alpha=ALPHA, shape="obrien_fleming",
                             n_sims=400_000, seed=12)
    print(obf)
    print()
    proof.check_close(
        "Pocock boundary for 5 looks matches the published 2.413",
        pocock.boundaries[0], 2.413, tolerance=0.02,
        detail="Pocock (1977), two-sided alpha 0.05, 5 equally spaced looks",
    )
    published_obf = (4.56, 3.23, 2.63, 2.28, 2.04)
    worst = max(abs(a - b) for a, b in zip(obf.boundaries, published_obf))
    proof.check_close(
        "O'Brien-Fleming boundaries for 5 looks match the published table",
        worst, 0.0, tolerance=0.03,
        detail=f"computed {tuple(round(b, 3) for b in obf.boundaries)} "
               f"against published {published_obf}; largest gap {worst:.4f}",
    )

    # ---- independent Brownian check ---------------------------------------
    for plan, name in ((pocock, "Pocock"), (obf, "O'Brien-Fleming")):
        rng = np.random.default_rng(987654)
        n_sims = 120_000
        fractions = np.asarray(plan.information_fractions)
        deltas = np.diff(np.concatenate([[0.0], fractions]))
        increments = rng.standard_normal((n_sims, len(fractions))) * np.sqrt(deltas)
        z = np.abs(np.cumsum(increments, axis=1) / np.sqrt(fractions))
        crossed = int((z >= np.asarray(plan.boundaries)).any(axis=1).sum())
        proof.check_rate(
            f"{name} boundaries on an independent simulation",
            crossed, n_sims, ALPHA,
            detail="different seed from the one used to calibrate them",
        )

    # ---- the real thing ----------------------------------------------------
    # Actual users, actual tests, no effect, five looks, boundaries applied.
    rng = np.random.default_rng(31415)
    n_sims, n_final, base = 4000, 8000, 0.10
    n_looks = 5
    per_look = n_final // n_looks
    crossed = 0
    for _ in range(n_sims):
        c_all = (rng.random(n_final) < base).astype(float)
        t_all = (rng.random(n_final) < base).astype(float)
        for look in range(1, n_looks + 1):
            n = per_look * look
            r = two_proportion_from_arrays(c_all[:n], t_all[:n])
            z = r.estimate / r.se if r.se > 0 else 0.0
            if abs(z) >= obf.boundaries[look - 1]:
                crossed += 1
                break
    proof.check_rate(
        "O'Brien-Fleming boundaries on real two-proportion tests under the null",
        crossed, n_sims, ALPHA,
        detail="this is the claim that matters: the boundaries were derived "
               "from a Brownian abstraction and are applied to actual "
               "accumulating conversion data",
    )

    # ---- and they must still find a real effect ---------------------------
    rng = np.random.default_rng(27182)
    effect = 0.02
    found, found_early = 0, 0
    for _ in range(1500):
        c_all = (rng.random(n_final) < base).astype(float)
        t_all = (rng.random(n_final) < base + effect).astype(float)
        for look in range(1, n_looks + 1):
            n = per_look * look
            r = two_proportion_from_arrays(c_all[:n], t_all[:n])
            z = r.estimate / r.se if r.se > 0 else 0.0
            if abs(z) >= obf.boundaries[look - 1]:
                found += 1
                found_early += look < n_looks
                break
    proof.check_true(
        "the boundaries still detect a real effect, and often early",
        found / 1500 > 0.70,
        detail=f"detected in {found}/1500 runs ({found / 1500:.1%}), of which "
               f"{found_early} stopped before the planned end. The point of a "
               f"sequential design is not only validity, it is finishing "
               f"sooner when the effect is large.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
