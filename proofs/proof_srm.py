#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The sample-ratio check: it must not cry wolf, and it must catch what it says it catches.

WHY THIS CHECK RUNS BEFORE EVERY EFFECT ESTIMATE
If users did not reach the arms in the proportions you designed, something
in the assignment, the logging or the exclusion filters is broken - and
whatever is broken has almost certainly removed a non-random slice of
users. The effect estimate is then measuring that slice, not the treatment.
No amount of statistical care further down the readout recovers from it,
which is why srm.py is the first thing analyze.py calls and why a failure
stops the readout rather than annotating it.

TWO THINGS HAVE TO BE TRUE FOR THE CHECK TO BE WORTH HAVING

1. IT MUST NOT CRY WOLF. The threshold is 0.001, not 0.05, precisely
   because this check runs on every experiment and a false alarm costs an
   investigation. So the false alarm rate has to actually be 0.001 - which
   means the chi-square tail from statlib has to be right out at p =
   0.001, where almost nothing else in this repository exercises it. This
   proof is the only place that tail is measured.

2. IT MUST CATCH WHAT IT ADVERTISES. SRMResult reports a `detectable`
   field: the smallest share deviation catchable at 80% power. A reported
   number that has not been verified is just a comment. Here the whole
   power curve is predicted from

       power(d) = P(reject | true share is 0.5 + d)
                = Phi(-z_crit - d/se) + 1 - Phi(z_crit - d/se)

   with se = sqrt(p(1-p)/total) and z_crit the two-sided 0.001 critical
   value, and then measured at five points along it, including the point
   the module claims is its 80% threshold.

THE TEACHING POINT THAT MATTERS MOST
An SRM is not a fixed-size defect. The same broken assignment - the same
percentage of users lost - is invisible in a small experiment and glaring
in a large one. The last section measures one imbalance at three sample
sizes to make that concrete, because the usual mistake is to see a passing
SRM check on a small test and conclude the assignment is sound.
"""
from __future__ import annotations

import math

import numpy as np

from _harness import header

from explab.srm import srm_check
from explab.statlib import norm_cdf, norm_ppf, norm_sf

THRESHOLD = 0.001


def predicted_power(deviation: float, total: int, share: float = 0.5,
                    threshold: float = THRESHOLD) -> float:
    """Power of the two-arm chi-square check against a share deviation.

    The chi-square statistic with 1 df is the square of a z-statistic, so
    the check is exactly a two-sided z-test on the arm's share, and its
    power is the ordinary two-sided power formula.
    """
    se = math.sqrt(share * (1.0 - share) / total)
    z_crit = norm_ppf(1.0 - threshold / 2.0)
    lam = deviation / se
    return norm_sf(z_crit - lam) + norm_cdf(-z_crit - lam)


def simulate(total: int, true_share: float, n_sims: int, seed: int) -> int:
    """How many of n_sims experiments the check flags."""
    rng = np.random.default_rng(seed)
    control = rng.binomial(total, true_share, size=n_sims)
    flagged = 0
    for c in control:
        result = srm_check(
            {"control": int(c), "treatment": int(total - c)},
            threshold=THRESHOLD,
        )
        flagged += not result.passed
    return flagged


def main() -> int:
    proof = header(
        "PROOF: the sample ratio mismatch check",
        "the check flags a correct split 0.1% of the time and catches the "
        "deviation it says it catches 80% of the time",
    )

    # ---- 1. it must not cry wolf ------------------------------------------
    total = 200_000
    n_sims = 60_000
    flagged = simulate(total, 0.5, n_sims, 4242)
    proof.check_rate(
        "a correctly split experiment is flagged at the stated threshold",
        flagged, n_sims, THRESHOLD,
        detail=f"{flagged} false alarms in {n_sims:,} correctly randomised "
               f"experiments of {total:,} users. This is also the only place "
               f"in the suite that exercises the chi-square tail at p = "
               f"{THRESHOLD}, four orders of magnitude out from where the "
               f"rest of the library uses it.",
    )

    # Three arms, unequal declared shares: the df and the expected counts
    # both change, so the tail is being exercised at 2 df as well.
    rng = np.random.default_rng(777)
    shares = {"control": 0.5, "variant_a": 0.3, "variant_b": 0.2}
    n_sims_multi = 30_000
    draws = rng.multinomial(total, [0.5, 0.3, 0.2], size=n_sims_multi)
    flagged_multi = 0
    for row in draws:
        result = srm_check(
            {"control": int(row[0]), "variant_a": int(row[1]),
             "variant_b": int(row[2])},
            expected_shares=shares, threshold=THRESHOLD,
        )
        flagged_multi += not result.passed
    proof.check_rate(
        "a three-arm 50/30/20 split is flagged at the stated threshold too",
        flagged_multi, n_sims_multi, THRESHOLD,
        detail=f"{flagged_multi} false alarms in {n_sims_multi:,} "
               f"experiments; 2 degrees of freedom and unequal expected "
               f"counts, which is where a wrong df would show up",
    )

    # ---- 2. the power curve, predicted then measured ----------------------
    se = math.sqrt(0.25 / total)
    reference = srm_check({"control": total // 2, "treatment": total // 2},
                          threshold=THRESHOLD)
    advertised = reference.detectable
    print()
    print(f"  srm_check reports detectable = {advertised:.6f} at "
          f"{total:,} users")
    print(f"  (a share of {0.5 + advertised:.6f} instead of 0.500000, i.e. "
          f"{advertised * total * 2:.0f} users in the wrong arm)")
    print()
    print("     deviation   as z   predicted power   measured")
    print("  " + "-" * 52)

    power_sims = 6_000
    for i, lam in enumerate((1.0, 2.0, 3.0, advertised / se, 5.0)):
        deviation = lam * se
        predicted = predicted_power(deviation, total)
        hits = simulate(total, 0.5 + deviation, power_sims, 5100 + i)
        print(f"  {deviation:10.6f}   {lam:4.2f}   {predicted:15.4f}   "
              f"{hits / power_sims:8.4f}")
        proof.check_rate(
            f"power against a deviation of {lam:.2f} standard errors",
            hits, power_sims, predicted,
            detail=f"predicted {predicted:.4f} from the two-sided power "
                   f"formula at threshold {THRESHOLD}",
        )
    print()

    # The advertised number is the 80% point. That is the claim on the box.
    proof.check_close(
        "the `detectable` field srm_check reports really is the 80% point",
        predicted_power(advertised, total), 0.80, tolerance=0.005,
        detail=f"srm_check advertises {advertised:.6f} as detectable at 80% "
               f"power; the power formula puts that deviation at "
               f"{predicted_power(advertised, total):.4f}. The field is "
               f"checked against theory here and measured against simulation "
               f"in the row above, so it is not just a comment.",
    )

    # ---- 3. the same defect, three sample sizes ---------------------------
    # One broken assignment, expressed as a fixed share deviation, seen at
    # three scales. Nothing about the defect changes.
    print("  THE SAME DEFECT AT THREE SCALES")
    print("  A fixed 0.35% share deviation - one user in 286 sent to the")
    print("  wrong arm - held constant while only the sample size changes.")
    print()
    print("       users   detection rate   verdict")
    print("  " + "-" * 48)
    defect = 0.0035
    rates: list[float] = []
    for i, n_users in enumerate((50_000, 200_000, 1_000_000)):
        hits = simulate(n_users, 0.5 + defect, 4_000, 6200 + i)
        rate = hits / 4_000
        rates.append(rate)
        verdict = ("invisible" if rate < 0.2 else
                   "sometimes caught" if rate < 0.8 else "caught")
        print(f"  {n_users:10,}   {rate:14.4f}   {verdict}")
        proof.check_rate(
            f"detection of a 0.35% deviation at {n_users:,} users",
            hits, 4_000, predicted_power(defect, n_users),
            detail=f"predicted {predicted_power(defect, n_users):.4f}",
        )
    print()
    proof.check_true(
        "a passing SRM check on a small experiment is not evidence of a "
        "sound assignment",
        rates[0] < 0.25 and rates[-1] > 0.95,
        detail=f"the identical defect is detected {rates[0]:.1%} of the time "
               f"at 50,000 users and {rates[-1]:.1%} of the time at "
               f"1,000,000. The assignment was equally broken in both. This "
               f"is why `detectable` is reported alongside the verdict: a "
               f"PASS means nothing without the deviation it could have "
               f"caught.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
