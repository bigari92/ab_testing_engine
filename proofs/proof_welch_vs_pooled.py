#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why inference.py offers Welch's t-test and not Student's pooled one.

THE CHOICE BEING DEFENDED
Student's two-sample t-test assumes both groups have the SAME variance. It
pools them into one estimate:

    s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2) / (n1 + n2 - 2)
    SE    = s_p * sqrt(1/n1 + 1/n2)

Welch's refuses to pool:

    SE = sqrt(s1^2/n1 + s2^2/n2)

and spends degrees of freedom (Welch-Satterthwaite) to pay for it.

THE USUAL STORY IS WRONG, AND THAT IS WHY THIS PROOF EXISTS
The textbook line is "use Welch when the variances differ". It is not that
simple, and the simple version leads people to use the pooled test when
they have checked the variances and think they are close enough.

What actually breaks the pooled test is unequal variance AND unequal group
sizes TOGETHER. With equal n it survives a 4x variance ratio almost
untouched. With unequal n it fails - and the DIRECTION of the failure
flips depending on which arm is the bigger one:

  big arm has the SMALL variance  ->  SE too small  ->  false positives
  big arm has the LARGE variance  ->  SE too big    ->  power thrown away

Both are measured below, and both are predicted first. The prediction:

    E[s_p^2]  = ((n1-1)sig1^2 + (n2-1)sig2^2) / (n1 + n2 - 2)
    SE_pooled = sqrt(E[s_p^2] * (1/n1 + 1/n2))
    SE_true   = sqrt(sig1^2/n1 + sig2^2/n2)

The statistic is a mean difference with true spread SE_true, divided by
something close to SE_pooled, so it behaves like a normal with standard
deviation SE_true/SE_pooled rather than 1, and

    false positive rate = 2 * (1 - Phi(t_crit * SE_pooled / SE_true))

THE POINT
An unbalanced experiment is not an exotic case. It is what you get from a
90/10 ramp, from one arm crashing, from excluding bots unevenly, or from
a holdout. And a variance difference between arms is what a treatment that
changes behaviour produces. So the conditions for the pooled test to fail
are the ordinary conditions of an online experiment, the failure is
invisible in the output, and its direction is not something you can reason
about at the keyboard. Hence: always Welch. The last claim here measures
what that costs when the variances really are equal, which is the only
reason anyone would hesitate.
"""
from __future__ import annotations

import math

import numpy as np

from _harness import header

from explab.inference import welch_t
from explab.statlib import norm_sf, t_ppf, t_sf

ALPHA = 0.05


def pooled_t(control: np.ndarray, treatment: np.ndarray) -> float:
    """Student's two-sample t-test. Returns the two-sided p-value.

    This is the test the library deliberately does NOT provide. It lives
    here, in the proof that explains why, so that nobody can import it by
    accident.
    """
    n1, n2 = control.size, treatment.size
    v1 = control.var(ddof=1)
    v2 = treatment.var(ddof=1)
    df = n1 + n2 - 2
    s_p2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    se = math.sqrt(s_p2 * (1.0 / n1 + 1.0 / n2))
    t_stat = (treatment.mean() - control.mean()) / se
    return 2.0 * t_sf(abs(t_stat), df)


def predicted_pooled_fpr(n1: int, n2: int, sd1: float, sd2: float) -> float:
    """What the pooled test's false positive rate must be. Theory only."""
    df = n1 + n2 - 2
    e_sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df
    se_pooled = math.sqrt(e_sp2 * (1.0 / n1 + 1.0 / n2))
    se_true = math.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    t_crit = t_ppf(1.0 - ALPHA / 2.0, df)
    return 2.0 * norm_sf(t_crit * se_pooled / se_true)


def measure(n1: int, n2: int, sd1: float, sd2: float, n_sims: int,
            seed: int) -> tuple[float, float]:
    """False positive rate of both tests under a true null."""
    rng = np.random.default_rng(seed)
    hits_pooled = hits_welch = 0
    for _ in range(n_sims):
        c = rng.standard_normal(n1) * sd1
        t = rng.standard_normal(n2) * sd2
        hits_pooled += pooled_t(c, t) < ALPHA
        hits_welch += welch_t(c, t).p_value < ALPHA
    return hits_pooled / n_sims, hits_welch / n_sims


def case(label: str, n1: int, n2: int, sd1: float, sd2: float, n_sims: int,
         seed: int, proof) -> None:
    """One configuration: predict, measure, check both tests."""
    predicted = predicted_pooled_fpr(n1, n2, sd1, sd2)
    rate_pooled, rate_welch = measure(n1, n2, sd1, sd2, n_sims, seed)

    print(f"  {label}")
    print(f"    n = {n1} vs {n2},  sd = {sd1} vs {sd2}")
    print(f"    pooled (Student)  false positive rate {rate_pooled:.4f}   "
          f"theory predicted {predicted:.4f}")
    print(f"    Welch             false positive rate {rate_welch:.4f}   "
          f"nominal {ALPHA}")
    print()

    proof.check_rate(
        f"pooled test lands on its predicted rate - {label}",
        int(round(rate_pooled * n_sims)), n_sims, predicted,
        detail=f"predicted {predicted:.4f} from the variances and group "
               f"sizes alone, before the simulation ran",
    )
    proof.check_rate(
        f"Welch holds alpha - {label}",
        int(round(rate_welch * n_sims)), n_sims, ALPHA,
        detail="the same data, the same nominal level, no assumption about "
               "equal variances",
    )


def main() -> int:
    proof = header(
        "PROOF: Welch's t-test against Student's pooled one",
        "the pooled test fails when variances and group sizes are both "
        "unequal, in a direction that depends on which arm is larger, and "
        "Welch does not",
    )

    # 15,000 is enough: the gap between theory and measurement here is
    # systematic (the prediction treats the pooled SE as fixed when it is
    # really estimated), not noise, so it does not shrink with more
    # simulations - at 40,000 it was 0.1795 against 0.1785 predicted, and
    # 0.00455 against 0.00422. More runs would only narrow the tolerance
    # around an offset that stays put.
    n_sims = 15_000

    # 1. The baseline: nothing unequal. Both tests are correct here, which
    #    is what makes the pooled test feel safe.
    case("equal variances, equal sizes", 400, 400, 1.0, 1.0, n_sims, 11, proof)

    # 2. A 4x variance ratio with balanced arms. Still fine - and this is
    #    the case that convinces people the pooled test is robust.
    case("4x variance ratio, equal sizes", 400, 400, 1.0, 2.0, n_sims, 12,
         proof)

    # 3. Unbalanced, with the large arm on the low-variance side. The SE is
    #    too small and the test rejects far too often.
    case("4x variance ratio, large arm has the small variance", 800, 200,
         1.0, 2.0, n_sims, 13, proof)

    # 4. The same numbers, the arms swapped. Now the SE is too big and the
    #    test has thrown its power away instead.
    case("4x variance ratio, large arm has the large variance", 200, 800,
         1.0, 2.0, n_sims, 14, proof)

    # ---- the direction flip, stated as its own claim ----------------------
    up = predicted_pooled_fpr(800, 200, 1.0, 2.0)
    down = predicted_pooled_fpr(200, 800, 1.0, 2.0)
    proof.check_true(
        "the same variance ratio breaks the pooled test in both directions",
        up > 3.0 * ALPHA and down < ALPHA / 3.0,
        detail=f"identical variance ratio, identical total sample size, only "
               f"the allocation reversed: the false positive rate goes to "
               f"{up:.1%} one way and {down:.2%} the other. There is no "
               f"'conservative' reading of the pooled test to fall back on, "
               f"which is why the choice cannot be left to judgement at the "
               f"keyboard.",
    )

    # ---- and what Welch costs when the assumption really does hold -------
    # The only argument for the pooled test is power under equal variances.
    # Measure it, so the trade-off is a number rather than an opinion.
    rng = np.random.default_rng(99)
    effect = 0.18
    n = 400
    power_sims = 10_000
    hits_pooled = hits_welch = 0
    for _ in range(power_sims):
        c = rng.standard_normal(n)
        t = rng.standard_normal(n) + effect
        hits_pooled += pooled_t(c, t) < ALPHA
        hits_welch += welch_t(c, t).p_value < ALPHA
    gap = (hits_pooled - hits_welch) / power_sims
    proof.check_close(
        "Welch costs almost no power when the variances are genuinely equal",
        gap, 0.0, tolerance=0.005,
        detail=f"at {n} per arm with a true effect of {effect}: pooled "
               f"detected {hits_pooled / power_sims:.2%}, Welch "
               f"{hits_welch / power_sims:.2%} - a difference of {gap:+.3%}. "
               f"That is the entire price of never having to check the "
               f"assumption, against a false positive rate that goes to "
               f"{up:.0%} when the assumption fails quietly.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
