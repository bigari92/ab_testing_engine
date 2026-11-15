#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The corrections control what they claim to - and NOT what they do not claim to.

WHAT IS BEING TESTED, AND WHY P-VALUES ARE FED IN DIRECTLY
bonferroni, holm and benjamini_hochberg take p-values and return verdicts.
They never see data. So the honest way to test them is to hand them
p-values whose truth is known exactly: a true null produces a p-value that
is Uniform(0, 1) by definition, and a false null produces one drawn from a
genuine non-central test. No estimation error, no approximation, nothing
between the procedure and its guarantee.

TWO DIFFERENT GUARANTEES, AND THE CONFUSION BETWEEN THEM IS THE POINT

  FWER  P(at least one false rejection anywhere). Bonferroni and Holm.
        "If I report nothing real, how often do I report something?"
  FDR   E[false rejections / total rejections]. Benjamini-Hochberg.
        "Of the things I report, what share are wrong?"

These are not stronger and weaker versions of one idea. FDR says nothing
about the probability of a single false finding - and this proof measures
BH's familywise error rate to show it running well above alpha, on purpose,
by design, correctly. A reader who thinks BH is "a less strict Bonferroni"
will read a BH-corrected segment scan as if one significant segment were
safe to act on. It is not, and the number below says how unsafe.

AND THE PROJECT'S OWN RULE, PUT TO THE TEST
multiplicity.py states a rule: guardrail metrics are never corrected. That
is a decision that costs something - more false alarms on guardrails - and
buys something, and the last claim here measures what it buys. The
asymmetry is the whole argument: a false guardrail alarm costs an
investigation, and a missed guardrail failure ships the harm.
"""
from __future__ import annotations

import math

import numpy as np

from _harness import header

from explab.inference import welch_t
from explab.multiplicity import benjamini_hochberg, bonferroni, holm
from explab.statlib import norm_cdf, norm_ppf, norm_sf

ALPHA = 0.05
M = 10


def null_p(rng, size) -> np.ndarray:
    """A true null's p-value is Uniform(0,1). That is the definition."""
    return rng.random(size)


def alt_p(rng, size, lam: float) -> np.ndarray:
    """A false null's p-value, from a two-sided test with non-centrality lam."""
    z = rng.standard_normal(size) + lam
    return 2.0 * np.array([norm_sf(abs(v)) for v in z])


def main() -> int:
    proof = header(
        "PROOF: multiple-comparison corrections",
        "Bonferroni and Holm hold the familywise error rate at alpha, "
        "Benjamini-Hochberg holds the false discovery rate at alpha, and "
        "BH does not hold the familywise rate at all",
    )

    n_sims = 40_000
    rng = np.random.default_rng(31337)

    # ---- 1. the global null: every hypothesis is true ---------------------
    # Every rejection here is a false one, so FWER and FDR coincide and the
    # theory is exact.
    raw_hits = bonf_hits = holm_hits = bh_hits = 0
    for _ in range(n_sims):
        p = null_p(rng, M)
        raw_hits += bool((p < ALPHA).any())
        bonf_hits += any(bonferroni(p, alpha=ALPHA).rejected)
        holm_hits += any(holm(p, alpha=ALPHA).rejected)
        bh_hits += any(benjamini_hochberg(p, alpha=ALPHA).rejected)

    uncorrected = 1.0 - (1.0 - ALPHA) ** M
    bonf_exact = 1.0 - (1.0 - ALPHA / M) ** M

    print(f"  {M} independent tests, every null TRUE, {n_sims:,} simulations")
    print(f"    no correction   {raw_hits / n_sims:.4f}   "
          f"theory {uncorrected:.4f}")
    print(f"    Bonferroni      {bonf_hits / n_sims:.4f}   "
          f"theory {bonf_exact:.4f}")
    print(f"    Holm            {holm_hits / n_sims:.4f}   "
          f"theory {bonf_exact:.4f} (identical to Bonferroni here)")
    print(f"    BH              {bh_hits / n_sims:.4f}   theory {ALPHA:.4f} "
          f"(exact under the global null)")
    print()

    proof.check_rate(
        "without correction, 10 tests at alpha 0.05 give a 40% error rate",
        raw_hits, n_sims, uncorrected,
        detail=f"1 - (1 - {ALPHA})^{M} = {uncorrected:.4f}. This is the "
               f"number that makes correction necessary: a readout with ten "
               f"secondary metrics finds something in two runs out of five "
               f"when nothing is there.",
    )
    proof.check_rate(
        "Bonferroni holds the familywise rate at alpha", bonf_hits, n_sims,
        bonf_exact,
        detail=f"1 - (1 - {ALPHA}/{M})^{M} = {bonf_exact:.4f}, slightly below "
               f"alpha: the conservatism Bonferroni is known for, and it is "
               f"only this small because the tests are independent",
    )
    proof.check_rate(
        "Holm holds it too, at exactly the same rate under the global null",
        holm_hits, n_sims, bonf_exact,
        detail="Holm's first step tests the smallest p-value against "
               "alpha/m, which is Bonferroni's condition exactly; when every "
               "null is true, making any rejection requires passing that "
               "step, so the two cannot differ here. Holm's advantage "
               "appears only once some nulls are false.",
    )
    proof.check_rate(
        "BH holds the familywise rate at alpha when every null is true",
        bh_hits, n_sims, ALPHA,
        detail="under the global null every discovery is false, so FDR and "
               "FWER are the same quantity and BH's control is exact rather "
               "than conservative. This is the one case where BH is as safe "
               "as Bonferroni - and the next claims show it is the only one.",
    )

    # ---- 2. a realistic mix, and the two guarantees come apart ------------
    m0, m1 = 4, 6          # 4 true nulls, 6 real effects
    lam = 4.0              # the real effects are unmistakable
    n_mix = 40_000
    rng = np.random.default_rng(80808)
    fdp_bh: list[float] = []
    bh_familywise = 0
    bonf_familywise = 0
    holm_never_fewer = True
    holm_strictly_more = 0
    for _ in range(n_mix):
        p = np.concatenate([null_p(rng, m0), alt_p(rng, m1, lam)])
        is_null = np.array([True] * m0 + [False] * m1)

        bh = np.asarray(benjamini_hochberg(p, alpha=ALPHA).rejected)
        n_rej = int(bh.sum())
        n_false = int((bh & is_null).sum())
        fdp_bh.append(n_false / n_rej if n_rej else 0.0)
        bh_familywise += n_false > 0

        b = np.asarray(bonferroni(p, alpha=ALPHA).rejected)
        h = np.asarray(holm(p, alpha=ALPHA).rejected)
        bonf_familywise += int((b & is_null).sum()) > 0
        if int(h.sum()) < int(b.sum()):
            holm_never_fewer = False
        holm_strictly_more += int(h.sum()) > int(b.sum())

    fdr_theory = ALPHA * m0 / (m0 + m1)
    print(f"  {m0} true nulls and {m1} real effects, {n_mix:,} simulations")
    print(f"    BH false discovery rate   {np.mean(fdp_bh):.4f}   "
          f"theory alpha*m0/m = {fdr_theory:.4f}")
    print(f"    BH FAMILYWISE rate        "
          f"{bh_familywise / n_mix:.4f}   against alpha = {ALPHA}")
    print(f"    Bonferroni familywise     "
          f"{bonf_familywise / n_mix:.4f}")
    print()

    proof.check_mean(
        "BH holds the false discovery rate at alpha*m0/m", fdp_bh, fdr_theory,
        detail=f"mean false-discovery proportion over {n_mix:,} simulations "
               f"against the exact independent-case value "
               f"{ALPHA} * {m0}/{m0 + m1} = {fdr_theory:.4f}. Note what this "
               f"means in practice: BH gets MORE permissive as more of your "
               f"hypotheses turn out to be real.",
    )
    proof.check_true(
        "BH does NOT control the familywise error rate, by design",
        bh_familywise / n_mix > 2.5 * ALPHA,
        detail=f"BH made at least one false rejection in "
               f"{bh_familywise / n_mix:.1%} of simulations, against a "
               f"familywise alpha of {ALPHA:.0%} - "
               f"{bh_familywise / n_mix / ALPHA:.1f} times higher, while "
               f"holding its own guarantee exactly. Both facts are correct "
               f"simultaneously. This is why a BH-corrected segment scan is "
               f"labelled EXPLORATORY in report.py: the list of segments is "
               f"trustworthy as a list, and no single segment on it is "
               f"trustworthy on its own.",
    )
    proof.check_true(
        "Holm never rejects less than Bonferroni, in any simulation",
        holm_never_fewer,
        detail=f"checked per-simulation across all {n_mix:,} runs, not on "
               f"average: Holm rejected strictly more in "
               f"{holm_strictly_more / n_mix:.1%} of them and fewer in none. "
               f"Same guarantee, uniformly more power, which is why "
               f"bonferroni() exists in this library mainly to be compared "
               f"against.",
    )
    proof.check_rate(
        "Bonferroni still holds the familywise rate once some nulls are false",
        bonf_familywise, n_mix, ALPHA * m0 / (m0 + m1), k=5.0,
        detail=f"with only {m0} of {m0 + m1} nulls true, Bonferroni's "
               f"effective level is alpha*m0/m = {ALPHA * m0 / (m0 + m1):.4f} "
               f"- it is spending correction on hypotheses that did not need "
               f"it, which is exactly the conservatism Holm removes",
    )

    # ---- 3. why guardrails are never corrected ----------------------------
    # A real harm in one of five guardrails. Correcting the family protects
    # against a false alarm and pays for it by missing the harm.
    n_guardrails = 5
    z_un = norm_ppf(1.0 - ALPHA / 2.0)
    z_co = norm_ppf(1.0 - ALPHA / (2.0 * n_guardrails))
    # Size the harm so the uncorrected test has 80% power.
    lam_harm = z_un + norm_ppf(0.80)
    power_un_theory = norm_sf(z_un - lam_harm) + norm_cdf(-z_un - lam_harm)
    power_co_theory = norm_sf(z_co - lam_harm) + norm_cdf(-z_co - lam_harm)

    # Measured on real data, not on p-values: a continuous guardrail with a
    # genuine regression in one of the five.
    rng = np.random.default_rng(1234)
    n_per_arm = 2000
    shift = lam_harm * math.sqrt(2.0 / n_per_arm)   # so the z-stat has mean lam
    caught_un = caught_co = 0
    sims = 8_000
    for _ in range(sims):
        p_values = []
        for g in range(n_guardrails):
            c = rng.standard_normal(n_per_arm)
            t = rng.standard_normal(n_per_arm) + (shift if g == 0 else 0.0)
            p_values.append(welch_t(c, t).p_value)
        caught_un += p_values[0] < ALPHA
        caught_co += bool(bonferroni(p_values, alpha=ALPHA).rejected[0])

    print("  ONE REAL HARM IN FIVE GUARDRAILS")
    print(f"    uncorrected   caught {caught_un / sims:.4f}   "
          f"theory {power_un_theory:.4f}")
    print(f"    Bonferroni    caught {caught_co / sims:.4f}   "
          f"theory {power_co_theory:.4f}")
    print()

    proof.check_rate(
        "an uncorrected guardrail catches a real harm at the designed rate",
        caught_un, sims, power_un_theory,
        detail=f"the harm was sized for 80% power, and the measurement "
               f"confirms it: {caught_un / sims:.1%}",
    )
    proof.check_rate(
        "correcting the guardrail family loses that detection",
        caught_co, sims, power_co_theory,
        detail=f"the identical harm, the identical data, Bonferroni applied "
               f"across the five guardrails: caught "
               f"{caught_co / sims:.1%} of the time",
    )
    lost = (caught_un - caught_co) / sims
    proof.check_true(
        "so guardrails are left uncorrected, and that is the reason",
        lost > 0.15,
        detail=f"correcting costs {lost:.1%} of real harms detected - "
               f"{caught_un / sims:.0%} down to {caught_co / sims:.0%} - in "
               f"exchange for holding the guardrail family's false alarm rate "
               f"at {ALPHA:.0%} instead of "
               f"{1 - (1 - ALPHA) ** n_guardrails:.0%}. The two errors are "
               f"not comparable: a false guardrail alarm costs an "
               f"investigation that ends in 'no, it was fine', and a missed "
               f"one ships a regression to everybody. Correction is for "
               f"findings you want to announce, not for harms you are "
               f"watching for.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
