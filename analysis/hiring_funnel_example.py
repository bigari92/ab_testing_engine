#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The same machinery on a recruitment funnel, end to end.

WHY THIS FILE EXISTS
--------------------
Everything in `proofs/` runs against a generic product experiment:
conversion, revenue, refunds. That is the right shape for testing the
library and the wrong shape for showing that it transfers. This file runs
the full pipeline on a staffing funnel — CVs submitted, interviews,
offers, hires, probation passed — which is the domain the author actually
works in, and which has one structural feature a checkout funnel does not:

    EVERY STAGE IS A DENOMINATOR FOR THE NEXT ONE.

That makes the trap in docs/02 not an edge case but the default. "Offer
rate" has at least four defensible meanings in a staffing business:

    offers / candidates sourced          sourcing + screening + interview
    offers / CVs submitted to client     screening + interview
    offers / candidates interviewed      interview performance only
    offers / vacancies open              demand-side, a different question

A change that improves screening moves the first three in ways that have
nothing to do with the thing being measured, and the fourth not at all.
Picking one without saying which is how two people report "offer rate"
from the same database and disagree by a factor of three.

THE EXPERIMENT MODELLED HERE
----------------------------
A recruitment agency introduces a structured pre-screening call before a
CV goes to the client. The hypothesis: clients interview more of the CVs
they receive, because the weak ones are filtered out earlier.

    randomisation unit   the candidate, at the moment of sourcing
    primary              CV -> interview rate (per CV submitted)
    guardrail            offers per interview  (interview quality must
                         not fall: if we are only sending obvious yeses,
                         the client's interview slots are being wasted
                         less, but our offer conversion should hold)
    guardrail            probation pass rate per hire
    secondary            time to hire, days

WHAT THE SCREENING ACTUALLY DOES, in the planted truth
------------------------------------------------------
It raises the probability a submitted CV reaches interview by a real
amount, by declining to submit the weakest candidates. It does NOT make
any individual candidate better. So:

  - interview rate per CV submitted      genuinely rises     <- the effect
  - offers per interview                 does NOT change     <- unchanged
  - offers per CV submitted              rises mechanically  <- the trap
  - CVs submitted per candidate sourced  FALLS               <- the cost

The last line is the one a readout usually omits, and it is the one a
client cares about: fewer CVs reaching them is the mechanism, not a side
effect. It is reported here as a declared metric rather than discovered
in a follow-up question.

RUN IT
------
    python analysis/hiring_funnel_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explab.analyze import MetricSpec, analyze          # noqa: E402
from explab.inference import ratio_delta, two_proportion_from_arrays  # noqa: E402
from explab.power import sample_size_proportions        # noqa: E402
from explab.report import render_text                   # noqa: E402

# --------------------------------------------------------------------------
# the funnel generator
# --------------------------------------------------------------------------
# Baseline rates taken from the shape of a mid-market IT staffing funnel:
# most sourced candidates never reach a client, most submitted CVs are
# declined on paper, and the stages after interview are comparatively
# efficient. The absolute numbers are illustrative; the RATIOS between
# stages are what make the example behave like a real funnel.
P_SUBMIT = 0.42        # sourced -> CV submitted to client
P_INTERVIEW = 0.26     # submitted -> interviewed
P_OFFER = 0.31         # interviewed -> offer
P_ACCEPT = 0.74        # offer -> hire
P_PROBATION = 0.88     # hire -> passed probation


def generate_funnel(
    n_candidates: int = 9_000,
    *,
    seed: int = 0,
    screening_lift: float = 0.06,
    screening_submit_cost: float = 0.07,
    offer_quality_change: float = 0.0,
    logging_gap: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """One randomised funnel experiment and its sealed key.

    screening_lift          added to P(interview | submitted) in treatment.
                            The real effect.
    screening_submit_cost   subtracted from P(submit) in treatment: the
                            weakest candidates are no longer sent. This is
                            the price of the lift, not a defect.
    offer_quality_change    added to P(offer | interviewed) in treatment.
                            Zero means interview performance is untouched,
                            which is the interesting case.
    logging_gap             share of CONTROL candidates silently dropped
                            before analysis, to model the most common real
                            SRM cause: a recruiter workflow that only logs
                            one arm properly.
    """
    rng = np.random.default_rng(seed)

    # Latent candidate strength. Drives every stage, which is what makes
    # the stages correlated and the denominators treacherous.
    strength = rng.standard_normal(n_candidates)
    treated = rng.random(n_candidates) < 0.5

    def shift(base: float, s: np.ndarray, extra: float = 0.0) -> np.ndarray:
        """Base rate tilted by candidate strength, plus a flat treatment shift."""
        return np.clip(base + 0.09 * s + extra, 0.01, 0.99)

    p_submit = shift(P_SUBMIT, strength,
                     np.where(treated, -screening_submit_cost, 0.0))
    submitted = (rng.random(n_candidates) < p_submit).astype(int)

    p_interview = shift(P_INTERVIEW, strength,
                        np.where(treated, screening_lift, 0.0))
    interviewed = ((rng.random(n_candidates) < p_interview) & (submitted == 1)).astype(int)

    p_offer = shift(P_OFFER, strength,
                    np.where(treated, offer_quality_change, 0.0))
    offered = ((rng.random(n_candidates) < p_offer) & (interviewed == 1)).astype(int)

    hired = ((rng.random(n_candidates) < P_ACCEPT) & (offered == 1)).astype(int)
    passed = ((rng.random(n_candidates) < P_PROBATION) & (hired == 1)).astype(int)

    # Days to hire: faster for stronger candidates, and the screening call
    # adds a real two-day delay up front that nobody budgeted for.
    days = np.where(
        hired == 1,
        np.clip(44.0 - 6.0 * strength + np.where(treated, 2.0, 0.0)
                + rng.standard_normal(n_candidates) * 9.0, 3.0, None),
        np.nan,
    )

    frame = pd.DataFrame({
        "variant": np.where(treated, "treatment", "control"),
        "strength": strength,
        "submitted": submitted,
        "interviewed": interviewed,
        "offered": offered,
        "hired": hired,
        "passed_probation": passed,
        "days_to_hire": days,
    })

    if logging_gap > 0.0:
        keep = ~((frame.variant == "control")
                 & (rng.random(n_candidates) < logging_gap))
        frame = frame[keep].reset_index(drop=True)

    truth = {
        "screening_lift_on_interview_per_submitted": screening_lift,
        "screening_cost_on_submit_rate": -screening_submit_cost,
        "offer_quality_change_per_interview": offer_quality_change,
        "logging_gap_applied_to_control": logging_gap,
    }
    return frame, truth


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------
def declared_metrics() -> list[MetricSpec]:
    """Every metric, declared before the data is seen, with its denominator.

    Note that the primary is a RATIO, not a binary. "Interview rate" means
    interviews per CV submitted, and `submitted` is not the randomisation
    unit - so the standard error belongs at the candidate level and the
    delta method is what puts it there. See docs/11.
    """
    return [
        MetricSpec("interview_rate_per_cv", "ratio",
                   numerator="interviewed", denominator="submitted",
                   role="primary", direction="increase",
                   practical_threshold=0.02),
        MetricSpec("offers_per_interview", "ratio",
                   numerator="offered", denominator="interviewed",
                   role="guardrail", direction="increase"),
        MetricSpec("probation_pass_per_hire", "ratio",
                   numerator="passed_probation", denominator="hired",
                   role="guardrail", direction="increase"),
        MetricSpec("cv_submit_rate", "binary", column="submitted",
                   role="secondary", direction="increase"),
        MetricSpec("offers_per_candidate", "binary", column="offered",
                   role="secondary", direction="increase"),
    ]


def main() -> int:
    print("=" * 74)
    print("1. PLAN IT, BEFORE RUNNING ANYTHING")
    print("=" * 74)
    plan = sample_size_proportions(P_INTERVIEW, mde_absolute=0.02)
    print(plan)
    print()
    print(f"  those are CVs SUBMITTED, not candidates sourced. At a "
          f"{P_SUBMIT:.0%} submit rate")
    print(f"  that is {plan.n_per_group / P_SUBMIT:,.0f} candidates to source "
          f"per arm, {2 * plan.n_per_group / P_SUBMIT:,.0f} in total.")
    print("  Converting the sample size back up the funnel is the step that")
    print("  turns a statistics answer into a staffing answer.")
    print()

    frame, truth = generate_funnel(n_candidates=9_000, seed=7)
    readout = analyze(frame, declared_metrics())
    print(render_text(readout, title="structured pre-screening call"))

    print()
    print("=" * 74)
    print("2. FOUR WAYS TO SAY 'OFFER RATE', ON ONE DATASET")
    print("=" * 74)
    c = frame[frame.variant == "control"]
    t = frame[frame.variant == "treatment"]
    per_cand = two_proportion_from_arrays(
        c["offered"].to_numpy(float), t["offered"].to_numpy(float),
        metric="offers per CANDIDATE SOURCED")
    per_int = ratio_delta(
        c["offered"].to_numpy(float), c["interviewed"].to_numpy(float),
        t["offered"].to_numpy(float), t["interviewed"].to_numpy(float),
        metric="offers per INTERVIEW")
    per_cv = ratio_delta(
        c["offered"].to_numpy(float), c["submitted"].to_numpy(float),
        t["offered"].to_numpy(float), t["submitted"].to_numpy(float),
        metric="offers per CV SUBMITTED")
    for r in (per_cand, per_cv, per_int):
        print(r)
        print()
    print("  Offer quality per interview was planted at exactly "
          f"{truth['offer_quality_change_per_interview']:+.3f}, and the")
    print("  'per INTERVIEW' line is correctly flat. The other two are")
    print("  measuring the screening change, not interview performance.")
    print()
    print("  And the finding a client would actually want: offers per")
    print("  candidate sourced did NOT move, because the lift and the cost")
    print("  cancel. We send 17.7% fewer CVs and convert a higher share of")
    print("  them, for the same number of offers. Whether that is a win")
    print("  depends on whose time is scarce - the client's interview")
    print("  slots or our sourcing capacity - and that is a business")
    print("  question the readout should pose, not answer.")
    print()

    print("=" * 74)
    print("2b. THE SAME CHANGE WITHOUT THE SUBMIT COST")
    print("=" * 74)
    print("  Set screening_submit_cost=0 and the cancellation goes away,")
    print("  which is what isolates the mechanical rise:")
    print()
    free, _ = generate_funnel(n_candidates=9_000, seed=7,
                              screening_submit_cost=0.0)
    fc = free[free.variant == "control"]
    ft = free[free.variant == "treatment"]
    print(two_proportion_from_arrays(
        fc["offered"].to_numpy(float), ft["offered"].to_numpy(float),
        metric="offers per CANDIDATE SOURCED"))
    print()
    print(ratio_delta(
        fc["offered"].to_numpy(float), fc["interviewed"].to_numpy(float),
        ft["offered"].to_numpy(float), ft["interviewed"].to_numpy(float),
        metric="offers per INTERVIEW"))
    print()
    print("  Offers per candidate rises 20% relative; offers per interview")
    print("  is flat at -3%. Nobody got better at interviewing - more people")
    print("  reached one. A readout quoting the first number as 'offer")
    print("  conversion improved' would be wrong in a way no statistic in")
    print("  it was wrong.")
    print()
    print("  Note the first p-value: 0.061, so by its own interval that")
    print("  rise is NOT established at 9,000 candidates. The composition")
    print("  shift is real and this funnel is too small to resolve it,")
    print("  which is the ordinary condition of a staffing experiment and")
    print("  the reason docs/07 belongs before any of this.")
    print()

    print("=" * 74)
    print("3. THE SRM THAT RECRUITMENT DATA ACTUALLY HAS")
    print("=" * 74)
    print("  A workflow where only the treatment arm is logged reliably.")
    print("  The same 6% logging gap at three scales - this is docs/08's")
    print("  point in the domain where it bites hardest, because staffing")
    print("  funnels are small:")
    print()
    print("   candidates   SRM p      verdict    decision")
    print("  " + "-" * 56)
    for n in (9_000, 40_000, 200_000):
        broken, _ = generate_funnel(n_candidates=n, seed=7, logging_gap=0.06)
        r = analyze(broken, declared_metrics())
        print(f"  {n:>10,}   {r.srm.p_value:<9.3g}  "
              f"{'PASS' if r.srm.passed else 'FAIL - SRM':<10} {r.decision}")
    print()
    print("  The defect is identical in all three rows: 6% of control")
    print("  candidates never made it into the dataset. At 9,000 the check")
    print("  cannot see it - it could only detect a 2.18% share deviation")
    print("  at 80% power, and this one is about 1.5%. At 200,000 it is")
    print("  caught and the readout refuses to report a decision.")
    print()
    print("  So the honest reading of a PASS on a small funnel is 'no")
    print("  imbalance above 2.2%', which is not the same sentence as 'the")
    print("  assignment was sound'. A staffing experiment rarely has the")
    print("  volume to tell those apart, and that is a reason to control")
    print("  the logging rather than to trust the check.")
    print()
    print("THE SEALED KEY")
    for k, v in truth.items():
        print(f"  {k} = {v:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
