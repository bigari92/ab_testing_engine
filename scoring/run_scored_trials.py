#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the DECISION, not the statistics. Blind, against a sealed key.

WHAT THIS ADDS THAT THE PROOFS DO NOT
proofs/ checks statistical properties one at a time: this interval covers,
that test holds alpha, this correction controls FDR. Every one of those can
hold while the thing built on top of them still gives bad advice, because
the advice comes from `analyze._decide` - an ordering of checks and a set
of thresholds that no individual statistic validates.

So this runs the whole pipeline the way it would actually be used:

  1. Draw a random truth. Effect size, sample size, whether a guardrail
     harm is planted, whether the assignment is broken, how heavy the
     tails are, whether there is a novelty effect.
  2. Generate the data from it.
  3. Run `analyze` on the frame ALONE. The metric declaration is fixed
     before any trial runs and never varies, because a declaration that
     adapts to the data is the thing experiment discipline exists to
     prevent. The truth object is not in scope.
  4. Only then open the key, and score.

THE DISCIPLINE, AND WHY IT IS STRUCTURAL RATHER THAN PROMISED
`_analyse_blind` takes a DataFrame and returns a Readout. It cannot see
the truth because the truth is not an argument. `_score` takes the readout
and the truth and never touches the frame. Nothing is stopping anyone from
merging the two functions, and that is precisely why they are separate and
why this paragraph exists.

WHAT IS SCORED, in the order it matters
  FALSE SHIP   shipping when there was nothing worth shipping, or when a
               guardrail was harmed. The expensive error, and the one a
               readout is for preventing.
  FALSE FLAT   declaring "no effect big enough to matter" when there WAS
               one. The quiet error: it closes the question and the
               effect is never looked for again.
  HARM MISSED  a planted guardrail regression that did not block the ship.
  SRM MISSED / FALSE ALARM
  WIN FOUND    plain statistical power, reported last because it is the
               only one of these a bigger sample fixes.

Usage:
    python scoring/run_scored_trials.py              400 trials
    python scoring/run_scored_trials.py --trials 50  a quick pass
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder in ("src", "data"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from generate import ExperimentTruth, generate_experiment  # noqa: E402

from explab.analyze import MetricSpec, Readout, analyze  # noqa: E402

# --------------------------------------------------------------------------
# The declaration. Fixed before any trial runs, identical for every trial.
# --------------------------------------------------------------------------
PRACTICAL_THRESHOLD = 0.004      # 0.4 points of conversion

METRICS = [
    MetricSpec("conversion", "binary", column="converted", role="primary",
               direction="increase", practical_threshold=PRACTICAL_THRESHOLD),
    # Refunds per CONVERTER, not per user.
    #
    # The first version of this file declared the guardrail the obvious
    # way - `binary` on the `refunded` column - and this harness is how
    # that was caught. `refunded` is 0 for everyone who did not convert,
    # so that metric is refunds per USER, which is P(convert) times
    # P(refund | converted). A treatment that raises conversion raises it
    # mechanically, with refund propensity untouched, and the guardrail
    # then blocked 77% of genuine wins.
    #
    # proof_guardrail_denominator.py measures the whole thing: the
    # decomposition is exact, composition accounts for 110% of the
    # apparent harm, and the propensity term - the only part a refund
    # guardrail is asking about - comes out NEGATIVE.
    MetricSpec("refund_rate", "ratio", numerator="refunded",
               denominator="converted", role="guardrail",
               direction="decrease"),
    MetricSpec("revenue", "continuous", column="revenue", role="secondary"),
    MetricSpec("minutes", "continuous", column="minutes",
               covariate="pre_minutes", role="secondary"),
    MetricSpec("ctr", "ratio", numerator="clicks", denominator="impressions",
               role="secondary"),
]


def _analyse_blind(frame) -> Readout:
    """The only function that sees the data. It cannot see the truth."""
    return analyze(frame, METRICS)


# --------------------------------------------------------------------------
# drawing a truth
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Trial:
    kind: str
    kwargs: dict


def draw_trial(rng: np.random.Generator, index: int) -> Trial:
    """One random experiment, from a mixture weighted like real traffic.

    Most experiments do nothing - that is the single most important fact
    about experimentation programmes and it has to be reflected here, or
    the false-ship rate is measured on a population where shipping is
    usually right. So 40% of trials are true nulls.

    `kind` is the category the decision will be scored against, and it is
    derived from the parameters being PLANTED, never from the data.
    """
    n_users = int(rng.choice([20_000, 50_000, 120_000, 300_000]))
    base = dict(
        n_users=n_users,
        seed=1_000_000 + index,
        days=21,
        # Realistic nuisance, present in every trial regardless of kind,
        # so the decision is never scored on clean textbook data.
        outlier_rate=float(rng.choice([0.0, 0.002, 0.01])),
        outlier_scale=40.0,
        novelty_strength=float(rng.choice([0.0, 0.0, 1.5])),
        ratio_heterogeneity=float(rng.choice([8.0, 20.0])),
    )
    roll = rng.random()

    if roll < 0.40:
        # Nothing happening. The majority case.
        return Trial("true_null", {**base, "effect_conversion": 0.0})

    if roll < 0.62:
        # A real effect worth shipping for: at least the practical
        # threshold, so "ship" is the right answer if it is detected.
        effect = float(rng.uniform(PRACTICAL_THRESHOLD, 0.02))
        return Trial("real_win", {**base, "effect_conversion": effect})

    if roll < 0.74:
        # Real, but smaller than the threshold. Shipping is wrong even
        # though the effect exists - it is not worth the change.
        effect = float(rng.uniform(0.0005, PRACTICAL_THRESHOLD * 0.9))
        return Trial("below_threshold", {**base,
                                         "effect_conversion": effect})

    if roll < 0.86:
        # A guardrail regression, with or without a primary win. Either
        # way the answer is not "ship".
        return Trial("guardrail_harm", {
            **base,
            "effect_conversion": float(rng.choice([0.0, 0.01])),
            "effect_refund": float(rng.uniform(0.01, 0.04)),
        })

    if roll < 0.95:
        # Broken assignment. Everything below the SRM check is void.
        share = float(rng.uniform(0.504, 0.52))
        if rng.random() < 0.5:
            share = 1.0 - share
        return Trial("broken_assignment", {
            **base, "n_users": max(n_users, 120_000),
            "assignment_share": share,
            "effect_conversion": float(rng.choice([0.0, 0.01])),
        })

    # Heterogeneous: a real effect in one segment only. The overall
    # average may or may not clear the threshold, so this is scored
    # loosely - it is here to make sure nothing crashes or lies.
    # hetero_segment names a LEVEL, not a column: the effect lands on
    # mobile users only, and the other devices get nothing.
    return Trial("segment_only", {
        **base, "effect_conversion": 0.0,
        "hetero_segment": "mobile", "hetero_effect": 0.02,
    })


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
SHIP = "ship"


@dataclass
class Tally:
    """Counts, kept as counts so every rate printed can be recomputed."""
    trials: int = 0
    by_kind: Counter = None
    decisions: Counter = None
    # the errors that matter
    false_ship: int = 0
    false_ship_eligible: int = 0
    false_flat: int = 0
    false_flat_eligible: int = 0
    harm_missed: int = 0
    harm_total: int = 0
    srm_caught: int = 0
    srm_total: int = 0
    srm_false_alarm: int = 0
    srm_clean_total: int = 0
    win_found: int = 0
    win_total: int = 0
    powered_win_found: int = 0
    powered_win_total: int = 0
    sub_threshold_shipped: int = 0
    sub_threshold_total: int = 0
    null_flat: int = 0
    null_total: int = 0
    confusion: Counter = None
    failures: list = None

    def __post_init__(self):
        self.by_kind = Counter()
        self.decisions = Counter()
        self.confusion = Counter()
        self.failures = []


def _score(tally: Tally, kind: str, readout: Readout,
           truth: ExperimentTruth, index: int) -> None:
    """Open the key and judge. Never touches the data."""
    decision = readout.decision
    tally.trials += 1
    tally.by_kind[kind] += 1
    tally.decisions[decision] += 1
    tally.confusion[(kind, decision)] += 1

    srm_planted = bool(truth.srm_planted)
    if srm_planted:
        tally.srm_total += 1
        if not readout.srm.passed:
            tally.srm_caught += 1
    else:
        tally.srm_clean_total += 1
        if not readout.srm.passed:
            tally.srm_false_alarm += 1

    harmed = truth.effect_refund > 0.0
    if harmed:
        tally.harm_total += 1
        if decision == SHIP:
            tally.harm_missed += 1
            tally.failures.append(
                f"trial {index}: SHIPPED with {truth.effect_refund:+.3f} "
                f"planted on the refund guardrail"
            )

    # Everything below is judged against the REALISED effect from the key,
    # not against the `kind` label. The labels describe how a trial was
    # generated; the key records what actually ended up in the rows. Those
    # differ - a "segment_only" trial with a +0.02 effect on mobile users
    # has a real overall effect of about +0.01, so shipping it is correct
    # even though no overall effect was requested. Scoring on the label
    # counted those as errors, which is a bug in the SCORER, and the sort
    # that quietly makes a tool look worse than it is.
    realised = float(truth.realised_effect_conversion)
    worth_shipping = realised >= PRACTICAL_THRESHOLD
    nothing_there = realised <= 0.0

    # FALSE SHIP, defined narrowly so it means something: shipped when
    # there was nothing at all, or shipped over a real guardrail harm. A
    # broken assignment is excluded and counted as an SRM miss instead,
    # because that is a different failure.
    if not srm_planted and (nothing_there or harmed):
        tally.false_ship_eligible += 1
        if decision == SHIP:
            tally.false_ship += 1
            if not harmed:
                tally.failures.append(
                    f"trial {index}: SHIPPED on a {kind} with a realised "
                    f"effect of {realised:+.5f}"
                )

    # Shipped on a real but sub-threshold effect. NOT scored as an error:
    # the effect existed, the test was right to find it, and whether a
    # +0.003 lift is worth a ship when you declared +0.004 is a judgement
    # the readout cannot make for you. Reported so it is visible rather
    # than hidden inside one of the other rates.
    if not srm_planted and not harmed and 0.0 < realised < PRACTICAL_THRESHOLD:
        tally.sub_threshold_total += 1
        if decision == SHIP:
            tally.sub_threshold_shipped += 1

    if worth_shipping and not srm_planted and not harmed:
        tally.false_flat_eligible += 1
        tally.win_total += 1
        if decision == SHIP:
            tally.win_found += 1
        # FALSE FLAT: "no effect big enough to matter" when there was one.
        if decision == "flat":
            tally.false_flat += 1
            tally.failures.append(
                f"trial {index}: called FLAT with a realised effect of "
                f"{realised:+.5f}, above the {PRACTICAL_THRESHOLD} threshold"
            )
        # Power is only interpretable against what the experiment could
        # detect. The readout reports its own MDE, so use it: of the wins
        # that were ADEQUATELY SIZED, how many were found? A miss on an
        # underpowered experiment is not a failure of the readout, it is
        # the readout correctly declining to claim what it cannot see.
        detectable = readout.primary.detectable if readout.primary else None
        if detectable is not None and realised >= detectable:
            tally.powered_win_total += 1
            if decision == SHIP:
                tally.powered_win_found += 1

    # Of the true nulls, how often is the readout decisive rather than
    # merely cautious? Both are correct; "flat" is more useful.
    if nothing_there and not srm_planted and not harmed:
        tally.null_total += 1
        if decision == "flat":
            tally.null_flat += 1


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _rate(num: int, den: int) -> str:
    if den == 0:
        return "    n/a"
    return f"{num / den:7.2%}"


def _mc_se(num: int, den: int) -> float:
    if den == 0:
        return float("nan")
    p = num / den
    return float(np.sqrt(max(p * (1 - p), 1e-12) / den))


def report(tally: Tally) -> int:
    line = "=" * 78
    print(line)
    print("BLIND SCORING: the decision function against a sealed key")
    print(line)
    print(f"{tally.trials} trials, each generated from a random truth, "
          f"analysed without it,")
    print("then scored against it. One fixed metric declaration throughout.")
    print()

    print("TRUTHS DRAWN")
    for kind, count in sorted(tally.by_kind.items()):
        print(f"  {kind:20s} {count:5d}  ({count / tally.trials:5.1%})")
    print()

    print("DECISIONS RETURNED")
    for decision, count in sorted(tally.decisions.items()):
        print(f"  {decision:20s} {count:5d}  ({count / tally.trials:5.1%})")
    print()

    print("CONFUSION MATRIX  (truth down, decision across)")
    decisions = sorted(tally.decisions)
    header = "  " + " " * 20 + "".join(f"{d:>14s}" for d in decisions)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kind in sorted(tally.by_kind):
        row = f"  {kind:20s}"
        for decision in decisions:
            count = tally.confusion[(kind, decision)]
            row += f"{count:>14d}" if count else f"{'-':>14s}"
        print(row)
    print()

    print("THE NUMBERS THAT MATTER")
    print("  " + "-" * 74)
    rows = [
        ("false ship rate", tally.false_ship, tally.false_ship_eligible,
         "shipped when there was nothing to ship or a guardrail was harmed"),
        ("harm missed", tally.harm_missed, tally.harm_total,
         "a planted guardrail regression that did not block the ship"),
        ("false flat rate", tally.false_flat, tally.false_flat_eligible,
         "called flat when a real effect above the threshold was present"),
        ("SRM missed", tally.srm_total - tally.srm_caught, tally.srm_total,
         "a planted assignment imbalance the check did not flag"),
        ("SRM false alarm", tally.srm_false_alarm, tally.srm_clean_total,
         "a clean split flagged as broken"),
        ("win found (all)", tally.win_found, tally.win_total,
         "a real effect above the threshold correctly shipped, including "
         "experiments far too small to see it"),
        ("win found (powered)", tally.powered_win_found,
         tally.powered_win_total,
         "the same, restricted to experiments whose own reported MDE was "
         "below the planted effect - the number that judges the readout "
         "rather than the sample size"),
        ("shipped sub-threshold", tally.sub_threshold_shipped,
         tally.sub_threshold_total,
         "a real effect smaller than the declared threshold, shipped. NOT "
         "an error: a judgement the readout cannot make for you"),
        ("decisive on a null", tally.null_flat, tally.null_total,
         "true nulls called flat rather than inconclusive - both correct, "
         "flat is more useful"),
    ]
    for label, num, den, note in rows:
        se = _mc_se(num, den)
        se_text = f" +/- {se:.2%}" if den else ""
        print(f"  {label:22s} {_rate(num, den)}{se_text:>12s}  "
              f"({num}/{den})")
        print(f"  {'':22s} {note}")
    print()

    if tally.failures:
        print(f"EVERY SCORED FAILURE ({len(tally.failures)}), listed")
        print("  Not summarised away: each one is a case where the readout")
        print("  gave advice the key says was wrong.")
        for text in tally.failures[:40]:
            print(f"  - {text}")
        if len(tally.failures) > 40:
            print(f"  ... and {len(tally.failures) - 40} more")
        print()

    # ---- the verdict ------------------------------------------------------
    # These are the thresholds the decision function is held to. They are
    # not statistical guarantees - a decision rule has no alpha - so they
    # are stated as what a reader should be able to rely on, and a run that
    # breaches one exits non-zero.
    problems: list[str] = []
    if tally.false_ship_eligible:
        rate = tally.false_ship / tally.false_ship_eligible
        if rate > 0.05:
            problems.append(f"false ship rate {rate:.2%} exceeds 5%")
    if tally.harm_total:
        rate = tally.harm_missed / tally.harm_total
        if rate > 0.25:
            problems.append(f"harm missed {rate:.2%} exceeds 25%")
    if tally.false_flat_eligible:
        rate = tally.false_flat / tally.false_flat_eligible
        if rate > 0.10:
            problems.append(f"false flat rate {rate:.2%} exceeds 10%")
    if tally.srm_total:
        rate = (tally.srm_total - tally.srm_caught) / tally.srm_total
        if rate > 0.30:
            problems.append(f"SRM missed {rate:.2%} exceeds 30%")
    if tally.srm_clean_total:
        rate = tally.srm_false_alarm / tally.srm_clean_total
        if rate > 0.01:
            problems.append(f"SRM false alarm {rate:.2%} exceeds 1%")
    if tally.powered_win_total:
        rate = tally.powered_win_found / tally.powered_win_total
        if rate < 0.70:
            problems.append(
                f"adequately powered wins found only {rate:.2%} of the time, "
                f"below 70%"
            )

    print(line)
    if problems:
        print("SCORING FAILED")
        for text in problems:
            print(f"  - {text}")
        print(line)
        return 1
    print("SCORING PASSED: every rate inside the limits stated above")
    print(line)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    tally = Tally()

    for index in range(args.trials):
        trial = draw_trial(rng, index)
        frame, truth = generate_experiment(**trial.kwargs)
        readout = _analyse_blind(frame)          # the truth is not in scope
        _score(tally, trial.kind, readout, truth, index)
        if (index + 1) % 50 == 0:
            print(f"  ... {index + 1}/{args.trials} trials", flush=True)

    print()
    return report(tally)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
