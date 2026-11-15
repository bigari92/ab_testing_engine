#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""From a table of assigned users to a decision somebody can act on.

This is where the pieces become a readout. The order of operations is the
opinionated part, and it is the part worth arguing about, so it is written
out rather than buried:

    1. SAMPLE RATIO FIRST. If assignment is broken, no effect estimate in
       the readout means anything, so nothing else runs. A readout that
       prints a lift above an SRM failure is worse than no readout,
       because somebody will quote the lift.

    2. THE PRIMARY METRIC, declared in advance, uncorrected, at the alpha
       the experiment was designed for. One metric decides. If two metrics
       decide, neither does.

    3. GUARDRAILS, uncorrected and two-sided. Deliberately not corrected:
       a guardrail exists to catch harm, and raising its bar makes the
       readout blind to the thing it was put there for. See the discussion
       in multiplicity.py.

    4. SECONDARY METRICS, Benjamini-Hochberg corrected, labelled
       exploratory, and structurally unable to change the decision. They
       exist to generate the next hypothesis.

    5. THE DECISION, from rules stated in `decide()` below, including the
       distinction most readouts miss: "no effect" and "we could not have
       seen an effect" are different outcomes and get different words.

WHAT "FLAT" MEANS AND WHY IT IS NOT "INCONCLUSIVE"
-------------------------------------------------
A non-significant result splits in two, and the split is the most useful
thing a readout can say:

  FLAT         the whole confidence interval lies inside the range of
               effects too small to care about. The experiment has
               ANSWERED the question: there is no effect of practical
               size. Ship it for other reasons or do not, but stop
               investigating.
  INCONCLUSIVE the interval still contains effects you would have shipped
               for. The experiment has not answered anything. Running it
               longer is the only honest next step.

Telling them apart needs a number nobody wants to supply: the smallest
effect worth shipping. `MetricSpec.practical_threshold`. Without it this
module cannot distinguish the two cases and says so in the readout rather
than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .cuped import CupedResult, cuped_adjust
from .inference import (TestResult, ratio_delta, two_proportion_from_arrays,
                        welch_t)
from .multiplicity import Adjusted, benjamini_hochberg
from .power import mde_means, mde_proportions
from .srm import SRMResult, srm_check

__all__ = ["MetricSpec", "Readout", "analyze"]

Kind = Literal["binary", "continuous", "ratio"]
Role = Literal["primary", "secondary", "guardrail"]
Direction = Literal["increase", "decrease"]

Decision = Literal["ship", "do-not-ship", "flat", "inconclusive", "invalid"]


@dataclass(frozen=True)
class MetricSpec:
    """One metric, declared before the data is looked at.

    name                  what it is called in the readout
    kind                  binary (0/1), continuous, or ratio of two columns
    column                the value column, for binary and continuous
    numerator/denominator the two columns, for ratio
    role                  primary | secondary | guardrail
    direction             which way counts as an improvement
    practical_threshold   the smallest effect, in metric units, that would
                          change a decision. Required to tell "flat" from
                          "inconclusive"; omit it and the readout says the
                          distinction could not be made.
    covariate             a PRE-TREATMENT column to use for CUPED on a
                          continuous metric. Must be measured before
                          assignment; see cuped.py for what goes wrong
                          when it is not.
    """
    name: str
    kind: Kind
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    role: Role = "secondary"
    direction: Direction = "increase"
    practical_threshold: float | None = None
    covariate: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "ratio":
            if not (self.numerator and self.denominator):
                raise ValueError(
                    f"metric {self.name!r} is a ratio and needs numerator and "
                    "denominator columns"
                )
            if self.covariate:
                raise ValueError(
                    f"metric {self.name!r}: CUPED on a ratio metric needs the "
                    "covariate carried through the delta method, which this "
                    "project does not implement. Drop the covariate or use a "
                    "per-user metric."
                )
        else:
            if not self.column:
                raise ValueError(f"metric {self.name!r} needs a column")
        if self.kind == "binary" and self.covariate:
            raise ValueError(
                f"metric {self.name!r}: CUPED is implemented for continuous "
                "metrics here. On a 0/1 outcome the pre-period correlation is "
                "usually near zero anyway - see docs/10."
            )
        if self.practical_threshold is not None and self.practical_threshold <= 0:
            raise ValueError(
                f"metric {self.name!r}: practical_threshold must be positive "
                "(it is a magnitude, and direction is given separately)"
            )


@dataclass(frozen=True)
class MetricOutcome:
    """A metric's result plus how it was obtained and what it could detect."""
    spec: MetricSpec
    result: TestResult
    cuped: CupedResult | None
    detectable: float | None
    q_value: float | None = None
    moved_in_good_direction: bool = False
    beyond_practical: bool | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Readout:
    """Everything the readout concluded, and the evidence for it."""
    srm: SRMResult
    primary: MetricOutcome | None
    guardrails: tuple[MetricOutcome, ...]
    secondary: tuple[MetricOutcome, ...]
    secondary_correction: Adjusted | None
    decision: Decision
    reasons: tuple[str, ...]
    alpha: float
    n_control: int
    n_treatment: int

    @property
    def all_outcomes(self) -> tuple[MetricOutcome, ...]:
        out: list[MetricOutcome] = []
        if self.primary is not None:
            out.append(self.primary)
        out.extend(self.guardrails)
        out.extend(self.secondary)
        return tuple(out)


def _split(
    frame: pd.DataFrame, variant_column: str, control_label: str, treatment_label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if variant_column not in frame.columns:
        raise ValueError(f"no {variant_column!r} column in the frame")
    control = frame[frame[variant_column] == control_label]
    treatment = frame[frame[variant_column] == treatment_label]
    if control.empty or treatment.empty:
        raise ValueError(
            f"one arm is empty: control={len(control)}, treatment={len(treatment)}. "
            f"Labels looked for were {control_label!r} and {treatment_label!r}; "
            f"the column holds {sorted(frame[variant_column].unique())[:5]}"
        )
    return control, treatment


def _run_metric(
    spec: MetricSpec,
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    alpha: float,
) -> MetricOutcome:
    notes: list[str] = []
    cuped_result: CupedResult | None = None
    confidence = 1.0 - alpha

    if spec.kind == "ratio":
        assert spec.numerator and spec.denominator
        for col in (spec.numerator, spec.denominator):
            if col not in control.columns:
                raise ValueError(f"metric {spec.name!r}: no column {col!r}")
        result = ratio_delta(
            control[spec.numerator].to_numpy(float),
            control[spec.denominator].to_numpy(float),
            treatment[spec.numerator].to_numpy(float),
            treatment[spec.denominator].to_numpy(float),
            metric=spec.name, confidence=confidence,
        )
        detectable = None
        notes.append(
            "randomisation unit is the row, denominator unit is finer; "
            "the delta method keeps the standard error at the randomised unit"
        )
    else:
        assert spec.column
        if spec.column not in control.columns:
            raise ValueError(f"metric {spec.name!r}: no column {spec.column!r}")
        c = control[spec.column].to_numpy(float)
        t = treatment[spec.column].to_numpy(float)

        if spec.kind == "binary":
            result = two_proportion_from_arrays(
                c, t, metric=spec.name, confidence=confidence
            )
            pooled_rate = float(np.concatenate([c, t]).mean())
            try:
                detectable = mde_proportions(
                    pooled_rate, min(c.size, t.size), alpha=alpha
                )
            except ValueError:
                detectable = None
        else:
            if spec.covariate:
                if spec.covariate not in control.columns:
                    raise ValueError(
                        f"metric {spec.name!r}: no covariate column "
                        f"{spec.covariate!r}"
                    )
                cuped_result = cuped_adjust(
                    c, control[spec.covariate].to_numpy(float),
                    t, treatment[spec.covariate].to_numpy(float),
                )
                result = welch_t(
                    cuped_result.adjusted_control, cuped_result.adjusted_treatment,
                    metric=spec.name, confidence=confidence,
                )
                notes.append(
                    f"CUPED on {spec.covariate}: {cuped_result.reduction:.1%} of "
                    f"variance removed, worth {cuped_result.effective_n_gain:.2f}x "
                    "the sample"
                )
                notes.extend(cuped_result.warnings)
                sd = float(np.concatenate([
                    cuped_result.adjusted_control, cuped_result.adjusted_treatment
                ]).std(ddof=1))
            else:
                result = welch_t(c, t, metric=spec.name, confidence=confidence)
                sd = float(np.concatenate([c, t]).std(ddof=1))
            try:
                detectable = mde_means(sd, min(c.size, t.size), alpha=alpha)
            except ValueError:
                detectable = None

    notes.extend(result.notes)

    good = (
        result.estimate > 0 if spec.direction == "increase" else result.estimate < 0
    )
    beyond: bool | None = None
    if spec.practical_threshold is not None:
        # Does the interval contain any effect big enough to act on?
        lo, hi = result.ci
        thr = spec.practical_threshold
        beyond = bool(hi > thr or lo < -thr)
    return MetricOutcome(
        spec=spec, result=result, cuped=cuped_result, detectable=detectable,
        moved_in_good_direction=bool(good), beyond_practical=beyond,
        notes=tuple(notes),
    )


def _decide(
    srm: SRMResult,
    primary: MetricOutcome | None,
    guardrails: Sequence[MetricOutcome],
    alpha: float,
) -> tuple[Decision, tuple[str, ...]]:
    """The decision rules, in one place, in order.

    Written as a function of the outcomes alone so that
    scoring/run_scored_trials.py can measure how often it is right.
    """
    reasons: list[str] = []

    if not srm.passed:
        return "invalid", (
            f"sample ratio mismatch (p={srm.p_value:.3g}): the arms are not "
            "exchangeable, so no effect estimate in this readout is "
            "interpretable. Fix assignment or logging and re-run.",
        )

    # Guardrails are checked before the primary is believed, because harm
    # outranks improvement.
    harmed = [
        g for g in guardrails
        if g.result.significant and not g.moved_in_good_direction
    ]
    for g in harmed:
        reasons.append(
            f"guardrail {g.spec.name} moved the wrong way: "
            f"{g.result.estimate:+.6g} "
            f"[{g.result.ci[0]:+.4g}, {g.result.ci[1]:+.4g}], "
            f"p={g.result.p_value:.4g} (uncorrected, by design)"
        )

    if primary is None:
        reasons.append("no metric was declared primary, so nothing decides")
        return ("do-not-ship" if harmed else "inconclusive"), tuple(reasons)

    r = primary.result
    if r.significant:
        if primary.moved_in_good_direction:
            reasons.insert(
                0,
                f"primary metric {primary.spec.name} moved "
                f"{r.estimate:+.6g} ({r.relative_lift:+.2%}), interval "
                f"[{r.ci[0]:+.4g}, {r.ci[1]:+.4g}] excludes zero at "
                f"{r.confidence:.0%}",
            )
            if harmed:
                reasons.append(
                    "the primary metric won and a guardrail lost: this is a "
                    "trade-off for a human, not a verdict for a script"
                )
                return "do-not-ship", tuple(reasons)
            return "ship", tuple(reasons)
        reasons.insert(
            0,
            f"primary metric {primary.spec.name} moved against the declared "
            f"direction: {r.estimate:+.6g} "
            f"[{r.ci[0]:+.4g}, {r.ci[1]:+.4g}], p={r.p_value:.4g}",
        )
        return "do-not-ship", tuple(reasons)

    # Not significant. Flat or inconclusive?
    if primary.beyond_practical is None:
        reasons.insert(
            0,
            f"primary metric {primary.spec.name} is not significant "
            f"(interval [{r.ci[0]:+.4g}, {r.ci[1]:+.4g}] contains zero). "
            "No practical_threshold was declared, so this readout cannot "
            "tell 'no effect' from 'could not have seen one'. Declare one.",
        )
        return ("do-not-ship" if harmed else "inconclusive"), tuple(reasons)

    if primary.beyond_practical:
        reasons.insert(
            0,
            f"primary metric {primary.spec.name} is not significant, but the "
            f"interval [{r.ci[0]:+.4g}, {r.ci[1]:+.4g}] still contains effects "
            f"beyond the practical threshold of "
            f"{primary.spec.practical_threshold:.6g}. The experiment has not "
            "answered the question; it needs more units, not a decision.",
        )
        return ("do-not-ship" if harmed else "inconclusive"), tuple(reasons)

    reasons.insert(
        0,
        f"primary metric {primary.spec.name} is flat: the whole interval "
        f"[{r.ci[0]:+.4g}, {r.ci[1]:+.4g}] lies inside the practical "
        f"threshold of +/-{primary.spec.practical_threshold:.6g}. This is an "
        "answer, not a failure: there is no effect of a size worth shipping for.",
    )
    return ("do-not-ship" if harmed else "flat"), tuple(reasons)


def analyze(
    frame: pd.DataFrame,
    metrics: Sequence[MetricSpec],
    *,
    variant_column: str = "variant",
    control_label: str = "control",
    treatment_label: str = "treatment",
    expected_shares: dict[str, float] | None = None,
    alpha: float = 0.05,
    srm_threshold: float = 0.001,
) -> Readout:
    """Run the whole readout in the order described in the module docstring.

    frame            one row per randomisation unit
    metrics          the declared metric set. Exactly one may be primary.
    expected_shares  declared assignment shares, e.g. {"control": 0.5,
                     "treatment": 0.5}; defaults to even
    alpha            two-sided level for the primary and the guardrails
    srm_threshold    see srm.py; 0.001 by default

    Raises on more than one primary metric, because a readout with two
    primary metrics has none.
    """
    if not metrics:
        raise ValueError("no metrics declared")
    primaries = [m for m in metrics if m.role == "primary"]
    if len(primaries) > 1:
        raise ValueError(
            f"{len(primaries)} metrics declared primary "
            f"({', '.join(m.name for m in primaries)}). Pick one: a readout "
            "with two primary metrics has no primary metric."
        )
    names = [m.name for m in metrics]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate metric names: {names}")

    control, treatment = _split(frame, variant_column, control_label, treatment_label)

    counts = {
        control_label: int(len(control)),
        treatment_label: int(len(treatment)),
    }
    srm = srm_check(counts, expected_shares, threshold=srm_threshold)

    primary_outcome: MetricOutcome | None = None
    guardrail_outcomes: list[MetricOutcome] = []
    secondary_outcomes: list[MetricOutcome] = []

    # Metrics are still computed when SRM fails: the numbers are needed to
    # debug the break, and hiding them invites someone to compute them by
    # hand without the warning attached. The DECISION is what refuses.
    for spec in metrics:
        outcome = _run_metric(spec, control, treatment, alpha)
        if spec.role == "primary":
            primary_outcome = outcome
        elif spec.role == "guardrail":
            guardrail_outcomes.append(outcome)
        else:
            secondary_outcomes.append(outcome)

    correction: Adjusted | None = None
    if secondary_outcomes:
        correction = benjamini_hochberg(
            {o.spec.name: o.result.p_value for o in secondary_outcomes}, alpha=alpha
        )
        q_by_name = dict(zip(correction.labels, correction.p_adjusted))
        secondary_outcomes = [
            MetricOutcome(
                spec=o.spec, result=o.result, cuped=o.cuped,
                detectable=o.detectable, q_value=q_by_name.get(o.spec.name),
                moved_in_good_direction=o.moved_in_good_direction,
                beyond_practical=o.beyond_practical, notes=o.notes,
            )
            for o in secondary_outcomes
        ]

    decision, reasons = _decide(srm, primary_outcome, guardrail_outcomes, alpha)
    return Readout(
        srm=srm, primary=primary_outcome,
        guardrails=tuple(guardrail_outcomes), secondary=tuple(secondary_outcomes),
        secondary_correction=correction, decision=decision, reasons=reasons,
        alpha=alpha, n_control=counts[control_label],
        n_treatment=counts[treatment_label],
    )
