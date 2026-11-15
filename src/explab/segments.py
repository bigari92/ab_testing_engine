#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segment analysis: where false findings are manufactured.

"It didn't win overall, but it won for mobile users in Germany."

That sentence has ended more experiments than any other, and it is almost
always noise. Twelve segments at alpha 0.05 give a 46% chance of at least
one spurious winner on a feature that does nothing. Slice by device,
country, tenure and plan at once and you are guaranteed a story.

This module does segment analysis anyway, because heterogeneous effects
are real and worth finding, and does it with four guards:

1. A MINIMUM SIZE. A segment with 40 users per arm cannot detect anything
   an experiment would act on. Segments below the floor are reported as
   unpowered rather than tested, with the effect they COULD have detected,
   so "no effect in enterprise accounts" is never confused with "we had 31
   enterprise accounts".

2. AN INTERACTION TEST, not a comparison of two p-values. "Significant in
   mobile, not significant in desktop" is not evidence that the effect
   differs between them — both could be the same effect measured with
   different precision. The question "does the effect differ by segment"
   has its own test, and it is a test on the DIFFERENCE of the two effects:

       z = (effect_a - effect_b) / sqrt(se_a^2 + se_b^2)

   This is the single most common error in segment readouts and the
   cheapest to avoid.

3. BENJAMINI-HOCHBERG across the segments, with the result labelled
   exploratory. A segment finding is a hypothesis for the next experiment,
   not a decision from this one.

4. A MIX-SHIFT CHECK. If the segments themselves are distributed
   differently across arms, something is wrong before any effect is
   estimated: either randomisation is broken, or — far more often — the
   segment is defined by something that happened AFTER assignment.
   "Engaged users" is the classic: the treatment changes who becomes
   engaged, so comparing engaged users across arms compares two different
   populations. That is the mechanism behind every real Simpson's paradox
   in experimentation, and it is a data-modelling error, not a statistical
   one.

THE RULE
Segments declared before the experiment, from pre-treatment attributes
only, analysed with an interaction test and an FDR correction, reported as
exploratory. Anything else is a story.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .inference import TestResult, two_proportion_from_arrays, welch_t
from .multiplicity import benjamini_hochberg
from .power import mde_means, mde_proportions
from .statlib import chi2_cdf, norm_cdf

__all__ = ["SegmentFinding", "SegmentScan", "segment_scan", "mix_shift_check",
           "interaction_test"]


@dataclass(frozen=True)
class SegmentFinding:
    """One segment's result, with the reasons to doubt it attached."""
    segment: str
    n_control: int
    n_treatment: int
    result: TestResult | None
    powered: bool
    detectable_effect: float | None
    q_value: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        if self.result is None:
            return (
                f"  {self.segment:24s} n={self.n_control}/{self.n_treatment}  "
                f"UNPOWERED - could only detect "
                f"{self.detectable_effect:.4g} or larger"
                if self.detectable_effect is not None else
                f"  {self.segment:24s} n={self.n_control}/{self.n_treatment}  "
                f"not tested"
            )
        q = f"q={self.q_value:.4g}" if self.q_value is not None else "q=n/a"
        flag = "" if self.powered else "  [unpowered]"
        return (
            f"  {self.segment:24s} n={self.n_control}/{self.n_treatment}  "
            f"effect {self.result.estimate:+.6g} "
            f"[{self.result.ci[0]:+.4g}, {self.result.ci[1]:+.4g}]  "
            f"p={self.result.p_value:.4g} {q}{flag}"
        )


@dataclass(frozen=True)
class SegmentScan:
    """Every segment, plus the checks that say how much to believe them."""
    metric: str
    segment_column: str
    findings: tuple[SegmentFinding, ...]
    mix_shift_p: float
    mix_shift_ok: bool
    interaction_p: float | None
    overall: TestResult
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        lines = [
            f"segment scan: {self.metric} by {self.segment_column} "
            f"(EXPLORATORY - segment findings are hypotheses, not decisions)",
            f"  overall effect {self.overall.estimate:+.6g} "
            f"[{self.overall.ci[0]:+.4g}, {self.overall.ci[1]:+.4g}] "
            f"p={self.overall.p_value:.4g}",
        ]
        lines += [str(f) for f in self.findings]
        mix = "ok" if self.mix_shift_ok else "BROKEN"
        lines.append(
            f"  segment mix across arms: {mix} (p={self.mix_shift_p:.4g})"
        )
        if self.interaction_p is not None:
            lines.append(
                f"  effects differ across segments: p={self.interaction_p:.4g} "
                "(this is the question 'is it heterogeneous', tested once)"
            )
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def interaction_test(a: TestResult, b: TestResult) -> float:
    """p-value for "these two effects are different from each other".

    The test is on the difference of the two estimates, with the standard
    errors added in quadrature because the segments are disjoint samples
    and therefore independent:

        z = (est_a - est_b) / sqrt(se_a^2 + se_b^2)

    Use this instead of eyeballing two p-values. An effect of +0.05
    (p=0.01) in one segment and +0.04 (p=0.30) in another are not
    different effects; they are the same effect, one measured on more
    users.
    """
    se = float(np.sqrt(a.se ** 2 + b.se ** 2))
    if se == 0.0:
        return 1.0
    z = (a.estimate - b.estimate) / se
    return float(2.0 * (1.0 - norm_cdf(abs(z))))


def _omnibus_interaction(results: Sequence[TestResult]) -> float | None:
    """Do these k effects differ from each other at all? One test, k-1 df.

    Cochran's Q for heterogeneity, the statistic meta-analysis uses:

        w_i = 1 / se_i^2                 (inverse-variance weights)
        pooled = sum(w_i * est_i) / sum(w_i)
        Q = sum(w_i * (est_i - pooled)^2)   ~ chi-square with k-1 df

    Q asks one question once, so it needs no multiplicity correction. A
    small p means the segments really do respond differently; a large p
    means the spread you are looking at is what independent estimates of
    one common effect look like.
    """
    usable = [r for r in results if r.se > 0]
    k = len(usable)
    if k < 2:
        return None
    weights = np.array([1.0 / (r.se ** 2) for r in usable])
    estimates = np.array([r.estimate for r in usable])
    pooled = float((weights * estimates).sum() / weights.sum())
    q_stat = float((weights * (estimates - pooled) ** 2).sum())
    return float(1.0 - chi2_cdf(q_stat, k - 1))


def mix_shift_check(
    frame: pd.DataFrame,
    *,
    segment_column: str,
    variant_column: str = "variant",
    threshold: float = 0.001,
) -> tuple[float, bool]:
    """Chi-square test of independence: is the segment mix the same in both arms?

    Randomisation makes pre-treatment attributes independent of the arm, so
    on a correctly run experiment with a pre-treatment segment this test
    passes by construction. A failure therefore means one of two things,
    and both are serious:

      - the segment is not pre-treatment (it was computed from behaviour
        during the experiment), or
      - assignment is broken, in which case srm_check is probably also
        failing.

    Threshold defaults to 0.001 for the same reason srm_check does: this
    runs on every segment of every readout.
    """
    table = pd.crosstab(frame[segment_column], frame[variant_column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 1.0, True
    observed = table.to_numpy(dtype=float)
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row @ col / total
    if expected.min() < 5:
        # Chi-square unreliable; report a pass with the reason, rather than
        # a number nobody should act on.
        return 1.0, True
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p = float(1.0 - chi2_cdf(chi2, df))
    return p, bool(p >= threshold)


def segment_scan(
    frame: pd.DataFrame,
    *,
    metric_column: str,
    segment_column: str,
    variant_column: str = "variant",
    control_label: str = "control",
    treatment_label: str = "treatment",
    binary: bool | None = None,
    min_per_arm: int = 200,
    alpha: float = 0.05,
    target_effect: float | None = None,
) -> SegmentScan:
    """Estimate the effect inside every level of `segment_column`.

    frame           one row per randomisation unit
    metric_column   the outcome
    segment_column  a PRE-TREATMENT attribute. If it is not pre-treatment,
                    the mix-shift check will say so.
    binary          force the 0/1 path; inferred from the data when None
    min_per_arm     segments smaller than this are reported unpowered and
                    not tested. 200 is a deliberately low floor; for a
                    conversion metric the honest floor is in the thousands,
                    and `detectable_effect` tells you which you are in.
    target_effect   if given, a segment counts as powered only when it
                    could detect an effect this small

    Returns every segment, an FDR-corrected q-value per segment, one
    omnibus heterogeneity test, and the mix-shift check.
    """
    for col in (metric_column, segment_column, variant_column):
        if col not in frame.columns:
            raise ValueError(f"column {col!r} not in the frame")
    variants = set(frame[variant_column].unique())
    missing = {control_label, treatment_label} - variants
    if missing:
        raise ValueError(f"variant column has no rows for {sorted(missing)}")

    values = frame[metric_column].to_numpy(dtype=float)
    if binary is None:
        binary = bool(np.isin(values[~np.isnan(values)], (0.0, 1.0)).all())

    def test(c: np.ndarray, t: np.ndarray, name: str) -> TestResult:
        if binary:
            return two_proportion_from_arrays(c, t, metric=name)
        return welch_t(c, t, metric=name)

    is_c = frame[variant_column] == control_label
    is_t = frame[variant_column] == treatment_label
    overall = test(values[is_c.to_numpy()], values[is_t.to_numpy()], metric_column)

    findings: list[SegmentFinding] = []
    tested: list[tuple[str, TestResult]] = []
    for level, part in frame.groupby(segment_column, sort=True, observed=True):
        seg_values = part[metric_column].to_numpy(dtype=float)
        seg_variant = part[variant_column].to_numpy()
        c = seg_values[seg_variant == control_label]
        t = seg_values[seg_variant == treatment_label]
        n_c, n_t = c.size, t.size
        label = str(level)

        if min(n_c, n_t) < max(2, min_per_arm):
            detectable = None
            if min(n_c, n_t) >= 2:
                pooled = np.concatenate([c, t])
                try:
                    detectable = (
                        mde_proportions(float(pooled.mean()), min(n_c, n_t), alpha=alpha)
                        if binary else
                        mde_means(float(pooled.std(ddof=1)), min(n_c, n_t), alpha=alpha)
                    )
                except ValueError:
                    detectable = None
            findings.append(SegmentFinding(
                segment=label, n_control=n_c, n_treatment=n_t, result=None,
                powered=False, detectable_effect=detectable,
                notes=(f"below the floor of {min_per_arm} units per arm",),
            ))
            continue

        result = test(c, t, f"{metric_column}[{label}]")
        pooled = np.concatenate([c, t])
        try:
            detectable = (
                mde_proportions(float(pooled.mean()), min(n_c, n_t), alpha=alpha)
                if binary else
                mde_means(float(pooled.std(ddof=1)), min(n_c, n_t), alpha=alpha)
            )
        except ValueError:
            detectable = None
        powered = True
        notes: list[str] = []
        if target_effect is not None and detectable is not None:
            powered = bool(abs(detectable) <= abs(target_effect))
            if not powered:
                notes.append(
                    f"could only detect {detectable:.4g}, target was "
                    f"{target_effect:.4g}"
                )
        findings.append(SegmentFinding(
            segment=label, n_control=n_c, n_treatment=n_t, result=result,
            powered=powered, detectable_effect=detectable, notes=tuple(notes),
        ))
        tested.append((label, result))

    # FDR across the segments that were actually tested.
    if tested:
        adj = benjamini_hochberg(
            {name: res.p_value for name, res in tested}, alpha=alpha
        )
        q_by_name = dict(zip(adj.labels, adj.p_adjusted))
        findings = [
            SegmentFinding(
                segment=f.segment, n_control=f.n_control, n_treatment=f.n_treatment,
                result=f.result, powered=f.powered,
                detectable_effect=f.detectable_effect,
                q_value=q_by_name.get(f.segment), notes=f.notes,
            )
            for f in findings
        ]

    mix_p, mix_ok = mix_shift_check(
        frame, segment_column=segment_column, variant_column=variant_column
    )
    interaction_p = _omnibus_interaction([res for _, res in tested])

    warnings: list[str] = []
    if not mix_ok:
        warnings.append(
            f"segment mix differs across arms (p={mix_p:.3g}): either this "
            "segment is not pre-treatment, or assignment is broken. Fix that "
            "before reading any segment effect."
        )
    untested = [f.segment for f in findings if f.result is None]
    if untested:
        warnings.append(
            f"{len(untested)} segment(s) too small to test: "
            f"{', '.join(untested[:5])}"
        )
    if interaction_p is not None and interaction_p > 0.10 and any(
        f.q_value is not None and f.q_value <= alpha for f in findings
    ):
        warnings.append(
            f"a segment looks significant but the heterogeneity test does not "
            f"reject (p={interaction_p:.3g}): the spread across segments is "
            "consistent with one common effect measured at different precisions"
        )
    return SegmentScan(
        metric=metric_column, segment_column=segment_column,
        findings=tuple(findings), mix_shift_p=mix_p, mix_shift_ok=mix_ok,
        interaction_p=interaction_p, overall=overall, warnings=tuple(warnings),
    )
