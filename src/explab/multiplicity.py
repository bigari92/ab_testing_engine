#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing twenty things and believing all twenty.

THE PROBLEM
-----------
Run one test with alpha = 0.05 on a feature that does nothing, and there is
a 5% chance of a false positive. Run twenty, and the chance that at least
one comes up significant is 1 - 0.95^20 = 64%. Run a readout with four
primary metrics, six guardrails and twelve segments and you will find
something every single time. The finding is the procedure, not the feature.

TWO DIFFERENT GUARANTEES
------------------------
They answer different questions and are not interchangeable.

FWER, family-wise error rate (Bonferroni, Holm)
    P(at least one false positive among all tests) <= alpha.
    Strict. Use it when one false positive is costly on its own: you are
    about to ship on the strength of a single metric, or make a claim in
    public.

FDR, false discovery rate (Benjamini-Hochberg)
    E[share of your rejections that are false] <= alpha.
    Of the things you flagged, on average no more than alpha of them are
    noise. Use it when you are screening: twenty segments, fifty metrics,
    and the output is a shortlist to look into rather than a decision.

Bonferroni on fifty metrics requires p < 0.001 and will find nothing.
BH on the same fifty will hand you a shortlist in which roughly one in
twenty is junk. Neither is wrong; they are answers to different questions.

WHERE NOT TO CORRECT — THE ONE THAT MATTERS
-------------------------------------------
Do not apply a multiplicity correction to guardrail metrics.

A guardrail exists to catch harm: crash rate, latency, unsubscribes,
refund rate. Correcting it raises the bar for noticing damage, which is
backwards. The asymmetry is the point — a false alarm on a guardrail costs
an investigation, a missed signal costs users. If anything, guardrails
deserve a LOOSER alpha than the primary metric, and a readout that applies
Bonferroni across "all metrics" has quietly made itself blind to the
failure mode it was built to catch.

The rule this project follows, and that `analyze.py` implements:
  - one pre-declared primary metric: no correction, alpha as designed
  - secondary metrics: BH, reported as exploratory
  - segments: BH, reported as exploratory, never as a decision
  - guardrails: no correction, and flagged on the raw p-value
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["Adjusted", "bonferroni", "holm", "benjamini_hochberg"]


@dataclass(frozen=True)
class Adjusted:
    """The outcome of a correction.

    labels      names in the order given
    p_raw       the p-values as supplied
    p_adjusted  comparable to the ORIGINAL alpha after correction
    rejected    which nulls are rejected at `alpha`
    method      which correction ran
    alpha       the level the rejections were taken at
    guarantee   one line on what is actually controlled
    """
    labels: tuple[str, ...]
    p_raw: tuple[float, ...]
    p_adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    method: str
    alpha: float
    guarantee: str

    def significant(self) -> list[str]:
        return [lbl for lbl, rej in zip(self.labels, self.rejected) if rej]

    def __str__(self) -> str:
        lines = [
            f"{self.method} at alpha {self.alpha:g} - {self.guarantee}",
            f"{'metric':34s} {'p raw':>10s} {'p adj':>10s}  verdict",
        ]
        order = np.argsort(self.p_raw)
        for i in order:
            mark = "REJECT null" if self.rejected[i] else "-"
            lines.append(
                f"{self.labels[i][:34]:34s} {self.p_raw[i]:10.4g} "
                f"{self.p_adjusted[i]:10.4g}  {mark}"
            )
        return "\n".join(lines)


def _prepare(
    p_values: Sequence[float] | dict[str, float],
    labels: Sequence[str] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(p_values, dict):
        if labels is not None:
            raise ValueError("pass labels OR a dict, not both")
        labels = tuple(p_values)
        arr = np.asarray([p_values[k] for k in labels], dtype=float)
    else:
        arr = np.asarray(list(p_values), dtype=float)
        labels = tuple(labels) if labels is not None else tuple(
            f"test_{i + 1}" for i in range(arr.size)
        )
    if arr.size == 0:
        raise ValueError("no p-values given")
    if len(labels) != arr.size:
        raise ValueError(f"{len(labels)} labels for {arr.size} p-values")
    if not np.isfinite(arr).all():
        raise ValueError("p-values contain NaN or inf")
    if ((arr < 0) | (arr > 1)).any():
        bad = arr[(arr < 0) | (arr > 1)]
        raise ValueError(f"p-values outside [0,1]: {bad[:3]}")
    return labels, arr


def bonferroni(
    p_values: Sequence[float] | dict[str, float],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> Adjusted:
    """Multiply every p-value by the number of tests. Clip at 1.

    Controls FWER <= alpha for any dependence structure whatsoever, which
    is its virtue: it needs no assumption about how the tests relate. That
    generality is also why it is conservative. Correlated metrics (revenue
    per user and revenue per session move together) are effectively fewer
    than m independent tests, and Bonferroni still charges you for m.

    Equivalent to testing each hypothesis at alpha/m. Reported as adjusted
    p-values rather than a lowered alpha so that every method in this
    module is read against the same threshold.
    """
    lbl, p = _prepare(p_values, labels)
    m = p.size
    adj = np.minimum(p * m, 1.0)
    return Adjusted(
        labels=lbl, p_raw=tuple(p.tolist()), p_adjusted=tuple(adj.tolist()),
        rejected=tuple((adj <= alpha).tolist()), method="Bonferroni",
        alpha=alpha,
        guarantee=f"P(any false positive among {m}) <= {alpha:g}, any dependence",
    )


def holm(
    p_values: Sequence[float] | dict[str, float],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> Adjusted:
    """Holm-Bonferroni: same guarantee as Bonferroni, uniformly more power.

    Sort the p-values ascending. Test the smallest against alpha/m, the
    next against alpha/(m-1), and so on, stopping at the first failure.
    Everything after the stop is not rejected, which is the step-down rule
    and the reason the adjusted values must be made monotone with a running
    maximum.

    There is no reason to prefer plain Bonferroni over Holm: Holm rejects
    everything Bonferroni does and sometimes more, with the same FWER
    control. Bonferroni survives because it can be done in your head.
    """
    lbl, p = _prepare(p_values, labels)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)  # enforce monotonicity
        adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return Adjusted(
        labels=lbl, p_raw=tuple(p.tolist()), p_adjusted=tuple(adj.tolist()),
        rejected=tuple((adj <= alpha).tolist()), method="Holm-Bonferroni",
        alpha=alpha,
        guarantee=f"P(any false positive among {m}) <= {alpha:g}, any dependence",
    )


def benjamini_hochberg(
    p_values: Sequence[float] | dict[str, float],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> Adjusted:
    """Benjamini-Hochberg: control the share of your findings that are false.

    Sort ascending. Find the largest rank k where p_(k) <= (k/m) * alpha,
    and reject everything up to it — including any larger p-value below
    that cut, which is the step-UP rule and is why the adjusted values are
    made monotone with a running minimum from the top.

    Adjusted p-value (the "q-value"): min over ranks j >= k of
    (m/j) * p_(j), clipped at 1.

    THE GUARANTEE, PRECISELY
    E[false discoveries / total discoveries] <= alpha, where the ratio is
    taken as 0 when nothing is discovered. Note the expectation: on any
    SINGLE readout the realised share can be higher. BH is a statement
    about your practice over many readouts, not a warranty on this one.

    Valid for independent tests, and also for positively dependent ones,
    which covers the usual case of correlated product metrics. Under
    arbitrary dependence the guarantee needs the Benjamini-Yekutieli
    correction, which multiplies alpha by 1/sum(1/i) and costs most of the
    power back. In practice: use BH, say it is exploratory, and confirm
    anything load-bearing in a follow-up experiment.
    """
    lbl, p = _prepare(p_values, labels)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = p_sorted * m / ranks
    # Step-up: make monotone by sweeping from the largest p downwards.
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return Adjusted(
        labels=lbl, p_raw=tuple(p.tolist()), p_adjusted=tuple(q.tolist()),
        rejected=tuple((q <= alpha).tolist()), method="Benjamini-Hochberg",
        alpha=alpha,
        guarantee=(
            f"E[false share of rejections] <= {alpha:g} "
            "(independent or positively dependent tests)"
        ),
    )
