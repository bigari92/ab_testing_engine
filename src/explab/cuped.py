#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUPED: make the same experiment answer a smaller question.

THE IDEA IN ONE PARAGRAPH
-------------------------
Most of the variance in a per-user metric is not caused by your feature. It
is caused by users differing from each other: some buy every week, some
never. If you know how much each user spent BEFORE the experiment started,
you can subtract the part of their post-period spend that was predictable
from that, and test the remainder. The treatment effect survives the
subtraction untouched, because pre-period behaviour cannot have been
affected by a treatment that had not happened yet. The noise does not
survive. Same users, same effect, narrower interval.

The name is from Deng, Xu, Kohavi and Walker (2013), "Improving the
Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment
Data": Controlled-experiment Using Pre-Existing Data.

THE ADJUSTMENT

    theta = Cov(Y, X) / Var(X)
    Y_adj = Y - theta * (X - mean(X))

where Y is the outcome and X the pre-period covariate, and mean(X) is taken
over EVERY unit in the experiment, both arms pooled.

    Var(Y_adj) = Var(Y) * (1 - rho^2)

rho is the correlation between X and Y. That is the whole payoff and it is
worth memorising: variance falls by the square of the correlation. A
covariate correlated 0.5 with the outcome removes 25% of the variance; one
correlated 0.9 removes 81%. Since required sample size is proportional to
variance, that 0.9 covariate turns a five-week experiment into a one-week
experiment.

THREE WAYS TO GET THIS WRONG
----------------------------
1. A covariate that is not pre-treatment. If X is measured during or after
   the experiment, the treatment can move it, the subtraction removes part
   of the real effect, and the estimate is biased toward zero. This is not
   a small effect and it is invisible in the output — the interval gets
   narrower and the point estimate quietly wrong. `cuped_adjust` cannot
   check this for you; it is a property of your data pipeline, and it is
   why `pre_` is a naming convention worth enforcing.

2. theta estimated per arm. Fitting a separate theta to control and
   treatment lets the adjustment absorb some of the treatment effect.
   Pool the arms. `cuped_adjust` does, and refuses to do otherwise.

3. Forgetting to centre, or centring per arm. Y - theta*X (no centring)
   changes the level of the metric, so the adjusted means are no longer on
   the metric's own scale and "control converted at 4.1%" stops being
   true. Centring on the POOLED mean keeps both adjusted means
   interpretable and keeps the difference unbiased.

WHAT IT DOES NOT DO
-------------------
CUPED does not fix bias, confounding, or a broken randomisation. It is a
variance reduction technique for an experiment that is already valid. If
SRM fails, CUPED makes a wrong answer more precise.

It also does nothing for users with no history. New users have no
pre-period, X is zero for all of them, and the correlation across a
population that is mostly new is near zero. On a signup funnel CUPED is
close to useless; on a retention or revenue metric over an established
base it is the highest-leverage change available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["CupedResult", "cuped_adjust", "variance_reduction"]


@dataclass(frozen=True)
class CupedResult:
    """Adjusted outcomes plus the diagnostics needed to trust them.

    adjusted_control    Y_adj for the control units, original order
    adjusted_treatment  Y_adj for the treatment units, original order
    theta               the coefficient that was subtracted
    correlation         rho between covariate and outcome, pooled
    variance_before     pooled variance of Y
    variance_after      pooled variance of Y_adj
    reduction           1 - variance_after/variance_before, measured
    predicted_reduction rho^2, what theory says it should be
    effective_n_gain    how much more sample the raw metric would need to
                        match this precision, as a multiplier
    warnings            anything a reader should be told
    """
    adjusted_control: np.ndarray
    adjusted_treatment: np.ndarray
    theta: float
    correlation: float
    variance_before: float
    variance_after: float
    reduction: float
    predicted_reduction: float
    effective_n_gain: float
    warnings: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = [
            f"CUPED: theta = {self.theta:.6g}, rho = {self.correlation:+.4f}",
            f"  variance {self.variance_before:.6g} -> {self.variance_after:.6g}"
            f"  ({self.reduction:.1%} removed, rho^2 predicts "
            f"{self.predicted_reduction:.1%})",
            f"  equivalent to {self.effective_n_gain:.2f}x the sample size",
        ]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _clean(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def variance_reduction(correlation: float) -> float:
    """rho -> share of variance CUPED will remove. The whole theory, one line."""
    return float(correlation ** 2)


def cuped_adjust(
    outcome_control: Sequence[float] | np.ndarray,
    covariate_control: Sequence[float] | np.ndarray,
    outcome_treatment: Sequence[float] | np.ndarray,
    covariate_treatment: Sequence[float] | np.ndarray,
) -> CupedResult:
    """Apply CUPED and report what it bought.

    Pass the pre-experiment covariate for each unit alongside that unit's
    outcome, in matching order, per arm. The function pools the arms to
    estimate theta and the centring constant, then returns the adjusted
    outcomes for you to hand to `welch_t` exactly as you would have handed
    it the raw ones.

    The adjusted values are NOT a different metric. They estimate the same
    difference of means; only the noise is smaller. That is why it is safe
    to report "revenue per user +$0.42" from an adjusted test — but not
    safe to report the adjusted LEVEL of revenue per user as if it were
    the actual level, which is why centring on the pooled mean is done:
    it keeps the levels honest too.
    """
    y_c = _clean(outcome_control, "outcome_control")
    x_c = _clean(covariate_control, "covariate_control")
    y_t = _clean(outcome_treatment, "outcome_treatment")
    x_t = _clean(covariate_treatment, "covariate_treatment")
    if y_c.size != x_c.size:
        raise ValueError(
            f"control outcome/covariate lengths differ: {y_c.size} vs {x_c.size}"
        )
    if y_t.size != x_t.size:
        raise ValueError(
            f"treatment outcome/covariate lengths differ: {y_t.size} vs {x_t.size}"
        )
    if y_c.size < 2 or y_t.size < 2:
        raise ValueError("need at least 2 units per arm")

    # Pool both arms. The covariate is pre-treatment, so pooling cannot
    # leak the treatment effect into theta, and pooling gives the most
    # precise estimate of it.
    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])

    var_x = float(x_all.var(ddof=1))
    var_y = float(y_all.var(ddof=1))
    warnings: list[str] = []

    if var_x == 0.0:
        # A constant covariate carries no information. Return the outcomes
        # untouched rather than dividing by zero.
        return CupedResult(
            adjusted_control=y_c.copy(), adjusted_treatment=y_t.copy(),
            theta=0.0, correlation=0.0, variance_before=var_y,
            variance_after=var_y, reduction=0.0, predicted_reduction=0.0,
            effective_n_gain=1.0,
            warnings=("covariate has zero variance; no adjustment possible",),
        )

    cov_xy = float(np.cov(np.vstack([y_all, x_all]), ddof=1)[0, 1])
    theta = cov_xy / var_x
    rho = cov_xy / float(np.sqrt(var_x * var_y)) if var_y > 0 else 0.0

    x_mean = float(x_all.mean())
    adj_c = y_c - theta * (x_c - x_mean)
    adj_t = y_t - theta * (x_t - x_mean)

    var_after = float(np.concatenate([adj_c, adj_t]).var(ddof=1))
    reduction = 1.0 - (var_after / var_y) if var_y > 0 else 0.0
    gain = (var_y / var_after) if var_after > 0 else float("inf")

    if abs(rho) < 0.1:
        warnings.append(
            f"correlation is only {rho:+.3f}; CUPED will remove about "
            f"{rho**2:.1%} of the variance, which is not worth the "
            "complexity it adds to the readout"
        )
    if reduction < 0:
        warnings.append(
            "variance went UP: this happens when the covariate is noise and "
            "the sample is small. Drop the adjustment."
        )
    # A correlation this high between a pre-period and a post-period metric
    # usually means the covariate is not actually pre-period.
    if abs(rho) > 0.95:
        warnings.append(
            f"correlation {rho:+.3f} is suspiciously high: check that the "
            "covariate really is measured before the experiment started and "
            "is not the outcome under another name"
        )
    return CupedResult(
        adjusted_control=adj_c, adjusted_treatment=adj_t, theta=float(theta),
        correlation=float(rho), variance_before=var_y, variance_after=var_after,
        reduction=float(reduction), predicted_reduction=float(rho ** 2),
        effective_n_gain=float(gain), warnings=tuple(warnings),
    )
