#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample size, minimum detectable effect, and power — the step people skip.

An experiment with too few units is not a cheap experiment. It is an
expensive way to produce a result that cannot be acted on either way: a
null that means nothing and a positive that is probably noise. The numbers
in this module are the ones to compute BEFORE the experiment runs, and the
same numbers tell you afterwards what the null you just got was capable of
ruling out.

THREE QUANTITIES, ONE EQUATION
------------------------------
Sample size n, minimum detectable effect MDE, and power are three views of
one relationship. Fix any two and the third follows:

    MDE = (z_{1-alpha/2} + z_{power}) * se(n)

For two independent groups of equal size n each, comparing means with
standard deviation sigma:

    se(n) = sigma * sqrt(2/n)
    =>  n = 2 * sigma^2 * (z_{1-alpha/2} + z_{power})^2 / MDE^2

Read the shape of that, because it governs every planning conversation:
n scales with 1/MDE^2. Halving the effect you want to catch quadruples the
sample. An effect a tenth the size needs a hundred times the users. This is
why "we'll just run it and see" fails for small effects, and why variance
reduction (see cuped.py) is worth real effort — it lowers sigma, and sigma
enters squared.

WHY z AND NOT t
---------------
The planning formulas use normal quantiles even though the analysis will
use a t test. At the sample sizes where planning matters (hundreds up), the
difference is under a percent of the sample size, and the inputs — your
guess at sigma, your guess at the baseline rate — are wrong by far more
than that. Using t here would be precision theatre. `sample_size_means`
takes an `exact` flag that iterates with the t distribution anyway, for the
small-n case where it genuinely moves the answer.

ON "OBSERVED POWER"
-------------------
Computing power from the effect you just measured is circular and tells you
nothing: it is a deterministic function of your p-value. `power_at` here
takes the effect size you CARE about as an argument, never the one you
observed. That is the difference between "could this experiment have found
a 2% lift?" (answerable, useful) and "was this experiment well powered for
the effect it happened to see?" (meaningless).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .statlib import norm_cdf, norm_ppf, t_cdf, t_ppf

__all__ = [
    "PowerPlan", "sample_size_means", "sample_size_proportions",
    "mde_means", "mde_proportions", "power_at", "power_at_proportions",
]


@dataclass(frozen=True)
class PowerPlan:
    """The answer to a planning question, with its inputs attached.

    n_per_group     units needed in EACH arm
    total_n         both arms together
    mde             the effect the plan is sized for, in metric units
    mde_relative    the same effect as a share of the baseline
    power           probability of detecting `mde` if it is real
    alpha           two-sided false-positive rate
    assumptions     what had to be guessed, so a reader can challenge it
    """
    n_per_group: int
    total_n: int
    mde: float
    mde_relative: float
    power: float
    alpha: float
    assumptions: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = [
            f"{self.n_per_group:,} units per group ({self.total_n:,} total)",
            f"sized to detect {self.mde:.6g} ({self.mde_relative:+.2%} relative)",
            f"at {self.power:.0%} power, alpha {self.alpha:.3g} two-sided",
        ]
        if self.assumptions:
            lines.append("assumes:")
            lines += [f"  - {a}" for a in self.assumptions]
        return "\n".join(lines)


def _check_rates(alpha: float, power: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0,1), got {power}")
    if power <= alpha:
        raise ValueError(
            f"power={power} <= alpha={alpha}: a coin flip would do better; "
            "check the arguments are not swapped"
        )


def sample_size_means(
    sd: float,
    mde: float,
    *,
    baseline: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    exact: bool = False,
) -> PowerPlan:
    """Units per arm to detect an absolute difference `mde` in a mean.

    sd        standard deviation of the metric in the population, per unit.
              Take it from historical data, not from intuition; it is the
              single input most often wrong, and it enters squared.
    mde       the smallest absolute difference worth detecting. Choose it
              from what would change a decision, not from what you hope to
              see.
    baseline  the control mean, used only to express `mde` as a percentage.
    exact     iterate with the t distribution instead of the normal. Worth
              it below roughly n=50 per arm, pointless above it.
    """
    _check_rates(alpha, power)
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    if mde == 0:
        raise ValueError("mde of zero needs infinite sample; pick an effect size")

    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    n = 2.0 * (sd ** 2) * ((z_alpha + z_power) ** 2) / (mde ** 2)
    n_per = max(2, math.ceil(n))

    assumptions = [f"per-unit standard deviation = {sd:.6g}"]
    if exact:
        # The t-based critical value depends on df, which depends on n:
        # iterate to a fixed point. Converges in a handful of passes.
        for _ in range(100):
            df = 2.0 * (n_per - 1)
            t_alpha = t_ppf(1.0 - alpha / 2.0, df)
            t_power = t_ppf(power, df)
            n_new = max(2, math.ceil(2.0 * (sd ** 2) * ((t_alpha + t_power) ** 2) / (mde ** 2)))
            if n_new == n_per:
                break
            n_per = n_new
        assumptions.append("sized with the t distribution (exact=True)")
    else:
        assumptions.append("normal approximation for the critical values")

    rel = (mde / baseline) if baseline else float("nan")
    return PowerPlan(
        n_per_group=n_per, total_n=2 * n_per, mde=float(mde), mde_relative=rel,
        power=power, alpha=alpha, assumptions=tuple(assumptions),
    )


def sample_size_proportions(
    baseline_rate: float,
    mde_absolute: float | None = None,
    *,
    mde_relative: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerPlan:
    """Units per arm to detect a change in a conversion rate.

    Give the effect either way round:
        mde_absolute = 0.01       catch 4.0% -> 5.0%
        mde_relative = 0.10       catch 4.0% -> 4.4%

    For a 0/1 metric the variance is not free to choose: it is p(1-p). Both
    arms are allowed their own, so the standard error under the alternative
    uses p_c(1-p_c) + p_t(1-p_t) rather than twice one of them. That makes
    this slightly more conservative than the pooled textbook formula, and
    agrees with the unpooled interval that `two_proportion` reports — a
    plan sized with one variance assumption and analysed with another is
    how experiments end up underpowered on paper and overpowered in
    practice, or the reverse.

    A rate near 0.5 has the largest variance and needs the most traffic; a
    1% conversion rate is cheap per unit of variance and expensive in
    relative terms, because a 10% relative move on 1% is 0.1 points.
    """
    _check_rates(alpha, power)
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")
    if (mde_absolute is None) == (mde_relative is None):
        raise ValueError("give exactly one of mde_absolute or mde_relative")

    if mde_relative is not None:
        mde_absolute = baseline_rate * mde_relative
    assert mde_absolute is not None
    if mde_absolute == 0:
        raise ValueError("mde of zero needs infinite sample; pick an effect size")

    p_c = baseline_rate
    p_t = p_c + mde_absolute
    if not 0.0 < p_t < 1.0:
        raise ValueError(
            f"baseline {p_c:.4g} plus effect {mde_absolute:.4g} = {p_t:.4g}, "
            "which is not a rate"
        )

    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    var_sum = p_c * (1 - p_c) + p_t * (1 - p_t)
    n = var_sum * ((z_alpha + z_power) ** 2) / (mde_absolute ** 2)
    n_per = max(2, math.ceil(n))
    return PowerPlan(
        n_per_group=n_per, total_n=2 * n_per, mde=float(mde_absolute),
        mde_relative=mde_absolute / p_c, power=power, alpha=alpha,
        assumptions=(
            f"baseline rate = {p_c:.4%}",
            f"treatment rate under the alternative = {p_t:.4%}",
            "unpooled variance, matching two_proportion's interval",
        ),
    )


def mde_means(
    sd: float,
    n_per_group: int,
    *,
    baseline: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """The smallest absolute mean difference `n_per_group` units can detect.

    The question to ask when someone says "we only have 3,000 users": not
    "is that enough" but "enough for what". This returns the what.
    """
    _check_rates(alpha, power)
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be >= 2, got {n_per_group}")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)
    return float((z_alpha + z_power) * sd * math.sqrt(2.0 / n_per_group))


def mde_proportions(
    baseline_rate: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """The smallest absolute rate change `n_per_group` units can detect.

    Solved by bisection rather than algebra: the variance under the
    alternative depends on the effect being detected, so the relationship
    is implicit. Bisection on "required n at this effect" is exact to the
    tolerance and immune to the algebra slips that closed forms invite.
    """
    _check_rates(alpha, power)
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be >= 2, got {n_per_group}")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")

    lo, hi = 1e-12, min(baseline_rate, 1.0 - baseline_rate) * 0.999999
    if hi <= lo:
        raise ValueError("baseline rate too close to 0 or 1 for this helper")

    def needed(effect: float) -> float:
        return float(sample_size_proportions(
            baseline_rate, effect, alpha=alpha, power=power).n_per_group)

    if needed(hi) > n_per_group:
        # Even the largest representable effect needs more units than we have.
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if needed(mid) > n_per_group:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return float(hi)


def power_at(
    sd: float,
    n_per_group: int,
    effect: float,
    *,
    alpha: float = 0.05,
    exact: bool = False,
) -> float:
    """Probability of a significant result if the true mean difference is `effect`.

    THE DERIVATION
    Under the alternative, the test statistic is centred not on 0 but on
    effect/se. We reject when |statistic| exceeds the critical value, so

        power = P(Z > z_crit - effect/se) + P(Z < -z_crit - effect/se)

    The second term is the chance of a significant result in the WRONG
    direction. It is negligible for a well-powered test and is included
    anyway, because for a badly powered test it is not, and a "significant"
    result with the sign flipped is the worst outcome an experiment can
    produce.

    `exact=True` uses the t distribution's central form as an
    approximation to the noncentral t. That is a real approximation, not an
    exact calculation, and it is close enough that proof_power.py cannot
    distinguish it from simulation at n >= 20; below that, trust the
    simulation.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be >= 2, got {n_per_group}")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")

    se = sd * math.sqrt(2.0 / n_per_group)
    shift = effect / se
    if exact:
        df = 2.0 * (n_per_group - 1)
        crit = t_ppf(1.0 - alpha / 2.0, df)
        return float((1.0 - t_cdf(crit - shift, df)) + t_cdf(-crit - shift, df))
    crit = norm_ppf(1.0 - alpha / 2.0)
    return float((1.0 - norm_cdf(crit - shift)) + norm_cdf(-crit - shift))


def power_at_proportions(
    baseline_rate: float,
    n_per_group: int,
    effect_absolute: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Power for a rate change, with each arm's own variance."""
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be >= 2, got {n_per_group}")
    p_c = baseline_rate
    p_t = p_c + effect_absolute
    if not 0.0 <= p_t <= 1.0:
        raise ValueError(f"baseline + effect = {p_t:.4g}, which is not a rate")
    se = math.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / n_per_group)
    if se == 0:
        return 1.0 if effect_absolute != 0 else alpha
    shift = effect_absolute / se
    crit = norm_ppf(1.0 - alpha / 2.0)
    return float((1.0 - norm_cdf(crit - shift)) + norm_cdf(-crit - shift))
