#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The four comparisons an experiment readout is allowed to make.

    welch_t         two means, variances not assumed equal
    two_proportion  two rates, 0/1 outcomes
    ratio_delta     a ratio of sums, randomised at a coarser unit
    bootstrap_diff  anything, when you do not trust the formula

Each returns the same `TestResult` so the rest of the project never has to
know which one ran. Each reports a confidence interval, not only a p-value,
because "no significant difference" and "we could not have detected a
difference this small" are different sentences and only the interval tells
them apart.

A NOTE ON WHAT IS BEING ESTIMATED
---------------------------------
Every function here estimates the same thing: the difference between the
treatment group's metric and the control group's, plus a range of values
the data cannot rule out. `relative_lift` divides that difference by the
control value, which is how the result will be reported, and which is also
the number people quietly misread: a 2-point lift on a 4% base is a 50%
relative lift, and a 2-point lift on a 40% base is a 5% one. Both appear in
the readout so neither can hide.

ONE-SIDED TESTS ARE NOT OFFERED
-------------------------------
Deliberately. A one-sided test is defensible only if you committed to the
direction before seeing the data, and in practice nobody can prove they
did. Two-sided keeps the arithmetic honest at the cost of sample size, and
the cost is 1.96/1.645 squared, about 42% more units — which is a price
worth paying for a number you can defend in a room.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .statlib import norm_cdf, norm_ppf, norm_sf, t_ppf, t_sf

__all__ = [
    "TestResult", "welch_t", "two_proportion", "two_proportion_from_arrays",
    "ratio_delta", "bootstrap_diff",
]


@dataclass(frozen=True)
class TestResult:
    """One comparison, with everything needed to argue about it.

    metric            what was measured
    method            which function produced this
    n_control         units in control
    n_treatment       units in treatment
    mean_control      control's value of the metric
    mean_treatment    treatment's value of the metric
    estimate          mean_treatment - mean_control (the absolute effect)
    se                standard error of `estimate`
    ci                (low, high) for `estimate` at `confidence`
    p_value           two-sided, against the null of no difference
    df                degrees of freedom used, None for z-based methods
    confidence        the level `ci` was computed at, e.g. 0.95
    notes             anything the method wants the reader to know
    """
    metric: str
    method: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    df: float | None
    confidence: float = 0.95
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def relative_lift(self) -> float:
        """estimate / mean_control. NaN when control is zero."""
        if self.mean_control == 0:
            return float("nan")
        return self.estimate / self.mean_control

    @property
    def relative_ci(self) -> tuple[float, float]:
        """The interval expressed as relative lift.

        Dividing both ends by the control mean is an approximation: it
        treats the control mean as known rather than estimated, so the real
        interval on a relative lift is slightly wider than this. It is the
        convention every experimentation tool uses, and at the sample sizes
        where you would ship a decision the difference is in the third
        decimal. Stated so that nobody has to wonder.
        """
        if self.mean_control == 0:
            return (float("nan"), float("nan"))
        return (self.ci[0] / self.mean_control, self.ci[1] / self.mean_control)

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero at `confidence`."""
        return not (self.ci[0] <= 0.0 <= self.ci[1])

    def __str__(self) -> str:
        lo, hi = self.ci
        rlo, rhi = self.relative_ci
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.metric}: control {self.mean_control:.6g} -> "
            f"treatment {self.mean_treatment:.6g}\n"
            f"  absolute {self.estimate:+.6g}  "
            f"[{lo:+.6g}, {hi:+.6g}] at {self.confidence:.0%}\n"
            f"  relative {self.relative_lift:+.2%}  [{rlo:+.2%}, {rhi:+.2%}]\n"
            f"  p = {self.p_value:.4g}  ({verdict}, {self.method}, "
            f"n = {self.n_control} vs {self.n_treatment})"
        )


# The smallest p-value this project will print. Below it the honest
# statement is "smaller than any number this arithmetic can carry", not a
# figure. 1e-300 is just inside double precision's normal range.
P_FLOOR = 1e-300


def _two_sided_z(z: float) -> tuple[float, str | None]:
    """Two-sided p from a z statistic, via the survival function.

    2 * norm_sf(|z|) rather than 2 * (1 - norm_cdf(|z|)): the second form
    gives exactly 0.0 for |z| beyond about 8.3, because norm_cdf has
    already rounded to 1.0. See statlib.norm_sf.
    """
    p = 2.0 * norm_sf(abs(z))
    if p < P_FLOOR:
        return P_FLOOR, (
            f"p-value underflowed (|z| = {abs(z):.1f}); reported at the "
            f"floor of {P_FLOOR:g}. The effect is far outside sampling "
            "noise, and no finite dataset justifies a smaller figure."
        )
    return min(1.0, p), None


def _two_sided_t(t_stat: float, df: float) -> tuple[float, str | None]:
    """Two-sided p from a t statistic, with the same underflow guard."""
    p = 2.0 * t_sf(abs(t_stat), df)
    if p < P_FLOOR:
        return P_FLOOR, (
            f"p-value underflowed (|t| = {abs(t_stat):.1f} on {df:.0f} df); "
            f"reported at the floor of {P_FLOOR:g}."
        )
    return min(1.0, p), None


def _as_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf; clean or drop before testing")
    return arr


# --------------------------------------------------------------------------
# means
# --------------------------------------------------------------------------
def welch_t(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    *,
    metric: str = "mean",
    confidence: float = 0.95,
) -> TestResult:
    """Compare two means without assuming the groups have equal variance.

    THE ARITHMETIC

        se = sqrt(var_c/n_c + var_t/n_t)          independent groups, so
                                                  variances add
        t  = (mean_t - mean_c) / se
        df = se^4 / (  (var_c/n_c)^2/(n_c-1)
                     + (var_t/n_t)^2/(n_t-1) )     Welch-Satterthwaite

    `var` is the sample variance with the n-1 divisor (ddof=1): dividing by
    n would understate the spread, because the sample mean is already fitted
    to this sample and sits closer to the data than the true mean does.

    WHY WELCH AND NOT THE POOLED (STUDENT) T-TEST
    The pooled test assumes one common variance and borrows strength across
    groups to estimate it. In an experiment that assumption is exactly what
    the treatment may have broken: a feature that helps heavy users and does
    nothing for light users changes the variance as well as the mean. Welch
    costs a little power when the variances really are equal, and does not
    quietly inflate the false-positive rate when they are not. The default
    in a production experiment platform should be Welch, and this project
    has a simulation that shows what the pooled test does when variances
    differ by 4x: see proofs/proof_welch_vs_pooled.py.

    DEGREES OF FREEDOM
    Welch's df is generally fractional. That is not a bug; it is an
    interpolation between the t distributions that bracket the true
    sampling distribution.
    """
    c = _as_float_array(control, "control")
    t_arr = _as_float_array(treatment, "treatment")
    if c.size < 2 or t_arr.size < 2:
        raise ValueError("welch_t needs at least 2 observations per group")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")

    n_c, n_t = c.size, t_arr.size
    mean_c, mean_t = float(c.mean()), float(t_arr.mean())
    var_c, var_t = float(c.var(ddof=1)), float(t_arr.var(ddof=1))
    term_c, term_t = var_c / n_c, var_t / n_t
    se = float(np.sqrt(term_c + term_t))

    notes: list[str] = []
    if se == 0.0:
        # Both groups constant. No variation means no inference to do.
        return TestResult(
            metric=metric, method="welch_t", n_control=n_c, n_treatment=n_t,
            mean_control=mean_c, mean_treatment=mean_t,
            estimate=mean_t - mean_c, se=0.0,
            ci=(mean_t - mean_c, mean_t - mean_c),
            p_value=1.0 if mean_t == mean_c else 0.0, df=None,
            confidence=confidence,
            notes=("zero variance in both groups; no sampling error to report",),
        )

    denom = (term_c ** 2) / (n_c - 1) + (term_t ** 2) / (n_t - 1)
    df = (se ** 4) / denom
    estimate = mean_t - mean_c
    t_stat = estimate / se
    # Two-sided: the probability of a statistic at least this extreme in
    # either direction. t_cdf(-|t|) is one tail; double it.
    p_value, p_note = _two_sided_t(t_stat, df)
    notes += [p_note] if p_note else []
    crit = t_ppf(0.5 + confidence / 2.0, df)
    ci = (estimate - crit * se, estimate + crit * se)

    ratio = (var_c / var_t) if var_t > 0 else float("inf")
    if ratio and (ratio > 4.0 or ratio < 0.25):
        notes.append(
            f"variances differ {max(ratio, 1/ratio):.1f}x between groups; "
            "Welch is handling it, but check for outliers or a split effect"
        )
    return TestResult(
        metric=metric, method="welch_t", n_control=n_c, n_treatment=n_t,
        mean_control=mean_c, mean_treatment=mean_t, estimate=estimate,
        se=se, ci=ci, p_value=min(1.0, p_value), df=df,
        confidence=confidence, notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------
def two_proportion(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    *,
    metric: str = "rate",
    confidence: float = 0.95,
) -> TestResult:
    """Compare two rates from 0/1 outcomes.

    THE DELIBERATE INCONSISTENCY
    The p-value uses a POOLED standard error and the interval uses an
    UNPOOLED one. That looks sloppy and is correct, and the reason is worth
    holding on to:

      - The p-value is computed under the null hypothesis. If the null is
        true both groups share one rate, so the best estimate of that rate
        uses every observation: p_pool = (s_c + s_t) / (n_c + n_t). Using
        separate rates to test the hypothesis that the rates are equal
        throws away information the null gives you for free.

      - The interval is not computed under the null. It has to cover the
        true difference whatever it is, including the case where the rates
        genuinely differ, so it must let each group have its own variance.

    Mixing them means a p-value of 0.049 can sit beside an interval whose
    endpoint is a hair past zero. proofs/proof_ci_coverage.py measures both
    and shows each doing its own job: the pooled test holds the false
    positive rate at alpha, the unpooled interval holds coverage at
    1 - alpha. The textbook alternative is to use the unpooled SE for both,
    which keeps them consistent with each other and makes the test slightly
    worse at the only thing a test is for.

    SMALL SAMPLES
    This is the normal approximation to the binomial. It gets shaky when a
    cell is small; the usual guard is at least 10 successes AND 10 failures
    in each group, which is checked and reported in `notes` rather than
    raised, because a warning you can read beats an exception you work
    around.
    """
    for name, value in (("n_control", n_control), ("n_treatment", n_treatment)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, s, n in (("control", successes_control, n_control),
                       ("treatment", successes_treatment, n_treatment)):
        if not 0 <= s <= n:
            raise ValueError(f"successes_{name}={s} outside 0..{n}")

    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    estimate = p_t - p_c

    p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)
    se_pooled = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment)))
    se_unpooled = float(np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment))

    notes: list[str] = []
    if se_pooled == 0.0:
        p_value = 1.0
        notes.append("no successes in either group, or no failures; test is vacuous")
    else:
        z = estimate / se_pooled
        p_value, p_note = _two_sided_z(z)
        if p_note:
            notes.append(p_note)

    crit = norm_ppf(0.5 + confidence / 2.0)
    ci = (estimate - crit * se_unpooled, estimate + crit * se_unpooled)

    cells = [successes_control, n_control - successes_control,
             successes_treatment, n_treatment - successes_treatment]
    if min(cells) < 10:
        notes.append(
            f"smallest cell is {min(cells)} (<10): the normal approximation is "
            "strained, prefer bootstrap_diff or an exact test"
        )
    return TestResult(
        metric=metric, method="two_proportion", n_control=n_control,
        n_treatment=n_treatment, mean_control=p_c, mean_treatment=p_t,
        estimate=estimate, se=se_unpooled, ci=ci, p_value=min(1.0, p_value),
        df=None, confidence=confidence, notes=tuple(notes),
    )


def two_proportion_from_arrays(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    *,
    metric: str = "rate",
    confidence: float = 0.95,
) -> TestResult:
    """two_proportion for 0/1 arrays instead of counts."""
    c = _as_float_array(control, "control")
    t_arr = _as_float_array(treatment, "treatment")
    for name, arr in (("control", c), ("treatment", t_arr)):
        bad = arr[(arr != 0) & (arr != 1)]
        if bad.size:
            raise ValueError(
                f"{name} must be 0/1 for a proportion test; found {bad[:3]}"
            )
    return two_proportion(
        int(c.sum()), c.size, int(t_arr.sum()), t_arr.size,
        metric=metric, confidence=confidence,
    )


# --------------------------------------------------------------------------
# ratio metrics
# --------------------------------------------------------------------------
def ratio_delta(
    num_control: Sequence[float] | np.ndarray,
    den_control: Sequence[float] | np.ndarray,
    num_treatment: Sequence[float] | np.ndarray,
    den_treatment: Sequence[float] | np.ndarray,
    *,
    metric: str = "ratio",
    confidence: float = 0.95,
) -> TestResult:
    """Compare sum(numerator)/sum(denominator) between groups, correctly.

    THE PROBLEM THIS SOLVES
    Randomise users; measure clicks per impression. The metric is a ratio of
    two sums whose unit — the impression — is finer than the unit that was
    randomised. Two wrong things get done with it:

      1. The common and serious one: analyse at the DENOMINATOR's level.
         Pool every impression, call the metric a proportion, and compute
         sqrt(p(1-p)/n_impressions). That treats 27,000 impressions from
         3,000 users as 27,000 independent observations. They are not:
         impressions from one user share that user's behaviour, so the real
         sampling variance is larger — by a factor that grows with how much
         users differ from one another. Coverage falls below nominal,
         p-values come out optimistic, and nothing in the output hints at
         it. How bad it gets depends entirely on heterogeneity: with every
         user clicking at the same underlying rate the understatement is
         small, and with realistic user-to-user variation in rates it is
         enough to turn a null into a "win". proofs/proof_delta_method.py
         measures both cases.

      2. The subtle one: average the per-user ratios, mean(x_i / y_i).
         This is not wrong arithmetic, it is a different question — the
         average of individual users' rates, in which a user with one
         impression counts as much as a user with a thousand. Sometimes
         that is what you want. It is almost never what the metric
         definition said.

    This function estimates the ratio of sums, at the unit that was
    randomised, with the variance that unit implies.

    THE FIX: THE DELTA METHOD
    Write the metric as R = X_bar / Y_bar where X_bar is the mean numerator
    per unit and Y_bar the mean denominator per unit. A first-order Taylor
    expansion of R around (E X, E Y) gives

        Var(R) ~= (1/Y_bar^2) * ( Var(X)        - 2R*Cov(X,Y) + R^2*Var(Y) )
                              / n

    The covariance term is the one the naive calculation drops, and it is
    the one that matters: it is large and positive whenever heavy users are
    heavy in both columns, which is almost always.

    Each group gets its own variance; the two are then added, as independent
    groups always are. proofs/proof_delta_method.py measures the coverage of
    this interval against the naive one on data where the truth is known.

    INPUTS
    One row per randomisation unit, in matching order: a user's numerator
    and that same user's denominator. Units with a zero denominator are
    kept, because dropping them changes the estimand from "clicks per
    impression across all users" to "clicks per impression among users who
    had an impression", which is a different question with a different
    answer.
    """
    xc = _as_float_array(num_control, "num_control")
    yc = _as_float_array(den_control, "den_control")
    xt = _as_float_array(num_treatment, "num_treatment")
    yt = _as_float_array(den_treatment, "den_treatment")
    if xc.size != yc.size:
        raise ValueError(f"control numerator/denominator lengths differ: {xc.size} vs {yc.size}")
    if xt.size != yt.size:
        raise ValueError(f"treatment numerator/denominator lengths differ: {xt.size} vs {yt.size}")
    if xc.size < 2 or xt.size < 2:
        raise ValueError("ratio_delta needs at least 2 units per group")

    def group(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
        n = x.size
        x_bar, y_bar = float(x.mean()), float(y.mean())
        if y_bar == 0.0:
            raise ValueError("denominator sums to zero in one group; ratio undefined")
        ratio = x_bar / y_bar
        cov = np.cov(np.vstack([x, y]), ddof=1)
        var_x, var_y, cov_xy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
        var_ratio = (var_x - 2.0 * ratio * cov_xy + ratio * ratio * var_y) / (n * y_bar * y_bar)
        return ratio, max(var_ratio, 0.0), n

    r_c, v_c, n_c = group(xc, yc)
    r_t, v_t, n_t = group(xt, yt)
    estimate = r_t - r_c
    se = float(np.sqrt(v_c + v_t))

    notes = [
        "delta-method standard error: includes the numerator/denominator "
        "covariance that a per-row mean would drop",
    ]
    if se == 0.0:
        p_value, ci = 1.0, (estimate, estimate)
    else:
        z = estimate / se
        p_value, p_note = _two_sided_z(z)
        if p_note:
            notes.append(p_note)
        crit = norm_ppf(0.5 + confidence / 2.0)
        ci = (estimate - crit * se, estimate + crit * se)
    zero_den = int((yc == 0).sum() + (yt == 0).sum())
    if zero_den:
        notes.append(
            f"{zero_den} units have a zero denominator and were kept; "
            "dropping them would silently change the question"
        )
    return TestResult(
        metric=metric, method="ratio_delta", n_control=n_c, n_treatment=n_t,
        mean_control=r_c, mean_treatment=r_t, estimate=estimate, se=se,
        ci=ci, p_value=min(1.0, p_value), df=None, confidence=confidence,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def bootstrap_diff(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    *,
    metric: str = "statistic",
    statistic: Literal["mean", "median"] | str = "mean",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> TestResult:
    """Difference in a statistic, with the interval resampled rather than derived.

    WHEN TO REACH FOR THIS
    When the formula does not exist or you do not trust it: a median, a
    trimmed mean, a 90th percentile, revenue so heavy-tailed that the
    central limit theorem has not arrived yet at your sample size. The
    bootstrap buys generality by spending CPU.

    HOW IT WORKS
    Resample each group WITH REPLACEMENT to its own original size, compute
    the statistic on each resample, difference them, and repeat `n_boot`
    times. The spread of those differences estimates the sampling
    distribution of the real difference. The interval reported is the
    percentile interval: the 2.5th and 97.5th percentiles of the resampled
    differences for a 95% level.

    Resampling to the ORIGINAL size matters. The width of a sampling
    distribution depends on n, so a resample of a different size answers a
    different question.

    THE P-VALUE HERE
    Inverted from the interval by bisection on the confidence level: the
    p-value is one minus the narrowest level whose interval still excludes
    zero. Equivalent in spirit to the usual "share of resamples on the
    other side of zero, doubled", and better behaved when the whole
    bootstrap distribution lies on one side, where that share is 0 and the
    naive p-value collapses to exactly 0 — a claim no 10,000-draw
    simulation can support. Here the floor is 1/n_boot and is labelled.

    LIMITS, STATED
    The percentile bootstrap is not magic. It inherits whatever bias the
    statistic has, it needs the sample to be large enough to contain the
    shape of the distribution, and for a mean it is very slightly
    anti-conservative in small samples. For a mean with n in the hundreds,
    welch_t is both faster and better; use the bootstrap where the formula
    is missing.
    """
    c = _as_float_array(control, "control")
    t_arr = _as_float_array(treatment, "treatment")
    if n_boot < 1000:
        raise ValueError(f"n_boot={n_boot} is too few for a stable interval; use >= 1000")

    if statistic == "mean":
        fn = np.mean
    elif statistic == "median":
        fn = np.median
    else:
        raise ValueError(f"unknown statistic {statistic!r}; use 'mean' or 'median'")

    rng = np.random.default_rng(seed)
    n_c, n_t = c.size, t_arr.size
    # Draw all resample indices at once: (n_boot, n) integer matrices.
    idx_c = rng.integers(0, n_c, size=(n_boot, n_c))
    idx_t = rng.integers(0, n_t, size=(n_boot, n_t))
    boot = fn(t_arr[idx_t], axis=1) - fn(c[idx_c], axis=1)

    observed = float(fn(t_arr) - fn(c))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Invert the interval for a p-value: find the smallest level at which
    # the interval still excludes zero.
    share_below = float((boot <= 0.0).mean())
    share_above = float((boot >= 0.0).mean())
    p_value = min(1.0, 2.0 * min(share_below, share_above))
    notes = [f"percentile bootstrap, {n_boot} resamples, statistic={statistic}"]
    floor = 1.0 / n_boot
    if p_value < floor:
        p_value = floor
        notes.append(f"p-value floored at 1/n_boot = {floor:g}; no resample crossed zero")
    return TestResult(
        metric=metric, method=f"bootstrap_{statistic}", n_control=n_c,
        n_treatment=n_t, mean_control=float(fn(c)), mean_treatment=float(fn(t_arr)),
        estimate=observed, se=float(boot.std(ddof=1)), ci=(float(lo), float(hi)),
        p_value=p_value, df=None, confidence=confidence, notes=tuple(notes),
    )
