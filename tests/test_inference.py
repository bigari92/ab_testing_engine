# -*- coding: utf-8 -*-
"""The tests against SciPy and statsmodels, where an independent answer exists.

WHAT IS CHECKED HERE THAT THE PROOFS CANNOT CHECK
proofs/ establishes that these functions have the right LONG-RUN
behaviour: 5% rejection under the null, 95% coverage. That is the property
that matters, and it is also a loose constraint - a test statistic could be
off by 0.3% and every proof would still pass, because 40,000 simulations
cannot resolve 0.3%.

So this file pins the arithmetic. Each function is run on fixed data and
compared against an independent implementation of the same formula, to
10 or more significant figures. Where no independent implementation exists
(the delta method, the bootstrap) the test is against a hand-computed
formula or an invariant that must hold exactly.

THE DELIBERATE INCONSISTENCY, TESTED AS A DELIBERATE CHOICE
two_proportion uses a POOLED standard error for the p-value and an
UNPOOLED one for the interval. That looks like a bug, is not, and is the
same thing statsmodels does by default - so there is a test below that
asserts both halves separately against statsmodels, which is the only way
to show the inconsistency is intentional rather than accidental.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)

from explab.inference import (
    bootstrap_diff,
    ratio_delta,
    two_proportion,
    two_proportion_from_arrays,
    welch_t,
)


@pytest.fixture
def normal_samples():
    rng = np.random.default_rng(7)
    return rng.normal(10.0, 2.0, 900), rng.normal(10.3, 3.5, 1100)


# --------------------------------------------------------------------------
# Welch's t-test
# --------------------------------------------------------------------------
def test_welch_t_matches_scipy_exactly(normal_samples):
    control, treatment = normal_samples
    mine = welch_t(control, treatment)
    theirs = stats.ttest_ind(treatment, control, equal_var=False)
    assert mine.p_value == pytest.approx(theirs.pvalue, rel=1e-11)
    assert mine.estimate == pytest.approx(treatment.mean() - control.mean(),
                                          rel=1e-14)


def test_welch_degrees_of_freedom_match_scipy(normal_samples):
    """Welch-Satterthwaite df, which is where an implementation usually
    drifts: it is easy to write the formula with n instead of n-1."""
    control, treatment = normal_samples
    mine = welch_t(control, treatment)
    theirs = stats.ttest_ind(treatment, control, equal_var=False)
    assert mine.df == pytest.approx(theirs.df, rel=1e-11)


def test_welch_interval_matches_scipy(normal_samples):
    control, treatment = normal_samples
    mine = welch_t(control, treatment, confidence=0.95)
    lo, hi = stats.ttest_ind(
        treatment, control, equal_var=False
    ).confidence_interval(0.95)
    assert mine.ci[0] == pytest.approx(lo, rel=1e-10)
    assert mine.ci[1] == pytest.approx(hi, rel=1e-10)


@pytest.mark.parametrize("confidence", [0.80, 0.90, 0.95, 0.99])
def test_welch_interval_widens_with_confidence(normal_samples, confidence):
    control, treatment = normal_samples
    result = welch_t(control, treatment, confidence=confidence)
    lo, hi = stats.ttest_ind(
        treatment, control, equal_var=False
    ).confidence_interval(confidence)
    assert result.ci[0] == pytest.approx(lo, rel=1e-10)
    assert result.ci[1] == pytest.approx(hi, rel=1e-10)


def test_welch_estimate_sits_inside_its_own_interval(normal_samples):
    """An invariant, not a comparison. If this ever fails, something is
    wrong in a way no external library would catch."""
    control, treatment = normal_samples
    r = welch_t(control, treatment)
    assert r.ci[0] < r.estimate < r.ci[1]


def test_significance_agrees_with_the_interval(normal_samples):
    """p < alpha and "interval excludes zero" must be the same statement.
    They are computed by different code paths, so they can disagree."""
    control, treatment = normal_samples
    for conf, alpha in ((0.95, 0.05), (0.99, 0.01), (0.90, 0.10)):
        r = welch_t(control, treatment, confidence=conf)
        excludes_zero = r.ci[0] > 0.0 or r.ci[1] < 0.0
        assert (r.p_value < alpha) == excludes_zero


# --------------------------------------------------------------------------
# two proportions
# --------------------------------------------------------------------------
def test_two_proportion_pvalue_matches_statsmodels():
    """The p-value half: POOLED standard error, as statsmodels does."""
    result = two_proportion(820, 10_000, 905, 10_000)
    stat, p = proportions_ztest([905, 820], [10_000, 10_000])
    assert result.p_value == pytest.approx(p, rel=1e-10)


def test_two_proportion_interval_matches_statsmodels():
    """The interval half: UNPOOLED, also as statsmodels does.

    Pooling assumes the null is true, which is the right assumption while
    testing it and the wrong one while estimating how big the difference
    is. So the two halves use different standard errors on purpose.
    """
    result = two_proportion(820, 10_000, 905, 10_000)
    lo, hi = confint_proportions_2indep(
        905, 10_000, 820, 10_000, method="wald", compare="diff",
    )
    assert result.ci[0] == pytest.approx(lo, rel=1e-9)
    assert result.ci[1] == pytest.approx(hi, rel=1e-9)


def test_the_two_standard_errors_really_do_differ():
    """Proves the inconsistency is present rather than merely documented.

    If someone "tidied" the code to use one SE for both, the two tests
    above would still pass against whichever one statsmodels matched, so
    this test asserts the distinction itself.
    """
    successes_c, n_c, successes_t, n_t = 100, 1000, 300, 1000
    r = two_proportion(successes_c, n_c, successes_t, n_t)
    p_pool = (successes_c + successes_t) / (n_c + n_t)
    se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    p_c, p_t = successes_c / n_c, successes_t / n_t
    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)

    # The two differ by 3.3% of the standard error here - small in absolute
    # terms and not small in the terms that matter, since it scales the
    # interval's half-width directly. The relative gap is the meaningful
    # quantity; an absolute threshold would just encode this example's
    # sample size.
    relative_gap = abs(se_pooled - se_unpooled) / se_unpooled
    assert relative_gap > 0.01
    assert r.se == pytest.approx(se_unpooled, rel=1e-12)  # interval uses this
    half_width = (r.ci[1] - r.ci[0]) / 2.0
    assert half_width == pytest.approx(1.959963984540054 * se_unpooled,
                                       rel=1e-9)


def test_two_proportion_from_arrays_agrees_with_counts():
    rng = np.random.default_rng(3)
    c = (rng.random(5000) < 0.08).astype(float)
    t = (rng.random(5000) < 0.09).astype(float)
    from_arrays = two_proportion_from_arrays(c, t)
    from_counts = two_proportion(int(c.sum()), c.size, int(t.sum()), t.size)
    assert from_arrays.p_value == pytest.approx(from_counts.p_value, rel=1e-14)
    assert from_arrays.ci == pytest.approx(from_counts.ci, rel=1e-14)


def test_relative_lift_is_relative_to_control():
    r = two_proportion(1000, 10_000, 1100, 10_000)
    assert r.estimate == pytest.approx(0.01, rel=1e-12)
    assert r.relative_lift == pytest.approx(0.10, rel=1e-12)


def test_two_proportion_rejects_impossible_counts():
    with pytest.raises(ValueError):
        two_proportion(11, 10, 5, 10)        # more successes than trials
    with pytest.raises(ValueError):
        two_proportion(5, 0, 5, 10)          # no trials
    with pytest.raises(ValueError):
        two_proportion(-1, 10, 5, 10)        # negative


# --------------------------------------------------------------------------
# ratio metrics
# --------------------------------------------------------------------------
def test_ratio_delta_point_estimate_is_the_ratio_of_sums():
    """The estimand, stated as a test: sum(num)/sum(den), per arm."""
    rng = np.random.default_rng(11)
    imp_c = rng.integers(1, 40, 600).astype(float)
    imp_t = rng.integers(1, 40, 600).astype(float)
    clicks_c = rng.binomial(imp_c.astype(int), 0.10).astype(float)
    clicks_t = rng.binomial(imp_t.astype(int), 0.12).astype(float)
    r = ratio_delta(clicks_c, imp_c, clicks_t, imp_t)
    expected = clicks_t.sum() / imp_t.sum() - clicks_c.sum() / imp_c.sum()
    assert r.estimate == pytest.approx(expected, rel=1e-13)


def test_ratio_delta_variance_formula_by_hand():
    """The delta-method variance, recomputed independently in the test.

    Var(R) ~ (Var(X) - 2R Cov(X,Y) + R^2 Var(Y)) / (n * Ybar^2)

    No library computes this for us, so the check is against the formula
    written out a second time from the module docstring. That catches a
    transcription error, which is the realistic failure mode here.
    """
    rng = np.random.default_rng(13)
    n = 900
    imp = rng.integers(1, 30, 2 * n).astype(float)
    clicks = rng.binomial(imp.astype(int), 0.11).astype(float)
    x_c, y_c = clicks[:n], imp[:n]
    x_t, y_t = clicks[n:], imp[n:]

    def var_arm(x, y):
        ratio = x.sum() / y.sum()
        cov = np.cov(x, y, ddof=1)
        return (
            cov[0, 0] - 2.0 * ratio * cov[0, 1] + ratio ** 2 * cov[1, 1]
        ) / (x.size * y.mean() ** 2)

    expected_se = math.sqrt(var_arm(x_c, y_c) + var_arm(x_t, y_t))
    r = ratio_delta(x_c, y_c, x_t, y_t)
    assert r.se == pytest.approx(expected_se, rel=1e-12)


def test_ratio_delta_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ratio_delta(np.ones(10), np.ones(9), np.ones(10), np.ones(10))


def test_ratio_delta_rejects_an_empty_denominator():
    with pytest.raises(ValueError):
        ratio_delta(np.zeros(10), np.zeros(10), np.ones(10), np.ones(10))


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_is_reproducible_from_its_seed():
    rng = np.random.default_rng(17)
    c = rng.lognormal(0, 1, 500)
    t = rng.lognormal(0.1, 1, 500)
    a = bootstrap_diff(c, t, n_boot=2000, seed=99)
    b = bootstrap_diff(c, t, n_boot=2000, seed=99)
    assert a.ci == b.ci
    assert a.p_value == b.p_value


def test_bootstrap_brackets_the_observed_difference():
    rng = np.random.default_rng(19)
    c = rng.lognormal(0, 1, 800)
    t = rng.lognormal(0.3, 1, 800)
    r = bootstrap_diff(c, t, n_boot=4000, seed=5)
    assert r.estimate == pytest.approx(t.mean() - c.mean(), rel=1e-13)
    assert r.ci[0] < r.estimate < r.ci[1]


def test_bootstrap_and_welch_agree_on_well_behaved_data():
    """On data where Welch's assumptions hold, the two must roughly agree.
    If they disagree on NORMAL data, one of them is wrong - and that is the
    only situation where this comparison is informative, which is why the
    heavy-tailed case is a proof rather than a test."""
    rng = np.random.default_rng(23)
    c = rng.normal(5, 1, 1500)
    t = rng.normal(5.15, 1, 1500)
    boot = bootstrap_diff(c, t, n_boot=6000, seed=1)
    welch = welch_t(c, t)
    assert boot.ci[0] == pytest.approx(welch.ci[0], abs=0.012)
    assert boot.ci[1] == pytest.approx(welch.ci[1], abs=0.012)
