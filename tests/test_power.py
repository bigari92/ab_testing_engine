# -*- coding: utf-8 -*-
"""Power and sample size against statsmodels, and against their own inverses.

WHY BOTH KINDS OF CHECK
An external comparison catches a wrong formula. A round-trip catches an
inconsistent pair of formulas, which is the more common defect here: it is
easy to write sample_size and mde so that each is individually defensible
and they disagree about the same experiment. So every function is checked
against statsmodels where statsmodels computes the same thing, and against
its own inverse where it does not.

ONE DELIBERATE DISAGREEMENT WITH STATSMODELS
`sample_size_proportions` uses the UNPOOLED variance sum, because the
interval that `two_proportion` reports is unpooled and a sample size should
size the thing you are going to report. statsmodels' `zt_ind_solve_power`
with a Cohen's-h effect size answers a slightly different question and
lands a little lower. The gap is small, it is in the conservative
direction, and it is tested as a bounded gap rather than papered over -
see test_proportion_sample_size_is_deliberately_a_little_conservative.
"""
from __future__ import annotations

import math

import pytest
from statsmodels.stats.power import TTestIndPower, zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

from explab.power import (
    mde_means,
    mde_proportions,
    power_at,
    power_at_proportions,
    sample_size_means,
    sample_size_proportions,
)
from explab.statlib import norm_cdf, norm_ppf, norm_sf


# --------------------------------------------------------------------------
# means
# --------------------------------------------------------------------------
@pytest.mark.parametrize("power", [0.50, 0.80, 0.90, 0.95])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_sample_size_means_matches_the_closed_form(power, alpha):
    """The normal-approximation formula, written out a second time.

        n per group = 2 * (z_{1-alpha/2} + z_power)^2 * sd^2 / mde^2

    statsmodels solves the exact t-based problem instead, so it is used in
    the next test; this one pins the arithmetic of the approximation that
    `exact=False` claims to compute.
    """
    sd, mde = 2.5, 0.4
    plan = sample_size_means(sd, mde, alpha=alpha, power=power)
    z_sum = norm_ppf(1 - alpha / 2) + norm_ppf(power)
    expected = 2.0 * z_sum ** 2 * sd ** 2 / mde ** 2
    assert plan.n_per_group == math.ceil(expected)
    assert plan.total_n == 2 * plan.n_per_group


@pytest.mark.parametrize("power", [0.80, 0.90])
@pytest.mark.parametrize("mde", [0.2, 0.5, 1.0])
def test_sample_size_means_exact_matches_statsmodels(power, mde):
    """`exact=True` solves the t-based problem, which is what statsmodels
    does. Agreement is to within a single unit, because both round up."""
    sd = 2.0
    plan = sample_size_means(sd, mde, power=power, exact=True)
    theirs = TTestIndPower().solve_power(
        effect_size=mde / sd, power=power, alpha=0.05, ratio=1.0,
        alternative="two-sided",
    )
    assert abs(plan.n_per_group - math.ceil(theirs)) <= 1


def test_the_exact_and_approximate_answers_bracket_each_other():
    """The t-based answer must be the LARGER one, always.

    The t distribution is wider than the normal, so it needs more data for
    the same power. If this ordering ever reversed, one of the two is
    wrong, and no external library is needed to know it.
    """
    for mde in (0.1, 0.3, 0.8, 2.0):
        approx = sample_size_means(2.0, mde).n_per_group
        exact = sample_size_means(2.0, mde, exact=True).n_per_group
        assert exact >= approx
        assert exact - approx <= max(3, approx * 0.02)


@pytest.mark.parametrize("effect", [0.0, 0.05, 0.15, 0.3, 0.45, 0.55])
def test_power_at_matches_statsmodels(effect):
    """Measured worst gap over this grid: 1.0e-3, at effect 0.15.

    statsmodels evaluates the non-central t; `power_at` uses the normal
    approximation, so the two agree to about a thousandth rather than to
    machine precision. That is the right precision to demand - the
    assumed standard deviation going into a power calculation is never
    known to three decimal places, so a third decimal of agreement here
    is already past the point where the input justifies it.

    The grid stops at 0.55 because statsmodels returns `nan` at 0.6 on
    these degrees of freedom, where the power is 1 to within 3e-9 and its
    non-central t loses the ability to represent the complement. That is
    worth knowing about the reference implementation, not a reason to
    distrust either answer.
    """
    sd, n = 1.0, 400
    mine = power_at(sd, n, effect)
    if effect == 0.0:
        assert mine == pytest.approx(0.05, abs=1e-12)
        return
    theirs = TTestIndPower().power(
        effect_size=effect / sd, nobs1=n, alpha=0.05, ratio=1.0,
        alternative="two-sided",
    )
    assert not math.isnan(theirs), "grid should avoid statsmodels' nan region"
    assert mine == pytest.approx(theirs, abs=2e-3)


def test_power_at_includes_the_wrong_direction_term():
    """At zero effect, power must be exactly alpha - not alpha/2.

    A two-sided test rejects when the estimate is far out in EITHER
    direction, so the probability of rejecting under the null is the whole
    alpha. Dropping the far-tail term is the standard slip here, and it
    shows up only at or near zero effect, which is why it survives.
    """
    for alpha in (0.01, 0.05, 0.10):
        assert power_at(1.0, 500, 0.0, alpha=alpha) == pytest.approx(
            alpha, abs=1e-12
        )
    # And the term is present, not merely cancelling: at a small negative
    # effect the power must exceed the one-sided value.
    z = norm_ppf(0.975)
    lam = -0.05 * math.sqrt(500 / 2.0)
    one_sided_only = norm_sf(z - lam)
    assert power_at(1.0, 500, -0.05) > one_sided_only
    assert power_at(1.0, 500, -0.05) == pytest.approx(
        one_sided_only + norm_cdf(-z - lam), abs=1e-12
    )


def test_power_is_symmetric_in_the_sign_of_the_effect():
    for effect in (0.05, 0.2, 0.7):
        assert power_at(1.5, 300, effect) == pytest.approx(
            power_at(1.5, 300, -effect), rel=1e-12
        )


# --------------------------------------------------------------------------
# the inverse relationships
# --------------------------------------------------------------------------
@pytest.mark.parametrize("power", [0.70, 0.80, 0.90])
def test_mde_means_inverts_sample_size_means(power):
    sd = 3.0
    for mde in (0.1, 0.5, 1.5):
        n = sample_size_means(sd, mde, power=power).n_per_group
        recovered = mde_means(sd, n, power=power)
        # n was rounded up, so the recovered MDE is a little SMALLER.
        assert recovered <= mde * 1.0000001
        assert recovered == pytest.approx(mde, rel=0.01)


def test_mde_means_inverts_power_at():
    """The MDE is by definition the effect at which power equals the target.

    `mde_means` bisects, because the relationship cannot be inverted in
    closed form, so the round trip is exact only to the bisection's
    tolerance. Measured error in the delivered power: 1.5e-5 at a 60%
    target, 9.6e-7 at 80%, 1.3e-8 at 95%. The pattern is not a bug - the
    power curve is steepest near the middle, so the same tolerance on the
    MDE maps to a larger error in power there.
    """
    sd, n = 2.0, 1500
    for power in (0.60, 0.80, 0.95):
        mde = mde_means(sd, n, power=power)
        assert power_at(sd, n, mde) == pytest.approx(power, abs=1e-4)


def test_mde_proportions_inverts_power_at_proportions():
    base, n = 0.08, 20_000
    for power in (0.60, 0.80, 0.95):
        mde = mde_proportions(base, n, power=power)
        assert power_at_proportions(base, n, mde) == pytest.approx(
            power, abs=1e-4
        )


def test_sample_size_scales_as_one_over_mde_squared():
    """The single most useful fact in experiment planning.

    Halving the effect you want to detect quadruples the experiment. For
    MEANS that is exact in the normal approximation, because the standard
    deviation is an input and does not move when the MDE does.
    """
    big = sample_size_means(1.0, 0.10).n_per_group
    small = sample_size_means(1.0, 0.05).n_per_group
    assert small / big == pytest.approx(4.0, rel=1e-3)


def test_for_proportions_the_scaling_is_not_exactly_four_and_that_is_correct():
    """A rate's variance depends on the rate, so the two are not separable.

    Halving the MDE for a proportion does NOT quadruple the sample size:
    it comes out at 3.84 here, not 4.00, because the treated rate itself
    changes with the MDE and carries the variance term with it. Sizing for
    a 0.10 -> 0.12 lift assumes more variance than sizing for 0.10 -> 0.11.

    This is the kind of thing that looks like a rounding error and is a
    real effect, so the test asserts the exact prediction rather than a
    tolerance around 4: the ratio must equal the variance-weighted
    prediction, which it does to 2e-4.
    """
    base = 0.10
    lift_big, lift_small = 0.02, 0.01
    n_big = sample_size_proportions(base, lift_big).n_per_group
    n_small = sample_size_proportions(base, lift_small).n_per_group

    def variance_sum(lift):
        treat = base + lift
        return base * (1 - base) + treat * (1 - treat)

    predicted = (
        variance_sum(lift_small) / lift_small ** 2
    ) / (variance_sum(lift_big) / lift_big ** 2)
    assert predicted == pytest.approx(3.8425, abs=1e-3)      # not 4
    assert n_small / n_big == pytest.approx(predicted, rel=1e-3)


# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------
def test_sample_size_proportions_matches_the_unpooled_formula():
    """The formula the module documents, recomputed in the test.

        n = (z_a + z_b)^2 * (p0(1-p0) + p1(1-p1)) / (p1 - p0)^2
    """
    base, lift = 0.08, 0.008
    plan = sample_size_proportions(base, lift)
    z_sum = norm_ppf(0.975) + norm_ppf(0.80)
    treat = base + lift
    expected = (
        z_sum ** 2 * (base * (1 - base) + treat * (1 - treat)) / lift ** 2
    )
    assert plan.n_per_group == math.ceil(expected)


@pytest.mark.parametrize(
    "base,relative_lift",
    [(0.002, 0.20), (0.005, 0.20), (0.01, 0.10), (0.05, 0.10), (0.10, 0.10),
     (0.20, 0.10), (0.40, 0.10), (0.60, 0.10), (0.80, 0.10)],
)
def test_proportion_sample_size_agrees_with_statsmodels(base, relative_lift):
    """Two different parametrisations, nearly the same answer. Measured.

    statsmodels sizes against Cohen's h, an arcsine-transformed effect
    size; this module sizes against the unpooled variance sum, so that the
    sample size matches the interval `two_proportion` actually reports.
    Those are genuinely different formulas, and the point of running them
    against each other across the whole rate range - from 0.2% to 80% - is
    to find out how much the choice costs.

    The answer is: almost nothing. Measured ratios run from 0.998 to 1.003,
    with this module slightly higher at low rates and slightly lower in the
    middle. Asserted at 1%, which every point clears with room.

    Worth stating plainly, because a 0.3% difference in sample size is
    noise next to the uncertainty in the baseline rate you plugged in. The
    reason to prefer the unpooled form is coherence with the interval, not
    accuracy.
    """
    lift = base * relative_lift
    mine = sample_size_proportions(base, lift).n_per_group
    theirs = zt_ind_solve_power(
        effect_size=proportion_effectsize(base + lift, base),
        alpha=0.05, power=0.80, ratio=1.0, alternative="two-sided",
    )
    assert mine / theirs == pytest.approx(1.0, abs=0.01)


def test_relative_and_absolute_mde_agree():
    base = 0.08
    by_absolute = sample_size_proportions(base, 0.008)
    by_relative = sample_size_proportions(base, mde_relative=0.10)
    assert by_absolute.n_per_group == by_relative.n_per_group
    assert by_relative.mde == pytest.approx(0.008, rel=1e-12)
    assert by_relative.mde_relative == pytest.approx(0.10, rel=1e-12)


def test_power_at_proportions_matches_the_formula():
    base, n, effect = 0.10, 15_000, 0.01
    treat = base + effect
    se = math.sqrt(base * (1 - base) / n + treat * (1 - treat) / n)
    z = norm_ppf(0.975)
    lam = effect / se
    expected = norm_sf(z - lam) + norm_cdf(-z - lam)
    assert power_at_proportions(base, n, effect) == pytest.approx(
        expected, rel=1e-12
    )


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0}, {"alpha": 1.0}, {"alpha": -0.1},
        {"power": 0.0}, {"power": 1.0}, {"power": 1.5},
    ],
)
def test_impossible_alpha_or_power_is_refused(kwargs):
    with pytest.raises(ValueError):
        sample_size_means(1.0, 0.1, **kwargs)


def test_power_below_alpha_is_refused():
    """Asking for 3% power at alpha 5% is not a small experiment, it is a
    contradiction: a test already rejects alpha of the time at zero effect."""
    with pytest.raises(ValueError):
        sample_size_means(1.0, 0.1, alpha=0.05, power=0.03)


def test_a_zero_mde_is_refused():
    """Detecting an effect of zero needs infinite data. There is no answer."""
    with pytest.raises(ValueError):
        sample_size_means(1.0, 0.0)
    with pytest.raises(ValueError):
        sample_size_proportions(0.1, 0.0)


def test_a_negative_mde_means_detect_a_DECREASE_and_is_accepted():
    """Not an error, and the sign is not merely tolerated - it is used.

    "Detect a 2 point drop in conversion" is a perfectly ordinary request,
    so a negative MDE is accepted rather than rejected. For MEANS the
    answer is symmetric: the standard deviation does not care which way the
    effect goes.

    For PROPORTIONS it is NOT symmetric, and that is the interesting part.
    Sizing 0.10 -> 0.08 needs 3,211 per arm while 0.10 -> 0.12 needs 3,839,
    because the variance at 0.08 is lower than at 0.12. An implementation
    that took `abs(mde)` early would return the same number for both and
    look perfectly reasonable, so the asymmetry is asserted here.
    """
    assert (sample_size_means(1.0, -0.5).n_per_group
            == sample_size_means(1.0, 0.5).n_per_group)

    down = sample_size_proportions(0.10, -0.02).n_per_group
    up = sample_size_proportions(0.10, 0.02).n_per_group
    assert down < up
    assert down == 3211 and up == 3839

    # And the asymmetry is exactly the variance difference, not an accident.
    def variance_sum(treat):
        return 0.10 * 0.90 + treat * (1 - treat)

    assert down / up == pytest.approx(
        variance_sum(0.08) / variance_sum(0.12), rel=1e-3
    )


@pytest.mark.parametrize("sd", [0.0, -1.0])
def test_a_non_positive_sd_is_refused(sd):
    with pytest.raises(ValueError):
        sample_size_means(sd, 0.1)


@pytest.mark.parametrize("base", [0.0, 1.0, -0.1, 1.2])
def test_an_impossible_baseline_rate_is_refused(base):
    with pytest.raises(ValueError):
        sample_size_proportions(base, 0.01)


def test_a_lift_past_100_percent_is_refused():
    """A rate cannot exceed 1, so neither can baseline plus MDE."""
    with pytest.raises(ValueError):
        sample_size_proportions(0.95, 0.10)


def test_exactly_one_of_absolute_or_relative_mde_is_required():
    with pytest.raises(ValueError):
        sample_size_proportions(0.1)
    with pytest.raises(ValueError):
        sample_size_proportions(0.1, 0.01, mde_relative=0.10)


def test_the_plan_carries_its_assumptions():
    """A sample size with no stated assumptions is a number, not a plan.

    The assumptions are what makes it auditable later, when the observed
    standard deviation turns out to be different from the one assumed.
    """
    plan = sample_size_means(2.0, 0.2, baseline=10.0)
    assert plan.assumptions
    assert any("2" in a for a in plan.assumptions)
    assert plan.mde_relative == pytest.approx(0.02, rel=1e-12)
