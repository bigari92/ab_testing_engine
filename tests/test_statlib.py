# -*- coding: utf-8 -*-
"""statlib against SciPy, function by function.

WHY THIS SUITE EXISTS AND WHAT IT IS NOT
The proofs in `proofs/` establish that the library's METHODS behave
correctly - that a 95% interval covers, that a test rejects at alpha. They
do that by simulation, against known truths, and they would still pass if
a distribution function were slightly off, because simulation error hides
numerical error. Ninety-two simulation claims pass in this repository and
not one of them would have caught any of the three defects listed below.

This suite does the other half: it checks the NUMBERS, against an
independent implementation, to far more digits than any simulation could
resolve. statlib.py implements the normal, Student-t and chi-square
distributions from the incomplete beta and gamma functions rather than
importing SciPy. That is the point of the repository - a p-value you can
derive all the way down - and it is also a liability, because handwritten
special functions are exactly the kind of code that is subtly wrong in the
tails. So SciPy is a test dependency and never a runtime one.

WHAT IT FOUND, all three of them cancellation in disguise
  norm_cdf  returned exactly 0.0 for z below about -8.3, because
            `1 + erf(z/sqrt 2)` cancels. norm_ppf bisects on it, so the
            quantile inherited the damage: 8e-7 of error at p = 1e-12,
            which the docstring had confidently explained as an inherent
            limit of the input. It was not.
  t_cdf     computed its LOWER tail as `0.5 - half_central`, and at
            t = -8 on 500 df that subtraction left two correct digits out
            of sixteen - a relative error of 4.7e-3 in a function whose
            only job is to be exact.
  t_sf      returned exactly 0.5 across a band around the median at large
            df, because its argument df/(df + t^2) rounds to 1.0 there.
            Its own docstring claimed accuracy "from the median all the
            way out" at the time.

Every tolerance below is a MEASURED number, not a round one. Where a
function has a real limit, the limit is asserted from both sides, so it
cannot silently get worse and cannot silently be "fixed" by loosening the
test.
"""
from __future__ import annotations

import math

import pytest
from scipy import special, stats

from explab import statlib as sl


# --------------------------------------------------------------------------
# the special functions underneath everything else
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x", [0.5, 1.0, 1.5, 2.0, 7.0, 25.0, 140.0, 1e4])
def test_log_gamma_matches_scipy(x):
    assert sl.log_gamma(x) == pytest.approx(special.gammaln(x), rel=1e-13)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.5, 10.0, 200.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 2.5, 10.0, 200.0])
@pytest.mark.parametrize("x", [1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6])
def test_betainc_matches_scipy(a, b, x):
    """The regularised incomplete beta, over a grid that crosses the
    continued fraction's two branches (x below and above (a+1)/(a+b+2))."""
    mine = sl.betainc(a, b, x)
    theirs = special.betainc(a, b, x)
    assert mine == pytest.approx(theirs, rel=1e-11, abs=1e-300)


@pytest.mark.parametrize("a", [0.5, 1.0, 5.0, 50.0, 500.0])
@pytest.mark.parametrize("x", [1e-8, 0.1, 1.0, 5.0, 50.0, 500.0, 2000.0])
def test_gammainc_lower_matches_scipy(a, x):
    """Crosses the series/continued-fraction switch at x ~ a + 1."""
    mine = sl.gammainc_lower(a, x)
    theirs = special.gammainc(a, x)
    assert mine == pytest.approx(theirs, rel=1e-11, abs=1e-300)


def test_betainc_endpoints_are_exact():
    for a, b in ((0.5, 0.5), (3.0, 7.0), (100.0, 2.0)):
        assert sl.betainc(a, b, 0.0) == 0.0
        assert sl.betainc(a, b, 1.0) == 1.0


def test_betacf_raises_rather_than_returning_a_half_converged_answer():
    """There is no silent non-convergence anywhere underneath.

    This matters for interpreting the large-df accuracy limit further
    down: that error is the argument running out of significant digits,
    not the continued fraction quietly giving up after 300 iterations. If
    it ever did give up, it would raise, and a raise cannot be mistaken
    for a number.
    """
    import inspect
    source = inspect.getsource(sl._betacf)
    assert "raise" in source, "_betacf must fail loudly, not approximately"


# --------------------------------------------------------------------------
# normal
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "z", [-37.0, -20.0, -10.0, -8.5, -6.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0,
          3.0, 6.0, 8.5, 10.0, 20.0, 37.0]
)
def test_norm_cdf_matches_scipy(z):
    """Measured worst relative error over this grid: 2.1e-13."""
    assert sl.norm_cdf(z) == pytest.approx(stats.norm.cdf(z), rel=1e-12,
                                           abs=1e-300)


@pytest.mark.parametrize("z", [0.0, 0.5, 2.0, 5.0, 8.0, 8.3, 10.0, 20.0,
                               30.0, 37.0])
def test_norm_sf_survives_the_far_tail(z):
    """The whole reason norm_sf exists.

    At z = 10 the survival probability is 7.6e-24. Computing it as
    1 - norm_cdf(z) gives exactly 0.0, because 1 - 7.6e-24 rounds to 1.0.
    A p-value of 0.0 is not a small p-value, it is a broken one.
    """
    assert sl.norm_sf(z) == pytest.approx(stats.norm.sf(z), rel=1e-12)


def test_the_cancellation_that_norm_sf_avoids():
    """Documents the trap, so the test explains itself."""
    z = 10.0
    assert 1.0 - sl.norm_cdf(z) == 0.0        # the wrong way
    assert sl.norm_sf(z) > 0.0                # the right way
    assert sl.norm_sf(z) == pytest.approx(7.619853e-24, rel=1e-6)


def test_norm_cdf_far_lower_tail_regression():
    """REGRESSION: norm_cdf used to return exactly 0.0 here.

    `0.5 * (1 + erf(z/sqrt 2))` at z = -8.5 adds 1 to -0.9999999999999999
    and keeps nothing. The fix is an `erfc` branch for negative z, which
    is the same symmetry norm_sf uses. The values below are SciPy's, and
    before the fix the first two were 0.0.
    """
    assert sl.norm_cdf(-8.5) == pytest.approx(9.479534822203e-18, rel=1e-11)
    assert sl.norm_cdf(-10.0) == pytest.approx(7.619853024161e-24, rel=1e-11)
    assert sl.norm_cdf(-20.0) == pytest.approx(2.753624118606e-89, rel=1e-11)
    # And the identity that had been broken: the normal is symmetric.
    for z in (0.5, 3.0, 6.0, 8.5, 10.0, 20.0, 30.0):
        assert sl.norm_cdf(-z) == pytest.approx(sl.norm_sf(z), rel=1e-14)


@pytest.mark.parametrize("p", [1e-15, 1e-12, 1e-8, 1e-4, 0.001, 0.025, 0.1,
                               0.5, 0.9, 0.975, 0.999, 1 - 1e-7, 1 - 1e-9,
                               1 - 1e-11, 1 - 1e-13])
def test_norm_ppf_matches_scipy(p):
    """Measured worst absolute error over this grid: 1.9e-13.

    The grid deliberately runs past p = 1 - 1e-7, where the docstring
    used to concede roughly 1e-8 of error. That concession described a
    bug in norm_cdf, now fixed, and the tolerance here is what the solver
    actually delivers.
    """
    assert sl.norm_ppf(p) == pytest.approx(stats.norm.ppf(p), abs=1e-11)


def test_the_literal_is_what_is_lossy_not_the_solver():
    """The limit that IS real, separated from the one that was a bug.

    `1 - 1e-11` in source does not produce that number. It produces the
    nearest double, up to 5.5e-17 away, and the normal density out at
    z = 6.7 is about 2.2e-11, so the rounding alone displaces the true
    quantile by roughly 3e-6. Nothing downstream can recover it.

    This test asserts the mechanism rather than asserting an error: the
    double is shown to differ from the decimal, which is why passing the
    tail probability directly - and negating - is the actual fix.
    """
    p = 1 - 1e-11
    assert p != 1 - 1e-11 + 0.0 or True          # the literal is a double
    tail_from_subtraction = 1.0 - p              # exact, by Sterbenz
    assert tail_from_subtraction != 1e-11        # but not the number meant
    assert abs(tail_from_subtraction - 1e-11) < 1e-16

    # The lossless route: ask for the tail directly and use symmetry.
    direct = -sl.norm_ppf(1e-11)
    assert direct == pytest.approx(-stats.norm.ppf(1e-11), abs=1e-12)
    # Both routes agree to the precision the damaged input allows.
    assert sl.norm_ppf(p) == pytest.approx(direct, abs=1e-5)


def test_norm_ppf_round_trips_through_norm_cdf():
    for p in (1e-12, 1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-6):
        assert sl.norm_cdf(sl.norm_ppf(p)) == pytest.approx(p, rel=1e-11)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_norm_ppf_rejects_impossible_probabilities(p):
    with pytest.raises(ValueError):
        sl.norm_ppf(p)


def test_the_1_96_everyone_quotes():
    """The number that turns a confidence level into a multiplier."""
    assert sl.norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=1e-13)


# --------------------------------------------------------------------------
# Student's t
# --------------------------------------------------------------------------
@pytest.mark.parametrize("df", [1.0, 2.0, 4.5, 30.0, 500.0, 5000.0, 1e5, 1e7])
@pytest.mark.parametrize("t", [-200.0, -60.0, -8.0, -2.5, -1.0, -1e-7, 0.0,
                               1e-7, 1.0, 2.5, 8.0, 60.0, 200.0])
def test_t_cdf_matches_scipy(t, df):
    """Measured worst relative error over this grid: 3.6e-9, at df = 1e7.

    The tolerance is set by the largest df in the grid; see
    test_t_accuracy_degrades_only_with_df for where that comes from and
    why it is left as it is.
    """
    assert sl.t_cdf(t, df) == pytest.approx(stats.t.cdf(t, df), rel=1e-8,
                                            abs=1e-300)


@pytest.mark.parametrize("df", [1.0, 4.5, 30.0, 500.0, 1e6])
@pytest.mark.parametrize("t", [0.0, 1.0, 3.0, 9.0, 30.0, 200.0])
def test_t_sf_survives_the_far_tail(t, df):
    assert sl.t_sf(t, df) == pytest.approx(stats.t.sf(t, df), rel=1e-8,
                                           abs=1e-300)


def test_t_cdf_lower_tail_regression():
    """REGRESSION: t_cdf's lower tail was computed by cancellation.

    The old code used `0.5 - half_central` whenever t^2 < df, which covers
    t = -8 at 500 df. The true value is 4.3648e-15 and it returned
    4.3854e-15 - a relative error of 4.7e-3. At df = 1e5 it was 2.9e-2,
    nearly 3%. The fix routes t < 0 through t_sf(-t) by symmetry.

    Asserted to 1e-10 relative, which the cancelling version misses by
    seven orders of magnitude.
    """
    assert sl.t_cdf(-8.0, 500.0) == pytest.approx(4.364831396940e-15,
                                                  rel=1e-10)
    assert sl.t_cdf(-8.0, 1e5) == pytest.approx(6.286959938128e-16, rel=1e-9)
    assert sl.t_cdf(-20.0, 30.0) == pytest.approx(3.374541832886e-19,
                                                  rel=1e-10)
    # The identity that had been broken: the t is symmetric.
    for df in (2.0, 4.5, 30.0, 500.0, 1e5):
        for t in (1.0, 2.5, 8.0, 20.0, 60.0):
            assert sl.t_cdf(-t, df) == pytest.approx(sl.t_sf(t, df),
                                                     rel=1e-14)


def test_t_sf_is_not_flat_near_the_median_regression():
    """REGRESSION: t_sf returned exactly 0.5 across a band at large df.

    `df/(df + t^2)` rounds to 1.0 for |t| below about 1e-5 once df is
    1e7, and I_1(a,b) is exactly 1, so the function returned 0.5 where
    the truth was 0.4999960. The docstring claimed accuracy "from the
    median all the way out" while this was true.

    The fix switches to the central parametrisation once the tail form's
    argument is within 1e-12 of 1.0 - the region where the answer is near
    0.5 anyway and the central form is the accurate one.
    """
    for df in (1e5, 1e7):
        for t in (1e-9, 1e-7, 1e-5):
            mine = sl.t_sf(t, df)
            assert mine != 0.5, f"t_sf flat at t={t}, df={df}"
            assert mine == pytest.approx(stats.t.sf(t, df), abs=1e-13)


def test_t_sf_reaches_the_pvalues_a_real_test_produces():
    """A CUPED-adjusted test in this repository produced |t| = 8.9.

    Computed as 1 - t_cdf that is exactly 0.0. The tail form gives
    2.9e-19. That number is why inference.py routes every p-value through
    t_sf rather than 1 - t_cdf, and why P_FLOOR exists for the cases
    beyond even this.
    """
    p = 2.0 * sl.t_sf(8.9, 39_998.0)
    assert p == pytest.approx(2.0 * stats.t.sf(8.9, 39_998.0), rel=1e-9)
    assert 0.0 < p < 1e-17
    assert 1.0 - sl.t_cdf(8.9, 39_998.0) == 0.0     # the trap being avoided


def test_t_cdf_is_strictly_monotone_through_the_median_at_large_df():
    """REGRESSION for the bug that passed every obvious check.

    The original parametrisation computed x = df/(df + t^2), which rounds
    to exactly 1.0 for tiny t at large df, making the CDF numerically flat
    for a band around zero. Nothing looked wrong: t_cdf(0) was 0.5, the
    tails were right, the function was still monotone NON-decreasing. But
    t_ppf bisects on it, and a flat region through the median meant
    t_ppf(0.5 + tiny) came back wrong by ~1e-6.

    STRICT monotonicity is what was broken, so strict monotonicity is what
    is asserted - through both the t_cdf and the t_sf paths, since the fix
    now lives in t_sf and t_cdf delegates to it.
    """
    for df in (1e5, 1e7):
        previous = sl.t_cdf(-1e-6, df)
        for t in (-1e-7, -1e-8, 0.0, 1e-8, 1e-7, 1e-6):
            current = sl.t_cdf(t, df)
            assert current > previous, f"t_cdf flat at t={t}, df={df}"
            previous = current

        previous = sl.t_sf(1e-6, df)
        for t in (1e-7, 1e-8, 0.0, -1e-8, -1e-7, -1e-6):
            current = sl.t_sf(t, df)
            assert current > previous, f"t_sf flat at t={t}, df={df}"
            previous = current


def test_t_accuracy_degrades_only_with_df_and_only_this_far():
    """The one limit left, pinned from BOTH sides so it cannot drift.

    t_sf's argument is x = df/(df + t^2). At df = 1e7 that is
    0.9999999000000, and a double has only about ten significant digits
    of `1 - x` left to give - which `betainc` needs internally. So the
    information is gone before the algorithm starts. It is not
    non-convergence: `_betacf` raises rather than returning a
    half-converged answer.

    It could be fixed by letting betainc take the complement directly,
    and deliberately is not: 4e-9 of relative error on a p-value changes
    no decision, and at a million degrees of freedom norm_sf is the right
    tool anyway. The upper bounds below stop it getting worse. The lower
    bound on the df = 1e7 case stops anyone "fixing" this test by
    loosening it - if the error ever drops below 1e-10 there, the limit
    has genuinely been addressed and this test should be rewritten, not
    relaxed.
    """
    grid = (-200.0, -60.0, -8.0, -2.5, -1.0, 1.0, 2.5, 8.0, 60.0, 200.0)

    def worst(df):
        return max(
            abs(sl.t_cdf(t, df) - stats.t.cdf(t, df)) / stats.t.cdf(t, df)
            for t in grid if stats.t.cdf(t, df) > 0
        )

    assert worst(2.0) < 1e-13         # measured 1.9e-15
    assert worst(30.0) < 1e-13        # measured 1.2e-14
    assert worst(500.0) < 1e-11       # measured 4.8e-13
    assert worst(1e4) < 1e-10         # measured 2.0e-11
    assert worst(1e5) < 1e-9          # measured 4.8e-11
    assert worst(1e7) < 1e-8          # measured 3.6e-9
    assert worst(1e7) > 1e-10, "the large-df limit is gone; rewrite this test"


@pytest.mark.parametrize("df", [1.0, 3.0, 12.0, 250.0, 1e4, 1e6])
@pytest.mark.parametrize("p", [1e-8, 0.001, 0.025, 0.25, 0.5, 0.75, 0.975,
                               0.999])
def test_t_ppf_matches_scipy(p, df):
    """Relative AND absolute tolerance, because the quantiles span 1e7.

    At p = 1e-8 on 1 df the quantile is -3.18e7, where 1.1e-6 of absolute
    error is 3.4e-14 relative. At p = 0.5 it is 0, where relative error is
    meaningless. pytest.approx passes on either, which is the correct
    semantics for a function whose output has no natural scale.
    """
    assert sl.t_ppf(p, df) == pytest.approx(stats.t.ppf(p, df), rel=1e-8,
                                            abs=1e-8)


def test_t_ppf_approaches_the_normal_as_df_grows():
    """A property check, not a comparison: the t must become the normal."""
    for p in (0.025, 0.5, 0.975, 0.999):
        assert sl.t_ppf(p, 1e8) == pytest.approx(sl.norm_ppf(p), abs=1e-6)


# --------------------------------------------------------------------------
# chi-square
# --------------------------------------------------------------------------
@pytest.mark.parametrize("df", [1, 2, 3, 5, 20, 100])
@pytest.mark.parametrize("x", [1e-6, 0.1, 1.0, 3.84, 10.0, 50.0, 300.0])
def test_chi2_cdf_matches_scipy(x, df):
    """Measured worst relative error over this grid: 5.7e-14."""
    assert sl.chi2_cdf(x, df) == pytest.approx(stats.chi2.cdf(x, df),
                                               rel=1e-12, abs=1e-300)


def test_chi2_tail_at_the_srm_threshold():
    """srm.py tests at p = 0.001, four orders out from where anything else
    in the library uses this tail. If the chi-square tail were wrong there,
    every SRM verdict in the project would be wrong and nothing else would
    notice - which is why proof_srm.py measures the false alarm rate at
    exactly that threshold, and why this test pins the arithmetic."""
    critical = stats.chi2.ppf(1 - 0.001, 1)
    assert 1.0 - sl.chi2_cdf(critical, 1) == pytest.approx(0.001, rel=1e-9)


def test_chi2_with_1_df_is_the_squared_normal():
    """An identity, checkable without SciPy: P(chi2_1 <= x) = P(|Z| <= sqrt x).

    Worth having because it ties two independently implemented functions
    together. gammainc_lower and erf share no code, so agreement here is
    real evidence rather than a tautology.
    """
    for x in (0.5, 1.0, 3.84, 10.0, 25.0):
        assert sl.chi2_cdf(x, 1) == pytest.approx(
            2.0 * sl.norm_cdf(math.sqrt(x)) - 1.0, rel=1e-11
        )


@pytest.mark.parametrize("df", [1, 2, 3, 5, 20, 100])
@pytest.mark.parametrize("x", [1e-6, 0.1, 1.0, 3.84, 10.0, 50.0, 300.0, 1000.0])
def test_chi2_sf_matches_scipy(x, df):
    """Measured worst relative error over this grid: 2.7e-14."""
    assert sl.chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df),
                                              rel=1e-12, abs=1e-300)


def test_chi2_sf_survives_the_far_tail_where_the_subtraction_does_not():
    """The regression for the seventh defect found in this project.

    srm.py computed its p-value as `1 - chi2_cdf(chi2, df)`. On 1 degree
    of freedom that returns exactly 0.0 for every chi-square past about
    100, and a sample ratio mismatch is the single test most likely to
    produce one: srm.py's own docstring calls an 11-sigma deviation the
    ordinary case at two million users.

    So the asserts below are in pairs. The first of each pair is what the
    old expression returned; the second is the truth.
    """
    for chi in (100.0, 200.0, 400.0, 1000.0):
        assert 1.0 - sl.chi2_cdf(chi, 1) == 0.0          # what it used to say
        assert sl.chi2_sf(chi, 1) > 0.0                  # what is true
        assert sl.chi2_sf(chi, 1) == pytest.approx(stats.chi2.sf(chi, 1),
                                                   rel=1e-11)
    # The specific values, so a rewrite cannot quietly regress them.
    assert sl.chi2_sf(100.0, 1) == pytest.approx(1.5230178e-23, rel=1e-6)
    assert sl.chi2_sf(400.0, 1) == pytest.approx(2.5597525e-89, rel=1e-6)


def test_chi2_sf_and_cdf_sum_to_one_where_both_are_accurate():
    """Both branches of gammainc_upper, checked against their complement.

    In the middle of the distribution neither function is near its limits,
    so the identity must hold to machine precision. This is the test that
    catches a mirrored branch applied on the wrong side of x = a + 1.
    """
    for df in (1, 2, 7, 30):
        for x in (0.3, 1.0, 2.5, 6.0, 12.0, 25.0):
            assert sl.chi2_cdf(x, df) + sl.chi2_sf(x, df) == pytest.approx(
                1.0, rel=1e-14
            )


def test_srm_reports_a_real_p_value_on_an_impossible_split():
    """The defect as it actually appeared: a readout printing `p = 0`."""
    from explab.srm import srm_check

    r = srm_check({"control": 94_000, "treatment": 100_000})
    assert not r.passed
    assert r.p_value > 0.0
    assert r.p_value == pytest.approx(stats.chi2.sf(r.chi_square, 1), rel=1e-10)
    assert "p = 0," not in str(r)
