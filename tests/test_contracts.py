# -*- coding: utf-8 -*-
"""The guards: what the library refuses, and what it warns about instead.

WHY THIS IS A WHOLE FILE
Most of what makes an analysis wrong is not a wrong formula. It is a right
formula applied to the wrong thing: a covariate measured after the
treatment started, a chi-square test on cells with two units in them, a
second primary metric added after the first one came back flat, a sixth
look at an experiment that declared five. None of those produce an error.
They produce a number, and the number is confident.

So the library refuses some things and warns about others, and the
difference between those two lists is a design decision that deserves
testing:

  REFUSE  when there is no correct answer to give. A ratio metric with no
          denominator, an experiment with two primary metrics, a look that
          was never declared. Returning something would be worse than
          stopping.
  WARN    when the answer is correct but the reader needs to know
          something about it. A correlation of 0.04 that makes CUPED
          pointless, a retention curve of length one, a CAC denominator
          that includes organic signups. The caller may have good reason;
          the reader still needs the caveat.

A guard with no test is a guard that gets refactored away, which is why
these are here rather than left to the docstrings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explab.analyze import MetricSpec, analyze
from explab.cuped import cuped_adjust
from explab.segments import segment_scan
from explab.sequential import calibrate_boundary
from explab.srm import srm_check
from explab.unit_economics import (
    compute,
    expected_lifetime,
    ltv_from_retention,
    ltv_geometric,
    retention_from_activity,
)

# The smallest calibration sequential.py will accept. Used in the guard
# tests below so that the n_sims check does not fire before the guard
# actually under test - it is deliberately the first thing validated.
MIN_SIMS = 10_000


# --------------------------------------------------------------------------
# MetricSpec: declared before the data is looked at, so it is checked then
# --------------------------------------------------------------------------
def test_a_ratio_metric_needs_both_columns():
    with pytest.raises(ValueError, match="numerator and denominator"):
        MetricSpec("ctr", "ratio", numerator="clicks")


def test_cuped_on_a_ratio_metric_is_refused_rather_than_approximated():
    """The honest refusal: the delta method would have to carry the
    covariate through, and this project does not implement that. Silently
    applying CUPED to the numerator would produce a narrower interval
    around a subtly different estimand."""
    with pytest.raises(ValueError, match="delta method"):
        MetricSpec("ctr", "ratio", numerator="clicks",
                   denominator="impressions", covariate="pre_ctr")


def test_cuped_on_a_binary_metric_is_refused():
    with pytest.raises(ValueError, match="continuous"):
        MetricSpec("conv", "binary", column="converted",
                   covariate="pre_converted")


def test_a_non_ratio_metric_needs_a_column():
    with pytest.raises(ValueError, match="needs a column"):
        MetricSpec("revenue", "continuous")


@pytest.mark.parametrize("threshold", [0.0, -0.01])
def test_a_practical_threshold_must_be_a_positive_magnitude(threshold):
    """Direction is declared separately, so the threshold is a size."""
    with pytest.raises(ValueError, match="positive"):
        MetricSpec("conv", "binary", column="converted",
                   practical_threshold=threshold)


def test_a_valid_spec_survives():
    spec = MetricSpec("minutes", "continuous", column="minutes",
                      covariate="pre_minutes", role="secondary",
                      practical_threshold=0.5)
    assert spec.covariate == "pre_minutes"


# --------------------------------------------------------------------------
# analyze: the structural rules of a readout
# --------------------------------------------------------------------------
@pytest.fixture
def small_frame():
    rng = np.random.default_rng(5)
    n = 2000
    return pd.DataFrame({
        "variant": ["control"] * n + ["treatment"] * n,
        "converted": rng.binomial(1, 0.1, 2 * n),
        "revenue": rng.lognormal(0, 1, 2 * n),
        "clicks": rng.binomial(10, 0.1, 2 * n),
        "impressions": np.full(2 * n, 10),
    })


def test_two_primary_metrics_is_refused(small_frame):
    """The whole point of a primary metric is that it decides. Two of them
    means the decision rule is "whichever one wins", which is a false
    positive rate nobody computed."""
    with pytest.raises(ValueError, match="primary"):
        analyze(small_frame, [
            MetricSpec("a", "binary", column="converted", role="primary"),
            MetricSpec("b", "continuous", column="revenue", role="primary"),
        ])


def test_no_metrics_is_refused(small_frame):
    with pytest.raises(ValueError, match="no metrics"):
        analyze(small_frame, [])


def test_duplicate_metric_names_are_refused(small_frame):
    with pytest.raises(ValueError, match="duplicate"):
        analyze(small_frame, [
            MetricSpec("same", "binary", column="converted", role="primary"),
            MetricSpec("same", "continuous", column="revenue"),
        ])


def test_a_missing_variant_column_is_refused(small_frame):
    with pytest.raises(ValueError, match="variant"):
        analyze(small_frame.drop(columns=["variant"]),
                [MetricSpec("a", "binary", column="converted",
                            role="primary")])


def test_a_missing_metric_column_is_refused(small_frame):
    with pytest.raises(ValueError, match="nope"):
        analyze(small_frame, [MetricSpec("a", "binary", column="nope",
                                         role="primary")])


# --------------------------------------------------------------------------
# srm_check
# --------------------------------------------------------------------------
def test_srm_needs_at_least_two_arms():
    with pytest.raises(ValueError, match="at least 2 arms"):
        srm_check({"control": 1000})


def test_srm_refuses_negative_counts():
    with pytest.raises(ValueError, match="negative"):
        srm_check({"control": 1000, "treatment": -5})


def test_srm_refuses_an_empty_experiment():
    with pytest.raises(ValueError, match="no units"):
        srm_check({"control": 0, "treatment": 0})


def test_srm_refuses_shares_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        srm_check({"control": 500, "treatment": 500},
                  expected_shares={"control": 0.5, "treatment": 0.4})


def test_srm_refuses_an_arm_with_no_declared_share():
    with pytest.raises(ValueError, match="no expected share"):
        srm_check({"control": 400, "a": 300, "b": 300},
                  expected_shares={"control": 0.5, "a": 0.5})


def test_srm_refuses_a_declared_arm_that_received_nothing():
    """And the error says what to do: pass it as 0, because an arm that
    got no units is itself the mismatch you are looking for."""
    with pytest.raises(ValueError, match="pass it as 0"):
        srm_check({"control": 500, "treatment": 500},
                  expected_shares={"control": 0.4, "treatment": 0.4,
                                   "ghost": 0.2})


def test_srm_refuses_cells_too_small_for_chi_square():
    """Below five expected units the chi-square approximation does not
    hold, and the function says so rather than returning a p-value that
    looks like any other."""
    with pytest.raises(ValueError, match="chi-square approximation"):
        srm_check({"control": 4, "treatment": 4})


def test_srm_expects_shares_against_the_observed_total_not_a_planned_one():
    """A quiet weekend is not an SRM. The test asks whether the units you
    logged were SPLIT as designed, not whether there were as many as you
    hoped - conflating those is how a real alert gets dismissed."""
    big = srm_check({"control": 50_000, "treatment": 50_000})
    small = srm_check({"control": 500, "treatment": 500})
    assert big.passed and small.passed
    assert big.p_value == pytest.approx(small.p_value, rel=1e-12)


# --------------------------------------------------------------------------
# cuped
# --------------------------------------------------------------------------
def test_cuped_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        cuped_adjust(np.ones(10), np.ones(9), np.ones(10), np.ones(10))


def test_cuped_refuses_nan():
    bad = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="NaN"):
        cuped_adjust(bad, np.ones(4), np.ones(4), np.ones(4))


def test_cuped_refuses_a_single_unit_per_arm():
    with pytest.raises(ValueError, match="at least 2"):
        cuped_adjust(np.ones(1), np.ones(1), np.ones(1), np.ones(1))


def test_cuped_warns_rather_than_failing_on_a_useless_covariate():
    """A near-zero correlation is not an error - it is a correct answer
    that happens to be worthless, and the right response is to say so."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(4000)
    x = rng.standard_normal(4000)          # independent of y
    result = cuped_adjust(y[:2000], x[:2000], y[2000:], x[2000:])
    assert abs(result.correlation) < 0.1
    assert any("correlation is only" in w for w in result.warnings)
    assert result.adjusted_control.size == 2000


def test_cuped_warns_when_the_covariate_looks_post_treatment():
    """A pre-period metric correlated 0.99 with the outcome is usually the
    outcome under another name, and proof_cuped.py measures what that does
    to the estimate: 100% of the real effect, gone, with a narrow
    interval and no complaint."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(4000)
    y = 0.995 * x + np.sqrt(1 - 0.995 ** 2) * rng.standard_normal(4000)
    result = cuped_adjust(y[:2000], x[:2000], y[2000:], x[2000:])
    assert any("suspiciously high" in w for w in result.warnings)


def test_cuped_survives_a_constant_covariate():
    """Zero variance means theta is undefined. Return the data unchanged
    with a warning, rather than dividing by zero."""
    rng = np.random.default_rng(6)
    y = rng.standard_normal(1000)
    x = np.ones(1000)
    result = cuped_adjust(y[:500], x[:500], y[500:], x[500:])
    assert result.theta == 0.0
    assert any("zero variance" in w for w in result.warnings)
    assert np.allclose(result.adjusted_control, y[:500])


# --------------------------------------------------------------------------
# sequential
# --------------------------------------------------------------------------
def test_an_undeclared_extra_look_voids_the_plan():
    """The single most important guard in the file.

    Boundaries are calibrated for a declared number of looks. A sixth look
    on a five-look plan has no boundary, and quietly reusing the fifth
    one spends alpha that was never budgeted. proof_peeking.py measures
    what that costs: five unplanned looks take 5% to 14.2%.
    """
    plan = calibrate_boundary(3, n_sims=MIN_SIMS, seed=1)
    assert plan.decide([0.5, 1.2, 1.9])["stop"] is False
    with pytest.raises(ValueError, match="voids the guarantee"):
        plan.decide([0.5, 1.2, 1.9, 2.4])


def test_sequential_refuses_a_calibration_too_coarse_to_trust():
    """The guard that fires before every other argument check, on purpose.

    A boundary calibrated on 1,000 simulations carries a Monte Carlo
    standard error of about 0.007 on an alpha of 0.05 - a seventh of the
    quantity being calibrated. It would return a number, and the number
    would be indistinguishable from a correct one. Since the whole claim
    of sequential.py is that its boundaries hold alpha, a calibration run
    too short to establish that is refused rather than reported with a
    caveat.
    """
    with pytest.raises(ValueError, match="too coarse"):
        calibrate_boundary(3, n_sims=1000)


def test_sequential_refuses_an_unknown_boundary_shape():
    with pytest.raises(ValueError, match="unknown shape"):
        calibrate_boundary(3, shape="aggressive", n_sims=MIN_SIMS)


def test_sequential_refuses_information_fractions_out_of_order():
    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_boundary(3, information_fractions=[0.3, 0.2, 1.0],
                           n_sims=MIN_SIMS)


def test_sequential_refuses_information_fractions_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        calibrate_boundary(3, information_fractions=[0.3, 0.6, 1.4],
                           n_sims=MIN_SIMS)


def test_sequential_refuses_a_fraction_count_that_disagrees_with_n_looks():
    with pytest.raises(ValueError):
        calibrate_boundary(5, information_fractions=[0.5, 1.0],
                           n_sims=MIN_SIMS)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_sequential_refuses_an_impossible_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        calibrate_boundary(3, alpha=alpha, n_sims=MIN_SIMS)


def test_sequential_refuses_zero_looks():
    with pytest.raises(ValueError, match="n_looks"):
        calibrate_boundary(0, n_sims=MIN_SIMS)


# --------------------------------------------------------------------------
# unit economics
# --------------------------------------------------------------------------
def test_negative_revenue_is_refused_with_an_instruction():
    """Refunds are real, and averaging them into revenue silently is how
    ARPU ends up meaning nothing. The error says to handle them
    explicitly rather than guessing which convention the caller meant."""
    with pytest.raises(ValueError, match="refunds explicitly"):
        compute([10.0, 5.0, -3.0, 20.0])


def test_an_increasing_retention_curve_is_refused():
    """Retention cannot go up: it is the share of an original cohort still
    present, and people do not un-churn into a cohort they already left.
    An increasing curve means the denominator changed between periods,
    which is survivorship, and the lifetime computed from it is fiction."""
    with pytest.raises(ValueError, match="retention"):
        expected_lifetime([1.0, 0.6, 0.7])


@pytest.mark.parametrize("curve", [[1.0, 0.5, 1.2], [1.0, -0.1]])
def test_retention_outside_zero_to_one_is_refused(curve):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_lifetime(curve)


def test_an_empty_retention_curve_is_refused():
    with pytest.raises(ValueError, match="empty"):
        expected_lifetime([])


def test_a_negative_discount_rate_is_refused():
    with pytest.raises(ValueError, match="discount_rate"):
        ltv_from_retention(10.0, [1.0, 0.7, 0.5], discount_rate=-0.1)


@pytest.mark.parametrize("churn", [0.0, 1.5, -0.2])
def test_an_impossible_churn_rate_is_refused(churn):
    with pytest.raises(ValueError, match="churn_rate"):
        ltv_geometric(10.0, churn)


def test_a_non_2d_activity_matrix_is_refused():
    with pytest.raises(ValueError, match="2-D"):
        retention_from_activity(np.ones(10))


def test_zero_customers_acquired_is_refused():
    with pytest.raises(ValueError, match="customers_acquired"):
        compute([10.0, 20.0], acquisition_spend=500.0, customers_acquired=0)


def test_a_length_one_retention_curve_warns_rather_than_guessing():
    """One period of retention cannot imply a lifetime, but it is also
    what you have in week one of a new product. So: compute it, and say
    the horizon is an assumption rather than a measurement."""
    result = compute([10.0, 20.0, 30.0, 15.0], retention=[1.0])
    assert result.warnings


def test_a_wide_arpu_interval_warns():
    """ARPU on 30 users with lognormal revenue has an interval wide enough
    that any LTV built on it is decoration. The warning is the point."""
    rng = np.random.default_rng(8)
    result = compute(rng.lognormal(2.0, 1.5, 30))
    assert result.warnings


# --------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------
def test_segment_scan_refuses_a_missing_column(small_frame):
    with pytest.raises(ValueError, match="not in the frame"):
        segment_scan(small_frame, metric_column="converted",
                     segment_column="country")


def test_segment_scan_refuses_a_variant_label_with_no_rows(small_frame):
    with pytest.raises(ValueError, match="no rows"):
        segment_scan(small_frame, metric_column="converted",
                     segment_column="variant", treatment_label="variant_b")


def test_an_underpowered_segment_is_reported_but_not_tested():
    """Reported, not hidden - and that distinction is the design.

    A segment with six users per arm cannot support an estimate: the
    interval would be wider than the metric's range. There are two wrong
    ways to handle it. Testing it anyway puts a meaningless row in the
    table and makes the real segments pay for it in the multiplicity
    correction. Dropping it silently leaves the reader thinking the
    segment was fine, or that it was never there.

    So it appears in `findings` with `result=None` and `powered=False`,
    carrying the effect it WOULD have needed to show something - which is
    the one genuinely useful thing to say about an underpowered segment -
    and it is excluded from the correction.
    """
    rng = np.random.default_rng(9)
    n = 6000
    draw = rng.random(n)
    # Three sizes: comfortably testable, below the floor but estimable,
    # and too small for even a detectable effect to be computed.
    segment = np.where(draw < 0.94, "big",
                       np.where(draw < 0.995, "small", "minuscule"))
    frame = pd.DataFrame({
        "variant": np.where(rng.random(n) < 0.5, "control", "treatment"),
        "converted": rng.binomial(1, 0.1, n),
        "segment": segment,
    })
    scan = segment_scan(frame, metric_column="converted",
                        segment_column="segment", min_per_arm=200)

    by_name = {f.segment: f for f in scan.findings}
    assert set(by_name) == {"big", "small", "minuscule"}   # all reported

    big, small, minuscule = (by_name["big"], by_name["small"],
                             by_name["minuscule"])
    assert big.powered and big.result is not None
    assert big.detectable_effect is not None

    # Below the floor: no estimate, and the floor is explained.
    assert not small.powered and small.result is None
    assert min(small.n_control, small.n_treatment) < 200
    assert any("below the floor" in note for note in small.notes)

    # And smaller still. In both cases the detectable effect may be
    # genuinely uncomputable - at 150 units per arm on a 10% rate, no
    # effect this side of a 100% relative change is detectable at all. The
    # requirement is that it then reports None, never NaN: a NaN prints as
    # "nan" and compares False against every threshold it meets, which is
    # how a missing number becomes a silent one.
    assert min(minuscule.n_control, minuscule.n_treatment) < 20
    assert minuscule.result is None
    for finding in (small, minuscule):
        assert finding.detectable_effect is None or (
            finding.detectable_effect > big.detectable_effect
        )
        if finding.detectable_effect is not None:
            assert not np.isnan(finding.detectable_effect)

    # None of the unpowered segments carries a corrected p-value, because
    # none of them was tested - they cost the real segments nothing.
    assert small.q_value is None and minuscule.q_value is None


# --------------------------------------------------------------------------
# NaN never reaches a reader
# --------------------------------------------------------------------------
# Three separate places returned NaN, all three of them printed it, and all
# three were found by this file rather than by any of the 92 simulation
# claims in proofs/. A NaN is worse than an error: it compares False
# against every threshold it meets, propagates through arithmetic without
# complaint, and renders as "nan" in a readout that is otherwise careful
# about every number on it. The project already refuses to print a p-value
# of 0.0 for the same reason.
def test_mde_proportions_raises_rather_than_returning_nan():
    """The fix that matters most, because a caller already handles the raise.

    segments.py wraps this in `except ValueError: detectable = None`. With
    NaN returned instead, that handler never fired and the NaN went into a
    SegmentFinding and then into the report.
    """
    from explab.power import mde_proportions

    with pytest.raises(ValueError, match="cannot detect anything"):
        mde_proportions(0.1, 157)
    # And where an answer does exist, it is still returned.
    assert mde_proportions(0.1, 1000) == pytest.approx(0.040685, rel=1e-4)


def test_a_power_plan_with_no_baseline_does_not_print_nan():
    """`mde_relative` is NaN when no baseline was given, because a share of
    an unknown baseline is not a quantity. The plan says so in words."""
    from explab.power import sample_size_means

    rendered = str(sample_size_means(2.0, 0.3))
    assert "nan" not in rendered.lower()
    assert "no baseline given" in rendered
    # With a baseline, the relative figure appears as normal.
    assert "+3.00%" in str(sample_size_means(2.0, 0.3, baseline=10.0))


def test_a_zero_control_mean_does_not_print_nan():
    """0 -> 0.008 is an infinite relative lift, not a missing one.

    The readout says the ratio is undefined and points at the absolute
    change, which is the whole story in that case: the treatment produced
    behaviour that did not exist before, and no percentage expresses that.
    """
    from explab.inference import two_proportion

    result = two_proportion(0, 5000, 40, 5000)
    rendered = str(result)
    assert "nan" not in rendered.lower()
    assert "relative undefined" in rendered
    assert "+0.008" in rendered          # the absolute change is still there
    # The property itself still returns NaN, which is the right value for a
    # ratio with a zero denominator - it is the DISPLAY that must not.
    assert np.isnan(result.relative_lift)


def test_no_readout_in_the_repository_prints_nan():
    """A sweep, so a fourth NaN cannot appear somewhere new unnoticed."""
    from explab.inference import ratio_delta, two_proportion, welch_t
    from explab.power import sample_size_means, sample_size_proportions
    from explab.srm import srm_check

    rng = np.random.default_rng(42)
    rendered = [
        str(welch_t(np.zeros(500), rng.standard_normal(500))),
        str(two_proportion(0, 1000, 0, 1000)),
        str(two_proportion(1000, 1000, 1000, 1000)),
        str(ratio_delta(np.zeros(300), np.ones(300),
                        np.zeros(300), np.ones(300))),
        str(sample_size_means(1.0, 0.1)),
        str(sample_size_proportions(0.5, 0.01)),
        str(srm_check({"control": 5000, "treatment": 5000})),
    ]
    for text in rendered:
        assert "nan" not in text.lower(), text
