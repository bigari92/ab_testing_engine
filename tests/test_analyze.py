# -*- coding: utf-8 -*-
"""The readout end to end: does the decision follow from the evidence?

WHAT THIS FILE TESTS THAT NOTHING ELSE DOES
Every other test here checks a number. These check a JUDGEMENT. `analyze`
takes a frame and a list of declared metrics and produces one of a small
set of decisions, and the rule that produces it is the most opinionated
code in the repository:

  1. SRM first. If the split is broken, nothing else is reported, because
     every estimate below it is measuring a non-random slice of users.
  2. Guardrails next, uncorrected. A harm blocks a ship even when the
     primary metric won.
  3. Then the primary metric, and only then.

And one distinction most readouts collapse into a single "not
significant":

  FLAT           the whole interval is inside the practical threshold. The
                 experiment answered the question: there is no effect big
                 enough to matter. That is a result.
  INCONCLUSIVE   the interval still contains effects you would ship for.
                 The experiment did not answer the question. That is not a
                 result, and calling it one is how a real effect gets
                 abandoned.

Those two have the same p-value and opposite meanings, which is why they
are tested separately below. The data comes from `generate.py` with a
known planted truth, so each test states what was planted and what the
readout is required to conclude from it.
"""
from __future__ import annotations

import numpy as np
import pytest

from generate import generate_experiment

from explab.analyze import MetricSpec, analyze
from explab.report import render_markdown, render_text


def conversion(threshold: float | None = 0.004, **kw) -> MetricSpec:
    return MetricSpec("conversion", "binary", column="converted",
                      role="primary", practical_threshold=threshold, **kw)


REFUND_GUARDRAIL = MetricSpec("refund_rate", "binary", column="refunded",
                              role="guardrail", direction="decrease")


# --------------------------------------------------------------------------
# the SRM gate
# --------------------------------------------------------------------------
def test_a_broken_split_stops_the_readout_before_any_effect_is_reported():
    """The planted SRM dataset: 50.7% / 49.3% on 200,000 users.

    The decision must be the SRM, not the conversion result - and the
    reasons must say so, because an analyst who sees only "invalid" will
    go looking for the cause and an analyst who sees a lift will ship it.
    """
    frame, truth = generate_experiment(
        n_users=200_000, seed=2, assignment_share=0.507,
        effect_conversion=0.008,
    )
    assert truth.srm_planted

    readout = analyze(frame, [conversion(), REFUND_GUARDRAIL])
    assert not readout.srm.passed
    assert readout.decision in ("invalid", "srm")
    assert any("ratio" in r.lower() or "srm" in r.lower()
               for r in readout.reasons)

    # The effect IS still computed - hiding it would stop anyone
    # diagnosing the imbalance - but the decision does not rest on it.
    assert readout.primary is not None
    rendered = render_text(readout)
    assert "sample ratio" in rendered.lower()


def test_a_clean_split_passes_the_gate():
    frame, _ = generate_experiment(n_users=40_000, seed=1,
                                   effect_conversion=0.008)
    readout = analyze(frame, [conversion(), REFUND_GUARDRAIL])
    assert readout.srm.passed
    assert readout.decision != "invalid"


# --------------------------------------------------------------------------
# flat versus inconclusive: the same p-value, opposite meanings
# --------------------------------------------------------------------------
def test_a_real_effect_is_called_a_win():
    """0.8 points planted on an 8% base, 40,000 users. That is detectable
    at this size, so the readout is required to find it."""
    frame, truth = generate_experiment(n_users=40_000, seed=1,
                                       effect_conversion=0.008)
    readout = analyze(frame, [conversion(), REFUND_GUARDRAIL])
    assert readout.primary.result.significant
    assert readout.primary.moved_in_good_direction
    assert readout.decision == "ship"

    # The estimate must land near the planted truth - but NOT via "the
    # interval contains it", which is the tempting assertion and the wrong
    # one: a 95% interval misses 5% of the time by construction, so that
    # test would fail on one seed in twenty and teach everyone to re-run
    # it until it passed. Coverage is a property of the METHOD over many
    # samples, and proof_ci_coverage.py is where it is measured.
    #
    # What can be asserted on a single sample is that the estimate is
    # within a few standard errors of the truth. At four SEs this holds
    # 99.99% of the time, so it catches a real bias and not a coin flip.
    result = readout.primary.result
    assert abs(result.estimate - truth.realised_effect_conversion) < (
        4.0 * result.se
    )


def test_a_true_zero_at_a_large_sample_is_FLAT_not_inconclusive():
    """No effect planted, and enough users to rule out one that matters.

    The whole interval lands inside the 0.4-point practical threshold, so
    the experiment HAS answered the question: there is nothing here worth
    shipping for. The decision must say that, not "we need more data".
    """
    frame, truth = generate_experiment(n_users=400_000, seed=11,
                                       effect_conversion=0.0)
    assert truth.effect_conversion == 0.0
    readout = analyze(frame, [conversion(threshold=0.004)])
    lo, hi = readout.primary.result.ci
    assert abs(lo) < 0.004 and abs(hi) < 0.004
    assert readout.primary.beyond_practical is False
    assert readout.decision == "flat"
    assert "flat" in render_text(readout).lower()


def test_a_true_zero_at_a_small_sample_is_INCONCLUSIVE_not_flat():
    """The same zero effect, a tenth of the users, the same threshold.

    Now the interval still admits a full point of lift either way - more
    than the threshold - so the experiment has NOT answered the question.
    Identical truth, identical p-value regime, opposite conclusion, and
    the difference is entirely the width of the interval.
    """
    frame, _ = generate_experiment(n_users=8_000, seed=12,
                                   effect_conversion=0.0)
    readout = analyze(frame, [conversion(threshold=0.004)])
    lo, hi = readout.primary.result.ci
    assert not readout.primary.result.significant
    assert max(abs(lo), abs(hi)) > 0.004
    assert readout.primary.beyond_practical is True
    assert readout.decision == "inconclusive"


def test_without_a_practical_threshold_the_distinction_is_refused_not_guessed():
    """No threshold declared, so "flat" cannot be claimed.

    The readout must fall back to inconclusive and say why, rather than
    inventing a threshold. This is the one place where the library's
    answer gets worse because the caller declared less, and it says so
    instead of hiding it.
    """
    frame, _ = generate_experiment(n_users=400_000, seed=11,
                                   effect_conversion=0.0)
    readout = analyze(frame, [conversion(threshold=None)])
    assert readout.primary.beyond_practical is None
    assert readout.decision != "flat"
    assert any("threshold" in r.lower() for r in readout.reasons)


# --------------------------------------------------------------------------
# guardrails
# --------------------------------------------------------------------------
def test_a_guardrail_regression_blocks_a_winning_primary():
    """Both planted: conversion up 1 point, refunds up 3 points.

    The primary metric wins and the readout must still not say ship. A
    guardrail exists precisely for the case where the thing you were
    optimising improved.
    """
    frame, _ = generate_experiment(
        n_users=120_000, seed=21, effect_conversion=0.01,
        effect_refund=0.03,
    )
    readout = analyze(frame, [conversion(), REFUND_GUARDRAIL])
    assert readout.srm.passed
    assert readout.primary.result.significant
    assert readout.primary.moved_in_good_direction

    guardrail = readout.guardrails[0]
    assert guardrail.result.significant
    assert not guardrail.moved_in_good_direction
    assert readout.decision != "ship"
    assert any("guardrail" in r.lower() for r in readout.reasons)


def test_guardrails_are_reported_uncorrected_and_the_readout_says_so():
    """proof_multiplicity.py measures why: correcting five guardrails costs
    21 points of real-harm detection, 80% down to 58%. The readout has to
    make the choice visible, because an uncorrected family of five has a
    23% false alarm rate and a reader needs to know which trade was made.
    """
    frame, _ = generate_experiment(n_users=60_000, seed=22)
    guardrails = [
        MetricSpec(f"guard_{col}", "binary", column=col, role="guardrail",
                   direction="decrease")
        for col in ("refunded",)
    ]
    readout = analyze(frame, [conversion()] + guardrails)
    rendered = render_text(readout)
    assert "uncorrected" in rendered.lower()
    for outcome in readout.guardrails:
        assert outcome.q_value is None


# --------------------------------------------------------------------------
# secondary metrics and correction
# --------------------------------------------------------------------------
def test_secondary_metrics_are_corrected_and_labelled_exploratory():
    frame, _ = generate_experiment(n_users=60_000, seed=31)
    metrics = [
        conversion(),
        MetricSpec("revenue", "continuous", column="revenue"),
        MetricSpec("minutes", "continuous", column="minutes",
                   covariate="pre_minutes"),
        MetricSpec("ctr", "ratio", numerator="clicks",
                   denominator="impressions"),
    ]
    readout = analyze(frame, metrics)
    assert readout.secondary_correction is not None
    assert len(readout.secondary) == 3
    for outcome in readout.secondary:
        assert outcome.q_value is not None
        assert outcome.q_value >= outcome.result.p_value - 1e-12
    assert "exploratory" in render_text(readout).lower()


def test_the_primary_metric_is_never_corrected():
    """One primary metric means one test means nothing to correct for.

    Correcting it would be the mirror of the guardrail mistake: paying a
    multiplicity price for a comparison that was declared in advance and
    is the only one deciding anything.
    """
    frame, _ = generate_experiment(n_users=60_000, seed=32)
    readout = analyze(frame, [
        conversion(),
        MetricSpec("revenue", "continuous", column="revenue"),
    ])
    assert readout.primary.q_value is None


def test_cuped_is_applied_when_declared_and_narrows_the_interval():
    """pre_minutes correlates about 0.77 with minutes in the generator, so
    CUPED should remove roughly 59% of the variance - and the interval
    must come out narrower than the unadjusted one on the same data."""
    frame, _ = generate_experiment(n_users=40_000, seed=41,
                                   effect_minutes=0.4)
    adjusted = analyze(frame, [
        conversion(),
        MetricSpec("minutes", "continuous", column="minutes",
                   covariate="pre_minutes"),
    ])
    plain = analyze(frame, [
        conversion(),
        MetricSpec("minutes", "continuous", column="minutes"),
    ])

    adj_outcome = next(o for o in adjusted.secondary
                       if o.spec.name == "minutes")
    plain_outcome = next(o for o in plain.secondary
                         if o.spec.name == "minutes")
    assert adj_outcome.cuped is not None
    assert plain_outcome.cuped is None
    assert adj_outcome.cuped.correlation > 0.6

    adj_width = adj_outcome.result.ci[1] - adj_outcome.result.ci[0]
    plain_width = plain_outcome.result.ci[1] - plain_outcome.result.ci[0]
    assert adj_width < plain_width * 0.85

    # And it must not have moved the answer, only tightened it. The
    # estimates agree well inside the unadjusted interval's half-width.
    assert adj_outcome.result.estimate == pytest.approx(
        plain_outcome.result.estimate, abs=plain_width / 2
    )


# --------------------------------------------------------------------------
# the rendered report
# --------------------------------------------------------------------------
@pytest.fixture
def full_readout():
    frame, _ = generate_experiment(n_users=50_000, seed=51,
                                   effect_conversion=0.006)
    return analyze(frame, [
        conversion(),
        REFUND_GUARDRAIL,
        MetricSpec("minutes", "continuous", column="minutes",
                   covariate="pre_minutes"),
        MetricSpec("ctr", "ratio", numerator="clicks",
                   denominator="impressions"),
    ])


def test_the_report_leads_with_the_decision(full_readout):
    """Not with a table. The first section is the conclusion and the
    reasons for it, because that is what the reader came for and what they
    will quote. Everything after it is the evidence."""
    text = render_text(full_readout)
    assert "DECISION" in text
    assert text.index("DECISION") < text.index("ASSIGNMENT CHECK")
    assert text.index("DECISION") < text.index("PRIMARY METRIC")

    # And the decision carries its reasons with it, rather than leaving the
    # reader to reconstruct them from the tables below.
    decision_block = text[text.index("DECISION"):text.index("ASSIGNMENT")]
    assert full_readout.reasons
    assert all(reason[:40] in decision_block
               for reason in full_readout.reasons)


def test_the_report_puts_the_interval_before_the_p_value(full_readout):
    """Ordering as an argument: the interval says how big the effect might
    be, which is the decision-relevant quantity. The p-value says only
    whether zero is in it."""
    text = render_text(full_readout)
    primary_line = next(ln for ln in text.splitlines()
                        if "conversion" in ln and "[" in ln)
    assert primary_line.index("[") < primary_line.find("p =") or \
        "p =" not in primary_line


def test_the_report_states_the_sample_ratio_check_above_the_effects(
        full_readout):
    """The assignment check comes before any effect estimate.

    Ordering as an argument again: if the split is broken, every number
    below is measuring a non-random slice of users, so the reader needs
    the check before they have formed an opinion about the effect - not
    in a footnote after it. The section header even says so.

    Compared against the EVIDENCE sections, not the whole document: the
    decision block above names the primary metric in its reasons, which is
    correct and should not count as reporting an effect.
    """
    text = render_text(full_readout)
    assert "ASSIGNMENT CHECK" in text
    assert "sample ratio check:" in text
    assert text.index("ASSIGNMENT CHECK") < text.index("PRIMARY METRIC")
    assert text.index("ASSIGNMENT CHECK") < text.index("GUARDRAILS")
    assert "before anything is believed" in text


def test_the_report_carries_the_mde_alongside_the_result(full_readout):
    """An estimate without the effect the experiment could have detected
    is not interpretable: "no significant difference" at an MDE of 5
    points means something entirely different from the same words at an
    MDE of 0.2 points."""
    assert full_readout.primary.detectable is not None
    text = render_text(full_readout).lower()
    assert "detect" in text


def test_both_renderers_produce_output_and_neither_prints_nan(full_readout):
    for text in (render_text(full_readout), render_markdown(full_readout)):
        assert len(text.splitlines()) > 8
        assert "nan" not in text.lower()
    assert "|" in render_markdown(full_readout)      # it really is a table


def test_the_readout_is_reproducible(full_readout):
    """Same seed, same frame, same words. A readout that changes between
    runs cannot be reviewed."""
    frame, _ = generate_experiment(n_users=50_000, seed=51,
                                   effect_conversion=0.006)
    again = analyze(frame, [
        conversion(),
        REFUND_GUARDRAIL,
        MetricSpec("minutes", "continuous", column="minutes",
                   covariate="pre_minutes"),
        MetricSpec("ctr", "ratio", numerator="clicks",
                   denominator="impressions"),
    ])
    assert render_text(again) == render_text(full_readout)


# --------------------------------------------------------------------------
# the generator's own contract
# --------------------------------------------------------------------------
def test_the_generator_plants_what_it_says_it_plants():
    """The sealed key has to be true, or every proof resting on it is void.

    `realised_effect_conversion` is the effect actually present in the
    rows, which differs slightly from the requested one because
    probabilities get clipped and users get assigned by a coin flip. The
    key records the realised figure, and this checks the two agree to
    within what the sampling allows.
    """
    frame, truth = generate_experiment(n_users=200_000, seed=61,
                                       effect_conversion=0.01)
    control = frame.loc[frame.variant == "control", "converted"].mean()
    treatment = frame.loc[frame.variant == "treatment", "converted"].mean()
    observed = treatment - control
    assert truth.realised_effect_conversion == pytest.approx(0.01, abs=0.002)
    assert observed == pytest.approx(truth.realised_effect_conversion,
                                     abs=0.004)


def test_the_generator_can_produce_a_true_null():
    """Used by proof_false_positive.py, so it has to really be null."""
    frame, truth = generate_experiment(
        n_users=100_000, seed=62, effect_conversion=0.0, effect_revenue=0.0,
        effect_ctr=0.0, effect_minutes=0.0, effect_refund=0.0,
    )
    assert truth.realised_effect_conversion == pytest.approx(0.0, abs=1e-9)
    readout = analyze(frame, [conversion()])
    assert readout.srm.passed


def test_the_engaged_column_is_post_treatment_and_labelled_as_such():
    """A deliberate trap in the generator, kept for the segment lesson.

    `engaged` is produced AFTER the treatment and is moved by it, so
    segmenting on it manufactures an effect difference out of nothing.
    This is the real mechanism behind most Simpson's-paradox readouts, and
    it is in the data on purpose - with the key saying so.
    """
    frame, truth = generate_experiment(n_users=40_000, seed=63,
                                       effect_conversion=0.01)
    assert truth.post_treatment_segment
    assert "engaged" in frame.columns
    assert any("engaged" in note for note in truth.notes)
