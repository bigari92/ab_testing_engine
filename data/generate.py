#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic experiments where the right answer is known, and written down separately.

WHY SYNTHETIC DATA IS THE POINT, NOT A COMPROMISE
-------------------------------------------------
On real experiment data you can check that an analysis runs. You cannot
check that it is right, because nobody knows the true effect — that is why
the experiment was run. So a portfolio built on real data can demonstrate
competence but cannot demonstrate correctness.

Generate the data and you know the truth exactly. Then "does this analysis
recover the effect" becomes a measurable question with a number for an
answer, and "does it catch the defect" becomes a scoreable test. Every
claim this project makes about its own accuracy comes from here.

THE SEALED KEY
--------------
`generate_experiment` returns two things: a DataFrame, and an
`ExperimentTruth` holding every parameter used to build it. The analysis
functions never see the truth object. `scoring/run_scored_trials.py`
generates, analyses blind, and only then opens the key to score. That
separation is the whole method, and it is enforced by the function
signatures: nothing in `explab` accepts an `ExperimentTruth`.

THE DEFECTS THAT CAN BE PLANTED
-------------------------------
Each is a real failure mode that has shipped a wrong decision somewhere.

srm_share           assignment probability away from 0.50. Detected by
                    srm_check. At 0.505 on 200k users this is invisible to
                    the eye and fatal to the analysis.
outlier_rate        a few users with revenue multiplied by `outlier_scale`.
                    Breaks the mean, leaves the median alone, and makes the
                    t-interval wider than it looks like it should be.
novelty_strength    effect inflated for users assigned early, decaying with
                    a time constant. A cumulative readout shows the effect
                    shrinking day by day, and an experiment stopped early
                    "while it was winning" ships the decay.
hetero_segment      the effect exists in ONE segment and nowhere else. The
                    overall effect is diluted; the segment scan should find
                    it and the heterogeneity test should reject.
post_treatment_seg  include an "engaged" column computed from behaviour
                    DURING the experiment. Because the treatment changes
                    who becomes engaged, slicing on it compares different
                    populations and can reverse the sign. This is the
                    Simpson's paradox generator, and it is a data-modelling
                    bug wearing a statistics costume.
ratio_heterogeneity per-user click rates drawn from a Beta rather than
                    fixed. With fixed rates, analysing clicks at the
                    impression level is only slightly wrong; with
                    realistic spread it is badly wrong. Feeds
                    proof_delta_method.py.

THE LATENT VARIABLE
-------------------
Every user gets a quality score q ~ Normal(0,1) that drives pre-period
revenue, conversion propensity, impression volume and revenue size at once.
That single line is what makes the dataset behave like real data: it
creates the pre/post correlation CUPED exploits, the numerator/denominator
covariance the delta method needs, and the heavy revenue tail that breaks
naive means. Without it, every technique in this project would look
unnecessary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ExperimentTruth", "generate_experiment", "write_dataset"]

COUNTRIES = ("UA", "PL", "DE", "US", "BR")
DEVICES = ("mobile", "desktop", "tablet")
PLANS = ("free", "pro")


@dataclass(frozen=True)
class ExperimentTruth:
    """Everything the analysis is not allowed to see.

    Effects are ABSOLUTE and on the metric's own scale:
      effect_conversion   added to the conversion probability
      effect_revenue      multiplicative on revenue among converters
      effect_ctr          added to each user's click-through rate
      effect_minutes      added to minutes in the period, per user
      effect_refund       added to the refund probability among converters
    """
    n_users: int
    seed: int
    base_conversion: float
    effect_conversion: float
    effect_revenue: float
    effect_ctr: float
    effect_minutes: float
    effect_refund: float
    assignment_share: float
    srm_planted: bool
    outlier_rate: float
    outlier_scale: float
    novelty_strength: float
    novelty_tau: float
    hetero_segment: str | None
    hetero_effect: float
    post_treatment_segment: bool
    ratio_heterogeneity: float
    days: int
    realised_effect_conversion: float = 0.0
    realised_effect_revenue_per_user: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def has_conversion_effect(self) -> bool:
        return abs(self.effect_conversion) > 0 or (
            self.hetero_segment is not None and abs(self.hetero_effect) > 0
        )


def generate_experiment(
    n_users: int = 40_000,
    *,
    seed: int = 0,
    base_conversion: float = 0.08,
    effect_conversion: float = 0.008,
    effect_revenue: float = 0.0,
    effect_ctr: float = 0.0,
    effect_minutes: float = 0.0,
    effect_refund: float = 0.0,
    assignment_share: float = 0.5,
    outlier_rate: float = 0.0,
    outlier_scale: float = 40.0,
    novelty_strength: float = 0.0,
    novelty_tau: float = 5.0,
    hetero_segment: str | None = None,
    hetero_effect: float = 0.0,
    post_treatment_segment: bool = True,
    ratio_heterogeneity: float = 8.0,
    days: int = 21,
) -> tuple[pd.DataFrame, ExperimentTruth]:
    """Build one experiment dataset and its sealed key.

    Defaults give a clean, well-behaved experiment with a real 0.8pp lift
    on an 8% base: the shape of an ordinary product test that a 40,000-user
    sample is adequately powered to find. Pass the defect parameters to
    break it in one specific way at a time.

    ratio_heterogeneity is the concentration of the Beta that per-user
    click rates are drawn from. Large means every user is near the mean
    (homogeneous); small means wide spread. 8 is realistic; set it to a
    few thousand to make users effectively identical.
    """
    if n_users < 100:
        raise ValueError(f"n_users={n_users} is too small to be interesting")
    if not 0.0 < base_conversion < 1.0:
        raise ValueError(f"base_conversion must be in (0,1), got {base_conversion}")
    if not 0.05 < assignment_share < 0.95:
        raise ValueError(f"assignment_share {assignment_share} is outside (0.05, 0.95)")
    if hetero_segment is not None and hetero_segment not in (
        COUNTRIES + DEVICES + PLANS
    ):
        raise ValueError(
            f"hetero_segment {hetero_segment!r} is not one of the generated "
            f"levels: {COUNTRIES + DEVICES + PLANS}"
        )

    rng = np.random.default_rng(seed)

    # ---- the latent user quality that makes everything correlate --------
    q = rng.standard_normal(n_users)

    # ---- pre-experiment behaviour: the CUPED covariate ------------------
    # Gamma shaped by q, so heavy users were heavy before the test too.
    pre_revenue = rng.gamma(shape=2.0, scale=np.exp(0.45 * q) * 6.0)
    pre_sessions = rng.poisson(np.exp(0.5 + 0.35 * q))

    # A well-behaved continuous metric with a strong pre-period counterpart:
    # minutes in the app. Both periods share a persistent per-user level
    # `habit`, which is what makes the pre-period a useful covariate at all.
    # This is the metric CUPED is for. Conversion and revenue are not: they
    # are zero for most users, so their correlation with any pre-period
    # measure is weak, and docs/10 shows that contrast rather than hiding it.
    habit = 18.0 + 9.0 * q + rng.standard_normal(n_users) * 2.0
    pre_minutes = np.clip(habit + rng.standard_normal(n_users) * 5.0, 0.0, None)
    minutes = np.clip(habit + rng.standard_normal(n_users) * 5.0, 0.0, None)

    # ---- pre-treatment segments -----------------------------------------
    country = rng.choice(COUNTRIES, size=n_users, p=(0.30, 0.20, 0.18, 0.22, 0.10))
    device = rng.choice(DEVICES, size=n_users, p=(0.62, 0.33, 0.05))
    plan = np.where(q + rng.standard_normal(n_users) * 0.6 > 0.9, "pro", "free")

    # ---- assignment ------------------------------------------------------
    treated = rng.random(n_users) < assignment_share
    variant = np.where(treated, "treatment", "control")
    assigned_day = rng.integers(1, days + 1, size=n_users)

    # ---- the treatment effect, possibly decaying, possibly segmented ----
    # Novelty: users assigned on day 1 see the inflated effect; it decays
    # with assignment day, so a cumulative readout drifts downward.
    novelty = 1.0 + novelty_strength * np.exp(-(assigned_day - 1) / novelty_tau)

    in_segment = np.zeros(n_users, dtype=bool)
    if hetero_segment is not None:
        for column, values in (
            (country, COUNTRIES), (device, DEVICES), (plan, PLANS)
        ):
            if hetero_segment in values:
                in_segment = column == hetero_segment
                break

    effect_per_user = effect_conversion * novelty + np.where(
        in_segment, hetero_effect, 0.0
    )

    # ---- conversion ------------------------------------------------------
    # Logistic in q so propensity varies, then shift the PROBABILITY by the
    # effect. Shifting the probability rather than the log-odds keeps the
    # truth stated in the same units the analysis reports, which is what
    # makes blind scoring possible.
    base_logit = np.log(base_conversion / (1 - base_conversion)) + 0.8 * q
    p_control = 1.0 / (1.0 + np.exp(-base_logit))
    p_user = np.clip(p_control + np.where(treated, effect_per_user, 0.0), 1e-6, 1 - 1e-6)
    converted = (rng.random(n_users) < p_user).astype(int)

    # ---- revenue ---------------------------------------------------------
    # Lognormal among converters, scaled by q: a heavy tail that is not an
    # accident of the distribution but a property of who the users are.
    revenue = np.zeros(n_users)
    conv_idx = converted == 1
    n_conv = int(conv_idx.sum())
    revenue[conv_idx] = rng.lognormal(mean=2.4 + 0.5 * q[conv_idx], sigma=0.9, size=n_conv)
    if effect_revenue:
        revenue[conv_idx & treated] *= 1.0 + effect_revenue

    # ---- planted outliers ------------------------------------------------
    n_outliers = 0
    if outlier_rate > 0:
        # Spread evenly across arms so they add variance without adding bias.
        candidates = np.flatnonzero(conv_idx)
        n_outliers = int(round(outlier_rate * n_users))
        n_outliers = min(n_outliers, candidates.size)
        if n_outliers:
            chosen = rng.choice(candidates, size=n_outliers, replace=False)
            revenue[chosen] *= outlier_scale

    # ---- the ratio metric: impressions and clicks ------------------------
    # Impression volume rises with q, so numerator and denominator
    # covary — the thing the delta method exists for.
    impressions = rng.poisson(np.exp(1.6 + 0.5 * q)) + 1
    base_ctr = 0.11
    if ratio_heterogeneity <= 0:
        raise ValueError("ratio_heterogeneity must be positive")
    # Beta with the requested concentration, mean base_ctr.
    a = base_ctr * ratio_heterogeneity
    b = (1.0 - base_ctr) * ratio_heterogeneity
    ctr_user = rng.beta(a, b, size=n_users)
    ctr_user = np.clip(ctr_user + np.where(treated, effect_ctr, 0.0), 1e-6, 1 - 1e-6)
    clicks = rng.binomial(impressions, ctr_user)

    # The minutes effect lands on the post-period only, never on pre_minutes:
    # a covariate the treatment can touch is not a covariate (see cuped.py).
    if effect_minutes:
        minutes = minutes + np.where(treated, effect_minutes * novelty, 0.0)

    # A guardrail metric: refunds among converters. A treatment that wins
    # conversion by overselling shows up here and nowhere else, which is the
    # entire reason guardrails exist and the reason they are never
    # multiplicity-corrected (see multiplicity.py).
    # The q coefficient is kept small enough that clipping never binds, so
    # the refund effect in the data equals the parameter exactly and blind
    # scoring has an unambiguous target.
    base_refund = 0.08
    refund_p = np.clip(
        base_refund + np.where(treated, effect_refund, 0.0) + 0.01 * q,
        1e-6, 1 - 1e-6,
    )
    refunded = ((rng.random(n_users) < refund_p) & (converted == 1)).astype(int)

    frame = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "variant": variant,
        "assigned_day": assigned_day,
        "pre_revenue": pre_revenue,
        "pre_sessions": pre_sessions,
        "pre_minutes": pre_minutes,
        "country": country,
        "device": device,
        "plan": plan,
        "converted": converted,
        "revenue": revenue,
        "refunded": refunded,
        "minutes": minutes,
        "impressions": impressions,
        "clicks": clicks,
    })

    # ---- the post-treatment segment trap ---------------------------------
    if post_treatment_segment:
        # "Engaged" is defined by behaviour DURING the experiment, so the
        # treatment moves who lands in it. Slicing on this column compares
        # two different populations; mix_shift_check is built to catch it.
        engaged_score = clicks + 3 * converted + rng.random(n_users)
        frame["engaged"] = (engaged_score > np.quantile(engaged_score, 0.70)).astype(int)

    # ---- the realised effect, for scoring --------------------------------
    # The TRUE effect in the data as generated, not the parameter: with
    # finite users the realised difference in expected conversion differs
    # slightly from `effect_conversion`, and scoring should be against what
    # was actually built.
    realised_conv = float(
        np.mean(np.clip(p_control[treated] + effect_per_user[treated], 0, 1))
        - np.mean(p_control[treated])
    ) if treated.any() else 0.0

    rev_c = revenue[~treated].mean() if (~treated).any() else 0.0
    rev_t = revenue[treated].mean() if treated.any() else 0.0

    notes = []
    if n_outliers:
        notes.append(f"{n_outliers} revenue outliers planted at {outlier_scale}x")
    if novelty_strength:
        notes.append(
            f"novelty: effect inflated {1 + novelty_strength:.2f}x on day 1, "
            f"decaying with tau={novelty_tau} days"
        )
    if hetero_segment:
        notes.append(
            f"effect {hetero_effect:+.4f} confined to {hetero_segment} "
            f"({int(in_segment.sum())} users)"
        )
    if post_treatment_segment:
        notes.append("'engaged' is post-treatment and must not be sliced on")

    truth = ExperimentTruth(
        n_users=n_users, seed=seed, base_conversion=base_conversion,
        effect_conversion=effect_conversion, effect_revenue=effect_revenue,
        effect_ctr=effect_ctr, effect_minutes=effect_minutes,
        effect_refund=effect_refund,
        assignment_share=assignment_share,
        srm_planted=abs(assignment_share - 0.5) > 1e-9,
        outlier_rate=outlier_rate, outlier_scale=outlier_scale,
        novelty_strength=novelty_strength, novelty_tau=novelty_tau,
        hetero_segment=hetero_segment, hetero_effect=hetero_effect,
        post_treatment_segment=post_treatment_segment,
        ratio_heterogeneity=ratio_heterogeneity, days=days,
        realised_effect_conversion=realised_conv,
        realised_effect_revenue_per_user=float(rev_t - rev_c),
        notes=tuple(notes),
    )
    return frame, truth


def write_dataset(
    directory: str | Path = "data/generated",
    **kwargs,
) -> tuple[Path, Path]:
    """Write one dataset plus its key to disk, as two separate files.

    Two files, not one, and the key is named `*_answer_key.json` so that an
    accidental read is obvious in a diff. The analysis entry points take a
    CSV path; they have no parameter that could accept the key.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    frame, truth = generate_experiment(**kwargs)
    tag = f"seed{truth.seed}_n{truth.n_users}"
    data_path = out / f"experiment_{tag}.csv"
    key_path = out / f"experiment_{tag}_answer_key.json"
    frame.to_csv(data_path, index=False)
    truth.to_json(key_path)
    return data_path, key_path


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("writing the three datasets the docs refer to\n")
    specs = [
        ("clean", dict(seed=1, n_users=40_000)),
        ("broken_srm", dict(seed=2, n_users=200_000, assignment_share=0.507,
                            effect_conversion=0.0)),
        ("messy", dict(seed=3, n_users=60_000, effect_conversion=0.0,
                       hetero_segment="mobile", hetero_effect=0.02,
                       outlier_rate=0.0008, novelty_strength=1.5,
                       effect_revenue=0.0)),
    ]
    for name, kwargs in specs:
        data_path, key_path = write_dataset(**kwargs)
        frame, truth = generate_experiment(**kwargs)
        print(f"{name:12s} -> {data_path.name}")
        print(f"             {len(frame):,} rows, "
              f"conversion {frame['converted'].mean():.4%}, "
              f"revenue/user {frame['revenue'].mean():.4f}")
        for note in truth.notes:
            print(f"             planted: {note}")
        print()
