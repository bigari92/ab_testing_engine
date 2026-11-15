#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARPU, LTV, CAC, payback — with the definitions written down and an interval attached.

WHY THIS MODULE IS IN AN EXPERIMENT PROJECT
-------------------------------------------
Because these are the numbers an experiment is usually trying to move, and
because the same discipline applies to both: an estimate without an
interval is a guess with a decimal point. In 27 of the 128 job postings
this project was built against, unit economics appear by name. In almost
every dashboard that reports them, LTV appears as a single number with no
range, computed from a formula nobody can reproduce.

THE DEFINITIONS, WHICH ARE THE ACTUAL DELIVERABLE
-------------------------------------------------
ARPU      average revenue per USER, over the measurement window:
          total revenue / all users in the cohort, payers and non-payers.
ARPPU     average revenue per PAYING user: total revenue / payers only.
          ARPU = ARPPU * conversion_rate. Confusing the two is the most
          common error in this area and it inflates LTV by 1/conversion,
          which on a 3% conversion rate is a factor of 33.
CAC       customer acquisition cost: acquisition spend / customers
          acquired BY THAT SPEND in the same window. The denominator is
          where this goes wrong — organic signups do not belong in it.
LTV       expected revenue from one acquired user over a stated horizon,
          discounted if you say so. There is no such thing as LTV without
          a horizon; "lifetime" is not a number.
          LTV = ARPU_per_period * sum over k of retention_k / (1+d)^k
payback   periods of ARPU needed to recover CAC, on the retention curve.

WHAT GOES WRONG, IN ORDER OF HOW OFTEN
--------------------------------------
1. NO HORIZON. "LTV is $340" with no window. If the horizon is not stated
   the number cannot be checked, and it will be compared against a CAC
   measured over a month. This module requires a horizon and prints it.

2. A SINGLE CHURN RATE FOR A CURVE THAT IS NOT GEOMETRIC. LTV = ARPU/churn
   assumes a constant hazard: the same share leaves every period. Real
   curves decay fast then flatten, because the people most likely to leave
   leave first and what remains is a hardier population. Fit the rate on
   period 1 and you understate the tail badly; use a late-period rate and
   you extrapolate the survivors' behaviour onto everyone. Both are
   available here, side by side, so the gap between them is visible —
   `ltv_geometric` against `ltv_from_retention`.

3. SURVIVORSHIP IN THE COHORT SELECTION. Measuring 12-month LTV requires
   12 months of history, so the estimate can only use cohorts acquired at
   least a year ago — a different acquisition mix, different pricing,
   different product. The number is real and it is about the past. Nothing
   here can fix that; it can only be labelled, and `warnings` labels it.

4. NO INTERVAL. ARPU on heavy-tailed revenue is dominated by a few whales,
   so the mean is unstable in a way its three decimal places conceal. Every
   figure here carries a bootstrap interval, and on real data those
   intervals are embarrassingly wide. That is information, not a defect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "UnitEconomics", "arpu", "retention_from_activity", "expected_lifetime",
    "ltv_from_retention", "ltv_geometric", "payback_periods", "compute",
    "METRIC_DICTIONARY",
]

METRIC_DICTIONARY = {
    "arpu": "total revenue in the window / all users in the cohort "
            "(payers and non-payers)",
    "arppu": "total revenue in the window / users with revenue > 0",
    "conversion_rate": "users with revenue > 0 / all users in the cohort",
    "cac": "acquisition spend in the window / customers acquired by that "
           "spend in the same window; excludes organic",
    "retention_k": "share of the cohort still active in period k, with "
                   "period 0 = 1.0 by construction",
    "expected_lifetime": "sum of retention_k over the stated horizon, in "
                         "periods; not a prediction past the horizon",
    "ltv": "arpu_per_period * sum_k retention_k / (1+d)^k over the stated "
           "horizon, d = discount rate per period",
    "ltv_cac": "ltv / cac; above 1 means the cohort pays back inside the "
               "horizon, which is not the same as being profitable",
    "payback_periods": "smallest k where cumulative discounted revenue per "
                       "acquired user >= cac; None when it never happens "
                       "inside the horizon",
}


@dataclass(frozen=True)
class UnitEconomics:
    """One cohort's economics, with the inputs and doubts attached."""
    n_users: int
    horizon: int
    arpu: float
    arpu_ci: tuple[float, float]
    arppu: float
    conversion_rate: float
    revenue_total: float
    cac: float | None
    ltv: float
    ltv_ci: tuple[float, float]
    ltv_geometric: float | None
    expected_lifetime: float
    ltv_cac: float | None
    payback: int | None
    discount_rate: float
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        lines = [
            f"unit economics over {self.horizon} periods, n = {self.n_users:,}",
            f"  ARPU            {self.arpu:12.4f}  "
            f"[{self.arpu_ci[0]:.4f}, {self.arpu_ci[1]:.4f}] 95%",
            f"  ARPPU           {self.arppu:12.4f}  "
            f"(conversion {self.conversion_rate:.2%})",
            f"  expected life   {self.expected_lifetime:12.4f}  periods "
            f"(sum of the retention curve)",
            f"  LTV             {self.ltv:12.4f}  "
            f"[{self.ltv_ci[0]:.4f}, {self.ltv_ci[1]:.4f}] 95%",
        ]
        if self.ltv_geometric is not None:
            gap = self.ltv_geometric - self.ltv
            lines.append(
                f"  LTV geometric   {self.ltv_geometric:12.4f}  "
                f"({gap:+.4f} vs the curve: this gap is the cost of "
                f"assuming constant churn)"
            )
        if self.cac is not None:
            lines.append(f"  CAC             {self.cac:12.4f}")
            lines.append(
                f"  LTV/CAC         {self.ltv_cac:12.4f}"
                if self.ltv_cac is not None else "  LTV/CAC         n/a"
            )
            lines.append(
                f"  payback         {self.payback:12d}  periods"
                if self.payback is not None
                else f"  payback         {'never':>12s}  inside the horizon"
            )
        if self.discount_rate:
            lines.append(f"  discount rate   {self.discount_rate:12.4%} per period")
        if self.assumptions:
            lines.append("  assumes:")
            lines += [f"    - {a}" for a in self.assumptions]
        if self.warnings:
            lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _clean(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    if (arr < 0).any():
        raise ValueError(f"{name} contains negative revenue; handle refunds explicitly")
    return arr


def arpu(
    revenue_per_user: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Mean revenue per user and a percentile bootstrap interval for it.

    Bootstrapped rather than t-based because revenue is heavy-tailed: a
    handful of users supply most of it, the sample mean inherits their
    variance, and at realistic cohort sizes the normal approximation for
    the mean is still arriving. The bootstrap makes no shape assumption.
    """
    rev = _clean(revenue_per_user, "revenue_per_user")
    mean = float(rev.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, rev.size, size=(n_boot, rev.size))
    boots = rev[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, (float(lo), float(hi))


def retention_from_activity(active_matrix: np.ndarray) -> np.ndarray:
    """Retention curve from a users x periods boolean activity matrix.

    Row = user, column = period, True = active in that period. Returns the
    share of the cohort active in each period, with period 0 forced to 1.0
    if everyone in the cohort was active at acquisition.

    Note what this does NOT do: it does not require activity to be
    contiguous. A user active in periods 0, 1 and 5 counts in all three.
    That is "active in period k", which is what a retention curve means.
    If your definition is "has not yet churned", compute that upstream —
    the two differ, and which one a dashboard uses is usually
    undocumented.
    """
    mat = np.asarray(active_matrix)
    if mat.ndim != 2:
        raise ValueError(f"expected a 2-D users x periods matrix, got {mat.shape}")
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError("activity matrix is empty")
    return mat.astype(float).mean(axis=0)


def expected_lifetime(retention: Sequence[float] | np.ndarray) -> float:
    """Sum of the retention curve: expected periods active inside the horizon.

    Not "lifetime". The sum of a curve you observed for 12 periods is an
    expectation over 12 periods, and anyone who extends it is forecasting.
    """
    r = np.asarray(retention, dtype=float)
    if r.size == 0:
        raise ValueError("retention curve is empty")
    if (r < 0).any() or (r > 1).any():
        raise ValueError("retention values must lie in [0, 1]")
    if np.any(np.diff(r) > 1e-9):
        # Retention can only fall. A rise means resurrection is being
        # counted, or the cohort definition changes between periods.
        raise ValueError(
            "retention curve increases somewhere; a cohort cannot grow. "
            "Check for resurrected users or a shifting cohort definition."
        )
    return float(r.sum())


def ltv_from_retention(
    arpu_per_period: float,
    retention: Sequence[float] | np.ndarray,
    *,
    discount_rate: float = 0.0,
) -> float:
    """LTV as revenue per active period, summed along the observed curve.

        LTV = arpu_per_period * sum_k retention_k / (1 + d)^k

    `arpu_per_period` is revenue per user per period among ACTIVE users of
    that period, so that multiplying by retention_k gives the cohort's
    revenue in period k. Passing a whole-window ARPU here double-counts
    the retention decay, which is a quiet way to halve or double the
    answer depending on which way you got it wrong.
    """
    r = np.asarray(retention, dtype=float)
    if discount_rate < 0:
        raise ValueError(f"discount_rate cannot be negative, got {discount_rate}")
    periods = np.arange(r.size, dtype=float)
    discount = (1.0 + discount_rate) ** periods
    return float(arpu_per_period * (r / discount).sum())


def ltv_geometric(arpu_per_period: float, churn_rate: float) -> float:
    """LTV under a constant hazard: arpu / churn.

    Kept so the gap against `ltv_from_retention` can be shown rather than
    argued about. Valid only if the retention curve really is geometric,
    which it is not: hazard declines as the churn-prone leave first.
    """
    if not 0.0 < churn_rate <= 1.0:
        raise ValueError(f"churn_rate must be in (0,1], got {churn_rate}")
    return float(arpu_per_period / churn_rate)


def payback_periods(
    cac: float,
    arpu_per_period: float,
    retention: Sequence[float] | np.ndarray,
    *,
    discount_rate: float = 0.0,
) -> int | None:
    """First period where cumulative discounted revenue per user covers CAC.

    Returns None when it does not happen inside the observed curve, which
    is an answer and should be reported as one rather than extrapolated
    into a number that looks better.
    """
    r = np.asarray(retention, dtype=float)
    cumulative = 0.0
    for k in range(r.size):
        cumulative += arpu_per_period * r[k] / ((1.0 + discount_rate) ** k)
        if cumulative >= cac:
            return k
    return None


def compute(
    revenue_per_user: Sequence[float] | np.ndarray,
    *,
    retention: Sequence[float] | np.ndarray | None = None,
    acquisition_spend: float | None = None,
    customers_acquired: int | None = None,
    horizon: int | None = None,
    discount_rate: float = 0.0,
    n_boot: int = 10_000,
    seed: int = 0,
) -> UnitEconomics:
    """Compute the whole set from one cohort's revenue, with intervals.

    revenue_per_user   total revenue in the window, one entry per acquired
                       user, zeros included
    retention          share active per period; defaults to a flat curve of
                       length 1, i.e. "we measured one period and claim
                       nothing about the future"
    acquisition_spend  spend that bought this cohort
    customers_acquired denominator for CAC; defaults to len(revenue_per_user)
    horizon            periods the figures cover; defaults to len(retention)

    The LTV interval comes from bootstrapping the users, so it carries the
    uncertainty in ARPU but treats the retention curve as known. That is
    the honest decomposition: the revenue side is sampled, the curve is
    usually taken from a different and larger population. If the curve is
    also noisy, the real interval is wider than the one printed, and the
    warning says so.
    """
    rev = _clean(revenue_per_user, "revenue_per_user")
    n = rev.size
    r = (np.ones(1, dtype=float) if retention is None
         else np.asarray(retention, dtype=float))
    life = expected_lifetime(r)
    h = int(horizon if horizon is not None else r.size)

    arpu_value, arpu_ci = arpu(rev, n_boot=n_boot, seed=seed)
    payers = rev[rev > 0]
    conversion = float(payers.size / n)
    arppu = float(payers.mean()) if payers.size else 0.0

    # Revenue per ACTIVE period, so that multiplying by retention does not
    # apply the decay twice.
    arpu_per_period = arpu_value / life if life > 0 else 0.0
    ltv = ltv_from_retention(arpu_per_period, r, discount_rate=discount_rate)

    # Bootstrap the users; hold the curve fixed.
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_arpu = rev[idx].mean(axis=1)
    boot_ltv = np.array([
        ltv_from_retention(a / life if life > 0 else 0.0, r,
                           discount_rate=discount_rate)
        for a in boot_arpu
    ])
    ltv_lo, ltv_hi = np.quantile(boot_ltv, [0.025, 0.975])

    geometric = None
    if r.size >= 2 and r[-1] > 0:
        # Churn implied by the last observed step, the rate a dashboard
        # would most likely use.
        last_churn = float(1.0 - r[-1] / r[-2]) if r[-2] > 0 else 1.0
        if 0.0 < last_churn <= 1.0:
            geometric = ltv_geometric(arpu_per_period, last_churn)

    cac_value = None
    ltv_cac = None
    payback = None
    if acquisition_spend is not None:
        denominator = customers_acquired if customers_acquired is not None else n
        if denominator <= 0:
            raise ValueError("customers_acquired must be positive")
        cac_value = float(acquisition_spend / denominator)
        if cac_value > 0:
            ltv_cac = float(ltv / cac_value)
            payback = payback_periods(cac_value, arpu_per_period, r,
                                      discount_rate=discount_rate)

    assumptions = [
        f"horizon of {h} period(s); nothing is claimed beyond it",
        "ARPU is per acquired user, zeros included",
        f"retention curve of length {r.size}, treated as known",
    ]
    if discount_rate == 0:
        assumptions.append("no discounting (d = 0): future revenue counted at face value")

    warnings: list[str] = []
    width = arpu_ci[1] - arpu_ci[0]
    if arpu_value > 0 and width / arpu_value > 0.4:
        warnings.append(
            f"the ARPU interval spans {width / arpu_value:.0%} of ARPU itself: "
            "this cohort is too small or too heavy-tailed for a point estimate "
            "to be quoted without the range"
        )
    top_share = float(np.sort(rev)[::-1][: max(1, n // 100)].sum() / rev.sum()) if rev.sum() > 0 else 0.0
    if top_share > 0.5:
        warnings.append(
            f"the top 1% of users supply {top_share:.0%} of revenue; ARPU is a "
            "statement about those users, not about a typical one"
        )
    if r.size == 1:
        warnings.append(
            "no retention curve supplied, so LTV equals one period of ARPU. "
            "This is not a lifetime value and should not be compared to CAC."
        )
    if customers_acquired is None and acquisition_spend is not None:
        warnings.append(
            "CAC used every user in the cohort as its denominator; if any "
            "arrived organically, CAC is understated"
        )
    if geometric is not None and ltv > 0 and abs(geometric - ltv) / ltv > 0.25:
        warnings.append(
            f"the constant-churn shortcut gives {geometric:.2f} against "
            f"{ltv:.2f} from the curve, a {abs(geometric - ltv) / ltv:.0%} gap: "
            "do not let a dashboard quote the shortcut"
        )
    return UnitEconomics(
        n_users=n, horizon=h, arpu=arpu_value, arpu_ci=arpu_ci, arppu=arppu,
        conversion_rate=conversion, revenue_total=float(rev.sum()),
        cac=cac_value, ltv=ltv, ltv_ci=(float(ltv_lo), float(ltv_hi)),
        ltv_geometric=geometric, expected_lifetime=life, ltv_cac=ltv_cac,
        payback=payback, discount_rate=discount_rate,
        assumptions=tuple(assumptions), warnings=tuple(warnings),
    )
