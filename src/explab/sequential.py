#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Looking at the result early, without lying about it.

WHY PEEKING BREAKS A TEST
-------------------------
A p-value of 0.05 means: if nothing is going on, a statistic this extreme
turns up 5% of the time. That promise is about ONE look at ONE dataset of a
pre-declared size.

Watch the p-value every day for a month and stop the moment it dips below
0.05, and you are no longer running that procedure. You are running "reject
if the random walk ever crosses the line", and a random walk crosses any
fixed line much more often than it ends up past it. Five looks takes the
false positive rate from 5% to about 14%; twenty daily looks takes it past
25%. The number is measured, not asserted: proofs/proof_peeking.py runs
the simulation and prints the table.

Nobody peeks on purpose. It happens because the dashboard is right there,
because a stakeholder asks on Tuesday, and because stopping early when you
are winning feels like diligence rather than a methodological choice. The
cure is not discipline. The cure is to design for looking.

TWO HONEST WAYS TO LOOK EARLY
-----------------------------
1. GROUP SEQUENTIAL BOUNDARIES (this module). Decide in advance how many
   looks you will take, then raise the bar at each look so the total false
   positive rate across all of them comes out at alpha. You get to stop
   early; you pay a slightly higher bar at the final look than a
   single-look test would have had.

2. ALWAYS-VALID / ANYTIME-VALID INFERENCE (not implemented here). Confidence
   sequences that hold at every sample size simultaneously, so you may look
   continuously and even choose when to stop based on what you see. More
   flexible and more expensive in sample size. Deliberately out of scope —
   see docs/19-what-this-does-not-cover.md.

HOW THE BOUNDARIES ARE OBTAINED HERE
------------------------------------
The classical route is Lan-DeMets alpha spending, which needs recursive
numerical integration over the joint distribution of the interim
statistics. That integration is where a hand-rolled implementation goes
quietly wrong.

This module takes the other route: pick a boundary SHAPE, then calibrate
its one free constant by Monte Carlo so that the measured false positive
rate equals alpha. The shape comes from the literature; the constant comes
from simulation; and the result is checked by an independent simulation in
proofs/proof_sequential.py, which must come out within Monte Carlo error
of alpha or the build fails.

This is the project's thesis applied to itself: if you cannot derive a
number, you may still earn it, as long as you then measure it.

THE SIMULATION MODEL
Under the null, with information accruing in equal increments, the
cumulative test statistic behaves like a Brownian motion observed at K
equally spaced times. Concretely: draw K independent standard normals as
the per-look increments, take their cumulative sum S_k, and the z-statistic
at look k is S_k / sqrt(k). That is all a group-sequential null looks like.
Unequal spacing is supported through `information_fractions`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

__all__ = ["SequentialPlan", "calibrate_boundary", "naive_peeking_alpha"]

Shape = Literal["pocock", "obrien_fleming"]


@dataclass(frozen=True)
class SequentialPlan:
    """Critical values for a pre-declared set of interim looks.

    boundaries            |z| required to stop at each look
    information_fractions share of the planned sample at each look
    alpha                 total two-sided false positive rate targeted
    shape                 which boundary family
    measured_alpha        the rate the calibration simulation produced
    mc_standard_error     Monte Carlo error on `measured_alpha`
    n_sims                simulations used to calibrate
    """
    boundaries: tuple[float, ...]
    information_fractions: tuple[float, ...]
    alpha: float
    shape: str
    measured_alpha: float
    mc_standard_error: float
    n_sims: int

    @property
    def n_looks(self) -> int:
        return len(self.boundaries)

    def decide(self, z_values: Sequence[float]) -> dict:
        """Given the z-statistics observed so far, say whether to stop.

        Pass one z per look taken, in order. Returns the look number that
        crossed (1-based) or None, plus the boundary it had to beat.
        """
        z = list(z_values)
        if len(z) > self.n_looks:
            raise ValueError(
                f"{len(z)} looks supplied but the plan declared {self.n_looks}; "
                "an undeclared extra look voids the guarantee"
            )
        for i, value in enumerate(z):
            if abs(value) >= self.boundaries[i]:
                return {
                    "stop": True, "look": i + 1, "z": value,
                    "boundary": self.boundaries[i],
                    "reason": f"|z| {abs(value):.3f} >= boundary "
                              f"{self.boundaries[i]:.3f} at look {i + 1}",
                }
        return {
            "stop": False, "look": len(z), "z": z[-1] if z else None,
            "boundary": self.boundaries[len(z) - 1] if z else None,
            "reason": "no boundary crossed; keep collecting",
        }

    def __str__(self) -> str:
        lines = [
            f"{self.shape} boundaries, {self.n_looks} looks, "
            f"two-sided alpha {self.alpha:g}",
            f"{'look':>5s} {'information':>12s} {'|z| needed':>11s} "
            f"{'~one-look p':>12s}",
        ]
        from .statlib import norm_cdf
        for i, (frac, b) in enumerate(
            zip(self.information_fractions, self.boundaries), start=1
        ):
            nominal = 2.0 * (1.0 - norm_cdf(b))
            lines.append(f"{i:5d} {frac:11.0%} {b:11.3f} {nominal:12.4g}")
        lines.append(
            f"calibrated on {self.n_sims:,} simulations: measured alpha "
            f"{self.measured_alpha:.4f} +/- {self.mc_standard_error:.4f}"
        )
        return "\n".join(lines)


def _shape_weights(shape: Shape, fractions: np.ndarray) -> np.ndarray:
    """Relative height of the boundary at each look, before scaling.

    pocock           flat. Same |z| at every look. Spends alpha evenly, so
                     it stops early most often and pays the most at the
                     final look.
    obrien_fleming   proportional to 1/sqrt(information). Very high early,
                     approaching the single-look critical value at the end.
                     Spends almost nothing early, so an early stop means
                     something dramatic, and the final look is barely
                     penalised. The usual default when the experiment will
                     probably run to completion anyway.
    """
    if shape == "pocock":
        return np.ones_like(fractions)
    if shape == "obrien_fleming":
        return 1.0 / np.sqrt(fractions)
    raise ValueError(f"unknown shape {shape!r}; use 'pocock' or 'obrien_fleming'")


def calibrate_boundary(
    n_looks: int,
    *,
    alpha: float = 0.05,
    shape: Shape = "obrien_fleming",
    information_fractions: Sequence[float] | None = None,
    n_sims: int = 200_000,
    seed: int = 20261115,
) -> SequentialPlan:
    """Find the boundary constant that makes the total error rate equal alpha.

    n_looks                how many interim analyses, including the final one
    alpha                  total two-sided false positive rate across all looks
    shape                  boundary family, see _shape_weights
    information_fractions  share of planned sample at each look; defaults to
                           equally spaced (1/K, 2/K, ..., 1)
    n_sims                 simulations per calibration step

    Bisects on a single scale factor c, with boundaries c * weight(look),
    until the simulated crossing rate hits alpha. One set of simulated
    paths is reused across every candidate c, which makes the search
    monotone in c and therefore safe to bisect: with common random
    numbers, raising the boundary can only ever reduce the crossing rate.
    """
    if n_looks < 1:
        raise ValueError(f"n_looks must be >= 1, got {n_looks}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if n_sims < 10_000:
        raise ValueError(
            f"n_sims={n_sims} gives a Monte Carlo error around "
            f"{np.sqrt(alpha * (1 - alpha) / n_sims):.4f}, too coarse to "
            "calibrate a boundary; use >= 10,000"
        )

    if information_fractions is None:
        fractions = np.arange(1, n_looks + 1, dtype=float) / n_looks
    else:
        fractions = np.asarray(list(information_fractions), dtype=float)
        if fractions.size != n_looks:
            raise ValueError(
                f"{fractions.size} information fractions for {n_looks} looks"
            )
        if np.any(np.diff(fractions) <= 0):
            raise ValueError("information_fractions must be strictly increasing")
        if fractions[0] <= 0 or fractions[-1] > 1.0:
            raise ValueError("information_fractions must lie in (0, 1]")

    rng = np.random.default_rng(seed)
    # Brownian motion under the null, observed at the given information
    # times. Increment variance is proportional to the information gained,
    # so draw N(0, delta_t) increments and divide the cumulative sum by
    # sqrt(t) to get the standardised statistic at each look.
    deltas = np.diff(np.concatenate([[0.0], fractions]))
    increments = rng.standard_normal((n_sims, n_looks)) * np.sqrt(deltas)
    brownian = np.cumsum(increments, axis=1)
    z_paths = np.abs(brownian / np.sqrt(fractions))

    weights = _shape_weights(shape, fractions)

    def crossing_rate(scale: float) -> float:
        return float((z_paths >= scale * weights).any(axis=1).mean())

    lo, hi = 0.1, 20.0
    if crossing_rate(hi) > alpha:
        raise RuntimeError("could not bracket the boundary; alpha too small")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if crossing_rate(mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    scale = 0.5 * (lo + hi)
    boundaries = scale * weights
    measured = crossing_rate(scale)
    mc_se = float(np.sqrt(measured * (1.0 - measured) / n_sims))
    return SequentialPlan(
        boundaries=tuple(float(b) for b in boundaries),
        information_fractions=tuple(float(f) for f in fractions),
        alpha=alpha, shape=shape, measured_alpha=measured,
        mc_standard_error=mc_se, n_sims=n_sims,
    )


def naive_peeking_alpha(
    n_looks: int,
    *,
    alpha: float = 0.05,
    information_fractions: Sequence[float] | None = None,
    n_sims: int = 200_000,
    seed: int = 7,
) -> float:
    """The real false positive rate when you peek `n_looks` times at alpha.

    Same simulation machinery, no boundary adjustment: the fixed critical
    value is applied at every look and we count how often any look crosses
    it. This is the number to show a stakeholder who wants to call the
    experiment on Wednesday.
    """
    from .statlib import norm_ppf

    if information_fractions is None:
        fractions = np.arange(1, n_looks + 1, dtype=float) / n_looks
    else:
        fractions = np.asarray(list(information_fractions), dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.diff(np.concatenate([[0.0], fractions]))
    increments = rng.standard_normal((n_sims, fractions.size)) * np.sqrt(deltas)
    z_paths = np.abs(np.cumsum(increments, axis=1) / np.sqrt(fractions))
    crit = norm_ppf(1.0 - alpha / 2.0)
    return float((z_paths >= crit).any(axis=1).mean())
