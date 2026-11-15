#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""explab - experiment analysis where every number can be derived or measured.

The package is organised so you can read it in the order the work happens:

    power        before the experiment: how many units, for what effect
    srm          first thing after it: did assignment work
    inference    the comparison itself: means, rates, ratios, bootstrap
    cuped        variance reduction with pre-experiment data
    sequential   looking early without inflating the error rate
    multiplicity many metrics, many segments, one error budget
    segments     heterogeneous effects, with the guards that make them
                 readable
    unit_economics  ARPU, LTV, CAC, payback - with intervals
    analyze      all of it, in order, into one decision
    report       that decision, rendered for a person
    statlib      the distribution functions the rest is built on

Nothing here imports SciPy. The distributions are implemented in statlib
and cross-checked against SciPy in tests/, so the library runs on numpy
alone and still has an independent reference behind every number.

Quickstart
----------
    import pandas as pd
    from explab.analyze import MetricSpec, analyze
    from explab.report import render_text

    frame = pd.read_csv("data/generated/experiment_seed1_n40000.csv")
    metrics = [
        MetricSpec("conversion", "binary", column="converted",
                   role="primary", practical_threshold=0.004),
        MetricSpec("minutes", "continuous", column="minutes",
                   covariate="pre_minutes", role="secondary"),
        MetricSpec("ctr", "ratio", numerator="clicks",
                   denominator="impressions", role="secondary"),
    ]
    print(render_text(analyze(frame, metrics)))
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "statlib", "inference", "power", "srm", "cuped", "multiplicity",
    "sequential", "segments", "unit_economics", "analyze", "report",
]
