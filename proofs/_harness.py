#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared machinery for the proofs: state a claim, measure it, judge it.

A "proof" here is a simulation study. Each one states a property the
library claims, generates data where the truth is known, measures whether
the property holds, and fails the build when it does not.

THE TOLERANCE PROBLEM
---------------------
A simulated false-positive rate will never equal 0.0500 exactly, so every
check needs a tolerance, and a tolerance chosen by eye is a tolerance
chosen to pass. These helpers derive it instead.

For a measured RATE from n independent simulations, the Monte Carlo
standard error is sqrt(p(1-p)/n). `check_rate` allows `k` of those (3 by
default, about a 99.7% interval under the normal approximation). That
makes the tolerance a function of how much simulation you paid for: want
a tighter claim, run more sims. It also means a proof cannot be made to
pass by loosening a number — only by running more of them, or by fixing
the code.

For a measured MEAN, the same idea with the standard error of the mean.

WHY `expected` IS ALWAYS WRITTEN OUT
------------------------------------
Every check names the number it expects and where that number comes from:
alpha for a false positive rate, rho^2 for a variance reduction, an
analytic power formula for a power curve. A simulation that only reports
what happened proves nothing; it is the comparison against an
independently derived target that makes it evidence.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make `explab` and the data generator importable when a proof is run
# directly from the repository root or from inside proofs/.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Claim:
    """One checkable statement."""
    label: str
    measured: float
    expected: float
    tolerance: float
    passed: bool
    detail: str = ""

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"  [{mark}] {self.label}\n"
            f"         measured {self.measured:.6g}   expected {self.expected:.6g}"
            f"   tolerance +/-{self.tolerance:.3g}"
            + (f"\n         {self.detail}" if self.detail else "")
        )


@dataclass
class Proof:
    """A named set of claims, with a header and a verdict."""
    title: str
    claim_list: list[Claim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  {text}")

    def check_rate(
        self, label: str, successes: int, trials: int, expected: float,
        *, k: float = 3.0, detail: str = "",
    ) -> Claim:
        """Check a simulated rate against its target, within k Monte Carlo SEs."""
        if trials <= 0:
            raise ValueError("trials must be positive")
        measured = successes / trials
        se = math.sqrt(max(expected * (1.0 - expected), 1e-12) / trials)
        tol = k * se
        claim = Claim(
            label=label, measured=measured, expected=expected, tolerance=tol,
            passed=abs(measured - expected) <= tol,
            detail=detail or f"{successes:,} of {trials:,} simulations",
        )
        self.claim_list.append(claim)
        print(claim.line())
        return claim

    def check_mean(
        self, label: str, values, expected: float, *, k: float = 3.0,
        detail: str = "",
    ) -> Claim:
        """Check a simulated mean against its target, within k standard errors."""
        import numpy as np

        arr = np.asarray(values, dtype=float)
        n = arr.size
        if n < 2:
            raise ValueError("need at least 2 values")
        measured = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(n))
        tol = k * se
        claim = Claim(
            label=label, measured=measured, expected=expected, tolerance=tol,
            passed=abs(measured - expected) <= tol,
            detail=detail or f"mean of {n:,} values, se {se:.4g}",
        )
        self.claim_list.append(claim)
        print(claim.line())
        return claim

    def check_close(
        self, label: str, measured: float, expected: float, *,
        tolerance: float, detail: str = "",
    ) -> Claim:
        """Check against an explicit tolerance, for non-sampling comparisons."""
        claim = Claim(
            label=label, measured=measured, expected=expected,
            tolerance=tolerance,
            passed=abs(measured - expected) <= tolerance, detail=detail,
        )
        self.claim_list.append(claim)
        print(claim.line())
        return claim

    def check_true(self, label: str, condition: bool, *, detail: str = "") -> Claim:
        """Check a yes/no property."""
        claim = Claim(
            label=label, measured=float(bool(condition)), expected=1.0,
            tolerance=0.0, passed=bool(condition), detail=detail,
        )
        self.claim_list.append(claim)
        print(claim.line())
        return claim

    @property
    def failed(self) -> list[Claim]:
        return [c for c in self.claim_list if not c.passed]

    def finish(self) -> int:
        """Print the verdict and return a process exit code."""
        elapsed = time.perf_counter() - self._started
        n = len(self.claim_list)
        bad = self.failed
        print()
        if bad:
            print(f"  {len(bad)} of {n} claims FAILED in {elapsed:.1f}s")
            for c in bad:
                print(f"    - {c.label}")
            return 1
        print(f"  all {n} claims hold ({elapsed:.1f}s)")
        return 0


def header(title: str, claim: str) -> Proof:
    """Open a proof: print the title and the property being tested."""
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"CLAIM: {claim}")
    print()
    return Proof(title=title)
