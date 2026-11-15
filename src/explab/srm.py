#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample ratio mismatch: the check that runs before you read the result.

WHAT IT IS
----------
You asked for a 50/50 split. You got 49.6/50.4. Is that fine?

On 1,000 users: fine, that is a coin landing 496 heads.
On 2,000,000 users: not fine. That is roughly an 11-sigma deviation, and
the probability of it happening by chance is smaller than the probability
of your analysis being correct about anything.

A sample ratio mismatch means units did not reach the arms the way the
design said they would. Once that is true, every effect estimate in the
readout is suspect, because the two groups are no longer exchangeable:
whatever mechanism dropped or duplicated units may well correlate with the
outcome. Common causes, in rough order of how often they turn out to be
the culprit:

  - a redirect or bot filter applied to one arm and not the other
  - logging that fires after the feature loads, so the slower arm loses
    users who bounce first (this one biases toward the treatment looking
    good, and is the reason "treatment is winning" should make you check
    SRM harder, not less)
  - a cached assignment that reshuffles on session boundaries
  - deduplication on a key that exists in one arm only
  - an analyst filtering the dataset after assignment (the most common of
    all, and the easiest to fix)

WHY THE THRESHOLD IS NOT 0.05
-----------------------------
This test runs on every experiment and every metric readout, so it is a
monitor rather than a hypothesis. At alpha = 0.05 one experiment in twenty
would raise a false alarm and the team would learn to ignore it. The
industry convention is p < 0.001, which this module defaults to: rare
enough to be believed, sensitive enough that a real break of more than a
fraction of a percent shows up quickly at scale.

That tightness has a direct consequence worth naming: SRM is a test with
enormous power at large n and very little at small n. On 1,000 users per
arm you will not detect a 1% assignment bias and you should not pretend
otherwise. `srm_check` reports the smallest imbalance it could have caught,
so a pass is readable as "no imbalance above X" rather than "no imbalance".
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .statlib import chi2_cdf, norm_ppf

__all__ = ["SRMResult", "srm_check"]


@dataclass(frozen=True)
class SRMResult:
    """The outcome of one sample-ratio check.

    counts         observed units per arm, in the order given
    expected       what the declared split implies, same order
    chi_square     the test statistic
    df             arms - 1
    p_value        probability of a deviation at least this large by chance
    threshold      the alpha this was judged against
    passed         True when p_value >= threshold
    worst_arm      the arm furthest from its expectation
    observed_share the share actually seen in `worst_arm`
    expected_share the share declared for `worst_arm`
    detectable     smallest absolute share deviation this n could have
                   caught at 80% power and this threshold (two arms only)
    """
    counts: tuple[int, ...]
    expected: tuple[float, ...]
    chi_square: float
    df: int
    p_value: float
    threshold: float
    passed: bool
    worst_arm: str
    observed_share: float
    expected_share: float
    detectable: float | None

    def __str__(self) -> str:
        head = "PASS" if self.passed else "FAIL - SRM"
        lines = [
            f"sample ratio check: {head}  (p = {self.p_value:.3g}, "
            f"threshold {self.threshold:g})",
            f"  chi-square {self.chi_square:.4g} on {self.df} df, "
            f"n = {sum(self.counts):,}",
            f"  worst arm {self.worst_arm}: observed {self.observed_share:.4%} "
            f"vs expected {self.expected_share:.4%}",
        ]
        if self.detectable is not None:
            lines.append(
                f"  this sample could detect a share deviation of "
                f"{self.detectable:.4%} or more at 80% power"
            )
        if not self.passed:
            lines.append(
                "  do not read the effect estimates until this is explained: "
                "the arms are not exchangeable"
            )
        return "\n".join(lines)


def srm_check(
    counts: dict[str, int],
    expected_shares: dict[str, float] | None = None,
    *,
    threshold: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test on the units that landed in each arm.

    counts           {"control": 100432, "treatment": 99781}
    expected_shares  {"control": 0.5, "treatment": 0.5}; defaults to an
                     even split across whatever arms appear in `counts`
    threshold        alpha. Default 0.001, see the module docstring.

    THE ARITHMETIC

        chi2 = sum over arms of (observed - expected)^2 / expected
        df   = number of arms - 1

    df is arms-1, not arms, because the total is fixed: once you know how
    many units landed in every arm but the last, the last is determined.
    One constraint, one degree of freedom gone.

    `expected` is the declared share times the OBSERVED total. The test
    asks whether the units you logged were split as designed, not whether
    you logged as many units as you expected to — that second question is
    a traffic question, and conflating the two is how an SRM alert gets
    dismissed as "we just had a quiet weekend".
    """
    if len(counts) < 2:
        raise ValueError(f"need at least 2 arms, got {list(counts)}")
    if any(v < 0 for v in counts.values()):
        raise ValueError(f"counts cannot be negative: {counts}")
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no units in any arm")

    arms = list(counts)
    if expected_shares is None:
        expected_shares = {arm: 1.0 / len(arms) for arm in arms}
    missing = set(arms) - set(expected_shares)
    if missing:
        raise ValueError(f"no expected share given for arm(s) {sorted(missing)}")
    extra = set(expected_shares) - set(arms)
    if extra:
        raise ValueError(
            f"expected shares name arm(s) with no counts: {sorted(extra)}; "
            "an arm that received zero units is itself an SRM, pass it as 0"
        )
    share_sum = sum(expected_shares[a] for a in arms)
    if abs(share_sum - 1.0) > 1e-9:
        raise ValueError(f"expected shares sum to {share_sum}, not 1")

    expected = tuple(expected_shares[a] * total for a in arms)
    if min(expected) < 5:
        # The chi-square approximation needs a few units in every cell.
        raise ValueError(
            f"expected count in the smallest arm is {min(expected):.2f} (<5); "
            "the chi-square approximation does not hold, use an exact test"
        )

    chi_square = sum(
        (counts[a] - e) ** 2 / e for a, e in zip(arms, expected)
    )
    df = len(arms) - 1
    p_value = 1.0 - chi2_cdf(chi_square, df)

    # The arm furthest from expectation, measured in standardised units so
    # a big arm and a small arm are compared fairly.
    worst = max(
        arms,
        key=lambda a: abs(counts[a] - expected_shares[a] * total)
        / math.sqrt(max(expected_shares[a] * total, 1e-12)),
    )

    detectable: float | None = None
    if len(arms) == 2:
        # Two-arm case: the count in one arm is Binomial(total, share). The
        # smallest share deviation detectable at 80% power and this
        # threshold is (z_alpha + z_power) * sqrt(p(1-p)/total).
        p0 = expected_shares[arms[0]]
        se = math.sqrt(p0 * (1.0 - p0) / total)
        detectable = float((norm_ppf(1.0 - threshold / 2.0) + norm_ppf(0.80)) * se)

    return SRMResult(
        counts=tuple(counts[a] for a in arms),
        expected=expected,
        chi_square=float(chi_square),
        df=df,
        p_value=float(p_value),
        threshold=threshold,
        passed=bool(p_value >= threshold),
        worst_arm=worst,
        observed_share=counts[worst] / total,
        expected_share=expected_shares[worst],
        detectable=detectable,
    )
