#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The distribution functions this project needs, written from scratch.

WHY THIS FILE EXISTS
--------------------
Every other module in `explab` needs four numbers:

    norm_cdf / norm_ppf     the standard normal, for z-tests and power
    t_cdf    / t_ppf        Student's t, for small-sample mean tests
    chi2_cdf                chi-square, for sample-ratio mismatch
    (plus the two special functions those are built on)

SciPy has all of them. We implement them anyway, for three reasons:

1. You cannot defend a p-value you cannot derive. If a client asks "where does
   0.034 come from", the answer has to go all the way down, not stop at
   `scipy.stats`.
2. The library then depends on numpy alone, so it runs anywhere.
3. It gives us an independent reference to check against. `tests/` compares
   every function here to SciPy and fails on a disagreement past 1e-10. Two
   implementations that agree to ten decimals are rarely both wrong in the
   same direction.

HOW TO READ IT
--------------
Work bottom-up. `log_gamma`, `_betacf` and `_gamma_series` are machinery.
`betainc` and `gammainc_lower` are the two special functions. Everything
public is a thin wrapper over those. The inverses (`*_ppf`) are solved
numerically by bisection rather than by a rational approximation: slower,
but there is no table of magic constants to get wrong, and the error is
bounded by the tolerance you pass in.

NOTATION
--------
`a`, `b`, `x` follow the standard notation for the incomplete beta and gamma
functions so the code can be read next to a textbook. Elsewhere names are
spelled out.
"""
from __future__ import annotations

import math

__all__ = [
    "log_gamma", "betainc", "gammainc_lower",
    "norm_pdf", "norm_cdf", "norm_sf", "norm_ppf",
    "t_cdf", "t_sf", "t_ppf", "chi2_cdf",
]

# Smallest number we let a continued-fraction denominator become, so a zero
# denominator cannot produce an inf. Taken from Numerical Recipes.
_TINY = 1e-300
# Relative tolerance for the continued fractions. Double precision gives us
# about 2.2e-16; asking for much less than that just burns iterations.
_EPS = 3e-16
_MAX_ITER = 300
# Where t_sf stops using the tail parametrisation and switches to the
# central one. The tail form's argument is df/(df + t^2), which is
# 1 - t^2/(df + t^2); once that distance from 1.0 falls below this, the
# argument rounds to exactly 1.0 and the tail form goes flat. Doubles run
# out at about 1.1e-16, so 1e-12 leaves four orders of magnitude of margin
# while still sitting deep inside the region where the central form is the
# more accurate of the two anyway. See t_sf for the full argument.
_CENTRAL_SWITCH = 1e-12


# --------------------------------------------------------------------------
# machinery
# --------------------------------------------------------------------------
def log_gamma(x: float) -> float:
    """log(Gamma(x)) for x > 0.

    Python's stdlib already has this (`math.lgamma`), and unlike the
    incomplete functions below there is nothing instructive about
    re-deriving Lanczos coefficients, so we delegate. It is wrapped rather
    than used directly only so that every special function in the project
    is reached through one named door.
    """
    return math.lgamma(x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function.

    This is Lentz's algorithm: evaluate a continued fraction from the top
    down, carrying a running numerator `c` and denominator `d`, and stop when
    a new term changes the product by less than _EPS. Each loop iteration
    applies two terms of the fraction (the even and the odd one), which is
    why `aa` is computed twice.

    Only valid for 0 <= x <= 1. Callers arrange that.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise RuntimeError(f"_betacf did not converge for a={a}, b={b}, x={x}")


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b).

    I_x(a,b) is the share of a Beta(a,b) distribution lying below x, so it
    runs from 0 at x=0 to 1 at x=1. Student's t CDF is a reparametrisation
    of it, which is the only reason we need it.

    The continued fraction converges quickly only on one side of the
    distribution's centre of mass, so for large x we evaluate the mirror
    image I_{1-x}(b,a) and subtract. That is the `else` branch.
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"betainc needs 0 <= x <= 1, got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"betainc needs a, b > 0, got a={a}, b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    # log of the leading factor x^a (1-x)^b / (a B(a,b)), kept in logs
    # because x^a underflows for the sample sizes we care about.
    front = math.exp(
        log_gamma(a + b) - log_gamma(a) - log_gamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _gamma_series(a: float, x: float) -> float:
    """Series expansion of the regularised lower incomplete gamma P(a, x).

    Converges fast when x is small relative to a.
    """
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    else:
        raise RuntimeError(f"_gamma_series did not converge for a={a}, x={x}")
    return total * math.exp(-x + a * math.log(x) - log_gamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Continued fraction for the regularised UPPER incomplete gamma Q(a, x).

    Converges fast when x is large relative to a: the complement of the
    region where the series works. Lentz's algorithm again.
    """
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    else:
        raise RuntimeError(f"_gamma_cf did not converge for a={a}, x={x}")
    return math.exp(-x + a * math.log(x) - log_gamma(a)) * h


def gammainc_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x), in [0, 1].

    Switches between the series and the continued fraction at x = a + 1,
    where each is in its comfortable range. chi2_cdf is a thin wrapper.
    """
    if x < 0.0 or a <= 0.0:
        raise ValueError(f"gammainc_lower needs x >= 0, a > 0; got a={a}, x={x}")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


# --------------------------------------------------------------------------
# the standard normal
# --------------------------------------------------------------------------
def norm_pdf(z: float) -> float:
    """Density of the standard normal at z."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_cdf(z: float) -> float:
    """P(Z <= z) for Z ~ Normal(0, 1).

    The error function is exactly this integral up to a change of variable,
    and `math.erf` is already correct to the last bit, so there is nothing
    to hand-roll here.

    WHY THERE ARE TWO BRANCHES, when `0.5 * (1 + erf(z/sqrt 2))` looks like
    it covers the whole line. For z very negative, erf(z/sqrt 2) approaches
    -1, and `1 + (-0.99999999999999999...)` throws away every significant
    digit the tail had. At z = -8.5 it returns exactly 0.0 when the true
    value is 9.48e-18; at z = -7 it keeps about six digits instead of
    sixteen. That is the same catastrophic cancellation `norm_sf` exists to
    avoid, just at the other end, and it mattered in practice: `norm_ppf`
    bisects on this function, so a left tail with six good digits became a
    quantile with six good digits - an error of 8e-7 at p = 1e-12.

    `erfc` computes the complement directly, so by symmetry the negative
    side is `0.5 * erfc(-z/sqrt 2)` with nothing cancelling. Both branches
    are exact restatements of the same integral; only their floating-point
    behaviour differs.

    Found by tests/test_statlib.py, not by any of the simulations in
    proofs/ - a tail probability off by ten orders of magnitude at z = -8.5
    changes no coverage rate, because no simulation ever lands there.
    """
    if z < 0.0:
        return 0.5 * math.erfc(-z / math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z: float) -> float:
    """P(Z > z), the survival function, computed without cancellation.

    Why this exists separately from `1 - norm_cdf(z)`: at z = 9 the true
    tail is 1.1e-19, but norm_cdf(9) rounds to exactly 1.0, so the
    subtraction gives 0.0 and a p-value of zero appears in a readout. No
    test on finite data has a p-value of zero, and printing one is a
    credibility problem, not a rounding problem.

    `math.erfc` computes the complement directly and stays accurate to
    about 1e-308, which covers |z| up to roughly 37. Everything in this
    project that reports a two-sided p-value uses 2 * norm_sf(|z|).
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _bisect(fn, target: float, lo: float, hi: float, tol: float = 1e-13) -> float:
    """Solve fn(x) == target on a bracket where fn is increasing.

    Plain bisection. `tol` is relative to the size of the root, so the same
    call works for a root near 1e-3 and a root near 1e12. Used for every
    inverse CDF in this module: slower than a rational approximation, with
    no table of magic constants to get wrong, and the only error is the one
    you asked for. Inverse CDFs are called once per test to get a critical
    value, never once per row, so the cost is invisible.
    """
    if fn(lo) - target > 0.0 or fn(hi) - target < 0.0:
        raise ValueError(f"target {target} not bracketed by [{lo}, {hi}]")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if fn(mid) - target < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(lo), abs(hi)):
            break
    return 0.5 * (lo + hi)


def norm_ppf(p: float) -> float:
    """Inverse of norm_cdf: the z with P(Z <= z) == p.

    This is the number that turns a confidence level into a multiplier:
    norm_ppf(0.975) == 1.959963... is where "1.96" comes from.

    Solved on the lower tail and reflected when p > 0.5, because the lower
    tail is the better-conditioned side: there the CDF value and its
    floating-point resolution shrink together.

    ACCURACY, and a correction to what this docstring used to say.

    It used to claim that past about p = 1 - 1e-7 the answer degraded to
    roughly 1e-8 absolute, and it explained that as an unavoidable
    consequence of how doubles are spaced near 1.0 - not a flaw in the
    solver. The explanation was plausible, the arithmetic in it was right,
    and the conclusion was wrong. The degradation was real but it was a
    BUG, in `norm_cdf`, which this function bisects on: the old `norm_cdf`
    computed `0.5 * (1 + erf(z/sqrt 2))` on the whole line, and in the far
    left tail that sum cancelled away most of its significant digits. The
    quantile inherited the damage - 8e-7 of absolute error at p = 1e-12.
    Once `norm_cdf` was given a separate `erfc` branch for negative z, the
    same solver came back accurate to about 1e-13 across the entire range
    tested, p from 1e-15 to 1 - 1e-13.

    The lesson is worth more than the fix: a believable theoretical
    account of why something cannot be better is exactly what stops anyone
    looking. The bug was found by comparing against SciPy in
    tests/test_statlib.py, which is the one thing a convincing explanation
    cannot talk its way out of.

    WHAT IS GENUINELY LIMITED, as opposed to what was broken: fidelity to
    the quantile you MEANT. `1 - 1e-11` written in source does not produce
    that number; it produces the nearest double, which sits up to 5.5e-17
    away. The normal density out at z = 6.7 is about 2.2e-11, so that
    rounding alone moves the true quantile by around 3e-6, and no
    implementation can recover what the literal threw away. This function
    agrees with SciPy to 1e-13 there because SciPy is handed the same
    damaged double, and both solve the problem they were actually given.
    If you need quantiles that far out, pass the tail probability directly
    and negate: `-norm_ppf(1e-11)` keeps every digit.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")
    if p > 0.5:
        return -_bisect(norm_cdf, 1.0 - p, -40.0, 40.0)
    return _bisect(norm_cdf, p, -40.0, 40.0)


# --------------------------------------------------------------------------
# Student's t
# --------------------------------------------------------------------------
def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for T ~ Student-t with `df` degrees of freedom.

    Derivation of the identity used here: for T with df = v,

        P(|T| > t) = I_x(v/2, 1/2)    where x = v / (v + t^2)

    so the two tails together are that incomplete beta value, and one tail
    is half of it. The sign of t decides which end we are on.

    The t distribution is wider than the normal because estimating the
    variance from the same sample costs you something; with df above ~200
    the difference stops mattering at three decimal places.

    This function is a thin wrapper over `t_sf`, which does the numerical
    work and chooses between the two algebraically equivalent forms of the
    identity. Deriving the CDF from the survival function rather than the
    other way round is what keeps the lower tail accurate:

        t < 0   P(T <= t) = P(T >= -t) = t_sf(-t)     by symmetry, exact
        t > 0   P(T <= t) = 1 - t_sf(t)

    The direct route for t < 0 would be `0.5 - half_central`, and that
    subtraction is where the accuracy went. At t = -8 on 500 df the true
    value is 4.36e-15, while `0.5 - 0.49999999999999563` keeps only two
    digits - a relative error of 4.7e-3, in a function whose whole job is
    to be exact. The symmetry route never subtracts nearly equal numbers,
    so the lower tail is as accurate as the upper one, which is the only
    defensible state for a symmetric distribution.

    For t > 0 the `1.0 - t_sf(t)` form loses nothing that is recoverable:
    any CDF value that close to 1 cannot carry its tail's digits in a
    double no matter how it is computed, which is exactly why `t_sf`
    exists as a separate function rather than as a convenience.
    """
    if df <= 0.0:
        raise ValueError(f"t_cdf needs df > 0, got {df}")
    if t == 0.0:
        return 0.5
    if t < 0.0:
        return t_sf(-t, df)
    return 1.0 - t_sf(t, df)


def t_sf(t: float, df: float) -> float:
    """P(T > t), computed in the form that stays accurate far out in the tail.

    The companion to `norm_sf`, and the primitive that `t_cdf` is built
    from. At t = 8.9 on 39,998 df the true tail is 2.9068e-19; anything
    computed as "0.5 minus something that rounds to 0.5" returns zero
    instead, and a p-value of zero in a readout is a credibility problem
    rather than a rounding one.

    TWO FORMS OF ONE IDENTITY, AND WHERE EACH ONE BREAKS
    Since I_x(a,b) = 1 - I_{1-x}(b,a), the same tail has two expressions:

        tail form     P(|T| > t) = I_x(df/2, 1/2),  x = df/(df + t^2)
        central form  P(|T| < t) = I_x(1/2, df/2),  x = t^2/(df + t^2)

    The tail form never subtracts nearly equal numbers, so it holds its
    relative accuracy out to where the incomplete beta itself underflows,
    around 1e-300. But its argument goes to 1 as t goes to 0, and on 1e7
    degrees of freedom `df/(df + t^2)` rounds to exactly 1.0 for every
    |t| below about 1e-5 - so the tail form returns exactly 0.5 across a
    band around the median where the true answer is already 0.4999960.
    That is a flat spot, and anything bisecting on this function cannot
    resolve a quantile inside it.

    The central form has the mirror-image problem: its argument goes to 0
    as t goes to 0, which doubles represent happily, but it computes the
    far tail as a difference of two numbers near 0.5 and loses everything.

    So the switch is on whether the tail form's argument is still
    distinguishable from 1.0. `t^2/(df + t^2)` is exactly `1 - x_tail`, so
    it measures that distance directly: below 1e-12 the tail form has
    nothing left to work with and the central form is used, which is also
    precisely the region where the answer is near 0.5 and the central form
    is at its best. Above it, the tail form, all the way down to
    underflow. Neither is ever evaluated where it loses precision.

    The flat spot was a real defect in this file, found by
    tests/test_statlib.py after the docstring here had claimed accuracy
    "from the median all the way out" for some time. The claim is now
    true; it was not before, and no simulation in proofs/ would ever have
    said so.

    THE ONE LIMIT THAT REMAINS, measured and left alone deliberately.
    Agreement with SciPy is about 1e-14 relative for df up to a few
    thousand, and degrades to 4e-9 at df = 1e7 and 2e-8 at df = 1e8. That
    is not the continued fraction giving up - `_betacf` raises rather than
    returning a half-converged answer, so there is no silent failure here.
    It is the argument: x = df/(df + t^2) is 0.9999999000000 at df = 1e7,
    and a double has only about ten significant digits of `1 - x` left to
    give. `betainc` needs that complement internally, so the information
    was already gone before the algorithm started.

    It could be fixed, by letting `betainc` accept the complement directly
    - this function computes it accurately as t^2/(df + t^2) a few lines
    below - and that would restore full precision at any df. It is not
    fixed, and the reason is worth stating rather than hiding: 4e-9 of
    relative error on a p-value changes no decision anyone will ever make,
    and at a million degrees of freedom the t distribution and the normal
    agree to more places than this error spans, so `norm_sf` is the better
    tool there anyway. Extra machinery for accuracy nobody can use is a
    cost with no benefit. The limit is asserted in the test suite so that
    it cannot quietly get worse.
    """
    if df <= 0.0:
        raise ValueError(f"t_sf needs df > 0, got {df}")
    if t == 0.0:
        return 0.5
    t2 = t * t
    distance_from_one = t2 / (df + t2)
    if distance_from_one < _CENTRAL_SWITCH:
        half_central = 0.5 * betainc(0.5, 0.5 * df, distance_from_one)
        return 0.5 - half_central if t > 0.0 else 0.5 + half_central
    half_tails = 0.5 * betainc(0.5 * df, 0.5, df / (df + t2))
    return half_tails if t > 0.0 else 1.0 - half_tails


def t_ppf(p: float, df: float) -> float:
    """Inverse of t_cdf. Reflected onto the lower tail, as norm_ppf is."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_ppf needs 0 < p < 1, got {p}")
    if df <= 0.0:
        raise ValueError(f"t_ppf needs df > 0, got {df}")
    # With 1 degree of freedom the t distribution is Cauchy: tails so heavy
    # that the 1e-12 quantile sits near -1e12. Widen the bracket for small df.
    bound = 1e5 if df > 2.0 else 1e13
    if p > 0.5:
        return -_bisect(lambda t: t_cdf(t, df), 1.0 - p, -bound, bound)
    return _bisect(lambda t: t_cdf(t, df), p, -bound, bound)


# --------------------------------------------------------------------------
# chi-square
# --------------------------------------------------------------------------
def chi2_cdf(x: float, df: int) -> float:
    """P(X <= x) for X ~ chi-square with `df` degrees of freedom.

    A chi-square with v degrees of freedom is a Gamma(v/2, scale 2), so its
    CDF is the regularised lower incomplete gamma at a = v/2, x/2.
    Used by srm.py: a sample-ratio mismatch test is a goodness-of-fit test,
    and goodness-of-fit is chi-square.
    """
    if df < 1:
        raise ValueError(f"chi2_cdf needs df >= 1, got {df}")
    if x <= 0.0:
        return 0.0
    return gammainc_lower(0.5 * df, 0.5 * x)
