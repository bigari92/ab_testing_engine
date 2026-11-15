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
    """
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

    ACCURACY LIMIT, stated because it is real: for p past about 1 - 1e-7
    the answer degrades to roughly 1e-8 absolute. Not a flaw in the solver.
    `1 - 1e-9` as a double has an absolute resolution of about 1e-16, while
    the normal density out at z = 6 is about 6e-9, so the input itself does
    not carry enough information to pin z down more tightly than
    1e-16 / 6e-9. SciPy's `isf` dodges this by taking the tail probability
    as the argument instead of 1 minus it. If you ever need quantiles that
    far out, pass the tail probability and negate.
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

    The identity above is used in two algebraically equivalent forms, and
    which one we pick matters numerically. I_x(a,b) = 1 - I_{1-x}(b,a), so
    the same tail can be written with x = df/(df+t^2) or with
    x = t^2/(df+t^2). The first sends x to 1 as t goes to 0; for large df,
    df/(df+t^2) rounds to exactly 1.0 for every |t| below about 1e-6, which
    would make the CDF numerically flat through the middle of the
    distribution and leave t_ppf unable to resolve the median. The second
    form sends x to 0 instead, which floating point represents happily.
    So: small |t| takes the second form, large |t| the first, and neither
    is ever evaluated where it loses precision.
    """
    if df <= 0.0:
        raise ValueError(f"t_cdf needs df > 0, got {df}")
    if t == 0.0:
        return 0.5
    t2 = t * t
    if t2 < df:
        # middle of the distribution: half the central mass, then reflect
        half_central = 0.5 * betainc(0.5, 0.5 * df, t2 / (df + t2))
        return 0.5 + half_central if t > 0.0 else 0.5 - half_central
    # out in a tail
    half_tails = 0.5 * betainc(0.5 * df, 0.5, df / (df + t2))
    return 1.0 - half_tails if t > 0.0 else half_tails


def t_sf(t: float, df: float) -> float:
    """P(T > t), computed in the form that stays accurate far out in the tail.

    The companion to `norm_sf`, and needed for the same reason. `t_cdf`
    uses the central parametrisation for |t| below sqrt(df) so that the
    middle of the distribution is not numerically flat — necessary for
    `t_ppf`, fatal for p-values. At t = 8.9 on 40,000 df the true tail is
    about 2.4e-19, while the central form computes 0.5 minus something
    that rounds to 0.5, and returns zero.

    This function always uses the tail form

        P(|T| > t) = I_x(df/2, 1/2),   x = df/(df + t^2)

    which never subtracts two nearly equal numbers, so it is accurate from
    the median all the way out to where the incomplete beta itself
    underflows, around 1e-300.
    """
    if df <= 0.0:
        raise ValueError(f"t_sf needs df > 0, got {df}")
    if t == 0.0:
        return 0.5
    t2 = t * t
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
