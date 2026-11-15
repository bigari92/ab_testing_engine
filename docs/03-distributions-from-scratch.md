# 03 — Distributions from scratch

## The question

A client asks: "where does 0.034 come from?"

"SciPy" is not an answer. This document is the answer, all the way down to
arithmetic you could do by hand if you were patient. The file is
[`src/explab/statlib.py`](../src/explab/statlib.py), 500 lines, and it is
the only module in the project with no statistics in it at all — just four
functions and the two special functions they are made of.

## What a p-value is, mechanically

You computed a test statistic — call it `z = 2.12`. The p-value is a
**probability under an assumption**: if the effect were exactly zero, what
fraction of repeated experiments would produce a statistic at least this
extreme?

That fraction is an area under a curve:

```
p = 2 * integral from 2.12 to infinity of  (1/sqrt(2 pi)) e^(-x^2/2) dx
```

Nothing more mysterious than that. Every p-value in this project is one of
three such integrals, and each one has a closed form in terms of a special
function. So the job splits in two:

1. Know which integral your statistic needs → that is documents 04–05.
2. Evaluate it to sixteen digits → that is this document.

## Four public functions, two special functions

```
norm_cdf / norm_sf / norm_ppf     the standard normal     <- math.erf / erfc
t_cdf    / t_sf    / t_ppf        Student's t             <- betainc
chi2_cdf                          chi-square              <- gammainc_lower
```

The `*_ppf` functions are inverses, solved numerically. So there are really
only **two** things to compute: the regularised incomplete beta function
and the regularised incomplete gamma function. Everything else is a change
of variable.

## The normal: already solved by the standard library

The error function is defined as

```
erf(x) = (2/sqrt pi) * integral from 0 to x of e^(-s^2) ds
```

Substitute `s = x/sqrt(2)` into the normal integral and you get the normal
CDF in terms of `erf` exactly. `math.erf` is correct to the last bit, and
re-deriving a Chebyshev approximation for it would teach nothing, so:

```python
def norm_cdf(z: float) -> float:
    if z < 0.0:
        return 0.5 * math.erfc(-z / math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
```

**Why two branches**, when the second line is algebraically valid on the
whole line — this is the most important paragraph in the document, and the
whole of the "do it yourself" section at the end.

For very negative `z`, `erf(z/sqrt 2)` approaches −1. So you are computing
`1 + (-0.9999999999999999...)`. The leading digits cancel, and what is
left is whatever rounding noise was in the last bits. At z = −8.5 that
expression returns **exactly 0.0** when the true value is 9.48e-18. At
z = −7 it keeps about six correct digits out of sixteen.

`erfc` computes `1 − erf` directly, without ever forming the difference.
Same integral, different floating-point path, sixteen digits kept:

```
norm_cdf(-8.5)  = 9.479534822203e-18     (asserted in the test suite)
norm_cdf(-20.0) = 2.753624118606e-89
```

This was a **bug in this file**, found after the proofs were all passing.
It is in document 00's list and it is in here because of what it did
downstream: `norm_ppf` bisects on `norm_cdf`, so a left tail with six good
digits produced a quantile with six good digits.

## Student's t: where the incomplete beta comes in

Now a real derivation, because this is the one worth being able to
reproduce.

`t` is defined as a standard normal divided by the square root of an
independent chi-square over its degrees of freedom:

```
T = Z / sqrt(V/v)       Z ~ Normal(0,1),  V ~ chi2(v),  independent
```

**Step 1.** Square it: `T² = v Z²/V`. Now form the quantity
`T²/(v + T²)` and simplify:

```
v + T²  =  v + vZ²/V  =  v(V + Z²)/V

   T²         vZ²/V            V              Z²
--------  =  --------- * ------------  =  ---------
 v + T²         V         v(V + Z²)        Z² + V
```

The `v` cancels completely. So `T²/(v+T²)` is exactly `Z²/(Z² + V)`.

**Step 2.** `Z²` is chi-square with 1 degree of freedom, which is
`Gamma(1/2)` with scale 2; `V` is `Gamma(v/2)` with the same scale 2. And
there is a standard fact about gammas: for independent `X ~ Gamma(a)` and
`Y ~ Gamma(b)` on a common scale,

```
X / (X + Y)  ~  Beta(a, b)
```

(the scale cancels in the ratio, which is why it must be common). So

```
Z² / (Z² + V)  ~  Beta(1/2, v/2)
```

**Step 3.** `t ↦ t²/(v+t²)` is increasing in `|t|`, so the events line up
and we can read off the probability directly:

```
P(|T| < t) = P( T²/(v+T²) < t²/(v+t²) ) = I_{t²/(v+t²)}(1/2, v/2)
```

where `I_x(a,b)` is the share of a `Beta(a,b)` lying below `x` — the
regularised incomplete beta function. That is the entire t distribution,
in one line, from two facts.

**Step 4.** Mirror it. `I_x(a,b) = 1 − I_{1−x}(b,a)` (swap which tail you
are measuring), and `1 − t²/(v+t²) = v/(v+t²)`, so the same quantity has a
second expression:

```
P(|T| > t) = I_{v/(v+t²)}(v/2, 1/2)
```

Both are exact. They are not interchangeable in floating point, and that
difference is the next section.

## The two forms, and why the code holds both

```python
t2 = t * t
distance_from_one = t2 / (df + t2)
if distance_from_one < _CENTRAL_SWITCH:            # 1e-12
    half_central = 0.5 * betainc(0.5, 0.5 * df, distance_from_one)
    return 0.5 - half_central if t > 0.0 else 0.5 + half_central
half_tails = 0.5 * betainc(0.5 * df, 0.5, df / (df + t2))
return half_tails if t > 0.0 else 1.0 - half_tails
```

- The **tail form** (`betainc(df/2, 1/2, df/(df+t²))`) computes the tail
  directly, never subtracting, so it keeps relative accuracy all the way
  down to where the incomplete beta itself underflows around 1e-300. But
  its argument `df/(df+t²)` goes to 1 as t goes to 0. On 1e7 degrees of
  freedom that argument **rounds to exactly 1.0** for every |t| below
  about 1e-5, and the function returns a flat 0.5 across a band where the
  truth is already 0.499996.
- The **central form** has the mirror problem: its argument goes to 0,
  which doubles handle beautifully, but the far tail then has to be
  computed as `0.5 − (something that rounds to 0.5)` and there is nothing
  left.

So the switch is on **whether the tail form's argument is still
distinguishable from 1.0**, and `t²/(df+t²)` *is* `1 − x_tail` computed
without cancellation — it measures that distance directly. Below 1e-12,
use the central form (which is also exactly where the answer is near 0.5
and the central form is at its best). Above it, the tail form. Neither is
ever evaluated where it loses precision.

The flat spot was the second bug in this file. Its docstring claimed
accuracy "from the median all the way out" while being numerically
constant around the median at large df.

## And the third bug: deriving the CDF the right way round

`t_cdf` does no numerical work at all. It is four lines:

```python
if t == 0.0:
    return 0.5
if t < 0.0:
    return t_sf(-t, df)        # symmetry. exact, no subtraction.
return 1.0 - t_sf(t, df)
```

The direct route for `t < 0` is `0.5 − half_central`. At t = −8 on 500 df
the true value is 4.36e-15, and `0.5 − 0.49999999999999563` keeps **two**
correct digits out of sixteen — a relative error of 4.7e-3 in a function
whose entire job is to be exact.

Deriving the CDF *from* the survival function rather than the other way
round costs nothing and fixes it, because a symmetric distribution's lower
tail is its upper tail. The lesson generalises: **when you have a choice of
algebraically identical expressions, pick the one that never subtracts two
numbers that are nearly equal.** That one rule accounts for all three bugs
in this file.

## Chi-square: the incomplete gamma, in one line

A chi-square with `v` degrees of freedom is `Gamma(v/2)` with scale 2. So:

```python
def chi2_cdf(x: float, df: int) -> float:
    return gammainc_lower(0.5 * df, 0.5 * x)
```

That is the whole function. It exists because a sample-ratio mismatch test
is a goodness-of-fit test and goodness-of-fit is chi-square — see
[document 08](08-sample-ratio-mismatch.md).

## How the two special functions are actually computed

Both are continued fractions evaluated by **Lentz's algorithm**, which
is worth understanding once because it appears everywhere in numerical
work.

A continued fraction is naturally written top-down and naturally evaluated
bottom-up, which is useless when you do not know where the bottom is.
Lentz's trick is to carry a running numerator `c` and denominator `d` so
each new term multiplies the running answer by something near 1, and you
stop when that something is within `_EPS` of 1:

```python
delta = d * c
h *= delta
if abs(delta - 1.0) < _EPS:
    return h
```

Two details that are not decoration:

```python
_TINY = 1e-300        # floor on a denominator, so a zero cannot make an inf
_EPS  = 3e-16         # doubles give ~2.2e-16; asking for less burns iterations
_MAX_ITER = 300
```

and, at the end of the loop:

```python
raise RuntimeError(f"_betacf did not converge for a={a}, b={b}, x={x}")
```

**It raises rather than returning what it has.** That line is why the
accuracy numbers at the end of this document can be trusted: there is no
path by which a half-converged continued fraction becomes a plausible
p-value. A test asserts that the exception is reachable and is an
exception.

`betainc` also mirrors its own argument, for the same reason the t
distribution has two forms — the continued fraction converges quickly only
on one side of the distribution's centre of mass:

```python
if x < (a + 1.0) / (a + b + 2.0):
    return front * _betacf(a, b, x) / a
return 1.0 - front * _betacf(b, a, 1.0 - x) / b
```

and the leading factor is kept in logs, because `x**a` underflows at the
sample sizes a real experiment has:

```python
front = math.exp(
    log_gamma(a + b) - log_gamma(a) - log_gamma(b)
    + a * math.log(x) + b * math.log1p(-x)
)
```

`math.log1p(-x)` rather than `math.log(1-x)`, for the third time in this
document: for small `x`, `1-x` loses digits that `log1p` keeps.

`gammainc_lower` is the same idea with a different split — a power series
when `x < a+1` (fast when x is small relative to a) and a continued
fraction above it (fast when x is large). Each is used only in its
comfortable range.

## The inverses: bisection, deliberately

`norm_ppf(0.975) = 1.959963984540054` is where "1.96" comes from. It is
found by bisection:

```python
def _bisect(fn, target, lo, hi, tol=1e-13):
    if fn(lo) - target > 0.0 or fn(hi) - target < 0.0:
        raise ValueError(f"target {target} not bracketed by [{lo}, {hi}]")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        ...
```

A rational approximation (Acklam, Wichura) would be fifty times faster and
is what most libraries use. Bisection is here because:

- there is **no table of magic constants to get wrong**, and
- the error is exactly the tolerance you asked for, not whatever the fit
  happened to achieve in the region you are in.

An inverse CDF is called once per test to get a critical value, never once
per row. Fifty times faster than invisible is still invisible.

Both inverses solve on the **lower** tail and reflect:

```python
if p > 0.5:
    return -_bisect(norm_cdf, 1.0 - p, -40.0, 40.0)
return _bisect(norm_cdf, p, -40.0, 40.0)
```

because that is the better-conditioned side — the CDF's value and its
floating-point resolution shrink together there. And `t_ppf` widens its
bracket for small df, because t with 1 degree of freedom is Cauchy and its
1e-12 quantile sits out near −1e12:

```python
bound = 1e5 if df > 2.0 else 1e13
```

## The measurement

`tests/test_statlib.py` — **503 of the project's 665 tests** — compares
every function here against SciPy across a parameter grid, and the
tolerances are the measured agreement, not round numbers:

| function | asserted | measured worst |
|---|---|---|
| `log_gamma` | rel 1e-13 | — |
| `betainc` | rel 1e-11 | — |
| `norm_cdf` | rel 1e-12 | 2.1e-13 |
| `norm_ppf` | abs 1e-11 | 1.9e-13 |
| `t_cdf` / `t_sf` | rel 1e-8 | 3.6e-9 |
| `chi2_cdf` | rel 1e-12 | 5.7e-14 |

Plus a regression test per bug, asserting the specific values the broken
versions got wrong, and this, which is the test I am most pleased with:

```python
assert worst(2.0)   < 1e-13         # measured 1.9e-15
assert worst(500.0) < 1e-11         # measured 4.8e-13
assert worst(1e7)   < 1e-8          # measured 3.6e-9
assert worst(1e7)   > 1e-10, "the large-df limit is gone; rewrite this test"
```

The last line asserts that a **known limitation still exists**. At
df = 1e7 the argument `x = df/(df+t²)` is 0.9999999000000, and a double
has only about ten significant digits of `1−x` left to give; `betainc`
needs that complement internally, so the information was gone before the
algorithm started. It is fixable — pass the complement in directly — and
it is deliberately not fixed, because 4e-9 of relative error on a p-value
changes no decision, and at a million degrees of freedom you should be
using `norm_sf` anyway.

The test exists so that the limit **cannot quietly get better or worse
without someone noticing**. A test suite that only asserts upper bounds
drifts; one that pins the known shape of its own inaccuracy does not.

## Do it yourself

**1. Watch the cancellation happen.** This is the single most transferable
thing in the document:

```python
import math, sys; sys.path.insert(0, "src")
from explab.statlib import norm_cdf

for z in (-5, -6, -7, -8, -8.5, -10):
    naive = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    good = norm_cdf(z)
    lost = 0 if naive == good else abs(math.log10(abs(naive - good) / good))
    print(f"{z:6}  naive {naive:.6e}   erfc {good:.6e}   rel err 1e-{lost:.1f}")
```

```
    -5  naive 2.866516e-07   erfc 2.866516e-07   rel err 1e-10.4
    -7  naive 1.279810e-12   erfc 1.279813e-12   rel err 1e-5.6
    -8  naive 6.106227e-16   erfc 6.220961e-16   rel err 1e-1.7
  -8.5  naive 0.000000e+00   erfc 9.479535e-18   rel err 1e-0.0
   -10  naive 0.000000e+00   erfc 7.619853e-24   rel err 1e-0.0
```

Ten good digits at z = −5, five at z = −7, **one** at z = −8, and then
nothing at all. The decay is the signature: every step further into the
tail cancels one more digit, because the quantity being added to 1 is one
step closer to −1. If you ever see accuracy degrade smoothly along a
parameter like that, suspect a subtraction before you suspect an
algorithm.

**2. Verify the t derivation numerically** rather than taking step 2 on
faith. Draw the ingredients and check the Beta claim directly:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab.statlib import betainc

rng = np.random.default_rng(0)
v = 7
t = rng.standard_normal(2_000_000) / np.sqrt(rng.chisquare(v, 2_000_000) / v)
w = t ** 2 / (v + t ** 2)
for q in (0.001, 0.01, 0.05, 0.2, 0.5, 0.8):
    print(f"q={q:<6g} empirical {(w < q).mean():.5f}   "
          f"betainc {betainc(0.5, v / 2, q):.5f}")
```

```
q=0.001  empirical 0.06433   betainc 0.06437
q=0.01   empirical 0.20204   betainc 0.20203
q=0.05   empirical 0.43695   betainc 0.43697
q=0.2    empirical 0.77295   betainc 0.77255
q=0.5    empirical 0.96672   betainc 0.96685
q=0.8    empirical 0.99884   betainc 0.99887
```

Worst disagreement 4.0e-04 — which is the right answer, because the
Monte Carlo noise on a proportion near 0.77 at two million draws is
`sqrt(0.77 × 0.23 / 2e6) = 3.0e-04`. The check is not "are they close";
it is "do they differ by no more than sampling noise explains". You have
now verified `T²/(v+T²) ~ Beta(1/2, v/2)` with your own data, and the t
distribution is no longer something you look up.

**3. Break the switch on purpose.** Set `_CENTRAL_SWITCH = 0.0` in
`statlib.py` — which forces the tail form everywhere — and run:

```bash
python -m pytest tests/test_statlib.py -q -k "flat_near_the_median or monotone"
```

Watch it fail, then put it back. A constant you have seen fail is a
constant you understand.

→ [04: Comparing two means](04-comparing-two-means.md)
