# 10 — Variance reduction with CUPED

## The question

[Document 07](07-power-and-sample-size.md) established that
`n ∝ sd²/mde²`. So there are exactly two ways to detect a smaller effect:
more traffic, or less variance.

Traffic you have to buy. **Variance you may be able to delete for free,
using data you already have.**

## The idea

Most of the variance in a per-user metric is not caused by your feature.
It is caused by users differing from each other: some buy every week, some
never. Your treatment is a small nudge on top of enormous pre-existing
heterogeneity, and that heterogeneity is pure noise as far as your
comparison is concerned.

But you can often *predict* it. If you know how much each user spent
**before the experiment started**, you can subtract the part of their
post-period spend that was predictable from that, and test the remainder.

The treatment effect survives the subtraction untouched, because
**pre-period behaviour cannot have been affected by a treatment that had
not happened yet.** The noise does not survive.

Same users, same effect, narrower interval. The name is from Deng, Xu,
Kohavi and Walker (2013) — **C**ontrolled-experiment **U**sing
**P**re-**E**xisting **D**ata.

## The adjustment, derived

```
theta = Cov(Y, X) / Var(X)
Y_adj = Y - theta * (X - mean(X))
```

Where does `theta` come from? You want the subtraction that removes the
most variance. Write the adjusted variance for an arbitrary coefficient
`c`:

```
Var(Y - cX) = Var(Y) - 2c*Cov(Y,X) + c^2*Var(X)
```

That is a quadratic in `c`, opening upward. Differentiate and set to zero:

```
-2*Cov(Y,X) + 2c*Var(X) = 0        ->       c = Cov(Y,X)/Var(X)
```

So `theta` is not a choice. It is the unique minimiser, and it is also
exactly the slope of the regression of Y on X — CUPED is linear regression
wearing different clothes.

Substitute it back:

```
Var(Y_adj) = Var(Y) - Cov(Y,X)^2/Var(X) = Var(Y) * (1 - rho^2)
```

**Variance falls by the square of the correlation.** That is the whole
payoff, and it is worth memorising:

| ρ | variance removed | sample-size multiplier |
|---|---|---|
| 0.1 | 1% | 0.99× |
| 0.3 | 9% | 0.91× |
| 0.5 | 25% | 0.75× |
| 0.7 | 49% | 0.51× |
| 0.9 | 81% | **0.19×** |
| 0.95 | 90% | **0.10×** |

A covariate correlated 0.9 with the outcome turns a five-week experiment
into a one-week experiment. For a revenue or retention metric on an
established user base, last month's value of the same metric is routinely
correlated 0.6–0.8 with this month's. **This is usually the
highest-leverage change available to an experimentation programme**, and
it costs one join.

## The measurement

`proofs/proof_cuped.py`, planting a known correlation and comparing the
realised variance reduction against `1 − ρ²`:

| ρ | theory `ρ²` | measured | effective n gain |
|---|---|---|---|
| 0.1 | 0.0100 | 0.0088 | 1.00× |
| 0.3 | 0.0900 | 0.0900 | 1.10× |
| 0.5 | 0.2500 | 0.2541 | 1.34× |
| 0.7 | 0.4900 | 0.4824 | 1.93× |
| 0.9 | 0.8100 | 0.8075 | 5.19× |
| 0.95 | 0.9025 | 0.9027 | 10.28× |

And the thing you actually care about — detecting something you otherwise
would have missed. An effect of 0.08 at 1,200 per arm:

```
raw      detected  734/1500   (48.9%)
adjusted detected 1272/1500   (84.8%)
```

**Same users, same effect, same α.** The raw test is a coin flip; the
adjusted test is a properly powered experiment. Nothing was bought.

## Why the narrowing is only the square root

```
[PASS] the interval narrows by 1 - sqrt(1 - rho^2)
       measured 0.338858   expected 0.338562
       intervals shrank 33.9% on average
```

That claim runs at ρ = 0.75, and the arithmetic is worth doing by hand
once:

```
variance removed   = rho^2            = 0.5625      -> 56% less variance
interval narrowing = 1 - sqrt(1-rho^2) = 1 - 0.6614  -> 34% narrower
```

The variance falls 56% and the interval narrows only 34%, because
**interval width scales with the standard deviation** — the square root of
the variance. Measured 33.886% against a prediction of 33.856%.

The practical consequence is a communication trap: **in interval terms
CUPED looks much less impressive than it is in sample-size terms.**
Sample size tracks the variance (56% less traffic needed), the interval
tracks its square root (34% narrower). Both are true. If you demo CUPED
by showing a stakeholder the interval, you will undersell it by half;
quote the sample size saved instead.

## Two claims that make the narrowing trustworthy

A narrower interval is easy to produce and worthless unless it is still
honest. So:

```
[PASS] the CUPED-adjusted estimate is unbiased
       mean 0.249128 against a truth of 0.25 (unadjusted mean 0.248213)

[PASS] the CUPED-adjusted 95% interval still covers
       measured 0.946 against 0.95
```

That second claim is annotated in the proof as "this is the claim that
makes the narrowing safe rather than just flattering", and that is exactly
right. Any number of transformations will shrink your interval. Only one
that keeps coverage at 95% has shrunk it legitimately.

## The three ways to get it wrong

The module docstring lists these, and the third is the subtle one.

### 1. A covariate that is not pre-treatment

This is the dangerous one, and the proof measures how dangerous:

```
[PASS] a post-treatment covariate biases the estimate toward zero
       mean estimate -0.000056 against a truth of 0.25
       bias -0.250056, which is 100% of the real effect
```

**The entire effect, gone.** A real +0.25 effect reported as −0.00006.

And the annotation that matters:

> The interval was narrow and the number was wrong, and nothing in the
> output said so.

If X is measured during the experiment, the treatment moved X. Subtracting
the part of Y predictable from X then subtracts part of the real effect.
Your interval gets tighter and your point estimate gets wrong, and the
readout looks *better* than before — more precision, more confidence,
around a number that is now meaningless.

`cuped_adjust` **cannot check this for you.** It is a property of your data
pipeline, not of the arrays handed in. Which is why the naming convention
is worth enforcing: a column called `pre_revenue_30d` carries its
timestamp in its name, and `revenue_30d` does not.

This is the same class of error as [document 02](02-metrics-and-the-unit-of-analysis.md)'s
guardrail denominator and [document 01](01-what-an-experiment-is.md)'s
slice on `engaged`: **conditioning on something the treatment moved.** It
is the single most productive mistake in analytics, in the sense that it
produces the most wrong answers per unit of effort.

### 2. Theta estimated per arm

Fitting separate thetas to control and treatment lets the adjustment
absorb some of the treatment effect — the same leak, by another route.
`cuped_adjust` pools the arms and refuses to do otherwise:

```python
# leak the treatment effect into theta, and pooling gives the most
```

### 3. Forgetting to centre, or centring per arm

```python
x_mean = ...                                  # pooled across both arms
adj_c = y_c - theta * (x_c - x_mean)
adj_t = y_t - theta * (x_t - x_mean)
```

`Y − theta*X` without centring changes the *level* of the metric. The
difference between arms is unaffected, so the p-value is fine — but the
adjusted means are no longer on the metric's own scale, and "control
converted at 4.1%" stops being a true sentence. Centring on the **pooled**
mean keeps both adjusted means interpretable and keeps the difference
unbiased. Centring per arm would re-introduce leak 2.

This is a small bug with a specific signature: everything is right except
the two numbers a stakeholder actually reads.

## What CUPED does not do

> CUPED does not fix bias, confounding, or a broken randomisation. It is a
> variance reduction technique for an experiment that is already valid. If
> SRM fails, CUPED makes a wrong answer more precise.

And the limit that decides whether it is worth building:

> It also does nothing for users with no history. New users have no
> pre-period, X is zero for all of them, and the correlation across a
> population that is mostly new is near zero. **On a signup funnel CUPED
> is close to useless; on a retention or revenue metric over an
> established base it is the highest-leverage change available.**

So the question to ask before investing in it is not "what is our
correlation" but "what share of our experiment population has a
pre-period at all". If you are testing acquisition, the answer is none,
and no amount of implementation effort changes that.

## Do it yourself

**1. Watch `ρ²` appear** out of nothing but a covariance:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab.cuped import cuped_adjust

rng = np.random.default_rng(0)
for rho in (0.3, 0.5, 0.7, 0.9):
    x = rng.standard_normal(20_000)
    y = rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal(20_000)
    # NOTE the argument order: outcome and covariate PER ARM, interleaved.
    r = cuped_adjust(y[:10_000], x[:10_000], y[10_000:], x[10_000:])
    print(f"rho {rho}  theory {rho ** 2:.4f}  measured {r.reduction:.4f}"
          f"  theta {r.theta:.4f}")
```

```
rho 0.3  theory 0.0900  measured 0.0899  theta 0.3033
rho 0.5  theory 0.2500  measured 0.2492  theta 0.4993
rho 0.7  theory 0.4900  measured 0.4929  theta 0.7070
rho 0.9  theory 0.8100  measured 0.8077  theta 0.8999
```

Note that `theta` comes out equal to `rho`. That is not a coincidence: in
this construction `Var(X) = 1`, so `theta = Cov(Y,X)/Var(X) = Cov(Y,X) =
rho`. On real data where X and Y have different scales they will differ,
and `theta` carries the units.

**1b. Now get the argument order wrong on purpose**, grouping by variable
instead of by arm:

```python
r = cuped_adjust(y[:10_000], y[10_000:], x[:10_000], x[10_000:])   # WRONG
```

```
rho 0.9  theory 0.8100  measured 0.0000  theta 0.0014
```

**No error, no warning, theta ≈ 0, no variance removed.** I made exactly
this mistake while writing this document. The failure is benign — a
mis-ordered call cannot bias the estimate, it just silently does nothing —
but it is benign *by luck*, and the only thing that surfaced it was
checking the measured number against the theory. If I had printed the
reduction without a prediction beside it, "0.0000" would have looked like
a property of the data.

That is the argument for `proofs/` in one sentence: **a number with
nothing to compare it to cannot be wrong.**

**2. Then derive theta yourself** by minimising `Var(Y − cX)` on paper. It
is two lines of calculus and it is the difference between CUPED being a
technique you apply and a thing you understand. Check your answer against
`r.theta`.

**3. Reproduce the 100% bias**, because it is the finding worth being
afraid of. Build `x` so that it *depends on* the treatment — for instance
`x = 0.5*y + noise`, where `y` already contains the effect — then run
`cuped_adjust` and look at the estimate and the interval. The interval
will be beautiful.

**4. Ask the question that decides it.** In your own data, what fraction
of users in a typical experiment have a non-null pre-period value for your
primary metric? Compute it. That number, not the correlation, is what
determines whether CUPED is worth building where you work.

→ [11: Ratio metrics and the delta method](11-ratio-metrics-and-the-delta-method.md)
