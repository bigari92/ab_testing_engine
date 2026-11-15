# 11 — Ratio metrics and the delta method

## The question

You randomised **users**. You are measuring **clicks per impression**.
Those are different units, and the gap between them is worth about 31
points of coverage.

This is the second-most expensive mistake in this repository, after
[document 02](02-metrics-and-the-unit-of-analysis.md)'s denominator trap —
and it is the same family of error seen from a different angle. There the
denominator moved; here the denominator is simply *finer* than the thing
you randomised.

## The two wrong things people do

### 1. Analyse at the denominator's level

Pool every impression, call the metric a proportion, compute
`sqrt(p(1−p)/n_impressions)`.

That treats 27,000 impressions from 3,000 users as **27,000 independent
observations.** They are not. Impressions from one user share that user's
behaviour. The real sampling variance is larger, the standard error comes
out too small, p-values come out optimistic, and **nothing in the output
hints at it** — the numbers all look fine, there are more of them than
before, and the intervals are satisfyingly tight.

### 2. Average the per-user ratios

`mean(x_i / y_i)`. This is not wrong arithmetic — it is a **different
question**. It is the average of individual users' rates, in which a user
with one impression counts as much as a user with a thousand.

Sometimes that is genuinely what you want. It is almost never what the
metric definition said. And note the proof's annotation on this one:

> per-user ratios  coverage 0.9487  *(different estimand; covering THIS
> truth is not its job)*

It covers 95% of the time — of its own truth. It is a correct answer to a
question nobody asked, which is harder to catch than a wrong answer.

## The fix: the delta method

Write the metric as `R = X̄ / Ȳ`, where `X̄` is the mean numerator **per
randomised unit** and `Ȳ` the mean denominator per unit. A first-order
Taylor expansion of `R` around `(EX, EY)` gives

```
Var(R)  ~=   (1/Ybar^2) * ( Var(X) - 2R*Cov(X,Y) + R^2*Var(Y) ) / n
```

Three terms, and **the middle one is the whole point.** It is the term the
naive calculation drops, and it is large and positive whenever heavy users
are heavy in both columns — which is almost always. A user with many
impressions also has many clicks; the numerator and denominator move
together, and ignoring that correlation is what breaks the variance.

Each group gets its own variance, and the two are added, as independent
groups always are.

## The governing quantity: the design effect

How bad is the naive version? Not a vague "somewhat". It is computable in
advance, and the proof computes it **from the generator's parameters
before the simulation runs**:

```python
def theory_design_effect(concentration):
    e_lam = math.exp(LOG_MEAN + LOG_SD * LOG_SD / 2.0)
    e_lam2 = math.exp(2.0 * LOG_MEAN + 2.0 * LOG_SD * LOG_SD)
    e_n = e_lam + 1.0
    e_n2 = (e_lam2 + e_lam) + 2.0 * e_lam + 1.0
    m_eff = e_n2 / e_n                     # E[n^2]/E[n]
    icc = 1.0 / (concentration + 1.0)
    return 1.0 + (m_eff - 1.0) * icc, m_eff, icc
```

```
DE = 1 + (m_eff - 1) * ICC
```

**This is the cluster design effect**, and it is one of the most useful
formulas in applied statistics:

- **ICC**, the intra-cluster correlation: how much of the variation in the
  outcome is *between* users rather than within them. For a
  Beta-Binomial click process with concentration `c`, `ICC = 1/(c+1)`.
- **m_eff**, the effective cluster size: `E[n²]/E[n]`, **not** the mean.
  It is larger than the mean whenever cluster sizes vary, because big
  clusters contribute disproportionately to the variance. At a lognormal
  impression count with mean ~25, `m_eff` is **37.4**.

And then the standard error understatement is `sqrt(DE)`.

## The measurement

Same users per arm, same impressions per user, same effect, **same code**.
Only how much users differ from one another changes:

| user heterogeneity | ICC | DE | `sqrt(DE)` | delta coverage | impression-level coverage |
|---|---|---|---|---|---|
| identical (c = 4000) | 0.0002 | 1.009 | 1.005× | 0.9482 | 0.9477 |
| mild (c = 40) | 0.0244 | 1.887 | 1.374× | 0.9448 | 0.8495 |
| **realistic (c = 8)** | 0.1111 | **5.043** | **2.246×** | 0.9560 | **0.6298** |

Read the last row. The impression-level interval, computed with perfectly
correct arithmetic from perfectly correct data, covers the truth
**63% of the time** when it claims 95%.

> same users per arm, same impressions per user, same effect, same code:
> the impression-level interval covers 94.8% when users are identical and
> 63.0% when they differ as much as real ones do — a span of 31.8%. **One
> realistic readout in 2.7 is wrong**, with p-values optimistic in the
> same proportion, and nothing in the output hints at it.

And the prediction is the part that makes this knowledge rather than
observation:

| level | predicted coverage | measured |
|---|---|---|
| identical | 0.9490 | 0.9477 |
| mild | 0.8463 | 0.8495 |
| realistic | 0.6172 | 0.6298 |

Predicted from `DE`, which was computed from the generator's parameters
with no simulation involved. **Heterogeneity, not sample size, is what
breaks it** — more data makes the broken interval tighter and no more
correct, exactly as in [document 01](01-what-an-experiment-is.md)'s
selection bias.

## The claim that makes the delta interval believable

Coverage is one check. This is the sharper one:

```
[PASS] the delta SE equals the actual spread of the estimates
       mean reported SE 0.005366  against the standard deviation
       of 6,000 point estimates 0.005315
```

> An interval is only honest if its width matches how much the estimate
> actually moves.

That is a stronger statement than coverage, because coverage can be right
for the wrong reasons — a skewed interval can cover 95% while being the
wrong shape. Comparing the *reported* standard error against the *realised*
standard deviation of the estimates checks the actual quantity the formula
claims to compute. At every heterogeneity level it agrees within 2%.

And one honest residual, which the proof states rather than smoothing
over:

> The measured shortfall sits a little below theory at every level, and in
> the same direction: the clustering formula ignores the denominator's own
> variance and its covariance with the numerator, which the delta method
> carries and which shaves a little off.

Measured 2.211 against a theory of 2.246 — 1.5% low, consistently. The
design-effect formula is the simpler model; the delta method is the more
complete one; the gap between them is small, systematic, and explained.
A claim that got the direction of its own residual right is doing more
work than one that just passed.

## The input contract, and the estimand hidden in it

```
One row per randomisation unit, in matching order: a user's numerator
and that same user's denominator. Units with a zero denominator are
kept, because dropping them changes the estimand from "clicks per
impression across all users" to "clicks per impression among users who
had an impression", which is a different question with a different
answer.
```

Dropping zero-denominator rows looks like obvious hygiene — you cannot
divide by zero, so why keep them? Because `ratio_delta` never divides by
anything per-row; it works with the *sums*. A user with zero impressions
contributes 0 to both sums and correctly dilutes nothing.

Drop them and you have conditioned on **having an impression**, which is
post-treatment if the treatment affects whether people see the thing. That
is [document 02](02-metrics-and-the-unit-of-analysis.md)'s mistake all
over again, arrived at through what felt like data cleaning. Most
instances of this error in the wild are committed in a `WHERE` clause by
someone being careful.

## When to use which

| metric | unit randomised | use |
|---|---|---|
| conversion per user | user | `two_proportion` |
| revenue per user | user | `welch_t` |
| clicks per impression | user | **`ratio_delta`** |
| refunds per converter | user | **`ratio_delta`** |
| items per order | user | **`ratio_delta`** |

The rule from [document 02](02-metrics-and-the-unit-of-analysis.md),
restated: if the metric's denominator is not the thing you randomised,
you need `ratio_delta`. And if the denominator is something the treatment
can *move*, you need to think harder before you need any function at all.

## Do it yourself

**1. Produce the 63% yourself.** This is the most convincing experiment in
the whole course, because both calculations are right:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab.inference import ratio_delta, two_proportion

rng = np.random.default_rng(0)
TRUE_CTR, N_USERS, CONC = 0.11, 1200, 8.0
delta_cov = naive_cov = 0
for _ in range(400):
    arms = []
    for shift in (0.0, 0.0):                 # no effect; truth is 0
        imps = rng.poisson(rng.lognormal(3.2, 0.5, N_USERS)) + 1
        rate = rng.beta((TRUE_CTR + shift) * CONC,
                        (1 - TRUE_CTR - shift) * CONC, N_USERS)
        arms.append((rng.binomial(imps, rate), imps))
    (xc, yc), (xt, yt) = arms
    lo, hi = ratio_delta(xc, yc, xt, yt).ci
    delta_cov += lo <= 0.0 <= hi
    lo, hi = two_proportion(int(xc.sum()), int(yc.sum()),
                            int(xt.sum()), int(yt.sum())).ci
    naive_cov += lo <= 0.0 <= hi
print("delta", delta_cov / 400, " impression-level", naive_cov / 400)
```

```
CONC 8.0     delta 0.9425   impression-level 0.63
CONC 4000.0  delta 0.9625   impression-level 0.965
```

Set `CONC = 4000` — users all clicking at the same rate — and the two
converge. The difference is **entirely** user heterogeneity, and your own
data has more of it than you think.

(The `int()` casts are needed because `two_proportion` takes counts and
validates `0 <= successes <= n`. That it insists on integers is the same
guard discussed in [document 05](05-comparing-two-rates.md) — and here it
is about to be fed a numerator and denominator at the *wrong grain*, and
cannot possibly know.)

**2. Compute your own design effect.** For your real metric: what is the
mean denominator per user, what is `E[n²]/E[n]`, and what fraction of the
rate variance is between users? `DE = 1 + (m_eff − 1)·ICC` tells you how
wrong your current readout is, as a single number, before you change any
code.

**3. Find the third estimand.** On the same data, compute `mean(x_i/y_i)`
over users with at least one impression, and compare it to
`sum(x)/sum(y)`. They will differ, sometimes substantially. Work out which
one the metric *definition* at your workplace actually describes — and
then which one the dashboard computes. In my experience those are not
always the same.

→ [12: Outliers and heavy tails](12-outliers-and-heavy-tails.md)
