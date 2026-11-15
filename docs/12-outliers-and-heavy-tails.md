# 12 — Outliers and heavy tails

## The question

Revenue per user. One customer bought 40 times what anybody else did. What
do you do?

The honest answer has three parts, and the usual advice — "use the median,
it's robust" — turns out to be the one option that does not work at all
here. Everything below is measured on this project's generator.

## First, see how bad it is

40,000 users, 4,130 of them with any revenue at all:

```
mean 3.581   sd 55.719
top     1 user  =  4.55% of all revenue
top    10 users = 19.64%
top   100 users = 37.65%
single largest user's share of total VARIANCE = 34.1%
```

**One user out of 40,000 carries a third of the variance.** From
[document 07](07-power-and-sample-size.md), `n ∝ sd²` — so that one user
is costing you about a third of your sample size. Not a third of a
percent. A third.

That is the whole problem in one number, and it is why this is a power
problem rather than a validity problem — which is the next section, and it
is the part that surprised me.

## Validity is fine. Power is destroyed.

With a genuine null — no conversion effect, no revenue effect — 600
simulations:

| data | `welch_t` false positive rate |
|---|---|
| clean lognormal revenue | 0.0533 |
| with 0.1% of users at 40× | **0.0333** |

Outliers make Welch **conservative**, not anti-conservative. The heavy
tail inflates the sample variance, the standard error goes up, the
interval widens, and the test rejects *less* than it should. Your α is
safe. Nobody is publishing false findings because of a whale.

Now the same setup with a **real** effect planted — a +12% revenue lift
among converters, 20,000 users per arm:

| data | power |
|---|---|
| clean | **0.573** |
| with 0.1% of users at 40× | **0.147** |

**Forty users out of 40,000 took the experiment from 57% power to 15%.**
That is the cost: not wrong answers, but no answers. The experiment runs,
the readout says "not significant", and everyone concludes the feature did
nothing — when the feature worked and the measurement did not.

This is the single most useful reframing in the document. The question is
never "are outliers biasing my result". It is **"how much of my sample
size am I donating to forty people"**.

## What does not work: the median

The standard advice for a heavy tail is to use a robust statistic. So:

```python
bootstrap_diff(c, t, statistic="median", n_boot=2000)
```

Measured power, on data with a real +12% effect: **0.000.** Not low —
zero, in every one of 120 simulations, with and without outliers.

Here is why:

```
median control 0.0000   median treatment 0.0000
  q=0.50  control    0.000  treatment    0.000
  q=0.80  control    0.000  treatment    0.000
  q=0.90  control    0.000  treatment    1.603
  q=0.95  control   14.208  treatment   17.631
```

**Only 9.9% of users convert**, so the median user's revenue is zero in
both arms and always will be. The median is not robust here; it is blind.
It is measuring a quantile that contains no information about the thing
you changed.

The lesson generalises beyond this dataset: **"use the median" is advice
for a symmetric heavy tail, and revenue is not that.** Revenue is
*zero-inflated* — a point mass at zero plus a heavy tail — and robust
statistics designed for the first problem do nothing for the second. Check
your metric's median before reaching for it. If it is zero, so is your
power.

## What does work: winsorising, at the right quantile

Cap values at a high quantile instead of discarding them:

```python
cap = np.quantile(np.concatenate([c, t]), q)   # pooled, so the cap is
c, t = np.minimum(c, cap), np.minimum(t, cap)  # identical in both arms
```

Measured power on the +12% effect, 400 simulations:

| | raw | winsorise 99.9% | **winsorise 99%** | winsorise 95% |
|---|---|---|---|---|
| no outliers | 0.573 | 0.620 | **0.580** | 0.212 |
| 0.1% at 40× | 0.147 | 0.505 | **0.580** | 0.217 |

Read the 99% column twice. It gives **0.580 in both rows** — it fully
restores the power the outliers destroyed, *and* it costs nothing
(0.580 vs 0.573) when there were no outliers to begin with. That is as
close to a free lunch as this subject offers.

Now read the 95% column, which is the warning. Winsorising at the 95th
percentile **collapses power to 0.21** whether or not outliers exist. It
is far worse than doing nothing.

### Why 95% is a disaster and 99% is right

The arithmetic is worth doing, because it tells you where *your* cap
belongs rather than copying a number out of a blog post:

```
share of users converting:            0.099
95th percentile of ALL users:        14.988
median among CONVERTERS:             15.186     <- the same number
```

With 9.9% of users converting, the top 10% of all users *are* the
converters. So the 95th percentile of the full population is the
**median of the people who actually spent money**. Winsorising there caps
half of your converters — it is not trimming a tail, it is flattening the
signal.

The 99th percentile of all users sits at the **90th percentile among
converters**, which is a tail. Hence:

> **Pick the quantile on the non-zero population, not the full one.** A
> cap at the 99th percentile of all users is a cap at the 90th among
> spenders. Compute both before choosing.

If 50% of your users convert, the usable range is entirely different. The
number 99% is not a rule; the *method* is.

### Three properties that make winsorising safe

1. **The cap is computed on the pooled data**, so it is identical in both
   arms. A per-arm cap is a different transformation per arm, which can
   move the difference — the same leak as a per-arm `theta` in
   [document 10](10-variance-reduction-with-cuped.md).
2. **It is declared before you look.** A cap chosen after seeing the
   result is a free parameter, and a free parameter is a p-value you did
   not compute. This belongs with `practical_threshold` in
   [document 02](02-metrics-and-the-unit-of-analysis.md).
3. **It changes the estimand, and you must say so.** You are no longer
   estimating mean revenue per user; you are estimating mean
   *winsorised* revenue per user. For a decision about whether the feature
   helps, that is usually the better question. For a finance forecast it
   is the wrong number, and the difference is exactly the whales you
   capped.

Point 3 is why this project does not winsorise for you. There is no
`winsorise=True` flag in `explab`, and that is deliberate: the choice
changes what is being measured, so it belongs to the caller, in the open.

## The library's weak warning, measured honestly

`welch_t` does flag the symptom:

```python
ratio = (var_c / var_t) if var_t > 0 else float("inf")
if ratio and (ratio > 4.0 or ratio < 0.25):
    notes.append(
        f"variances differ {max(ratio, 1/ratio):.1f}x between groups; "
        "Welch is handling it, but check for outliers or a split effect")
```

How often does it fire when outliers are present? **12.5%.**

So it is a weak detector, and it is worth saying so rather than presenting
it as a safety net. The reason it misses is structural: with outliers
spread evenly across both arms, both variances inflate *together*, and the
note only triggers on a variance **ratio**. It catches the case where the
whales landed lopsidedly — which is a real and worse case — and is silent
on symmetric heavy tails.

**The reliable check is not a note in a readout. It is looking at the
concentration before you start:** the top-1, top-10, top-100 shares at the
head of this document took four lines and told you everything the note
did not.

## And the bootstrap

From [document 06](06-what-an-interval-means.md), on lognormal revenue at
n = 500 per arm:

```
percentile bootstrap coverage  0.9450
welch_t coverage               0.9550
```

Welch wins on heavy-tailed data at this n, which is the opposite of what
the bootstrap's reputation suggests. Use the bootstrap for statistics with
no formula — a 90th percentile, a trimmed mean — not as a defence against
skew in a mean. The central limit theorem is doing more for you than the
resampling is.

## Do it yourself

**1. Measure your own concentration**, which is four lines and will
probably surprise you:

```python
import numpy as np
rev = ...                      # your per-user revenue array
s = np.sort(rev)[::-1]
for k in (1, 10, 100):
    print(k, "%.2f%% of revenue" % (100 * s[:k].sum() / rev.sum()))
print("largest user's share of variance: %.1f%%" %
      (100 * (1 - np.sort(rev)[:-1].var(ddof=1) * (rev.size - 2)
              / ((rev.size - 1) * rev.var(ddof=1)))))
```

**2. Find your own cap.** Print the 95th and 99th percentile of your full
population, then the same percentiles computed **among non-zero users
only**, and compare. The gap between those two tables is the mistake this
document is about.

**3. Reproduce the 0.573 → 0.147 collapse** with the generator, then fix
it with a 99% cap. Then try a 95% cap and watch it get worse than doing
nothing. Doing all three in one sitting is what makes the lesson stick,
because the middle option is the one everybody reaches for.

**4. Check your median before trusting it.** One line:
`np.median(your_metric)`. If it is zero, every robust-statistic
recommendation you have read does not apply to you.

→ [13: Segments, multiplicity and Simpson](13-segments-multiplicity-and-simpson.md)
