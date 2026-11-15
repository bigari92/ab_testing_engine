# 06 — What an interval means

## The question

"No significant difference" and "we could not have detected a difference
this small" are different sentences, and a p-value cannot tell them apart.
Only an interval can. This document is about what the interval actually
claims, which is narrower than most people think, and about what to do
when no formula for it exists.

## The claim a 95% interval makes

```
absolute +0.0122  [+0.0031, +0.0213] at 95%
```

What that means: **if you repeated this experiment many times, 95% of the
intervals so constructed would contain the true effect.**

What it does **not** mean: there is a 95% probability the true effect is
between 0.0031 and 0.0213. The true effect is a fixed number. It is either
in this interval or it is not — there is no probability left in it. The
95% is a property of the **procedure**, not of this one interval.

That distinction sounds like pedantry until you notice it is testable, and
that testing it is the only way to know a library is correct. "95% of
intervals cover the truth" is a frequency. You can simulate it. A
statement about *this* interval's probability is not checkable against
anything.

So `proofs/proof_ci_coverage.py` does exactly that: plant a known truth,
build the interval, count how often it covers.

| method | design | coverage | target |
|---|---|---|---|
| `welch_t` | equal variances, true effect +0.4 | 0.947 | 0.95 |
| `welch_t` | 4× variance ratio, 4× size ratio | 0.957 | 0.95 |
| `two_proportion` | 8% base, +1.2pp effect | 0.9478 | 0.95 |
| `ratio_delta` | +1pp CTR applied per user | 0.9493 | 0.95 |
| `bootstrap_diff` | lognormal revenue, n = 500 | 0.9450 | ≥ 0.93 |

That is the whole justification for trusting any interval this library
prints. Note what it required: **a known truth.** You cannot measure
coverage on real data, because you do not know what you are covering.
This is the reason the project has a data generator at all — see
[document 16](16-how-the-proofs-work.md).

## Where the interval comes from

```python
crit = t_ppf(0.5 + confidence / 2.0, df)
ci = (estimate - crit * se, estimate + crit * se)
```

Three ingredients: the estimate, its standard error, and a multiplier that
turns a confidence level into a number of standard errors.
`t_ppf(0.975, df)` for 95%, which is 1.96 at large df and larger at small
df.

`0.5 + confidence/2` rather than `confidence`, because the interval is
two-sided: 95% in the middle leaves 2.5% in each tail, so the upper
endpoint is the 97.5th percentile. Writing `t_ppf(0.95, df)` gives a 90%
interval, and it is a mistake that produces plausible-looking numbers
about 20% too narrow.

## Why the interval is the part that matters

A p-value of 0.4 is compatible with two completely different situations,
and the interval is what separates them:

```
A:  absolute +0.0004  [-0.0006, +0.0014]     p = 0.42
B:  absolute +0.0400  [-0.0560, +0.1360]     p = 0.41
```

**A is an answer.** The effect is somewhere between −0.06pp and +0.14pp.
Whatever it is, it is too small to matter. Stop asking.

**B is not an answer.** The effect could be −5.6pp or +13.6pp. The
experiment was too small to say anything at all.

Same p-value, opposite conclusions. The project turns this into a
structural distinction — *flat* versus *inconclusive* — which is what
`practical_threshold` is for ([document 02](02-metrics-and-the-unit-of-analysis.md))
and what [document 15](15-the-decision-and-the-write-up.md) measures.

This is also the single most common failure in commercial analytics:
reporting B as "no effect" and moving on. It is not "no effect". It is "no
experiment".

## The relative interval, and the approximation in it

```python
return (self.ci[0] / self.mean_control, self.ci[1] / self.mean_control)
```

Dividing both endpoints by the control mean treats the control mean as
**known** rather than estimated. It is not — it has its own sampling
error, so the honest interval on a *relative* lift is slightly wider than
this.

The docstring says so, and then says why it is done anyway: it is the
convention every experimentation tool uses, and at sample sizes where you
would ship a decision the difference is in the third decimal. The project
reports both the absolute and the relative figure for a related reason:

> a 2-point lift on a 4% base is a 50% relative lift, and a 2-point lift
> on a 40% base is a 5% one. Both appear in the readout so neither can
> hide.

Relative lift is how a result gets communicated and absolute lift is how
it gets sized. Showing one without the other is how "a 50% improvement!"
turns out to mean four extra conversions.

## When no formula exists: the bootstrap

Revenue per user is lognormal-ish: most users spend nothing, a few spend a
lot. A median, a 90th percentile, a trimmed mean — these have sampling
distributions nobody has written down for your case. `bootstrap_diff`
gets an interval without one.

**The idea**, which is genuinely remarkable once it lands: the only reason
you cannot see the sampling distribution is that you have one sample
instead of many. But your sample *is* an estimate of the population. So
draw new samples **from your sample**, with replacement, each the same
size as the original, and the spread of the statistic across those
resamples estimates the spread it would have across real repeated
experiments.

```python
idx_c = rng.integers(0, n_c, size=(n_boot, n_c))
idx_t = rng.integers(0, n_t, size=(n_boot, n_t))
boot = fn(t_arr[idx_t], axis=1) - fn(c[idx_c], axis=1)
lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
```

Two details that are not stylistic:

**Resample to the original size.** The width of a sampling distribution
depends on n. A resample of 100 from a sample of 1000 answers a question
about experiments of size 100, which is not your experiment.

**With replacement.** Without it, every resample is a permutation of the
original and every statistic is identical. The variation comes entirely
from drawing some rows twice and others not at all.

And the implementation detail worth copying: the indices are drawn as one
`(n_boot, n)` integer matrix and the statistic is taken along an axis.
10,000 resamples of 500 rows is a single vectorised operation, not a Python
loop — about 200× faster, which is the difference between a tool you use
and one you avoid.

## The p-value, and a docstring that was wrong

```python
share_below = float((boot <= 0.0).mean())
share_above = float((boot >= 0.0).mean())
p_value = min(1.0, 2.0 * min(share_below, share_above))
```

The share of resampled differences on the far side of zero, doubled. It is
the percentile interval read the other way round — the confidence level at
which the interval would just touch zero — so here the test and the
interval can never disagree. (Contrast
[document 05](05-comparing-two-rates.md), where they can, on purpose.)

```python
floor = 1.0 / n_boot
if p_value < floor:
    p_value = floor
    notes.append(f"p-value floored at 1/n_boot = {floor:g}; "
                 "no resample crossed zero")
```

Once no resample crosses zero the share is exactly 0.0, and **p = 0 from
10,000 draws is a claim the draws cannot support.** All they establish is
`p < 1/n_boot`. The floor is the resolution of the method, stated, rather
than a zero that reads as certainty.

### The sixth defect

This document is why there is a sixth entry in document 00's list of
defects. The docstring of `bootstrap_diff` said the p-value was

> "inverted from the interval by bisection on the confidence level: one
> minus the narrowest level whose interval still excludes zero"

There is no bisection in the function. There never was. The code has
always computed the doubled share above.

Nothing downstream was wrong — the two methods agree in spirit, and the
floor handles the case the old text was worried about. But a docstring
describing an implementation **that does not exist** is worse than no
docstring, because it is the thing a reviewer checks the code against.
It is the same failure mode as `norm_ppf`'s confident explanation of its
own bug in [document 03](03-distributions-from-scratch.md): a plausible
account of a mechanism is exactly what stops anyone from reading the
mechanism.

It was found by writing this document. That is worth generalising:
**explaining a mechanism to somebody else is a check that no test suite
performs.** A test asserts behaviour. Documentation asserts the *reason*
for the behaviour, and the reason is where the misunderstandings live.

## The bootstrap's limits, measured not hedged

```
percentile bootstrap on lognormal revenue, n = 500 per arm:  0.9450
welch_t on the same data:                                    0.9550
```

The percentile bootstrap **under-covers** here — 94.5% against a nominal
95%, with a Monte Carlo standard error of 0.0089, so the shortfall is
inside noise but sits consistently on the low side, which is the known
behaviour for a mean with this much skew at this n. Welch, on the same
heavy-tailed data, does slightly better.

The claim in the proof is therefore written as a **bound, not an
equality** — "does not badly under-cover" — and it prints the measured
number so the size of the shortfall is visible. That is the honest shape
for a claim about a method that is known to be approximate. Asserting
`== 0.95` would have required a tolerance wide enough to hide a real
problem.

The practical rule follows directly, and it is the opposite of how the
bootstrap is usually sold:

> **For a mean at n in the hundreds, use `welch_t`.** It is faster and
> better. Use the bootstrap where the formula is *missing* — a median, a
> percentile, a trimmed mean — not as a way of avoiding the formula you
> have.

See [document 12](12-outliers-and-heavy-tails.md) for what to do about the
one whale that is actually driving your revenue metric, which is a
different problem from this one.

## Do it yourself

**1. Measure coverage yourself.** This is the exercise that makes the
definition click, because you will watch the frequency appear:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab.inference import welch_t

rng = np.random.default_rng(3)
TRUE = 0.4
covered = 0
for _ in range(2000):
    c = rng.normal(0.0, 1.0, 300)
    t = rng.normal(TRUE, 1.0, 300)
    lo, hi = welch_t(c, t).ci
    covered += lo <= TRUE <= hi
print(covered / 2000)        # ~0.95
```

Then print the first ten intervals and notice that they are all
different, most contain 0.4, and one or two do not. **That** is what 95%
means. No single line of that output has a probability attached to it.

**2. Then try to do it with real data** and work out why you cannot. The
impossibility is the point: coverage is only measurable against a known
truth, which is why this project generates its own.

**3. Find the n where the bootstrap stops under-covering.** Run the
lognormal coverage check at n = 100, 500, 2000, 10000. You will see the
shortfall shrink. Write down the n at which it stops mattering for your
own data's skew — that number is a fact about your data, not about the
bootstrap, and nobody can tell you it in advance.

**4. Break the two-sided multiplier.** Change `0.5 + confidence/2.0` to
`confidence` in `welch_t`, re-run exercise 1, and watch coverage drop to
about 90%. Then put it back. The numbers in the broken version look
completely reasonable, which is the lesson.

→ [07: Power and sample size](07-power-and-sample-size.md)
