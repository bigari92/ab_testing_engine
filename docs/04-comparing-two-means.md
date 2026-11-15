# 04 — Comparing two means

## The question

Two groups, a continuous metric — revenue per user, session minutes, days
to hire. Control averages 41.2, treatment averages 43.8. Is 2.6 real?

## The arithmetic

Three lines, and then the rest of this document is about the third one.

```
se = sqrt( var_c/n_c + var_t/n_t )
t  = (mean_t - mean_c) / se
df = se^4 / ( (var_c/n_c)^2/(n_c-1) + (var_t/n_t)^2/(n_t-1) )
```

**Line 1.** The sampling variance of a mean is `var/n` — that is the
central limit theorem, and it is the only distributional fact the test
needs. The two groups are independent, and **variances of independent
things add**, so the variance of the *difference* of the two means is the
sum of the two. Take the square root and you have the standard error of
the quantity you care about. Note that it is the variances that add, not
the standard errors: `sqrt(a² + b²)`, never `a + b`.

**Line 2.** Divide the effect by its own standard error and you have a
unitless number: how many standard errors from zero. Feed it to `t_sf` from
[document 03](03-distributions-from-scratch.md) and you have a p-value.

**Line 3** is Welch–Satterthwaite, and it is where the interesting part is.

## Why the variance divisor is n−1

```python
var_c, var_t = float(c.var(ddof=1)), float(t_arr.var(ddof=1))
```

`ddof=1` is not a convention, and NumPy's default of `ddof=0` is the wrong
one here. The sample variance measures spread around the **sample** mean,
and the sample mean is fitted to this very sample — it sits closer to these
particular data points than the true mean does. So dividing by `n`
systematically understates the spread. Dividing by `n − 1` corrects it
exactly, in expectation.

One degree of freedom was spent estimating the mean. You get `n − 1`
independent pieces of information about spread, not `n`. At n = 400 the
difference is a quarter of a percent and invisible; at n = 5 it is 25% and
it is the whole answer.

## The pooled t-test, and why this library refuses to provide it

The test most people were taught assumes **one common variance** in both
groups and estimates it by combining the two samples:

```
sp² = ( (n_c-1) var_c + (n_t-1) var_t ) / (n_c + n_t - 2)
se  = sqrt( sp² (1/n_c + 1/n_t) )
df  = n_c + n_t - 2            <- a whole number, which is the tell
```

It is a better test **when the assumption is true**, because it pools
information from both arms to estimate one quantity instead of two, and so
earns more degrees of freedom.

In an experiment, that assumption is exactly the thing the treatment might
have broken. A feature that helps heavy users and does nothing for light
users **changes the variance as well as the mean**. There is no reason to
expect equal variances after an intervention; expecting them is expecting
the treatment to have been uniform, which is usually the hypothesis under
test.

The pooled test is deliberately **not in `src/explab/`**. It lives in
`proofs/proof_welch_vs_pooled.py`, which is the file that explains why you
should not use it — the only place a tempting wrong tool belongs.

## What goes wrong, measured

With no effect planted at all, 15,000 simulations per cell, at a nominal
5%:

| design | sd control | sd treatment | pooled FPR | Welch FPR |
|---|---|---|---|---|
| n = 400 vs 400 | 1.0 | 1.0 | 0.0469 | 0.0469 |
| n = 400 vs 400 | 1.0 | 2.0 | 0.0507 | 0.0504 |
| n = **800 vs 200** | 1.0 | 2.0 | **0.1802** | 0.0477 |
| n = **200 vs 800** | 1.0 | 2.0 | **0.0049** | 0.0513 |

Read the last two rows together, because that pair is the entire argument.

**Identical variance ratio. Identical total sample size. Only the
allocation reversed.** The pooled test's false positive rate goes to
**18%** one way and **0.49%** the other. Rows 1 and 2 show that unequal
variance *alone* costs nothing — the 4× variance ratio at equal sizes is
fine. It is unequal variance **together with** unequal n that breaks it.

And note which way each one breaks:

- The **larger arm has the smaller variance** → the pooled estimate of the
  common variance is dragged down toward the large arm's small one, the
  standard error comes out too small, every statistic is inflated, and you
  find effects that are not there one time in five.
- The **larger arm has the larger variance** → the pooled variance is too
  big, the standard error too large, and the test becomes so conservative
  it finds almost nothing. You are running at α = 0.005 while reporting
  0.05, throwing away most of your power and never knowing.

There is no "it is at least conservative" fallback. The direction of the
error depends on a design detail — which arm got more traffic — that
nobody checks against the variances. **That is why the choice cannot be
left to judgement at the keyboard**, and why this library does not offer
the option.

## The prediction, made before the simulation ran

A simulation that only reports what happened is a weak instrument. The
proof predicts each of those four numbers from the design alone, with no
simulation involved:

```python
def predicted_pooled_fpr(n1, n2, sd1, sd2):
    df = n1 + n2 - 2
    e_sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df
    se_pooled = math.sqrt(e_sp2 * (1.0 / n1 + 1.0 / n2))
    se_true = math.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    t_crit = t_ppf(1.0 - ALPHA / 2.0, df)
    return 2.0 * norm_sf(t_crit * se_pooled / se_true)
```

The logic: the pooled test rejects when `|estimate| > t_crit × se_pooled`,
but the estimate's **actual** standard deviation is `se_true`. So in units
of the real sampling distribution, the test's threshold sits at
`t_crit × se_pooled / se_true` standard errors instead of 1.96 — and the
rejection rate is the normal tail beyond that. The ratio
`se_pooled/se_true` is the whole defect in one number.

| design | predicted | measured |
|---|---|---|
| 400 vs 400, sd 1 vs 1 | 0.0497 | 0.0469 |
| 400 vs 400, sd 1 vs 2 | 0.0497 | 0.0507 |
| 800 vs 200, sd 1 vs 2 | 0.1785 | 0.1802 |
| 200 vs 800, sd 1 vs 2 | 0.0042 | 0.0049 |

Predicting 0.1785 and measuring 0.1802 is a different kind of evidence
from measuring 0.1802 and calling it high. The first says *I understand
the mechanism*; the second says *I observed a number*. Only the first
tells you what will happen on a design you have not simulated yet.

(The predictions sit at 0.0497 rather than exactly 0.05 in the first two
rows because the formula uses a normal tail for a t-distributed statistic.
That approximation is in the **prediction**, not in the test, and the proof
sets its tolerance from Monte Carlo noise so the gap is accounted for
rather than ignored.)

## What Welch costs

This is the question that decides it, and the answer is better than the
textbooks suggest. At 400 per arm with a true effect of 0.18 and
**genuinely equal variances** — the one case where the pooled test is
supposed to win:

```
pooled detected 72.39%
Welch  detected 72.39%      a difference of +0.000%
```

Zero. Not "a small price": nothing measurable at 10,000 simulations. The
pooled test's advantage is extra degrees of freedom, and the difference
between `t(798)` and `t(797.6)` is not a difference. The pooled test only
pays you back at sample sizes small enough that you should not be shipping
a decision from them anyway.

So: **no detectable cost when the assumption holds, and an 18% false
positive rate when it fails quietly.** There is no decision to make.

## Welch's fractional degrees of freedom

```
df = 25.3  on a comparison of 20 against 20
```

Not a bug. Welch's formula interpolates between the t distributions that
bracket the true sampling distribution of the statistic — which is not
exactly a t distribution at all, and the fractional df is the best single
t that approximates it. `statlib.t_sf` takes `df` as a float throughout for
exactly this reason, and a whole-number df in a two-sample comparison is a
sign that something pooled when it should not have.

Two bounds worth carrying in your head: Welch's df is never more than
`n_c + n_t − 2`, and never less than `min(n_c, n_t) − 1`. If you see one
outside that range, the arithmetic is wrong.

And a third fact that is more useful than either bound: **the df is
governed by whichever arm contributes most of the standard error**, not by
the total sample size. On 20 against 2000 with equal variances, the small
arm supplies `1/20` of the variance budget against the large arm's
`1/2000` — 99% of it — so the df collapses to 19.4, essentially
`min(n) − 1`. Give the *large* arm five times the standard deviation and
the df rises to 24.9, because the large arm now carries a real share. This
is why a 50/50 split is worth defending: a 90/10 split has roughly the
degrees of freedom of the 10, and the standard error to match.

## What the code does beyond the formula

```python
if se == 0.0:
    return TestResult(..., p_value=1.0 if mean_t == mean_c else 0.0,
                      notes=("zero variance in both groups; "
                             "no sampling error to report",))
```

Both groups constant. The formula would divide by zero; the honest answer
is that there is no sampling error to report, and the result says so rather
than producing a NaN. Every guard like this in the project exists because
something downstream would otherwise print a number that looks like a
measurement — see the NaN sweep in
[document 15](15-the-decision-and-the-write-up.md).

```python
if not np.isfinite(arr).all():
    raise ValueError(f"{name} contains NaN or inf; clean or drop before testing")
```

Refusing NaN input rather than silently ignoring it. `np.mean` of an array
with one NaN is NaN, which at least propagates visibly; worse is code that
calls `np.nanmean` as a kindness and quietly changes the denominator. If
rows are missing, **that** is the finding, and it belongs in the readout.

## One-sided tests are not offered

Also deliberate, and stated in the module docstring so nobody has to
wonder:

> A one-sided test is defensible only if you committed to the direction
> before seeing the data, and in practice nobody can prove they did.

The cost of two-sided is `(1.96/1.645)² ≈ 1.42` — about 42% more units. It
is worth paying for a number you can defend in a room, because the moment
you have seen the data, the direction you would have "pre-committed" to is
unknowable even to you.

## Do it yourself

**1. Build the 18% yourself**, so it is a thing you have seen rather than
a thing you read:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab.inference import welch_t

rng = np.random.default_rng(1)
alerts_welch = 0
for _ in range(2000):
    c = rng.normal(0, 1.0, 800)        # big arm, small variance
    t = rng.normal(0, 2.0, 200)        # small arm, big variance. No effect.
    alerts_welch += welch_t(c, t).significant
print("Welch", alerts_welch / 2000)    # I measured 0.055
```

(0.055 on 2,000 simulations, where the Monte Carlo noise is
`sqrt(0.05 × 0.95 / 2000) = 0.005`. One standard error high, so: 0.05.
Judging a simulated rate without computing its noise first is how people
convince themselves of effects that are not there — the proofs derive
every tolerance this way, see [document 16](16-how-the-proofs-work.md).)

Now write `pooled_t` yourself from the three formulas above (eight lines),
run the same loop, and watch it land near 0.18. Then swap the two sample
sizes and watch it land near 0.005. If you copy `pooled_t` out of
`proofs/proof_welch_vs_pooled.py` instead of writing it, you will have
learned nothing — the point is that you can derive a broken test from a
textbook without any step feeling wrong.

**2. Find the n−1.** Run this and explain the result before reading on:

```python
x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
print(x.var(), x.var(ddof=1))        # 4.0 and 4.571428...
```

The ratio is exactly `8/7`. Then take 100,000 samples of size 5 from a
known-variance normal and check which divisor recovers the truth on
average. One of them will be biased low by exactly 20%.

**3. Check the df bounds, and predict before you run.** Take arrays of 20
and 2000. Write down what you expect `result.df` to be when the variances
are equal, and when the large arm has five times the standard deviation.
Then run it:

```python
print(welch_t(rng.normal(0, 1, 20), rng.normal(0, 1, 2000)).df)   # 19.4
print(welch_t(rng.normal(0, 1, 20), rng.normal(0, 5, 2000)).df)   # 24.9
```

Most people predict this backwards, as I did when writing this document —
more variance in the big arm *raising* the df looks wrong until you write
out which term dominates `var_c/n_c + var_t/n_t`. Do that by hand; it is
the fact that makes the Welch formula stop being a black box.

→ [05: Comparing two rates](05-comparing-two-rates.md)
