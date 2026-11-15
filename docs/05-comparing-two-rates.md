# 05 — Comparing two rates

## The question

8.0% of control converted, 9.2% of treatment. Is 1.2 points real?

This is the most common comparison in commercial analytics and it has one
subtlety worth the whole document: **the p-value and the confidence
interval are computed from different standard errors, on purpose.**

## A rate is a mean

A conversion rate is the mean of a 0/1 column. So everything in
[document 04](04-comparing-two-means.md) applies — and you could run
`welch_t` on a 0/1 array and get a defensible answer. `two_proportion`
exists because a 0/1 variable has one property a general continuous one
does not:

```
Var(X) = p (1 - p)         for X in {0, 1}
```

The variance is a **function of the mean**. You do not estimate it
separately; knowing the rate tells you the spread exactly. That is worth a
dedicated function, because it makes two things possible: the standard
error can be computed from counts alone (no array needed), and — the
important one — you can compute what the variance *would be under the null
hypothesis*, which is the next section.

The standard error of a single rate follows immediately:

```
se(p_hat) = sqrt( p(1-p) / n )
```

and for a difference of two independent rates, variances add again:

```
se = sqrt( p_c(1-p_c)/n_c + p_t(1-p_t)/n_t )
```

## The deliberate inconsistency

Here is the code, and it is the part that looks like a bug:

```python
p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)
se_pooled   = sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
se_unpooled = sqrt(p_c * (1 - p_c)/n_control + p_t * (1 - p_t)/n_treatment)

z = estimate / se_pooled                                    # the p-value
ci = (estimate - crit * se_unpooled,                        # the interval
      estimate + crit * se_unpooled)
```

Two different standard errors, in one function, on one dataset. The reason
is that **they answer different questions**, and this is the single
clearest example in the project of a statistic being defined by the
question rather than by the data.

**The p-value is computed under the null hypothesis.** The null says the
two rates are equal. If they are equal, there is only one rate in the
world, and the best estimate of it uses every observation you have:
`(s_c + s_t)/(n_c + n_t)`. Using two separate rate estimates to test the
hypothesis that the rates are the same throws away information the null
hands you for free. Section 04's logic in reverse: here the pooling
assumption is not an assumption about reality, it is *the thing being
assumed in order to be tested*.

**The interval is not computed under the null.** Its job is to contain the
true difference whatever that difference is — including the case where the
rates genuinely differ, which is precisely when pooling them is wrong. So
each group must be allowed its own variance.

The consequence is visible and worth stating out loud: **a p-value of
0.049 can sit next to an interval whose endpoint is a hair past zero.**
The two can disagree at the boundary. The textbook alternative is to use
the unpooled SE for both so they always agree — and that makes the test
slightly worse at the only thing a test is for.

This project asserts that the inconsistency is **present** rather than
merely documented, in `tests/test_inference.py`:

```
the two standard errors differ by 3.3% on the worked example
```

A comment in a docstring rots. A test fails.

## The measurement: both doing their own job

| what | measured | target |
|---|---|---|
| pooled test's false positive rate, 8% base, 1500/arm | 0.0498 | 0.05 |
| unpooled interval's coverage, 8% base, +1.2pp effect | 0.9478 | 0.95 |

Each choice is validated against the thing it is responsible for — the
test against α, the interval against coverage — not against each other.
`proofs/proof_false_positive.py` and `proofs/proof_ci_coverage.py`.

## Small cells

```python
if min(cells) < 10:
    notes.append(f"smallest cell is {min(cells)} (<10): the normal "
                 "approximation is strained, prefer bootstrap_diff "
                 "or an exact test")
```

`cells` is all four of successes and failures in both arms. This is the
normal approximation to the binomial, and it needs enough mass in every
cell for the binomial to look normal. The rule of thumb is 10 successes
**and** 10 failures per group.

It is a **note, not an exception**, and that is a considered choice: a
warning you can read beats an exception you work around. Someone who hits
a raised error at 9 successes will pass the data through
`welch_t` instead and get no warning at all.

How bad is it when strained? Measured, at a 1% base with 300 per arm —
about three successes per arm, far below the guard:

```
false positive rate 0.0487 against alpha 0.05
```

**Conservative, not anti-conservative.** The approximation loses power
rather than inventing findings, which is the harmless direction. That is
worth knowing because it tells you what the warning means: "you may be
missing things", not "you may be reporting things that are not there".
The interval is where small cells hurt, and `bootstrap_diff` from
[document 06](06-what-an-interval-means.md) is the answer there.

## Guards that are not formulas

```python
if not 0 <= s <= n:
    raise ValueError(f"successes_{name}={s} outside 0..{n}")
```

More successes than trials. This happens constantly in real work — a
numerator joined at one grain and a denominator at another, so a user with
two sessions is counted twice in the numerator and once in the
denominator. The function raises, because the alternative is a rate above
1.0 flowing into a `sqrt(p(1-p))` that returns NaN, and a NaN p-value in
a readout that nobody questions. (The same joined-grain mistake, surviving
all the way to a shipped decision, is the subject of
[document 02](02-metrics-and-the-unit-of-analysis.md).)

```python
if se_pooled == 0.0:
    p_value = 1.0
    notes.append("no successes in either group, or no failures; "
                 "test is vacuous")
```

Nobody converted in either arm, or everybody did. There is no information
about a difference in rates, and the honest report is `p = 1.0` with a
note, not a division by zero.

```python
bad = arr[(arr != 0) & (arr != 1)]
if bad.size:
    raise ValueError(f"{name} must be 0/1 for a proportion test; found {bad[:3]}")
```

`two_proportion_from_arrays` checks that the column really is binary before
treating `Var = p(1-p)` as true. A column of counts passed to a proportion
test produces a confidently wrong standard error, because the variance
formula is simply false for it. **And it prints the offending values** —
`found [2. 3. 2.]` — so the diagnosis takes a second rather than a
debugging session.

## Which function to use

| situation | use |
|---|---|
| 0/1 per randomised unit | `two_proportion` / `two_proportion_from_arrays` |
| continuous per randomised unit | `welch_t` |
| a ratio whose denominator is finer than the unit | `ratio_delta` ([11](11-ratio-metrics-and-the-delta-method.md)) |
| heavy tails, small cells, or no formula you trust | `bootstrap_diff` ([06](06-what-an-interval-means.md)) |

The one that catches people is row three. "Click-through rate" is a 0/1
outcome **per impression**, and impressions are not what you randomised.
Running `two_proportion` on impression-level rows gives a standard error
that is too small by a factor you can calculate — measured at **2.2×** in
this project. That is document 11, and it is the second-most expensive
mistake in here.

## Do it yourself

**1. See the two standard errors diverge.** They agree when the rates are
similar and separate when they are not:

```python
import sys; sys.path.insert(0, "src")
from explab.inference import two_proportion

for s_t in (80, 100, 150, 300):
    r = two_proportion(80, 1000, s_t, 1000)
    print(f"treatment {s_t/1000:.1%}  se reported (unpooled) {r.se:.6f}")
```

Then compute the pooled SE by hand for each row and plot the gap against
the size of the effect. You will see why the choice is immaterial at the
null — where the p-value is decided — and material far from it, where the
interval is.

**2. Confirm the null-hypothesis logic is real, not aesthetic.** Run a
null simulation (two arms at the same 8% rate) and count rejections using
the pooled SE, then again using the unpooled SE. Both should land near
0.05 — the pooled one slightly closer. Then do the same at a 1% base with
200 per arm, where the difference becomes visible. The pooled SE is better
calibrated because at the null it is estimating a real single quantity
from twice the data.

**3. Break it on purpose.** Pass a count column to
`two_proportion_from_arrays` and read the error. Then pass the same column
to `welch_t` and notice that it answers cheerfully. Keep that asymmetry in
mind: **the function that refuses your data is doing you a favour.**

→ [06: What an interval means](06-what-an-interval-means.md)
