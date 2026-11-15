# 08 — Sample ratio mismatch

## The question

You asked for a 50/50 split. You got 49.6/50.4. Is that fine?

**On 1,000 users:** fine. That is a coin landing 496 heads.

**On 2,000,000 users:** not fine. That is roughly an 11-sigma deviation,
and the probability of it happening by chance is smaller than the
probability of your analysis being correct about anything.

This is the only check in the project that runs **before** any effect is
estimated, and the only one whose failure invalidates everything
downstream rather than adjusting it.

## Why it comes first

[Document 01](01-what-an-experiment-is.md) established that randomisation
is the one thing that kills selection bias, and that **randomisation is a
property of the assignment, not of the data** — you cannot verify it by
looking at the outcome.

What you *can* verify is that the assignment mechanism did what you told
it to. If you asked for 50/50 and the data is not 50/50, then units did
not reach the arms the way the design said, and the two groups are no
longer exchangeable. Whatever mechanism dropped or duplicated units may
correlate with the outcome — and usually does.

Common causes, in rough order of how often they turn out to be the
culprit:

- a redirect or bot filter applied to one arm and not the other
- **logging that fires after the feature loads**, so the slower arm loses
  users who bounce first
- a cached assignment that reshuffles on session boundaries
- deduplication on a key that exists in one arm only
- **an analyst filtering the dataset after assignment** — the most common
  of all, and the easiest to fix

The second one deserves its own sentence, because of its direction: if
treatment loads slower and logs later, treatment loses its impatient
users, who are also its worst converters. **The bias makes treatment look
good.** So "treatment is winning" is a reason to check SRM *harder*, not
a reason to relax. The most dangerous SRM is the one in the direction you
were hoping for.

## The arithmetic

A sample ratio mismatch test is a goodness-of-fit test, and goodness-of-fit
is chi-square:

```
chi2 = sum over arms of (observed - expected)^2 / expected
df   = number of arms - 1
```

**Why `df = arms − 1` and not `arms`:** the total is fixed. Once you know
how many units landed in every arm but the last, the last is determined by
subtraction. One constraint, one degree of freedom gone. Getting this
wrong inflates or deflates every p-value the check produces, which is why
the proof includes a three-arm case — with two degrees of freedom and
unequal expected counts, a wrong `df` shows up immediately.

And `expected` is the declared share times the **observed** total, not the
planned total. The test asks whether the split is right, not whether the
experiment reached its planned size. Those are different questions and
only one of them is about assignment.

The p-value comes from `chi2_cdf`, which is `gammainc_lower` in disguise —
see [document 03](03-distributions-from-scratch.md).

## Why the threshold is 0.001, not 0.05

```python
def srm_check(counts, expected_shares=None, *, threshold: float = 0.001):
```

This check runs on **every experiment and every readout**. It is a
monitor, not a hypothesis test. At α = 0.05, one experiment in twenty
raises a false alarm, and within a quarter the team has learned to ignore
it — at which point the monitor has negative value, because it is now
training people to dismiss exactly the signal it exists to raise.

p < 0.001 is the industry convention and the default here: rare enough to
be believed, sensitive enough that a real break of more than a fraction of
a percent shows up quickly at scale.

Measured, on correctly randomised experiments:

```
60,000 experiments of 200,000 users:  63 false alarms = 0.00105
30,000 three-arm 50/30/20 splits:     21 false alarms = 0.0007
```

Both land on 0.001. That second row is the one that would catch a `df`
error, and the first is also the **only place in the whole suite that
exercises the chi-square tail at p = 0.001** — four orders of magnitude
further out than anywhere else the library uses it. A proof that happens
to be the sole coverage of a numerical regime is worth knowing about as a
proof; it is doing two jobs.

## A PASS means nothing on its own

This is the part most implementations get wrong, and it follows directly
from [document 07](07-power-and-sample-size.md): a tight threshold means
**enormous power at large n and very little at small n.**

Take one fixed defect — a 0.35% share deviation, one user in 286 sent to
the wrong arm — and change only the sample size:

| users | detection rate | predicted |
|---|---|---|
| 50,000 | 0.0425 | 0.0422 |
| 200,000 | 0.4380 | 0.4364 |
| 1,000,000 | 0.9998 | 0.9999 |

**The assignment is equally broken in all three rows.** At 50,000 users
the check passes 96% of the time; at a million it fails essentially
always. A PASS at 50,000 users is not evidence of a sound assignment — it
is evidence that you ran a small experiment.

So `srm_check` reports the imbalance it **could** have caught:

```python
detectable: float | None    # smallest absolute share deviation this n
                            # could have caught at 80% power
```

```
sample ratio check: PASS  (p = 0.431, threshold 0.001)
  chi-square 0.6195 on 1 df, n = 200,000
  worst arm control: observed 49.9120% vs expected 50.0000%
  this sample could detect a share deviation of 0.4620% or more at 80% power
```

Which makes the verdict readable as **"no imbalance above 0.46%"** rather
than "no imbalance". That is the same move as reporting a guardrail's MDE
in [document 02](02-metrics-and-the-unit-of-analysis.md), and the same
move as distinguishing *flat* from *inconclusive* in
[document 06](06-what-an-interval-means.md). It is the central habit of
this whole project: **a negative result is only meaningful next to the
sensitivity that produced it.**

## The `detectable` field is checked, not asserted

A field like that is easy to compute slightly wrong and impossible to
notice, because nothing downstream consumes it. So the proof pins it from
two directions. First, a predicted power curve at the 0.001 threshold:

```python
def predicted_power(deviation, total, share, threshold):
    ...
    return norm_sf(z_crit - lam) + norm_cdf(-z_crit - lam)
```

(Both tails, because a deviation in either direction fails the check — the
second term is negligible here but it is the correct formula, and omitting
it is a mistake that only shows up at small λ.)

| deviation | as z | predicted | measured |
|---|---|---|---|
| 0.001118 | 1.0 | 0.0110 | 0.0115 |
| 0.002236 | 2.0 | 0.0984 | 0.1037 |
| 0.003354 | 3.0 | 0.3857 | 0.3858 |
| **0.004620** | **4.13** | **0.8000** | **0.8020** |
| 0.005590 | 5.0 | 0.9563 | 0.9513 |

Note row four: `0.004620` is exactly the number `srm_check` printed as its
`detectable` field, and the simulation detects it 80.2% of the time. Then
a separate claim closes the loop against theory:

> `srm_check` advertises 0.004620 as detectable at 80% power; the power
> formula puts that deviation at 0.8000. The field is checked against
> theory here and measured against simulation in the row above, so it is
> not just a comment.

Notice also what row one tells you about the threshold: a **one-sigma**
deviation is caught 1.1% of the time. At p < 0.001 you need about 3.3
standard errors before the check even starts responding. That is the price
of a monitor you can trust, and it is why `detectable` is not optional.

## What the output says when it fails

```python
if not self.passed:
    lines.append(
        "  do not read the effect estimates until this is explained: "
        "the arms are not exchangeable"
    )
```

Not "consider investigating". The readout in
[document 15](15-the-decision-and-the-write-up.md) puts the assignment
check **above** the primary metric for this reason, and `analyze` refuses
to present a decision when SRM fails. An SRM failure is not a caveat to
attach to a result; it is the absence of a result.

## Do it yourself

**1. Build the intuition that n is everything.** Same deviation, three
scales:

```python
import sys; sys.path.insert(0, "src")
from explab.srm import srm_check

for n in (1_000, 50_000, 1_000_000):
    bad = int(n * 0.5035)            # 0.35% too many in treatment
    print(n, srm_check({"control": n - bad, "treatment": bad}).p_value)
```

```
1000      0.8495
50000     0.1196
1000000   2.596e-12
```

The share is identical in all three. The p-value goes from "obviously
fine" through "mildly interesting" to "impossible". Nothing about the
assignment changed. Note the middle row especially: 0.12 is the kind of
number that gets written up as "no evidence of imbalance", and the
imbalance is real and is 0.35%.

**2. Read a passing result properly.** Run `print(srm_check(...))` on a
clean 200,000-user split and read the `detectable` line aloud. Then say
what the PASS actually licenses you to believe. If you can do that in one
sentence, you have the habit this project is trying to teach.

**3. Find the SRM in your own data.** This is the one exercise with
immediate practical value: take any dataset at work that has a group
column and a declared split, and run `srm_check` on it. Filtered datasets
fail this constantly — a `WHERE` clause applied after assignment is an
SRM, and it is the most common cause on the list above. If it fails, you
have found a real defect in a real pipeline, using about four lines of
code, before doing any analysis at all.

→ [09: Peeking and sequential tests](09-peeking-and-sequential-tests.md)
