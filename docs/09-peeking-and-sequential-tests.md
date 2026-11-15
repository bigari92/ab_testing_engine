# 09 — Peeking and sequential tests

## The question

The test has been running four days. Someone checks the dashboard, sees
p = 0.04, and says ship it.

What is wrong with that? Nothing about the arithmetic. The p-value is
correctly computed. The problem is that it was **looked at**, and the
rule "stop when you see p < 0.05" is a different test from "compute p once
at the end" — with a different false positive rate that nobody computed.

## The cost, measured

Stop at the first look where p < 0.05, on data with **no effect at all**.
200,000 simulations per row:

| looks | false positive rate | inflation |
|---|---|---|
| 1 | 0.0499 | 1.00× |
| 2 | 0.0833 | 1.67× |
| 3 | 0.1063 | 2.13× |
| 5 | 0.1424 | 2.85× |
| 10 | 0.1935 | 3.87× |
| 20 | 0.2467 | 4.93× |
| 30 | 0.2802 | 5.60× |

**Five looks nearly triples it. Twenty looks puts it past one in four.**

Two looks — just one mid-experiment glance, which nobody would describe as
p-hacking — already costs 67%. The whole industry practice of "watching
the dashboard" is somewhere between rows 10 and 30.

And the inflation is asserted to be **monotone**, because it must be:
every additional look adds chances to cross and removes none.

## Why, in one sentence

A p-value is the probability of a statistic this extreme **under a fixed
sampling plan.** Peeking changes the plan. You are no longer asking "is
this one statistic extreme?" but "did this *random walk* ever go above a
line?" — and a random walk given more chances to cross a line will cross
it more often. That is all.

The curve's shape follows: the gain from each extra look shrinks, because
consecutive looks are highly correlated — the data at day 5 is mostly the
data at day 4. Which is also why the fix is not Bonferroni across looks;
that would assume independence and be far too conservative.

## The second cost, which is worse

The inflated error rate is the famous problem. This one gets less
attention and does more damage.

```
run to the end:            mean estimate 0.00998   truth 0.01
stop at the first p < 0.05: mean estimate 0.01982   truth 0.01
```

**Stopping early doubles the effect you measure.** 1.98× here.

This is the **winner's curse**, and the mechanism is worth stating
carefully: you stop on the draws where noise was *helping*. Among all the
parallel worlds in which the experiment reached significance early, the
over-estimates are over-represented, because an over-estimate is what
crossing the line early requires.

So the effect you ship is not the effect you measured. You forecast the
annual impact from 0.0198 and deliver 0.0099, and six months later someone
asks why the roadmap's numbers never materialise. Running to the end gives
0.00998 against a truth of 0.01 — unbiased, as
[document 01](01-what-an-experiment-is.md) promised.

Note also what the proof reports alongside: of 616 false positives across
5 looks, **558 were declared before the planned end.** Those are not
borderline calls at the finish line. They are exactly the experiments
somebody would have called a win and stopped.

## Doing it honestly: group-sequential boundaries

You can look early. You just have to **demand more evidence** at each look
so the *total* error rate across all looks is still 5%.

```
obrien_fleming boundaries, 5 looks, two-sided alpha 0.05
 look  information  |z| needed  ~one-look p
    1         20%       4.563    5.045e-06
    2         40%       3.226     0.001253
    3         60%       2.634     0.008429
    4         80%       2.281      0.02252
    5        100%       2.041      0.04129
calibrated on 400,000 simulations: measured alpha 0.0500 +/- 0.0003
```

Read the last column. To stop at 20% of the data you need p < 5e-06 — an
effect so large it is unmistakable. By the final look the bar is p <
0.0413, barely tighter than the 0.05 you would have used with no interim
looks at all.

**That is the trade, and it is a good one:** you buy five chances to stop
early for a 17% tightening of the final threshold. The alternative is one
chance and 0.05.

Two shapes, and the choice between them is about what you want:

```python
if shape == "pocock":
    return np.ones_like(fractions)
if shape == "obrien_fleming":
    return 1.0 / np.sqrt(fractions)
```

- **Pocock** — flat. Same `|z| = 2.412` at every look, which is p < 0.0159
  at *every* look including the last. Spends α evenly, so it stops early
  most often, and pays for it at the end.
- **O'Brien–Fleming** — proportional to `1/sqrt(information)`. Spends
  almost nothing early, so an early stop means something dramatic, and the
  final look is barely penalised. **The right default when the experiment
  will probably run to completion anyway**, which is nearly always.

Put the two final looks side by side and the choice is concrete:
**Pocock 0.0159, O'Brien–Fleming 0.0413.** If your experiment goes the
distance — and most do — Pocock has cost you two thirds of your final
threshold to buy early stops you did not take.

## How the boundaries are found

Not from a table. Simulated:

```python
deltas = np.diff(np.concatenate([[0.0], fractions]))
increments = rng.standard_normal((n_sims, n_looks)) * np.sqrt(deltas)
brownian = np.cumsum(increments, axis=1)
z_paths = np.abs(brownian / np.sqrt(fractions))
```

The accumulating test statistic behaves like **Brownian motion observed at
the information times**. Information accrues with sample size, the
increment variance is proportional to the information gained, so draw
`N(0, Δt)` increments, cumulate, and divide by `sqrt(t)` to standardise.
That gives you 200,000 complete null experiments as a matrix.

Then bisect on a single scale factor:

```python
def crossing_rate(scale: float) -> float:
    return float((z_paths >= scale * weights).any(axis=1).mean())
```

And one line in the docstring is the reason this is sound rather than
merely convenient:

> One set of simulated paths is reused across every candidate c, which
> makes the search monotone in c and therefore safe to bisect: with common
> random numbers, raising the boundary can only ever reduce the crossing
> rate.

**Common random numbers.** If each candidate `c` were evaluated on fresh
paths, the crossing rate would be a noisy function of `c`, not monotone,
and bisection would be invalid — it would converge to whichever side the
noise happened to favour. Reusing the paths makes the function exactly
monotone, because on a *fixed* path a higher boundary can only fail to be
crossed. This is a general technique and it is the kind of thing that
separates a simulation you can trust from one that produces a number.

```python
if n_sims < 10_000:
    raise ValueError(
        f"n_sims={n_sims} gives a Monte Carlo error around "
        f"{np.sqrt(alpha * (1 - alpha) / n_sims):.4f}, too coarse to "
        "calibrate a boundary; use >= 10,000")
```

The guard computes the error it is complaining about and puts it in the
message, so the limit is arguable rather than arbitrary.

## The four-way validation

Simulated boundaries need more checking than a formula does, so
`proofs/proof_sequential.py` closes it from four directions:

**1. Against the published literature.** Pocock (1977) gives 2.413 for 5
looks at two-sided 0.05; this computes **2.41142**. O'Brien–Fleming's
published table is (4.56, 3.23, 2.63, 2.28, 2.04); this computes (4.563,
3.226, 2.634, 2.281, 2.041), largest gap **0.0044**. Fifty years of
independent arithmetic, reproduced from scratch.

**2. On an independent simulation** — a different seed from the one used to
calibrate: Pocock 0.0516, O'Brien–Fleming 0.0507. This is the check
against overfitting the boundary to its own random numbers.

**3. On real data.** The boundaries come from a Brownian abstraction;
conversion data is not Brownian. Applied to actual accumulating
`two_proportion` tests under the null: **0.0458 against 0.05.** That is
the claim that matters, because it is the one that could have failed.

**4. That it still works.** A valid test that never fires is easy. With a
real effect planted, the O'Brien–Fleming design detected it in
**1474/1500 runs (98.3%), of which 1377 stopped before the planned end.**

Point four is the one to remember. The purpose of a sequential design is
not only validity — it is **finishing sooner when the effect is large.**
92% of the detections came early. That is the actual business case: a
clear winner ships in days instead of weeks, and the price is paid only by
experiments that were ambiguous anyway.

## What this does not fix

The winner's curse is **still there**. A sequential design controls the
false positive rate; it does not make an early-stopped estimate unbiased.
If you stop at look 2, your effect estimate is still conditioned on having
crossed a high boundary, and it is still too big.

So if you stop early, ship the decision and **not the number**. The
magnitude needs a fresh measurement — a holdback, or a later read on the
full sample. Nothing in this library will correct it for you, and
pretending otherwise is how the forecast and the outcome part company.

## Do it yourself

**1. Produce the inflation yourself**, since it is six lines and it is the
thing nobody believes until they see it:

```python
import numpy as np
rng = np.random.default_rng(0)
LOOKS = 5
alerts = 0
for _ in range(20_000):
    z = np.cumsum(rng.standard_normal(LOOKS) * np.sqrt(1/LOOKS))
    z = np.abs(z / np.sqrt(np.arange(1, LOOKS+1) / LOOKS))
    alerts += (z >= 1.96).any()
print(alerts / 20_000)        # ~0.142
```

Then change `LOOKS` to 1, 2, 20 and watch the table above appear:

```
 1   0.0482
 2   0.0852
 5   0.1458
20   0.2543
```

Six lines, 20,000 simulations, no library. If you can write this from
memory you will never again be talked into "just a quick look at the
dashboard".

**2. Measure the winner's curse.** In the same loop, plant a real effect,
record the estimate at the look where you stop, and compare its mean
against the truth. Then record the estimate at the **final** look for the
same runs and compare that. The second is unbiased; the first is not. The
gap is what you would have over-promised.

**3. Calibrate your own boundary and check it against Pocock:**

```python
import sys; sys.path.insert(0, "src")
from explab.sequential import calibrate_boundary

print(calibrate_boundary(5, shape="pocock", n_sims=200_000))
```

You should see 2.41 at every look. Reproducing a 1977 result from a random
number generator in two seconds is the moment this stops being something
you look up in a table.

**4. Break the common random numbers.** Modify `crossing_rate` to draw
fresh paths on every call, re-run the calibration several times, and watch
the answer wobble. That wobble is why the docstring's sentence about
monotonicity is load-bearing.

→ [10: Variance reduction with CUPED](10-variance-reduction-with-cuped.md)
