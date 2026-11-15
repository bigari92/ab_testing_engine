# 16 — How the proofs work

## The question

Why should anyone believe a p-value this library produced?

Not "is the code readable" or "does it have tests". **Why believe the
number.** This document is the answer, and it is the part that makes the
other seventeen defensible to somebody who does not trust you yet.

There are three independent mechanisms, and each one catches things the
others cannot.

## Mechanism 1: proofs — simulation against a known truth

```
A "proof" here is a simulation study. Each one states a property the
library claims, generates data where the truth is known, measures whether
the property holds, and fails the build when it does not.
```

**11 proof files, 104 claims, about six minutes.** The structure is always
the same: plant a truth, run the library, compare.

The reason this is possible at all is the data generator. On real data you
cannot measure coverage, because you do not know what you are covering
([document 06](06-what-an-interval-means.md)). On generated data you know
`Y(0)` and `Y(1)` for every user, so every property becomes a frequency
you can count.

And the generator is built to make that comparison exact. From
[document 01](01-what-an-experiment-is.md): the treatment adds
`effect_per_user` to the conversion **probability**, not to the log-odds —
a deliberate choice, so the planted truth is stated in the same units the
analysis reports. A log-odds shift would be more natural statistically and
useless for scoring, because the true effect in percentage points would
then depend on each user's baseline and there would be no single number to
compare the estimate against.

### The tolerance problem, and why it is the crux

```
A simulated false-positive rate will never equal 0.0500 exactly, so every
check needs a tolerance, and a tolerance chosen by eye is a tolerance
chosen to pass.
```

That sentence is the whole methodological argument. A simulation study
with hand-picked tolerances proves nothing, because the author adjusted
them until it passed — usually without noticing, which is worse.

So the tolerance is **derived**:

```python
measured = successes / trials
se = math.sqrt(max(expected * (1.0 - expected), 1e-12) / trials)
tol = k * se                                   # k = 3.0 by default
```

`sqrt(p(1−p)/n)` is the Monte Carlo standard error of the measured rate.
Three of them is about a 99.7% interval. Which means:

> It also means a proof cannot be made to pass by loosening a number —
> only by running more of them, or by fixing the code.

**The tolerance is a function of how much simulation you paid for.** Want
a tighter claim? Run more sims. There is no dial marked "accept this
anyway". That is the difference between a test suite and a demonstration.

`check_mean` does the same with the standard error of the mean.
`check_close` takes an explicit tolerance and exists for comparisons that
are not sampling — reproducing Pocock's 2.413, for instance — and
`check_true` for yes/no structural properties.

### Why `expected` is always written out

```
Every check names the number it expects and where that number comes from:
alpha for a false positive rate, rho^2 for a variance reduction, an
analytic power formula for a power curve. A simulation that only reports
what happened proves nothing; it is the comparison against an
independently derived target that makes it evidence.
```

This is the second half of the argument, and the course is full of cases
where it did real work:

- [Document 04](04-comparing-two-means.md): the pooled test's false
  positive rate **predicted at 0.1785** from the variances and group sizes
  alone, then measured at 0.1802.
- [Document 11](11-ratio-metrics-and-the-delta-method.md): impression-level
  coverage **predicted at 0.6172** from the cluster design effect, then
  measured at 0.6298.
- [Document 08](08-sample-ratio-mismatch.md): a five-point SRM power curve
  predicted and then measured.

Measuring 0.1802 and calling it high says *I observed a number*.
Predicting 0.1785 first says *I understand the mechanism* — and only the
second tells you what will happen on a design nobody has simulated.

It also catches the error the first cannot: if your implementation and
your understanding are both wrong in the same way, a self-comparison
passes.

### The proofs catch their own failures honestly

`_betacf` raises rather than returning a half-converged answer
([document 03](03-distributions-from-scratch.md)). `run_all.py` runs each
proof as a **separate process**, so one crashing cannot take the others
with it and each exit code is its own verdict. Non-zero exit if any claim
fails — it is a gate, not a report.

And when a claim failed during development, the rule was: **measure
first, do not loosen.** Several of my own predicted severities were wrong
and had to be replaced with correct predictions — the delta-method proof
was rebuilt around the cluster design effect after my guessed thresholds
failed, and the guardrail proof's "blocks over 90% of wins" became a
measured 76.7% expressed as a ratio claim. In every case the simulation
was right and I was wrong, which is the only useful direction for that
disagreement to run.

## Mechanism 2: tests — cross-check against SciPy and statsmodels

**665 tests, 503 of them on `statlib` alone.** Every distribution function
compared against SciPy across a parameter grid; every power calculation
against statsmodels.

Why this is a genuinely independent check:

> Two implementations that agree to ten decimals are rarely both wrong in
> the same direction.

And it found what the proofs could not. **Five defects, with every proof
claim passing:**

| defect | why no proof saw it |
|---|---|
| `norm_cdf` returned exactly 0.0 below z ≈ −8.3 | no simulation ever lands at z = −8.5 |
| `t_cdf` kept 2 of 16 digits in its lower tail | ditto |
| `t_sf` was numerically flat around the median at large df | ditto |
| `mde_proportions` returned NaN into a `ValueError` handler | not a statistical property |
| `bootstrap_diff`'s docstring described code that did not exist | not behaviour at all |

The first three share a cause: **a simulation visits the region where data
lands, and a tail probability of 1e-18 changes no coverage rate.** The
proofs were measuring frequencies, and these bugs did not move any
frequency. They would have moved a p-value in a readout.

The tolerances in the tests are **measured, not chosen** — same principle
as the proofs, different mechanism:

```python
assert sl.norm_cdf(z) == pytest.approx(stats.norm.cdf(z), rel=1e-12)   # measured 2.1e-13
assert sl.chi2_cdf(x, df) == pytest.approx(..., rel=1e-12)             # measured 5.7e-14
```

And the test I would point at if asked to show one:

```python
assert worst(1e7) < 1e-8          # measured 3.6e-9
assert worst(1e7) > 1e-10, "the large-df limit is gone; rewrite this test"
```

The second line asserts that a **known limitation still exists.** A suite
that only asserts upper bounds drifts quietly as code changes; one that
pins the shape of its own inaccuracy tells you when something has changed
even for the better.

## Mechanism 3: the scoring harness — blind decisions against a sealed key

The proofs check properties. The tests check numbers. Neither checks
whether the **whole pipeline makes good decisions**, and that is a
different question — one about metric declarations, decision rules and
defaults, none of which is a property of a function.

```python
def _analyse_blind(frame):     # sees the data. never sees the truth.
def _score(readout, truth):    # sees both. cannot change the analysis.
```

Two functions, structurally separated. The analysis **cannot** consult the
answer key, because it is not in scope. That is not a convention; it is
the design, and it is what makes the resulting numbers evidence rather
than self-assessment.

And it earned its place immediately. It found the single most valuable
finding in the repository — a refund guardrail declared the obvious way
blocking **77% of genuine wins**, with every statistic correct throughout
([document 02](02-metrics-and-the-unit-of-analysis.md)).

**No proof could have found that.** Every claim in
`proof_false_positive.py` and `proof_ci_coverage.py` was passing on the
metric in question, because the arithmetic was right. The metric was
wrong, and only something that scored *decisions* could see it.

One detail that matters: the harness scores from
`truth.realised_effect_conversion`, **not** from the `kind` label the trial
was generated with. An early version scored against the label and reported
a 12.5% false-ship rate, which turned out to be partly my labelling —
a +0.02 mobile-only effect is a real ~+0.01 overall effect, so shipping it
is correct. Scoring against what was actually planted rather than what I
called it removed the phantom and left the real defect visible.

## What each mechanism is for

| | catches | blind to |
|---|---|---|
| **proofs** | statistical properties on known truths | numerical accuracy where data never lands; wrong metrics |
| **tests** | numerical accuracy, contracts, regressions | whether the right quantity is being computed |
| **scoring** | wrong quantities, bad defaults, bad decision rules | anything that does not change a decision |

Three mechanisms because no one of them is sufficient, and the evidence
for that is not an argument — it is the list of five defects above, each
found by exactly one of the three, after the other two were green.

## The honest limits of all this

Worth stating, since the point of the document is credibility:

- **The generator is not reality.** Everything is measured against a model
  of an experiment. The model was built to contain the specific hazards
  the library claims to handle, which means it is a fair test of those
  claims and says nothing about hazards neither of us thought of.
- **Agreement with SciPy is not correctness.** It is very strong evidence,
  because the implementations are independent, and it would not catch an
  error both make — for instance a shared convention that is itself
  wrong.
- **Monte Carlo claims have Monte Carlo error.** A `k = 3` tolerance
  passes a broken implementation about 0.3% of the time by chance. With
  104 claims that is roughly a 27% chance that at least one claim passes
  for the wrong reason on any given run, which is why the suite is run
  repeatedly rather than once, and why a claim that fails intermittently
  is treated as a failure.

That last bullet is the kind of thing a project like this usually leaves
out. It is in because the alternative — a suite presented as proof — is
exactly the overclaiming the whole repository was built to avoid.

## Do it yourself

**1. Run the gate:**

```bash
python proofs/run_all.py
```

Six minutes, 104 claims. Then run one on its own:

```bash
python proofs/run_all.py guardrail
```

**2. Try to make a proof pass by cheating.** Open any proof, find a
`check_rate` call, and try to widen the tolerance. You cannot — there is
no tolerance argument to widen, only `k` (which is visible and
indefensible to change) and `trials` (which costs real simulation).
Understanding *why* you cannot is the point of the exercise.

**3. Break the library and watch the right thing fail.** Change
`_CENTRAL_SWITCH` in `statlib.py` to `0.0` and run both suites. Measured,
just now:

```
python -m pytest tests/ -q               17 failed, 648 passed
python proofs/proof_false_positive.py    PASSED
python proofs/proof_power.py             PASSED
python proofs/proof_ci_coverage.py       PASSED
```

Seventeen test failures, and the proofs do not notice at all. The reason
is structural: at the degrees of freedom and `t` values a simulated
experiment actually produces, `t²/(df+t²)` is nowhere near 1e-12, so the
tail form is used either way and no frequency moves. **The proofs stay
green on a real numerical defect.**

That contrast, produced with your own hands in two minutes, is the whole
argument for mechanism 2 — and it is the reason this project has three.
Remember to put the constant back.

**4. Then do the reverse.** Redeclare the refund guardrail as
`MetricSpec("refund_rate", "binary", column="refunded", ...)` in
`scoring/run_scored_trials.py` and run the harness. The tests pass. The
proofs pass. The harness fails, and tells you that three winning
experiments in four are being blocked.

→ [17: A worked example — a hiring funnel](17-a-worked-example-hiring-funnel.md)
