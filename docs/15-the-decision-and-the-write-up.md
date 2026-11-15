# 15 — The decision and the write-up

## The question

Every document before this one produced a number. This one produces a
**decision**, and the shape of the write-up is part of the decision —
because what a reader can see first is what they will act on.

## Four outcomes, not two

Most readouts offer *significant* or *not significant*. `analyze` returns
four, and the extra two are where the value is:

| decision | meaning |
|---|---|
| **ship** | the primary metric moved by an amount worth having |
| **do-not-ship** | a guardrail was harmed, or the primary moved the wrong way |
| **flat** | there is no effect large enough to matter. **An answer.** |
| **inconclusive** | the experiment did not answer the question |

The split between the last two is the whole point, and it needs
`practical_threshold` declared in advance
([document 02](02-metrics-and-the-unit-of-analysis.md)). Same truth — no
effect planted — only the sample size differs:

```
400,000 users -> flat
   primary metric conversion is flat: the whole interval
   [-0.003676, +3.471e-05] lies inside the practical threshold of
   +/-0.004. This is an answer, not a failure: there is no effect of a
   size worth shipping for.

8,000 users -> inconclusive
   primary metric conversion is not significant, but the interval
   [-0.01455, +0.01175] still contains effects beyond the practical
   threshold of 0.004. The experiment has not answered the question;
   it needs more units, not a decision.
```

Both are "not significant". One says **stop asking**; the other says
**the money has not been spent yet**. Collapsing them into "no effect" is
the most expensive wording error in commercial analytics, because it
retires questions that were never tested.

Across 400 blind trials, true nulls came out *flat* **32.9%** of the time
and *inconclusive* the rest — both correct, and the proportion is a fact
about how much traffic the experiments had.

## The order of the readout is load-bearing

```
DECISION: SHIP
  - primary metric conversion moved +0.0109969 (+10.96%), interval
    [+0.008301, +0.01369] excludes zero at 95%

------------------------------------------------------------------
ASSIGNMENT CHECK (runs before anything is believed)
------------------------------------------------------------------
sample ratio check: PASS  (p = 0.116, threshold 0.001)
  chi-square 2.464 on 1 df, n = 200,000
  worst arm control: observed 49.8245% vs expected 50.0000%
  this sample could detect a share deviation of 0.4620% or more at 80% power
```

**Decision first.** Not because it is the most important thing
statistically, but because it is the only line some readers will see. If
the decision is buried under methodology, the reader forms one from the
biggest number on the page.

**Assignment check second, before any effect.** The section header says
why: *runs before anything is believed*. If SRM fails,
[document 08](08-sample-ratio-mismatch.md) applies and there is no result
to read. A test in `tests/test_analyze.py` asserts that ordering against
the rendered text, because a refactor that moved it would silently undo
the argument.

**Then the primary metric, alone, labelled "declared in advance, not
corrected".** One metric decides. The label is there so a reader can
check that it was not chosen after the fact.

**Then guardrails, labelled "uncorrected on purpose: a correction would
hide harm"** — the 21.7-point measurement from
[document 13](13-segments-multiplicity-and-simpson.md), compressed into a
section header so nobody has to ask.

**Then secondary metrics, labelled EXPLORATORY**, with the BH guarantee
spelled out at the bottom of the block:

```
  Benjamini-Hochberg across this group. These cannot change the decision;
  they are hypotheses for the next experiment.
  ...
  E[false share of rejections] <= 0.05 (independent or positively
  dependent tests)
```

"Cannot change the decision" is a structural claim, not advice:
`analyze` derives the decision from the primary and the guardrails only.

## Every metric line carries its own sensitivity

```
  conversion  (binary, increase is better)
    control 0.100352  ->  treatment 0.111349
    effect  +0.0109969  [+0.00830097, +0.0136929]  at 95%
    relative +10.96%  [+8.27%, +13.64%]
    p 1.325e-15   method two_proportion
    this sample could detect 0.00389263 or larger at 80% power
    practical threshold +/-0.004: interval still admits an effect worth
    shipping for
```

Seven things on every metric, and each one exists because of a failure
measured earlier in this course:

1. **Both levels**, so "+10.96%" can be checked against reality.
2. **Absolute effect with interval** — [document 06](06-what-an-interval-means.md).
3. **Relative effect with interval**, because a 2-point lift on a 4% base
   and on a 40% base are the same absolute number and different news.
4. **The p-value**, floored rather than printed as zero
   ([document 03](03-distributions-from-scratch.md)).
5. **The method**, so the reader knows which document applies.
6. **The MDE** — *this sample could detect 0.0039 or larger* — so a null
   result is readable. [Documents 02](02-metrics-and-the-unit-of-analysis.md)
   and [08](08-sample-ratio-mismatch.md) are both about this habit.
7. **The practical threshold and what the interval does about it**, which
   is the flat/inconclusive distinction made visible per metric.

## Notes are where the methodology lives

The guardrail in the real readout above carries three:

```
note: randomisation unit is the row, denominator unit is finer; the delta
      method keeps the standard error at the randomised unit
note: delta-method standard error: includes the numerator/denominator
      covariance that a per-row mean would drop
note: 178826 units have a zero denominator and were kept; dropping them
      would silently change the question
```

That third note is [document 11](11-ratio-metrics-and-the-delta-method.md)
in one line, printed with a count. A reader who wondered "shouldn't you
drop the non-converters?" gets the answer without asking, and a reader who
*would not* have wondered is told that a choice was made.

And on a secondary metric:

```
note: CUPED on pre_minutes: 59.2% of variance removed, worth 2.45x the
      sample
```

59.2% removed implies a correlation of about 0.77, and
`1/(1 − 0.592) = 2.45` — the sample-size multiple from
[document 10](10-variance-reduction-with-cuped.md), quoted in the units a
stakeholder cares about rather than as a variance ratio.

## The NaN sweep

A readout is a rendering, and renderings leak. `TestResult.__str__`,
`PowerPlan.__str__` and `SRMResult.__str__` were each capable of printing
`nan` into a report:

```python
if math.isnan(self.relative_lift):
    relative_line = (
        "  relative undefined (control mean is zero; the absolute "
        "change above is the whole story)"
    )
```

A relative lift needs something to be relative to; from a control mean of
zero it is undefined, and **that fact is itself worth reading** — the
treatment produced behaviour that did not exist before, which no
percentage can express. `+nan%` is the same credibility problem as a
p-value of exactly zero.

`tests/test_contracts.py` sweeps **seven separate renderings** for a
leaked NaN, including `test_no_readout_in_the_repository_prints_nan`. The
sweep exists because the same mistake was independently available in three
places and was being made in all three. A guard applied once is a fix; a
sweep is a policy.

## The scoring harness: does any of this work?

Everything above is a design argument. `scoring/run_scored_trials.py`
tests it, by generating experiments with randomly chosen planted truths,
analysing them **blind**, and scoring the decisions against a sealed key.

The structure is the point:

```python
def _analyse_blind(frame):     # sees the data. never sees the truth.
def _score(readout, truth):    # sees both. cannot change the analysis.
```

Two functions, separated so that the analysis **cannot** consult the
answer. That separation is what makes the numbers below evidence rather
than self-report — and it is why the harness was able to find the refund
guardrail defect that the 104 proof claims could not.

400 trials:

| | rate | |
|---|---|---|
| false ship (shipped a null or a harm) | **4.07%** | 9/221 |
| harm missed (shipped a real regression) | **8.77%** | 5/57 |
| false flat (called flat when a real effect existed) | **0.00%** | 0/101 |
| SRM missed | **7.89%** | 3/38 |
| SRM false alarm | **0.00%** | 0/362 |
| real win found, when powered for it | **94.12%** | 80/85 |

`SCORING PASSED`. And two diagnostics that are not pass/fail:

- **shipped sub-threshold: 22.50%** — of the ships, nearly a quarter were
  for effects below the practical threshold. These are not errors by the
  harness's definition, but they are decisions a stricter rule would not
  have made.
- **decisive on a null: 32.93%** — the *flat* rate from the top of this
  document, which is a measure of how much traffic the trials had rather
  than of the library.

Those two rows are in the output because they are the ones that would
change if someone made the decision rule more aggressive. A scorecard that
only reports the metrics you pass is a scorecard you tuned.

## Do it yourself

**1. Produce the readout.** This is the whole project in fifteen lines:

```python
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "data")
from generate import generate_experiment
from explab.analyze import MetricSpec, analyze
from explab.report import render_text

frame, truth = generate_experiment(n_users=200_000, seed=404,
                                   effect_conversion=0.01, effect_minutes=0.4)
metrics = [
    MetricSpec("conversion", "binary", column="converted", role="primary",
               direction="increase", practical_threshold=0.004),
    MetricSpec("refund_rate", "ratio", numerator="refunded",
               denominator="converted", role="guardrail", direction="decrease"),
    MetricSpec("minutes", "continuous", column="minutes", role="secondary",
               covariate="pre_minutes"),
    MetricSpec("revenue", "continuous", column="revenue", role="secondary"),
]
print(render_text(analyze(frame, metrics), title="checkout redesign v3"))
print(truth)          # the sealed key, AFTER you have read the readout
```

Read the readout first and write down your decision. **Then** print
`truth`. Doing it in that order once is worth more than reading this
document twice.

**2. Produce flat and inconclusive from the same truth:**

```python
for n in (400_000, 8_000):
    frame, _ = generate_experiment(n_users=n, seed=77, effect_conversion=0.0)
    r = analyze(frame, [metrics[0]])
    print(n, "->", r.decision, "\n   ", r.reasons[0], "\n")
```

**3. Remove the practical threshold** and run it again. Watch the
*flat* case become *inconclusive*, and read what `analyze` says about why
it cannot make the distinction. That is the cost of not declaring a
threshold, stated by the tool rather than by a colleague six weeks later.

**4. Run the harness.** `python scoring/run_scored_trials.py`. It takes a
few minutes and it is the only thing in this repository that could tell
you the library is wrong in a way the proofs would not catch.

→ [16: How the proofs work](16-how-the-proofs-work.md)
