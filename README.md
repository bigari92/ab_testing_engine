# ab_testing_engine

Experiment analysis where every statistical claim is either derived from
first principles or measured by simulation — and usually both.

**Status: library and proof suite complete and passing. Teaching docs,
unit tests and the blind-scoring harness are not written yet.** See
[What is still missing](#what-is-still-missing).

---

## Why this exists

This project was not chosen by taste. It was chosen by measuring what
employers actually ask for and comparing it to what the portfolio could
prove.

On 2026-11-15 I read **128 live job postings** for data / BI / product
analyst roles — 91 from Djinni (`primary_keyword=Data Analyst`, 8 result
pages, analyst titles only) and 37 from DOU (RSS feeds for the Analyst,
Data Science, Big Data, Data Engineer and AI/ML categories across five
experience bands, filtered to the data-BI / product-marketing / HR
analytics title families). For each posting the requirements section was
separated from the benefits section — otherwise "education budget" and
"wellness budget" register as finance experience — and a term appearing
before a "will be a plus" marker was counted as required, after it as
nice-to-have.

Then every requirement was scored against the portfolio, harshly: a
capability with no showable artifact does not count as proof, because it
cannot be put in front of an employer.

The result (`analysis/gap_scan.py` reproduces it):

| requirement | postings | % of 128 | hard requirement in | evidence |
|---|---|---|---|---|
| Statistics / hypothesis testing | 44 | 34% | 36 | **none** |
| A/B testing | 28 | 22% | 23 | **claimed on the CV, nothing behind it** |
| LTV / CAC / unit economics | 27 | 21% | 24 | **none** |
| Segmentation / clustering | 21 | 16% | 20 | **none** |

**44% of the 128 postings require statistical inference or A/B testing.**
The portfolio contained no hypothesis test, no confidence interval and no
p-value anywhere — while `resume/build_resume.py` listed "A/B testing" as
a skill. That gap, between a claim and its evidence, is the one worth
closing first.

This repository closes it.

## The thesis

An analyst's defensible edge is not knowing which function to call. It is
being able to show that the number is right.

So nothing here is asserted. `src/explab/statlib.py` implements the normal,
Student-t and chi-square distributions from the incomplete beta and gamma
functions rather than importing SciPy, so that a p-value can be derived all
the way down. And every property the library claims is measured against a
known truth in `proofs/`, which fails the build when a claim stops holding.

Three examples of what that produces:

- The group-sequential boundaries in `sequential.py` are calibrated by
  Monte Carlo, with no closed form anywhere in the code. They come out at
  **2.411** (Pocock) and **4.563, 3.226, 2.634, 2.281, 2.041**
  (O'Brien-Fleming) for five looks at two-sided alpha 0.05 — against the
  published tables of 2.413 and 4.56, 3.23, 2.63, 2.28, 2.04, computed by
  numerical integration decades ago by people with no connection to this
  code.
- Peeking five times and stopping at the first p < 0.05 raises the false
  positive rate from 5% to **14.2%**, and the effect you stop on is
  **1.98x** the true effect. Both measured, not quoted.
- CUPED on a pre-period covariate correlated 0.77 with the outcome removes
  **59.0%** of the variance, against the 59.0% that rho-squared predicts,
  and turns the sample into the equivalent of **2.44x** as many users.

## What is in the library

```
src/explab/
  statlib.py        normal, Student-t, chi-square from first principles
  inference.py      Welch t, two-proportion, delta-method ratio, bootstrap
  power.py          sample size, MDE, power — the step people skip
  srm.py            sample ratio mismatch, the check that runs first
  cuped.py          variance reduction with pre-experiment data
  sequential.py     peeking, and how to look early without lying
  multiplicity.py   Bonferroni, Holm, Benjamini-Hochberg
  segments.py       heterogeneous effects, with the guards that make them readable
  unit_economics.py ARPU, LTV, CAC, payback — with intervals
  analyze.py        the whole readout, in order, into one decision
  report.py         that decision, rendered for a person
data/generate.py    synthetic experiments with planted defects and a sealed key
proofs/             the simulation studies that make the claims checkable
analysis/gap_scan.py the measurement that chose this project
```

Every module's docstring explains not just what it does but why it does it
that way, including the choices that look wrong and are not — for example
why `two_proportion` deliberately uses a **pooled** standard error for the
p-value and an **unpooled** one for the interval.

## Running it

```bash
pip install -r requirements.txt
```

```bash
python data/generate.py
```

```bash
python proofs/run_all.py
```

A readout, end to end:

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from explab.analyze import MetricSpec, analyze
from explab.report import render_text

frame = pd.read_csv("data/generated/experiment_seed1_n40000.csv")
metrics = [
    MetricSpec("conversion", "binary", column="converted",
               role="primary", practical_threshold=0.004),
    MetricSpec("refund_rate", "binary", column="refunded",
               role="guardrail", direction="decrease"),
    MetricSpec("minutes", "continuous", column="minutes",
               covariate="pre_minutes", role="secondary"),
    MetricSpec("ctr", "ratio", numerator="clicks",
               denominator="impressions", role="secondary"),
]
print(render_text(analyze(frame, metrics)))
```

The readout leads with a decision, puts the sample-ratio check above every
effect estimate, reports intervals before p-values, and distinguishes two
outcomes that most readouts collapse into one: **flat** (the whole interval
is smaller than the effect you would have shipped for — an answer) from
**inconclusive** (the interval still contains effects worth shipping for —
not an answer).

## Proof suite

`python proofs/run_all.py` runs every simulation study. Current state:

| proof | claims | what it establishes |
|---|---|---|
| `proof_false_positive.py` | 6 | every method rejects at exactly alpha under the null, including the data generator itself |
| `proof_ci_coverage.py` | 5 | a 95% interval contains the truth 95% of the time, for means, rates, ratios and the bootstrap |
| `proof_power.py` | 13 | an experiment run at the recommended size detects the stated effect at the stated rate, across a grid |
| `proof_peeking.py` | 7 | the measured cost of peeking, and the winner's curse on the effect size |
| `proof_sequential.py` | 6 | calibrated boundaries hold alpha on data they were never fitted to, and match the published tables |

All 37 claims currently hold. Tolerances are derived from Monte Carlo
standard error rather than chosen, so a proof cannot be made to pass by
loosening a number — only by running more simulations, or by fixing the
code.

## What is still missing

Written down rather than quietly omitted:

- `docs/` — the teaching series. The module docstrings carry most of the
  explanation already, but the step-by-step course is not written.
- `tests/` — unit tests cross-checking `statlib` against SciPy and the
  inference functions against SciPy and statsmodels. These checks have been
  run (agreement to 1e-11 or better on every function) but are not yet
  committed as a suite.
- `scoring/` — the blind-scoring harness: generate many experiments with
  random truths, run `analyze` without the key, open the key, and score the
  decision function's hit rate and error rate.
- `proofs/` — four more are planned: CUPED, the delta method against
  impression-level analysis, SRM detection power, and FDR control.
- CI, and the push to GitHub.

## Licence

MIT. The data is synthetic; there is no client material in this repository.
