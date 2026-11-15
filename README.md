# ab_testing_engine

Experiment analysis where every statistical claim is either derived from
first principles or measured by simulation — and usually both.

**Status: complete.** Library, 11 proofs (104 claims), 716 tests, a
blind-scoring harness, and a 19-document teaching course. Everything
passes; the known limits are listed in
[docs/18](docs/18-what-this-does-not-cover.md) rather than omitted.

**New here?** The course starts at
[docs/00-start-here.md](docs/00-start-here.md). It is written so you can
rebuild every piece of `src/explab/` yourself and know why each line is
there.

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

So nothing here is asserted. `src/explab/statlib.py` implements the
normal, Student-t and chi-square distributions from the incomplete beta
and gamma functions rather than importing SciPy, so that a p-value can be
derived all the way down. Every property the library claims is measured
against a known truth in `proofs/`, which fails the build when a claim
stops holding. And every number is cross-checked against an independent
implementation in `tests/`.

Four examples of what that produces:

- The group-sequential boundaries in `sequential.py` are calibrated by
  Monte Carlo, with no closed form anywhere in the code. They come out at
  **2.411** (Pocock) and **4.563, 3.226, 2.634, 2.281, 2.041**
  (O'Brien–Fleming) for five looks at two-sided alpha 0.05 — against the
  published tables of 2.413 and 4.56, 3.23, 2.63, 2.28, 2.04, computed by
  numerical integration decades ago by people with no connection to this
  code.
- Peeking five times and stopping at the first p < 0.05 raises the false
  positive rate from 5% to **14.2%**, and the effect you stop on is
  **1.98×** the true effect. Both measured, not quoted.
- Analysing a click-through rate at the impression level — correct
  arithmetic, wrong unit — gives a 95% interval that covers the truth
  **63.0%** of the time at realistic user heterogeneity. Predicted at
  61.7% from the cluster design effect before the simulation ran.
- A refund guardrail declared the obvious way blocks **76.7%** of genuine
  winning experiments, with every statistic in the readout correct. Found
  by this project's own scoring harness, not by its author.

## The three verification mechanisms

Three, because no one of them is sufficient — and the evidence for that
claim is the list of eight defects below, each found by exactly one of the
three while the other two were green.

| | catches | blind to |
|---|---|---|
| **`proofs/`** — 11 files, 104 claims, simulation against a known truth | statistical properties | numerical accuracy where data never lands; wrong metrics |
| **`tests/`** — 716 tests against SciPy and statsmodels | numerical accuracy, contracts, regressions | whether the right quantity is being computed |
| **`scoring/`** — blind decisions against a sealed key | wrong quantities, bad defaults, bad decision rules | anything that does not change a decision |

Tolerances everywhere are **derived, not chosen** — from Monte Carlo
standard error in the proofs, from measured agreement in the tests — so a
claim cannot be made to pass by loosening a number, only by running more
simulations or fixing the code. [docs/16](docs/16-how-the-proofs-work.md)
explains the methodology and its limits.

### The eight defects they found

Recorded because a project that shows only the finished thing teaches you
the finished thing:

| defect | found by |
|---|---|
| `norm_cdf` returned exactly `0.0` below z ≈ −8.3, with a confident and wrong docstring explaining the resulting inaccuracy | tests |
| `t_cdf` computed its lower tail by cancellation, keeping 2 of 16 digits at t = −8 | tests |
| `t_sf` was numerically flat around the median at large df while claiming accuracy "from the median all the way out" | tests |
| `mde_proportions` returned `NaN` into a caller already catching `ValueError` for exactly that case | tests |
| a refund guardrail blocked 76.7% of winning experiments | scoring |
| `bootstrap_diff`'s docstring described a p-value computed by bisection; the code has always used a doubled tail share | writing docs/06 |
| ratio metrics reported no minimum detectable effect at all | writing docs/17 |
| `srm.py` computed `1 - chi2_cdf(...)`, returning `p = 0` for any chi-square past 100 — in the one check whose docstring calls an 11-sigma deviation ordinary | writing docs/17 |

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
tests/              the cross-check against SciPy and statsmodels
scoring/            blind trials scored against the key
docs/               the course: 19 documents, 00 through 18
analysis/gap_scan.py             the measurement that chose this project
analysis/hiring_funnel_example.py  the same machinery on a staffing funnel
```

Every module's docstring explains not just what it does but why it does it
that way, including the choices that look wrong and are not — for example
why `two_proportion` deliberately uses a **pooled** standard error for the
p-value and an **unpooled** one for the interval.

## Running it

```bash
pip install -r requirements.txt
```

The gate — every proof, non-zero exit if any claim fails:

```bash
python proofs/run_all.py
```

The cross-check:

```bash
python -m pytest tests/ -q
```

The blind scoring harness:

```bash
python scoring/run_scored_trials.py
```

A worked example on a recruitment funnel, end to end:

```bash
python analysis/hiring_funnel_example.py
```

A readout, in code:

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
]
print(render_text(analyze(frame, metrics), title="checkout redesign v3"))
print(truth)        # the sealed key — after you have read the readout
```

The readout leads with a decision, puts the sample-ratio check above every
effect estimate, reports intervals before p-values, carries each metric's
minimum detectable effect so that a null result is readable, and
distinguishes two outcomes that most readouts collapse into one: **flat**
(the whole interval is smaller than the effect you would have shipped for
— an answer) from **inconclusive** (the interval still contains effects
worth shipping for — not an answer).

Note the guardrail declaration. `refund_rate` is a **ratio** with
`converted` as its denominator, not a binary on `refunded`, and
[docs/02](docs/02-metrics-and-the-unit-of-analysis.md) is the whole story
of why that one line matters more than anything else in the file.

## Proof suite

| proof | claims | what it establishes |
|---|---|---|
| `proof_false_positive.py` | 6 | every method rejects at exactly alpha under the null, including the data generator itself |
| `proof_ci_coverage.py` | 5 | a 95% interval contains the truth 95% of the time, for means, rates, ratios and the bootstrap |
| `proof_power.py` | 13 | an experiment run at the recommended size detects the stated effect at the stated rate, across a grid |
| `proof_peeking.py` | 7 | the measured cost of peeking, and the winner's curse on the effect size |
| `proof_sequential.py` | 6 | calibrated boundaries hold alpha on data they were never fitted to, and match the published tables |
| `proof_welch_vs_pooled.py` | 10 | the pooled t-test's false positive rate goes to 18% one way and 0.5% the other on the same variance ratio — predicted from the design, then measured |
| `proof_srm.py` | 12 | the SRM check holds 0.001, and the `detectable` field it advertises really is its 80% point |
| `proof_cuped.py` | 11 | CUPED removes rho-squared of the variance and a post-treatment covariate erases 100% of a real effect |
| `proof_delta_method.py` | 11 | impression-level coverage collapses to the cluster design effect's prediction; the delta-method SE equals the realised spread |
| `proof_multiplicity.py` | 11 | BH holds FDR at alpha·m₀/m exactly while its familywise rate runs at 2.6× alpha |
| `proof_guardrail_denominator.py` | 12 | the per-user refund guardrail's false alarm rate, and the exact algebraic decomposition behind it |

**All 104 claims hold**, in about 6.5 minutes.

## Scoring harness

400 blind trials, analysis structurally separated from the key:

| | rate | |
|---|---|---|
| false ship | 4.07% | 9/221 |
| harm missed | 8.77% | 5/57 |
| false flat | 0.00% | 0/101 |
| SRM missed | 7.89% | 3/38 |
| SRM false alarm | 0.00% | 0/362 |
| real win found, when powered for it | 94.12% | 80/85 |

Plus two diagnostics that are not pass/fail, because they are the ones
that would move if the decision rule were made more aggressive: **22.50%**
of ships were for sub-threshold effects, and true nulls were called *flat*
**32.93%** of the time.

## The course

| # | document | what it is for |
|---|---|---|
| 00 | [Start here](docs/00-start-here.md) | how to use it, and the defects it records |
| 01 | [What an experiment is](docs/01-what-an-experiment-is.md) | randomisation, the counterfactual, what it buys you |
| 02 | [Metrics and the unit of analysis](docs/02-metrics-and-the-unit-of-analysis.md) | the denominator trap; primary, guardrail, secondary |
| 03 | [Distributions from scratch](docs/03-distributions-from-scratch.md) | normal, t, chi-square from the incomplete beta and gamma |
| 04 | [Comparing two means](docs/04-comparing-two-means.md) | Welch, and why never the pooled t-test |
| 05 | [Comparing two rates](docs/05-comparing-two-rates.md) | two proportions, pooled vs unpooled |
| 06 | [What an interval means](docs/06-what-an-interval-means.md) | coverage, and the bootstrap when the formula runs out |
| 07 | [Power and sample size](docs/07-power-and-sample-size.md) | MDE, planning, and the conversation this replaces |
| 08 | [Sample ratio mismatch](docs/08-sample-ratio-mismatch.md) | the check that runs before anything is believed |
| 09 | [Peeking and sequential tests](docs/09-peeking-and-sequential-tests.md) | the measured cost, and how to look early honestly |
| 10 | [Variance reduction with CUPED](docs/10-variance-reduction-with-cuped.md) | more power from data you already have |
| 11 | [Ratio metrics and the delta method](docs/11-ratio-metrics-and-the-delta-method.md) | when the denominator is finer than the unit |
| 12 | [Outliers and heavy tails](docs/12-outliers-and-heavy-tails.md) | revenue, and what to do about one whale |
| 13 | [Segments, multiplicity and Simpson](docs/13-segments-multiplicity-and-simpson.md) | slicing without fooling yourself |
| 14 | [Unit economics](docs/14-unit-economics.md) | ARPU, LTV, CAC, payback — with intervals |
| 15 | [The decision and the write-up](docs/15-the-decision-and-the-write-up.md) | flat vs inconclusive, and the order of a readout |
| 16 | [How the proofs work](docs/16-how-the-proofs-work.md) | derived tolerances, blind scoring, and the limits |
| 17 | [A worked example: a hiring funnel](docs/17-a-worked-example-hiring-funnel.md) | the same machinery on recruitment data |
| 18 | [What this does not cover](docs/18-what-this-does-not-cover.md) | the honest boundary of the project |

Each document follows the same shape: the question a real readout has to
answer, the derivation, the code that implements it, the measurement that
establishes it, and something to run or re-derive yourself. **Every
exercise in the course was executed and its real output pasted back in** —
including the three where my prediction was wrong and the measurement
corrected it.

## Licence

MIT. The data is synthetic; there is no client material in this
repository.
