# Start here

This is a course, not documentation. It is written so that you can rebuild
every piece of `src/explab/` yourself and know why each line is there.

I wrote it because I claimed "A/B testing" as a skill on my CV and could
not point at a single p-value I had produced. That gap is the whole reason
this repository exists — see the measurement in the top-level
[README](../README.md) — and the fastest way to close it permanently is to
build the thing from the bottom and be able to explain it.

## How to use this

Read in order. Each document follows the same shape:

1. **The question** a real readout has to answer.
2. **The derivation**, from something you already accept.
3. **The code** in this repository that implements it, quoted, with the
   choices that look wrong and are not.
4. **The measurement** — which file in `proofs/` establishes that the code
   does what the derivation says, and what number it got.
5. **Do it yourself** — something to run or re-derive. Not optional if you
   want this to stick.

The order is not the order a textbook would use. It is the order a readout
is built in: what you are measuring, then whether the data is trustworthy,
then the test, then what the answer means, then the decision.

## What you need

```bash
pip install -r ../requirements.txt
```

NumPy and pandas for the library. SciPy, statsmodels and pytest are test
dependencies only — the library never imports them at runtime, and that is
deliberate: the point is that nothing here is taken on faith from another
package.

Python 3.10 or later (the type syntax `float | None` needs it).

## The three things this project is actually about

**One.** A p-value is not a thing you get from a library. It is an integral
you can compute, and `statlib.py` computes it from the incomplete beta
function so that you can follow it all the way down. Document 03 derives
that.

**Two.** Being able to show a number is right matters more than knowing
which function produced it. Every property this library claims is measured
against a known truth in `proofs/`, and the proof suite fails the build
when a claim stops holding. Document 16 explains how that works and why
the tolerances are derived rather than chosen.

**Three.** Most wrong analyses are not wrong arithmetic. They are correct
arithmetic on the wrong quantity. The sharpest example in this repository
was found by its own scoring harness: a refund guardrail declared the
obvious way blocked 77% of genuine wins, with every statistic correct
throughout. Document 02 is about that, and it is the most useful document
here.

## The map

| # | document | what it is for |
|---|---|---|
| 01 | [What an experiment is](01-what-an-experiment-is.md) | randomisation, the counterfactual, what it buys you |
| 02 | [Metrics and the unit of analysis](02-metrics-and-the-unit-of-analysis.md) | the denominator trap; primary, guardrail, secondary |
| 03 | [Distributions from scratch](03-distributions-from-scratch.md) | normal, t, chi-square from the incomplete beta and gamma |
| 04 | [Comparing two means](04-comparing-two-means.md) | Welch, and why never the pooled t-test |
| 05 | [Comparing two rates](05-comparing-two-rates.md) | two proportions, pooled vs unpooled |
| 06 | [What an interval means](06-what-an-interval-means.md) | coverage, and the bootstrap when the formula runs out |
| 07 | [Power and sample size](07-power-and-sample-size.md) | MDE, planning, and the conversation this replaces |
| 08 | [Sample ratio mismatch](08-sample-ratio-mismatch.md) | the check that runs before anything is believed |
| 09 | [Peeking and sequential tests](09-peeking-and-sequential-tests.md) | the measured cost, and how to look early honestly |
| 10 | [Variance reduction with CUPED](10-variance-reduction-with-cuped.md) | more power from data you already have |
| 11 | [Ratio metrics and the delta method](11-ratio-metrics-and-the-delta-method.md) | when the denominator is finer than the unit |
| 12 | [Outliers and heavy tails](12-outliers-and-heavy-tails.md) | revenue, and what to do about one whale |
| 13 | [Segments, multiplicity and Simpson](13-segments-multiplicity-and-simpson.md) | slicing without fooling yourself |
| 14 | [Unit economics](14-unit-economics.md) | ARPU, LTV, CAC, payback — with intervals |
| 15 | [The decision and the write-up](15-the-decision-and-the-write-up.md) | flat vs inconclusive, and the order of a readout |
| 16 | [How the proofs work](16-how-the-proofs-work.md) | simulation studies, derived tolerances, blind scoring |
| 17 | [A worked example: a hiring funnel](17-a-worked-example-hiring-funnel.md) | the same machinery on recruitment data |
| 18 | [What this does not cover](18-what-this-does-not-cover.md) | the honest boundary of the project |

## If you only read three

03, 02, 16 — in that order of difficulty, not importance. 02 will change
how you declare metrics tomorrow. 16 is the part that makes the rest
defensible to somebody who does not trust you yet.

## A note on the mistakes in here

This course records several defects found in this repository's own code,
by its own tests, after the code was written and the proofs were passing:

- `norm_cdf` returned exactly `0.0` for z below −8.3, and the docstring
  had a confident theoretical explanation for the resulting inaccuracy
  that was simply wrong.
- `t_cdf` computed its lower tail by cancellation and kept two correct
  digits out of sixteen at t = −8.
- `t_sf` was numerically flat around the median at large degrees of
  freedom while claiming in its own docstring to be accurate "from the
  median all the way out".
- `mde_proportions` returned `NaN` into a function whose caller was
  already catching `ValueError` for exactly that case.
- A refund guardrail blocked 77% of winning experiments.
- `bootstrap_diff`'s docstring described a p-value computed by bisecting
  on the confidence level. The function has always computed a doubled tail
  share instead. Found by writing document 06 — not by any test, because a
  test asserts behaviour and this was a false claim about *mechanism*.
- `proof_cuped.py` printed the caption `theory says 1 - rho^2` above a
  number that was `rho^2`. The claim it checked was correct; only the
  caption was wrong.
- Ratio metrics reported no minimum detectable effect at all —
  `analyze()` set `detectable = None` and said nothing. That left a
  guardrail's "no harm detected" unreadable in exactly the case document
  02 is about, since the refund guardrail there *is* a ratio metric.
- `srm.py` computed its p-value as `1 - chi2_cdf(...)`, which on one
  degree of freedom returns exactly `0.0` for every chi-square past about
  100 — where the truth is 1.5e-23. The check whose own docstring calls an
  11-sigma deviation ordinary was reporting those deviations as `p = 0`.
  There was no `chi2_sf`; now there is.

The last three were found by writing documents 10, 17 and 17. Writing the
explanation is the cheapest defect-finding tool in this project, because a
test asserts behaviour while an explanation has to assert the *reason* —
and the reason is where the misunderstandings live.

They are in here on purpose, with the measurements that found them. A
course that only shows the finished thing teaches you the finished thing.
A course that shows what the guards are for teaches you to build the next
one.

→ [01: What an experiment is](01-what-an-experiment-is.md)
