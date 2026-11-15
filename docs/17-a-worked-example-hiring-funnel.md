# 17 — A worked example: a hiring funnel

## The question

Everything so far has run on a generic product experiment: conversion,
revenue, refunds. Does any of it transfer to a business that does not have
a checkout?

This document runs the whole pipeline on a **staffing funnel**, which is
the domain I actually work in, and which has one structural feature a
checkout does not:

> **Every stage is a denominator for the next one.**

That makes [document 02](02-metrics-and-the-unit-of-analysis.md)'s trap
not an edge case but the default condition.

The code is [`analysis/hiring_funnel_example.py`](../analysis/hiring_funnel_example.py)
and it runs in a few seconds:

```bash
python analysis/hiring_funnel_example.py
```

## "Offer rate" has four meanings

```
offers / candidates sourced        sourcing + screening + interview
offers / CVs submitted to client   screening + interview
offers / candidates interviewed    interview performance only
offers / vacancies open            demand-side, a different question
```

Four quantities, one phrase. A change that improves screening moves the
first three in ways that have nothing to do with the thing being measured,
and the fourth not at all. **Picking one without saying which is how two
people report "offer rate" from the same database and disagree by a factor
of three** — which is a thing I have watched happen.

## The experiment

A recruitment agency introduces a structured pre-screening call before a
CV goes to the client. The hypothesis: clients interview more of the CVs
they receive, because the weak ones are filtered out earlier.

| | |
|---|---|
| randomisation unit | the candidate, at the moment of sourcing |
| primary | CV → interview rate, **per CV submitted** |
| guardrail | offers per interview |
| guardrail | probation pass rate per hire |
| secondary | CV submit rate, offers per candidate |

The planted truth — which `_score` sees and the analysis never does:

```
interview rate per CV submitted      genuinely rises     +0.06
offers per interview                 does NOT change      0.00
CVs submitted per candidate sourced  FALLS               -0.07
```

The screening call does not make any individual candidate better. It
declines to submit the weakest ones. So the lift is real **and** it has a
cost, and both are planted on purpose.

## Step 1: plan it, before running anything

```
7,732 units per group (15,464 total)
sized to detect 0.02 (+7.69% relative)
at 80% power, alpha 0.05 two-sided
assumes:
  - baseline rate = 26.0000%
  - treatment rate under the alternative = 28.0000%
  - unpooled variance, matching two_proportion's interval

  those are CVs SUBMITTED, not candidates sourced. At a 42% submit rate
  that is 18,410 candidates to source per arm, 36,819 in total.
```

That last conversion is the step that turns a statistics answer into a
staffing answer. Nobody sources CVs; they source candidates. 15,464
"units" is an abstraction and **36,819 candidates** is a conversation about
whether this experiment is possible at all — which, for most agencies,
it is not at that MDE.

This is [document 07](07-power-and-sample-size.md) doing the only thing
that matters: making the cost visible before the money is spent. If the
answer is "we cannot", that is worth knowing in the planning meeting
rather than after a quarter of inconclusive data.

## Step 2: the primary metric is a ratio, and that is not a detail

```python
MetricSpec("interview_rate_per_cv", "ratio",
           numerator="interviewed", denominator="submitted",
           role="primary", direction="increase",
           practical_threshold=0.02)
```

"Interview rate" means interviews **per CV submitted**, and `submitted` is
not the randomisation unit — we randomised candidates. So the standard
error belongs at the candidate level, and the delta method is what puts it
there ([document 11](11-ratio-metrics-and-the-delta-method.md)).

```
  interview_rate_per_cv  (ratio, increase is better)
    control 0.268345  ->  treatment 0.329598
    effect  +0.061253  [+0.030209, +0.0922969]  at 95%
    relative +22.83%  [+11.26%, +34.39%]
    p 0.0001101   method ratio_delta
    this sample could detect 0.0443744 or larger at 80% power
    practical threshold +/-0.02: interval still admits an effect worth
    shipping for
    note: 5616 units have a zero denominator and were kept; dropping
    them would silently change the question
```

The planted truth was +0.06; the estimate is +0.0613.

Now read the MDE line against the threshold below it: **this sample could
detect 0.044, and the effect worth acting on was declared at 0.02.** The
experiment found its effect only because the effect was three times the
size anyone thought worth having. Had the real lift been +0.025 — still
worth shipping by the team's own declared standard — this design would
have missed it most of the time.

That is not visible anywhere else in the readout, and until I wrote this
document it was not visible at all: ratio metrics printed no MDE line.
See "What writing this document found" below.

And the last note: 5,616 candidates were never submitted to a client at
all. **They are kept**, because dropping them would change the question
from "interviews per CV across all sourced candidates" to "…among
candidates we chose to submit" — and whether we submit is exactly what the
treatment changed. That is one `WHERE` clause away from a wrong answer,
and it is the kind of `WHERE` clause that gets added by someone being
careful.

## Step 3: the trap, on three denominators at once

Offer quality per interview was planted at **exactly zero**. Here are
three ways to report "offer rate" on that data:

```
offers per CANDIDATE SOURCED  -2.51%  [-22.47%, +17.45%]   p 0.81
offers per CV SUBMITTED      +18.39%  [ -2.99%, +39.77%]   p 0.09
offers per INTERVIEW          -3.61%  [-19.81%, +12.59%]   p 0.66
```

Only the third answers "did we get worse at interviews", and it is
correctly flat. The second is measuring the screening change wearing the
name of an interview metric.

### The finding a client would actually want

**Offers per candidate sourced did not move.** The lift and the cost
cancel: we send 17.7% fewer CVs and convert a higher share of them, for
the same number of offers.

```
cv_submit_rate   -17.66%  [-22.50%, -12.81%]   p 1.0e-12
```

Whether that is a win depends on **whose time is scarce** — the client's
interview slots, or our sourcing capacity. That is a business question,
and the right thing for a readout to do is pose it, not answer it. Note
that `cv_submit_rate` is a *declared secondary metric* here, not something
discovered when a client asked why volume had dropped. Declaring the known
cost of your change is how you avoid having it reported back to you as a
surprise.

### Isolating the mechanical rise

Set `screening_submit_cost=0` and the cancellation goes away:

```
offers per CANDIDATE SOURCED  +20.07%  [-0.95%, +41.09%]   p 0.061
offers per INTERVIEW           -3.33%  [-18.77%, +12.11%]  p 0.67
```

Offers per candidate rises 20% relative. Offers per interview is flat.
**Nobody got better at interviewing — more people reached one.** A readout
quoting the first number as "offer conversion improved" would be wrong in
a way no statistic in it was wrong.

And read that first p-value: **0.061.** By its own interval the rise is
not established at 9,000 candidates. The composition shift is real and
this funnel is too small to resolve it — which is the ordinary condition
of a staffing experiment, and the reason step 1 comes first.

## Step 4: the SRM that recruitment data actually has

The most common real cause of a sample ratio mismatch in staffing is not
exotic: **a recruiter workflow where only one arm gets logged properly.**
The script models it as 6% of control candidates silently missing.

The same defect at three scales:

| candidates | SRM p | verdict | decision |
|---|---|---|---|
| 9,000 | 0.0587 | PASS | ship |
| 40,000 | 1.58e-11 | **FAIL — SRM** | invalid |
| 200,000 | 2.86e-45 | **FAIL — SRM** | invalid |

**The assignment is equally broken in all three rows.** At 9,000
candidates the check cannot see it: it could only detect a 2.18% share
deviation at 80% power, and a 6% one-sided logging gap is about a 1.5%
share deviation. At 40,000 it is caught and the readout refuses to report
a decision at all.

This is [document 08](08-sample-ratio-mismatch.md) in the domain where it
bites hardest, because **staffing funnels are small.** So the honest
reading of a PASS here is "no imbalance above 2.2%", which is not the same
sentence as "the assignment was sound". A staffing experiment rarely has
the volume to tell those apart — and that is an argument for **controlling
the logging** rather than trusting the check.

## What writing this document found

Two defects, both in code that had been passing 104 proof claims and 665
tests:

**1. Ratio metrics reported no MDE.** `analyze()` set
`detectable = None` for every ratio metric and said nothing about it. Two
of the three guardrails here are ratios, so both came back flat with no
sensitivity attached — the exact thing
[document 02](02-metrics-and-the-unit-of-analysis.md) says never to do,
in a readout that claims to do it. Fixed by adding
`mde_from_standard_error`, which is the general identity from
[document 07](07-power-and-sample-size.md):

```python
return float((norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)) * se)
```

No closed form was possible — the design effect makes `sd·sqrt(2/n)` wrong
by `sqrt(DE)` — but the standard error the delta method **already
computed** carries it, so `effect/se ≥ z_α + z_power` applies directly.

**2. The SRM p-value underflowed to exactly zero.** The 200,000-candidate
row above originally printed `p = 0`. `srm.py` computed
`1 - chi2_cdf(chi2, df)`, and on one degree of freedom that returns
exactly `0.0` for every chi-square past about 100, where the truth is
1.5e-23.

There was no `chi2_sf` — the third member of the `norm_sf` / `t_sf` family
was simply missing, and the one function that needed it most was the one
whose own docstring describes an 11-sigma deviation as the ordinary case
at two million users. **The check whose job is to flag impossible splits
was reporting them with a p-value of zero**, in the readout where it is
most likely to be read aloud. Now 2.86e-45, agreeing with SciPy to
2.7e-14, with a regression test that asserts both what the old expression
returned and what is true.

Both found by running the machinery on a different shape of problem. That
is worth more than it sounds: the generic generator exercises the hazards
I built it to contain, and a funnel where every stage is a denominator
exercised two I had not thought about.

## Do it yourself

**1. Run it and read the readout before the key:**

```bash
python analysis/hiring_funnel_example.py
```

**2. Map your own funnel's denominators.** Write down every stage, and for
each metric you report, which stage is the denominator. Then mark the ones
the change you are testing could move. In my experience that marks most of
them, and the unmarked ones are the metrics worth having.

**3. Change the planted truth** and see whether the readout catches you
out. Set `offer_quality_change=-0.04` in `generate_funnel` — a real drop in
interview performance — and check whether the guardrail fires, and what
its MDE says about whether it could have. Then make the funnel bigger and
try again.

**4. Find your real logging gap.** For any funnel stage at your own work,
count rows per arm and run `srm_check`. Then read the `detectable` line
and ask whether your volume could have caught a gap worth caring about.
The answer is often no, and that is actionable: it means the logging has
to be verified rather than monitored.

→ [18: What this does not cover](18-what-this-does-not-cover.md)
