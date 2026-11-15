# 18 — What this does not cover

## Why this document exists

A portfolio project that lists only what it can do is a sales document.
This one is here because the boundary is part of the work: knowing which
problems a tool does **not** solve is what stops you applying it to them.

Each section below says what is missing, why it was left out, and — where
it matters — what to use instead.

## Interference between units

[Document 01](01-what-an-experiment-is.md) named the assumption:
randomisation gives an unbiased comparison **provided one user's treatment
does not change another user's outcome.** Nothing in this project does
anything about it.

The randomisation unit is the row and that is assumed to be the right
unit. It breaks whenever your users interact:

- a social feed change — treated users post more, their untreated friends
  see more posts, and the control group is contaminated
- a marketplace change — treated buyers bid more, raising prices for
  untreated buyers, so the control group moves in the *opposite*
  direction and the estimate is inflated
- a fixed-capacity business — a treated customer taking the last slot is a
  slot an untreated one cannot take
- **a staffing business with one client's vacancies** — two candidates
  competing for one role are not independent units, which makes
  [document 17](17-a-worked-example-hiring-funnel.md)'s design wrong at
  the vacancy level even though it is right at the candidate level

The honest options are **cluster randomisation** (randomise groups,
analyse at the group level) or **switchback designs** (alternate the whole
system over time). Both are real techniques and both are outside this
repository. The cluster design effect from
[document 11](11-ratio-metrics-and-the-delta-method.md) is the right
starting point if you go there — it is the same formula, applied to the
randomisation unit instead of the metric's denominator.

## Always-valid (anytime-valid) inference

[Document 09](09-peeking-and-sequential-tests.md) implements group
sequential boundaries, which require you to **declare the number of looks
in advance**. The more flexible alternative is a confidence sequence that
holds at every sample size simultaneously, so you may look continuously
and choose when to stop based on what you see.

> More flexible and more expensive in sample size. Deliberately out of
> scope.

It was left out because it would have been the one component I could not
check from first principles — and this project's claim is that every
number in it is derivable. An implementation I could not verify would
undermine the rest more than its absence does.

## Bayesian methods

No priors, no posteriors, no probability-to-be-best. Not because they are
worse — for a decision under a loss function they are often better — but
because they answer a different question, and mixing the two vocabularies
in one readout is how "95% confident" starts meaning two things in the
same meeting.

If you want the Bayesian version of
[document 15](15-the-decision-and-the-write-up.md)'s decision rule, that
is a different project, and a good one.

## Causal inference without randomisation

Difference-in-differences, synthetic control, instrumental variables,
regression discontinuity, propensity matching. **None of it is here.**

This matters because it is the realistic case: most questions arrive
without an experiment attached. The reason it is absent is that
[document 01](01-what-an-experiment-is.md)'s selection-bias term is
exactly what those methods try to argue away, and every one of them
depends on an assumption you cannot test with the data you have. That is
legitimate work and it is a different discipline from this one.

What this project gives you toward it is the thing most such analyses
lack: a precise statement of what the bias term *is*, so you can say which
assumption you are making instead of hoping nobody asks.

## Multi-armed tests and bandits

`analyze` takes two arms. `srm_check` handles any number of arms — the
proof covers a three-way 50/30/20 split — but nothing downstream does.

Three or more arms need a decision rule about which comparisons to make
and how to correct them, and a bandit needs an allocation policy and an
argument about why adaptive allocation has not destroyed the inference.
Both are real work. Running three pairwise tests at α = 0.05 and shipping
the winner is not, and
[document 13](13-segments-multiplicity-and-simpson.md) is why.

## Long-horizon and novelty effects

The generator *can* produce a novelty effect:

```python
novelty = 1.0 + novelty_strength * np.exp(-(assigned_day - 1) / novelty_tau)
```

so an effect can be inflated for users assigned early and decay with
tenure. But **nothing in `src/explab/` detects or corrects it.** There is
no holdback analysis, no cohort-over-time decomposition, no test for
whether an effect is still there in week six.

That is a real gap and the most commercially damaging one on this list. A
feature that wins in week one and is gone by week five is a very common
outcome, and this library will report the week-one number with a correct
interval around it. [Document 09](09-peeking-and-sequential-tests.md)'s
winner's-curse section is the closest thing here to a warning.

What to do: run long enough, and keep a holdback. Neither is a statistical
technique.

## Variance reduction beyond CUPED

[Document 10](10-variance-reduction-with-cuped.md) implements CUPED with a
single covariate. Stratified sampling, post-stratification, regression
adjustment with several covariates, and the machine-learning variants
(CUPAC and relatives) are all absent.

CUPED is the one with the best ratio of benefit to things that can go
wrong, and the document measures its failure mode — a post-treatment
covariate erasing 100% of a real effect. Multi-covariate adjustment
multiplies that risk surface without multiplying the benefit.

## Everything about the data pipeline

The library takes a DataFrame with a `variant` column. It does not know
where that came from, whether the assignment was deterministic, whether
the hash was salted per experiment, whether a user can appear twice, or
whether the exposure event fires before or after the feature loads.

**Most real experiment failures are here**, not in the statistics.
[Document 08](08-sample-ratio-mismatch.md) catches a subset of them after
the fact and the rest are invisible to anything downstream. `srm_check` is
a smoke alarm, not a fire-prevention system.

## Non-inferiority and equivalence testing

[Document 15](15-the-decision-and-the-write-up.md) distinguishes *flat*
from *inconclusive* using a practical threshold, which is a sensible
approximation to an equivalence test. The formal versions — TOST, proper
non-inferiority margins — are not implemented. If you need to demonstrate
that a cheaper system is **not worse** by more than a stated margin, the
threshold logic here is the right shape and not the right test.

## And the limits of the evidence itself

Repeating from [document 16](16-how-the-proofs-work.md), because it
belongs on this list too:

- **The generator is not reality.** It was built to contain the hazards
  the library claims to handle, which makes it a fair test of those claims
  and silent about hazards neither of us thought of. Two such hazards
  surfaced the moment I ran the machinery on a funnel instead
  ([document 17](17-a-worked-example-hiring-funnel.md)), which is direct
  evidence for this bullet rather than a hypothetical.
- **Agreement with SciPy is not correctness.** Very strong evidence,
  because the implementations are independent; blind to an error both
  make.
- **Monte Carlo claims carry Monte Carlo error.** At `k = 3`, roughly a
  27% chance that at least one of 104 claims passes for the wrong reason
  on a given run. Hence repeated runs, and hence an intermittent failure
  being treated as a failure.

## What is actually here

For balance, the positive list, since the rest of this document is
negative space:

- distribution functions derived from the incomplete beta and gamma, with
  the catastrophic-cancellation traps found and fixed (03)
- Welch, two-proportion, delta-method ratio, and bootstrap comparisons,
  each validated for false positive rate and interval coverage (04, 05,
  06, 11)
- power and MDE for means, proportions and now any computed standard
  error (07)
- an SRM monitor that reports its own sensitivity (08)
- Monte-Carlo-calibrated group sequential boundaries reproducing Pocock
  and O'Brien–Fleming to three decimals (09)
- CUPED, with its failure mode measured (10)
- multiplicity corrections with the FWER/FDR distinction measured rather
  than asserted (13)
- unit economics with intervals and a horizon you cannot omit (14)
- a decision rule with four outcomes and a readout ordered so the
  assignment check cannot be skipped (15)
- three independent verification mechanisms, and the eight defects they
  found in each other's blind spots (16)

## Do it yourself

**1. Pick the one that would bite you.** Read the list and decide which
absence matters most for the work you actually do. For a marketplace it is
interference; for a staffing business it is interference *and* the
pipeline; for a subscription product it is long-horizon effects.

**2. Then write down what you would do instead.** Not implement it — just
name the technique and the assumption it needs. That sentence is what
separates "we can't measure that" from "we can't measure that, and here is
what we would need".

**3. Add one.** The repository is laid out so that a new capability means a
module, a proof with derived tolerances, tests against an independent
implementation, and a document. If you work through this course and add
cluster randomisation to it, you will have learned more than this course
could teach you.

---

← [00: Start here](00-start-here.md)
