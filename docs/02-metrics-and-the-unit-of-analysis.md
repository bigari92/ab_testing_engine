# 02 — Metrics and the unit of analysis

This is the most useful document here, and it is second because everything
after it is wasted if you get this wrong. Every statistic in the story
below was correct. The readout was still wrong.

## The question

You have a metric. "Refund rate." Everyone in the room knows what it
means. Write it down as code and you will find out that they do not.

## What happened

The blind-scoring harness in `scoring/` runs the whole pipeline against
randomly generated truths and scores the decisions. It started returning
"do not ship" on experiments where a real win had been planted. Not
occasionally — at a rate that could not be chance.

The cause was the refund guardrail, declared the obvious way:

```python
MetricSpec("refund_rate", "binary", column="refunded",
           role="guardrail", direction="decrease")
```

`refunded` is a 0/1 column. It is `1` for a user who converted and then
refunded, and `0` for everyone else — **including everyone who never
converted at all.**

So the mean of that column is not the refund rate. It is:

```
refunds / users  =  P(convert) x P(refund | converted)
```

A treatment that raises conversion raises the first factor. Refunds per
user therefore go up **mechanically**, with nobody's propensity to refund
having moved at all.

## The measurement

`proofs/proof_guardrail_denominator.py`, with no refund harm planted
anywhere and refund propensity identical in both arms by construction:

| conversion lift | per-user guardrail alerts | per-converter guardrail alerts |
|---|---|---|
| +0.000 | 3.3% | 3.3% |
| +0.004 | 10.0% | 1.7% |
| +0.010 | 33.3% | 1.7% |
| +0.020 | **76.7%** | 0.0% |

Three winning experiments in four, blocked by a guardrail reporting a harm
that does not exist. And the shape of the error is the worst available:
**the bigger your win, the more certainly it is blocked.** The rate keeps
climbing with sample size too, because the composition shift is a fixed
quantity that more data resolves more sharply.

This is the symptom teams learn to live with — "our refund guardrail
always trips when we win, so we override it" — and overriding it is how
the one real regression ships.

## Why, exactly

Write `p` for P(convert) and `r` for refunds-per-converter. Then refunds
per user is `p × r`, and the difference between arms is an **identity**:

```
p_t r_t  −  p_c r_c   =   (p_t − p_c) r_c   +   p_t (r_t − r_c)
                          \__ composition __/   \__ propensity __/
```

Check it by expanding: the `p_t r_c` terms cancel. No approximation, no
assumption.

Only the second term is anything a refund guardrail is asking about. The
first is the treatment doing its job on conversion. Measured over 150
experiments with a +0.02 conversion lift and no refund harm:

```
per-user refund shift          +0.001580
composition  (p_t - p_c) r_c   +0.001739   110.0% of it
propensity   p_t (r_t - r_c)   -0.000159   -10.0% of it
```

The composition term accounts for **more than the whole** shift, and the
propensity term — the only real component — is **negative**. Refund
propensity went down. The guardrail alerted on a harm that pointed the
other way.

The proof asserts that decomposition to 1e-15, because it is algebra and
anything else appearing there would mean the two terms are not what they
are labelled. Measured largest residual across 150 experiments: 3.25e-18.

## The fix

Declare the metric at the denominator the question is about:

```python
MetricSpec("refund_rate", "ratio", numerator="refunded",
           denominator="converted", role="guardrail",
           direction="decrease")
```

The ratio of sums is total refunds over total conversions — refunds per
converter — and `ratio_delta` keeps the standard error at the randomised
unit, which matters for reasons that are the whole of
[document 11](11-ratio-metrics-and-the-delta-method.md).

## And the fix costs something, which the proof also measures

Here is the part a tidier write-up would leave out. The correct guardrail
is **less powerful**:

| planted regression | per-converter catches | per-user catches | per-converter MDE |
|---|---|---|---|
| +0.01 | 37.5% | 100% | 0.0141 |
| +0.03 | 100% | 100% | 0.0147 |

Only about a tenth of users convert, so the per-converter metric has about
a tenth of the units, and roughly three times the standard error. It
catches a large regression every time and a small one about a third of the
time.

The per-user version catches both every time — and also catches a harm
that is not there 77% of the time, which is why its sensitivity is worth
nothing. A test that always fires is not sensitive; it is constant.

The answer is not to go back to the wrong denominator. It is to **report
the guardrail's MDE next to its verdict**, which `analyze.py` does, so
that "no harm detected" is read as "no harm above 0.015 detected". That is
why every metric line in a readout from this project carries the effect it
could have detected — see
[document 15](15-the-decision-and-the-write-up.md).

Notice also what makes that table a real check rather than a boast: a
regression **larger** than the reported MDE is caught almost always, and
one **smaller** than it is not. If a metric claimed an MDE of 0.014 and
then reliably caught a 0.01 regression, the MDE would be lying in the
flattering direction. The proof asserts both halves.

## The general rule

> **Define a metric by its denominator, and make the denominator the thing
> you randomised — or a quantity the treatment cannot move.**

Three ways to satisfy it:

1. **Per randomised unit.** Revenue per user, minutes per user. Always
   safe, sometimes not the question.
2. **A ratio with the denominator in the metric**, analysed with the delta
   method at the randomised unit. Refunds per converter, clicks per
   impression. This is what `kind="ratio"` is for.
3. **A pre-treatment denominator.** Per user who was already a subscriber
   before the experiment started. Safe because the treatment cannot change
   who that was.

What is never safe: a denominator the treatment moves, which includes
every "among users who …" where the "…" happens after assignment.
Conditioning on a post-treatment quantity is the same mistake as slicing
on `engaged` in [document 01](01-what-an-experiment-is.md), and it is the
real mechanism behind most readouts labelled "Simpson's paradox" — see
[document 13](13-segments-multiplicity-and-simpson.md).

## The three roles

`MetricSpec` requires you to say what a metric is **for**, and the three
roles get deliberately different statistical treatment.

```python
role="primary"      exactly one. It decides. Not corrected.
role="guardrail"    things that must not get worse. Not corrected.
role="secondary"    everything else. BH-corrected, labelled exploratory.
```

**Exactly one primary.** `analyze` raises if you declare two. The point of
a primary metric is that it decides; two of them means the rule is
"whichever wins", and that is a false positive rate nobody computed. This
is a structural guard, not advice:

```python
if sum(m.role == "primary" for m in metrics) > 1:
    raise ValueError(...)
```

**Guardrails are never corrected for multiplicity.** This looks like
sloppiness and is the opposite. `proofs/proof_multiplicity.py` measures
what correcting them costs: with one real harm among five guardrails,
sized for 80% power,

```
uncorrected   caught 79.9%   (theory 80.0%)
Bonferroni    caught 58.2%   (theory 58.9%)
```

Correcting costs 21 points of real-harm detection, in exchange for holding
the guardrail family's false alarm rate at 5% instead of 23%. Those two
errors are not comparable. A false guardrail alarm costs an investigation
that ends in "no, it was fine". A missed one ships a regression to
everybody. Correction is for findings you want to **announce**, not for
harms you are **watching for**.

**Secondary metrics are corrected and labelled.** Ten metrics at α = 0.05
find something 40% of the time when nothing is there — measured at 0.4016
against the theoretical 0.4013. So they get Benjamini-Hochberg and the
word EXPLORATORY in the output, for a reason spelled out in
[document 13](13-segments-multiplicity-and-simpson.md): BH controls the
false discovery *rate*, which says nothing about the probability of any
single finding being real. Its familywise error rate, measured, runs at
12.9% — 2.6× α — while holding its own guarantee exactly.

## Declaring it before you look

`MetricSpec` is validated at construction, not at analysis time, and the
`practical_threshold` is part of the declaration:

```python
MetricSpec("conversion", "binary", column="converted", role="primary",
           direction="increase", practical_threshold=0.004)
```

`practical_threshold` is the smallest effect that would change your
decision. You have to supply it **before** seeing the result, because its
only job is to separate two outcomes that have the same p-value and
opposite meanings:

- the interval sits entirely inside ±threshold → **flat**. There is no
  effect big enough to matter. That is an answer.
- the interval still contains effects you would ship for →
  **inconclusive**. Not an answer.

Leave it out and `analyze` will not guess. It reports that the distinction
could not be made, and says so in the reasons. [Document
15](15-the-decision-and-the-write-up.md) has the measurement: across 400
blind trials, true nulls were called *flat* 32.9% of the time and
*inconclusive* the rest — both correct, and the difference between
"stop asking" and "keep asking".

## Do it yourself

**1. Reproduce the trap.** Run it both ways on the same data:

```python
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "data")
from generate import generate_experiment
from explab.analyze import MetricSpec, analyze

frame, _ = generate_experiment(n_users=200_000, seed=901,
                               effect_conversion=0.02, effect_refund=0.0)

primary = MetricSpec("conversion", "binary", column="converted",
                     role="primary", practical_threshold=0.004)
per_user = MetricSpec("refund_per_user", "binary", column="refunded",
                      role="guardrail", direction="decrease")
per_conv = MetricSpec("refund_per_converter", "ratio", numerator="refunded",
                      denominator="converted", role="guardrail",
                      direction="decrease")

for g in analyze(frame, [primary, per_user, per_conv]).guardrails:
    print(g.spec.name, "ALERT" if g.result.significant and
          not g.moved_in_good_direction else "ok")
```

**2. Then verify the identity by hand** on that frame. Compute `p_c`,
`p_t`, `r_c`, `r_t` yourself and check that
`(p_t − p_c) r_c + p_t (r_t − r_c)` equals the per-user difference to the
last bit. Once you have done it once you will see this metric everywhere:
"revenue per user" in a business where the treatment changes who buys,
"tickets per customer" where it changes who contacts support, "offers per
candidate" where it changes who reaches interview.

**3. Find one in your own work.** Take any metric you report that has the
word "per" in it, or the word "rate", and ask: can the treatment change
the denominator? If yes, you have been reporting a composition shift as an
effect.

→ [03: Distributions from scratch](03-distributions-from-scratch.md)
