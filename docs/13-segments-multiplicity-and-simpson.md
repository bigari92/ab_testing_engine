# 13 — Segments, multiplicity and Simpson

## The question

> "It didn't win overall, but it won for mobile users in Germany."

That sentence has ended more experiments than any other, and it is almost
always noise. This document is about how to slice data without fooling
yourself, and about the four specific guards `segments.py` applies in
order to do it.

## The arithmetic that makes correction compulsory

Ten independent tests at α = 0.05, **every null true**, 40,000
simulations:

```
no correction   0.4016      theory 1 - (1 - 0.05)^10 = 0.4013
```

**A readout with ten secondary metrics finds something in two runs out of
five when nothing is there.** Twelve segments gives 46%. Slice by device,
country, tenure and plan at once and you are not at risk of finding a
story — you are guaranteed one.

The theory and the measurement agree to four decimals, which is the point:
this is not a vague caution about "p-hacking". It is a computable number,
and you can compute it for your own readout right now.

## The three corrections, and what each one actually promises

This is the part that is usually taught as a list of recipes. The useful
version is that they control **different quantities**, and the difference
is measurable.

| | controls | 10 true nulls | 4 true / 6 real |
|---|---|---|---|
| none | nothing | 0.4016 | — |
| Bonferroni | FWER | 0.0495 | 0.0199 |
| Holm | FWER | 0.0495 | ≥ Bonferroni always |
| Benjamini–Hochberg | **FDR** | 0.0503 | FDR 0.0198, **FWER 0.1290** |

**FWER** — familywise error rate — is the probability of making *any*
false rejection. **FDR** — false discovery rate — is the expected
*proportion* of your rejections that are false. They are not the same
guarantee and BH only offers the second.

### Bonferroni and Holm are the same test under the global null

A detail worth knowing because it looks like a measurement error:
Bonferroni and Holm give **identical** rates (0.04945, to five digits)
when every null is true. The proof explains why rather than letting it
look like a coincidence:

> Holm's first step tests the smallest p-value against alpha/m, which is
> Bonferroni's condition exactly; when every null is true, making any
> rejection requires passing that step, so the two cannot differ here.
> Holm's advantage appears only once some nulls are false.

And when some are false, Holm wins **uniformly**:

> checked per-simulation across all 40,000 runs, not on average: Holm
> rejected strictly more in 21.2% of them and fewer in none.

Same guarantee, more power, never worse — per simulation, not on average,
which is the stronger claim. There is no reason to use Bonferroni over
Holm except to have something to compare against, and that is explicitly
why `bonferroni()` exists in this library.

### BH holds its promise exactly, and it is not the promise you wanted

With 4 true nulls among 10 hypotheses:

```
BH false discovery rate   0.0198   theory alpha*m0/m = 0.05 * 4/10 = 0.0200
BH FAMILYWISE rate        0.1290   against alpha = 0.05
```

Both facts are true simultaneously. BH controls FDR **exactly** —
`α·m₀/m` under independence, four decimal agreement — and its familywise
rate runs at **2.6× α**.

Two consequences nobody mentions:

**One.** `α·m₀/m` means **BH gets more permissive as more of your
hypotheses turn out to be real.** That is a feature if you are screening
genes. It is a surprise if you thought you had set a 5% error rate.

**Two.** A BH-corrected list is trustworthy *as a list* — about 2% of the
entries are junk — and **no single entry on it is trustworthy on its
own**. This is exactly why `report.py` labels a segment scan
**EXPLORATORY**. The label is not modesty; it is the precise statement of
what BH bought you.

## Guardrails are not corrected, and here is the price

From [document 02](02-metrics-and-the-unit-of-analysis.md), now with the
measurement. One real harm among five guardrails, sized for 80% power:

```
uncorrected   caught 0.7989   theory 0.8000
Bonferroni    caught 0.5816   theory 0.5893
```

Correcting costs **21.7 points of real-harm detection** — 80% down to 58%
— in exchange for holding the guardrail family's false alarm rate at 5%
instead of 23%.

> The two errors are not comparable: a false guardrail alarm costs an
> investigation that ends in "no, it was fine", and a missed one ships a
> regression to everybody. **Correction is for findings you want to
> announce, not for harms you are watching for.**

That sentence is the decision rule. It is not a statistical argument at
all — the statistics are symmetric — it is an argument about which of two
costs you would rather pay, made explicitly instead of by default.

## The four guards in `segments.py`

Segment analysis is done anyway, because heterogeneous effects are real
and worth finding. Four guards make it defensible.

### 1. A minimum size, with the MDE attached

```
A segment with 40 users per arm cannot detect anything an experiment
would act on. Segments below the floor are reported as unpowered rather
than tested, with the effect they COULD have detected, so "no effect in
enterprise accounts" is never confused with "we had 31 enterprise
accounts".
```

Note *reported as unpowered*, not skipped. `segment_scan` returns those
rows with `result=None, powered=False` — the segment stays visible, and so
does the reason it was not tested. Dropping it would hide the fact that
you looked, and a reader cannot audit a silence. (This is the design a
test in `tests/test_contracts.py` asserts across three size tiers.)

### 2. An interaction test, not two p-values

**The single most common error in segment readouts**, and the cheapest to
avoid:

> "Significant in mobile, not significant in desktop" is not evidence that
> the effect differs between them — both could be the same effect measured
> with different precision.

```
z = (effect_a - effect_b) / sqrt(se_a^2 + se_b^2)
```

The question "does the effect differ by segment" has its own test, and it
is a test on the **difference of the two effects**. Mobile might have a
third of the traffic and therefore a wider interval around the identical
effect; comparing verdicts compares sample sizes, not effects.

A useful rule of thumb that follows from the formula: because the two
standard errors add in quadrature, the interaction test needs roughly
**four times** the sample of either segment test to reach the same power.
Interactions are expensive. That is why they are hypotheses for the next
experiment rather than conclusions from this one.

### 3. BH across the segments, labelled exploratory

For the reasons measured above. A segment finding is a hypothesis for the
next experiment, not a decision from this one.

### 4. The mix-shift check — and this is the real Simpson's paradox

```
If the segments themselves are distributed differently across arms,
something is wrong before any effect is estimated: either randomisation
is broken, or — far more often — the segment is defined by something
that happened AFTER assignment.
```

This is the fourth appearance in this course of one single mistake:

- [Document 01](01-what-an-experiment-is.md): slicing on `engaged` produced
  a 33-point effect from zero-effect data.
- [Document 02](02-metrics-and-the-unit-of-analysis.md): a refund
  guardrail with `users` in the denominator blocked 77% of wins.
- [Document 10](10-variance-reduction-with-cuped.md): a post-treatment
  covariate erased 100% of a real effect.
- Here: a segment defined after assignment.

All four are **conditioning on something the treatment moved.** And the
note in the module is the claim worth carrying away:

> "Engaged users" is the classic: the treatment changes who becomes
> engaged, so comparing engaged users across arms compares two different
> populations. That is the mechanism behind **every real Simpson's paradox
> in experimentation, and it is a data-modelling error, not a statistical
> one.**

Simpson's paradox is usually taught as a curiosity about aggregation —
the kidney-stone tables, the Berkeley admissions. In a randomised
experiment it is not a curiosity and it is not about aggregation. If your
assignment was sound and your segments are pre-treatment, the paradox
cannot occur: the arms have the same segment mix by construction, so the
within-segment effects and the overall effect cannot point in opposite
directions by more than noise. If you see it, **you have a post-treatment
segment**, and no amount of statistical care will fix a definition.

Which is why the mix-shift check runs **before** any effect is estimated,
exactly as SRM does in [document 08](08-sample-ratio-mismatch.md) — and
for the same reason. It is the same test, applied within a slice.

## The rule, in full

```
Segments declared before the experiment, from pre-treatment attributes
only, analysed with an interaction test and an FDR correction, reported as
exploratory. Anything else is a story.
```

Five conditions. Each one removes a specific failure measured above, and
dropping any one of them gets you a readout that cannot be defended.

## Do it yourself

**1. Compute your own 40%.** For the readout you produce at work, count
the metrics and segments you look at. Then `1 - 0.95**k`. If you have
ever shipped a decision off the back of the most interesting cell in a
table, that number is your actual false positive rate.

**2. Reproduce the FWER/FDR gap**, since it is the thing that separates
understanding BH from reciting it:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab.multiplicity import benjamini_hochberg, bonferroni, holm

rng = np.random.default_rng(0)
fwer = fdr = 0
for _ in range(20_000):
    p = np.concatenate([rng.uniform(size=4),               # 4 true nulls
                        rng.uniform(size=6) * 0.001])      # 6 real effects
    rej = np.asarray(benjamini_hochberg(p, alpha=0.05).rejected)
    false_rej = int(rej[:4].sum())
    fwer += false_rej > 0
    fdr += false_rej / max(1, int(rej.sum()))
print("FWER %.4f   FDR %.4f" % (fwer / 20_000, fdr / 20_000))
```

```
FWER 0.1333   FDR 0.0200
```

FDR dead on `α·m₀/m = 0.02`, FWER at **2.7× α**. Both correct, from the
same 20,000 runs. If you can hold those two numbers in your head at once
you understand BH better than most people who use it.

**3. Build the interaction-test mistake and then fix it.** Take one
dataset, split it into a big segment and a small one with the **identical**
true effect, and run a separate test in each. One will be significant and
one will not. Then run the interaction test on the difference and watch it
return nothing — correctly, because there is nothing to find.

**4. Find a post-treatment segment in your own reporting.** Look at the
segment columns on any dashboard you own and ask of each: *was this value
determined before the user was assigned?* Anything about engagement,
activity level, feature usage, or lifecycle stage computed over the
experiment window is post-treatment. This is usually a short exercise with
an uncomfortable result.

→ [14: Unit economics](14-unit-economics.md)
