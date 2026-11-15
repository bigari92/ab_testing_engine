# 14 — Unit economics

## The question

"LTV is $340."

Over what horizon? Computed how? With what range? In almost every
dashboard that reports unit economics, LTV appears as a single number with
no interval, from a formula nobody can reproduce — and it is the number
the business plans against.

Unit economics appear by name in **27 of the 128 job postings** this
project was built against, which is why they are in an experimentation
repository. The deeper reason is that the same discipline applies to both:

> **An estimate without an interval is a guess with a decimal point.**

## The definitions, which are the actual deliverable

More damage is done here by ambiguous definitions than by wrong
arithmetic. `METRIC_DICTIONARY` exists so that a readout can carry them:

| | definition |
|---|---|
| **ARPU** | total revenue / **all** users in the cohort, payers and non-payers |
| **ARPPU** | total revenue / **payers only** |
| **CAC** | acquisition spend / customers acquired **by that spend**, same window |
| **LTV** | expected revenue from one acquired user **over a stated horizon** |
| **payback** | periods of ARPU needed to recover CAC, on the retention curve |

```
ARPU = ARPPU * conversion_rate
```

Measured on this project's data, where 10.32% of users pay:

```
ARPU  2.5656  [2.4387, 2.6978]
ARPPU 24.8485 [23.8465, 25.9185]
ARPPU * conversion = 24.8485 * 0.1032 = 2.5656      exactly ARPU
```

**Confusing those two inflates LTV by `1/conversion` — a factor of 9.7
here, and 33 on a 3% conversion rate.** It is the most common error in
this area, it is a single word in a SQL query, and nothing about the
result looks wrong: $340 instead of $35 is a plausible number in a
plausible range, and the whole business case rests on it.

And note "LTV": there is no such thing as LTV without a horizon.
**"Lifetime" is not a number.** This module requires a horizon and prints
it.

## The worst error: one churn rate for a curve that is not geometric

`LTV = ARPU / churn` assumes a **constant hazard** — the same share leaves
every period. Real curves decay fast and then flatten, because the people
most likely to leave leave first, and what remains is a hardier
population.

So `ltv_geometric` is kept in the library **specifically so the gap can be
shown rather than argued about**. Same ARPU, same 12-period horizon, same
retention curve — only the period you fit churn on changes:

| churn fitted at | churn rate | LTV | vs the curve |
|---|---|---|---|
| period 1 | 0.3800 | **6.75** | 0.52× |
| period 6 | 0.0588 | **43.62** | 3.38× |
| period 11 | 0.0069 | **372.01** | **28.87×** |
| — (the actual curve) | — | **12.89** | 1.00× |

**$6.75 to $372 from the same data.** A 55-fold range, every value
produced by the textbook formula, correctly applied, by someone who could
defend each step.

Fit on period 1 and you understate the tail badly — the early hazard is
the churn-prone leaving, and it does not continue. Use a late-period rate
and you extrapolate the survivors' behaviour onto everybody, which is how
you get $372.

The honest calculation sums the curve you actually have:

```
LTV = ARPU_per_period * sum over k of retention_k / (1 + d)^k
```

No single churn rate appears in it. `ltv_from_retention` gives **12.89**,
and the reason to trust it is not that it is in the middle — it is that it
makes no claim about periods you did not observe.

## The two limits that cannot be fixed, only labelled

### Survivorship in the cohort selection

Measuring 12-month LTV requires 12 months of history. So the estimate can
only use cohorts acquired at least a year ago — a different acquisition
mix, different pricing, different product.

> The number is real and it is about the past. Nothing here can fix that;
> it can only be labelled, and `warnings` labels it.

This is worth being blunt about with stakeholders. A 12-month LTV is not a
forecast. It is a measurement of a cohort that no longer exists, and the
faster the business is changing the less it transfers.

### The interval is embarrassingly wide, and that is information

```
ARPU 2.5656  [2.4387, 2.6978]      the interval spans 10.1% of the estimate
```

Ten percent, on 40,000 users, for the simplest quantity here. And
[document 12](12-outliers-and-heavy-tails.md) explains why — one user in
40,000 carries 34% of the variance. Every figure in this module carries a
bootstrap interval, and the module docstring says what to do about the
result:

> on real data those intervals are embarrassingly wide. **That is
> information, not a defect.**

A $340 LTV with a plausible range of $180–$620 is a different business
decision from $340 ± $20. Reporting the first as the second is the failure
this module exists to prevent, and it is the same failure as reporting
*inconclusive* as *flat* in [document 06](06-what-an-interval-means.md).

## Payback, and an answer that is `None`

```python
def payback_periods(cac, arpu_per_period, retention, *, discount_rate=0.0):
    """First period where cumulative discounted revenue per user covers CAC.

    Returns None when it does not happen inside the observed curve, which
    is an answer and should be reported as one rather than extrapolated
    into a number that looks better.
    """
```

On the curve above:

```
CAC  5.0  ->  payback in period 2
CAC 12.0  ->  payback in period 10
CAC 40.0  ->  None
```

**`None` is the most valuable of those three.** It means: on the retention
you have actually observed, this acquisition cost does not pay back. The
alternative — fitting a geometric tail and extrapolating — yields "payback
in month 19" from a curve that ends at month 12, and the difference
between those two answers is the entire difference between a measurement
and a hope.

Compare the same restraint in `mde_proportions` from
[document 07](07-power-and-sample-size.md): *"the honest answer is that
this sample cannot detect anything, not a number."* Both functions refuse
to manufacture a figure where the data has run out, and in both cases that
refusal was the fix for a real defect.

## The ARPPU warning this course earned

[Document 02](02-metrics-and-the-unit-of-analysis.md) ends with a
measurement that lands directly on ARPPU. Under a +2pp conversion lift
with **nobody spending differently**:

```
revenue per converter falls 4.8%, and 88.7% of runs show a decrease
```

ARPPU is revenue per *payer*, and `payer` is a post-treatment
denominator — the treatment changes who pays. A conversion lift brings in
marginal, lower-value customers, so **ARPPU falls mechanically on a
successful experiment.**

So: ARPPU is a useful descriptive figure and a **terrible** success or
guardrail metric in an experiment that moves conversion. If you report
ARPU and ARPPU side by side during a conversion test, expect them to point
in opposite directions, and expect to have to explain it. The explanation
is `ARPU = ARPPU × conversion`: the product rose because the second factor
rose more than the first fell.

## Do it yourself

**1. Reproduce the 55-fold range**, because it is the most persuasive
thing in this document when someone defends `ARPU/churn`:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from explab import unit_economics as ue

arpu_period = 2.5656
retention = np.array([1.0, 0.62, 0.48, 0.41, 0.37, 0.34,
                      0.32, 0.31, 0.30, 0.295, 0.29, 0.288])
print("curve:", ue.ltv_from_retention(arpu_period, retention))
for k in (1, 6, 11):
    churn = 1.0 - retention[k] / retention[k - 1]
    print(f"  churn at period {k}: {churn:.4f} -> "
          f"{ue.ltv_geometric(arpu_period, churn):.2f}")
```

**2. Find out whether your own curve is geometric.** Compute the
period-over-period churn at every period and print the list. If it is
declining — it will be — then `ARPU/churn` has no single correct input,
and whoever chose the one in your dashboard made a decision worth
auditing.

**3. Check ARPU against ARPPU × conversion** in your own reporting. They
must multiply out exactly. If they do not, one of the three is computed on
a different cohort, and you have found a real defect.

**4. Ask for the horizon.** Next time you see an LTV, ask what window it
covers and which cohorts it used. If nobody knows, the number cannot be
checked — which is a finding about the number, not about the person you
asked.

→ [15: The decision and the write-up](15-the-decision-and-the-write-up.md)
