# 07 — Power and sample size

## The question

Someone asks you to run a test. The real question is not "will it win" but
**"is this experiment capable of answering anything at all?"** — and that
is answerable before a single user is enrolled.

This is the document whose contents replace the most common conversation in
analytics: *"the test ran for two weeks and came back inconclusive."* That
was knowable on day zero.

## The formula, derived

Power is the probability of detecting an effect **given that it is real**.
You reject when your statistic clears the critical value:

```
|estimate| > z_alpha * se
```

Under the alternative, `estimate` is centred on the true effect `d`, with
standard deviation `se`. So the quantity `estimate/se` is centred on
`d/se` — call that `lambda`, the **noncentrality**: the true effect
measured in standard errors.

You detect it when `estimate/se` clears `z_alpha`, which happens with
probability

```
power = P( Z + lambda > z_alpha )  =  1 - Phi(z_alpha - lambda)
```

(plus a negligible term for rejecting in the wrong direction). Now demand
a given power, so `lambda` must be at least `z_alpha + z_power`:

```
d / se  >=  z_alpha + z_power
```

and substitute the standard error of a difference of means,
`se = sd * sqrt(2/n)`:

```
         2 * sd^2 * (z_alpha + z_power)^2
n   =    --------------------------------
                    mde^2
```

Which is the code, exactly:

```python
z_alpha = norm_ppf(1.0 - alpha / 2.0)
z_power = norm_ppf(power)
n = 2.0 * (sd ** 2) * ((z_alpha + z_power) ** 2) / (mde ** 2)
```

At the usual settings `z_alpha = 1.96`, `z_power = 0.84`, so
`(z_alpha + z_power)² = 7.85`, and the whole thing collapses to the rule
worth memorising:

```
n_per_arm  ≈  15.7 * (sd / mde)^2
```

## The three facts that follow immediately

**One: `n` scales with `1/mde²`.** Halving the effect you want to catch
**quadruples** the sample. Measured in the proof: `4.00` against a
tolerance of ±0.02. This is the single most important number in
experimentation economics, and it runs both ways — it is also why
variance reduction pays. Cut the variance by 36% and you have bought
yourself the same thing as 36% more traffic, for free. That is
[document 10](10-variance-reduction-with-cuped.md).

**Two: `sd` enters squared.** It is the input most often wrong and the one
that punishes you hardest. A standard deviation guessed 40% too low gives
a sample half the size it needs to be. The docstring is blunt about it:

> Take it from historical data, not from intuition; it is the single input
> most often wrong, and it enters squared.

**Three: power is not a property of the experiment.** It is a property of
the experiment *and an effect size*. "Is this test powered?" has no answer.
"Can this test detect a 1-point lift at 80%?" has one.

## MDE: the same formula, asked usefully

In practice nobody gets to choose `n` — traffic is what it is. So invert:

```python
mde = (z_alpha + z_power) * sd * math.sqrt(2.0 / n_per_group)
```

**Minimum detectable effect**: given the traffic you actually have, the
smallest effect you could reliably find. This is the number to bring to
the planning meeting, because it converts "we don't have enough traffic"
into a statement someone can act on:

> "At 6,000 users per arm we can detect a 1-point lift on an 8% base at
> 50% power. If you want 80%, we need 12,200 per arm, which is eleven
> days. If you only care about a 2-point lift, we have enough for four
> days."

And it is the honest reply to "the test was inconclusive": the test was
inconclusive *by design*, and the design was visible in advance.

## Rates are not quite the same

For a 0/1 metric the variance is not free to choose — it is `p(1−p)`, so
the baseline rate decides it:

```python
var_sum = p_c * (1 - p_c) + p_t * (1 - p_t)
n = var_sum * ((z_alpha + z_power) ** 2) / (mde_absolute ** 2)
```

Note `p_c(1−p_c) + p_t(1−p_t)`, **each arm with its own variance**, rather
than twice one of them. The docstring gives the reason, and it is a
consistency argument rather than a statistical one:

> a plan sized with one variance assumption and analysed with another is
> how experiments end up underpowered on paper and overpowered in
> practice, or the reverse.

The plan's variance assumption matches the unpooled interval that
`two_proportion` reports ([document 05](05-comparing-two-rates.md)). It
makes the plan slightly more conservative than the pooled textbook
formula, and it makes the plan and the analysis talk about the same
quantity.

**The consequence that catches people**: for proportions, `n ∝ 1/mde²` is
**not exact.** Halving the MDE does not quite quadruple the sample,
because `p_t = p_c + mde` moves when the MDE moves, and so does the
variance. A test in `tests/test_power.py` pins this down rather than
letting it look like sloppiness:

```python
def test_for_proportions_the_scaling_is_not_exactly_four_and_that_is_correct():
    ...  # the measured ratio is 3.8425, and the exact
         # variance-weighted prediction is 3.8425
```

The test asserts the exact prediction, not "approximately 4". A test that
accepted 4.0 within 5% would pass while hiding the mechanism.

And one more asymmetry worth knowing: **detecting a decrease costs less
than detecting an increase** on a low base rate. From the same test file:

```
detect -1pp from 8%:  3,211 per arm
detect +1pp from 8%:  3,839 per arm
```

The ratio is exactly the ratio of the variance sums, because 7% has less
variance than 9%. A guardrail watching for a drop is therefore cheaper
than a primary metric watching for a rise — which is a useful thing to
know when someone asks why the guardrail has a tighter MDE.

## What the plan prints, and why

```python
return PowerPlan(
    n_per_group=n_per, total_n=2 * n_per, mde=float(mde_absolute),
    mde_relative=mde_absolute / p_c, power=power, alpha=alpha,
    assumptions=(
        f"baseline rate = {p_c:.4%}",
        f"treatment rate under the alternative = {p_t:.4%}",
        "unpooled variance, matching two_proportion's interval",
    ),
)
```

`assumptions` is not decoration. A sample size is a **conditional**
statement — 12,206 per arm *if* the base rate is 8% and *if* you care
about a 1-point move. Attaching the conditions to the answer means the
reader can challenge the inputs instead of the conclusion, and
"12,206 users" stops being a fact handed down from an analyst.

This is the difference between a number and a defensible number, and it is
the same principle as the proofs printing their tolerances
([document 16](16-how-the-proofs-work.md)).

## Two guards, one of which was a bug

```python
if mde == 0:
    raise ValueError("mde of zero needs infinite sample; pick an effect size")
```

A zero MDE divides by zero, and the limit really is infinite — there is no
sample size that detects an arbitrarily small effect. The error says what
to do.

And `mde_proportions`, which asks "what can I detect at this n", can be
asked a question with no answer. At 50 users per arm on an 8% base, there
is no effect it can detect at 80% power — not a small one, not a large
one, because even a move to 100% would not clear the bar. It used to
return `NaN`:

```python
raise ValueError(
    f"no detectable effect exists at {n_per_group} units per group: "
    f"even an absolute change of {hi:.4g} from a baseline of "
    f"{baseline_rate:.4g} would need {needed(hi):,.0f} per group at "
    f"{power:.0%} power. The honest answer is that this sample "
    f"cannot detect anything, not a number."
)
```

That `NaN` was the fourth defect in document 00's list, and the
instructive part is where it went: into a caller **already catching
`ValueError` for exactly this case**. Someone had anticipated the
situation and handled it correctly; the function signalled it the one way
the handler could not see. A NaN is a value that passes every type check
and every exception handler, and then prints itself into a report as
`nan`. The lesson is in the message: *the honest answer is that this
sample cannot detect anything, not a number.*

`tests/test_contracts.py` sweeps seven separate renderings for a leaked
NaN, because the same mistake was available in `PowerPlan.__str__` and
`TestResult.__str__` too, and both were making it.

## The measurement

The claim that makes any of this worth printing: **an experiment run at
the recommended size detects the stated effect at the stated rate.**
`proofs/proof_power.py`, planting the effect the plan was sized for:

| plan | asked | measured |
|---|---|---|
| `sample_size_means`, n = 342/arm | 50% | 0.4963 |
| `sample_size_means`, n = 698/arm | 80% | 0.7987 |
| `sample_size_means`, n = 1,156/arm | 95% | 0.9520 |
| `sample_size_proportions`, 8% base +1pp, n = 12,206/arm | 80% | 0.7980 |

And the power *curve*, predicted from the formula and then measured at
each point — which is a stronger claim than hitting one target:

| true effect | formula | measured |
|---|---|---|
| 0.00 | 0.0500 | 0.0577 |
| 0.15 | 0.1855 | 0.1783 |
| 0.30 | 0.5641 | 0.5617 |
| 0.45 | 0.8891 | 0.8867 |
| 0.60 | 0.9888 | 0.9880 |

The first row is the check that ties this document to
[document 01](01-what-an-experiment-is.md): **at a true effect of zero,
power is just the false positive rate.** They are the same integral. If
that row had come back at 0.15, the power formula and the test would be
disagreeing about what "reject" means.

Both inversions are also verified to round-trip — `mde_means` inverts
`sample_size_means` to 0.4499 against 0.45, the gap being the ceiling on
the sample size, and `mde_proportions` inverts to 0.0099996 against 0.01.

## Do it yourself

**1. Price your own next experiment** before reading further:

```python
import sys; sys.path.insert(0, "src")
from explab.power import sample_size_proportions, mde_proportions

print(sample_size_proportions(0.08, mde_absolute=0.01))
print(sample_size_proportions(0.08, mde_relative=0.05))
print(mde_proportions(0.08, n_per_group=6000))
```

Then divide by your daily traffic. That number — days, not users — is the
one that ends arguments.

**2. Watch the quadratic bite, and watch it not be exactly quadratic:**

```python
for m in (0.02, 0.01, 0.005, 0.0025):
    print(m, sample_size_proportions(0.08, mde_absolute=m).n_per_group)
```

```
0.02      3,211
0.01     12,206      x3.80
0.005    47,525      x3.89
0.0025  187,487      x3.95
```

Each halving costs **not quite** four times as much — 58× across the whole
range rather than 64× — and the ratio creeps *toward* 4 as the MDE
shrinks, because `p_t = 0.08 + mde` gets closer to `p_c` and the two
variance terms converge. Predict that direction before you run it; if you
predicted 64× flat, you have just learned the difference between a
proportion and a mean. Either way, nothing else in analytics has this
shape, which is why "let's make it a bit more sensitive" is never a small
request.

**3. Get the sd wrong on purpose.**

```python
sample_size_means(sd=2.0, mde=0.3).n_per_group    # 698
sample_size_means(sd=2.8, mde=0.3).n_per_group    # 1368
```

A 40% under-estimate of the standard deviation buys you **51%** of the
sample you needed — and the ratio is exactly `(2.8/2.0)² = 1.96`, because
`sd` enters squared. Then ask the library what you actually bought:

```python
power_at(sd=2.8, n_per_group=698, effect=0.3)     # 0.5166
```

**52% power while the plan says 80%** — a coin flip, and nothing in the
output will ever mention it. An experiment planned that way is not
slightly underpowered; it is roughly twice as likely to miss a real effect
as its own plan claims. Now ask where your project's `sd` would come from,
and whether anyone would check it.

**4. Ask an impossible question** and read the error:

```python
mde_proportions(0.08, n_per_group=50)
```

The text of that exception is the most useful thing in the module.

→ [08: Sample ratio mismatch](08-sample-ratio-mismatch.md)
