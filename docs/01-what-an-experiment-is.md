# 01 — What an experiment is

## The question

You changed something. The number moved. Did your change move it?

That question is unanswerable from the number alone, and almost all of
analytics is various ways of pretending otherwise.

## The problem, stated properly

For each user there are two numbers you care about:

- what they would do **with** your change — call it `Y(1)`
- what they would do **without** it — `Y(0)`

The effect on that user is `Y(1) − Y(0)`. You want the average of that
across users:

```
ATE = mean( Y(1) − Y(0) )          the average treatment effect
```

And here is the whole difficulty: **you can only ever see one of the two
numbers for any given user.** The other one does not exist anywhere. It
is not missing data you could go and collect; it is a fact about a world
that did not happen. This is called the fundamental problem of causal
inference, and it is not a technicality — it is the reason the job is
hard.

## What a comparison actually gives you

Suppose you compare users who got the change against users who did not:

```
observed difference = mean( Y(1) | treated ) − mean( Y(0) | not treated )
```

Add and subtract the same term, `mean( Y(0) | treated )`:

```
observed = [ mean(Y(1)|treated) − mean(Y(0)|treated) ]   <- the real effect
                                                             on the treated
         + [ mean(Y(0)|treated) − mean(Y(0)|untreated) ] <- selection bias
```

The second bracket is the difference the two groups would have shown **if
you had changed nothing at all.** It is not an error term, not noise, and
not something a bigger sample reduces. Collect ten million users and it
sits there unchanged.

That bracket is what every before/after comparison, every "users who
engaged with the feature retained better", and every geographic rollout
comparison is made of.

## Randomisation is the one thing that kills it

Assign each user to a group by a coin flip. The coin knows nothing about
the user. So the two groups' `Y(0)` distributions differ only by sampling
noise:

```
mean( Y(0) | treated ) − mean( Y(0) | untreated )  →  0
```

and it goes to zero at a rate you can compute, which is what the rest of
this course is about. That is the entire trick. Not statistics —
**assignment**. The statistics only tell you how much of the remaining
difference is noise.

Two things follow that are worth more than they look:

**Randomisation balances things you did not think of.** It balances
tenure, device, country, timezone, spending habits, and the thing you will
only realise mattered in six months. No regression adjustment can balance
a variable you did not record.

**Randomisation is a property of the assignment, not of the data.** You
cannot check it by looking at the outcome. You can only check that the
assignment *mechanism* did what you told it to — which is exactly what
[document 08](08-sample-ratio-mismatch.md) is about, and why that check
runs before any effect is estimated in this project.

## What this buys you, and what it does not

Buys you:
- An unbiased estimate of the average effect, with a computable uncertainty.
- Protection against confounders you never thought of.

Does **not** buy you:
- The effect on any individual. `Y(1) − Y(0)` for one person remains
  unknowable. An average effect of +2% does not mean everybody went up 2%;
  it is consistent with 10% of users going up 20% and nobody else moving.
  [Document 13](13-segments-multiplicity-and-simpson.md) is about how far
  you can responsibly go toward finding out which users.
- Anything about a different population, a different season, or a longer
  horizon than you ran.
- Protection against measuring the wrong thing. Randomisation makes your
  comparison valid. It does not make your metric relevant. See
  [document 02](02-metrics-and-the-unit-of-analysis.md), where the most
  expensive mistake in this repository lives.

## The one assumption people forget

Randomisation gives you an unbiased comparison **provided one user's
treatment does not change another user's outcome.** Formally: no
interference. Informally: it breaks whenever your users interact.

- A social feed change: treated users post more, and their untreated
  friends see more posts. The control group is contaminated.
- A marketplace change: treated buyers bid more, which raises prices for
  untreated buyers. The control group is affected in the *opposite*
  direction, which inflates your estimate.
- A pricing change in a fixed-capacity business: a treated customer taking
  the last slot is a slot an untreated one cannot take.

This project does nothing about interference. The randomisation unit is
the row and that is assumed to be the right unit. If your users interact,
the honest options are cluster randomisation (randomise groups, analyse at
the group level) or switchback designs, and both are outside this
repository — stated plainly in
[document 18](18-what-this-does-not-cover.md) rather than left for you to
discover.

## The code

There is no module for this document, which is the point: the most
important part of an experiment happens before any code in `src/explab/`
runs. What there is, is a generator that makes the counterfactual visible —
because in synthetic data you can know `Y(0)` and `Y(1)`:

```python
# data/generate.py
q = rng.standard_normal(n_users)          # each user's latent quality

base_logit = np.log(base_conversion / (1 - base_conversion)) + 0.8 * q
p_control = 1.0 / (1.0 + np.exp(-base_logit))          # this user's Y(0)
p_user = np.clip(p_control + np.where(treated, effect_per_user, 0.0),
                 1e-6, 1 - 1e-6)                       # their Y(1) if treated
converted = (rng.random(n_users) < p_user).astype(int)
```

`p_control` is every user's `Y(0)` — their conversion probability without
the change — and it varies across users through `q`. The treatment adds
`effect_per_user` to that **probability**.

Adding to the probability rather than to the log-odds is a deliberate
choice, and the comment in the file says why: it keeps the planted truth
stated in the same units the analysis reports. A log-odds shift would be
a perfectly respectable way to generate data and a useless one to score
against, because the "true effect" in percentage points would then depend
on each user's baseline and there would be no single number to compare the
estimate to. See [document 16](16-how-the-proofs-work.md).

The generator knows both `Y(0)` and `Y(1)`. The analysis is handed only
the `converted` column and the `variant` label.

## The measurement

`proofs/proof_false_positive.py` establishes the claim that makes
randomisation worth anything: with **no** effect planted, the test says
"significant" 5% of the time and no more. Measured on the generator
itself, with every effect set to zero:

```
false positive rate 0.0567 against a nominal 0.05, 3000 simulations
```

If that number came out at 0.15, every readout built on this library
would be finding things that are not there, and no amount of careful
wording further down would help.

## Do it yourself

**1. Watch selection bias not go away.** Compare users who engaged with a
feature against users who did not, in data where the feature does
nothing:

```python
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "data")
from generate import generate_experiment
from explab.inference import two_proportion_from_arrays

frame, truth = generate_experiment(n_users=200_000, seed=7,
                                   effect_conversion=0.0)
assert truth.effect_conversion == 0.0            # nothing was planted

engaged = frame[frame.engaged == 1]["converted"].to_numpy(float)
not_engaged = frame[frame.engaged == 0]["converted"].to_numpy(float)
print(two_proportion_from_arrays(not_engaged, engaged))
```

What comes back:

```
rate: control 0 -> treatment 0.334167
  absolute +0.334167  [+0.330392, +0.337941] at 95%
  p = 1e-300  (significant, two_proportion, n = 140000 vs 60000)
```

A 33-point effect, with a p-value at the floor this library will print,
from data where the planted effect is exactly zero. It is that extreme
because `engaged` is built from behaviour that includes converting
(`engaged_score = clicks + 3 * converted + noise`), so nobody outside the
engaged group converted at all — the comparison is between converters and
non-converters wearing a different name.

Now run the same data the right way:

```python
c = frame[frame.variant == "control"]["converted"].to_numpy(float)
t = frame[frame.variant == "treatment"]["converted"].to_numpy(float)
print(two_proportion_from_arrays(c, t))
```

```
rate: control 0.100284 -> treatment 0.100216
  absolute -6.74155e-05  [-0.00269992, +0.00256508] at 95%
  p = 0.96  (not significant, two_proportion, n = 99717 vs 100283)
```

Then raise `n_users` tenfold and confirm the fake effect gets **more**
significant, not less. That is the signature of bias: more data makes you
more certain of the wrong number. Noise shrinks with n; bias does not.

(The `relative undefined` line in the first output, where a percentage
change from zero has no meaning, is itself a small piece of this project —
see the note on NaN in [document 15](15-the-decision-and-the-write-up.md).)

**2. Derive the decomposition above on paper** without looking. If you
can reproduce the add-and-subtract step, you will never again be talked
into a before/after comparison.

→ [02: Metrics and the unit of analysis](02-metrics-and-the-unit-of-analysis.md)
