#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ratio-metric bug, measured - and predicted before it is measured.

THE SETUP
Users are randomised. The metric is clicks per impression. Impressions are
finer than users: one user generates many. Three ways to compute the
interval, only one of them right:

  delta        ratio_delta: the ratio of sums, with the variance taken at
               the randomised unit, including the numerator/denominator
               covariance
  impression   pool all impressions, treat the metric as a proportion,
               sqrt(p(1-p)/n_impressions). This is the common bug.
  per-user     average each user's own ratio. Not the same estimand, and
               included so the difference is visible rather than argued

HOW BAD IS THE BUG? THE THEORY SAYS, BEFORE THE SIMULATION RUNS
Pooling impressions pretends every impression is an independent draw. They
are not: impressions from one user share that user's click rate. This is
the classic clustered-sampling problem, and the factor by which the pooled
variance is understated is the DESIGN EFFECT:

    DE = 1 + (m_eff - 1) * ICC

  ICC    the intra-user correlation of clicks. For a Beta-Binomial user
         with concentration c = a + b, Var(p_i) = p(1-p)/(c+1), so
         ICC = Var(p_i) / (p(1-p)) = 1 / (c + 1), exactly.
  m_eff  the variance-weighted impressions per user, E[n^2]/E[n], not the
         plain mean. A few very heavy users count for more than their
         share, which is why the plain average understates the damage.

Both come from the GENERATOR's parameters, with no reference to either
estimator. So the standard error pooled over impressions should fall short
of the honest one by sqrt(DE), and a nominal 95% interval that is sqrt(DE)
times too narrow covers

    2 * Phi(z / sqrt(DE)) - 1

of the time. That is a number computed from theory before any simulation,
and the simulation either lands on it or the theory is wrong.

WHY IT IS RUN AT THREE HETEROGENEITY LEVELS
One match is a coincidence. Three points on the curve, from users who are
all identical to users who differ as much as real ones do, is the claim:
the severity of this bug is a property of YOUR USERS, not of your code.
That is why it survives code review and ships. On a toy fixture where
every user behaves the same, the wrong formula is right.
"""
from __future__ import annotations

import math

import numpy as np

from _harness import header

from explab.inference import ratio_delta
from explab.statlib import norm_cdf, norm_ppf

CONF = 0.95
Z = norm_ppf(0.5 + CONF / 2.0)

# The impression-count process, shared by every level so that the only
# thing changing between levels is how much users differ from one another.
LOG_MEAN, LOG_SD = 3.2, 0.5   # impressions = Poisson(lognormal) + 1
BASE_CTR = 0.11
BUMP = 0.01
N_PER_ARM = 1200
N_SIMS = 6000


def theory_design_effect(concentration: float) -> tuple[float, float, float]:
    """DE, m_eff and ICC from the generator's parameters alone.

    No estimator, no simulation output, no fitted constant. Everything here
    is a property of how the data is made:

      impressions  n = Poisson(lam) + 1,  lam = lognormal(mu, sd)
      click rate   p_i ~ Beta(a, b),  a + b = concentration

    E[lam]    = exp(mu + sd^2/2)
    E[lam^2]  = exp(2mu + 2sd^2)
    E[n]      = E[lam] + 1
    E[n^2]    = E[Poisson^2] + 2E[Poisson] + 1
              = (E[lam^2] + E[lam]) + 2E[lam] + 1

    using Var(Poisson) = E[lam], so E[Poisson^2] = E[lam^2] + E[lam] by the
    law of total variance.
    """
    e_lam = math.exp(LOG_MEAN + LOG_SD * LOG_SD / 2.0)
    e_lam2 = math.exp(2.0 * LOG_MEAN + 2.0 * LOG_SD * LOG_SD)
    e_n = e_lam + 1.0
    e_n2 = (e_lam2 + e_lam) + 2.0 * e_lam + 1.0
    m_eff = e_n2 / e_n
    icc = 1.0 / (concentration + 1.0)
    return 1.0 + (m_eff - 1.0) * icc, m_eff, icc


def impression_level_ci(clicks_c, imp_c, clicks_t, imp_t):
    """The common bug: pool impressions and call it a proportion."""
    p_c = clicks_c.sum() / imp_c.sum()
    p_t = clicks_t.sum() / imp_t.sum()
    se = np.sqrt(
        p_c * (1 - p_c) / imp_c.sum() + p_t * (1 - p_t) / imp_t.sum()
    )
    est = p_t - p_c
    return est, se, (est - Z * se, est + Z * se)


def per_user_ci(clicks_c, imp_c, clicks_t, imp_t):
    """Average of per-user ratios: a different estimand, shown for contrast."""
    r_c = clicks_c / np.maximum(imp_c, 1)
    r_t = clicks_t / np.maximum(imp_t, 1)
    est = r_t.mean() - r_c.mean()
    se = np.sqrt(r_c.var(ddof=1) / r_c.size + r_t.var(ddof=1) / r_t.size)
    return est, (est - Z * se, est + Z * se)


def run(label: str, concentration: float, seed: int, proof) -> dict:
    """Measure all three intervals at one heterogeneity level."""
    de, m_eff, icc = theory_design_effect(concentration)
    predicted_coverage = 2.0 * norm_cdf(Z / math.sqrt(de)) - 1.0

    rng = np.random.default_rng(seed)
    cov_delta = cov_imp = cov_user = 0
    se_delta = se_imp = 0.0
    estimates = np.empty(N_SIMS)

    for i in range(N_SIMS):
        q = rng.standard_normal(2 * N_PER_ARM)
        imp = (rng.poisson(np.exp(LOG_MEAN + LOG_SD * q)) + 1).astype(int)
        ctr = rng.beta(BASE_CTR * concentration,
                       (1 - BASE_CTR) * concentration, size=2 * N_PER_ARM)
        ctr_t = np.clip(ctr[N_PER_ARM:] + BUMP, 1e-9, 1 - 1e-9)
        clicks_c = rng.binomial(imp[:N_PER_ARM], ctr[:N_PER_ARM]).astype(float)
        clicks_t = rng.binomial(imp[N_PER_ARM:], ctr_t).astype(float)
        imp_c = imp[:N_PER_ARM].astype(float)
        imp_t = imp[N_PER_ARM:].astype(float)

        d = ratio_delta(clicks_c, imp_c, clicks_t, imp_t, confidence=CONF)
        cov_delta += d.ci[0] <= BUMP <= d.ci[1]
        se_delta += d.se
        estimates[i] = d.estimate

        _, s_imp, ci = impression_level_ci(clicks_c, imp_c, clicks_t, imp_t)
        cov_imp += ci[0] <= BUMP <= ci[1]
        se_imp += s_imp

        _, ci_u = per_user_ci(clicks_c, imp_c, clicks_t, imp_t)
        cov_user += ci_u[0] <= BUMP <= ci_u[1]

    mean_se_delta = se_delta / N_SIMS
    mean_se_imp = se_imp / N_SIMS
    empirical_sd = float(estimates.std(ddof=1))
    shortfall = mean_se_delta / mean_se_imp

    print(f"  {label}")
    print(f"    theory:  ICC {icc:.4f}   effective impressions/user "
          f"{m_eff:.1f}   ->   design effect {de:.3f}, "
          f"SE shortfall sqrt(DE) = {math.sqrt(de):.3f}x")
    print(f"    delta method      coverage {cov_delta / N_SIMS:.4f}   "
          f"mean SE {mean_se_delta:.6f}   (spread of the estimates "
          f"themselves: {empirical_sd:.6f})")
    print(f"    impression level  coverage {cov_imp / N_SIMS:.4f}   "
          f"mean SE {mean_se_imp:.6f}   ({shortfall:.3f}x too small, theory "
          f"said {math.sqrt(de):.3f}x)")
    print(f"    per-user ratios   coverage {cov_user / N_SIMS:.4f}   "
          f"(different estimand; covering THIS truth is not its job)")
    print()

    proof.check_rate(
        f"delta method covers at nominal 95%, {label}",
        cov_delta, N_SIMS, CONF,
        detail="the honest interval is untouched by heterogeneity: its "
               "variance is taken at the unit that was randomised",
    )
    proof.check_close(
        f"the delta SE equals the actual spread of the estimates, {label}",
        mean_se_delta / empirical_sd, 1.0, tolerance=0.02,
        detail=f"mean reported SE {mean_se_delta:.6f} against the standard "
               f"deviation of {N_SIMS:,} point estimates {empirical_sd:.6f}. "
               f"An interval is only honest if its width matches how much the "
               f"estimate actually moves.",
    )
    proof.check_rate(
        f"impression-level coverage lands on the design-effect prediction, "
        f"{label}",
        cov_imp, N_SIMS, predicted_coverage,
        detail=f"predicted {predicted_coverage:.4f} from DE = {de:.3f}, "
               f"computed from the generator's parameters before the "
               f"simulation ran; measured {cov_imp / N_SIMS:.4f}",
    )
    return {
        "label": label, "de": de, "shortfall": shortfall,
        "coverage_imp": cov_imp / N_SIMS, "predicted": predicted_coverage,
    }


def run_all(proof) -> list[dict]:
    """Three points on the curve, from identical users to realistic ones."""
    print("  Each level changes ONE thing: how much users' click rates")
    print("  differ. Impressions per user, users per arm, the effect and the")
    print("  code are identical throughout.")
    print()
    return [
        run("identical users (Beta concentration 4000)", 4000.0, 101, proof),
        run("mildly varied users (Beta concentration 40)", 40.0, 303, proof),
        run("realistically varied users (Beta concentration 8)", 8.0, 202,
            proof),
    ]


def main() -> int:
    proof = header(
        "PROOF: the delta method against analysing at the denominator's unit",
        "a 95% interval on a ratio metric covers only when the variance is "
        "taken at the randomised unit, and the cost of getting it wrong is "
        "the design effect",
    )

    levels = run_all(proof)

    # ---- the cross-level claim: theory tracks the measurement everywhere ---
    worst = max(abs(r["shortfall"] / math.sqrt(r["de"]) - 1.0) for r in levels)
    proof.check_close(
        "the SE shortfall equals sqrt(design effect) at every level",
        worst, 0.0, tolerance=0.03,
        detail="; ".join(
            f"{r['label'].split(' (')[0]}: measured {r['shortfall']:.3f} vs "
            f"theory {math.sqrt(r['de']):.3f}"
            for r in levels
        ) + f". Largest relative gap {worst:.1%}. The measured shortfall sits "
            f"a little below theory at every level, and in the same "
            f"direction: the clustering formula ignores the denominator's own "
            f"variance and its covariance with the numerator, which the delta "
            f"method carries and which shaves a little off.",
    )

    span = levels[0]["coverage_imp"] - levels[-1]["coverage_imp"]
    proof.check_true(
        "heterogeneity, not sample size, is what breaks it",
        span > 0.25,
        detail=f"same users per arm, same impressions per user, same effect, "
               f"same code: the impression-level interval covers "
               f"{levels[0]['coverage_imp']:.1%} when users are identical and "
               f"{levels[-1]['coverage_imp']:.1%} when they differ as much as "
               f"real ones do - a span of {span:.1%}. One realistic readout "
               f"in {1 / (1 - levels[-1]['coverage_imp']):.1f} is wrong, with "
               f"p-values optimistic in the same proportion, and nothing in "
               f"the output hints at it.",
    )
    return proof.finish()


if __name__ == "__main__":
    raise SystemExit(main())
