#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn a Readout into something a person will actually read.

The format is deliberate, and the order is the argument:

    DECISION first, in one word, with the reasons under it. A readout that
    makes the reader scroll to find the verdict will have its verdict
    invented for it by whoever skims fastest.

    THE CHECK that could invalidate everything, second. Sample ratio
    before effects, always.

    THE PRIMARY METRIC, with its interval before its p-value. The interval
    is the finding; the p-value is a summary of the interval. Putting the
    p-value first is how readouts end up as a hunt for one number below
    0.05.

    WHAT THE EXPERIMENT COULD HAVE SEEN, next to what it did see. A null
    result without its minimum detectable effect is unreadable: you cannot
    tell whether it means "no" or "we don't know".

    GUARDRAILS, flagged on raw p-values and labelled as uncorrected.

    SECONDARY METRICS last, under the word EXPLORATORY, with q-values.

Two renderers, same content: `render_text` for a terminal and
`render_markdown` for anything that gets pasted into a document or a pull
request.
"""
from __future__ import annotations

from .analyze import MetricOutcome, Readout

__all__ = ["render_text", "render_markdown", "DECISION_WORDS"]

DECISION_WORDS = {
    "ship": "SHIP",
    "do-not-ship": "DO NOT SHIP",
    "flat": "NO EFFECT OF PRACTICAL SIZE",
    "inconclusive": "INCONCLUSIVE - NEEDS MORE DATA",
    "invalid": "INVALID - DO NOT READ THE NUMBERS",
}


def _effect_line(outcome: MetricOutcome, indent: str = "  ") -> list[str]:
    r = outcome.result
    lines = [
        f"{indent}{outcome.spec.name}  ({outcome.spec.kind}, "
        f"{outcome.spec.direction} is better)",
        f"{indent}  control {r.mean_control:.6g}  ->  treatment "
        f"{r.mean_treatment:.6g}",
        f"{indent}  effect  {r.estimate:+.6g}  "
        f"[{r.ci[0]:+.6g}, {r.ci[1]:+.6g}]  at {r.confidence:.0%}",
    ]
    if r.mean_control != 0:
        rlo, rhi = r.relative_ci
        lines.append(
            f"{indent}  relative {r.relative_lift:+.2%}  [{rlo:+.2%}, {rhi:+.2%}]"
        )
    tail = f"{indent}  p {r.p_value:.4g}"
    if outcome.q_value is not None:
        tail += f"   q {outcome.q_value:.4g} (BH)"
    tail += f"   method {r.method}"
    lines.append(tail)
    if outcome.detectable is not None:
        lines.append(
            f"{indent}  this sample could detect {outcome.detectable:.6g} or "
            f"larger at 80% power"
        )
    if outcome.spec.practical_threshold is not None:
        thr = outcome.spec.practical_threshold
        verdict = (
            "interval still admits an effect worth shipping for"
            if outcome.beyond_practical else
            "whole interval is smaller than the threshold"
        )
        lines.append(
            f"{indent}  practical threshold +/-{thr:.6g}: {verdict}"
        )
    for note in outcome.notes:
        lines.append(f"{indent}  note: {note}")
    return lines


def render_text(readout: Readout, *, title: str = "experiment readout") -> str:
    """Plain-text readout for a terminal or a log."""
    width = 74
    out: list[str] = ["=" * width, title.upper(), "=" * width, ""]

    out.append(f"DECISION: {DECISION_WORDS[readout.decision]}")
    for reason in readout.reasons:
        out.append(f"  - {reason}")
    out.append("")

    out.append("-" * width)
    out.append("ASSIGNMENT CHECK (runs before anything is believed)")
    out.append("-" * width)
    out.append(str(readout.srm))
    out.append("")

    out.append(f"units: {readout.n_control:,} control / "
               f"{readout.n_treatment:,} treatment   alpha {readout.alpha:g}")
    out.append("")

    out.append("-" * width)
    out.append("PRIMARY METRIC (declared in advance, not corrected)")
    out.append("-" * width)
    if readout.primary is None:
        out.append("  none declared - nothing decides this experiment")
    else:
        out.extend(_effect_line(readout.primary))
    out.append("")

    if readout.guardrails:
        out.append("-" * width)
        out.append("GUARDRAILS (uncorrected on purpose: a correction would "
                   "hide harm)")
        out.append("-" * width)
        for g in readout.guardrails:
            flag = (" <-- ALERT" if g.result.significant
                    and not g.moved_in_good_direction else "")
            out.append(f"  {g.spec.name}{flag}")
            out.extend(_effect_line(g, indent="    "))
            out.append("")

    if readout.secondary:
        out.append("-" * width)
        out.append("SECONDARY METRICS - EXPLORATORY")
        out.append("  Benjamini-Hochberg across this group. These cannot "
                   "change the decision;")
        out.append("  they are hypotheses for the next experiment.")
        out.append("-" * width)
        for s in readout.secondary:
            out.extend(_effect_line(s, indent="    "))
            out.append("")
        if readout.secondary_correction is not None:
            out.append(f"  {readout.secondary_correction.guarantee}")
            out.append("")

    out.append("=" * width)
    return "\n".join(out)


def _md_row(outcome: MetricOutcome) -> str:
    r = outcome.result
    rel = f"{r.relative_lift:+.2%}" if r.mean_control != 0 else "n/a"
    q = f"{outcome.q_value:.4g}" if outcome.q_value is not None else "-"
    det = f"{outcome.detectable:.4g}" if outcome.detectable is not None else "-"
    return (
        f"| {outcome.spec.name} | {outcome.spec.role} | "
        f"{r.mean_control:.6g} | {r.mean_treatment:.6g} | "
        f"{r.estimate:+.6g} | [{r.ci[0]:+.4g}, {r.ci[1]:+.4g}] | {rel} | "
        f"{r.p_value:.4g} | {q} | {det} |"
    )


def render_markdown(readout: Readout, *, title: str = "Experiment readout") -> str:
    """Markdown readout, for a document or a pull request."""
    out: list[str] = [f"# {title}", ""]
    out.append(f"**Decision: {DECISION_WORDS[readout.decision]}**")
    out.append("")
    for reason in readout.reasons:
        out.append(f"- {reason}")
    out.append("")

    srm = readout.srm
    status = "passed" if srm.passed else "**FAILED**"
    out.append("## Assignment check")
    out.append("")
    out.append(
        f"Sample ratio {status}: chi-square {srm.chi_square:.4g} on {srm.df} df, "
        f"p = {srm.p_value:.3g} against a threshold of {srm.threshold:g}. "
        f"Worst arm `{srm.worst_arm}` at {srm.observed_share:.4%} against "
        f"{srm.expected_share:.4%} expected."
    )
    if srm.detectable is not None:
        out.append("")
        out.append(
            f"At this sample size the check would have caught an imbalance of "
            f"{srm.detectable:.4%} or more."
        )
    out.append("")

    out.append("## Metrics")
    out.append("")
    out.append(
        "| metric | role | control | treatment | effect | "
        f"{readout.primary.result.confidence:.0%} interval | relative | p | "
        "q (BH) | MDE |"
        if readout.primary is not None else
        "| metric | role | control | treatment | effect | interval | relative "
        "| p | q (BH) | MDE |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for outcome in readout.all_outcomes:
        out.append(_md_row(outcome))
    out.append("")
    out.append(
        f"Units: {readout.n_control:,} control, {readout.n_treatment:,} "
        f"treatment. Alpha {readout.alpha:g}, two-sided throughout."
    )
    out.append("")
    out.append(
        "Guardrail p-values are uncorrected by design: correcting them would "
        "raise the bar for noticing harm. Secondary metrics carry "
        "Benjamini-Hochberg q-values and are exploratory - they cannot change "
        "the decision above."
    )
    notes = [
        f"- `{o.spec.name}`: {note}"
        for o in readout.all_outcomes for note in o.notes
    ]
    if notes:
        out.append("")
        out.append("## Notes raised during analysis")
        out.append("")
        out.extend(notes)
    out.append("")
    return "\n".join(out)
