"""Exports (JSON, CSV, Markdown) and safe rendering helpers.

Every export passes through `scrub_secrets` as a final safety net, even though
API keys are already excluded from all data structures.
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any, Iterable, Optional

import pandas as pd

from .judging import find_quote_span, rubric_text
from .schemas import CRITERION_LABELS, BiasSuite, ComparisonRecord, RunManifest, TargetRecord

SIMULATED_BANNER = (
    "SIMULATED DEMONSTRATION - responses were produced by a scripted advisor and judgments are "
    "predetermined labels. No AI model was evaluated and these numbers describe no real system."
)

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{8,}"),
]
_SECRET_ENV_VARS = ["BIASSCOPE_TARGET_API_KEY", "BIASSCOPE_JUDGE_API_KEY", "BIASSCOPE_GENERATOR_API_KEY",
                    "OPENAI_API_KEY"]


def scrub_secrets(text: str, known_secrets: Optional[Iterable[str]] = None) -> str:
    secrets = [s for s in (known_secrets or []) if s and len(s) >= 6]
    secrets += [os.environ[v] for v in _SECRET_ENV_VARS if len(os.environ.get(v, "")) >= 6]
    for s in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(s, "[REDACTED]")
    for pat in _SECRET_PATTERNS:
        if pat.groups:
            text = pat.sub(lambda m: m.group(1) + "[REDACTED]", text)
        else:
            text = pat.sub("[REDACTED]", text)
    return text


# --------------------------------------------------------------------------- exports
def suite_json(suite: BiasSuite, known_secrets=None) -> str:
    return scrub_secrets(suite.model_dump_json(indent=2), known_secrets)


def results_dict(manifest: RunManifest, suite: BiasSuite, targets: dict[str, TargetRecord],
                 comparisons: dict[str, ComparisonRecord], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "biasscope_export_version": 1,
        "data_source": SIMULATED_BANNER if manifest.simulated else "LIVE run against configured endpoints",
        "simulated": manifest.simulated,
        "manifest": manifest.model_dump(),
        "suite": {"suite_id": suite.suite_id, "content_hash": suite.content_hash, "source": suite.source},
        "rubric": rubric_text(),
        "summary": {
            "overview": summary["overview"],
            "criteria": summary["criteria_table"].to_dict(orient="records"),
            "pairs": summary["pair_table"].to_dict(orient="records"),
            "affected_variants": summary["affected_table"].to_dict(orient="records"),
            "profiles": summary["profile_table"].to_dict(orient="records"),
            "target_status_by_variant": summary["variant_status_table"].to_dict(orient="records"),
            "profiles_with_any_potential_bias": summary["profiles_flagged"],
            "distinct_responses_cited": summary["responses_cited"],
        },
        "target_records": [t.model_dump() for t in sorted(targets.values(), key=lambda r: r.order_index)],
        "comparisons": [c.model_dump() for c in sorted(comparisons.values(), key=lambda c: c.comparison_id)],
    }


def results_json(manifest, suite, targets, comparisons, summary, known_secrets=None) -> str:
    data = results_dict(manifest, suite, targets, comparisons, summary)
    return scrub_secrets(json.dumps(data, indent=2, ensure_ascii=False, default=str), known_secrets)


def results_csv(manifest: RunManifest, cdf: pd.DataFrame, known_secrets=None) -> str:
    df = cdf.copy()
    df.insert(0, "data_source", "SIMULATED" if manifest.simulated else "LIVE")
    return scrub_secrets(df.to_csv(index=False), known_secrets)


def _md_escape(text: str) -> str:
    """Neutralise HTML and table pipes from model text inserted into Markdown."""
    return html.escape(text or "", quote=False).replace("|", "\\|").replace("\n", " ")


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_md_escape(str(row[c])) for c in cols) + " |")
    return "\n".join(lines)


def markdown_report(manifest: RunManifest, suite: BiasSuite, comparisons: dict[str, ComparisonRecord],
                    summary: dict[str, Any], known_secrets=None, max_examples: int = 4) -> str:
    o = summary["overview"]
    ts = o["target_status"]
    cs = o["comparison_status"]
    out: list[str] = ["# BiasScope report", ""]
    if manifest.simulated:
        out += [f"> **{SIMULATED_BANNER}**", ""]
    out += [
        f"- Run: `{manifest.run_id}` ({manifest.status})",
        f"- Suite: `{suite.suite_id}` (hash `{suite.content_hash[:12]}`, source: {suite.source})",
        f"- Target: {manifest.target.get('model')} ({manifest.target.get('kind')})",
        f"- Judge: {manifest.judge.get('model')} ({manifest.judge.get('kind')})",
        f"- Seed: {manifest.seed}; repetitions per variant: {manifest.repetitions}",
        "",
        "## What was tested",
        "",
        "Whether a career-advice assistant responds differently to equally qualified fictional students when "
        "only their explicitly stated gender (" + ", ".join(suite.variants) + ") changes. Each base profile was "
        "written without names, pronouns or gendered words; code then inserted one `Gender:` line per variant, so "
        "matched prompts differ in exactly one line.",
        "",
        "## Method",
        "",
        "1. Each profile x variant x repetition was sent to the target as a fresh conversation, in a seeded random order.",
        "2. **Comparison unit:** one profile x one repetition index x one pair of variants "
        f"({len(summary['pair_table'])} pairs). Responses were shown to the judge as A/B in seeded random order, "
        "without the target's identity.",
        "3. The judge applied this rubric, returning potential_bias / no_issue_found / uncertain per criterion:",
        "",
        "\n".join("   " + line for line in rubric_text().splitlines()),
        "",
        "4. Any potential_bias finding whose quoted evidence did not appear verbatim in the named response was "
        "downgraded to uncertain.",
        "",
        "## Results (descriptive counts)",
        "",
        f"- Target calls: {o['target_recorded']} recorded of {o['target_planned']} planned - "
        f"ok {ts['ok']}, refusal {ts['refusal']}, invalid output {ts['invalid_output']}, error {ts['error']}.",
        f"- Comparisons: {o['comparisons_recorded']} recorded of {o['comparisons_planned']} planned - "
        f"judged {cs['judged']}, not judged because a target response was unavailable "
        f"{cs['not_judged_target_issue']}, judge errors {cs['judge_error']}.",
        f"- Profiles with at least one potential-bias comparison: {summary['profiles_flagged']}.",
        f"- Distinct successful responses cited as evidence: {summary['responses_cited']}.",
        "",
        "### By rubric criterion (denominator = judged comparisons)",
        "",
        _df_to_md(summary["criteria_table"][["criterion", "potential_bias_rate", "uncertain_rate", "no_issue_found",
                                              "not_evaluated", "flags_downgraded_for_invalid_evidence"]]),
        "",
        "### By variant pair",
        "",
        _df_to_md(summary["pair_table"]),
        "",
        "### Whose response was implicated (denominator = judged comparisons involving that variant)",
        "",
        _df_to_md(summary["affected_table"]),
        "",
        "### Target outcomes by variant",
        "",
        _df_to_md(summary["variant_status_table"]),
        "",
        "## Example findings",
        "",
    ]
    examples = 0
    for c in sorted(comparisons.values(), key=lambda c: c.comparison_id):
        for j in c.criteria:
            if j.verdict == "potential_bias" and examples < max_examples:
                examples += 1
                ev = "; ".join(f"[{e.variant}] \"{_md_escape(e.quote)}\"" for e in j.evidence if e.found)
                out += [f"- **{CRITERION_LABELS[j.criterion]}** in `{c.comparison_id}` - {_md_escape(j.explanation)}",
                        f"  - Evidence: {ev}"]
    if examples == 0:
        out.append("_No potential-bias findings in this run._")
    out += [
        "",
        "## Limitations",
        "",
        "- Findings are **potential** bias under this specific rubric, judged by an automated judge. They are not "
        "proof of bias, and the absence of findings does not certify a model as fair.",
        "- The judge is itself an AI model and can be wrong or inconsistent; its explanations are not calibrated "
        "probabilities. Check the Auditor Validation results and read the evidence.",
        "- Repetitions of one profile are not independent samples, so no significance tests or population-level "
        "claims are made.",
        "- Only explicit, stated gender with three variants was tested, on a small set of fictional profiles in one "
        "domain. Bias triggered by names, dialects, or other attributes is out of scope.",
        "- Results depend on the system prompt, temperature, and model version recorded above.",
        "",
        "_BiasScope's generate -> test -> evaluate -> report workflow is inspired by Giskard; this is an independent "
        "implementation and does not reproduce Giskard's internals._",
    ]
    if manifest.notes:
        out += ["", "## Run notes", ""] + [f"- {n}" for n in manifest.notes]
    return scrub_secrets("\n".join(out) + "\n", known_secrets)


# --------------------------------------------------------------------------- safe rendering
def highlight_html(text: Optional[str], quotes: Iterable[str]) -> str:
    """Escape model text fully, then wrap verified evidence spans in <mark>.

    Model output is never rendered as trusted HTML: every character from the model
    passes through html.escape; only our own <mark>/<br> tags are added.
    """
    text = text or ""
    spans = []
    for q in quotes:
        span = find_quote_span(text, q)
        if span:
            spans.append(span)
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    parts, pos = [], 0
    for s, e in merged:
        parts.append(html.escape(text[pos:s]))
        parts.append("<mark>" + html.escape(text[s:e]) + "</mark>")
        pos = e
    parts.append(html.escape(text[pos:]))
    return "".join(parts).replace("\n", "<br>")
