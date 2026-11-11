"""Descriptive analysis with explicit denominators.

Principles:
* Every rate is shown as "count / denominator". Zero denominators give "n/a".
* Uncertain judgments, judge errors and target failures are counted separately
  and never folded into "no issue found".
* Repeated observations of the same profile are NOT independent; we therefore
  report counts and profile-level summaries, not significance tests.
* There is no aggregate "fairness score".
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

from .judging import plan_comparisons
from .schemas import CRITERIA, CRITERION_LABELS, BiasSuite, ComparisonRecord, RunManifest, TargetRecord


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return f"{k} / 0 (n/a)"
    return f"{k} / {n} ({100 * k / n:.1f}%)"


def safe_ratio(k: int, n: int) -> Optional[float]:
    return None if n == 0 else k / n


def targets_df(targets: Iterable[TargetRecord]) -> pd.DataFrame:
    rows = [{
        "record_id": t.record_id, "profile_id": t.profile_id, "variant": t.variant, "repetition": t.repetition,
        "order_index": t.order_index, "status": t.status, "status_detail": t.status_detail,
        "error_message": t.error_message, "model": t.model, "latency_ms": t.latency_ms, "attempts": t.attempts,
        "prompt_tokens": (t.usage or {}).get("prompt_tokens"),
        "completion_tokens": (t.usage or {}).get("completion_tokens"),
        "simulated": t.simulated,
    } for t in targets]
    cols = ["record_id", "profile_id", "variant", "repetition", "order_index", "status", "status_detail",
            "error_message", "model", "latency_ms", "attempts", "prompt_tokens", "completion_tokens", "simulated"]
    return pd.DataFrame(rows, columns=cols)


def comparisons_df(comparisons: Iterable[ComparisonRecord]) -> pd.DataFrame:
    """One row per comparison x criterion. Unjudged comparisons get verdict 'not_evaluated'."""
    rows = []
    for c in comparisons:
        common = {
            "run_id": c.run_id, "comparison_id": c.comparison_id, "profile_id": c.profile_id,
            "repetition": c.repetition, "pair": c.pair_label(), "a_variant": c.a_variant,
            "b_variant": c.b_variant, "comparison_status": c.status, "status_detail": c.status_detail,
            "judge_model": c.judge_model, "simulated": c.simulated,
        }
        if c.status == "judged":
            for j in c.criteria:
                rows.append({**common, "criterion": j.criterion, "verdict": j.verdict,
                             "judge_original_verdict": j.original_verdict,
                             "affected_variants": ";".join(j.affected_variants),
                             "evidence_valid": j.evidence_valid, "explanation": j.explanation,
                             "evidence": " || ".join(f"[{e.response_label}={e.variant}] {e.quote}" for e in j.evidence),
                             "validation_notes": "; ".join(j.validation_notes)})
        else:
            for cid in CRITERIA:
                rows.append({**common, "criterion": cid, "verdict": "not_evaluated", "judge_original_verdict": None,
                             "affected_variants": "", "evidence_valid": None, "explanation": "",
                             "evidence": "", "validation_notes": ""})
    cols = ["run_id", "comparison_id", "profile_id", "repetition", "pair", "a_variant", "b_variant",
            "comparison_status", "status_detail", "criterion", "verdict", "judge_original_verdict",
            "affected_variants", "evidence_valid", "explanation", "evidence", "validation_notes",
            "judge_model", "simulated"]
    return pd.DataFrame(rows, columns=cols)


def summarize(manifest: RunManifest, suite: BiasSuite, targets: dict[str, TargetRecord],
              comparisons: dict[str, ComparisonRecord]) -> dict[str, Any]:
    tdf = targets_df(targets.values())
    cdf = comparisons_df(comparisons.values())
    planned_comparisons = len(plan_comparisons(suite, manifest.repetitions, manifest.seed))

    t_counts = {s: int((tdf["status"] == s).sum()) for s in ("ok", "refusal", "invalid_output", "error")}
    t_done = int(len(tdf))
    c_status = {s: sum(1 for c in comparisons.values() if c.status == s)
                for s in ("judged", "not_judged_target_issue", "judge_error")}
    judged = c_status["judged"]

    overview = {
        "profiles": len(suite.profiles),
        "variants": list(suite.variants),
        "repetitions": manifest.repetitions,
        "target_planned": manifest.planned_target_calls,
        "target_recorded": t_done,
        "target_status": t_counts,
        "target_pending": manifest.planned_target_calls - t_done,
        "comparisons_planned": planned_comparisons,
        "comparisons_recorded": len(comparisons),
        "comparison_status": c_status,
        "comparisons_pending": planned_comparisons - len(comparisons),
    }

    # per-criterion verdict table
    jdf = cdf[cdf["comparison_status"] == "judged"]
    crit_rows = []
    for cid in CRITERIA:
        sub = jdf[jdf["criterion"] == cid]
        pb = int((sub["verdict"] == "potential_bias").sum())
        un = int((sub["verdict"] == "uncertain").sum())
        ni = int((sub["verdict"] == "no_issue_found").sum())
        downgraded = int(((sub["judge_original_verdict"] == "potential_bias") & (sub["verdict"] != "potential_bias")).sum())
        crit_rows.append({
            "criterion": CRITERION_LABELS[cid], "criterion_id": cid,
            "potential_bias": pb, "uncertain": un, "no_issue_found": ni, "judged_comparisons": judged,
            "not_evaluated": planned_comparisons - judged,
            "potential_bias_rate": fmt_rate(pb, judged), "uncertain_rate": fmt_rate(un, judged),
            "flags_downgraded_for_invalid_evidence": downgraded,
        })
    criteria_table = pd.DataFrame(crit_rows)

    # per pair
    pair_rows = []
    from .judging import variant_pairs

    for x, y in variant_pairs(list(suite.variants)):
        pair = f"{x} vs {y}"
        comps = [c for c in comparisons.values() if c.pair_label() == pair]
        j = [c for c in comps if c.status == "judged"]
        anyb = sum(1 for c in j if any(x.verdict == "potential_bias" for x in c.criteria))
        anyu = sum(1 for c in j if any(x.verdict == "uncertain" for x in c.criteria))
        pair_rows.append({"pair": pair, "judged": len(j), "not_evaluated": len(comps) - len(j),
                          "with_any_potential_bias": anyb, "with_any_uncertain": anyu,
                          "potential_bias_rate": fmt_rate(anyb, len(j))})
    pair_table = pd.DataFrame(pair_rows, columns=["pair", "judged", "not_evaluated", "with_any_potential_bias",
                                                  "with_any_uncertain", "potential_bias_rate"])

    # which variant's response was implicated
    aff_rows = []
    for v in suite.variants:
        involving = [c for c in comparisons.values() if c.status == "judged" and v in c.pair]
        row: dict[str, Any] = {"variant": v, "judged_comparisons_involving_variant": len(involving)}
        for cid in CRITERIA:
            k = sum(1 for c in involving for x in c.criteria
                    if x.criterion == cid and x.verdict == "potential_bias" and v in x.affected_variants)
            row[CRITERION_LABELS[cid]] = fmt_rate(k, len(involving))
        aff_rows.append(row)
    affected_table = pd.DataFrame(aff_rows)

    # distinct responses cited as evidence of potential bias
    cited = set()
    for c in comparisons.values():
        for x in c.criteria:
            if x.verdict == "potential_bias":
                for e in x.evidence:
                    if e.found:
                        cited.add(c.a_record_id if e.response_label == "A" else c.b_record_id)
    ok_responses = t_counts["ok"]

    # profile-level view (respects non-independence of repetitions)
    prof_rows = []
    for p in suite.profiles:
        comps = [c for c in comparisons.values() if c.profile_id == p.profile_id]
        j = [c for c in comps if c.status == "judged"]
        flagged = [c for c in j if any(x.verdict == "potential_bias" for x in c.criteria)]
        prof_rows.append({"profile_id": p.profile_id, "education_level": p.education_level,
                          "judged": len(j), "not_evaluated": len(comps) - len(j),
                          "with_any_potential_bias": len(flagged),
                          "rate": fmt_rate(len(flagged), len(j))})
    profile_table = pd.DataFrame(prof_rows)
    profiles_with_judged = int((profile_table["judged"] > 0).sum()) if len(profile_table) else 0
    profiles_flagged = int((profile_table["with_any_potential_bias"] > 0).sum()) if len(profile_table) else 0

    # refusals / errors by variant
    var_rows = []
    for v in suite.variants:
        sub = tdf[tdf["variant"] == v]
        var_rows.append({"variant": v, "recorded": len(sub),
                         "refusal": fmt_rate(int((sub["status"] == "refusal").sum()), len(sub)),
                         "invalid_output": fmt_rate(int((sub["status"] == "invalid_output").sum()), len(sub)),
                         "error": fmt_rate(int((sub["status"] == "error").sum()), len(sub))})
    variant_status_table = pd.DataFrame(var_rows)

    return {
        "overview": overview,
        "criteria_table": criteria_table,
        "pair_table": pair_table,
        "affected_table": affected_table,
        "profile_table": profile_table,
        "variant_status_table": variant_status_table,
        "profiles_flagged": fmt_rate(profiles_flagged, profiles_with_judged),
        "responses_cited": fmt_rate(len(cited), ok_responses),
        "targets_df": tdf,
        "comparisons_df": cdf,
    }


# --------------------------------------------------------------------------- run comparison
BLOCKING_FIELDS = ["suite_hash", "variants", "repetitions", "prompt_template_version", "rubric_version"]


def compare_runs(a: RunManifest, b: RunManifest) -> dict[str, Any]:
    """Check two runs can be compared and list every configuration difference."""
    blocking = []
    for f in BLOCKING_FIELDS:
        if getattr(a, f) != getattr(b, f):
            blocking.append(f"{f} differs: {getattr(a, f)!r} vs {getattr(b, f)!r}")
    diffs = []

    def add(name, va, vb):
        if va != vb:
            diffs.append({"setting": name, a.run_id: str(va), b.run_id: str(vb)})

    for key in ("kind", "base_url", "model", "temperature", "max_tokens", "token_param", "mock_behavior"):
        add(f"target.{key}", a.target.get(key), b.target.get(key))
    for key in ("kind", "base_url", "model", "temperature", "max_tokens"):
        add(f"judge.{key}", a.judge.get(key), b.judge.get(key))
    add("target_system_prompt", a.target_system_prompt, b.target_system_prompt)
    add("seed (order and A/B assignment)", a.seed, b.seed)
    add("simulated", a.simulated, b.simulated)
    judge_differs = any(d["setting"].startswith("judge.") for d in diffs)
    warnings = []
    if judge_differs:
        warnings.append("The two runs used different judges, so differences may come from the judge, not the target.")
    if a.simulated or b.simulated:
        warnings.append("At least one run is SIMULATED; it does not describe any real model.")
    if a.status != "completed" or b.status != "completed":
        warnings.append("At least one run is not completed; counts may be partial.")
    return {"compatible": not blocking, "blocking": blocking, "differences": diffs, "warnings": warnings}


# --------------------------------------------------------------------------- sunburst
def sunburst_frame(comparisons: Iterable[ComparisonRecord], verdict: str = "potential_bias") -> pd.DataFrame:
    """Rows for a click-to-drill sunburst: whose answer -> bias type -> profile -> count.

    Each criterion judgment with the given verdict contributes one count to every
    variant it implicates ("Not attributed" when the judge named no answer).
    """
    rows = []
    for c in comparisons:
        if c.status != "judged":
            continue
        for j in c.criteria:
            if j.verdict != verdict:
                continue
            who = j.affected_variants or ["Not attributed"]
            for v in who:
                rows.append({"answer_of": v, "bias_type": CRITERION_LABELS[j.criterion],
                             "profile": c.profile_id, "pair": c.pair_label(), "count": 1})
    df = pd.DataFrame(rows, columns=["answer_of", "bias_type", "profile", "pair", "count"])
    if df.empty:
        return df
    return df.groupby(["answer_of", "bias_type", "profile"], as_index=False)["count"].sum()
