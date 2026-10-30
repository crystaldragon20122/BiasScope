"""Auditor validation on a small fixture set.

Reference labels are DEVELOPER-AUTHORED, not independent ground truth.
Passing these fixtures shows the auditor handles a few specific behaviours; it
does not validate the auditor in general.

Metrics (per criterion and overall), using only determinate reference labels:
  positive = potential_bias, negative = no_issue_found
  TP/FP/FN/TN computed only where the auditor gave a determinate verdict.
  'uncertain' auditor outputs and judge errors are reported separately.
  precision = TP/(TP+FP), recall = TP/(TP+FN); None when the denominator is 0.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import config
from .execution import classify_response
from .judging import judge_pair
from .scenarios import load_curated_profiles
from .schemas import CRITERIA, CRITERION_LABELS, ProviderConfig

LABELS = {"potential_bias", "no_issue_found", "indeterminate"}


def load_fixtures(path: Path = config.VALIDATION_FIXTURES_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("purpose") != "validation_only":
        raise ValueError("fixture file must be marked purpose=validation_only")
    for f in data["fixtures"]:
        for crit, lab in f.get("reference_labels", {}).items():
            if crit not in CRITERIA or lab not in LABELS:
                raise ValueError(f"{f['fixture_id']}: bad label {crit}={lab}")
    return data


def developer_labels(fixtures: dict[str, Any]) -> dict[tuple[str, str], str]:
    return {(f["fixture_id"], c): lab for f in fixtures["fixtures"] for c, lab in f.get("reference_labels", {}).items()}


def import_labels(content: str, filename: str, fixtures: dict[str, Any]) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Import independently reviewed labels from JSON or CSV.

    Expected fields: fixture_id, criterion, label, reviewer (optional).
    Returns (labels, problems). Rows with problems are skipped, never guessed.
    """
    ids = {f["fixture_id"] for f in fixtures["fixtures"]}
    rows: list[dict[str, Any]]
    if filename.lower().endswith(".json"):
        data = json.loads(content)
        rows = data.get("labels", data) if isinstance(data, dict) else data
    else:
        rows = list(csv.DictReader(io.StringIO(content)))
    labels: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for i, r in enumerate(rows, start=1):
        fid, crit, lab = str(r.get("fixture_id", "")).strip(), str(r.get("criterion", "")).strip(), str(r.get("label", "")).strip()
        if fid not in ids:
            problems.append(f"row {i}: unknown fixture_id {fid!r}")
        elif crit not in CRITERIA:
            problems.append(f"row {i}: unknown criterion {crit!r}")
        elif lab not in LABELS:
            problems.append(f"row {i}: label must be one of {sorted(LABELS)}")
        else:
            labels[(fid, crit)] = lab
    return labels, problems


def _verdict_map(criteria) -> dict[str, Any]:
    return {j.criterion: j for j in criteria}


def run_validation(judge, cfg: ProviderConfig, fixtures: dict[str, Any],
                   assistant_description: str = config.DEFAULT_ASSISTANT_DESCRIPTION,
                   order_check: bool = True, progress=None) -> dict[str, Any]:
    """Run the auditor on every fixture (and on the swapped order). Returns raw rows."""
    profiles = {p.profile_id: p for p in load_curated_profiles()}
    rows: list[dict[str, Any]] = []
    total = len(fixtures["fixtures"])
    for n, f in enumerate(fixtures["fixtures"], start=1):
        status_a, _ = classify_response(f["response_a"])
        status_b, _ = classify_response(f["response_b"])
        expected = f.get("expected_target_status", {"A": "ok", "B": "ok"})
        base = {"fixture_id": f["fixture_id"], "category": f["category"],
                "target_status_A": status_a, "target_status_B": status_b,
                "target_classification_correct": (status_a == expected["A"] and status_b == expected["B"])}
        if status_a != "ok" or status_b != "ok":
            rows.append({**base, "judged": False, "judge_status": "not_judged_target_issue",
                         "orders": {}, "notes": f.get("notes", "")})
            if progress:
                progress(n, total)
            continue
        profile = profiles[f["profile_id"]]
        orders = {}
        variants = [("forward", f["a_variant"], f["response_a"], f["b_variant"], f["response_b"])]
        if order_check:
            variants.append(("swapped", f["b_variant"], f["response_b"], f["a_variant"], f["response_a"]))
        for name, av, at, bv, bt in variants:
            ctx = {"comparison_id": f"{f['fixture_id']}-{name}", "a_text": at, "b_text": bt}
            out = judge_pair(judge, cfg, profile, av, at, bv, bt, assistant_description, ctx)
            orders[name] = out
        rows.append({**base, "judged": True, "judge_status": orders["forward"]["status"], "orders": orders,
                     "notes": f.get("notes", "")})
        if progress:
            progress(n, total)
    return {"rows": rows, "auditor": cfg.public_dict()}


def _ratio(k: int, n: int) -> Optional[float]:
    return None if n == 0 else k / n


def compute_metrics(result: dict[str, Any], labels: dict[tuple[str, str], str]) -> dict[str, Any]:
    per_crit = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "uncertain_on_positive": 0, "uncertain_on_negative": 0,
                    "judge_error": 0, "indeterminate_reference": 0, "unlabeled": 0} for c in CRITERIA}
    detail_rows = []
    order_rows = []
    for row in result["rows"]:
        if not row["judged"]:
            continue
        fwd = row["orders"]["forward"]
        fwd_map = _verdict_map(fwd["criteria"]) if fwd["status"] == "judged" else {}
        swp = row["orders"].get("swapped")
        swp_map = _verdict_map(swp["criteria"]) if swp and swp["status"] == "judged" else {}
        for c in CRITERIA:
            ref = labels.get((row["fixture_id"], c))
            m = per_crit[c]
            pred = fwd_map[c].verdict if c in fwd_map else "judge_error"
            outcome = ""
            if ref is None:
                m["unlabeled"] += 1
                outcome = "no reference label"
            elif ref == "indeterminate":
                m["indeterminate_reference"] += 1
                outcome = "reference indeterminate (excluded)"
            elif pred == "judge_error":
                m["judge_error"] += 1
                outcome = "judge error"
            elif pred == "uncertain":
                m["uncertain_on_positive" if ref == "potential_bias" else "uncertain_on_negative"] += 1
                outcome = "uncertain (reported separately)"
            else:
                key = {("potential_bias", "potential_bias"): "TP", ("potential_bias", "no_issue_found"): "FP",
                       ("no_issue_found", "potential_bias"): "FN", ("no_issue_found", "no_issue_found"): "TN"}[(pred, ref)]
                m[key] += 1
                outcome = key
            detail_rows.append({"fixture_id": row["fixture_id"], "category": row["category"],
                                "criterion": CRITERION_LABELS[c], "reference": ref or "-", "auditor": pred,
                                "outcome": outcome,
                                "explanation": fwd_map[c].explanation if c in fwd_map else fwd.get("detail", ""),
                                "notes": "; ".join(fwd_map[c].validation_notes) if c in fwd_map else ""})
            if swp is not None:
                if c in fwd_map and c in swp_map:
                    v1, v2 = fwd_map[c], swp_map[c]
                    changed = (v1.verdict != v2.verdict) or (
                        v1.verdict == "potential_bias" and set(v1.affected_variants) != set(v2.affected_variants))
                    order_rows.append({"fixture_id": row["fixture_id"], "criterion": CRITERION_LABELS[c],
                                       "forward": v1.verdict, "swapped": v2.verdict,
                                       "forward_affected": ",".join(v1.affected_variants),
                                       "swapped_affected": ",".join(v2.affected_variants),
                                       "changed": changed})
                else:
                    order_rows.append({"fixture_id": row["fixture_id"], "criterion": CRITERION_LABELS[c],
                                       "forward": fwd_map[c].verdict if c in fwd_map else "judge_error",
                                       "swapped": swp_map[c].verdict if c in swp_map else "judge_error",
                                       "forward_affected": "", "swapped_affected": "", "changed": None})

    def summarize(m):
        return {**m, "precision": _ratio(m["TP"], m["TP"] + m["FP"]), "recall": _ratio(m["TP"], m["TP"] + m["FN"]),
                "determinate_scored": m["TP"] + m["FP"] + m["FN"] + m["TN"]}

    table = [{"criterion": CRITERION_LABELS[c], **summarize(m)} for c, m in per_crit.items()]
    total = {k: sum(m[k] for m in per_crit.values()) for k in next(iter(per_crit.values()))}
    table.append({"criterion": "ALL (pooled)", **summarize(total)})

    order_df = pd.DataFrame(order_rows)
    compared = int(order_df["changed"].notna().sum()) if len(order_df) else 0
    changed = int((order_df["changed"] == True).sum()) if len(order_df) else 0  # noqa: E712
    target_rows = [r for r in result["rows"]]
    return {
        "metrics": pd.DataFrame(table),
        "details": pd.DataFrame(detail_rows),
        "order_sensitivity": order_df,
        "order_changed": changed,
        "order_compared": compared,
        "target_classification": pd.DataFrame([{
            "fixture_id": r["fixture_id"], "category": r["category"], "status_A": r["target_status_A"],
            "status_B": r["target_status_B"], "classified_as_expected": r["target_classification_correct"],
            "judged": r["judged"]} for r in target_rows]),
    }
