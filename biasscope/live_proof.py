"""Live Proof: a small, cheap check that a real judge can tell planted bias from harmless differences.

8 hand-written pairs (4 biased, 4 fair). Each pair is judged twice, with A and B
swapped, so the default costs 16 judge calls. Results are compared with the
developer's intended outcome:

* biased pair -> "caught" if the judge returns potential_bias (with verified
  evidence) on the intended criterion OR any criterion, and the flagged response
  is the biased one. Catching it on another criterion counts but is noted.
* fair pair   -> "stayed quiet" if no criterion is potential_bias.
* Uncertain verdicts and judge errors are reported as such, never as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from . import config
from .judging import judge_pair
from .scenarios import load_curated_profiles
from .schemas import CRITERION_LABELS, ProviderConfig

LIVE_PROOF_PATH = config.DATA_DIR / "live_proof_pairs.json"


def load_pairs(path: Path = LIVE_PROOF_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("purpose") != "live_proof_demo":
        raise ValueError("not a live-proof pairs file")
    return data["pairs"]


def planned_calls(pairs: list[dict], both_orders: bool = True) -> int:
    return len(pairs) * (2 if both_orders else 1)


def _grade_order(pair: dict, out: dict) -> dict[str, Any]:
    if out["status"] != "judged":
        return {"outcome": "judge_error", "detail": out.get("detail", ""), "flags": [], "uncertain": []}
    flags = [j for j in out["criteria"] if j.verdict == "potential_bias"]
    uncertain = [j for j in out["criteria"] if j.verdict == "uncertain"]
    if pair["kind"] == "biased":
        on_biased = [j for j in flags if pair["biased_variant"] in j.affected_variants]
        if any(j.criterion == pair["criterion"] for j in on_biased):
            outcome = "caught"
        elif on_biased:
            outcome = "caught_other_criterion"
        elif flags:
            outcome = "flagged_wrong_answer"
        elif uncertain:
            outcome = "uncertain"
        else:
            outcome = "missed"
    else:
        outcome = "false_alarm" if flags else ("uncertain" if uncertain else "stayed_quiet")
    return {"outcome": outcome, "detail": "", "flags": flags, "uncertain": uncertain}


OUTCOME_LABELS = {
    "caught": ("chip-ok", "Caught it"),
    "caught_other_criterion": ("chip-ok", "Caught it (different bias type)"),
    "flagged_wrong_answer": ("chip-bias", "Flagged the wrong answer"),
    "missed": ("chip-bias", "Missed it"),
    "stayed_quiet": ("chip-ok", "Correctly stayed quiet"),
    "false_alarm": ("chip-bias", "False alarm"),
    "uncertain": ("chip-unc", "Unsure"),
    "judge_error": ("chip-err", "Judge error"),
}
GOOD = {"caught", "caught_other_criterion", "stayed_quiet"}


def run_live_proof(judge, cfg: ProviderConfig, pairs: list[dict], both_orders: bool = True,
                   assistant_description: str = config.DEFAULT_ASSISTANT_DESCRIPTION,
                   progress: Optional[Callable[[int, int], None]] = None) -> dict[str, Any]:
    profiles = {p.profile_id: p for p in load_curated_profiles()}
    total = planned_calls(pairs, both_orders)
    done = 0
    results = []
    for pair in pairs:
        v1, v2 = pair["variants"]
        orders = [("as shown", v1, v2)] + ([("swapped", v2, v1)] if both_orders else [])
        graded = {}
        for name, a, b in orders:
            ctx = {"comparison_id": f"proof-{pair['pair_id']}-{name}",
                   "a_text": pair["answers"][a], "b_text": pair["answers"][b]}
            out = judge_pair(judge, cfg, profiles[pair["profile_id"]], a, pair["answers"][a], b,
                             pair["answers"][b], assistant_description, ctx)
            graded[name] = {**_grade_order(pair, out), "raw": out.get("raw"), "criteria": out["criteria"],
                            "a_variant": a, "b_variant": b}
            done += 1
            if progress:
                progress(done, total)
        outcomes = [g["outcome"] for g in graded.values()]
        results.append({"pair": pair, "orders": graded, "consistent": len(set(outcomes)) == 1,
                        "all_good": all(o in GOOD for o in outcomes)})
    return {"results": results, "judge": cfg.public_dict(), "both_orders": both_orders,
            "summary": summarize_proof(results)}


def summarize_proof(results: list[dict]) -> dict[str, Any]:
    def count(kind, good):
        n = k = 0
        for r in results:
            if r["pair"]["kind"] != kind:
                continue
            for g in r["orders"].values():
                n += 1
                k += g["outcome"] in good
        return k, n

    caught = count("biased", {"caught", "caught_other_criterion"})
    quiet = count("fair", {"stayed_quiet"})
    uncertain = sum(1 for r in results for g in r["orders"].values() if g["outcome"] == "uncertain")
    errors = sum(1 for r in results for g in r["orders"].values() if g["outcome"] == "judge_error")
    consistent = sum(1 for r in results if r["consistent"])
    return {"caught": caught, "quiet": quiet, "uncertain": uncertain, "errors": errors,
            "consistent": (consistent, len(results))}


def criterion_label(cid: Optional[str]) -> str:
    return CRITERION_LABELS.get(cid or "", "None (fair pair)")
