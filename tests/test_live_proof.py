import json

from biasscope.analysis import sunburst_frame
from biasscope.live_proof import load_pairs, planned_calls, run_live_proof
from biasscope.pipeline import run_offline_demo
from biasscope.schemas import CRITERIA, ProviderConfig, ProviderResponse


def _cfg():
    return ProviderConfig(role="judge", kind="openai_compatible", base_url="http://x", model="m", max_retries=0,
                          backoff_base_s=0)


def test_pairs_file_is_balanced_and_well_formed():
    pairs = load_pairs()
    assert len(pairs) == 8
    assert sum(p["kind"] == "biased" for p in pairs) == 4
    assert {p["criterion"] for p in pairs if p["kind"] == "biased"} == set(CRITERIA)
    for p in pairs:
        assert set(p["answers"]) == set(p["variants"])
        if p["kind"] == "biased":
            assert p["biased_variant"] in p["variants"]
    assert planned_calls(pairs) == 16


class _PerfectJudge:
    """Flags exactly the planted bias, using the context to find it (a stand-in for a good judge)."""
    model_name = "perfect"

    def __init__(self, pairs):
        self.by_text = {}
        for p in pairs:
            if p["kind"] == "biased":
                self.by_text[p["answers"][p["biased_variant"]]] = p["criterion"]

    def complete(self, messages, context=None):
        crits = []
        for c in CRITERIA:
            item = {"criterion": c, "verdict": "no_issue_found", "affected_response": "neither",
                    "explanation": "ok", "evidence": []}
            for label, key in (("A", "a_text"), ("B", "b_text")):
                if self.by_text.get(context[key]) == c:
                    item = {"criterion": c, "verdict": "potential_bias", "affected_response": label,
                            "explanation": "planted", "evidence": [{"response": label, "quote": context[key][:40]}]}
            crits.append(item)
        return ProviderResponse(text=json.dumps({"criteria": crits}), model="perfect")


class _AlwaysFlagA:
    model_name = "flag-a"

    def complete(self, messages, context=None):
        a = context["a_text"]
        return ProviderResponse(text=json.dumps({"criteria": [
            {"criterion": c, "verdict": "potential_bias", "affected_response": "A", "explanation": "x",
             "evidence": [{"response": "A", "quote": a[:30]}]} for c in CRITERIA]}), model="flag-a")


def test_perfect_judge_scores_full_marks_in_both_orders():
    pairs = load_pairs()
    res = run_live_proof(_PerfectJudge(pairs), _cfg(), pairs)
    s = res["summary"]
    assert s["caught"] == (8, 8) and s["quiet"] == (8, 8) and s["consistent"] == (8, 8)


def test_position_biased_judge_is_exposed():
    pairs = load_pairs()
    res = run_live_proof(_AlwaysFlagA(), _cfg(), pairs)
    s = res["summary"]
    assert s["quiet"] == (0, 8)            # false alarms on every fair pair
    assert s["caught"][0] < 8               # half the time it flags the wrong answer
    assert s["consistent"][0] < 8


def test_broken_judge_is_error_not_pass():
    class Broken:
        model_name = "broken"

        def complete(self, messages, context=None):
            return ProviderResponse(text="no json here", model="broken")

    pairs = load_pairs()
    res = run_live_proof(Broken(), _cfg(), pairs)
    s = res["summary"]
    assert s["errors"] == 16 and s["caught"][0] == 0 and s["quiet"][0] == 0


def test_sunburst_counts_match_attributed_findings(store, full_suite):
    m = run_offline_demo(store, full_suite, "stereotyped", simulate_failures=False)
    comps = store.load_comparisons(m.run_id)
    df = sunburst_frame(comps.values())
    expected = sum(max(1, len(j.affected_variants)) for c in comps.values() for j in c.criteria
                   if j.verdict == "potential_bias")
    assert int(df["count"].sum()) == expected
    assert set(df["answer_of"]) <= {"woman", "man", "nonbinary", "Not attributed"}
    neutral = run_offline_demo(store, full_suite, "neutral", simulate_failures=False)
    assert sunburst_frame(store.load_comparisons(neutral.run_id).values()).empty
