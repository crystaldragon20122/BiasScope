import json

import pytest
from conftest import chat_reply

from biasscope.baseline import baseline_config, baseline_judgment
from biasscope.generation import generate_profiles
from biasscope.providers import OpenAICompatibleProvider, build_provider
from biasscope.schemas import CRITERIA, ProviderConfig, ProviderResponse
from biasscope.validation import compute_metrics, developer_labels, import_labels, load_fixtures, run_validation


@pytest.fixture(scope="module")
def fixtures():
    return load_fixtures()


def test_fixture_set_covers_required_categories(fixtures):
    cats = {f["category"] for f in fixtures["fixtures"]}
    assert {"stereotyped", "fair_wording", "ambiguous", "injection", "refusal", "malformed"} <= cats
    assert "developer-authored" in fixtures["label_source"]


def test_baseline_validation_metrics_and_target_classification(fixtures):
    cfg = baseline_config()
    res = run_validation(build_provider(cfg), cfg, fixtures)
    mt = compute_metrics(res, developer_labels(fixtures))
    pooled = mt["metrics"].iloc[-1]
    assert pooled["criterion"] == "ALL (pooled)"
    assert pooled["TP"] + pooled["FN"] + pooled["uncertain_on_positive"] + pooled["judge_error"] == sum(
        1 for f in fixtures["fixtures"] for v in f["reference_labels"].values() if v == "potential_bias")
    # refusals / malformed outputs are classified and never judged
    tc = mt["target_classification"]
    assert tc["classified_as_expected"].all()
    assert not tc[tc["category"].isin(["refusal", "malformed"])]["judged"].any()
    # the keyword baseline is symmetric, so swapping A/B never changes its verdicts
    assert mt["order_changed"] == 0 and mt["order_compared"] > 0


def test_precision_recall_zero_denominators():
    res = {"rows": [], "auditor": {}}
    mt = compute_metrics(res, {})
    for row in mt["metrics"].itertuples():
        assert row.precision is None and row.recall is None
    assert mt["order_compared"] == 0


class _OrderBiasedJudge:
    """Always flags whichever response is shown as A - a maximally order-sensitive judge."""
    model_name = "order-biased"

    def complete(self, messages, context=None):
        a = context["a_text"]
        quote = a[: max(20, min(60, len(a)))]
        crits = [{"criterion": c, "verdict": "potential_bias", "affected_response": "A",
                  "explanation": "A looks worse.", "evidence": [{"response": "A", "quote": quote}]} for c in CRITERIA]
        return ProviderResponse(text=json.dumps({"criteria": crits}), model="order-biased")


def test_order_sensitivity_detects_position_bias(fixtures):
    cfg = ProviderConfig(role="judge", kind="openai_compatible", base_url="http://x", model="m",
                         max_retries=0, backoff_base_s=0)
    res = run_validation(_OrderBiasedJudge(), cfg, fixtures)
    mt = compute_metrics(res, developer_labels(fixtures))
    assert mt["order_compared"] > 0
    assert mt["order_changed"] == mt["order_compared"]  # implicated variant flips every time
    pooled = mt["metrics"].iloc[-1]
    assert pooled["FP"] > 0  # flags the fair fixtures


def test_uncertain_reported_separately_not_as_tp_or_fn(fixtures):
    class Unsure:
        model_name = "unsure"

        def complete(self, messages, context=None):
            return ProviderResponse(text=json.dumps({"criteria": [
                {"criterion": c, "verdict": "uncertain", "affected_response": "neither", "explanation": "?",
                 "evidence": []} for c in CRITERIA]}), model="unsure")

    cfg = ProviderConfig(role="judge", kind="openai_compatible", base_url="http://x", model="m", max_retries=0)
    mt = compute_metrics(run_validation(Unsure(), cfg, fixtures, order_check=False), developer_labels(fixtures))
    pooled = mt["metrics"].iloc[-1]
    assert pooled["TP"] == pooled["FP"] == pooled["FN"] == pooled["TN"] == 0
    assert pooled["uncertain_on_positive"] > 0 and pooled["uncertain_on_negative"] > 0
    assert pooled["precision"] is None


def test_import_independent_labels_csv_and_json(fixtures):
    csv_text = ("fixture_id,criterion,label,reviewer\nF01,explicit_stereotyping,potential_bias,R1\n"
                "F05,unequal_encouragement,no_issue_found,R1\nF99,explicit_stereotyping,potential_bias,R1\n"
                "F02,explicit_stereotyping,maybe,R1\n")
    labels, problems = import_labels(csv_text, "labels.csv", fixtures)
    assert labels == {("F01", "explicit_stereotyping"): "potential_bias",
                      ("F05", "unequal_encouragement"): "no_issue_found"}
    assert len(problems) == 2
    js = json.dumps({"labels": [{"fixture_id": "F03", "criterion": "unequal_opportunities", "label": "potential_bias"}]})
    labels2, problems2 = import_labels(js, "labels.json", fixtures)
    assert labels2 == {("F03", "unequal_opportunities"): "potential_bias"} and not problems2


def test_baseline_flags_obvious_stereotype_and_ignores_wording():
    out = baseline_judgment("Girls are naturally better suited to nursing.", "Medicine suits you.")
    st = next(c for c in out["criteria"] if c["criterion"] == "explicit_stereotyping")
    assert st["verdict"] == "potential_bias" and st["affected_response"] == "A"
    out2 = baseline_judgment("Great profile! Apply to the program.", "You have a strong profile; apply to the program.")
    assert all(c["verdict"] == "no_issue_found" for c in out2["criteria"])


# --------------------------------------------------------------------------- generation


def test_generator_output_is_validated_and_bad_profiles_rejected(fake_server):
    good = {"education_level": "College junior studying chemistry", "academic_strengths": ["analytical chemistry"],
            "grades": "GPA 3.4", "interests": ["pharmaceuticals"], "experience": ["Lab assistant"],
            "goals": "Wants to work in drug development."}
    gendered = dict(good, goals="She wants to work in drug development.")
    named = dict(good, name="Jordan")
    payload = {"profiles": [good, gendered, named, "garbage", dict(good, interests=[])]}

    calls = {"n": 0}

    def handler(body, headers, n):
        calls["n"] += 1
        if n == 1:
            return 200, chat_reply("```json\n" + json.dumps(payload) + "\n```")
        return 200, chat_reply("I cannot produce JSON today.")

    srv = fake_server(handler)
    cfg = ProviderConfig(role="generator", kind="openai_compatible", base_url=srv.url, model="gen",
                         backoff_base_s=0, max_retries=0)
    profiles, rejections = generate_profiles(OpenAICompatibleProvider(cfg), cfg, n=3, max_rounds=2)
    assert len(profiles) == 1 and profiles[0].profile_id == "G01"
    reasons = " ".join(r["reason"] for r in rejections)
    assert "demographic" in reasons and "schema" in reasons and "not valid JSON" in reasons
    assert calls["n"] == 2  # bounded rounds; nothing fabricated to fill the gap
