import pandas as pd
import streamlit as st

import ui_common as ui
from biasscope.baseline import baseline_config
from biasscope.providers import build_provider
from biasscope.schemas import CRITERIA
from biasscope.validation import compute_metrics, developer_labels, import_labels, load_fixtures, run_validation

ui.page_header(
    "Check the judge",
    "The judge is an AI too, so it can be wrong. This page tests it on 15 hand-written examples where we already "
    "know the answer: obvious stereotypes, fair answers with harmless differences, tricky in-between cases, "
    "answers that try to boss the judge around, refusals and broken outputs.",
    kicker="Who checks the checker?")
st.caption("The expected answers were written by the developer, not independent reviewers. Passing shows the judge "
           "handles these specific situations – not that it's right about everything. You can upload labels from "
           "other reviewers below.")

fixtures = load_fixtures()
c1, c2 = st.columns(2)
auditor = c1.radio("Judge to test", ["practice", "real"], key="val_auditor",
                   format_func=lambda a: {"practice": "Free word-matching checker",
                                          "real": "Real AI judge (your API key)"}[a])
label_set = c2.radio("Expected answers", ["dev", "import"], key="val_labels",
                     format_func=lambda a: {"dev": "Developer's labels", "import": "Upload reviewers' labels"}[a])
labels = developer_labels(fixtures)
if label_set == "import":
    up = st.file_uploader("Labels file (CSV or JSON with fixture_id, criterion, label, reviewer)",
                          type=["csv", "json"], key="labels_upload")
    st.download_button("Download a blank labels template", "fixture_id,criterion,label,reviewer\n" + "".join(
        f"{f['fixture_id']},{c},,\n" for f in fixtures["fixtures"] if f.get("reference_labels") for c in CRITERIA),
        "labels_template.csv", "text/csv")
    if up is None:
        st.stop()
    labels, problems = import_labels(up.getvalue().decode("utf-8"), up.name, fixtures)
    if problems:
        st.warning("Skipped rows: " + "; ".join(problems[:10]))
    st.success(f"Using {len(labels)} uploaded labels.")
order_check = st.checkbox("Also run each example with the two answers swapped (checks whether order sways the judge)",
                          True, key="val_order")

if auditor == "practice":
    cfg = baseline_config()
    ready = True
else:
    ui.provider_form("judge", default_preset="OpenAI")
    cfg = ui.config_from_state("judge")
    missing = ui.missing_fields(cfg)
    if missing:
        st.warning("Still needed: " + ", ".join(missing))
    n = sum(1 for f in fixtures["fixtures"] if f.get("reference_labels"))
    st.caption(f"About {n * (2 if order_check else 1)} judge requests (a few cents at most).")
    ready = not missing

if st.button("Test the judge", type="primary", icon=":material/fact_check:", disabled=not ready):
    bar = st.progress(0.0)
    st.session_state["val_result"] = run_validation(
        build_provider(cfg), cfg, fixtures, order_check=order_check,
        progress=lambda n, t: bar.progress(n / t, text=f"Example {n} of {t}"))

res = st.session_state.get("val_result")
if not res:
    with st.expander("Browse the 15 examples"):
        st.dataframe(pd.DataFrame(fixtures["fixtures"]).drop(columns=["reference_labels"]), hide_index=True,
                     width="stretch")
    ui.footer()
    st.stop()

mt = compute_metrics(res, labels)
pooled = mt["metrics"].iloc[-1]
st.subheader(f"Results for: {res['auditor']['model']}")


def ratio(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.0%}"


k1, k2, k3, k4 = st.columns(4)
k1.metric("Bias caught (recall)", ratio(pooled["recall"]),
          help=f"Of the examples that really contain bias, how many it flagged: {pooled['TP']} of "
               f"{pooled['TP'] + pooled['FN']} (unsure answers excluded).")
k2.metric("Flags that were right (precision)", ratio(pooled["precision"]),
          help=f"Of everything it flagged, how much really was bias: {pooled['TP']} of {pooled['TP'] + pooled['FP']}.")
k3.metric("Unsure answers", int(pooled["uncertain_on_positive"] + pooled["uncertain_on_negative"]),
          help="Reported separately, never counted as right.")
k4.metric("Changed when swapped", f"{mt['order_changed']} / {mt['order_compared']}" if mt["order_compared"] else "n/a")

met = mt["metrics"].copy()
for col in ("precision", "recall"):
    met[col] = met[col].map(ratio)
with st.expander("Detailed counts by bias type"):
    st.caption("TP = caught real bias · FP = false alarm · FN = missed bias · TN = correctly quiet.")
    st.dataframe(met[["criterion", "TP", "FP", "FN", "TN", "precision", "recall", "uncertain_on_positive",
                      "uncertain_on_negative", "judge_error", "indeterminate_reference"]],
                 hide_index=True, width="stretch")
with st.expander("Refusals and broken answers"):
    st.caption("These must be recognised and never judged or counted as fair.")
    st.dataframe(mt["target_classification"], hide_index=True, width="stretch")
with st.expander("Every example, one by one"):
    st.dataframe(mt["details"], hide_index=True, width="stretch")
if len(mt["order_sensitivity"]):
    ch = mt["order_sensitivity"][mt["order_sensitivity"]["changed"] == True]  # noqa: E712
    if len(ch):
        with st.expander(f"Verdicts that changed when swapped ({len(ch)})"):
            st.dataframe(ch, hide_index=True, width="stretch")

ui.footer()
