import html

import streamlit as st

import ui_common as ui
from biasscope.analysis import summarize
from biasscope.reporting import highlight_html
from biasscope.schemas import CRITERIA, CRITERION_LABELS

ui.page_header("Evidence", "Read two answers side by side. The sentence the judge is worried about is highlighted, "
               "so you can decide for yourself.", kicker="Check the judge's reasoning")

m = ui.pick_run("evidence_run")
if m is None:
    ui.no_runs_message()
    st.stop()
manifest, suite, targets, comps = ui.load_run(m.run_id)
ui.mode_banner(manifest.simulated)
cdf = summarize(manifest, suite, targets, comps)["comparisons_df"]
if cdf.empty:
    st.info("No pairs recorded yet.")
    st.stop()

f1, f2, f3, f4 = st.columns(4)
verd = f1.multiselect("Verdict", list(ui.VERDICT_NAMES), default=["potential_bias", "uncertain"],
                      format_func=lambda v: ui.VERDICT_NAMES[v], key="ev_verd")
crit = f2.multiselect("Bias type", list(CRITERIA), format_func=lambda c: CRITERION_LABELS[c], key="ev_crit")
pair = f3.multiselect("Gender pair", list(dict.fromkeys(cdf["pair"])), key="ev_pair")
prof = f4.multiselect("Profile", sorted(cdf["profile_id"].unique()), key="ev_prof")
view = cdf
if verd:
    view = view[view["verdict"].isin(verd)]
if crit:
    view = view[view["criterion"].isin(crit)]
if pair:
    view = view[view["pair"].isin(pair)]
if prof:
    view = view[view["profile_id"].isin(prof)]
if view.empty:
    st.info("Nothing matches these filters.")
    st.stop()

options = list(dict.fromkeys(zip(view["comparison_id"], view["criterion"])))
st.caption(f"{len(options)} matching results.")
idx = st.session_state.get("ev_idx", 0)
idx = min(idx, len(options) - 1)
n1, n2, n3 = st.columns([1, 4, 1])
if n1.button("← Previous", disabled=idx == 0, width="stretch"):
    idx -= 1
if n3.button("Next →", disabled=idx >= len(options) - 1, width="stretch"):
    idx += 1
st.session_state["ev_idx"] = idx
sel = options[idx]
n2.markdown(f"<div style='text-align:center;padding-top:6px'><b>{idx + 1}</b> of {len(options)} · "
            f"{html.escape(sel[0])}</div>", unsafe_allow_html=True)

comp = comps[sel[0]]
j = next((x for x in comp.criteria if x.criterion == sel[1]), None)
st.markdown(f"### {html.escape(CRITERION_LABELS[sel[1]])}")
st.caption(CRITERIA[sel[1]])
if j is None:
    st.markdown(ui.chip("chip-err", "not judged") + html.escape(comp.status_detail or comp.status),
                unsafe_allow_html=True)
else:
    cls, txt = ui.VERDICT_CHIP[j.verdict]
    extra = ""
    if j.original_verdict and j.original_verdict != j.verdict:
        extra = ui.chip("chip-err", f"judge said {j.original_verdict.replace('_', ' ')} – downgraded")
    sim = ui.chip("chip-sim", "SIMULATED") if comp.simulated else ""
    st.markdown(ui.chip(cls, txt) + extra + sim, unsafe_allow_html=True)
    st.markdown(f"<div class='resp'><b>Judge's reason:</b> {html.escape(j.explanation or '(none given)')}</div>",
                unsafe_allow_html=True)
    if j.validation_notes:
        st.caption("Checks: " + "; ".join(j.validation_notes))

st.write("")
cols = st.columns(2, gap="medium")
for col, variant in zip(cols, comp.pair):
    label = "A" if comp.a_variant == variant else "B"
    rid = comp.a_record_id if label == "A" else comp.b_record_id
    rec = targets.get(rid)
    quotes = [e.quote for e in (j.evidence if j else []) if e.response_label == label and e.found]
    with col:
        st.markdown(f"**Gender: {variant}** <span class='muted'>(judge saw this as answer {label})</span>",
                    unsafe_allow_html=True)
        if rec is None:
            st.warning("No answer recorded.")
            continue
        if rec.status != "ok":
            st.markdown(ui.chip("chip-err", rec.status) + html.escape(rec.status_detail or rec.error_message or ""),
                        unsafe_allow_html=True)
        st.markdown(f"<div class='resp'>{highlight_html(rec.response_text, quotes)}</div>", unsafe_allow_html=True)
        for q in [e.quote for e in (j.evidence if j else []) if e.response_label == label and not e.found]:
            st.markdown(ui.chip("chip-err", "quote NOT found in this answer") + html.escape(q), unsafe_allow_html=True)
        st.caption(f"{rec.model} · {rec.latency_ms} ms · tries: {rec.attempts}")
        with st.expander("Exact question sent"):
            for msg in rec.messages:
                st.caption(msg["role"])
                st.code(msg["content"], language=None)
with st.expander("Raw judge reply"):
    st.code(comp.judge_raw_output or "(none)", language="json")

ui.footer()
