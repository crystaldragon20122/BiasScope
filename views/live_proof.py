import html

import streamlit as st

import ui_common as ui
from biasscope.baseline import baseline_config
from biasscope.live_proof import (
    OUTCOME_LABELS,
    criterion_label,
    load_pairs,
    planned_calls,
    run_live_proof,
)
from biasscope.providers import build_provider
from biasscope.reporting import highlight_html

ui.page_header(
    "Live proof",
    "Can a real AI judge tell a biased answer from a fair one? Here are <b>8 pairs of answers</b> we wrote: "
    "<b>4 contain planted bias</b> and <b>4 only differ in harmless ways</b> (wording, order, length). "
    "The judge doesn't know which is which. Each pair is checked twice, with the two answers swapped, "
    "to make sure the order doesn't sway it.",
    kicker="Proof that the checker works")

pairs = load_pairs()
calls = planned_calls(pairs)

with st.expander("Peek at the 8 answer pairs", icon=":material/visibility:"):
    for p in pairs:
        tag = ui.chip("chip-bias", "planted bias") if p["kind"] == "biased" else ui.chip("chip-ok", "fair")
        st.markdown(f"**{html.escape(p['title'])}** {tag}", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for col, v in zip((c1, c2), p["variants"]):
            col.markdown(f"<div class='muted'>Gender: {v}</div><div class='resp'>{html.escape(p['answers'][v])}</div>",
                         unsafe_allow_html=True)
        st.write("")

st.subheader("Choose a judge")
mode = st.radio("Judge", ["real", "practice"], horizontal=True, label_visibility="collapsed", key="proof_mode",
                format_func=lambda m: {"real": "Real AI judge (your API key, ~1 cent)",
                                       "practice": "Practice run – simple word-matching checker (free)"}[m])
if mode == "real":
    st.caption(f"This makes {calls} small requests to the judge (8 pairs × 2 orders), plus a few retries if "
               "something fails. Your key is used only for this session and never saved.")
    ui.provider_form("judge", default_preset="OpenAI")
    cfg = ui.config_from_state("judge")
    missing = ui.missing_fields(cfg)
    if missing:
        st.warning("Still needed: " + ", ".join(missing))
    ready = not missing
else:
    st.caption("The practice checker just looks for phrases like “women are naturally…”. It's free and instant, "
               "but it doesn't understand meaning – so it's a good baseline to compare a real AI judge against.")
    cfg = baseline_config()
    ready = True

if st.button("Run the live proof", type="primary", icon=":material/verified:", disabled=not ready):
    bar = st.progress(0.0, text="Judging…")
    try:
        res = run_live_proof(build_provider(cfg), cfg, pairs,
                             progress=lambda d, t: bar.progress(d / t, text=f"Judging: {d} of {t}"))
        st.session_state["proof_result"] = res
    except Exception as exc:
        st.error(f"Couldn't finish: {type(exc).__name__}: {exc}")

res = st.session_state.get("proof_result")
if res:
    st.divider()
    s = res["summary"]
    judge_name = res["judge"]["model"]
    st.subheader(f"Results for judge: {judge_name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bias caught", f"{s['caught'][0]} / {s['caught'][1]}", help="Checks of biased pairs where the judge "
              "flagged the biased answer (each pair is checked twice).")
    m2.metric("Fair pairs left alone", f"{s['quiet'][0]} / {s['quiet'][1]}", help="Checks of fair pairs with no "
              "bias flag. A flag here is a false alarm.")
    m3.metric("Same verdict both orders", f"{s['consistent'][0]} / {s['consistent'][1]}",
              help="Pairs where swapping A and B didn't change the outcome.")
    m4.metric("Unsure / errors", f"{s['uncertain']} / {s['errors']}", help="Never counted as correct.")
    st.caption("Each pair is checked twice (once with the answers swapped), so 4 biased pairs = 8 checks and "
               "4 fair pairs = 8 checks.")
    if res["judge"]["kind"] == "keyword_baseline":
        st.info("This was the free word-matching checker. Notice which subtle cases it misses – then try a real AI "
                "judge and compare.")
    else:
        st.caption("These pairs were written by the developer, so this is a demonstration, not an independent "
                   "benchmark. For a fuller check of the judge, see **Check the judge**.")

    for r in res["results"]:
        p = r["pair"]
        outcomes = [g["outcome"] for g in r["orders"].values()]
        icon = ":material/check_circle:" if r["all_good"] else ":material/error:"
        kind = "Planted bias" if p["kind"] == "biased" else "Fair"
        with st.expander(f"{kind} · {p['title']}", icon=icon, expanded=not r["all_good"]):
            st.markdown(" ".join(f"<b>{html.escape(name)}:</b> " + ui.chip(*OUTCOME_LABELS[g['outcome']])
                                 for name, g in r["orders"].items()), unsafe_allow_html=True)
            st.caption(f"Planted bias type: {criterion_label(p['criterion'])}")
            first = r["orders"]["as shown"]
            quotes = {v: [] for v in p["variants"]}
            for j in first["flags"] + first["uncertain"]:
                for e in j.evidence:
                    if e.found and e.variant in quotes:
                        quotes[e.variant].append(e.quote)
            c1, c2 = st.columns(2)
            for col, v in zip((c1, c2), p["variants"]):
                label = " (planted bias)" if v == p.get("biased_variant") else ""
                col.markdown(f"<div class='muted'>Gender: {v}{label}</div>"
                             f"<div class='resp'>{highlight_html(p['answers'][v], quotes[v])}</div>",
                             unsafe_allow_html=True)
            for j in first["flags"] + first["uncertain"]:
                cls, txt = ui.VERDICT_CHIP[j.verdict]
                st.markdown(ui.chip(cls, txt) + f"<b>{html.escape(criterion_label(j.criterion))}</b> – "
                            f"{html.escape(j.explanation or '')}", unsafe_allow_html=True)
            if first["outcome"] == "judge_error":
                st.caption(f"Judge error: {first['detail']}")

ui.footer()
