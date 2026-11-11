import streamlit as st

import ui_common as ui
from biasscope.analysis import summarize
from biasscope.pipeline import create_run
from biasscope.providers import offline_judge_config, offline_target_config
from biasscope.scenarios import build_suite, load_curated_profiles, save_suite

ui.page_header(
    "Try the demo",
    "A full scan in one click, on a <b>pretend</b> AI advisor. It's free, works offline, and shows exactly "
    "what a real scan looks like.",
    kicker="No key needed")
st.markdown(ui.chip("chip-sim", "SIMULATED") +
            '<span class="muted">The advisor’s answers are pre-written and the judge uses pre-set labels. '
            "This shows how BiasScope works – it doesn’t test a real AI.</span>", unsafe_allow_html=True)
st.write("")

st.markdown('<span class="bs-step">1</span> <b style="margin-left:6px">Pick a pretend advisor</b>',
            unsafe_allow_html=True)
choice = st.radio(
    "Pretend advisor", ["stereotyped", "neutral"], horizontal=True, label_visibility="collapsed", key="demo_behavior",
    format_func=lambda b: {"stereotyped": "Biased advisor – slips in obvious stereotypes",
                           "neutral": "Fair advisor – only harmless wording changes"}[b])
with st.expander("Options"):
    n = st.slider("Number of student profiles", 2, 12, 10, key="demo_n")
    reps = st.slider("Times each question is asked", 1, 5, 3, key="demo_reps")
    fails = st.checkbox("Include a few staged problems (a timeout, an error, a refusal, a broken judge reply, "
                        "a made-up quote) to show how BiasScope reports them honestly", True, key="demo_fails")

st.write("")
st.markdown('<span class="bs-step">2</span> <b style="margin-left:6px">Run it</b>', unsafe_allow_html=True)
st.caption(f"{n} profiles × 3 genders × {reps} = {n * 3 * reps} answers, compared in {n * 3 * reps} pairs. Takes a second.")
if st.button("Run the demo scan", type="primary", icon=":material/play_arrow:"):
    suite = build_suite(load_curated_profiles()[:n], source="curated")
    save_suite(suite)
    t_cfg, j_cfg = offline_target_config(choice, fails), offline_judge_config(fails)
    m = create_run(ui.store(), suite, t_cfg, j_cfg, repetitions=reps, seed=42)
    if ui.execute_with_progress(m.run_id, t_cfg, j_cfg):
        st.session_state["demo_last_run"] = m.run_id

run_id = st.session_state.get("demo_last_run")
if run_id:
    st.write("")
    st.markdown('<span class="bs-step">3</span> <b style="margin-left:6px">See what it found</b>',
                unsafe_allow_html=True)
    try:
        manifest, suite, targets, comps = ui.load_run(run_id)
    except Exception:
        st.stop()
    s = summarize(manifest, suite, targets, comps)
    ct = s["criteria_table"]
    judged = s["overview"]["comparison_status"]["judged"]
    c = st.columns(4)
    for col, row in zip(c, ct.itertuples()):
        col.metric(row.criterion, f"{row.potential_bias} / {judged}", help="Pairs flagged as potential bias, "
                   "out of pairs the judge actually checked.")
    who = s["affected_table"]
    st.caption(f"Profiles with at least one flag: {s['profiles_flagged']}. "
               + ("The biased advisor's planted lines were found – open the evidence to read them."
                  if manifest.target.get("mock_behavior") == "stereotyped"
                  else "The fair advisor's wording changes were not counted as bias."))
    l1, l2 = st.columns(2)
    l1.page_link("views/results.py", label="Explore the charts", icon=":material/donut_large:", width="stretch")
    l2.page_link("views/evidence.py", label="Read the evidence side by side", icon=":material/compare:",
                 width="stretch")
    st.info("Tip: run it again with the other advisor and compare the two on the Results page.")

ui.footer()
