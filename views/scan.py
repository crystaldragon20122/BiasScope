import os

import pandas as pd
import streamlit as st

import ui_common as ui
from biasscope import config
from biasscope.generation import generate_profiles
from biasscope.pipeline import create_run, planned_calls, same_endpoint
from biasscope.providers import build_provider, config_from_public, offline_judge_config, offline_target_config
from biasscope.reporting import suite_json
from biasscope.scenarios import build_suite, list_saved_suites, load_curated_profiles, load_suite, load_suite_json, save_suite

ui.page_header(
    "Run a scan",
    "Test a real AI advisor, step by step. You'll pick the AI to test, pick a judge, choose how big the test is, "
    "and press run.",
    kicker="Full scan")

with st.expander("Before you start: what you need", icon=":material/info:"):
    st.markdown(
        "- **An API key** for each AI you use (OpenAI: platform.openai.com · Claude: platform.claude.com). "
        "A ChatGPT or Claude chat subscription does **not** include API access.\n"
        "- **A little credit.** A default scan is about 90 questions + 90 judge checks – usually well under "
        "$1 with small models. Set a spending limit in your provider's settings.\n"
        "- **Your keys stay private.** They're kept only in this browser session and never saved.\n"
        "- No key? Use the **Try the demo** page instead.")

mode = st.radio("What do you want to test?", ["live", "sim"], horizontal=True, key="scan_mode",
                format_func=lambda m: {"live": "A real AI model", "sim": "The pretend advisor (simulated, free)"}[m])
live = mode == "live"

# ---------------------------------------------------------------- 1 the AI
st.markdown('<span class="bs-step">1</span> <b style="margin-left:6px">The AI being tested</b>', unsafe_allow_html=True)
if live:
    st.caption("This is the AI whose career advice you're checking for bias.")
    ui.provider_form("target", default_preset="OpenAI")
    target_cfg = ui.config_from_state("target")
else:
    behavior = st.radio("Pretend advisor", ["stereotyped", "neutral"], horizontal=True, key="scan_mock",
                        format_func=lambda b: {"stereotyped": "Biased advisor", "neutral": "Fair advisor"}[b])
    target_cfg = offline_target_config(behavior, True)

with st.expander("Optional: describe the assistant or change its instructions"):
    desc = st.text_area("Short description (shown to the judge)", config.DEFAULT_ASSISTANT_DESCRIPTION,
                        key="assistant_description", height=70)
    sys_prompt = st.text_area("Instructions given to the AI being tested (system prompt)",
                              config.DEFAULT_TARGET_SYSTEM_PROMPT, key="custom_prompt", height=100)

# ---------------------------------------------------------------- 2 the judge
st.write("")
st.markdown('<span class="bs-step">2</span> <b style="margin-left:6px">The judge</b>', unsafe_allow_html=True)
if live:
    st.caption("A second AI that compares answers. Use a different model from the one being tested, and keep "
               "the same judge for every scan you want to compare.")
    same = st.checkbox("Use the same model as the judge (not recommended)", key="judge_same")
    if same:
        judge_cfg = target_cfg.model_copy(update={"role": "judge"})
        st.warning("An AI may be less likely to notice problems in its own style of writing. Results will be "
                   "marked with this limitation.")
    else:
        ui.provider_form("judge", default_preset="OpenAI")
        judge_cfg = ui.config_from_state("judge")
    if same_endpoint(target_cfg, judge_cfg) and not same:
        st.warning("The judge is the same model as the AI being tested.")
else:
    judge_cfg = offline_judge_config(True)
    st.caption("Simulated judge: it uses pre-set labels and doesn't read the answers.")

# ---------------------------------------------------------------- 3 the test
st.write("")
st.markdown('<span class="bs-step">3</span> <b style="margin-left:6px">The test</b>', unsafe_allow_html=True)
sources = ["Built-in student profiles", "A saved test", "Upload a test file"]
if live:
    sources.append("Write new profiles with an AI")
source = st.radio("Student profiles", sources, horizontal=True, key="suite_source")
suite = None
curated = load_curated_profiles()
if source.startswith("Built-in"):
    n = st.slider("How many student profiles?", 1, len(curated), min(config.DEFAULT_NUM_PROFILES, len(curated)),
                  key="n_curated")
    suite = build_suite(curated[:n], source="curated")
elif source.startswith("A saved"):
    saved = list_saved_suites()
    if not saved:
        st.info("No saved tests yet. Tests are saved automatically whenever you run a scan.")
    else:
        choice = st.selectbox("Saved test", saved, format_func=lambda p: p.stem, key="saved_suite")
        try:
            suite = load_suite(choice)
        except Exception as exc:
            st.error(f"Couldn't load that test: {exc}")
elif source.startswith("Upload"):
    up = st.file_uploader("Test file (.json downloaded from BiasScope)", type=["json"], key="suite_upload")
    if up is not None:
        try:
            suite = load_suite_json(up.getvalue().decode("utf-8"))
            st.success("Test file checked: nothing was changed since it was saved.")
        except Exception as exc:
            st.error(f"That file didn't pass the check: {exc}")
else:
    with st.expander("Profile-writing AI", expanded=True):
        ui.provider_form("generator", default_preset="OpenAI")
    gen_cfg = ui.config_from_state("generator")
    n_gen = st.number_input("Profiles to write", 1, 30, config.DEFAULT_NUM_PROFILES, key="n_gen")
    st.caption("Each new profile is checked automatically and thrown out if it mentions names, pronouns or gender.")
    if st.button("Write profiles", key="gen_btn"):
        missing = ui.missing_fields(gen_cfg)
        if missing:
            st.error("Still needed: " + ", ".join(missing))
        else:
            with st.spinner("Writing profiles…"):
                profiles, rejections = generate_profiles(build_provider(gen_cfg), gen_cfg, int(n_gen))
            st.session_state["gen_rejections"] = rejections
            if profiles:
                g = build_suite(profiles, source="generated", generator=gen_cfg.public_dict())
                save_suite(g)
                st.session_state["generated_suite"] = g.model_dump_json()
                st.success(f"Kept {len(profiles)} of {int(n_gen)} profiles.")
            else:
                st.error("No usable profiles came back. Nothing was made up; see the rejected ones below.")
    if st.session_state.get("gen_rejections"):
        with st.expander(f"Rejected profiles ({len(st.session_state['gen_rejections'])})"):
            st.dataframe(pd.DataFrame(st.session_state["gen_rejections"]), width="stretch")
    if st.session_state.get("generated_suite"):
        suite = load_suite_json(st.session_state["generated_suite"])

c1, c2 = st.columns(2)
reps = c1.number_input("Times each question is asked", 1, 10, config.DEFAULT_REPETITIONS, key="reps",
                       help="AI answers vary from chat to chat, so each profile is asked more than once.")
seed = c2.number_input("Random seed", 0, 10_000_000, config.DEFAULT_SEED, key="seed",
                       help="Fixes the shuffle order and which answer the judge sees first. Keep it the same "
                            "to compare scans fairly.")

if suite is not None:
    with st.expander(f"Preview: {len(suite.profiles)} profiles → {len(suite.cases)} questions"):
        first = [c for c in suite.cases if c.profile_id == suite.profiles[0].profile_id]
        cols = st.columns(len(first))
        for col, case in zip(cols, first):
            col.markdown(f"**{case.variant}**")
            col.code(case.user_prompt, language=None)
        st.caption("Only the “Gender:” line changes between the three versions.")
    st.download_button("Download this test (to reuse later)", suite_json(suite, ui.known_secrets()),
                       file_name=f"{suite.suite_id}.json", mime="application/json")

# ---------------------------------------------------------------- 4 run
st.write("")
st.markdown('<span class="bs-step">4</span> <b style="margin-left:6px">Run</b>', unsafe_allow_html=True)
if suite is None:
    st.info("Choose the student profiles above to continue.")
else:
    calls = planned_calls(len(suite.profiles), len(suite.variants), int(reps))
    st.markdown(f"This will send **{calls['target_calls']} questions** to the AI being tested and make up to "
                f"**{calls['judge_calls']} judge checks**. Retries after errors can add a few more.")
    ready = True
    if live:
        missing = [f"{who}: {', '.join(m)}" for who, m in (("AI being tested", ui.missing_fields(target_cfg)),
                                                          ("judge", ui.missing_fields(judge_cfg))) if m]
        if missing:
            st.warning("Still needed – " + " · ".join(missing))
            ready = False
        ack = st.checkbox("I understand this uses my API credit.", key="ack_cost")
        ready = ready and ack
        if ready:
            st.caption("Tip: try 2 profiles and 1 repeat first to check everything works (it costs almost nothing).")
    if st.button("Run scan", type="primary", icon=":material/play_arrow:", disabled=not ready):
        save_suite(suite)
        manifest = create_run(ui.store(), suite, target_cfg, judge_cfg, repetitions=int(reps), seed=int(seed),
                              assistant_description=desc, target_system_prompt=sys_prompt)
        if ui.execute_with_progress(manifest.run_id, target_cfg, judge_cfg):
            st.success("Scan complete.")
            l1, l2 = st.columns(2)
            l1.page_link("views/results.py", label="See the results", icon=":material/donut_large:", width="stretch")
            l2.page_link("views/evidence.py", label="Read the evidence", icon=":material/compare:", width="stretch")

incomplete = [m for m in ui.store().list_runs() if m.status != "completed"]
if incomplete:
    with st.expander(f"Finish a scan that stopped early ({len(incomplete)})", icon=":material/restart_alt:"):
        st.caption("Finished answers are kept; only the missing ones are re-tried. For a real-AI scan, enter the "
                   "same keys above first (keys are never stored with scans).")
        m = st.selectbox("Scan", incomplete, key="resume_pick",
                         format_func=lambda m: f"{m.run_id} · {m.status} · {'simulated' if m.simulated else 'live'}")
        if st.button("Resume"):
            s = st.session_state
            tk = s.get("target_api_key") or os.environ.get("BIASSCOPE_TARGET_API_KEY", "")
            jk = tk if m.judge_same_as_target else (s.get("judge_api_key")
                                                    or os.environ.get("BIASSCOPE_JUDGE_API_KEY", ""))
            ui.execute_with_progress(m.run_id, config_from_public(m.target, tk), config_from_public(m.judge, jk))

ui.footer()
