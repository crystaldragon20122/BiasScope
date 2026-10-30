"""BiasScope - Streamlit interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from biasscope import config
from biasscope.analysis import compare_runs, summarize
from biasscope.baseline import baseline_config
from biasscope.generation import generate_profiles
from biasscope.pipeline import create_run, execute_run, planned_calls, same_endpoint
from biasscope.providers import (
    build_provider,
    config_from_public,
    offline_judge_config,
    offline_target_config,
)
from biasscope.reporting import (
    SIMULATED_BANNER,
    highlight_html,
    markdown_report,
    results_csv,
    results_json,
    suite_json,
)
from biasscope.scenarios import (
    build_suite,
    list_saved_suites,
    load_curated_profiles,
    load_suite,
    load_suite_json,
    save_suite,
)
from biasscope.schemas import CRITERIA, CRITERION_LABELS, ProviderConfig
from biasscope.storage import RunStore
from biasscope.validation import compute_metrics, developer_labels, import_labels, load_fixtures, run_validation

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="BiasScope", page_icon="🔎", layout="wide")
config.ensure_workspace()
STORE = RunStore()

st.markdown(
    """
<style>
  .block-container {max-width: 1250px; padding-top: 2rem;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600;
         border:1px solid transparent; margin-right:6px;}
  .chip-sim {background:#fff4e0; color:#8a5300; border-color:#f0d29a;}
  .chip-live {background:#e6f0fa; color:#1f4f7d; border-color:#b9d1ea;}
  .chip-bias {background:#fbe9e7; color:#9c2b1c; border-color:#efc2bb;}
  .chip-ok {background:#e8f3ec; color:#1e6a3a; border-color:#bcdcc7;}
  .chip-unc {background:#f1eefa; color:#4f3a8c; border-color:#d3caee;}
  .chip-err {background:#eeeeee; color:#444; border-color:#ccc;}
  .resp {background:#ffffff; border:1px solid #e1e3e1; border-radius:8px; padding:14px 16px;
         font-size:0.93rem; line-height:1.55; white-space:normal;}
  .resp mark {background:#ffe08a; padding:0 2px; border-radius:3px;}
  .banner-sim {background:#fff4e0; border:1px solid #f0d29a; color:#6b4100; padding:10px 14px;
               border-radius:8px; margin-bottom:12px; font-size:0.92rem;}
  .banner-live {background:#e6f0fa; border:1px solid #b9d1ea; color:#1f4f7d; padding:10px 14px;
                border-radius:8px; margin-bottom:12px; font-size:0.92rem;}
  .muted {color:#5b6168; font-size:0.88rem;}
</style>
""",
    unsafe_allow_html=True,
)

VERDICT_CHIP = {
    "potential_bias": ("chip-bias", "potential bias"),
    "no_issue_found": ("chip-ok", "no issue found"),
    "uncertain": ("chip-unc", "uncertain"),
    "not_evaluated": ("chip-err", "not evaluated"),
}
VERDICT_COLORS = {"potential_bias": "#c0533f", "uncertain": "#7a63b8", "no_issue_found": "#5f9b74",
                  "not_evaluated": "#b5b5b5"}


def chip(cls: str, text: str) -> str:
    return f'<span class="chip {cls}">{html.escape(text)}</span>'


def mode_banner(simulated: bool) -> None:
    if simulated:
        st.markdown(f'<div class="banner-sim"><b>SIMULATED</b> - {html.escape(SIMULATED_BANNER)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner-live"><b>LIVE</b> - responses and judgments came from the configured '
                    'endpoints. Findings are potential bias under this rubric, not proof.</div>',
                    unsafe_allow_html=True)


def known_secrets() -> list[str]:
    return [v for k, v in st.session_state.items() if k.endswith("_api_key") and isinstance(v, str) and v]


# ============================================================================ sidebar
st.sidebar.markdown("## 🔎 BiasScope")
st.sidebar.caption("Automated, rubric-based bias testing for a career-advice AI.")
page = st.sidebar.radio("View", ["Setup and Run", "Results", "Evidence Explorer", "Auditor Validation"],
                        label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    "Workflow inspired by Giskard's generate → test → evaluate → report scan. Independent implementation; "
    "not affiliated with Giskard."
)
st.sidebar.caption(f"Workspace: `{config.WORKSPACE_DIR.name}/` (local files only)")


# ============================================================================ provider config UI
def provider_form(role: str, defaults_prefix: str, default_temp: float) -> ProviderConfig:
    env = lambda k, d="": os.environ.get(f"BIASSCOPE_{defaults_prefix}_{k}", d)  # noqa: E731
    c1, c2 = st.columns(2)
    base_url = c1.text_input("Base URL", value=env("BASE_URL", "https://api.openai.com/v1"), key=f"{role}_base_url",
                             help="Any OpenAI-compatible endpoint. Local examples: http://localhost:11434/v1 "
                                  "(Ollama), http://localhost:1234/v1 (LM Studio).")
    model = c2.text_input("Model", value=env("MODEL", ""), key=f"{role}_model")
    key_env = f"BIASSCOPE_{defaults_prefix}_API_KEY"
    st.text_input(f"API key (session only; leave blank to use ${key_env})", type="password",
                  key=f"{role}_api_key", help="Never written to disk, logs, or exports. Local servers often need no key.")
    c3, c4, c5, c6 = st.columns(4)
    temp = c3.number_input("Temperature", 0.0, 2.0, default_temp, 0.1, key=f"{role}_temp")
    max_tokens = c4.number_input("Max output tokens", 16, 32000, 700 if role != "generator" else 3000, 50,
                                 key=f"{role}_max_tokens")
    token_param = c5.selectbox("Token-limit parameter", ["max_tokens", "max_completion_tokens", "omit"],
                               key=f"{role}_token_param",
                               help="Newer OpenAI models expect max_completion_tokens; most local servers accept max_tokens.")
    send_temp = c6.checkbox("Send temperature", True, key=f"{role}_send_temp",
                            help="Some reasoning models reject a temperature value.")
    c7, c8 = st.columns(2)
    timeout = c7.number_input("Timeout (s)", 5, 600, 90, 5, key=f"{role}_timeout")
    retries = c8.number_input("Max retries", 0, 6, 3, 1, key=f"{role}_retries")
    key = st.session_state.get(f"{role}_api_key") or os.environ.get(key_env, "")
    from pydantic import SecretStr

    return ProviderConfig(role=role, kind="openai_compatible", base_url=base_url.strip(), model=model.strip(),
                          api_key=SecretStr(key) if key else None, temperature=temp if send_temp else None,
                          max_tokens=int(max_tokens), token_param=token_param, timeout_s=float(timeout),
                          max_retries=int(retries))


# ============================================================================ SETUP AND RUN
def page_setup() -> None:
    st.title("Setup and Run")
    with st.expander("How BiasScope works (plain-language explanation)", expanded=False):
        st.markdown(
            """
**The question:** if two students are *identical on paper* except for the gender they state, does a
career-advice AI treat them differently?

1. **Generate** – we start from fictional student profiles written with *no* names, pronouns or gendered words.
   Code then makes three copies of each profile that differ in exactly one line: `Gender: woman`,
   `Gender: man`, or `Gender: nonbinary`.
2. **Test** – each copy is sent to the AI being tested, several times, each in a brand-new conversation.
3. **Evaluate** – a second AI (the *judge*) compares two answers at a time using a fixed rubric of four
   criteria. It must quote the exact sentences that support any finding; quotes that don't really appear
   in the answer are rejected.
4. **Report** – we count findings, always showing *what out of how many*, and let you read the evidence.

**What it is not:** a fairness certificate. Finding nothing does not prove a model is unbiased, and the
judge can make mistakes – which is why the *Auditor Validation* page exists.
"""
        )

    # ---- mode
    st.subheader("1 · Mode")
    mode = st.radio("Scan mode", ["Offline demo (simulated, no API key)", "Live (OpenAI-compatible endpoint)"],
                    horizontal=True, key="mode")
    live = mode.startswith("Live")
    if live:
        st.info("Live mode sends real requests. **Hosted APIs may charge you per request/token.** Local models "
                "(Ollama, LM Studio, vLLM) are free to call but need suitable hardware. A chat subscription "
                "(e.g. a Claude or ChatGPT app plan) does not by itself include API access.")
    else:
        st.markdown(chip("chip-sim", "SIMULATED") +
                    '<span class="muted">Responses come from a scripted advisor; judgments are predetermined '
                    'demonstration labels. Nothing here evaluates a real model.</span>', unsafe_allow_html=True)

    # ---- assistant
    st.subheader("2 · Assistant under test")
    desc = st.text_area("Description of the career-advice assistant", config.DEFAULT_ASSISTANT_DESCRIPTION,
                        key="assistant_description", height=70)
    use_custom = st.checkbox("Use a custom target system prompt", key="use_custom_prompt")
    sys_prompt = config.DEFAULT_TARGET_SYSTEM_PROMPT
    if use_custom:
        sys_prompt = st.text_area("Target system prompt", config.DEFAULT_TARGET_SYSTEM_PROMPT, key="custom_prompt")
    else:
        st.caption(f"Default system prompt: “{config.DEFAULT_TARGET_SYSTEM_PROMPT}”")

    # ---- models
    st.subheader("3 · Models")
    if not live:
        c1, c2 = st.columns(2)
        behavior = c1.selectbox("Simulated target behaviour", ["stereotyped", "neutral"],
                                format_func=lambda b: {"stereotyped": "Advisor with deliberate, obvious stereotypes",
                                                       "neutral": "Neutral advisor (harmless wording differences only)"}[b],
                                key="mock_behavior")
        sim_fail = c2.checkbox("Include simulated failures (1 timeout, 1 persistent error, 1 refusal, "
                               "1 malformed judge reply, 1 fabricated quote)", True, key="sim_fail",
                               help="Shows how errors and invalid evidence are recorded instead of counted as passes.")
        target_cfg = offline_target_config(behavior, sim_fail)
        judge_cfg = offline_judge_config(sim_fail)
        gen_cfg = None
        st.caption("Judge: simulated demo judge – it looks up predetermined labels and performs no semantic evaluation.")
    else:
        with st.expander("Target model (the AI being tested)", expanded=True):
            target_cfg = provider_form("target", "TARGET", 0.7)
        with st.expander("Judge model", expanded=True):
            same = st.checkbox("Use the same endpoint and model as the target", key="judge_same")
            if same:
                judge_cfg = target_cfg.model_copy(update={"role": "judge", "temperature": 0.0
                                                          if target_cfg.temperature is not None else None})
                st.warning("Limitation: the judge is the same model as the target. A model may be less likely to "
                           "notice problems in its own style of output. Prefer a separate judge when practical.")
            else:
                judge_cfg = provider_form("judge", "JUDGE", 0.0)
        with st.expander("Generator model (only needed to generate new profiles)", expanded=False):
            gen_cfg = provider_form("generator", "GENERATOR", 0.9)
        if same_endpoint(target_cfg, judge_cfg):
            st.warning("Target and judge point to the same model/endpoint (limitation noted in the run record).")

    # ---- suite
    st.subheader("4 · Test suite")
    sources = ["Curated profiles (built in)", "Load a saved suite", "Upload a suite JSON"]
    if live:
        sources.append("Generate new profiles with the generator model")
    source = st.radio("Profiles source", sources, key="suite_source")
    suite = None
    curated = load_curated_profiles()
    if source.startswith("Curated"):
        n = st.slider("Number of base profiles", 1, len(curated), min(config.DEFAULT_NUM_PROFILES, len(curated)),
                      key="n_curated")
        suite = build_suite(curated[:n], source="curated")
    elif source.startswith("Load"):
        saved = list_saved_suites()
        if not saved:
            st.info("No saved suites yet. Build one from curated profiles first; suites are saved automatically.")
        else:
            choice = st.selectbox("Saved suite", saved, format_func=lambda p: p.name, key="saved_suite")
            try:
                suite = load_suite(choice)
            except Exception as exc:
                st.error(f"Could not load suite: {exc}")
    elif source.startswith("Upload"):
        up = st.file_uploader("Suite JSON (exported from BiasScope)", type=["json"], key="suite_upload")
        if up is not None:
            try:
                suite = load_suite_json(up.getvalue().decode("utf-8"))
                st.success(f"Suite verified: {suite.suite_id} (hash matches, cases re-expand identically).")
            except Exception as exc:
                st.error(f"Suite rejected: {exc}")
    else:
        n_gen = st.number_input("Profiles to generate", 1, 30, config.DEFAULT_NUM_PROFILES, key="n_gen")
        st.caption("Uses the generator model (≈1–3 requests). Each profile is schema-checked and rejected if it "
                   "contains names, pronouns or gendered terms.")
        if st.button("Generate profiles", key="gen_btn"):
            if not gen_cfg or not gen_cfg.model:
                st.error("Configure the generator model first.")
            else:
                with st.spinner("Generating…"):
                    profiles, rejections = generate_profiles(build_provider(gen_cfg), gen_cfg, int(n_gen))
                st.session_state["gen_rejections"] = rejections
                if profiles:
                    g = build_suite(profiles, source="generated", generator=gen_cfg.public_dict())
                    save_suite(g)
                    st.session_state["generated_suite"] = g.model_dump_json()
                    st.success(f"Accepted {len(profiles)} of {int(n_gen)} requested profiles.")
                else:
                    st.error("No valid profiles were generated. Nothing was fabricated; see rejections below.")
        if st.session_state.get("gen_rejections"):
            with st.expander(f"Rejected generator outputs ({len(st.session_state['gen_rejections'])})"):
                st.dataframe(pd.DataFrame(st.session_state["gen_rejections"]), width="stretch")
        if st.session_state.get("generated_suite"):
            suite = load_suite_json(st.session_state["generated_suite"])

    if suite is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Base profiles", len(suite.profiles))
        c2.metric("Gender variants", len(suite.variants))
        c3.metric("Matched cases", len(suite.cases))
        with st.expander("Preview profiles and a matched set"):
            st.dataframe(pd.DataFrame([{**p.model_dump(), "academic_strengths": "; ".join(p.academic_strengths),
                                        "interests": "; ".join(p.interests), "experience": "; ".join(p.experience)}
                                       for p in suite.profiles]), width="stretch", hide_index=True)
            first = [c for c in suite.cases if c.profile_id == suite.profiles[0].profile_id]
            cols = st.columns(len(first))
            for col, case in zip(cols, first):
                col.markdown(f"**{case.variant}**")
                col.code(case.user_prompt, language=None)
            st.caption("Only the `Gender:` line differs between matched prompts.")
        st.download_button("Download test suite (JSON)", suite_json(suite, known_secrets()),
                           file_name=f"{suite.suite_id}.json", mime="application/json")

    # ---- settings
    st.subheader("5 · Repetitions and seed")
    c1, c2 = st.columns(2)
    reps = c1.number_input("Repetitions per gender variant", 1, 10, config.DEFAULT_REPETITIONS, key="reps")
    seed = c2.number_input("Random seed", 0, 10_000_000, config.DEFAULT_SEED, key="seed",
                           help="Controls request order and which response is shown to the judge as A or B.")

    # ---- plan + run
    st.subheader("6 · Run")
    if suite is None:
        st.info("Choose or build a test suite to continue.")
        return
    calls = planned_calls(len(suite.profiles), len(suite.variants), int(reps))
    st.markdown(
        f"**Planned calls:** {calls['target_calls']} target calls "
        f"({len(suite.profiles)} profiles × {len(suite.variants)} variants × {int(reps)} repetitions) and "
        f"up to {calls['judge_calls']} judge calls ({len(suite.profiles)} × {calls['pairs_per_profile']} pairs × "
        f"{int(reps)}). Retries after timeouts, rate limits or unparseable judge output can increase usage "
        f"(up to {1 + target_cfg.max_retries}× per target call and {1 + judge_cfg.max_retries}× per judge call)."
    )
    ready = True
    if live:
        missing = [r for r, c in (("target", target_cfg), ("judge", judge_cfg)) if not c.base_url or not c.model]
        if missing:
            st.error(f"Missing base URL/model for: {', '.join(missing)}")
            ready = False
        ack = st.checkbox("I understand live requests may incur charges.", key="ack_cost")
        ready = ready and ack
    if st.button("▶ Run Scan", type="primary", disabled=not ready):
        save_suite(suite)
        manifest = create_run(STORE, suite, target_cfg, judge_cfg, repetitions=int(reps), seed=int(seed),
                              assistant_description=desc, target_system_prompt=sys_prompt)
        _execute(manifest.run_id, target_cfg, judge_cfg)

    # ---- resume
    incomplete = [m for m in STORE.list_runs() if m.status != "completed"]
    if incomplete:
        with st.expander(f"Resume an incomplete run ({len(incomplete)})"):
            st.caption("Completed results are kept; only missing or failed calls are retried. Live runs use the "
                       "API keys currently entered above (keys are never stored with runs).")
            m = st.selectbox("Run", incomplete, format_func=lambda m: f"{m.run_id} · {m.status} · "
                             f"{'simulated' if m.simulated else 'live'}", key="resume_pick")
            if st.button("Resume selected run"):
                tk = (st.session_state.get("target_api_key") or os.environ.get("BIASSCOPE_TARGET_API_KEY", ""))
                jk = (st.session_state.get("judge_api_key") or os.environ.get("BIASSCOPE_JUDGE_API_KEY", ""))
                if m.judge_same_as_target:
                    jk = tk
                _execute(m.run_id, config_from_public(m.target, tk), config_from_public(m.judge, jk))


def _execute(run_id: str, target_cfg: ProviderConfig, judge_cfg: ProviderConfig) -> None:
    bar = st.progress(0.0, text="Starting…")
    log = st.empty()

    def progress(phase, done, total, msg):
        label = "Target model" if phase == "target" else "Judge"
        bar.progress(min(done / max(total, 1), 1.0), text=f"{label}: {done}/{total}")
        log.caption(html.escape(msg))

    try:
        m = execute_run(STORE, run_id, target_cfg, judge_cfg, progress=progress)
    except Exception as exc:
        st.error(f"Run stopped: {type(exc).__name__}: {exc}. Completed results were saved; you can resume.")
        return
    st.session_state["current_run"] = run_id
    if m.status == "completed":
        st.success(f"Scan complete: {run_id}. Open **Results** or **Evidence Explorer** in the sidebar.")
    else:
        st.warning(f"Run {run_id} ended with status '{m.status}'. Completed results were saved.")


# ============================================================================ helpers for run pages
def pick_run(key: str):
    runs = STORE.list_runs()
    if not runs:
        st.info("No runs yet. Go to **Setup and Run** and click Run Scan (offline demo works without a key).")
        return None
    ids = [m.run_id for m in runs]
    current = st.session_state.get("current_run")
    idx = ids.index(current) if current in ids else 0
    m = st.selectbox("Run", runs, index=idx, key=key,
                     format_func=lambda m: f"{m.run_id} · {'SIMULATED' if m.simulated else 'LIVE'} · "
                                           f"target={m.target.get('model')} · {m.status}")
    st.session_state["current_run"] = m.run_id
    return m


@st.cache_data(show_spinner=False)
def _load_run(run_id: str, mtime: float):
    manifest = STORE.load_manifest(run_id)
    suite = STORE.load_suite(run_id)
    targets = STORE.load_targets(run_id)
    comps = STORE.load_comparisons(run_id)
    return manifest, suite, targets, comps


def load_run(run_id: str):
    d = STORE.run_dir(run_id)
    mtime = max((p.stat().st_mtime for p in d.iterdir()), default=0)
    return _load_run(run_id, mtime)


# ============================================================================ RESULTS
def page_results() -> None:
    st.title("Results")
    m = pick_run("results_run")
    if m is None:
        return
    manifest, suite, targets, comps = load_run(m.run_id)
    mode_banner(manifest.simulated)
    for note in manifest.notes:
        if "same model" in note:
            st.warning(note)
    s = summarize(manifest, suite, targets, comps)
    o = s["overview"]
    ts, cs = o["target_status"], o["comparison_status"]

    c = st.columns(6)
    c[0].metric("Base profiles", o["profiles"])
    c[1].metric("Responses OK", f"{ts['ok']} / {o['target_planned']}")
    c[2].metric("Comparisons judged", f"{cs['judged']} / {o['comparisons_planned']}")
    c[3].metric("Refusals", ts["refusal"])
    c[4].metric("Target errors / invalid", f"{ts['error']} / {ts['invalid_output']}")
    c[5].metric("Judge errors", cs["judge_error"])
    st.caption(
        f"Comparison unit = one profile × one repetition × one pair of variants. "
        f"{cs['not_judged_target_issue']} comparisons were not judged because a target response was missing, "
        f"refused or invalid; {cs['judge_error']} failed at the judge. None of these count as 'no issue found'. "
        f"Pending (never recorded): {o['comparisons_pending']}."
    )

    st.subheader("Findings by rubric criterion")
    st.caption("Denominator = comparisons the judge actually evaluated. 'Uncertain' is shown separately.")
    ct = s["criteria_table"]
    st.dataframe(ct[["criterion", "potential_bias_rate", "uncertain_rate", "no_issue_found", "not_evaluated",
                     "flags_downgraded_for_invalid_evidence"]].rename(columns={
                         "potential_bias_rate": "potential bias", "uncertain_rate": "uncertain",
                         "no_issue_found": "no issue found", "not_evaluated": "not evaluated",
                         "flags_downgraded_for_invalid_evidence": "flags downgraded (invalid evidence)"}),
                 hide_index=True, width="stretch")
    long = ct.melt(id_vars=["criterion"], value_vars=["potential_bias", "uncertain", "no_issue_found", "not_evaluated"],
                   var_name="verdict", value_name="comparisons")
    fig = px.bar(long, x="comparisons", y="criterion", color="verdict", orientation="h",
                 color_discrete_map=VERDICT_COLORS, height=300,
                 category_orders={"verdict": ["potential_bias", "uncertain", "no_issue_found", "not_evaluated"]})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="", yaxis_title="",
                      plot_bgcolor="white", font=dict(size=13))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By variant pair")
        st.dataframe(s["pair_table"], hide_index=True, width="stretch")
    with c2:
        st.subheader("Whose response was implicated")
        st.caption("Potential-bias findings where that variant's response held the evidence ÷ judged comparisons "
                   "involving that variant.")
        st.dataframe(s["affected_table"], hide_index=True, width="stretch")

    st.subheader("By profile")
    st.caption(f"Profiles with at least one potential-bias comparison: **{s['profiles_flagged']}**. "
               f"Distinct successful responses cited as evidence: **{s['responses_cited']}**. Repetitions of one "
               "profile are not independent samples, so we report counts rather than significance tests.")
    cdf = s["comparisons_df"]
    heat = (cdf[cdf["verdict"] == "potential_bias"].groupby(["profile_id", "criterion"]).size()
            .unstack(fill_value=0).reindex(index=[p.profile_id for p in suite.profiles], columns=list(CRITERIA),
                                           fill_value=0))
    heat.columns = [CRITERION_LABELS[c] for c in heat.columns]
    fig2 = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale=["#ffffff", "#c0533f"],
                     labels=dict(color="potential-bias comparisons"), height=60 + 28 * len(heat))
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig2, width="stretch")
    with st.expander("Profile table"):
        st.dataframe(s["profile_table"], hide_index=True, width="stretch")
    with st.expander("Target outcomes by variant (refusals, invalid outputs, errors)"):
        st.dataframe(s["variant_status_table"], hide_index=True, width="stretch")
        errs = s["targets_df"][s["targets_df"]["status"] != "ok"]
        if len(errs):
            st.dataframe(errs[["record_id", "status", "status_detail", "error_message", "attempts"]],
                         hide_index=True, width="stretch")

    st.info("There is deliberately no overall 'fairness score'. Read the evidence before drawing conclusions; "
            "a flagged comparison is a lead to investigate, not a verdict.")

    st.subheader("Downloads")
    secrets_ = known_secrets()
    d = st.columns(4)
    d[0].download_button("Test suite (JSON)", suite_json(suite, secrets_), f"{suite.suite_id}.json", "application/json")
    d[1].download_button("Results (JSON)", results_json(manifest, suite, targets, comps, s, secrets_),
                         f"{manifest.run_id}-results.json", "application/json")
    d[2].download_button("Results (CSV)", results_csv(manifest, cdf, secrets_), f"{manifest.run_id}-results.csv",
                         "text/csv")
    d[3].download_button("Report (Markdown)", markdown_report(manifest, suite, comps, s, secrets_),
                         f"{manifest.run_id}-report.md", "text/markdown")

    with st.expander("Compare with another run (same saved suite required)"):
        others = [r for r in STORE.list_runs() if r.run_id != manifest.run_id]
        if not others:
            st.caption("Run another scan on the same suite (e.g. a different target) to compare.")
        else:
            other = st.selectbox("Other run", others, format_func=lambda r: f"{r.run_id} · target={r.target.get('model')}",
                                 key="cmp_other")
            cmp = compare_runs(manifest, other)
            if not cmp["compatible"]:
                st.error("Not comparable: " + " | ".join(cmp["blocking"]))
            else:
                for w in cmp["warnings"]:
                    st.warning(w)
                st.markdown("**Configuration differences**")
                st.dataframe(pd.DataFrame(cmp["differences"]) if cmp["differences"] else pd.DataFrame(
                    [{"setting": "none", "note": "identical configuration"}]), hide_index=True, width="stretch")
                m2, s2, t2, c2_ = load_run(other.run_id)
                s_other = summarize(m2, s2, t2, c2_)
                side = ct[["criterion", "potential_bias_rate", "uncertain_rate"]].merge(
                    s_other["criteria_table"][["criterion", "potential_bias_rate", "uncertain_rate"]],
                    on="criterion", suffixes=(f" · {manifest.target.get('model')}", f" · {m2.target.get('model')}"))
                st.dataframe(side, hide_index=True, width="stretch")


# ============================================================================ EVIDENCE EXPLORER
def page_evidence() -> None:
    st.title("Evidence Explorer")
    m = pick_run("evidence_run")
    if m is None:
        return
    manifest, suite, targets, comps = load_run(m.run_id)
    mode_banner(manifest.simulated)
    cdf = summarize(manifest, suite, targets, comps)["comparisons_df"]
    if cdf.empty:
        st.info("No comparisons recorded yet.")
        return

    f1, f2, f3, f4 = st.columns(4)
    prof = f1.multiselect("Profile", sorted(cdf["profile_id"].unique()), key="ev_prof")
    pair = f2.multiselect("Gender comparison", list(dict.fromkeys(cdf["pair"])), key="ev_pair")
    crit = f3.multiselect("Criterion", list(CRITERIA), format_func=lambda c: CRITERION_LABELS[c], key="ev_crit")
    verd = f4.multiselect("Verdict", ["potential_bias", "uncertain", "no_issue_found", "not_evaluated"],
                          default=["potential_bias", "uncertain"], key="ev_verd")
    view = cdf.copy()
    if prof:
        view = view[view["profile_id"].isin(prof)]
    if pair:
        view = view[view["pair"].isin(pair)]
    if crit:
        view = view[view["criterion"].isin(crit)]
    if verd:
        view = view[view["verdict"].isin(verd)]
    st.caption(f"{len(view)} matching criterion judgments (of {len(cdf)}).")
    st.dataframe(view[["comparison_id", "criterion", "verdict", "affected_variants", "evidence_valid",
                       "comparison_status"]], hide_index=True, width="stretch", height=220)
    if view.empty:
        return

    options = list(dict.fromkeys(zip(view["comparison_id"], view["criterion"])))
    sel = st.selectbox("Inspect", options, format_func=lambda o: f"{o[0]} · {CRITERION_LABELS[o[1]]}", key="ev_sel")
    comp = comps[sel[0]]
    j = next((x for x in comp.criteria if x.criterion == sel[1]), None)

    st.markdown("---")
    st.markdown(f"#### {html.escape(CRITERION_LABELS[sel[1]])}")
    st.caption(CRITERIA[sel[1]])
    if j is None:
        st.markdown(chip("chip-err", "not evaluated") + html.escape(comp.status_detail or comp.status),
                    unsafe_allow_html=True)
    else:
        cls, txt = VERDICT_CHIP[j.verdict]
        extra = ""
        if j.original_verdict and j.original_verdict != j.verdict:
            extra = chip("chip-err", f"judge said {j.original_verdict}; downgraded")
        sim = chip("chip-sim", "SIMULATED") if comp.simulated else ""
        st.markdown(chip(cls, txt) + extra + sim, unsafe_allow_html=True)
        st.markdown("**Judge's explanation** "
                    "<span class='muted'>(model-written; not a calibrated probability)</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='resp'>{html.escape(j.explanation or '(none)')}</div>", unsafe_allow_html=True)
        if j.validation_notes:
            st.caption("Validation notes: " + "; ".join(j.validation_notes))

    st.markdown(f"**Matched responses** · profile {comp.profile_id} · repetition {comp.repetition} · "
                f"{comp.pair_label()}")
    cols = st.columns(2)
    for col, variant in zip(cols, comp.pair):
        label = "A" if comp.a_variant == variant else "B"
        rid = comp.a_record_id if label == "A" else comp.b_record_id
        rec = targets.get(rid)
        quotes = [e.quote for e in (j.evidence if j else []) if e.response_label == label and e.found]
        with col:
            st.markdown(f"**Gender: {variant}** <span class='muted'>(shown to judge as {label})</span>",
                        unsafe_allow_html=True)
            if rec is None:
                st.warning("No target record.")
                continue
            if rec.status != "ok":
                st.markdown(chip("chip-err", rec.status) + html.escape(rec.status_detail or rec.error_message or ""),
                            unsafe_allow_html=True)
            st.markdown(f"<div class='resp'>{highlight_html(rec.response_text, quotes)}</div>", unsafe_allow_html=True)
            st.caption(f"model: {rec.model} · latency {rec.latency_ms} ms · attempts {rec.attempts}"
                       + (f" · tokens {rec.usage}" if rec.usage else ""))
            bad = [e.quote for e in (j.evidence if j else []) if e.response_label == label and not e.found]
            for q in bad:
                st.markdown(chip("chip-err", "quote NOT found in this response") + html.escape(q),
                            unsafe_allow_html=True)
            with st.expander("Exact prompt sent"):
                for msg in rec.messages:
                    st.caption(msg["role"])
                    st.code(msg["content"], language=None)
    with st.expander("Raw judge output"):
        st.code(comp.judge_raw_output or "(none)", language="json")


# ============================================================================ AUDITOR VALIDATION
def page_validation() -> None:
    st.title("Auditor Validation")
    st.markdown(
        "Before trusting the judge, check it on a small set of **fixtures**: hand-written response pairs whose "
        "expected outcome we already decided. They include clearly stereotyped answers, fair answers with harmless "
        "wording differences, ambiguous cases, answers that try to give the judge instructions, refusals, and "
        "malformed outputs."
    )
    st.warning("Reference labels are **developer-authored**, not independent ground truth. Passing these fixtures "
               "shows the auditor handles a few specific behaviours; it does **not** validate the auditor in general. "
               "These fixtures are kept separate from any prompt-tuning examples.")
    fixtures = load_fixtures()

    c1, c2 = st.columns(2)
    auditor = c1.radio("Auditor to validate", ["Keyword baseline (offline, non-semantic)",
                                              "Live judge (configured on Setup page)"], key="val_auditor")
    label_set = c2.radio("Reference labels", ["Developer-authored", "Imported independent labels"], key="val_labels")
    labels = developer_labels(fixtures)
    if label_set.startswith("Imported"):
        up = st.file_uploader("Labels file (CSV or JSON with fixture_id, criterion, label, reviewer)",
                              type=["csv", "json"], key="labels_upload")
        if up is None:
            st.info("Upload independently reviewed labels. Allowed labels: potential_bias, no_issue_found, indeterminate.")
            st.download_button("Download a blank labels template (CSV)",
                               "fixture_id,criterion,label,reviewer\n" + "".join(
                                   f"{f['fixture_id']},{c},,\n" for f in fixtures["fixtures"]
                                   if f.get("reference_labels") for c in CRITERIA),
                               "labels_template.csv", "text/csv")
            return
        labels, problems = import_labels(up.getvalue().decode("utf-8"), up.name, fixtures)
        if problems:
            st.warning("Skipped rows: " + "; ".join(problems[:10]))
        st.success(f"Using {len(labels)} imported labels.")
    order_check = st.checkbox("Run order-sensitivity check (evaluate again with A and B swapped)", True, key="val_order")

    if auditor.startswith("Keyword"):
        cfg = baseline_config()
        st.caption("The keyword baseline matches hand-written phrase lists. It does not understand meaning; it is a "
                   "weak reference point and lets the metrics run offline.")
    else:
        if st.session_state.get("judge_same"):
            st.warning("Using the target settings as judge (same-model limitation applies).")
        try:
            prefix = "target" if st.session_state.get("judge_same") else "judge"
            from pydantic import SecretStr

            key = st.session_state.get(f"{prefix}_api_key") or os.environ.get(f"BIASSCOPE_{prefix.upper()}_API_KEY", "")
            cfg = ProviderConfig(role="judge", kind="openai_compatible",
                                 base_url=st.session_state.get(f"{prefix}_base_url", ""),
                                 model=st.session_state.get(f"{prefix}_model", ""),
                                 api_key=SecretStr(key) if key else None,
                                 temperature=st.session_state.get(f"{prefix}_temp", 0.0)
                                 if st.session_state.get(f"{prefix}_send_temp", True) else None,
                                 max_tokens=int(st.session_state.get(f"{prefix}_max_tokens", 700)),
                                 token_param=st.session_state.get(f"{prefix}_token_param", "max_tokens"),
                                 timeout_s=float(st.session_state.get(f"{prefix}_timeout", 90)),
                                 max_retries=int(st.session_state.get(f"{prefix}_retries", 3)))
        except Exception as exc:
            st.error(f"Judge configuration invalid: {exc}")
            return
        if not cfg.base_url or not cfg.model:
            st.error("Configure the judge in Setup and Run (Live mode) first.")
            return
        judged = sum(1 for f in fixtures["fixtures"] if f.get("reference_labels"))
        st.caption(f"This makes about {judged * (2 if order_check else 1)} live judge calls (more with retries). "
                   "Hosted APIs may charge for these.")

    if st.button("Run validation", type="primary"):
        bar = st.progress(0.0)
        res = run_validation(build_provider(cfg), cfg, fixtures, order_check=order_check,
                             progress=lambda n, t: bar.progress(n / t, text=f"Fixture {n}/{t}"))
        st.session_state["val_result"] = res
    res = st.session_state.get("val_result")
    if not res:
        with st.expander("Browse fixtures"):
            st.dataframe(pd.DataFrame(fixtures["fixtures"]).drop(columns=["reference_labels"]), hide_index=True,
                         width="stretch")
        return

    mt = compute_metrics(res, labels)
    st.markdown(f"Auditor: **{html.escape(res['auditor']['model'])}** ({res['auditor']['kind']})")
    met = mt["metrics"].copy()
    for col in ("precision", "recall"):
        met[col] = met[col].map(lambda v: "n/a (0 denominator)" if v is None or pd.isna(v) else f"{v:.2f}")
    st.subheader("Determinate labels: confusion counts")
    st.caption("Positive = potential_bias. Only fixtures with a determinate reference label and a determinate "
               "auditor verdict enter TP/FP/FN/TN. Uncertain verdicts and judge errors are counted separately.")
    st.dataframe(met[["criterion", "TP", "FP", "FN", "TN", "precision", "recall", "uncertain_on_positive",
                      "uncertain_on_negative", "judge_error", "indeterminate_reference"]],
                 hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Order sensitivity")
        k, n = mt["order_changed"], mt["order_compared"]
        st.metric("Verdicts that changed when A/B swapped", f"{k} / {n}" if n else "n/a")
        st.caption("A change means the verdict (or which variant was implicated) depended on presentation order.")
        if len(mt["order_sensitivity"]):
            ch = mt["order_sensitivity"][mt["order_sensitivity"]["changed"] == True]  # noqa: E712
            if len(ch):
                st.dataframe(ch, hide_index=True, width="stretch")
    with c2:
        st.subheader("Refusals and malformed outputs")
        st.dataframe(mt["target_classification"], hide_index=True, width="stretch")
        st.caption("Refused or malformed responses must be classified as such and never judged or counted as passes.")

    st.subheader("Per-fixture results")
    st.dataframe(mt["details"], hide_index=True, width="stretch")


# ============================================================================ router
{"Setup and Run": page_setup, "Results": page_results, "Evidence Explorer": page_evidence,
 "Auditor Validation": page_validation}[page]()
