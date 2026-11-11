import html

import pandas as pd
import plotly.express as px
import streamlit as st

import ui_common as ui
from biasscope.analysis import compare_runs, summarize, sunburst_frame
from biasscope.reporting import markdown_report, results_csv, results_json, suite_json
from biasscope.schemas import CRITERIA, CRITERION_LABELS

ui.page_header("Results", "What the judge found, with every number shown as <b>count out of total</b>.",
               kicker="Explore a scan")

m = ui.pick_run("results_run")
if m is None:
    ui.no_runs_message()
    st.stop()

manifest, suite, targets, comps = ui.load_run(m.run_id)
ui.mode_banner(manifest.simulated)
for note in manifest.notes:
    if "same model" in note:
        st.warning(note)
s = summarize(manifest, suite, targets, comps)
o = s["overview"]
ts, cs = o["target_status"], o["comparison_status"]

c = st.columns(5)
c[0].metric("Student profiles", o["profiles"])
c[1].metric("Answers received", f"{ts['ok']} / {o['target_planned']}",
            help="Answers the AI gave successfully, out of questions planned.")
c[2].metric("Pairs judged", f"{cs['judged']} / {o['comparisons_planned']}",
            help="A pair = two answers to the same profile with different genders.")
c[3].metric("Refusals", ts["refusal"], help="The AI declined to answer. Never counted as a pass.")
c[4].metric("Errors", ts["error"] + ts["invalid_output"] + cs["judge_error"],
            help="Failed requests, broken answers or broken judge replies. Never counted as a pass.")
if cs["not_judged_target_issue"] or cs["judge_error"]:
    st.caption(f"{cs['not_judged_target_issue'] + cs['judge_error']} pairs couldn't be judged (a missing answer, "
               "a refusal or a judge error). They are left out of the totals below, not counted as fair.")

tab_explore, tab_types, tab_profiles, tab_more = st.tabs(
    ["Explore by gender", "By bias type", "By profile", "Downloads & compare"])

# ---------------------------------------------------------------- sunburst
with tab_explore:
    st.markdown("**Click a ring to zoom in.** The center splits the flags by *whose answer* was flagged. Click a "
                "gender to see which kinds of bias, then click again to see which student profiles. Click the "
                "center to zoom back out.")
    show = st.radio("Show", ["potential_bias", "uncertain"], horizontal=True,
                    format_func=lambda v: ui.VERDICT_NAMES[v], key="sb_verdict")
    df = sunburst_frame(comps.values(), show)
    if df.empty:
        st.success(f"No “{ui.VERDICT_NAMES[show].lower()}” verdicts in this scan, so there's nothing to drill into.")
    else:
        df = df.assign(root="All flags")
        colors = {"woman": "#3B3FD8", "man": "#E8902A", "nonbinary": "#2A9D8F", "Not attributed": "#9AA0B5",
                  "(?)": "#EDE8DC"}
        fig = px.sunburst(df, path=["root", "answer_of", "bias_type", "profile"], values="count",
                          color="answer_of", color_discrete_map=colors, maxdepth=3, height=560)
        fig.update_traces(insidetextorientation="radial", sort=False,
                          hovertemplate="<b>%{label}</b><br>%{value} flags<br>%{percentParent:.0%} of the ring "
                                        "it belongs to<extra></extra>",
                          marker=dict(line=dict(color="#FBF8F1", width=2)))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Space Grotesk, sans-serif", size=14))
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        with right:
            st.markdown("**Fair comparison by gender**")
            st.caption("Flags on that gender's answers ÷ judged pairs that included that gender.")
            st.dataframe(s["affected_table"].rename(columns={
                "variant": "gender", "judged_comparisons_involving_variant": "judged pairs"}),
                hide_index=True, width="stretch")
            st.markdown("**Read the flagged sentences**")
            g = st.selectbox("Gender", sorted(df["answer_of"].unique()), key="sb_g")
            b = st.selectbox("Bias type", sorted(df[df["answer_of"] == g]["bias_type"].unique()), key="sb_b")
            crit = next(k for k, v in CRITERION_LABELS.items() if v == b)
            shown = 0
            for comp in sorted(comps.values(), key=lambda x: x.comparison_id):
                for j in comp.criteria:
                    if j.criterion != crit or j.verdict != show:
                        continue
                    if g != "Not attributed" and g not in j.affected_variants:
                        continue
                    for e in j.evidence:
                        if e.found and (e.variant == g or g == "Not attributed") and shown < 4:
                            shown += 1
                            st.markdown(f"<div class='resp' style='margin-bottom:8px'>“{html.escape(e.quote)}”"
                                        f"<div class='muted'>{comp.profile_id} · {comp.pair_label()} · "
                                        f"try {comp.repetition + 1}</div></div>", unsafe_allow_html=True)
            if shown == 0:
                st.caption("No quoted sentences for this combination.")
            st.page_link("views/evidence.py", label="See full answers side by side", icon=":material/compare:")

# ---------------------------------------------------------------- criteria
with tab_types:
    ct = s["criteria_table"]
    st.dataframe(ct[["criterion", "potential_bias_rate", "uncertain_rate", "no_issue_found", "not_evaluated",
                     "flags_downgraded_for_invalid_evidence"]].rename(columns={
                         "criterion": "bias type", "potential_bias_rate": "potential bias",
                         "uncertain_rate": "uncertain", "no_issue_found": "no issue found",
                         "not_evaluated": "couldn't judge",
                         "flags_downgraded_for_invalid_evidence": "flags thrown out (fake quote)"}),
                 hide_index=True, width="stretch")
    long = ct.melt(id_vars=["criterion"], value_vars=["potential_bias", "uncertain", "no_issue_found", "not_evaluated"],
                   var_name="verdict", value_name="pairs")
    long["verdict"] = long["verdict"].map(ui.VERDICT_NAMES)
    fig = px.bar(long, x="pairs", y="criterion", color="verdict", orientation="h", height=320,
                 color_discrete_map={ui.VERDICT_NAMES[k]: v for k, v in ui.VERDICT_COLORS.items()},
                 category_orders={"verdict": [ui.VERDICT_NAMES[k] for k in ui.VERDICT_COLORS]})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="", yaxis_title="",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Space Grotesk, sans-serif", size=13))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.markdown("**By gender pair**")
    st.dataframe(s["pair_table"], hide_index=True, width="stretch")

# ---------------------------------------------------------------- profiles
with tab_profiles:
    st.caption(f"Profiles with at least one flag: **{s['profiles_flagged']}**. Repeats of the same profile aren't "
               "independent, so BiasScope shows counts instead of claiming statistical significance.")
    cdf = s["comparisons_df"]
    heat = (cdf[cdf["verdict"] == "potential_bias"].groupby(["profile_id", "criterion"]).size()
            .unstack(fill_value=0).reindex(index=[p.profile_id for p in suite.profiles], columns=list(CRITERIA),
                                           fill_value=0))
    heat.columns = [CRITERION_LABELS[c] for c in heat.columns]
    fig2 = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale=["#FFFFFF", "#C0533F"],
                     height=80 + 30 * len(heat))
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                       paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Space Grotesk, sans-serif"))
    st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})
    st.dataframe(s["profile_table"], hide_index=True, width="stretch")
    with st.expander("Refusals and errors by gender"):
        st.dataframe(s["variant_status_table"], hide_index=True, width="stretch")
        errs = s["targets_df"][s["targets_df"]["status"] != "ok"]
        if len(errs):
            st.dataframe(errs[["record_id", "status", "status_detail", "error_message", "attempts"]],
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- downloads & compare
with tab_more:
    st.markdown("**Downloads**")
    sec = ui.known_secrets()
    d = st.columns(4)
    d[0].download_button("Report (Markdown)", markdown_report(manifest, suite, comps, s, sec),
                         f"{manifest.run_id}-report.md", "text/markdown", width="stretch")
    d[1].download_button("Results (CSV)", results_csv(manifest, s["comparisons_df"], sec),
                         f"{manifest.run_id}-results.csv", "text/csv", width="stretch")
    d[2].download_button("Results (JSON)", results_json(manifest, suite, targets, comps, s, sec),
                         f"{manifest.run_id}-results.json", "application/json", width="stretch")
    d[3].download_button("Test file (JSON)", suite_json(suite, sec), f"{suite.suite_id}.json",
                         "application/json", width="stretch")

    st.markdown("**Compare with another scan**")
    others = [r for r in ui.store().list_runs() if r.run_id != manifest.run_id]
    if not others:
        st.caption("Run another scan on the same test (for example with a different AI) to compare them here.")
    else:
        other = st.selectbox("Other scan", others, key="cmp_other",
                             format_func=lambda r: f"{'Simulated' if r.simulated else 'Live'} · "
                                                   f"{r.target.get('model')} · {r.created_at[:16].replace('T', ' ')}")
        cmp = compare_runs(manifest, other)
        if not cmp["compatible"]:
            st.error("These two scans can't be compared fairly: " + " | ".join(cmp["blocking"]))
        else:
            for w in cmp["warnings"]:
                st.warning(w)
            m2, s2_, t2, c2_ = ui.load_run(other.run_id)
            so = summarize(m2, s2_, t2, c2_)
            side = s["criteria_table"][["criterion", "potential_bias_rate"]].merge(
                so["criteria_table"][["criterion", "potential_bias_rate"]], on="criterion",
                suffixes=(f" · {manifest.target.get('model')}", f" · {m2.target.get('model')}"))
            st.dataframe(side, hide_index=True, width="stretch")
            if cmp["differences"]:
                with st.expander("Settings that differ"):
                    st.dataframe(pd.DataFrame(cmp["differences"]), hide_index=True, width="stretch")

ui.footer()
