import streamlit as st

import ui_common as ui

hero_l, hero_r = st.columns([1.05, 1], vertical_alignment="center", gap="large")
with hero_l:
    st.markdown(ui.logo_mark_svg(150, underline=True), unsafe_allow_html=True)
    st.markdown(ui.wordmark_html(76), unsafe_allow_html=True)
    st.markdown(
        '<p class="bs-lede" style="margin-top:18px">Two students. Same grades, same interests, same goals. '
        'The only difference is the gender they state. <b>Does an AI career advisor treat them the same?</b> '
        'BiasScope tests that automatically and shows you the evidence.</p>',
        unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.page_link("views/demo.py", label="Try the demo (no key)", icon=":material/play_circle:", width="stretch")
    b2.page_link("views/live_proof.py", label="See the live proof", icon=":material/verified:", width="stretch")
with hero_r:
    st.markdown(
        """
<div class="bs-card" style="padding:26px">
  <div class="bs-kicker">The idea in one picture</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px">
    <div class="resp" style="font-size:.86rem"><b>Gender: woman</b><br>GPA 3.9 · robotics team lead · wants to study engineering</div>
    <div class="resp" style="font-size:.86rem"><b>Gender: man</b><br>GPA 3.9 · robotics team lead · wants to study engineering</div>
    <div class="resp" style="font-size:.86rem;border-color:#EFC2BB">“…you might be happier in a <mark>supportive role</mark>…”</div>
    <div class="resp" style="font-size:.86rem">“…apply to a competitive research program…”</div>
  </div>
  <p class="muted" style="margin-top:12px">Identical profiles, one line changed. If the advice changes in ways that
  have nothing to do with the student's qualifications, BiasScope flags it and quotes the exact sentence.</p>
</div>
""",
        unsafe_allow_html=True)

st.write("")
st.markdown('<div class="bs-kicker">Where do you want to start?</div>', unsafe_allow_html=True)
cards = [
    ("views/demo.py", ":material/play_circle:", "Try the demo",
     "Free and instant. Watch a full scan on a pretend advisor, then explore the results. No account or key needed."),
    ("views/live_proof.py", ":material/verified:", "Live proof",
     "Give a real AI judge 8 answer pairs (4 biased, 4 fair) and see if it can tell them apart. About 1 cent."),
    ("views/scan.py", ":material/tune:", "Real scan",
     "Test a real AI model (like ChatGPT or Claude) with your own API key. About 180 requests."),
    ("views/about.py", ":material/menu_book:", "How it works",
     "The experiment in plain words: what's tested, how it's judged, and what the results can and can't tell you."),
]
cols = st.columns(4, gap="medium")
for col, (page, icon, title, text) in zip(cols, cards):
    with col:
        with st.container(border=True, height="stretch"):
            st.markdown(f"#### {title}")
            st.markdown(f'<p class="muted" style="min-height:96px">{text}</p>', unsafe_allow_html=True)
            st.page_link(page, label="Open", icon=icon)

st.write("")
st.markdown('<div class="bs-kicker">How a scan works</div>', unsafe_allow_html=True)
steps = [
    ("Make twins", "Fictional student profiles with no names or pronouns are copied three times: woman, man, nonbinary."),
    ("Ask the AI", "Each copy goes to the AI advisor in a brand-new chat, a few times each."),
    ("Judge in pairs", "A second AI compares two answers and must quote the exact sentence behind any flag."),
    ("Show the proof", "Counts, charts and side-by-side evidence. Errors and unsure verdicts are never hidden."),
]
s_cols = st.columns(4, gap="medium")
for i, (col, (t, d)) in enumerate(zip(s_cols, steps), start=1):
    col.markdown(f'<span class="bs-step">{i}</span> <b style="margin-left:6px">{t}</b>'
                 f'<p class="muted" style="margin-top:8px">{d}</p>', unsafe_allow_html=True)

ui.footer()
