"""BiasScope website entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="BiasScope", page_icon=str(ROOT / "assets" / "favicon.png"), layout="wide",
                   initial_sidebar_state="collapsed")

import ui_common  # noqa: E402  (after set_page_config)

ui_common.apply_style()

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/demo.py", title="Try the demo", icon=":material/play_circle:"),
    st.Page("views/live_proof.py", title="Live proof", icon=":material/verified:"),
    st.Page("views/scan.py", title="Run a scan", icon=":material/tune:"),
    st.Page("views/results.py", title="Results", icon=":material/donut_large:"),
    st.Page("views/evidence.py", title="Evidence", icon=":material/compare:"),
    st.Page("views/judge_check.py", title="Check the judge", icon=":material/fact_check:"),
    st.Page("views/about.py", title="How it works", icon=":material/menu_book:"),
]
st.navigation(pages, position="top").run()
