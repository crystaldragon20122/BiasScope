"""Shared look, logo and helpers for every BiasScope page."""

from __future__ import annotations

import html
import os

import streamlit as st
from pydantic import SecretStr

from biasscope import config
from biasscope.pipeline import execute_run
from biasscope.schemas import ProviderConfig
from biasscope.storage import RunStore

INK = "#1C1B3A"
ACCENT = "#E8902A"
BRAND = "#3B3FD8"

# --------------------------------------------------------------------------- logo
_EYE_PATHS = """
<g filter="url(#bs-rough)" fill="none" stroke-linecap="round" stroke-linejoin="round">
<path d="M14 92 C52 38 160 34 206 90 C160 146 58 148 18 96 C40 92 60 96 72 92 C70 68 92 52 112 54 C136 56 150 76 146 98 C142 122 118 134 98 128 C78 122 70 104 74 90" stroke="{ink}" stroke-width="4.5"/>
<path d="M18 88 C58 44 156 42 200 88" stroke="{ink}" stroke-width="1.6" opacity="0.55"/>
<path d="M60 44 L52 26 M90 34 L88 14 M122 33 L126 13 M154 42 L164 24" stroke="{ink}" stroke-width="3.5"/>
<path d="M92 84 C102 82 116 86 128 83" stroke="{accent}" stroke-width="6"/>
<path d="M92 101 C104 99 116 103 128 100" stroke="{accent}" stroke-width="6"/>
{underline}
</g>"""
_FILTER = """<defs><filter id="bs-rough" x="-10%" y="-10%" width="120%" height="120%">
<feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="9"/>
<feDisplacementMap in="SourceGraphic" scale="3"/></filter></defs>"""


def logo_mark_svg(width: int = 120, ink: str = INK, accent: str = ACCENT, underline: bool = False) -> str:
    """The BiasScope mark: a one-line hand-drawn eye whose pupil is an equals sign."""
    ul = f'<path d="M28 160 C80 150 140 166 196 154" stroke="{ink}" stroke-width="3"/>' if underline else ""
    h = int(width * (180 if underline else 150) / 220)
    vb = "0 0 220 180" if underline else "0 0 220 150"
    return (f'<svg width="{width}" height="{h}" viewBox="{vb}" role="img" '
            f'aria-label="BiasScope logo: a hand-drawn eye whose pupil is an equals sign">'
            + _FILTER + _EYE_PATHS.format(ink=ink, accent=accent, underline=ul) + "</svg>")


def wordmark_html(size: int = 64, color: str = INK) -> str:
    return (f'<span class="bs-wordmark" style="font-size:{size}px;color:{color}">'
            f'<b>Bias</b><span>Scope</span></span>')


# --------------------------------------------------------------------------- style
CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  .stApp button, .stApp input, .stApp textarea, .stApp a, .resp, .bs-card, .muted, .bs-lede {
    font-family: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  }
  [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
  .stApp { background: #FBF8F1; }
  header[data-testid="stHeader"] { background: rgba(251,248,241,.94); backdrop-filter: blur(6px);
                                   border-bottom: 1px solid #E7E2D6; }
  .block-container { max-width: 1180px; padding-top: 5rem; padding-bottom: 4rem; }
  [data-testid="stPageLink"] a { border: 1px solid #D9D3C4; border-radius: 12px; background: #FFFFFF;
                                 padding: 8px 14px; }
  [data-testid="stPageLink"] a:hover { border-color: #3B3FD8; background: #F3F2FF; }
  h1 { letter-spacing: -0.03em; font-weight: 700; color: #1C1B3A; }
  h2, h3 { letter-spacing: -0.02em; color: #1C1B3A; }
  .bs-wordmark { letter-spacing: -0.04em; line-height: 1; white-space: nowrap; }
  .bs-wordmark b { font-weight: 700; } .bs-wordmark span { font-weight: 400; }
  .bs-lede { font-size: 1.12rem; color: #3D4160; line-height: 1.55; max-width: 720px; }
  .bs-kicker { font-size: .78rem; letter-spacing: .2em; text-transform: uppercase; color: #6A6F8C; font-weight: 600; }
  .bs-card { background: #FFFFFF; border: 1px solid #E7E2D6; border-radius: 16px; padding: 22px 22px 16px;
             height: 100%; box-shadow: 0 1px 0 rgba(28,27,58,.04); }
  .bs-card h4 { margin: 6px 0 6px; font-size: 1.15rem; color: #1C1B3A; }
  .bs-card p { color: #4A4F6A; font-size: .95rem; line-height: 1.5; margin: 0 0 6px; }
  .bs-step { display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px;
             border-radius:50%; background:#1C1B3A; color:#fff; font-weight:700; font-size:.9rem; }
  .bs-hint { color:#5B6080; font-size:.9rem; margin-top:-4px; }
  .chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
          border:1px solid transparent; margin-right:6px; }
  .chip-sim { background:#FFF1DC; color:#8A5300; border-color:#F0D29A; }
  .chip-live { background:#E6ECFA; color:#23408A; border-color:#B9CBEA; }
  .chip-bias { background:#FBE6E2; color:#9C2B1C; border-color:#EFC2BB; }
  .chip-ok { background:#E5F2EA; color:#1E6A3A; border-color:#BCDCC7; }
  .chip-unc { background:#EFEBFA; color:#4F3A8C; border-color:#D3CAEE; }
  .chip-err { background:#EEEEEE; color:#444; border-color:#CCC; }
  .resp { background:#FFFFFF; border:1px solid #E7E2D6; border-radius:12px; padding:14px 16px;
          font-size:.93rem; line-height:1.6; }
  .resp mark { background:#FFD98A; padding:0 2px; border-radius:3px; }
  .banner { padding:10px 14px; border-radius:12px; margin:4px 0 14px; font-size:.92rem; }
  .banner-sim { background:#FFF1DC; border:1px solid #F0D29A; color:#6B4100; }
  .banner-live { background:#E6ECFA; border:1px solid #B9CBEA; color:#23408A; }
  .muted { color:#5B6080; font-size:.88rem; }
  .bs-footer { margin-top:48px; padding-top:16px; border-top:1px solid #E7E2D6; color:#6A6F8C; font-size:.82rem; }
  div[data-testid="stMetricValue"] { font-weight:700; color:#1C1B3A; }
  .stButton > button[kind="primary"] { background:#1C1B3A; border-color:#1C1B3A; }
  .stButton > button[kind="primary"]:hover { background:#3B3FD8; border-color:#3B3FD8; }
</style>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, lede: str = "", kicker: str = "") -> None:
    if kicker:
        st.markdown(f'<div class="bs-kicker">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f'<p class="bs-lede">{lede}</p>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="bs-footer">BiasScope · a TKS student project. Workflow inspired by Giskard\'s '
        'generate → test → evaluate → report scan; independent implementation, not affiliated with Giskard. '
        'Findings are <i>potential</i> bias under a stated rubric, never a fairness certificate.</div>',
        unsafe_allow_html=True)


def chip(cls: str, text: str) -> str:
    return f'<span class="chip {cls}">{html.escape(text)}</span>'


VERDICT_CHIP = {
    "potential_bias": ("chip-bias", "potential bias"),
    "no_issue_found": ("chip-ok", "no issue found"),
    "uncertain": ("chip-unc", "uncertain"),
    "not_evaluated": ("chip-err", "not evaluated"),
}
VERDICT_COLORS = {"potential_bias": "#C0533F", "uncertain": "#7A63B8", "no_issue_found": "#5F9B74",
                  "not_evaluated": "#B5B5B5"}
VERDICT_NAMES = {"potential_bias": "Potential bias", "uncertain": "Uncertain",
                 "no_issue_found": "No issue found", "not_evaluated": "Not evaluated"}


def mode_banner(simulated: bool) -> None:
    if simulated:
        st.markdown('<div class="banner banner-sim"><b>Simulated run.</b> The answers came from a pretend advisor '
                    'and the judge used pre-set labels, so these numbers show how BiasScope works – they don\'t '
                    'describe any real AI.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner banner-live"><b>Live run.</b> Real answers from the AI you tested, graded '
                    'by a real AI judge. Flags are <i>potential</i> bias to read and check, not proof.</div>',
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- storage
def store() -> RunStore:
    config.ensure_workspace()
    return RunStore()


def known_secrets() -> list[str]:
    return [v for k, v in st.session_state.items() if k.endswith("_api_key") and isinstance(v, str) and v]


# --------------------------------------------------------------------------- providers
PRESETS = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "send_temp": True,
               "max_tokens": 700, "token_param": "max_tokens",
               "help": "Key from platform.openai.com (starts with sk-). Billed per use."},
    "Claude (Anthropic)": {"base_url": "https://api.anthropic.com/v1", "model": "claude-haiku-5-5",
                           "send_temp": False, "max_tokens": 2000, "token_param": "max_tokens",
                           "help": "Key from platform.claude.com (starts with sk-ant-). Use a workspace key. "
                                   "Temperature is left off because newer Claude models reject it."},
    "Local model (Ollama)": {"base_url": "http://localhost:11434/v1", "model": "llama3.1:8b", "send_temp": True,
                             "max_tokens": 700, "token_param": "max_tokens",
                             "help": "Free, runs on your own computer. Only works when the app runs locally too."},
    "Other (OpenAI-compatible)": {"base_url": "", "model": "", "send_temp": True, "max_tokens": 700,
                                  "token_param": "max_tokens", "help": "Any server with a /chat/completions endpoint."},
}
JUDGE_DEFAULT_MODEL = {"OpenAI": "gpt-4.1-mini", "Claude (Anthropic)": "claude-haiku-5-5"}


def _apply_preset(role: str) -> None:
    p = PRESETS[st.session_state[f"{role}_preset"]]
    st.session_state[f"{role}_base_url"] = p["base_url"]
    model = p["model"]
    if role == "judge":
        model = JUDGE_DEFAULT_MODEL.get(st.session_state[f"{role}_preset"], model)
    st.session_state[f"{role}_model"] = model
    st.session_state[f"{role}_send_temp"] = p["send_temp"]
    st.session_state[f"{role}_max_tokens"] = max(p["max_tokens"], 1500 if role == "judge" else 0)
    st.session_state[f"{role}_token_param"] = p["token_param"]


def _init_role(role: str, preset: str) -> None:
    if f"{role}_preset" not in st.session_state:
        st.session_state[f"{role}_preset"] = preset
        _apply_preset(role)
        env_prefix = f"BIASSCOPE_{role.upper()}_"
        if os.environ.get(env_prefix + "BASE_URL"):
            st.session_state[f"{role}_base_url"] = os.environ[env_prefix + "BASE_URL"]
        if os.environ.get(env_prefix + "MODEL"):
            st.session_state[f"{role}_model"] = os.environ[env_prefix + "MODEL"]
        st.session_state.setdefault(f"{role}_temp", 0.0 if role == "judge" else 0.7)
        st.session_state.setdefault(f"{role}_timeout", 90)
        st.session_state.setdefault(f"{role}_retries", 3)


def provider_form(role: str, default_preset: str = "OpenAI") -> None:
    """Draw a provider form. Values live in st.session_state under '<role>_*' keys."""
    _init_role(role, default_preset)
    st.selectbox("Provider", list(PRESETS), key=f"{role}_preset", on_change=_apply_preset, args=(role,))
    st.caption(PRESETS[st.session_state[f"{role}_preset"]]["help"])
    c1, c2 = st.columns(2)
    c1.text_input("Model name", key=f"{role}_model")
    c2.text_input("API key", type="password", key=f"{role}_api_key",
                  placeholder=f"or set BIASSCOPE_{role.upper()}_API_KEY",
                  help="Kept only for this browser session. Never saved to runs, logs or downloads.")
    with st.expander("Advanced settings"):
        st.text_input("Base URL", key=f"{role}_base_url")
        a1, a2, a3 = st.columns(3)
        a1.checkbox("Send temperature", key=f"{role}_send_temp",
                    help="Untick for newer Claude models and reasoning models.")
        a2.number_input("Temperature", 0.0, 2.0, step=0.1, key=f"{role}_temp")
        a3.number_input("Max output tokens", 16, 32000, step=50, key=f"{role}_max_tokens")
        b1, b2, b3 = st.columns(3)
        b1.selectbox("Token-limit parameter", ["max_tokens", "max_completion_tokens", "omit"],
                     key=f"{role}_token_param",
                     help="Newer OpenAI reasoning models need max_completion_tokens.")
        b2.number_input("Timeout (seconds)", 5, 600, step=5, key=f"{role}_timeout")
        b3.number_input("Max retries", 0, 6, step=1, key=f"{role}_retries")


def config_from_state(role: str, as_role: str | None = None) -> ProviderConfig:
    _init_role(role, "OpenAI")
    s = st.session_state
    key = s.get(f"{role}_api_key") or os.environ.get(f"BIASSCOPE_{role.upper()}_API_KEY", "")
    return ProviderConfig(
        role=as_role or role, kind="openai_compatible",
        base_url=str(s.get(f"{role}_base_url", "")).strip(), model=str(s.get(f"{role}_model", "")).strip(),
        api_key=SecretStr(key) if key else None,
        temperature=float(s[f"{role}_temp"]) if s.get(f"{role}_send_temp", True) else None,
        max_tokens=int(s.get(f"{role}_max_tokens", 700)), token_param=s.get(f"{role}_token_param", "max_tokens"),
        timeout_s=float(s.get(f"{role}_timeout", 90)), max_retries=int(s.get(f"{role}_retries", 3)),
    )


def missing_fields(cfg: ProviderConfig) -> list[str]:
    out = []
    if not cfg.base_url:
        out.append("base URL")
    if not cfg.model:
        out.append("model name")
    if not cfg.api_key and "localhost" not in cfg.base_url and "127.0.0.1" not in cfg.base_url:
        out.append("API key")
    return out


# --------------------------------------------------------------------------- run helpers
def execute_with_progress(run_id: str, target_cfg: ProviderConfig, judge_cfg: ProviderConfig) -> bool:
    bar = st.progress(0.0, text="Starting…")
    log = st.empty()

    def progress(phase, done, total, msg):
        label = "Asking the AI" if phase == "target" else "Judging answers"
        bar.progress(min(done / max(total, 1), 1.0), text=f"{label}: {done} of {total}")
        log.caption(html.escape(msg))

    try:
        m = execute_run(store(), run_id, target_cfg, judge_cfg, progress=progress)
    except Exception as exc:
        st.error(f"The scan stopped: {type(exc).__name__}: {exc}. Finished results were saved, so you can resume it.")
        return False
    st.session_state["current_run"] = run_id
    if m.status != "completed":
        st.warning(f"The scan ended early ({m.status}). Finished results were saved.")
        return False
    return True


@st.cache_data(show_spinner=False)
def _load_run(root: str, run_id: str, mtime: float):
    s = RunStore(root)
    return s.load_manifest(run_id), s.load_suite(run_id), s.load_targets(run_id), s.load_comparisons(run_id)


def load_run(run_id: str):
    s = store()
    d = s.run_dir(run_id)
    mtime = max((p.stat().st_mtime for p in d.iterdir()), default=0)
    return _load_run(str(s.root), run_id, mtime)


def pick_run(key: str):
    runs = store().list_runs()
    if not runs:
        return None
    ids = [m.run_id for m in runs]
    current = st.session_state.get("current_run")
    idx = ids.index(current) if current in ids else 0

    def label(m):
        kind = "Simulated" if m.simulated else "Live"
        target = m.target.get("model")
        return f"{kind} · {target} · {m.created_at[:16].replace('T', ' ')} UTC · {m.status}"

    m = st.selectbox("Which scan?", runs, index=idx, key=key, format_func=label)
    st.session_state["current_run"] = m.run_id
    return m


def no_runs_message() -> None:
    st.info("There are no scans yet. Run the 1-minute demo first – it needs no key.")
    st.page_link("views/demo.py", label="Go to the demo", icon=":material/play_circle:")
