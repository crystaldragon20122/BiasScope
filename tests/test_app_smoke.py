"""Headless smoke tests of the multipage website: every page renders, and the demo and
live-proof practice run work end to end through the UI."""

import importlib
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["views/home.py", "views/demo.py", "views/live_proof.py", "views/scan.py", "views/results.py",
         "views/evidence.py", "views/judge_check.py", "views/about.py"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("BIASSCOPE_WORKSPACE", str(tmp_path / "ws"))
    import biasscope.config as cfg

    importlib.reload(cfg)
    yield AppTest.from_file(APP, default_timeout=120)
    monkeypatch.delenv("BIASSCOPE_WORKSPACE")
    importlib.reload(cfg)


def _errors(at):
    return [e.value for e in at.exception]


def test_every_page_renders_with_no_scans(app):
    app.run()
    assert not _errors(app)
    for page in PAGES:
        app.switch_page(page).run()
        assert not _errors(app), page


def test_demo_then_results_and_evidence(app):
    app.run()
    app.switch_page("views/demo.py").run()
    next(b for b in app.button if b.label == "Run the demo scan").click().run()
    assert not _errors(app)
    assert app.session_state["demo_last_run"]
    assert any("Explicit stereotyping" == m.label for m in app.metric)
    for page in ["views/results.py", "views/evidence.py"]:
        app.switch_page(page).run()
        assert not _errors(app), page
    # the evidence page's "Next" button steps through findings
    next(b for b in app.button if b.label == "Next →").click().run()
    assert not _errors(app)
    assert app.session_state["ev_idx"] == 1


def test_live_proof_practice_run(app):
    app.run()
    app.switch_page("views/live_proof.py").run()
    app.radio(key="proof_mode").set_value("practice").run()
    next(b for b in app.button if b.label == "Run the live proof").click().run()
    assert not _errors(app)
    res = app.session_state["proof_result"]
    assert res["summary"]["quiet"] == (8, 8)


def test_live_modes_need_a_key_before_running(app, monkeypatch):
    for v in ("BIASSCOPE_TARGET_API_KEY", "BIASSCOPE_JUDGE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    app.run()
    app.switch_page("views/live_proof.py").run()
    assert next(b for b in app.button if b.label == "Run the live proof").disabled
    app.switch_page("views/scan.py").run()
    assert not _errors(app)
    assert next(b for b in app.button if b.label == "Run scan").disabled


def test_judge_check_practice(app):
    app.run()
    app.switch_page("views/judge_check.py").run()
    next(b for b in app.button if b.label == "Test the judge").click().run()
    assert not _errors(app)
    assert app.session_state["val_result"]
