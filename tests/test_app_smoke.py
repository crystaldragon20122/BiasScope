"""Headless smoke test of the Streamlit app: offline scan end to end through the UI."""

import importlib
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("BIASSCOPE_WORKSPACE", str(tmp_path / "ws"))
    import biasscope.config as cfg

    importlib.reload(cfg)
    yield AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    monkeypatch.delenv("BIASSCOPE_WORKSPACE")
    importlib.reload(cfg)


def test_offline_scan_through_ui_and_all_views_render(app):
    app.run()
    assert not app.exception
    run_btn = next(b for b in app.button if "Run Scan" in b.label)
    run_btn.click().run()
    assert not app.exception
    assert any("Scan complete" in s.value for s in app.success)
    for page in ["Results", "Evidence Explorer", "Auditor Validation"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    next(b for b in app.button if b.label == "Run validation").click().run()
    assert not app.exception


def test_live_mode_requires_configuration_and_cost_acknowledgement(app):
    app.run()
    app.radio(key="mode").set_value("Live (OpenAI-compatible endpoint)").run()
    assert not app.exception
    run_btn = next(b for b in app.button if "Run Scan" in b.label)
    assert run_btn.disabled  # no live request is possible until configured + acknowledged
