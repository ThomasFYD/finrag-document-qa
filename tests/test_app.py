"""Exercise the English demo with the real local retrieval pipeline."""
import os

import pytest
from streamlit.testing.v1 import AppTest

from finrag.config import ROOT

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.getenv("FINRAG_INTEGRATION") != "1", reason="requires the real local model and index")]


def test_english_demo_accepts_chinese_question():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Every number. Backed by evidence."
    app.text_input[0].set_value("游戏业务第二季度的收入是多少？")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    result = app.session_state["result"]
    assert result["mode"] == "evidence"
    assert any("2.9 billion" in evidence["text"] for evidence in result["evidence"])
    assert len(app.metric) == 3
