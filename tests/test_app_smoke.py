"""Smoke tests for the Streamlit dashboard entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from careerpilot.embeddings import SemanticJobIndex


def test_streamlit_app_renders_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify app.py initializes, renders all tabs, and throws zero Streamlit exceptions."""
    orig_init = SemanticJobIndex.__init__

    def _fast_init(
        self: SemanticJobIndex,
        jobs,
        model_name: str = "all-MiniLM-L6-v2",
        prefer_transformer: bool = True,
    ) -> None:
        orig_init(self, jobs=jobs, model_name=model_name, prefer_transformer=False)

    monkeypatch.setattr(SemanticJobIndex, "__init__", _fast_init)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 4
