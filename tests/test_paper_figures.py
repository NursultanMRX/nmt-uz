"""The paper gets vector PDFs without in-figure titles; the caption carries the title."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts import figures, speed_quality

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _no_title(*args, **kwargs):
    raise AssertionError("paper figures must not draw a title")


def test_figures_paper_mode_writes_pdfs_without_titles(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "_title", _no_title)
    figures.main(paper=True, out=tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "data_funnel.pdf" in names
    assert all(n.endswith(".pdf") for n in names)   # no SVGs, no dark variants


def test_speed_quality_paper_mode_writes_pdf(tmp_path):
    speed_quality.draw(speed_quality.load_points(RESULTS / "efficiency"), tmp_path, paper=True)
    assert [p.name for p in tmp_path.iterdir()] == ["speed_quality.pdf"]
