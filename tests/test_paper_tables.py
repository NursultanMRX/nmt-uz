"""The paper's tables must say exactly what the result files say."""

import json
import re
from pathlib import Path

import pytest

from scripts import paper_tables as pt

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _row(table: str, name: str) -> list[str]:
    """Cells of the first LaTeX row whose first cell contains `name`."""
    for line in table.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) > 1 and name in cells[0]:
            return cells
    raise AssertionError(f"no row for {name!r}")


def _num(cell: str) -> float:
    return float(re.search(r"[-+−]?\d*\.?\d+", cell).group().replace("−", "-"))


def _bench(system: str) -> dict:
    return json.loads((RESULTS / "benchmark" / f"{system}_devtest.json").read_text())["full"]["compat"]


@pytest.mark.parametrize("system,label", [("ours", "nmt-uz-76m"), ("nllb", "600M"),
                                          ("nllb-1.3b", "1.3B")])
def test_main_table_matches_benchmark_files(system, label):
    cells = _row(pt.main_table(RESULTS), label)
    b = _bench(system)
    expected = [b["en2uz"]["bleu"], b["en2uz"]["chrf"], b["en2uz"]["comet"],
                b["uz2en"]["bleu"], b["uz2en"]["chrf"], b["uz2en"]["comet"]]
    got = [_num(c) for c in cells[2:8]]
    assert got == pytest.approx(expected, abs=0.006)


def test_main_table_bolds_the_best_score_per_column():
    table = pt.main_table(RESULTS)
    assert r"\textbf{19.89}" in table      # en->uz BLEU: ours
    assert r"\textbf{34.61}" in table      # uz->en BLEU: NLLB-1.3B
    assert r"\textbf{19.89}" not in _row(table, "1.3B")[2]


def test_significance_table_matches_deltas():
    deltas = json.loads((RESULTS / "significance" / "devtest.json").read_text())["deltas"]
    table = pt.significance_table(RESULTS)
    d = deltas["nllb"]["en2uz"]["bleu"]
    assert f"{d['delta']:+.2f}" in table
    assert f"[{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]" in table
    assert "p = 0.015" in table            # the one COMET difference that is not p < 0.001


def test_reported_table_lists_stronger_published_systems():
    table = pt.reported_table(RESULTS)
    assert "Tilmoch" in table and "uz-en-v1" in table
    ours = _row(table, "nmt-uz-76m")
    first200 = json.loads((RESULTS / "benchmark" / "ours_devtest.json").read_text())["first200"]
    assert _num(ours[2]) == pytest.approx(first200["compat"]["en2uz"]["bleu"])


def test_latency_table_matches_latency_files():
    table = pt.latency_table(RESULTS)
    ours = json.loads((RESULTS / "latency" / "ours_cpu.json").read_text())
    assert f"{ours['beam4']['en2uz']['p50_ms']:.0f}" in table
    int8 = json.loads((RESULTS / "latency" / "onnx-int8_cpu.json").read_text())
    assert f"{int8['beam4']['en2uz']['p50_ms']:.0f}" in table


def test_data_table_totals():
    table = pt.data_table(RESULTS)
    stats = json.loads((RESULTS / "data_stats.json").read_text())
    assert f"{stats['stores']['train']['pairs']:,}".replace(",", "{,}") in table
    assert f"{stats['stores']['clean']['pairs']:,}".replace(",", "{,}") in table


def test_write_all(tmp_path):
    pt.write_all(RESULTS, tmp_path)
    names = sorted(p.name for p in tmp_path.glob("*.tex"))
    assert names == ["data.tex", "latency.tex", "main.tex", "onnx.tex", "reported.tex",
                     "significance.tex"]
    for p in tmp_path.glob("*.tex"):
        text = p.read_text(encoding="utf-8")
        assert text.count(r"\begin{tabular}") == 1 and text.count(r"\end{tabular}") == 1
