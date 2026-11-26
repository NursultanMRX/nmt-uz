import numpy as np
import pytest
import sacrebleu

from src.significance import bootstrap_indices, compare, render_markdown

REFS = [f"the {w} sat on the mat number {i}" for i, w in enumerate(["cat", "dog", "bird"] * 40)]
GOOD = [r.replace("mat", "rug") for r in REFS]
BAD = [" ".join(r.split()[:3]) for r in REFS]


def _refs():
    return {"en2uz": REFS, "uz2en": REFS}


def test_point_estimates_are_corpus_scores():
    out = compare({"ours": {"en2uz": GOOD, "uz2en": BAD}}, _refs(), samples=50)
    scores = out["systems"]["ours"]
    assert scores["en2uz"]["bleu"]["score"] == pytest.approx(
        sacrebleu.corpus_bleu(GOOD, [REFS]).score)
    assert scores["uz2en"]["chrf"]["score"] == pytest.approx(
        sacrebleu.corpus_chrf(BAD, [REFS]).score)
    lo, hi = scores["en2uz"]["bleu"]["ci"]
    assert lo <= scores["en2uz"]["bleu"]["score"] <= hi


def test_a_system_against_itself_is_not_significant():
    hyps = {"en2uz": GOOD, "uz2en": GOOD}
    out = compare({"ours": hyps, "copy": hyps}, _refs(), samples=200)
    d = out["deltas"]["copy"]["en2uz"]["bleu"]
    assert d["delta"] == 0
    assert d["ci"][0] <= 0 <= d["ci"][1]
    assert d["p"] >= 0.5


def test_a_clearly_better_system_is_significant():
    out = compare({"ours": {"en2uz": REFS, "uz2en": REFS},
                   "other": {"en2uz": BAD, "uz2en": BAD}}, _refs(), samples=200)
    for metric in ("bleu", "chrf"):
        d = out["deltas"]["other"]["uz2en"][metric]
        assert d["delta"] > 0
        assert d["ci"][0] > 0
        assert d["p"] < 0.05


def test_same_seed_same_result():
    systems = {"ours": {"en2uz": GOOD, "uz2en": BAD}, "other": {"en2uz": BAD, "uz2en": GOOD}}
    assert compare(systems, _refs(), samples=100, seed=7) == \
        compare(systems, _refs(), samples=100, seed=7)
    idx = bootstrap_indices(10, samples=5, seed=7)
    assert idx.shape == (5, 10) and idx.min() >= 0 and idx.max() < 10


def test_comet_segment_scores_are_bootstrapped_with_the_same_indices():
    n = len(REFS)
    comet = {"ours": {"en2uz": [0.9] * n, "uz2en": [0.8] * n},
             "other": {"en2uz": [0.7] * n, "uz2en": [0.8] * n}}
    systems = {"ours": {"en2uz": GOOD, "uz2en": GOOD}, "other": {"en2uz": BAD, "uz2en": GOOD}}
    out = compare(systems, _refs(), comet=comet, samples=100)
    assert out["systems"]["ours"]["en2uz"]["comet"]["score"] == pytest.approx(0.9)
    d = out["deltas"]["other"]
    assert d["en2uz"]["comet"]["delta"] == pytest.approx(0.2)
    assert d["en2uz"]["comet"]["p"] == 0
    assert d["uz2en"]["comet"]["delta"] == pytest.approx(0.0)
    assert d["uz2en"]["comet"]["p"] == 1


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compare({"ours": {"en2uz": GOOD[:-1], "uz2en": GOOD}}, _refs(), samples=10)
    n = len(REFS)
    with pytest.raises(ValueError):
        compare({"ours": {"en2uz": GOOD, "uz2en": GOOD}}, _refs(),
                comet={"ours": {"en2uz": [0.5] * (n - 1), "uz2en": [0.5] * n}}, samples=10)


def test_render_markdown_lists_every_comparison():
    systems = {"ours": {"en2uz": GOOD, "uz2en": GOOD}, "nllb": {"en2uz": BAD, "uz2en": BAD}}
    md = render_markdown(compare(systems, _refs(), samples=50), names={"nllb": "NLLB-600M"})
    assert "NLLB-600M" in md and "en→uz" in md and "uz→en" in md
    assert np.isfinite(compare(systems, _refs(), samples=50)["deltas"]["nllb"]["en2uz"]["bleu"]
                       ["delta"])
