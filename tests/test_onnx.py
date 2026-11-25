import os
import subprocess
import sys

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from src.data_pipeline.tokenizer import Tokenizer
from src.export import export, load_release
from src.inference import Translator
from src.onnx_export import VARIANTS, export_onnx
from src.onnx_translator import OnnxTranslator

TEXTS = [
    "good morning",
    "hello. a good book about the city school",
    "read the book ❤️",
    "the good school in the city, a book about the morning, hello world",
]


@pytest.fixture
def release(tmp_path, checkpoint, spm_model):
    out = export(checkpoint[0], spm_model, tmp_path / "release")
    export_onnx(out, out, variants=VARIANTS)
    return out


def test_graphs_pass_the_checker(release):
    for name in ("encoder_model", "decoder_model"):
        for suffix in VARIANTS.values():
            onnx.checker.check_model(str(release / "onnx" / f"{name}{suffix}.onnx"))


@pytest.mark.parametrize("beam", [1, 4])
def test_fp32_generates_the_same_tokens_as_pytorch(release, beam):
    model, tok = load_release(release), Tokenizer(release / "spm.model")
    ids = [tok.source_ids(s, "uz") for s in tok.encode(["good morning", "hello world city"])]
    src = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)  # PAD = 0
    for row, s in enumerate(ids):
        src[row, : len(s)] = torch.tensor(s)
    expected = model.generate(src, 1, 2, max_len=20, beam=beam).tolist()
    got = OnnxTranslator.from_pretrained(release, precision="fp32").generate(
        src.numpy(), max_len=20, beam=beam).tolist()
    assert [r[: len(e)] for r, e in zip(got, expected)] == expected


def test_fp32_translate_matches_pytorch(release):
    pt = Translator.from_pretrained(release, device="cpu")
    ox = OnnxTranslator.from_pretrained(release, precision="fp32")
    assert ox.translate(TEXTS, "uz") == pt.translate(TEXTS, "uz")
    assert ox.translate("Ўзбекистон шаҳар", "en") == pt.translate("Ўзбекистон шаҳар", "en")
    assert ox.translate(TEXTS[0], "uz", script="old") == pt.translate(TEXTS[0], "uz", script="old")


@pytest.mark.parametrize("precision,tol", [("fp16", 0.02), ("int8", 0.3), ("int4", 0.5)])
def test_reduced_precision_stays_close_to_fp32(release, precision, tol):
    ref = OnnxTranslator.from_pretrained(release, precision="fp32")
    low = OnnxTranslator.from_pretrained(release, precision=precision)
    src = np.array([ref.tok.source_ids(ref.tok.encode(["good morning"])[0], "uz")])
    diff = np.abs(low.first_step(src) - ref.first_step(src)).mean()
    assert diff < tol
    assert isinstance(low.translate("good morning", "uz"), str)


def test_runtime_does_not_need_torch(release):
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from src.onnx_translator import OnnxTranslator\n"
        f"tr = OnnxTranslator.from_pretrained(r'{release}', precision='int8')\n"
        "print(tr.translate('good morning', 'uz'))\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         check=False)
    assert out.returncode == 0, out.stderr
