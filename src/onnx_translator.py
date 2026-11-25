"""Translate with the ONNX export: onnxruntime, numpy and sentencepiece, no PyTorch.

    from src.onnx_translator import OnnxTranslator
    tr = OnnxTranslator.from_pretrained("nickoo004/nmt-uz-76m-onnx", precision="int8")
    tr.translate("Good morning!", target="uz")

Same behaviour as src.inference.Translator; the beam search is a numpy port of
Seq2SeqTransformer.generate.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src import text
from src.data_pipeline.alphabet import to_old_latin
from src.data_pipeline.tokenizer import BOS, EOS, PAD, Tokenizer

SUFFIXES = {"fp32": "", "fp16": "_fp16", "int8": "_quantized", "int4": "_q4"}
MAX_RATIO = 3  # same cap as src.inference: longer output is a decoding loop


class OnnxTranslator:
    def __init__(self, path: str | Path, precision: str = "int8",
                 providers: list[str] | None = None, threads: int | None = None):
        path, suffix = Path(path), SUFFIXES[precision]
        cfg = json.loads((path / "config.json").read_text())["model"]
        self.max_len, self.n_layers = cfg["max_len"], cfg["dec_layers"]
        self.kv_shape = (cfg["n_kv_heads"], cfg["d_model"] // cfg["n_heads"])
        self.tok = Tokenizer(path / "spm.model")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        self.encoder, self.decoder = (
            ort.InferenceSession(str(path / "onnx" / f"{name}_model{suffix}.onnx"), options,
                                 providers=providers)
            for name in ("encoder", "decoder"))
        self.cross = [o.name for o in self.encoder.get_outputs()[1:]]
        self.past = [i.name for i in self.decoder.get_inputs()[3 + len(self.cross):]]

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, precision: str = "int8",
                        providers: list[str] | None = None,
                        threads: int | None = None) -> "OnnxTranslator":
        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            suffix = SUFFIXES[precision]
            files = ["config.json", "spm.model", f"onnx/encoder_model{suffix}.onnx",
                     f"onnx/decoder_model{suffix}.onnx"]
            path = Path(snapshot_download(str(name_or_path), allow_patterns=files))
        return cls(path, precision, providers, threads)

    def _encode(self, src: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        out = self.encoder.run(None, {"src_ids": src.astype(np.int64)})
        return out[0], out[1:]

    def _step(self, tokens, position, mask, cross, past) -> tuple[np.ndarray, list[np.ndarray]]:
        feeds = {"tokens": tokens, "position": np.array([position], dtype=np.int64),
                 "memory_mask": mask, **dict(zip(self.cross, cross)), **dict(zip(self.past, past))}
        out = self.decoder.run(None, feeds)
        return out[0], out[1:]

    def first_step(self, src: np.ndarray) -> np.ndarray:
        """Log-probabilities of the first target token (used to compare precisions)."""
        mask, cross = self._encode(src)
        empty = [np.zeros((len(src), *self.kv_shape[:1], 0, self.kv_shape[1]), np.float32)
                 for _ in self.past]
        tokens = np.full((len(src), 1), BOS, dtype=np.int64)
        return self._step(tokens, 0, mask, cross, empty)[0]

    def generate(self, src: np.ndarray, max_len: int = 256, beam: int = 4,
                 alpha: float = 0.6) -> np.ndarray:
        """Beam search with KV cache and GNMT length penalty. beam=1 is greedy."""
        b, k = src.shape[0], beam
        mask, cross = self._encode(src)
        mask, cross = np.repeat(mask, k, 0), [np.repeat(c, k, 0) for c in cross]

        tokens = np.full((b * k, 1), BOS, dtype=np.int64)
        scores = np.full((b, k), -np.inf, dtype=np.float32)
        scores[:, 0] = 0.0  # all beams start identical, keep only one alive
        scores = scores.reshape(-1)
        finished = np.zeros(b * k, dtype=bool)
        lengths = np.zeros(b * k, dtype=np.float32)
        past = [np.zeros((b * k, self.kv_shape[0], 0, self.kv_shape[1]), np.float32)
                for _ in self.past]
        offsets = (np.arange(b) * k)[:, None]

        for step in range(min(max_len, self.max_len - 1)):
            logp, past = self._step(tokens[:, -1:], step, mask, cross, past)
            vocab = logp.shape[-1]
            done_row = np.full(vocab, -np.inf, dtype=np.float32)
            done_row[EOS] = 0.0
            logp = np.where(finished[:, None], done_row, logp)

            cand = (scores[:, None] + logp).reshape(b, k * vocab)
            idx = np.argsort(-cand, axis=1, kind="stable")[:, :k]
            top = np.take_along_axis(cand, idx, axis=1)
            sel = (offsets + idx // vocab).reshape(-1)
            tok = (idx % vocab).reshape(-1)

            prev_done = finished[sel]
            tokens = np.concatenate((tokens[sel], tok[:, None]), axis=1)
            lengths = lengths[sel] + (~prev_done)
            finished = prev_done | (tok == EOS)
            scores = top.reshape(-1)
            past = [p[sel] for p in past]
            if finished.all():
                break

        penalty = ((5.0 + lengths) / 6.0) ** alpha
        best = (scores / penalty).reshape(b, k).argmax(-1)
        return tokens.reshape(b, k, -1)[np.arange(b), best, 1:]

    def _translate_segments(self, sentences: list[str], target: str, beam: int,
                            batch_size: int, max_len: int) -> list[str]:
        """Port of src.evaluation.evaluate.translate with max_ratio=MAX_RATIO."""
        norm = text.normalize_uz if target == "en" else text.normalize_en
        ids = [self.tok.source_ids(s[: max_len - 2], target)
               for s in self.tok.encode([norm(s) for s in sentences])]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [""] * len(ids)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            width = max(len(ids[i]) for i in chunk)
            src = np.full((len(chunk), width), PAD, dtype=np.int64)
            for row, i in enumerate(chunk):
                src[row, : len(ids[i])] = ids[i]
            limit = min(max_len, int(MAX_RATIO * width) + 10)
            hyp = self.generate(src, max_len=limit, beam=beam)
            for row, i in enumerate(chunk):
                out[i] = self.tok.decode(hyp[row].tolist())
        return out

    def translate(self, texts: str | list[str], target: str, beam: int = 4,
                  max_len: int | None = None, batch_size: int = 32,
                  script: str = "new") -> str | list[str]:
        if target not in ("uz", "en"):
            raise ValueError(f"target must be 'uz' or 'en', got {target!r}")
        if script not in ("new", "old"):
            raise ValueError(f"script must be 'new' or 'old', got {script!r}")
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if target == "en":
            texts = [text.normalize_uz(t) for t in texts]  # so Cyrillic splits into sentences too

        segments, layout = text.segment(texts, lambda s: len(self.tok.encode([s])[0]))
        max_len = min(max_len or self.max_len, self.max_len)
        out = self._translate_segments(segments, target, beam, batch_size,
                                       max_len) if segments else []
        if target == "uz" and script == "old":
            out = [to_old_latin(s) for s in out]
        joined = text.assemble(out, layout)
        return joined[0] if single else joined
