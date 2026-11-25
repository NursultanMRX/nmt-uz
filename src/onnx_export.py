"""Export a release to ONNX in four precisions: an encoder and a one-step decoder with KV cache.

    python -m src.onnx_export --release release/    # writes release/onnx/*.onnx

The encoder also returns the cross-attention keys and values of every decoder layer, so a
decoding step only runs the decoder. File names follow the transformers.js convention.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.export import load_release
from src.models.transformer import Seq2SeqTransformer, _heads

VARIANTS = {"fp32": "", "fp16": "_fp16", "int8": "_quantized", "int4": "_q4"}
OPSET = 17


def io_names(n_layers: int) -> tuple[list[str], list[str], list[str]]:
    """Names of the cross-attention, past and present tensors, k before v, layer by layer."""
    def kv(prefix: str) -> list[str]:
        return [f"{prefix}_{t}_{i}" for i in range(n_layers) for t in ("k", "v")]
    return kv("cross"), kv("past"), kv("present")


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class _RMSNorm(nn.Module):
    """nn.RMSNorm written out: opset 17 has no RMSNormalization op."""

    def __init__(self, norm: nn.RMSNorm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _exportable(model: Seq2SeqTransformer) -> Seq2SeqTransformer:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.RMSNorm):
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, _RMSNorm(module))
    return model


class _Encoder(nn.Module):
    def __init__(self, model: Seq2SeqTransformer):
        super().__init__()
        self.m = model

    def forward(self, src_ids):
        memory, mask = self.m.encode(src_ids)
        cross = [t for layer in self.m.decoder for t in layer.cross_attn.kv(memory)]
        return (mask[:, 0, 0, :], *cross)


class _Decoder(nn.Module):
    """Seq2SeqTransformer.decode for one token with an explicit cache. The RoPE offset is an
    input tensor, so one graph serves every step."""

    def __init__(self, model: Seq2SeqTransformer):
        super().__init__()
        self.m = model

    def forward(self, tokens, position, memory_mask, *kv):
        n = len(self.m.decoder)
        cross, past = kv[: 2 * n], kv[2 * n :]
        cos, sin = self.m.rope.cos[position], self.m.rope.sin[position]
        mask = memory_mask[:, None, None, :]
        x = self.m.embed(tokens)
        present = []
        for i, layer in enumerate(self.m.decoder):
            attn = layer.self_attn
            h = layer.norm1(x)
            q = _rope(_heads(attn.wq(h), attn.h), cos, sin)
            k, v = attn.kv(h)
            k = torch.cat((past[2 * i], _rope(k, cos, sin)), dim=2)
            v = torch.cat((past[2 * i + 1], v), dim=2)
            present += [k, v]
            x = x + attn.attend(q, k, v)
            ca = layer.cross_attn
            x = x + ca.attend(_heads(ca.wq(layer.norm2(x)), ca.h), cross[2 * i], cross[2 * i + 1],
                              mask)
            x = x + layer.ffn(layer.norm3(x))
        logprobs = self.m.lm_head(self.m.dec_norm(x))[:, -1].float().log_softmax(-1)
        return (logprobs, *present)


def _export_fp32(model: Seq2SeqTransformer, enc_path: Path, dec_path: Path) -> None:
    cfg = model.cfg
    cross, past, present = io_names(cfg.dec_layers)
    hd = cfg.d_model // cfg.n_heads
    src = torch.tensor([[5, 6, 7, 2], [5, 6, 2, 0]])
    batch_src = {0: "batch", 1: "src_len"}
    torch.onnx.export(
        _Encoder(model), (src,), enc_path, dynamo=False, opset_version=OPSET,
        input_names=["src_ids"], output_names=["memory_mask", *cross],
        dynamic_axes={"src_ids": batch_src, "memory_mask": batch_src,
                      **{n: {0: "batch", 2: "src_len"} for n in cross}},
    )
    b, s, t = 2, 4, 3
    kv = lambda length: torch.randn(b, cfg.n_kv_heads, length, hd)
    args = (torch.ones(b, 1, dtype=torch.long), torch.tensor([t]),
            torch.ones(b, s, dtype=torch.bool), *[kv(s) for _ in cross], *[kv(t) for _ in past])
    torch.onnx.export(
        _Decoder(model), args, dec_path, dynamo=False, opset_version=OPSET,
        input_names=["tokens", "position", "memory_mask", *cross, *past],
        output_names=["logprobs", *present],
        dynamic_axes={"tokens": {0: "batch"}, "memory_mask": batch_src, "logprobs": {0: "batch"},
                      **{n: {0: "batch", 2: "src_len"} for n in cross},
                      **{n: {0: "batch", 2: "past_len"} for n in past},
                      **{n: {0: "batch", 2: "total_len"} for n in present}},
    )


def _convert(fp32: Path, out: Path, variant: str) -> None:
    import onnx

    if variant == "fp16":
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel

        # keep_io_types adds input casts at the end of the node list; sort so the checker passes
        model = OnnxModel(convert_float_to_float16(onnx.load(fp32), keep_io_types=True))
        model.topological_sort()
        onnx.save(model.model, out)
    elif variant == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Gather too: the embedding table is a fifth of the model and would stay fp32
        quantize_dynamic(fp32, out, weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gather"])
    elif variant == "int4":
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        # Quantizing Gather raises the opset to 21 without upgrading the other ops; do it first.
        from onnx import version_converter

        model = version_converter.convert_version(onnx.load(fp32), 21)
        quant = MatMulNBitsQuantizer(model, block_size=32, is_symmetric=True,
                                     op_types_to_quantize=("MatMul", "Gather"),
                                     quant_axes=(("MatMul", 0), ("Gather", 1)))
        quant.process()
        quant.model.save_model_to_file(str(out), use_external_data_format=False)
    else:
        raise ValueError(f"unknown variant {variant!r}")


def export_onnx(release: str | Path, out: str | Path, variants=VARIANTS) -> Path:
    """Write out/onnx/{encoder,decoder}_model{suffix}.onnx for each variant; returns out/onnx."""
    model = _exportable(load_release(release))
    onnx_dir = Path(out) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32 = {name: onnx_dir / f"{name}.onnx" for name in ("encoder_model", "decoder_model")}
    with torch.no_grad():
        _export_fp32(model, fp32["encoder_model"], fp32["decoder_model"])
    for variant in variants:
        if variant != "fp32":
            for name, path in fp32.items():
                _convert(path, onnx_dir / f"{name}{VARIANTS[variant]}.onnx", variant)
    return onnx_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release")
    ap.add_argument("--out", help="defaults to --release")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args(argv)
    out = export_onnx(args.release, args.out or args.release, args.variants)
    for f in sorted(out.glob("*.onnx")):
        print(f"{f.name:34} {f.stat().st_size / 2**20:7.1f} MB")


if __name__ == "__main__":
    main()
