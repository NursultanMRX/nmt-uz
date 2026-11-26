"""Runs the pipeline on Modal: corpus build on CPU, training and eval on a T4.

One-time setup:
    modal secret create huggingface HF_TOKEN=hf_...      # FLORES+ is gated

Typical flow:
    modal run modal_app.py::build                        # ~33M raw pairs -> stores on nmt-data
    modal run modal_app.py::train --run base --max-steps 50          # smoke test
    modal run --detach modal_app.py::train --run base                 # full run, resumes if
                                                                      # runs/base/last.pt exists
    modal run modal_app.py::evaluate --run base --split devtest
    modal run modal_app.py::benchmark --system nllb                    # NLLB-600M baseline
    modal run modal_app.py::benchmark --system nllb-1.3b
    modal run modal_app.py::benchmark --system ours --run base         # + COMET, old-alphabet track
    modal run modal_app.py::significance                               # bootstrap CIs, p-values
    modal run modal_app.py::latency                                    # CPU and T4, ours vs NLLB
    modal run --detach modal_app.py::efficiency                        # fp32/fp16/int8/int4 sweep
    modal volume get nmt-ckpt runs/base/efficiency results/            # its results, any time later
    modal run --detach modal_app.py::backtranslate --run base --lang uz
    modal volume get nmt-ckpt runs/base results/base                   # pull results
"""

import shlex
import subprocess

import modal

app = modal.App("nmt-uz")


def _with_code(img: modal.Image) -> modal.Image:
    return (img.add_local_dir("src", "/root/src")
               .add_local_dir("configs", "/root/configs"))


base_image = modal.Image.debian_slim(python_version="3.11").uv_sync()
image = _with_code(base_image)
# NLLB only matters for the benchmark; keep it out of the training image.
bench_image = _with_code(base_image.uv_pip_install("transformers", "onnx", "onnxruntime",
                                                   "onnxscript"))
# int8 / int4 rows of the efficiency table
quant_image = _with_code(base_image.uv_pip_install("transformers", "accelerate", "bitsandbytes"))
# unbabel-comet pins an older huggingface-hub, which drags torch back to 1.x inside our
# environment, so COMET gets an image of its own.
comet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("unbabel-comet>=2.2", "sacrebleu", "polars", "setuptools<81")  # pkg_resources
    .add_local_dir("src", "/root/src")
)

data_vol = modal.Volume.from_name("nmt-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("nmt-ckpt", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

DATA, CKPT = "/data/processed", "/ckpt/runs"
DAY = 24 * 60 * 60


@app.function(image=image, cpu=16, memory=64 * 1024, timeout=DAY,
              volumes={"/data": data_vol}, secrets=[hf_secret])
def build_corpus(extra_args: str = ""):
    from src.data_pipeline.build import main

    main(["--out", DATA, "--cache", "/data/raw", *shlex.split(extra_args)])
    data_vol.commit()


# CPU and RAM are billed on top of the GPU, so the training box stays small.
@app.function(image=image, gpu="T4", cpu=2, memory=16 * 1024, timeout=DAY,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol})
def train_remote(run: str, config: str, extra_args: str = ""):
    from src.train import main

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    # GPU containers can be preempted and are restarted with the same input,
    # so every start resumes from last.pt when one exists.
    main(["--config", config, "--data", DATA, "--run-dir", f"{CKPT}/{run}", "--resume",
          *shlex.split(extra_args)], on_save=ckpt_vol.commit)
    ckpt_vol.commit()


@app.function(image=image, gpu="T4", memory=32 * 1024, timeout=4 * 60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol})
def evaluate_remote(run: str, split: str, beam: int, checkpoint: str):
    from src.evaluate_run import main

    main(["--run-dir", f"{CKPT}/{run}", "--data", DATA, "--split", split,
          "--beam", str(beam), "--checkpoint", checkpoint])
    ckpt_vol.commit()


@app.function(image=image, gpu="T4", memory=32 * 1024, timeout=DAY,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol})
def backtranslate_remote(run: str, mono: str, lang: str, out: str, limit: int):
    from src.backtranslate import main

    main(["--checkpoint", f"{CKPT}/{run}/best.pt", "--data", DATA, "--mono", f"/data/{mono}",
          "--lang", lang, "--out", out, *(["--limit", str(limit)] if limit else [])])
    data_vol.commit()


def _model(system: str, run: str) -> str:
    """Our release on the volume (exported on first use), or the baseline's hub name."""
    from pathlib import Path

    from src.benchmark import SYSTEMS

    if system != "ours" and not system.startswith("onnx-"):
        return SYSTEMS[system][2]
    release = Path(f"{CKPT}/{run}/release")
    if not release.exists():
        from src.export import export

        export(f"{CKPT}/{run}/best.pt", f"{DATA}/spm/spm.model", release)
    if system.startswith("onnx-") and not (release / "onnx").exists():
        from src.onnx_export import export_onnx

        export_onnx(release, release)
    ckpt_vol.commit()
    return str(release)


@app.function(image=bench_image, cpu=4, memory=16 * 1024, timeout=60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol})
def onnx_export_remote(run: str) -> list[str]:
    import shutil
    from pathlib import Path

    shutil.rmtree(Path(f"{CKPT}/{run}/release/onnx"), ignore_errors=True)
    out = Path(_model("onnx-fp32", run)) / "onnx"
    return [f"{f.name} {f.stat().st_size / 2**20:.1f} MB" for f in sorted(out.glob("*.onnx"))]


def _benchmark(system: str, run: str, split: str, beam: int):
    import json
    from pathlib import Path

    from src.benchmark import run as run_benchmark

    result = run_benchmark(system, _model(system, run), split,
                           Path(f"{CKPT}/{run}/benchmark"), beam, comet=comet_remote.remote)
    ckpt_vol.commit()
    print(json.dumps(result, indent=2))


@app.function(image=bench_image, gpu="T4", memory=16 * 1024, timeout=2 * 60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret])
def benchmark_remote(system: str, run: str, split: str, beam: int):
    _benchmark(system, run, split, beam)


@app.function(image=bench_image, cpu=8, memory=16 * 1024, timeout=4 * 60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret])
def benchmark_cpu_remote(system: str, run: str, split: str, beam: int):
    """The ONNX variants run on onnxruntime's CPU provider; a GPU would sit idle."""
    _benchmark(system, run, split, beam)


def _latency(system: str, run: str, device: str, threads: int | None = None) -> dict:
    from src.latency import run as run_latency

    return run_latency(system, _model(system, run), device, threads=threads)


@app.function(image=bench_image, cpu=4, memory=16 * 1024, timeout=2 * 60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret])
def latency_cpu(system: str, run: str) -> dict:
    return _latency(system, run, "cpu", threads=4)


@app.function(image=bench_image, gpu="T4", memory=16 * 1024, timeout=60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret])
def latency_gpu(system: str, run: str) -> dict:
    return _latency(system, run, "cuda")


@app.function(image=quant_image, gpu="T4", memory=16 * 1024, timeout=4 * 60 * 60,
              volumes={"/ckpt": ckpt_vol}, secrets=[hf_secret])
def efficiency_remote(system: str, precision: str, run: str) -> dict:
    import json
    from pathlib import Path

    from src.benchmark import SYSTEMS
    from src.efficiency import run as run_efficiency

    model = f"{CKPT}/{run}/release" if system == "ours" else SYSTEMS[system][2]
    result = run_efficiency(system, model, precision)
    # also kept on the volume, so a --detach run survives the local client going away
    out = Path(f"{CKPT}/{run}/efficiency")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{system}_{precision}.json").write_text(json.dumps(result, indent=2))
    ckpt_vol.commit()
    return result


@app.function(image=comet_image, gpu="T4", memory=16 * 1024, timeout=60 * 60)
def comet_remote(src: list[str], mt: list[str], ref: list[str]) -> list[float]:
    from src.benchmark import comet_scorer

    return list(comet_scorer()(src, mt, ref))


@app.function(image=bench_image, cpu=4, memory=16 * 1024, timeout=60 * 60,
              secrets=[hf_secret])
def significance_remote(hyps: dict[str, dict[str, list[str]]], split: str) -> dict:
    from src.benchmark import to_reference_script, writes_2026
    from src.data_pipeline.sources import load_flores
    from src.significance import compare, render_markdown

    rows = load_flores(split)
    en, uz = [r["en"] for r in rows], [r["uz"] for r in rows]
    refs, srcs = {"en2uz": uz, "uz2en": en}, {"en2uz": en, "uz2en": uz}
    systems = {s: {"en2uz": to_reference_script(h["en2uz"], uz) if writes_2026(s) else h["en2uz"],
                   "uz2en": h["uz2en"]} for s, h in hyps.items()}
    # one COMET call for every system and direction, so the model loads once
    keys = [(s, d) for s in systems for d in refs]
    seg = comet_remote.remote([x for s, d in keys for x in srcs[d]],
                              [x for s, d in keys for x in systems[s][d]],
                              [x for s, d in keys for x in refs[d]])
    n, comet = len(rows), {}
    for i, (s, d) in enumerate(keys):
        comet.setdefault(s, {})[d] = seg[i * n : (i + 1) * n]
    result = compare(systems, refs, comet=comet)
    names = {"ours": "nmt-uz-76m", "nllb": "NLLB-200-distilled-600M",
             "nllb-1.3b": "NLLB-200-distilled-1.3B"}
    return {"result": result, "markdown": render_markdown(result, names)}


@app.local_entrypoint()
def build(args: str = ""):
    build_corpus.remote(args)


@app.local_entrypoint()
def train(run: str = "base", config: str = "configs/base.yaml", max_steps: int = 0,
          overrides: str = ""):
    extra = [f"--max-steps={max_steps}"] if max_steps else []
    extra += [f"--set={s}" for s in overrides.split(",") if s]
    train_remote.remote(run, config, " ".join(extra))


@app.local_entrypoint()
def evaluate(run: str = "base", split: str = "devtest", beam: int = 5,
             checkpoint: str = "best.pt"):
    evaluate_remote.remote(run, split, beam, checkpoint)


@app.local_entrypoint()
def benchmark(system: str = "ours", run: str = "base", split: str = "devtest", beam: int = 5):
    remote = benchmark_cpu_remote if system.startswith("onnx-") else benchmark_remote
    remote.remote(system, run, split, beam)


@app.local_entrypoint()
def onnx(run: str = "base"):
    """(Re)export release/onnx on the volume; the ONNX benchmarks then reuse it."""
    for line in onnx_export_remote.remote(run):
        print(line)


@app.local_entrypoint()
def latency(systems: str = "ours,nllb", run: str = "base", out: str = "results/latency"):
    import json
    from pathlib import Path

    # the ONNX variants are measured on CPU only (onnxruntime's CPU package)
    calls = [(s, d, fn.spawn(s, run)) for s in systems.split(",")
             for d, fn in (("cpu", latency_cpu), ("cuda", latency_gpu))
             if d == "cpu" or not s.startswith("onnx-")]
    Path(out).mkdir(parents=True, exist_ok=True)
    for system, device, call in calls:
        result = call.get()
        Path(out, f"{system}_{device}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


@app.local_entrypoint()
def efficiency(systems: str = "ours,nllb", precisions: str = "fp32,fp16,int8,int4",
               run: str = "base", out: str = "results/efficiency"):
    import json
    from pathlib import Path

    calls = [(s, p, efficiency_remote.spawn(s, p, run))
             for s in systems.split(",") for p in precisions.split(",")]
    Path(out).mkdir(parents=True, exist_ok=True)
    for system, precision, call in calls:
        try:
            result = call.get()
        except (modal.exception.Error, RuntimeError, ImportError) as e:  # keep the others
            print(f"{system} {precision} failed: {e}")
            continue
        Path(out, f"{system}_{precision}.json").write_text(json.dumps(result, indent=2))
        print(f"{system} {precision} done: {json.dumps(result['seconds'])}")
    print(f"python -m src.efficiency --table {out}")


@app.local_entrypoint()
def significance(systems: str = "ours,nllb,nllb-1.3b", split: str = "devtest",
                 hyps_dir: str = "results/benchmark", out: str = "results/significance"):
    """Bootstrap CIs and p-values from the saved benchmark translations."""
    import json
    from pathlib import Path

    hyps = {s: {d: Path(hyps_dir, f"{s}_{split}.{d}.txt").read_text(encoding="utf-8")
                .splitlines() for d in ("en2uz", "uz2en")} for s in systems.split(",")}
    res = significance_remote.remote(hyps, split)
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, f"{split}.json").write_text(json.dumps(res["result"], indent=2))
    Path(out, f"{split}.md").write_text(res["markdown"], encoding="utf-8")
    print(res["markdown"])


@app.local_entrypoint()
def backtranslate(run: str = "base", mono: str = "mono/uz.txt", lang: str = "uz",
                  out: str = "stores/bt_uz", limit: int = 0):
    backtranslate_remote.remote(run, mono, lang, out, limit)
