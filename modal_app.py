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
bench_image = _with_code(base_image.uv_pip_install("transformers"))
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


@app.function(image=bench_image, gpu="T4", memory=16 * 1024, timeout=2 * 60 * 60,
              volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret])
def benchmark_remote(system: str, run: str, split: str, beam: int):
    import json
    from pathlib import Path

    from src.benchmark import SYSTEMS
    from src.benchmark import run as run_benchmark
    from src.export import export

    model = SYSTEMS[system][2]
    if system == "ours":
        model = f"{CKPT}/{run}/release"
        if not Path(model).exists():
            export(f"{CKPT}/{run}/best.pt", f"{DATA}/spm/spm.model", model)
    result = run_benchmark(system, model, split, Path(f"{CKPT}/{run}/benchmark"), beam,
                           comet=comet_remote.remote)
    ckpt_vol.commit()
    print(json.dumps(result, indent=2))


def _latency(system: str, run: str, device: str, threads: int | None = None) -> dict:
    from src.benchmark import SYSTEMS
    from src.latency import run as run_latency

    model = f"{CKPT}/{run}/release" if system == "ours" else SYSTEMS[system][2]
    return run_latency(system, model, device, threads=threads)


@app.function(image=bench_image, cpu=4, memory=16 * 1024, timeout=2 * 60 * 60,
              volumes={"/ckpt": ckpt_vol}, secrets=[hf_secret])
def latency_cpu(system: str, run: str) -> dict:
    return _latency(system, run, "cpu", threads=4)


@app.function(image=bench_image, gpu="T4", memory=16 * 1024, timeout=60 * 60,
              volumes={"/ckpt": ckpt_vol}, secrets=[hf_secret])
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
    benchmark_remote.remote(system, run, split, beam)


@app.local_entrypoint()
def latency(systems: str = "ours,nllb", run: str = "base", out: str = "results/latency"):
    import json
    from pathlib import Path

    calls = [(s, d, fn.spawn(s, run)) for s in systems.split(",")
             for d, fn in (("cpu", latency_cpu), ("cuda", latency_gpu))]
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
def backtranslate(run: str = "base", mono: str = "mono/uz.txt", lang: str = "uz",
                  out: str = "stores/bt_uz", limit: int = 0):
    backtranslate_remote.remote(run, mono, lang, out, limit)
