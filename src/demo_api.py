"""Backend of the demo Space: translation plus a step-by-step trace for the animation.

Works with OnnxTranslator, so the Space needs no PyTorch. The page itself is static files.
"""

import threading
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from src.text import detect_target

Text = Annotated[str, StringConstraints(max_length=2000, strip_whitespace=True, min_length=1)]


class TranslateRequest(BaseModel):
    text: Text
    target: Literal["uz", "en"]
    beam: int = Field(4, ge=1, le=5)
    script: Literal["new", "old"] = "new"


class DetectRequest(BaseModel):
    text: Annotated[str, StringConstraints(max_length=2000)]


def create_app(translator, static_dir: str | Path) -> FastAPI:
    lock = threading.Lock()  # one request on the model at a time; the CPU is small
    app = FastAPI(title="nmt-uz demo", docs_url=None, redoc_url=None)

    @app.post("/api/translate")
    def translate(req: TranslateRequest):
        with lock:
            start = time.perf_counter()
            output = translator.translate(req.text, req.target, beam=req.beam, script=req.script)
            seconds = time.perf_counter() - start
            trace = translator.trace(req.text, req.target, beam=req.beam, script=req.script)
        return {"output": output, "seconds": seconds, "trace": trace}

    @app.post("/api/detect")
    def detect(req: DetectRequest):
        return {"target": detect_target(req.text)}

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
