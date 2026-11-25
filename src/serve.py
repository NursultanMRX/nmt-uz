"""HTTP API around Translator.

    MODEL_ID=nickoo004/nmt-uz-76m uvicorn src.serve:app --host 0.0.0.0 --port 8000

MODEL_ID is a Hugging Face repo id or a local release directory; DEVICE is cpu, cuda or auto.
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

DEFAULT_MODEL = "nickoo004/nmt-uz-76m"

Text = Annotated[str, StringConstraints(max_length=2000)]


class TranslateRequest(BaseModel):
    text: Text | Annotated[list[Text], Field(min_length=1, max_length=64)]
    target: Literal["uz", "en"]
    beam: int = Field(4, ge=1, le=8)
    script: Literal["new", "old"] = "new"


class TranslateResponse(BaseModel):
    translations: list[str]
    model: str
    ms: float


def create_app(translator=None) -> FastAPI:
    state = {"translator": translator}
    lock = threading.Lock()  # one request on the model at a time

    @asynccontextmanager
    async def lifespan(_app):
        if state["translator"] is None:
            from src.inference import Translator

            state["translator"] = Translator.from_pretrained(
                os.environ.get("MODEL_ID", DEFAULT_MODEL),
                device=os.environ.get("DEVICE", "auto"),
            )
        yield

    app = FastAPI(title="nmt-uz", description="English <-> Uzbek translation", lifespan=lifespan)

    @app.get("/health")
    def health():
        tr = state["translator"]
        return {"status": "ok", "model": tr.name, "device": tr.device}

    @app.post("/translate", response_model=TranslateResponse)
    def translate(req: TranslateRequest):
        tr = state["translator"]
        texts = [req.text] if isinstance(req.text, str) else req.text
        start = time.perf_counter()
        with lock:
            out = tr.translate(texts, req.target, beam=req.beam, script=req.script)
        ms = round((time.perf_counter() - start) * 1000, 1)
        return TranslateResponse(translations=out, model=tr.name, ms=ms)

    return app


app = create_app()
