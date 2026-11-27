import pytest

pytest.importorskip("fastapi")  # the "serve" extra

from fastapi.testclient import TestClient

from src.demo_api import create_app


class _FakeTranslator:
    def translate(self, text, target, beam=4, script="new"):
        return f"{target}:{script}:{beam}:{text}"

    def trace(self, text, target, beam=4, script="new"):
        return {"output": self.translate(text, target, beam, script), "steps": []}


@pytest.fixture
def client(tmp_path):
    (tmp_path / "index.html").write_text("<h1>demo</h1>", encoding="utf-8")
    with TestClient(create_app(_FakeTranslator(), tmp_path)) as c:
        yield c


def test_translate_returns_output_trace_and_time(client):
    r = client.post("/api/translate", json={"text": "hello", "target": "uz"})
    assert r.status_code == 200
    body = r.json()
    assert body["output"] == "uz:new:4:hello"
    assert body["trace"]["output"] == body["output"]
    assert body["seconds"] >= 0


def test_translate_passes_options(client):
    r = client.post("/api/translate",
                    json={"text": "salom", "target": "en", "beam": 1, "script": "old"})
    assert r.json()["output"] == "en:old:1:salom"


@pytest.mark.parametrize("body", [
    {"text": "", "target": "uz"},
    {"text": "   ", "target": "uz"},
    {"text": "hi", "target": "ru"},
    {"text": "hi", "target": "uz", "beam": 9},
    {"text": "x" * 2001, "target": "uz"},
])
def test_invalid_requests(client, body):
    assert client.post("/api/translate", json=body).status_code == 422


@pytest.mark.parametrize("text,target", [
    ("Китоб ўқиш фойдали.", "en"), ("The weather is nice.", "uz"), ("12345", None),
])
def test_detect(client, text, target):
    assert client.post("/api/detect", json={"text": text}).json() == {"target": target}


def test_serves_the_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "demo" in r.text
