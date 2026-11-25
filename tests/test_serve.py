import pytest
from fastapi.testclient import TestClient

from src.serve import create_app


class _EchoTranslator:
    name = "echo"
    device = "cpu"

    def translate(self, texts, target, beam=4, script="new"):
        return [f"{target}:{t}" for t in texts]


@pytest.fixture
def client():
    with TestClient(create_app(_EchoTranslator())) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "echo", "device": "cpu"}


def test_translate_string_and_list(client):
    r = client.post("/translate", json={"text": "hello", "target": "uz"})
    assert r.status_code == 200
    assert r.json()["translations"] == ["uz:hello"]
    r = client.post("/translate", json={"text": ["a", "b"], "target": "en", "beam": 1})
    assert r.json()["translations"] == ["en:a", "en:b"]
    assert r.json()["model"] == "echo"


@pytest.mark.parametrize("body", [
    {"text": "hi", "target": "ru"},
    {"text": "hi", "target": "uz", "beam": 9},
    {"text": "hi", "target": "uz", "script": "cyrillic"},
    {"text": ["x"] * 65, "target": "uz"},
    {"text": "x" * 2001, "target": "uz"},
    {"target": "uz"},
])
def test_invalid_requests(client, body):
    assert client.post("/translate", json=body).status_code == 422
