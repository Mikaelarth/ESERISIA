import importlib
import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def _build_client(monkeypatch):
    monkeypatch.setenv("ESERISIA_API_TOKENS", "test-token")
    monkeypatch.setenv("ESERISIA_STRICT_MODE", "1")
    api_path = Path(__file__).resolve().parents[1] / "api" / "main.py"
    spec = importlib.util.spec_from_file_location("eserisia_api_main", api_path)
    api_main = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(api_main)
    return TestClient(api_main.app)


def test_root_contract_is_neutral(monkeypatch):
    client = _build_client(monkeypatch)
    resp = client.get("/")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["message"] == "ESERISIA AI API"
    assert "SOTA" not in " ".join(payload["capabilities"])


def test_chat_requires_token_and_returns_structured_model_info(monkeypatch):
    client = _build_client(monkeypatch)
    authorized = client.post(
        "/chat",
        headers={"Authorization": "Bearer test-token"},
        json={"message": "bonjour", "stream": False},
    )
    assert authorized.status_code == 200
    payload = authorized.json()
    assert payload["model_info"]["model"] == "eserisia-service"
    assert payload["model_info"]["strict_mode"] is True
