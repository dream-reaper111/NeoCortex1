import json
from typing import Any, Dict, List

import pytest

import services.model_orchestration.n8n_workflow as n8n_module
from services.model_orchestration.n8n_workflow import (
    N8nConfig,
    N8nWorkflowClient,
    N8nWorkflowError,
)


class DummyResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers: Dict[str, str] = {}
        self._text = json.dumps(payload)

    @property
    def content(self) -> bytes:
        return self._text.encode("utf-8")

    @property
    def text(self) -> str:
        return self._text

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise n8n_module.requests.HTTPError(f"HTTP {self.status_code}")


def test_config_validates_inputs():
    with pytest.raises(ValueError):
        N8nConfig(base_url="", api_key="token")
    with pytest.raises(ValueError):
        N8nConfig(base_url="http://localhost", api_key="")
    with pytest.raises(ValueError):
        N8nConfig(base_url="http://localhost", api_key="token", timeout=0)


def test_trigger_and_poll_success(monkeypatch):
    requests_log: List[Dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any):
        requests_log.append({"method": method, "url": url, **kwargs})
        if method == "POST":
            return DummyResponse({"executionId": "123"})
        return DummyResponse({"data": {"finished": True, "status": "success", "data": {"result": 42}}})

    monkeypatch.setattr(n8n_module.requests, "request", fake_request, raising=False)

    config = N8nConfig(base_url="http://n8n.local", api_key="secret", poll_interval=0.01)
    client = N8nWorkflowClient(config)
    client._sleep = lambda _: None

    result = client.trigger("7", payload={"foo": "bar"})

    assert requests_log[0]["method"] == "POST"
    assert requests_log[1]["method"] == "GET"
    assert result["data"]["finished"] is True


def test_trigger_timeout(monkeypatch):
    responses = [
        DummyResponse({"executionId": "55"}),
        DummyResponse({"data": {"finished": False, "status": "running"}}),
        DummyResponse({"data": {"finished": False, "status": "running"}}),
    ]

    def fake_request(method: str, url: str, **kwargs: Any):
        return responses.pop(0)

    monkeypatch.setattr(n8n_module.requests, "request", fake_request, raising=False)

    config = N8nConfig(
        base_url="http://n8n.local",
        api_key="secret",
        poll_interval=0.001,
        max_poll_attempts=2,
    )
    client = N8nWorkflowClient(config)
    client._sleep = lambda _: None

    with pytest.raises(TimeoutError):
        client.trigger("99")


def test_request_wraps_errors(monkeypatch):
    def failing_request(method: str, url: str, **kwargs: Any):
        raise n8n_module.requests.exceptions.RequestException("boom")

    monkeypatch.setattr(n8n_module.requests, "request", failing_request, raising=False)

    config = N8nConfig(base_url="http://n8n.local", api_key="secret")
    client = N8nWorkflowClient(config)

    with pytest.raises(N8nWorkflowError):
        client.trigger("1", wait=False)
