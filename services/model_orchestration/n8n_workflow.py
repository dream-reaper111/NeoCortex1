"""Lightweight client utilities for triggering n8n workflows.

These helpers deliberately avoid third-party SDKs so they can operate in the
same constrained environments as the rest of the `services` package.  They
wrap the n8n REST API so that Celery tasks or FastAPI routes can kick off
automations without shelling out to the n8n CLI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests


_REQUEST_EXCEPTION = getattr(requests, "RequestException", None)
if _REQUEST_EXCEPTION is None:
    _REQUEST_EXCEPTION = getattr(
        getattr(requests, "exceptions", object()), "RequestException", Exception
    )


class N8nWorkflowError(RuntimeError):
    """Raised when an n8n workflow invocation fails."""


@dataclass(frozen=True)
class N8nConfig:
    """Configuration bundle for connecting to an n8n instance."""

    base_url: str
    api_key: str
    timeout: float = 10.0
    poll_interval: float = 1.0
    max_poll_attempts: int = 30

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.max_poll_attempts <= 0:
            raise ValueError("max_poll_attempts must be positive")

    def headers(self) -> Dict[str, str]:
        return {
            "X-N8N-API-KEY": self.api_key,
            "Accept": "application/json",
        }

    def build_url(self, path: str) -> str:
        if not path:
            raise ValueError("path must not be empty")
        return urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))


class N8nWorkflowClient:
    """Simple REST client for n8n workflow executions."""

    def __init__(self, config: N8nConfig):
        self.config = config
        self._sleep = time.sleep

    # ------------------------------------------------------------------
    def trigger(
        self,
        workflow_id: str,
        payload: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        """Execute a workflow and optionally wait for completion.

        Parameters
        ----------
        workflow_id:
            The internal n8n workflow identifier.
        payload:
            Optional JSON payload that becomes available to the workflow as the
            first item in the execution data.
        wait:
            When ``True`` (default) the call polls ``/executions/<id>`` until
            the run finishes.  When ``False`` the raw response from the
            ``/workflows/<id>/run`` endpoint is returned immediately.
        """

        if not workflow_id:
            raise ValueError("workflow_id is required")

        request_body = {
            "runData": {
                "Manual Input": [
                    {
                        "json": payload or {},
                    }
                ]
            }
        }

        response = self._request(
            "POST",
            f"api/v1/workflows/{workflow_id}/run",
            json=request_body,
        )

        execution_id = response.get("executionId") or response.get("id")
        if not wait or not execution_id:
            return response
        return self._poll_execution(execution_id)

    # ------------------------------------------------------------------
    def get_execution(self, execution_id: str) -> Dict[str, Any]:
        if not execution_id:
            raise ValueError("execution_id is required")
        return self._request("GET", f"api/v1/executions/{execution_id}")

    # ------------------------------------------------------------------
    def _poll_execution(self, execution_id: str) -> Dict[str, Any]:
        for attempt in range(self.config.max_poll_attempts):
            execution = self.get_execution(execution_id)
            payload = execution.get("data") if "data" in execution else execution

            finished = payload.get("finished")
            status = payload.get("status")

            if isinstance(finished, bool) and finished:
                return execution
            if isinstance(status, str) and status.lower() in {"success", "error"}:
                return execution

            self._sleep(self.config.poll_interval)

        raise TimeoutError(
            f"n8n execution {execution_id} did not finish after "
            f"{self.config.max_poll_attempts} attempts"
        )

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = self.config.build_url(path)
        headers = {**self.config.headers(), **kwargs.pop("headers", {})}

        json_payload = kwargs.pop("json", None)
        data_payload = kwargs.pop("data", None)

        try:
            request_callable = getattr(requests, "request", None)
            if request_callable is not None:
                if json_payload is not None:
                    kwargs["json"] = json_payload
                if data_payload is not None:
                    kwargs["data"] = data_payload
                response = request_callable(
                    method,
                    url,
                    headers=headers,
                    timeout=self.config.timeout,
                    **kwargs,
                )
            else:
                if json_payload is not None:
                    data_payload = json.dumps(json_payload).encode("utf-8")
                    headers.setdefault("Content-Type", "application/json")
                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=self.config.timeout,
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url,
                        data=data_payload or b"",
                        headers=headers,
                        timeout=self.config.timeout,
                    )
                else:
                    raise N8nWorkflowError(
                        f"HTTP method {method!r} is not supported by the bundled requests shim"
                    )
            response.raise_for_status()
        except _REQUEST_EXCEPTION as exc:  # pragma: no cover - network errors are wrapped
            raise N8nWorkflowError(f"{method} {url} failed: {exc}") from exc

        if not response.content:
            return {}

        try:
            return response.json()
        except ValueError as exc:
            raise N8nWorkflowError(
                f"{method} {url} returned invalid JSON: {response.text}"
            ) from exc


__all__ = ["N8nConfig", "N8nWorkflowClient", "N8nWorkflowError"]

