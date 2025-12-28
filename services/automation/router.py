"""FastAPI router bridging NeoCortex with the n8n automation layer."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi import status

from services.automation.ict_pipeline import ICTPipelineConfig, build_ict_payload, coerce_config, parse_bars
automation = APIRouter(prefix="/automation", tags=["automation"])

N8N_URL = os.getenv("N8N_URL", "https://localhost:5678")
N8N_KEY = os.getenv("N8N_KEY", "")
N8N_BASIC_AUTH_USER = os.getenv("N8N_BASIC_AUTH_USER", "neo")
N8N_PASS = os.getenv("N8N_PASS", "")
AUDIT_DB_PATH = Path(
    os.getenv("N8N_AUDIT_DB", "/home/neo/repo/neo/trin/NeoCortex1/logs/n8n_audit.db")
)


def _ensure_audit_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS n8n_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            payload TEXT,
            response_status INTEGER,
            response_body TEXT,
            error TEXT,
            created_at TEXT NOT NULL
        )
        """
    )


def _basic_auth() -> Optional[httpx.Auth]:
    if N8N_PASS:
        return httpx.BasicAuth(N8N_BASIC_AUTH_USER, N8N_PASS)
    return None


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if N8N_KEY:
        headers["Authorization"] = f"Bearer {N8N_KEY}"
    return headers


def _serialise_payload(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"repr": repr(data)})


def _write_audit(
    event_type: str,
    payload: Any,
    response_status: Optional[int] = None,
    response_body: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUDIT_DB_PATH) as connection:
        _ensure_audit_schema(connection)
        connection.execute(
            """
            INSERT INTO n8n_audit (
                event_type,
                payload,
                response_status,
                response_body,
                error,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                _serialise_payload(payload),
                response_status,
                response_body,
                error,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()


async def _audit(
    event_type: str,
    payload: Any,
    response_status: Optional[int] = None,
    response_body: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    await asyncio.to_thread(
        _write_audit,
        event_type,
        payload,
        response_status,
        response_body,
        error,
    )


@automation.post("/start")
async def start_workflow(req: Request) -> Dict[str, Any]:
    data = await req.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{N8N_URL.rstrip('/')}/webhook/start",
                json=data,
                headers=_headers(),
                auth=_basic_auth(),
            )
        await _audit(
            "start",
            data,
            response_status=response.status_code,
            response_body=response.text,
        )
    except httpx.HTTPError as exc:  # pragma: no cover - network failure handling
        await _audit("start_error", data, error=str(exc))
        raise HTTPException(status_code=502, detail="Failed to reach n8n webhook") from exc

    return {"status": "started", "response": response.text}


@automation.get("/status")
async def workflow_status() -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{N8N_URL.rstrip('/')}/rest/workflows",
                headers=_headers(),
                auth=_basic_auth(),
            )
        response.raise_for_status()
        body: Dict[str, Any] = response.json()
        await _audit(
            "status",
            payload={},
            response_status=response.status_code,
            response_body=json.dumps(body, ensure_ascii=False),
        )
        return body
    except httpx.HTTPError as exc:  # pragma: no cover - network failure handling
        await _audit("status_error", payload={}, error=str(exc))
        raise HTTPException(status_code=502, detail="Failed to query n8n workflows") from exc


@automation.post("/ict-signals", status_code=status.HTTP_201_CREATED)
async def ict_signals(payload: Dict[str, Any]) -> Dict[str, Any]:
    bars_raw = payload.get("bars")
    if not isinstance(bars_raw, list):
        raise HTTPException(status_code=400, detail="bars must be a list of candle dictionaries")
    config = coerce_config(payload.get("config"))
    include_equations = bool(payload.get("include_equations", False))
    bars = parse_bars(bars_raw)
    metrics = build_ict_payload(bars, config, include_equations=include_equations)
    await _audit(
        "ict_signals",
        payload={
            "bars": len(bars),
            "config": payload.get("config", {}),
            "include_equations": include_equations,
        },
        response_status=201,
        response_body=_serialise_payload(metrics),
    )
    return {"ok": True, "metrics": metrics}
