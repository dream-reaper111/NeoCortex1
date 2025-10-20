# -*- coding: utf-8 -*-
"""Entrypoint for running the FastAPI server via ``python -m``."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import uvicorn

from server import app, run_preflight


load_dotenv(override=False)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


if __name__ == "__main__":
    # Emit the preflight diagnostic report before starting the server.
    print(json.dumps(run_preflight(), indent=2))

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
