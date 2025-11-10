"""Run the dashboard as a standalone FastAPI application."""
from __future__ import annotations

import uvicorn

from . import create_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
