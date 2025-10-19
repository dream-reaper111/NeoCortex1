# __main__.py
# -*- coding: utf-8 -*-
"""
Entry point for PineMimic Trader server
Run with:  python -m tensorflowtest
"""

import os
import json
from dotenv import load_dotenv

import uvicorn
from server import app, run_preflight  # import your FastAPI app and preflight

# Load environment variables
load_dotenv(override=False)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

if __name__ == "__main__":
    # Print preflight report
    print(json.dumps(run_preflight(expected_port=API_PORT), indent=2))

    # Launch Uvicorn
    uvicorn.run(
        app,                # pass the app object directly (no string needed)
        host=API_HOST,
        port=API_PORT,
        reload=False        # True if you want hot reload for dev
    )
# __main__.py
# -*- coding: utf-8 -*-
"""
Entry point for PineMimic Trader server
Run with:  python -m tensorflowtest
"""

import os
import json
from dotenv import load_dotenv

import uvicorn
from server import app, run_preflight  # import your FastAPI app and preflight

# Load environment variables
load_dotenv(override=False)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

if __name__ == "__main__":
    # Print preflight report
    print(json.dumps(run_preflight(expected_port=API_PORT), indent=2))

    # Launch Uvicorn
    uvicorn.run(
        app,                # pass the app object directly (no string needed)
        host=API_HOST,
        port=API_PORT,
        reload=False        # True if you want hot reload for dev
    )
