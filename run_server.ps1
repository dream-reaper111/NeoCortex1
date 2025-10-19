$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)  # into deploy/
Set-Location ..

# load env
. .\deploy\uvicorn_env.ps1

# (optional) set Python venv here
# & "V:\venvs\tradingai\Scripts\activate.ps1"

python server.py
