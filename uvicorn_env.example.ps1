# Copy to uvicorn_env.ps1 and edit:
$env:API_HOST = "0.0.0.0"
$env:API_PORT = "8000"
$env:TV_WEBHOOK_SECRET = "super-secret-here"
$env:PAPER = "true"
$env:NGROK_DOMAIN = "tamara-unleavened-nonpromiscuously.ngrok-free.dev"  # set to "" for a random domain
