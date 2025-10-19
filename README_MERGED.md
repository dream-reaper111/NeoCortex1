
# Neo AI — Full Merge (TensorFlow project + Enduser + ngrok)

This zip includes the **entire TensorFlow project** (minus heavy virtualenvs/caches and binary blobs) merged with:
- End User App served at `/enduser`
- VS Code debug configs
- ngrok runner
- `.env` for Alpaca paper + ngrok

## Run
```
python -m pip install -r requirements.txt
# Then in VS Code: choose a Run config
# Or:
python -m tensorflowtest   # if __main__ exists and runs the server
python tensorflowtest/run_with_ngrok.py
```
Copied code files: ~80
Skipped (dir-filter): 7741, (binary-ext): 0, (size>12MB): 0
