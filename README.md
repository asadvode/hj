# hj

This workspace contains a lightweight Streamlit talking-avatar prototype for CPU-only environments.

## What it does
- Provides a simple web UI for generating a portrait and creating a talking-head video.
- Uses NVIDIA NIM endpoints when API keys are available.
- Falls back to a local placeholder image and a simple video render if the heavyweight Wav2Lip model is unavailable or the NVIDIA services are overloaded.

## Run locally
1. Install dependencies:
   ```bash
   python -m pip install --user -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run webui_orchestrator.py --server.port 8501 --server.address 0.0.0.0
   ```

## Notes
- Keep prompts short when using NVIDIA services because the endpoints can return overload errors under high traffic.
- The app is designed to degrade gracefully and still produce a basic output when remote services are unavailable.
