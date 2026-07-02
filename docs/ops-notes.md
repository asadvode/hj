# Operations notes

## NVIDIA endpoint caution
The NVIDIA inference endpoints are shared and can become overloaded. Short prompts and small payloads are more likely to succeed than long or complex requests.

## Recovery strategy
- If image generation fails, the app falls back to a placeholder portrait.
- If text generation fails, the app uses a local fallback phrase.
- If the Wav2Lip ONNX model is absent, the app still produces a simple video file so the UI remains usable.

## Verification checklist
- Config file parses successfully.
- The Streamlit entry point launches on port 8501.
- The app can render the UI even without remote NVIDIA credentials.
