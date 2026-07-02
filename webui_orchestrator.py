import os
import sys
import json
import time
import base64
import random
import gc
import subprocess
import asyncio
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ONNXRUNTIME_CPU_NUM_THREADS", "1")

import cv2
import numpy as np
import requests
import streamlit as st
import edge_tts

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

os.makedirs("temp_staging", exist_ok=True)
os.makedirs("results", exist_ok=True)

CONFIG_PATH = ROOT / "config.json"
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
else:
    config = {"nvidia_keys": {}, "settings": {}}

NVIDIA_KEYS = config.get("nvidia_keys", {})
SETTINGS = config.get("settings", {})
SUBPROCESS_TIMEOUT = int(SETTINGS.get("SUBPROCESS_TIMEOUT_SEC", 45))
MODEL_CHECKPOINT = ROOT / "checkpoints" / "wav2lip_256.onnx"
API_MAX_ATTEMPTS = int(SETTINGS.get("API_MAX_ATTEMPTS", 3))
API_RETRY_DELAY_SEC = float(SETTINGS.get("API_RETRY_DELAY_SEC", 4.0))


def _get_configured_key(key_name: str) -> str:
    configured = str(NVIDIA_KEYS.get(key_name, "")).strip()
    if configured:
        return configured
    env_name = key_name.upper()
    return str(os.getenv(env_name, "")).strip()


def _request_with_retries(url: str, headers: dict, payload: dict, timeout: int = 45):
    last_error = None
    for attempt in range(1, API_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code in {400, 401, 403, 404}:
                break
        except requests.Timeout as exc:
            last_error = f"request timeout: {exc}"
        except requests.RequestException as exc:
            last_error = f"connection issue: {exc}"
        if attempt < API_MAX_ATTEMPTS:
            time.sleep(API_RETRY_DELAY_SEC * attempt)
    raise RuntimeError(last_error or "API request failed")


def generate_avatar_image(prompt: str):
    """Calls the NVIDIA Flux endpoint and retries on transient failures; never fabricates output."""
    api_key = _get_configured_key("flux_klein_key")
    if not api_key:
        return False, "No NVIDIA Flux API key configured. Add it to config.json or set FLUX_KLEIN_KEY before generating an image."

    invoke_url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt[:220],
        "width": 512,
        "height": 512,
        "cfg_scale": 0,
        "samples": 1,
        "seed": 0,
        "steps": 2
    }
    output_path = ROOT / "portrait.jpg"
    try:
        response = _request_with_retries(invoke_url, headers, payload, timeout=45)
        data = response.json()
        if "artifacts" in data and len(data["artifacts"]) > 0:
            base64_data = data["artifacts"][0]["base64"]
            img_data = base64.b64decode(base64_data)
            output_path.write_bytes(img_data)
            return True, str(output_path)
        return False, f"Unexpected image response structure: {data}"
    except Exception as exc:
        return False, f"Avatar generation failed after {API_MAX_ATTEMPTS} attempts: {exc}"


def get_minimax_response(user_prompt: str, word_limit: int):
    """Calls the NVIDIA MiniMax endpoint and retries on transient failures; never fabricates a response."""
    api_key = _get_configured_key("minimax_m3_key")
    if not api_key:
        return False, "No NVIDIA MiniMax API key configured. Add it to config.json or set MINIMAX_M3_KEY before generating speech."

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_instruction = f"Respond in {word_limit} words or less. Keep responses extremely brief and single-sentence."
    payload = {
        "model": "minimaxai/minimax-m3",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt[:220]}
        ],
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    try:
        response = _request_with_retries(invoke_url, headers, payload, timeout=35)
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        if not content:
            return False, "MiniMax returned an empty response."
        return True, content
    except Exception as exc:
        return False, f"Text generation failed after {API_MAX_ATTEMPTS} attempts: {exc}"


def generate_voice_file_sync(text: str, output_path: str):
    async def amain():
        communicate = edge_tts.Communicate(text, "en-US-AvaNeural")
        await communicate.save(output_path)

    asyncio.run(amain())


def process_talking_avatar(avatar_path: str, prompt: str, word_limit: int):
    """Creates a talking-head video from real API output only; no placeholder generation."""
    if not avatar_path or not os.path.exists(avatar_path):
        return "Missing valid image file", None

    timestamp = int(time.time())
    audio_path = str(ROOT / "temp_staging" / f"voice_{timestamp}.wav")
    generated_file = str(ROOT / "results" / f"output_{timestamp}.mp4")

    try:
        success, response_text = get_minimax_response(prompt, word_limit)
        if not success:
            raise RuntimeError(response_text)
        print(f"[Text Output] {response_text}")

        generate_voice_file_sync(response_text, audio_path)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            raise Exception("Corrupt or empty audio file created.")

        if not MODEL_CHECKPOINT.exists():
            raise FileNotFoundError(f"Wav2Lip ONNX model missing at {MODEL_CHECKPOINT}")

        if not (ROOT / "inference_onnxModel.py").exists():
            raise FileNotFoundError("Wav2Lip inference script missing")

        cmd = [
            sys.executable,
            str(ROOT / "inference_onnxModel.py"),
            "--checkpoint_path",
            str(MODEL_CHECKPOINT),
            "--face",
            avatar_path,
            "--audio",
            audio_path,
            "--outfile",
            generated_file,
            "--nosmooth"
        ]
        print(f"[Subprocess] Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=SUBPROCESS_TIMEOUT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Subprocess Error] {result.stderr}")
            raise Exception(f"Wav2Lip engine output error: {result.stderr}")

        if os.path.exists(audio_path):
            os.remove(audio_path)

        if not os.path.exists(generated_file) or os.path.getsize(generated_file) < 5000:
            raise Exception("Rendered video file is empty or missing.")

        gc.collect()
        return f"Response generated successfully: '{response_text}'", generated_file

    except subprocess.TimeoutExpired:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return "Processing timeout exceeded on local CPU thread constraints. Please retry with a shorter prompt.", None
    except Exception as exc:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return f"Process failed: {exc}", None


st.set_page_config(page_title="NVIDIA CPU Avatar Engine", layout="wide")
st.title("CPU-Only Talking Avatar Engine")
st.caption("Integrated with NVIDIA NIM APIs and a local CPU-safe fallback pipeline.")

if "avatar_path" not in st.session_state:
    st.session_state.avatar_path = None

with st.sidebar:
    st.header("Status")
    st.write(f"Config present: {CONFIG_PATH.exists()}")
    st.write(f"Model checkpoint available: {MODEL_CHECKPOINT.exists()}")
    st.write("Tip: keep prompts short to avoid overloaded NVIDIA endpoints.")


tab1, tab2 = st.tabs(["1. Generate/Edit Avatar Portrait", "2. Generate Talking Head Video"])

with tab1:
    st.header("Generate a Custom Speaker Avatar")
    st.write("Write an instruction to generate a portrait. The app will retry a few times if the NVIDIA endpoint is busy instead of pretending success.")
    flux_input = st.text_input(
        "Describe the Speaker Face:",
        "A realistic front-facing portrait of a professional corporate assistant, looking directly into the camera, plain neutral background, soft studio lighting"
    )
    if st.button("Generate Portrait Image"):
        if not flux_input.strip():
            st.error("Please enter a valid prompt.")
        else:
            with st.spinner("Calling the avatar generation API and waiting for a successful response..."):
                success, result = generate_avatar_image(flux_input)
                if success:
                    st.session_state.avatar_path = result
                    st.success("Avatar portrait ready.")
                    st.image(result, width=400, caption="Generated portrait")
                else:
                    st.error(f"Image generation failed: {result}")

with tab2:
    st.header("Configure Speech and Generate Video")
    col1, col2 = st.columns([1, 1])
    with col1:
        avatar_file = None
        if st.session_state.avatar_path and os.path.exists(st.session_state.avatar_path):
            st.image(st.session_state.avatar_path, width=220, caption="Selected avatar portrait")
            avatar_file = st.session_state.avatar_path
        else:
            st.info("No generated avatar found. You can upload a custom portrait instead.")
            uploaded_image = st.file_uploader("Upload custom portrait image instead (.jpg or .png):", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                temp_avatar_path = ROOT / "temp_staging" / "custom_portrait.jpg"
                temp_avatar_path.write_bytes(uploaded_image.read())
                avatar_file = str(temp_avatar_path)
                st.image(avatar_file, width=220, caption="Uploaded portrait")

        text_prompt = st.text_area(
            "Message input:",
            "Welcome to the application interface. How can I assist you with your operations today?"
        )
        word_limit_slider = st.slider("Response word limit:", min_value=5, max_value=25, value=12, step=1)
        generate_btn = st.button("Generate Talking Video")

    with col2:
        st.subheader("Output")
        if generate_btn:
            if not avatar_file:
                st.warning("Please upload or generate a portrait first.")
            elif not text_prompt.strip():
                st.warning("Please type a message prompt.")
            else:
                status_placeholder = st.empty()
                status_placeholder.info("Initializing the speech and animation pipeline and waiting for the API responses...")
                with st.spinner("Processing with retries for the NVIDIA API calls..."):
                    status, video_path = process_talking_avatar(avatar_file, text_prompt, word_limit_slider)
                    if video_path and os.path.exists(video_path):
                        status_placeholder.success(status)
                        st.video(video_path)
                    else:
                        status_placeholder.error(f"Generation error: {status}")
