from __future__ import annotations

import gc
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from transformers import WhisperForConditionalGeneration, WhisperProcessor


BASE_MODEL_ID = os.environ.get("WHISPER_BASE_MODEL", "openai/whisper-small")
FINETUNED_MODEL_ID = os.environ.get(
    "WHISPER_FINETUNED_MODEL",
    str(Path(__file__).resolve().parent.parent / "outputs" / "whisper-small-coral"),
)
TARGET_SAMPLE_RATE = 16000
TASK = "transcribe"


LANGUAGE_ALIASES = {
    "da": "Danish",
    "danish": "Danish",
    "dansk": "Danish",
    "en": "English",
    "english": "English",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    default_language: str | None


MODEL_SPECS = {
    "base": ModelSpec(key="base", model_id=BASE_MODEL_ID, default_language=None),
    "danish_finetuned": ModelSpec(
        key="danish_finetuned",
        model_id=FINETUNED_MODEL_ID,
        default_language="Danish",
    ),
}


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = language.strip().lower()
    if not value:
        return None
    return LANGUAGE_ALIASES.get(value, value)


def language_code_to_name(language_code: str | None) -> str | None:
    if language_code is None:
        return None
    return LANGUAGE_ALIASES.get(language_code.strip().lower(), language_code.strip().lower())


def load_audio_bytes(audio_bytes: bytes) -> tuple[Any, int]:
    try:
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        audio_array, sampling_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)

    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = audio_array.mean(axis=1)

    if sampling_rate != TARGET_SAMPLE_RATE:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sampling_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
        sampling_rate = TARGET_SAMPLE_RATE

    return audio_array, sampling_rate


class WhisperRuntime:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.current_key: str | None = None
        self.model: WhisperForConditionalGeneration | None = None
        self.processor: WhisperProcessor | None = None
        self.lock = Lock()

    def _unload_current(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_model(self, spec: ModelSpec) -> float:
        start = time.perf_counter()
        self._unload_current()

        processor = WhisperProcessor.from_pretrained(spec.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            spec.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        model.eval()

        self.processor = processor
        self.model = model
        self.current_key = spec.key
        return time.perf_counter() - start

    def ensure_model(self, model_key: str) -> tuple[str | None, float]:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {model_key}")

        previous_key = self.current_key
        if self.current_key == model_key and self.model is not None and self.processor is not None:
            return previous_key, 0.0

        transition_time = self._load_model(MODEL_SPECS[model_key])
        return previous_key, transition_time

    def detect_language(self, audio_bytes: bytes) -> tuple[str, str | None, float]:
        previous_key, transition_time = self.ensure_model("base")
        assert self.model is not None
        assert self.processor is not None

        audio_array, sampling_rate = load_audio_bytes(audio_bytes)
        inputs = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)
        lang_ids = self.model.detect_language(input_features=input_features)
        inverse_lang_map = {
            token_id: token for token, token_id in self.model.generation_config.lang_to_id.items()
        }
        lang_token = inverse_lang_map.get(int(lang_ids[0].item()))
        if lang_token is None:
            lang_token = self.processor.tokenizer.convert_ids_to_tokens(int(lang_ids[0].item()))
        detected_code = lang_token.replace("<|", "").replace("|>", "")
        return detected_code, previous_key, transition_time

    @torch.inference_mode()
    def transcribe(self, audio_bytes: bytes, language: str | None) -> dict[str, Any]:
        requested_language = normalize_language(language)
        total_transition = 0.0
        previous_model: str | None = None
        detected_language: str | None = None

        if requested_language == "Danish":
            selected_key = "danish_finetuned"
            transcription_language = "Danish"
        elif requested_language is not None:
            selected_key = "base"
            transcription_language = requested_language
        else:
            detected_language, previous_model, transition = self.detect_language(audio_bytes)
            total_transition += transition
            selected_key = "danish_finetuned" if detected_language == "da" else "base"
            transcription_language = "Danish" if selected_key == "danish_finetuned" else None

        prev, transition = self.ensure_model(selected_key)
        total_transition += transition
        if previous_model is None:
            previous_model = prev

        assert self.model is not None
        assert self.processor is not None

        audio_array, sampling_rate = load_audio_bytes(audio_bytes)
        inputs = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)

        inference_start = time.perf_counter()
        generated_ids = self.model.generate(
            input_features=input_features,
            task=TASK,
            language=transcription_language,
        )
        inference_time = time.perf_counter() - inference_start

        text = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return {
            "text": text,
            "requested_language": requested_language,
            "detected_language": detected_language,
            "detected_language_name": language_code_to_name(detected_language),
            "model_used": selected_key,
            "previous_model": previous_model,
            "transition_time_seconds": round(total_transition, 4),
            "inference_time_seconds": round(inference_time, 4),
        }


runtime = WhisperRuntime()
app = FastAPI(title="Whisper ASR API", version="1.0.0")


def render_home_page() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evil Tape</title>
  <style>
    :root {
      --bg: #120509;
      --panel: rgba(255, 245, 239, 0.95);
      --ink: #1f1115;
      --muted: #7f656e;
      --line: rgba(133, 56, 58, 0.16);
      --brand-a: #8c0f19;
      --brand-b: #d12b2d;
      --accent: #f24b24;
      --accent-soft: rgba(216, 74, 29, 0.24);
      --shadow: 0 32px 90px rgba(0, 0, 0, 0.34);
      --player-bg: #1b1015;
      --player-track: rgba(255, 255, 255, 0.16);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
      color: #f7e7df;
      background:
        radial-gradient(circle at 54% 34%, rgba(255, 122, 52, 0.2), transparent 16%),
        radial-gradient(circle at 72% 72%, rgba(160, 22, 35, 0.18), transparent 18%),
        linear-gradient(180deg, #2a090f 0%, #17070b 48%, var(--bg) 100%);
      padding: 32px 24px;
      overflow-x: hidden;
    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: auto;
      pointer-events: none;
      filter: blur(70px);
      opacity: 0.68;
      z-index: 0;
    }

    body::before {
      width: 300px;
      height: 300px;
      right: 14%;
      top: 20%;
      background: rgba(175, 28, 42, 0.42);
    }

    body::after {
      width: 220px;
      height: 220px;
      left: 8%;
      bottom: 8%;
      background: rgba(255, 129, 56, 0.1);
    }

    .shell {
      position: relative;
      z-index: 1;
      width: min(1180px, 100%);
      min-height: calc(100vh - 64px);
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(320px, 1fr);
      gap: 72px;
      align-items: center;
    }

    .upload-card {
      position: relative;
      width: 100%;
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 34px;
      padding: 24px 28px 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      animation: rise 320ms ease-out;
    }

    @keyframes rise {
      from {
        opacity: 0;
        transform: translateY(18px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .card-close {
      position: absolute;
      top: 20px;
      right: 20px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 245, 241, 0.96);
      color: var(--muted);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
      padding: 9px 14px;
      cursor: pointer;
      transition: transform 0.12s ease, border-color 0.12s ease;
    }

    .card-close:hover {
      transform: translateY(-1px);
      border-color: rgba(255, 255, 255, 0.14);
    }

    .plus-button {
      width: 62px;
      height: 62px;
      margin: 86px auto 20px;
      border: 0;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
      color: white;
      font-size: 44px;
      font-weight: 300;
      display: grid;
      place-items: center;
      box-shadow: 0 20px 40px rgba(140, 15, 25, 0.28);
      cursor: default;
    }

    .upload-title {
      margin: 0;
      text-align: center;
      font-size: clamp(2.75rem, 4vw, 3.8rem);
      line-height: 0.96;
      letter-spacing: -0.06em;
      font-weight: 800;
    }

    .upload-subtitle,
    .recording-duration {
      margin: 10px 0 0;
      text-align: center;
      font-size: clamp(1.2rem, 2vw, 1.7rem);
      color: var(--muted);
    }

    form {
      display: grid;
      gap: 16px;
      margin-top: 28px;
    }

    input[type="file"] {
      display: none;
    }

    .field {
      display: grid;
      gap: 8px;
    }

    .field label {
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #8e7074;
    }

    select {
      width: 100%;
      padding: 16px 20px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 252, 250, 0.96);
      font: inherit;
      font-size: 17px;
      color: var(--ink);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      margin-top: 6px;
      padding: 6px;
      border-radius: 20px;
      background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
      box-shadow: 0 18px 36px rgba(163, 15, 28, 0.14);
    }

    .mini-button,
    .mini-button-muted,
    .submit-button {
      border: 0;
      font: inherit;
      cursor: pointer;
      font-weight: 800;
      transition: transform 0.12s ease, opacity 0.12s ease, background 0.12s ease;
    }

    .mini-button,
    .mini-button-muted {
      border-radius: 16px;
      padding: 16px 18px;
      font-size: 18px;
      text-align: center;
    }

    .mini-button {
      background: white;
      color: var(--ink);
    }

    .mini-button-muted {
      background: transparent;
      color: white;
    }

    .mini-button-muted.recording {
      background: rgba(22, 17, 34, 0.2);
      color: white;
    }

    .language-field {
      margin-top: 2px;
    }

    .recording-note {
      min-height: 1.5em;
      margin: 2px 0 0;
      text-align: center;
      font-size: 15px;
      color: var(--muted);
      font-weight: 600;
    }

    .recording-note.live {
      color: #a30f1c;
    }

    .preview {
      display: none;
    }

    .recorder-panel {
      display: none;
      margin-top: 6px;
    }

    .recorder-panel.visible {
      display: block;
    }

    .player-row {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px 18px;
      background: var(--player-bg);
      color: white;
      border-radius: 20px;
      margin-bottom: 10px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .player-row:last-child {
      margin-bottom: 0;
    }

    .play-toggle {
      width: 0;
      height: 0;
      border-top: 12px solid transparent;
      border-bottom: 12px solid transparent;
      border-left: 19px solid white;
      cursor: pointer;
      flex: 0 0 auto;
    }

    .play-toggle.pause {
      width: 16px;
      height: 20px;
      border: 0;
      background: linear-gradient(to right, white 0 5px, transparent 5px 11px, white 11px 16px);
    }

    .play-toggle.disabled {
      opacity: 0.38;
      cursor: default;
    }

    .timeline-slider,
    .volume-slider {
      -webkit-appearance: none;
      appearance: none;
      height: 8px;
      border-radius: 999px;
      background: var(--player-track);
      outline: none;
    }

    .timeline-slider {
      flex: 1 1 auto;
      min-width: 0;
    }

    .volume-slider {
      width: 92px;
    }

    .timeline-slider::-webkit-slider-thumb,
    .volume-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: white;
      cursor: pointer;
    }

    .timeline-slider::-moz-range-thumb,
    .volume-slider::-moz-range-thumb {
      width: 16px;
      height: 16px;
      border: 0;
      border-radius: 50%;
      background: white;
      cursor: pointer;
    }

    .time-readout {
      min-width: 94px;
      text-align: center;
      font-size: 16px;
      color: rgba(255, 255, 255, 0.88);
    }

    .speaker-icon {
      position: relative;
      width: 18px;
      height: 18px;
      flex: 0 0 auto;
    }

    .speaker-icon::before {
      content: "";
      position: absolute;
      left: 0;
      top: 4px;
      width: 7px;
      height: 10px;
      background: white;
      clip-path: polygon(0 25%, 38% 25%, 72% 0, 72% 100%, 38% 75%, 0 75%);
    }

    .speaker-icon::after {
      content: "";
      position: absolute;
      right: 0;
      top: 3px;
      width: 7px;
      height: 11px;
      border-right: 2px solid white;
      border-radius: 0 14px 14px 0;
    }

    .submit-button {
      width: 100%;
      margin-top: 8px;
      padding: 18px 24px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
      color: white;
      font-size: 24px;
      box-shadow: 0 24px 42px rgba(122, 51, 255, 0.2);
    }

    .mini-button:hover,
    .mini-button-muted:hover,
    .submit-button:hover {
      transform: translateY(-1px);
    }

    .submit-button:disabled {
      opacity: 0.72;
      cursor: wait;
      transform: none;
    }

    .status {
      min-height: 1.5em;
      margin: 2px 0 0;
      text-align: center;
      font-size: 16px;
      color: var(--muted);
    }

    .result {
      display: none;
      margin-top: 4px;
      padding: 18px;
      border-radius: 24px;
      background: rgba(255, 244, 239, 0.92);
      border: 1px solid var(--line);
    }

    .result.visible {
      display: block;
    }

    .transcript {
      margin: 0 0 16px;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 16px;
    }

    .result-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid rgba(163, 15, 28, 0.08);
    }

    .metric span {
      display: block;
      margin-bottom: 6px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #8b7178;
    }

    .metric strong {
      font-size: 14px;
      line-height: 1.35;
      color: var(--ink);
    }

    .story-panel {
      display: grid;
      gap: 24px;
      max-width: 520px;
      padding-right: 12px;
    }

    .story-eyebrow {
      margin: 0;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #d28e83;
    }

    .story-kicker {
      margin: 0;
      font-size: clamp(4rem, 8vw, 6rem);
      line-height: 0.95;
      letter-spacing: -0.06em;
      color: var(--accent);
      text-shadow: 0 12px 30px rgba(242, 75, 36, 0.18);
    }

    .story-copy {
      margin: 0;
      font-size: clamp(1.35rem, 2.4vw, 1.9rem);
      line-height: 1.35;
      max-width: 16ch;
      color: rgba(255, 234, 226, 0.92);
    }

    @media (max-width: 1180px) {
      .shell {
        gap: 42px;
      }
    }

    @media (max-width: 760px) {
      body {
        padding: 22px 16px;
      }

      .shell {
        min-height: auto;
        grid-template-columns: 1fr;
        gap: 28px;
      }

      .upload-card {
        padding: 22px 18px 22px;
        border-radius: 28px;
      }

      .plus-button {
        margin-top: 72px;
      }

      .story-panel {
        padding-right: 0;
      }

      .story-copy {
        max-width: none;
      }

      .result-grid {
        grid-template-columns: 1fr;
      }

      .player-row {
        flex-wrap: wrap;
      }

      .volume-slider {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="upload-card">
      <button class="card-close" type="button" aria-label="Reset form">Reset</button>
      <button class="plus-button" type="button" aria-hidden="true">+</button>

      <h1 id="upload-title" class="upload-title">Upload here</h1>
      <p id="upload-subtitle" class="upload-subtitle">Transcribe now</p>
      <p id="recording-duration" class="recording-duration"></p>

      <form id="transcribe-form">
        <input id="file" name="file" type="file" accept="audio/*" />

        <div class="button-row">
          <button id="choose-file-button" class="mini-button" type="button">Upload</button>
          <button id="record-button" class="mini-button-muted" type="button">Record</button>
        </div>

        <div class="field language-field">
          <label for="language">Language</label>
          <select id="language" name="language">
            <option value="">Auto-detect</option>
            <option value="Danish">Danish</option>
            <option value="English">English</option>
            <option value="German">German</option>
            <option value="French">French</option>
            <option value="Spanish">Spanish</option>
          </select>
        </div>

        <p id="recording-note" class="recording-note">Use Upload or record directly in the browser.</p>
        <audio id="recording-preview" class="preview"></audio>

        <div id="recorder-panel" class="recorder-panel">
          <div class="player-row">
            <div id="play-toggle" class="play-toggle" role="button" tabindex="0" aria-label="Play recording"></div>
            <input id="timeline-slider" class="timeline-slider" type="range" min="0" max="1" step="0.01" value="0" />
            <div id="time-readout" class="time-readout">0:00 / 0:00</div>
            <div class="speaker-icon" aria-hidden="true"></div>
            <input id="volume-slider" class="volume-slider" type="range" min="0" max="1" step="0.05" value="1" />
          </div>
          <div class="player-row">
            <div id="play-toggle-secondary" class="play-toggle" role="button" tabindex="0" aria-label="Play recording duplicate"></div>
            <input id="timeline-slider-secondary" class="timeline-slider" type="range" min="0" max="1" step="0.01" value="0" />
            <div id="time-readout-secondary" class="time-readout">0:00 / 0:00</div>
            <div class="speaker-icon" aria-hidden="true"></div>
            <input id="volume-slider-secondary" class="volume-slider" type="range" min="0" max="1" step="0.05" value="1" />
          </div>
        </div>

        <button id="submit-button" class="submit-button" type="submit">Transcribe</button>
        <p id="status" class="status"></p>

        <section id="result" class="result" aria-live="polite">
          <p id="transcript" class="transcript"></p>
          <div class="result-grid">
            <div class="metric"><span>Model Used</span><strong id="model-used"></strong></div>
            <div class="metric"><span>Detected Language</span><strong id="detected-language"></strong></div>
            <div class="metric"><span>Previous Model</span><strong id="previous-model"></strong></div>
            <div class="metric"><span>Transition Time</span><strong id="transition-time"></strong></div>
            <div class="metric"><span>Inference Time</span><strong id="inference-time"></strong></div>
            <div class="metric"><span>Requested Language</span><strong id="requested-language"></strong></div>
          </div>
        </section>
      </form>
    </section>

    <section class="story-panel">
      <p class="story-eyebrow">Whisper Demo</p>
      <h2 class="story-kicker">Evil Tape</h2>
      <p class="story-copy">
        Upload a file or record directly in the browser. Danish switches into the fine-tuned checkpoint.
        Everything else stays on Whisper Small.
      </p>
    </section>

  </main>

  <script>
    const form = document.getElementById("transcribe-form");
    const fileInput = document.getElementById("file");
    const languageInput = document.getElementById("language");
    const chooseFileButton = document.getElementById("choose-file-button");
    const recordButton = document.getElementById("record-button");
    const submitButton = document.getElementById("submit-button");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const transcriptEl = document.getElementById("transcript");
    const uploadTitle = document.getElementById("upload-title");
    const uploadSubtitle = document.getElementById("upload-subtitle");
    const recordingDuration = document.getElementById("recording-duration");
    const recordingNote = document.getElementById("recording-note");
    const recordingPreview = document.getElementById("recording-preview");
    const recorderPanel = document.getElementById("recorder-panel");
    const closeButton = document.querySelector(".card-close");
    const playToggles = [
      document.getElementById("play-toggle"),
      document.getElementById("play-toggle-secondary"),
    ];
    const timelineSliders = [
      document.getElementById("timeline-slider"),
      document.getElementById("timeline-slider-secondary"),
    ];
    const timeReadouts = [
      document.getElementById("time-readout"),
      document.getElementById("time-readout-secondary"),
    ];
    const volumeSliders = [
      document.getElementById("volume-slider"),
      document.getElementById("volume-slider-secondary"),
    ];
    const metricEls = {
      model_used: document.getElementById("model-used"),
      detected_language: document.getElementById("detected-language"),
      previous_model: document.getElementById("previous-model"),
      transition_time: document.getElementById("transition-time"),
      inference_time: document.getElementById("inference-time"),
      requested_language: document.getElementById("requested-language"),
    };

    let mediaStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordedFile = null;
    let currentObjectUrl = null;
    let recordingStartedAt = 0;
    let durationTimer = null;
    let discardRecording = false;

    const setStatus = (message) => {
      statusEl.textContent = message;
    };

    const formatDuration = (seconds) => {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    const setTimeReadouts = (current, total) => {
      const text = `${formatDuration(current)} / ${formatDuration(total)}`;
      timeReadouts.forEach((node) => {
        node.textContent = text;
      });
    };

    const syncTimeline = () => {
      const total = Number.isFinite(recordingPreview.duration) ? recordingPreview.duration : 0;
      const current = Number.isFinite(recordingPreview.currentTime) ? recordingPreview.currentTime : 0;
      const ratio = total > 0 ? current / total : 0;
      timelineSliders.forEach((slider) => {
        slider.value = String(ratio);
      });
      setTimeReadouts(current, total);
      playToggles.forEach((toggle) => {
        toggle.classList.toggle("pause", !recordingPreview.paused && total > 0);
      });
    };

    const updatePreviewState = () => {
      const hasAudio = Boolean(recordingPreview.src);
      recorderPanel.classList.toggle("visible", hasAudio);
      playToggles.forEach((toggle) => {
        toggle.classList.toggle("disabled", !hasAudio);
      });
      timelineSliders.forEach((slider) => {
        slider.disabled = !hasAudio;
      });
      volumeSliders.forEach((slider) => {
        slider.disabled = !hasAudio;
      });
      if (!hasAudio) {
        setTimeReadouts(0, 0);
      } else {
        syncTimeline();
      }
    };

    const clearObjectUrl = () => {
      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
      }
    };

    const resetAudio = () => {
      clearObjectUrl();
      recordedFile = null;
      recordingPreview.pause();
      recordingPreview.removeAttribute("src");
      recordingPreview.load();
      recordingPreview.hidden = true;
      updatePreviewState();
    };

    const attachAudio = (blob, filename) => {
      clearObjectUrl();
      recordedFile = blob instanceof File ? blob : new File([blob], filename, { type: blob.type || "audio/webm" });
      currentObjectUrl = URL.createObjectURL(recordedFile);
      recordingPreview.src = currentObjectUrl;
      recordingPreview.hidden = false;
      updatePreviewState();
    };

    const setIdleState = () => {
      uploadTitle.textContent = "Upload here";
      uploadSubtitle.textContent = "Transcribe now";
      recordingDuration.textContent = "";
      recordingNote.textContent = "Use Upload or record directly in the browser.";
      recordingNote.classList.remove("live");
      recordButton.textContent = "Record";
      recordButton.classList.remove("mini-button");
      recordButton.classList.remove("recording");
      recordButton.classList.add("mini-button-muted");
    };

    const setRecordingState = () => {
      uploadTitle.textContent = "Recording...";
      uploadSubtitle.textContent = "Press stop when the clip is ready.";
      recordingDuration.textContent = "Duration: 0:00";
      recordingNote.textContent = "Microphone is live.";
      recordingNote.classList.add("live");
      recordButton.textContent = "Stop";
      recordButton.classList.remove("mini-button-muted");
      recordButton.classList.remove("mini-button");
      recordButton.classList.add("mini-button-muted", "recording");
    };

    const setRecordedState = (seconds) => {
      uploadTitle.textContent = "Recording complete";
      uploadSubtitle.textContent = "";
      recordingDuration.textContent = `Duration: ${formatDuration(seconds)}`;
      recordingNote.textContent = "Recording captured. Press Transcribe to send it.";
      recordingNote.classList.remove("live");
      recordButton.textContent = "Record again";
      recordButton.classList.remove("mini-button", "recording");
      recordButton.classList.add("mini-button-muted");
    };

    const stopDurationTimer = () => {
      if (durationTimer) {
        window.clearInterval(durationTimer);
        durationTimer = null;
      }
    };

    const releaseMic = () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
      }
    };

    const stopRecording = (discard = false) => {
      discardRecording = discard;
      stopDurationTimer();
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      } else {
        releaseMic();
      }
    };

    const togglePlayback = async () => {
      if (!recordingPreview.src) {
        return;
      }
      if (recordingPreview.paused) {
        await recordingPreview.play();
      } else {
        recordingPreview.pause();
      }
      syncTimeline();
    };

    chooseFileButton.addEventListener("click", () => {
      fileInput.click();
    });

    fileInput.addEventListener("change", () => {
      const chosenFile = fileInput.files && fileInput.files[0];
      if (!chosenFile) {
        return;
      }

      stopRecording(true);
      attachAudio(chosenFile, chosenFile.name);
      uploadTitle.textContent = "File loaded";
      uploadSubtitle.textContent = chosenFile.name;
      recordingDuration.textContent = "";
      recordingNote.textContent = "Ready to transcribe the uploaded clip.";
      recordingNote.classList.remove("live");
      setStatus("");
    });

    recordButton.addEventListener("click", async () => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
        return;
      }

      try {
        fileInput.value = "";
        resetAudio();
        audioChunks = [];
        discardRecording = false;
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.addEventListener("dataavailable", (event) => {
          if (event.data && event.data.size > 0) {
            audioChunks.push(event.data);
          }
        });
        mediaRecorder.addEventListener(
          "stop",
          () => {
            releaseMic();
            if (discardRecording) {
              discardRecording = false;
              audioChunks = [];
              return;
            }
            const mimeType = mediaRecorder.mimeType || "audio/webm";
            const blob = new Blob(audioChunks, { type: mimeType });
            const durationSeconds = Math.max(1, Math.round((Date.now() - recordingStartedAt) / 1000));
            attachAudio(blob, "evil-tape-recording.webm");
            audioChunks = [];
            setRecordedState(durationSeconds);
          },
          { once: true }
        );

        recordingStartedAt = Date.now();
        setRecordingState();
        durationTimer = window.setInterval(() => {
          const seconds = Math.max(0, Math.round((Date.now() - recordingStartedAt) / 1000));
          recordingDuration.textContent = `Duration: ${formatDuration(seconds)}`;
        }, 250);
        mediaRecorder.start();
        setStatus("");
      } catch (error) {
        stopDurationTimer();
        releaseMic();
        setIdleState();
        setStatus(`Microphone error: ${error.message}`);
      }
    });

    closeButton.addEventListener("click", () => {
      stopRecording(true);
      fileInput.value = "";
      resetAudio();
      setIdleState();
      setStatus("");
      transcriptEl.textContent = "";
      Object.values(metricEls).forEach((node) => {
        node.textContent = "";
      });
      resultEl.classList.remove("visible");
    });

    playToggles.forEach((toggle) => {
      toggle.addEventListener("click", () => {
        togglePlayback().catch((error) => {
          setStatus(`Playback error: ${error.message}`);
        });
      });
      toggle.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          togglePlayback().catch((error) => {
            setStatus(`Playback error: ${error.message}`);
          });
        }
      });
    });

    timelineSliders.forEach((slider) => {
      slider.addEventListener("input", (event) => {
        if (!Number.isFinite(recordingPreview.duration) || recordingPreview.duration <= 0) {
          return;
        }
        const ratio = Number(event.target.value);
        recordingPreview.currentTime = recordingPreview.duration * ratio;
        syncTimeline();
      });
    });

    volumeSliders.forEach((slider) => {
      slider.addEventListener("input", (event) => {
        recordingPreview.volume = Number(event.target.value);
        volumeSliders.forEach((node) => {
          if (node !== event.target) {
            node.value = event.target.value;
          }
        });
      });
    });

    recordingPreview.addEventListener("timeupdate", syncTimeline);
    recordingPreview.addEventListener("loadedmetadata", syncTimeline);
    recordingPreview.addEventListener("play", syncTimeline);
    recordingPreview.addEventListener("pause", syncTimeline);
    recordingPreview.addEventListener("ended", syncTimeline);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const chosenFile = (fileInput.files && fileInput.files[0]) || recordedFile;
      if (!chosenFile) {
        setStatus("Choose an audio file or record a clip first.");
        return;
      }

      const formData = new FormData();
      formData.append("file", chosenFile);
      if (languageInput.value) {
        formData.append("language", languageInput.value);
      }

      submitButton.disabled = true;
      setStatus("Transcribing...");

      try {
        const response = await fetch("/transcribe", {
          method: "POST",
          body: formData,
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Request failed");
        }

        transcriptEl.textContent = payload.transcript || "No transcript returned.";
        metricEls.model_used.textContent = payload.model_used || "n/a";
        metricEls.detected_language.textContent = payload.detected_language || "n/a";
        metricEls.previous_model.textContent = payload.previous_model || "n/a";
        metricEls.transition_time.textContent = payload.model_transition_time_seconds ?? "n/a";
        metricEls.inference_time.textContent = payload.inference_time_seconds ?? "n/a";
        metricEls.requested_language.textContent = payload.requested_language || "Auto-detect";
        resultEl.classList.add("visible");
        setStatus("Done.");
      } catch (error) {
        resultEl.classList.remove("visible");
        setStatus(`Error: ${error.message}`);
      } finally {
        submitButton.disabled = false;
      }
    });

    setIdleState();
    updatePreviewState();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_home_page()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "loaded_model": runtime.current_key,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str | None = Form(None)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="An audio file is required.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    try:
        with runtime.lock:
            return runtime.transcribe(audio_bytes=audio_bytes, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_whisper_api:app", host="0.0.0.0", port=8000, reload=False)
