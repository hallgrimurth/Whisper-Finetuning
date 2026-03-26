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
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE


def default_finetuned_model_dir() -> str:
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    preferred = outputs_dir / "whisper-small-coral"
    if preferred.exists():
        return str(preferred)

    for candidate in sorted(outputs_dir.glob("whisper-small-coral*")):
        if candidate.is_dir():
            return str(candidate)

    return str(preferred)


BASE_MODEL_ID = os.environ.get("WHISPER_BASE_MODEL", "openai/whisper-small")
FINETUNED_MODEL_ID = os.environ.get(
    "WHISPER_FINETUNED_MODEL",
    default_finetuned_model_dir(),
)
TARGET_SAMPLE_RATE = 16000
TASK = "transcribe"
LANGUAGE_DETECTION_WINDOW_SECONDS = float(os.environ.get("WHISPER_LANGUAGE_DETECTION_WINDOW_SECONDS", "8"))
LANGUAGE_DETECTION_MAX_WINDOWS = int(os.environ.get("WHISPER_LANGUAGE_DETECTION_MAX_WINDOWS", "3"))
LANGUAGE_DETECTION_MIN_VOICED_SECONDS = float(
    os.environ.get("WHISPER_LANGUAGE_DETECTION_MIN_VOICED_SECONDS", "1.5")
)
LANGUAGE_DETECTION_TRIM_TOP_DB = int(os.environ.get("WHISPER_LANGUAGE_DETECTION_TRIM_TOP_DB", "30"))
DANISH_DETECTION_CONFIDENCE_THRESHOLD = float(
    os.environ.get("WHISPER_DANISH_DETECTION_CONFIDENCE_THRESHOLD", "0.8")
)
DANISH_DETECTION_MARGIN_THRESHOLD = float(
    os.environ.get("WHISPER_DANISH_DETECTION_MARGIN_THRESHOLD", "0.15")
)


WHISPER_LANGUAGE_NAMES = {code: name.title() for code, name in LANGUAGES.items()}
LANGUAGE_ALIASES = {code: display_name for code, display_name in WHISPER_LANGUAGE_NAMES.items()}
LANGUAGE_ALIASES.update(
    {display_name.lower(): display_name for display_name in WHISPER_LANGUAGE_NAMES.values()}
)
LANGUAGE_ALIASES.update(
    {
        alias.lower(): WHISPER_LANGUAGE_NAMES[language_code]
        for alias, language_code in TO_LANGUAGE_CODE.items()
        if language_code in WHISPER_LANGUAGE_NAMES
    }
)
LANGUAGE_ALIASES["dansk"] = "Danish"
SUPPORTED_LANGUAGE_OPTIONS = sorted(set(WHISPER_LANGUAGE_NAMES.values()))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    default_language: str | None


@dataclass(frozen=True)
class LanguageDetectionResult:
    language_code: str | None
    language_name: str | None
    confidence: float
    margin: float
    voiced_seconds: float
    analyzed_segments: int


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


def trim_audio_for_detection(audio_array: Any, sampling_rate: int) -> tuple[np.ndarray, float]:
    normalized = np.asarray(audio_array, dtype=np.float32)
    if normalized.size == 0:
        return normalized, 0.0

    trimmed_audio, _ = librosa.effects.trim(normalized, top_db=LANGUAGE_DETECTION_TRIM_TOP_DB)
    if trimmed_audio.size == 0:
        trimmed_audio = normalized

    voiced_seconds = float(trimmed_audio.shape[0]) / float(sampling_rate) if sampling_rate else 0.0
    return trimmed_audio, voiced_seconds


def build_detection_segments(audio_array: np.ndarray, sampling_rate: int) -> list[np.ndarray]:
    if audio_array.size == 0:
        return []

    segment_length = max(1, int(LANGUAGE_DETECTION_WINDOW_SECONDS * sampling_rate))
    if audio_array.shape[0] <= segment_length:
        return [audio_array]

    max_start = audio_array.shape[0] - segment_length
    segment_count = min(LANGUAGE_DETECTION_MAX_WINDOWS, max(1, int(np.ceil(audio_array.shape[0] / segment_length))))
    start_positions = np.linspace(0, max_start, num=segment_count, dtype=int)
    return [audio_array[start : start + segment_length] for start in start_positions]


def resolve_model_source(model_id: str) -> tuple[str, bool]:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve()), True

    looks_like_local_path = bool(candidate.anchor) or model_id.startswith((".", "~")) or "\\" in model_id
    if looks_like_local_path:
        raise FileNotFoundError(f"Local model path not found: {candidate}")

    return model_id, False


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

        model_source, local_only = resolve_model_source(spec.model_id)
        processor = WhisperProcessor.from_pretrained(model_source, local_files_only=local_only)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
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

    def _language_probabilities(self, audio_array: np.ndarray, sampling_rate: int) -> dict[str, float]:
        assert self.model is not None
        assert self.processor is not None

        inputs = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)

        decoder_input_ids = torch.full(
            (input_features.shape[0], 1),
            self.model.generation_config.decoder_start_token_id,
            device=self.device,
            dtype=torch.long,
        )

        logits = self.model(
            input_features=input_features[:, :, :3000],
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits[:, -1]

        lang_to_id = self.model.generation_config.lang_to_id
        inverse_lang_map = {token_id: token for token, token_id in lang_to_id.items()}
        language_token_ids = torch.tensor(list(lang_to_id.values()), device=logits.device, dtype=torch.long)
        language_logits = logits.index_select(-1, language_token_ids)
        language_probs = torch.softmax(language_logits, dim=-1)[0].detach().cpu().tolist()

        scored_languages: dict[str, float] = {}
        for token_id, probability in zip(language_token_ids.tolist(), language_probs):
            token = inverse_lang_map[token_id]
            language_code = token.replace("<|", "").replace("|>", "")
            scored_languages[language_code] = float(probability)

        return scored_languages

    def should_route_to_danish_model(self, detection: LanguageDetectionResult) -> bool:
        return (
            detection.language_code == "da"
            and detection.voiced_seconds >= LANGUAGE_DETECTION_MIN_VOICED_SECONDS
            and detection.confidence >= DANISH_DETECTION_CONFIDENCE_THRESHOLD
            and detection.margin >= DANISH_DETECTION_MARGIN_THRESHOLD
        )

    @torch.inference_mode()
    def detect_language(self, audio_bytes: bytes) -> tuple[LanguageDetectionResult, str | None, float]:
        previous_key, transition_time = self.ensure_model("base")
        assert self.model is not None
        assert self.processor is not None

        audio_array, sampling_rate = load_audio_bytes(audio_bytes)
        trimmed_audio, voiced_seconds = trim_audio_for_detection(audio_array, sampling_rate)
        segments = build_detection_segments(trimmed_audio, sampling_rate)

        if not segments:
            detection = LanguageDetectionResult(
                language_code=None,
                language_name=None,
                confidence=0.0,
                margin=0.0,
                voiced_seconds=voiced_seconds,
                analyzed_segments=0,
            )
            return detection, previous_key, transition_time

        aggregate_scores: dict[str, float] = {}
        total_weight = 0.0
        for segment in segments:
            segment_probabilities = self._language_probabilities(segment, sampling_rate)
            segment_seconds = max(float(segment.shape[0]) / float(sampling_rate), 1e-3)
            total_weight += segment_seconds
            for language_code, probability in segment_probabilities.items():
                aggregate_scores[language_code] = aggregate_scores.get(language_code, 0.0) + (
                    probability * segment_seconds
                )

        ranked_languages = sorted(
            ((language_code, score / total_weight) for language_code, score in aggregate_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )

        detected_code = ranked_languages[0][0] if ranked_languages else None
        detected_confidence = ranked_languages[0][1] if ranked_languages else 0.0
        runner_up_confidence = ranked_languages[1][1] if len(ranked_languages) > 1 else 0.0
        detection = LanguageDetectionResult(
            language_code=detected_code,
            language_name=language_code_to_name(detected_code),
            confidence=detected_confidence,
            margin=detected_confidence - runner_up_confidence,
            voiced_seconds=voiced_seconds,
            analyzed_segments=len(segments),
        )
        return detection, previous_key, transition_time

    @torch.inference_mode()
    def transcribe(self, audio_bytes: bytes, language: str | None) -> dict[str, Any]:
        requested_language = normalize_language(language)
        total_transition = 0.0
        previous_model: str | None = None
        detected_language: str | None = None
        detection = LanguageDetectionResult(
            language_code=None,
            language_name=None,
            confidence=0.0,
            margin=0.0,
            voiced_seconds=0.0,
            analyzed_segments=0,
        )

        if requested_language == "Danish":
            selected_key = "danish_finetuned"
            transcription_language = "Danish"
        elif requested_language is not None:
            selected_key = "base"
            transcription_language = requested_language
        else:
            detection, previous_model, transition = self.detect_language(audio_bytes)
            total_transition += transition
            detected_language = detection.language_code
            selected_key = "danish_finetuned" if self.should_route_to_danish_model(detection) else "base"
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
            "transcript": text,
            "text": text,
            "requested_language": requested_language,
            "detected_language": detected_language,
            "detected_language_name": language_code_to_name(detected_language),
            "detected_language_confidence": round(detection.confidence, 4) if detected_language else None,
            "detected_language_margin": round(detection.margin, 4) if detected_language else None,
            "voiced_audio_seconds": round(detection.voiced_seconds, 4) if detected_language else None,
            "language_detection_segments": detection.analyzed_segments if detected_language else 0,
            "model_used": selected_key,
            "previous_model": previous_model,
            "model_transition_time_seconds": round(total_transition, 4),
            "transition_time_seconds": round(total_transition, 4),
            "inference_time_seconds": round(inference_time, 4),
        }


runtime = WhisperRuntime()
app = FastAPI(title="Whisper ASR API", version="1.0.0")


def render_home_page() -> str:
    language_options_html = "\n".join(
        f'            <option value="{language_name}">{language_name}</option>'
        for language_name in SUPPORTED_LANGUAGE_OPTIONS
    )
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
      grid-template-columns: minmax(0, 390px) minmax(0, 1fr);
      gap: 56px;
      align-items: center;
    }

    .upload-card {
      position: relative;
      width: 100%;
      max-width: 100%;
      min-width: 0;
      overflow: hidden;
      background: var(--panel);
      color: var(--ink);
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
      min-width: 0;
    }

    form > * {
      min-width: 0;
      max-width: 100%;
    }

    input[type="file"] {
      display: none;
    }

    .field {
      display: grid;
      gap: 8px;
      width: 100%;
      max-width: 100%;
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
      max-width: 100%;
      min-width: 0;
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
      width: 100%;
      max-width: 100%;
      overflow: hidden;
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
      width: 100%;
      max-width: 100%;
    }

    .recorder-panel.visible {
      display: block;
    }

    .player-row {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
      max-width: 100%;
      overflow: hidden;
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
      max-width: 100%;
      margin-top: 8px;
      padding: 18px 24px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
      color: white;
      font-size: 24px;
      box-shadow: 0 24px 42px rgba(140, 15, 25, 0.24);
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
      overflow-wrap: anywhere;
    }

    .result {
      display: none;
      margin-top: 4px;
      width: 100%;
      max-width: 100%;
      overflow: hidden;
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
      width: 100%;
      min-width: 0;
      padding-right: 12px;
      overflow-wrap: anywhere;
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
      font-size: clamp(3.4rem, 7vw, 5.4rem);
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

    @media (max-width: 1500px) {
      .shell {
        min-height: auto;
        grid-template-columns: 1fr;
        gap: 32px;
        max-width: 720px;
      }

      .story-panel {
        padding-right: 0;
      }
    }

    @media (max-width: 760px) {
      body {
        padding: 22px 16px;
      }

      .shell { gap: 28px; }

      .upload-card {
        padding: 22px 18px 22px;
        border-radius: 28px;
      }

      .plus-button {
        margin-top: 72px;
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
{{LANGUAGE_OPTIONS}}
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
    const playToggles = [document.getElementById("play-toggle")];
    const timelineSliders = [document.getElementById("timeline-slider")];
    const timeReadouts = [document.getElementById("time-readout")];
    const volumeSliders = [document.getElementById("volume-slider")];
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

        transcriptEl.textContent = payload.transcript || payload.text || "No transcript returned.";
        metricEls.model_used.textContent = payload.model_used || "n/a";
        metricEls.detected_language.textContent =
          payload.detected_language_name || payload.detected_language || "n/a";
        metricEls.previous_model.textContent = payload.previous_model || "n/a";
        metricEls.transition_time.textContent =
          payload.model_transition_time_seconds ?? payload.transition_time_seconds ?? "n/a";
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
""".replace("{{LANGUAGE_OPTIONS}}", language_options_html)


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
