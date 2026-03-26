from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DEFAULT_MODEL_NAME = "openai/whisper-small"
DEFAULT_LANGUAGE = "Danish"
DEFAULT_TASK = "transcribe"


@dataclass
class WhisperBundle:
    model: WhisperForConditionalGeneration
    processor: WhisperProcessor
    device: torch.device
    dtype: torch.dtype


def load_whisper_bundle(
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = DEFAULT_LANGUAGE,
    task: str = DEFAULT_TASK,
) -> WhisperBundle:
    """Load a Whisper model plus processor onto the active device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = language
    model.generation_config.task = task
    model.to(device)

    return WhisperBundle(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    bundle = load_whisper_bundle()
    print(f"Loaded {DEFAULT_MODEL_NAME} on {bundle.device} with dtype={bundle.dtype}.")
