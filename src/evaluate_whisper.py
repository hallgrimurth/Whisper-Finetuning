from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from load_whisper_model import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME, DEFAULT_TASK
from prepare_whisper_data import (
    DEFAULT_CORAL_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASET_SOURCE,
    DEFAULT_FLEURS_CONFIG,
    load_audio_array,
    prepare_whisper_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base Whisper model and a fine-tuned checkpoint on a held-out split.",
    )
    parser.add_argument("--dataset-source", choices=["samromur", "coral", "fleurs"], default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--coral-config", default=DEFAULT_CORAL_CONFIG)
    parser.add_argument("--fleurs-config", default=DEFAULT_FLEURS_CONFIG)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("hf_cache"))
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs") / "whisper_eval_results.json",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional test subset size for quick evaluation.",
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def load_model_and_processor(model_path: str | Path, language: str, task: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(model_path, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = language
    model.generation_config.task = task
    model.to(device)
    model.eval()

    return model, processor, device, dtype


@torch.inference_mode()
def transcribe_sample(model, processor, device, dtype, audio_source) -> str:
    audio_array, sampling_rate = load_audio_array(audio_source)
    inputs = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device=device, dtype=dtype)

    predicted_ids = model.generate(input_features=input_features)
    prediction = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return prediction.strip()


def evaluate_model(model_path: str | Path, dataset, language: str, task: str) -> dict:
    model, processor, device, dtype = load_model_and_processor(model_path, language, task)

    references = []
    predictions = []
    rows = []

    for example in dataset:
        prediction = transcribe_sample(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            audio_source=example["audio"],
        )
        reference = example["sentence"]

        references.append(reference)
        predictions.append(prediction)
        rows.append(
            {
                "id": example["id"],
                "audio": example["audio"],
                "reference": reference,
                "prediction": prediction,
            }
        )

    return {
        "wer": jiwer.wer(references, predictions),
        "cer": jiwer.cer(references, predictions),
        "predictions": rows,
    }


def main() -> None:
    args = parse_args()
    hf_token = args.hf_token or os.environ.get(args.hf_token_env)

    prepared = prepare_whisper_data(
        model_name=args.base_model,
        language=args.language,
        task=args.task,
        dataset_source=args.dataset_source,
        data_root=args.data_root,
        coral_config=args.coral_config,
        fleurs_config=args.fleurs_config,
        hf_token=hf_token,
        hf_cache_dir=args.hf_cache_dir,
    )
    test_dataset = prepared.raw_datasets["test"]
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(min(len(test_dataset), args.max_test_samples)))

    base_results = evaluate_model(
        model_path=args.base_model,
        dataset=test_dataset,
        language=args.language,
        task=args.task,
    )
    finetuned_results = evaluate_model(
        model_path=args.checkpoint_path,
        dataset=test_dataset,
        language=args.language,
        task=args.task,
    )

    results = {
        "base_model": args.base_model,
        "checkpoint_path": str(args.checkpoint_path),
        "num_test_samples": len(test_dataset),
        "base": {
            "wer": base_results["wer"],
            "cer": base_results["cer"],
        },
        "finetuned": {
            "wer": finetuned_results["wer"],
            "cer": finetuned_results["cer"],
        },
        "predictions": {
            "base": base_results["predictions"],
            "finetuned": finetuned_results["predictions"],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Evaluation complete.")
    print(json.dumps(results["base"], indent=2))
    print(json.dumps(results["finetuned"], indent=2))
    print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
