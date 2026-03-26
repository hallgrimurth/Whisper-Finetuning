from __future__ import annotations

import argparse
import os
from pathlib import Path

import jiwer
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from load_whisper_model import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME, DEFAULT_TASK, load_whisper_bundle
from prepare_whisper_data import (
    DEFAULT_CORAL_CONFIG,
    DEFAULT_CORAL_LANGUAGE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASET_SOURCE,
    DEFAULT_FLEURS_CONFIG,
    DEFAULT_FLEURS_LANGUAGE_DEFAULTS,
    prepare_whisper_data,
)


DEFAULT_OUTPUT_DIR = Path("outputs") / "whisper-small-coral"
DEFAULT_RUN_NAME = "whisper-small-coral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on CORAL, FLEURS, or Samromur.")
    parser.add_argument("--dataset-source", choices=["samromur", "coral", "fleurs"], default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--coral-config", default=DEFAULT_CORAL_CONFIG)
    parser.add_argument("--fleurs-config", default=DEFAULT_FLEURS_CONFIG)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    return parser.parse_args()


def maybe_select_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))


def resolve_language(dataset_source: str, language: str | None) -> str:
    if language:
        return language
    if dataset_source == "coral":
        return DEFAULT_CORAL_LANGUAGE
    if dataset_source == "fleurs":
        raise ValueError("Use resolve_fleurs_language for dataset_source='fleurs'.")
    return DEFAULT_LANGUAGE


def resolve_fleurs_language(config_name: str, language: str | None) -> str:
    if language:
        return language
    if config_name in DEFAULT_FLEURS_LANGUAGE_DEFAULTS:
        return DEFAULT_FLEURS_LANGUAGE_DEFAULTS[config_name]
    raise ValueError(
        f"No default language mapping for FLEURS config '{config_name}'. Pass --language explicitly."
    )


def build_compute_metrics(processor):
    def compute_metrics(eval_pred):
        prediction_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": jiwer.wer(references, predictions)}

    return compute_metrics


def resolve_reporting_target(report_to: str) -> list[str]:
    return [] if report_to == "none" else [report_to]


def main() -> None:
    args = parse_args()
    logging_dir = args.logging_dir or (args.output_dir / "logs")
    if args.dataset_source == "fleurs":
        language = resolve_fleurs_language(args.fleurs_config, args.language)
    else:
        language = resolve_language(args.dataset_source, args.language)
    hf_token = args.hf_token or os.environ.get(args.hf_token_env)

    prepared = prepare_whisper_data(
        model_name=args.model_name,
        language=language,
        task=args.task,
        dataset_source=args.dataset_source,
        data_root=args.data_root,
        coral_config=args.coral_config,
        fleurs_config=args.fleurs_config,
        hf_token=hf_token,
        hf_cache_dir=args.hf_cache_dir,
    )
    bundle = load_whisper_bundle(
        model_name=args.model_name,
        language=language,
        task=args.task,
    )
    prepared.collator.feature_dtype = bundle.dtype

    train_dataset = maybe_select_subset(prepared.raw_datasets["train"], args.max_train_samples)
    eval_split = "dev" if "dev" in prepared.raw_datasets else "validation"
    eval_dataset = maybe_select_subset(prepared.raw_datasets[eval_split], args.max_eval_samples)

    bundle.model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        logging_dir=str(logging_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="no",
        logging_strategy="steps",
        generation_max_length=225,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=resolve_reporting_target(args.report_to),
    )

    trainer = Seq2SeqTrainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=prepared.collator,
        compute_metrics=build_compute_metrics(bundle.processor),
        tokenizer=bundle.processor,
    )

    trainer.train()
    metrics = trainer.evaluate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    bundle.processor.save_pretrained(str(args.output_dir))

    print("Training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
