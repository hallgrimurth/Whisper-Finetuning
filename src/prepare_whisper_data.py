from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import WhisperProcessor

from load_whisper_model import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME, DEFAULT_TASK


DEFAULT_DATA_ROOT = Path("data") / "samromur"
DEFAULT_DATASET_SOURCE = "coral"
DEFAULT_TEXT_COLUMN = "sentence_norm"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_TARGET_SR = 16000
DEFAULT_CORAL_DATASET_ID = "alexandrainst/coral"
DEFAULT_CORAL_CONFIG = "read_aloud"
DEFAULT_CORAL_LANGUAGE = "Danish"
DEFAULT_FLEURS_DATASET_ID = "google/fleurs"
DEFAULT_FLEURS_CONFIG = "de_de"
DEFAULT_FLEURS_LANGUAGE_DEFAULTS = {
    "de_de": "German",
    "is_is": "Icelandic",
    "da_dk": "Danish",
    "en_us": "English",
}


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int
    feature_dtype: torch.dtype = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        audio_arrays = []
        label_features = []

        for feature in features:
            audio_array, _ = load_audio_array(feature[DEFAULT_AUDIO_COLUMN])
            audio_arrays.append(audio_array)
            label_features.append(
                {"input_ids": self.processor.tokenizer(feature["sentence"]).input_ids}
            )

        batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=DEFAULT_TARGET_SR,
            return_tensors="pt",
        )
        batch["input_features"] = batch["input_features"].to(dtype=self.feature_dtype)

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


@dataclass
class PreparedWhisperData:
    processor: WhisperProcessor
    raw_datasets: Any
    collator: WhisperDataCollator


def load_processor(
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = DEFAULT_LANGUAGE,
    task: str = DEFAULT_TASK,
) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_name, language=language, task=task)


def get_decoder_start_token_id(processor: WhisperProcessor) -> int:
    decoder_start_token_id = processor.tokenizer.bos_token_id
    if decoder_start_token_id is None:
        raise ValueError("Whisper tokenizer is missing bos_token_id.")
    return decoder_start_token_id


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_audio_path(data_root: Path, row: dict[str, Any]) -> Path:
    split_dir = str(row["status"]).strip()
    speaker_id = str(row["speaker_id"]).strip()
    filename = str(row["filename"]).strip()
    audio_path = data_root / split_dir / speaker_id / filename

    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file for row {row['id']}: {audio_path}")

    return audio_path


def _load_split_records(
    metadata_path: Path,
    data_root: Path,
    split_name: str,
    text_column: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("status", "")).strip() != split_name:
                continue
            if _as_float(row.get("empty")) != 0.0:
                continue
            if _as_float(row.get("is_valid"), 1.0) != 1.0:
                continue

            sentence = str(row.get(text_column) or row.get("sentence") or "").strip()
            if not sentence:
                continue

            records.append(
                {
                    "id": str(row["id"]).strip(),
                    "audio": str(_resolve_audio_path(data_root, row)),
                    "sentence": sentence,
                    "duration": _as_float(row.get("duration")),
                }
            )

    return records


def load_samromur_dataset(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> DatasetDict:
    data_root = Path(data_root)
    metadata_path = data_root / "metadata.tsv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    datasets = {}
    for split_name in ("train", "dev", "test"):
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        records = _load_split_records(metadata_path, data_root, split_name, text_column)
        datasets[split_name] = Dataset.from_list(records)

    return DatasetDict(datasets)


def _configure_hf_cache(cache_dir: Path | str | None) -> None:
    if cache_dir is None:
        return

    cache_dir = Path(cache_dir)
    hub_cache = cache_dir / "hub"
    xet_cache = cache_dir / "xet"
    datasets_cache = cache_dir.parent / "hf_datasets_cache"
    xdg_cache = cache_dir.parent / ".cache"
    tmp_dir = cache_dir.parent / "tmp"

    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _standardize_coral_split(dataset: Dataset) -> Dataset:
    dataset = dataset.cast_column(DEFAULT_AUDIO_COLUMN, Audio(decode=False))

    def convert_row(row: dict[str, Any]) -> dict[str, Any]:
        sentence = str(row.get("text") or "").strip()
        return {
            "id": str(row.get("id_recording") or row.get("id") or "").strip(),
            "audio": row[DEFAULT_AUDIO_COLUMN],
            "sentence": sentence,
            "duration": None,
        }

    dataset = dataset.filter(lambda row: bool(str(row.get("text") or "").strip()))
    return dataset.map(convert_row, remove_columns=dataset.column_names)


def load_coral_dataset(
    config_name: str = DEFAULT_CORAL_CONFIG,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> DatasetDict:
    _configure_hf_cache(cache_dir)

    raw_datasets = load_dataset(
        DEFAULT_CORAL_DATASET_ID,
        config_name,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    datasets = {}
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for source_split, target_split in split_map.items():
        if source_split not in raw_datasets:
            continue
        datasets[target_split] = _standardize_coral_split(raw_datasets[source_split])

    return DatasetDict(datasets)


def _standardize_fleurs_split(dataset: Dataset) -> Dataset:
    dataset = dataset.cast_column(DEFAULT_AUDIO_COLUMN, Audio(decode=False))

    def convert_row(row: dict[str, Any]) -> dict[str, Any]:
        sentence = str(row.get("transcription") or row.get("raw_transcription") or "").strip()
        return {
            "id": str(row.get("id") or "").strip(),
            "audio": row[DEFAULT_AUDIO_COLUMN],
            "sentence": sentence,
            "duration": None,
        }

    dataset = dataset.filter(
        lambda row: bool(str(row.get("transcription") or row.get("raw_transcription") or "").strip())
    )
    return dataset.map(convert_row, remove_columns=dataset.column_names)


def load_fleurs_dataset(
    config_name: str = DEFAULT_FLEURS_CONFIG,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> DatasetDict:
    _configure_hf_cache(cache_dir)

    raw_datasets = load_dataset(
        DEFAULT_FLEURS_DATASET_ID,
        config_name,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    datasets = {}
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for source_split, target_split in split_map.items():
        if source_split not in raw_datasets:
            continue
        datasets[target_split] = _standardize_fleurs_split(raw_datasets[source_split])

    return DatasetDict(datasets)


def load_audio_array(audio_source: Any) -> tuple[Any, int]:
    if isinstance(audio_source, dict):
        if audio_source.get("array") is not None and audio_source.get("sampling_rate") is not None:
            audio_array = audio_source["array"]
            sampling_rate = int(audio_source["sampling_rate"])
        elif audio_source.get("bytes") is not None:
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_source["bytes"]))
        elif audio_source.get("path"):
            audio_array, sampling_rate = sf.read(audio_source["path"])
        else:
            raise ValueError(f"Unsupported audio payload: {audio_source.keys()}")
    else:
        audio_array, sampling_rate = sf.read(audio_source)

    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = audio_array.mean(axis=1)

    if sampling_rate != DEFAULT_TARGET_SR:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sampling_rate,
            target_sr=DEFAULT_TARGET_SR,
        )
        sampling_rate = DEFAULT_TARGET_SR

    return audio_array, sampling_rate


def prepare_whisper_data(
    model_name: str = DEFAULT_MODEL_NAME,
    language: str = DEFAULT_LANGUAGE,
    task: str = DEFAULT_TASK,
    dataset_source: str = DEFAULT_DATASET_SOURCE,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    coral_config: str = DEFAULT_CORAL_CONFIG,
    fleurs_config: str = DEFAULT_FLEURS_CONFIG,
    hf_token: str | None = None,
    hf_cache_dir: Path | str | None = None,
) -> PreparedWhisperData:
    processor = load_processor(model_name=model_name, language=language, task=task)
    if dataset_source == "coral":
        raw_datasets = load_coral_dataset(
            config_name=coral_config,
            token=hf_token,
            cache_dir=hf_cache_dir,
        )
    elif dataset_source == "fleurs":
        raw_datasets = load_fleurs_dataset(
            config_name=fleurs_config,
            token=hf_token,
            cache_dir=hf_cache_dir,
        )
    elif dataset_source == "samromur":
        raw_datasets = load_samromur_dataset(data_root=data_root, text_column=text_column)
    else:
        raise ValueError(f"Unsupported dataset_source: {dataset_source}")

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=get_decoder_start_token_id(processor),
    )

    return PreparedWhisperData(
        processor=processor,
        raw_datasets=raw_datasets,
        collator=collator,
    )


if __name__ == "__main__":
    prepared = prepare_whisper_data()

    for split_name, dataset in prepared.raw_datasets.items():
        print(f"{split_name}: {len(dataset)} rows")

    sample = prepared.raw_datasets["train"][0]
    batch = prepared.collator([sample])

    print("\nSample:")
    print(f"id: {sample['id']}")
    print(f"audio: {sample['audio']}")
    print(f"sentence: {sample['sentence']}")
    print(f"duration: {sample['duration']}")

    print("\nBatch:")
    print(f"input_features shape: {tuple(batch['input_features'].shape)}")
    print(f"labels shape: {tuple(batch['labels'].shape)}")
