# Whisper Fine-Tuning

Minimal handoff repo for fine-tuning and serving `openai/whisper-small` for Danish ASR.

This repo keeps only the code needed to:

- prepare training data from Hugging Face datasets or a local Samromur export
- fine-tune Whisper Small
- evaluate a saved checkpoint against the base model
- serve the base and fine-tuned models behind a small FastAPI API

Large local artifacts such as datasets, checkpoints, archives, caches, and notebook outputs are intentionally excluded from version control.

## Project layout

- `src/prepare_whisper_data.py`: dataset loading and audio preprocessing
- `src/load_whisper_model.py`: model and processor loading
- `src/train_whisper.py`: training entrypoint
- `src/evaluate_whisper.py`: evaluation entrypoint
- `src/serve_whisper_api.py`: FastAPI service and minimal demo page

## Setup

Install PyTorch for your platform first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Default training targets the Danish CORAL `read_aloud` split:

```bash
python src/train_whisper.py \
  --dataset-source coral \
  --language Danish \
  --output-dir outputs/whisper-small-coral
```

Useful options:

- `--dataset-source coral|fleurs|samromur`
- `--coral-config read_aloud`
- `--fleurs-config da_dk`
- `--max-train-samples` and `--max-eval-samples` for quick smoke runs
- `--report-to wandb` if experiment tracking is wanted

## Evaluation

Evaluate a fine-tuned checkpoint against the base model on the held-out split for the chosen dataset source:

```bash
python src/evaluate_whisper.py \
  --dataset-source coral \
  --language Danish \
  --checkpoint-path outputs/whisper-small-coral
```

The script writes a detailed JSON report to `outputs/whisper_eval_results.json`.

## Serving

Run the API locally:

```bash
python -m uvicorn src.serve_whisper_api:app --host 0.0.0.0 --port 8000
```

Environment variables:

- `WHISPER_BASE_MODEL`: base model id, defaults to `openai/whisper-small`
- `WHISPER_FINETUNED_MODEL`: path to the fine-tuned checkpoint, defaults to `outputs/whisper-small-coral`

API behavior:

- `POST /transcribe` accepts an uploaded audio file and optional `language`
- Danish requests route to the fine-tuned checkpoint
- Other languages route to the base model
- If no language is provided, the service detects language first and then selects the model

Returned metadata includes:

- `model_used`
- `previous_model`
- `transition_time_seconds`
- `inference_time_seconds`
