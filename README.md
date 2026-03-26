# Whisper Fine-Tuning

Small repo for training, evaluating, and serving `openai/whisper-small` for speech recognition.

If you are reviewing this project, start with the API.

This project is built around three tasks:

- prepare speech datasets
- fine-tune Whisper
- serve a local transcription API

Large local files such as datasets, checkpoints, and caches are intentionally not tracked in Git.

## Start Here

The main deliverable is the FastAPI transcription service in `src/serve_whisper_api.py`.

What to look at first:

- run the API locally
- upload audio in the browser demo at `http://localhost:8000`
- test `POST /transcribe`
- check how the service switches between the base model and the fine-tuned model

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/hallgrimurth/Whisper-Finetuning.git
cd Whisper-Finetuning
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

Install the correct PyTorch build for your machine first.

Example for CUDA 12.4:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If you are using CPU only, install the CPU build from the PyTorch website.

### 4. Install project dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Point the API at your fine-tuned model

Local checkpoint example:

```powershell
$env:WHISPER_FINETUNED_MODEL="outputs/whisper-small-coral"
```

Hugging Face model example:

```powershell
$env:WHISPER_FINETUNED_MODEL="your-name/your-whisper-model"
```

If `WHISPER_FINETUNED_MODEL` is not set, the API uses `outputs/whisper-small-coral` by default.

### 6. Run the API

```bash
python -m uvicorn src.serve_whisper_api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser after startup.

## Serving

### What the API does

`POST /transcribe` accepts an audio file and an optional `language`.

Routing behavior:

- Danish requests use the Danish fine-tuned checkpoint
- other explicit languages use the base Whisper model
- if no language is provided, the service auto-detects language first and then selects the model

### Local demo

The root page at `GET /` serves a simple browser UI for:

- uploading audio
- recording audio
- choosing a language or using auto-detect
- viewing transcript and model metadata

### Useful endpoints

- `GET /`: demo UI
- `GET /health`: basic service health
- `POST /transcribe`: transcription endpoint

### Example API call

```bash
curl -X POST "http://localhost:8000/transcribe" ^
  -F "file=@sample.wav" ^
  -F "language=Danish"
```

### Important environment variables

- `WHISPER_BASE_MODEL`: base model id, default `openai/whisper-small`
- `WHISPER_FINETUNED_MODEL`: checkpoint path or Hugging Face repo id
- `WHISPER_LANGUAGE_DETECTION_WINDOW_SECONDS`: detection window size
- `WHISPER_LANGUAGE_DETECTION_MAX_WINDOWS`: number of windows used for auto-detect
- `WHISPER_LANGUAGE_DETECTION_MIN_VOICED_SECONDS`: minimum voiced audio before trusting detection
- `WHISPER_DANISH_DETECTION_CONFIDENCE_THRESHOLD`: minimum confidence required to route to the Danish model
- `WHISPER_DANISH_DETECTION_MARGIN_THRESHOLD`: minimum lead over the next-best detected language

### Response metadata

The API returns the transcript plus routing and timing fields such as:

- `transcript`
- `model_used`
- `detected_language`
- `detected_language_name`
- `previous_model`
- `model_transition_time_seconds`
- `inference_time_seconds`

## Training And Evaluation

### Train a model

Default training uses the Danish CORAL `read_aloud` split.

```bash
python src/train_whisper.py --dataset-source coral --language Danish --output-dir outputs/whisper-small-coral
```

Useful options:

- `--dataset-source coral|fleurs|samromur`
- `--coral-config read_aloud`
- `--fleurs-config da_dk`
- `--max-train-samples` for a quick smoke test
- `--max-eval-samples` for a smaller validation run
- `--report-to wandb` to log training metrics

### Evaluate a checkpoint

```bash
python src/evaluate_whisper.py --dataset-source coral --language Danish --checkpoint-path outputs/whisper-small-coral
```

The evaluation report is written to `outputs/whisper_eval_results.json`.

## Notes

- Keep datasets and checkpoints out of Git
- The repo assumes local outputs are written under `outputs/`
- If a model path does not exist locally, the code falls back to loading from Hugging Face
