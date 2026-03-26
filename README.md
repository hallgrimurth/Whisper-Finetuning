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

