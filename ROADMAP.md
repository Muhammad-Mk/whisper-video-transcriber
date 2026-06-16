# mlx-whisper Implementation Roadmap

> Platform: macOS · Apple M4 Pro · 48 GB RAM  
> Approach: Native mlx-whisper inside Python virtual environment  
> Goal: Video/audio speech-to-text transcription & translation, fully local

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Phase 1 — Environment Setup](#2-phase-1--environment-setup)
3. [Phase 2 — Install Dependencies](#3-phase-2--install-dependencies)
4. [Phase 3 — Verify Installation](#4-phase-3--verify-installation)
5. [Phase 4 — Core Transcription Script](#5-phase-4--core-transcription-script)
6. [Phase 5 — Batch Processing (Multiple Videos)](#6-phase-5--batch-processing-multiple-videos)
7. [Phase 6 — Output Formats](#7-phase-6--output-formats)
8. [Phase 7 — CLI Tool (Optional Enhancement)](#8-phase-7--cli-tool-optional-enhancement)
9. [Virtual Environment Cheatsheet](#9-virtual-environment-cheatsheet)
10. [Troubleshooting](#10-troubleshooting)
11. [Model Reference](#11-model-reference)

---

## 1. Project Structure

Final directory layout after completing all phases:

```
whisper/
├── .venv/                    ← virtual environment (never commit this)
├── videos/                   ← drop input videos here
│   └── sample.mp4
├── output/                   ← transcripts saved here
│   └── sample.txt
├── src/
│   ├── transcribe.py         ← single video transcription
│   ├── batch.py              ← process entire videos/ folder
│   └── utils.py              ← shared helpers (formatting, file I/O)
├── requirements.txt          ← pinned dependencies
├── .gitignore
├── ANALYSIS.md
└── ROADMAP.md
```

---

## 2. Phase 1 — Environment Setup

### Step 1.1 — Check Python version

mlx requires Python 3.9 or higher. Check your system Python:

```bash
python3 --version
```

Expected output: `Python 3.11.x` or similar. If below 3.9, install via Homebrew:

```bash
brew install python@3.11
```

### Step 1.2 — Navigate to project directory

```bash
cd /Users/muhammadmk/Downloads/Syntaxtechs/whisper
```

### Step 1.3 — Create the virtual environment

```bash
python3 -m venv .venv
```

This creates a `.venv/` folder inside your project. It contains:
- Its own Python interpreter copy
- Its own `pip`
- Its own `site-packages/` (isolated from system)

> Nothing installed here affects your system-level Python or any other project.

### Step 1.4 — Activate the virtual environment

```bash
source .venv/bin/activate
```

Your terminal prompt will change to show `(.venv)` at the start:

```
(.venv) muhammadmk@MacBookPro whisper %
```

> Every time you open a new terminal tab to work on this project, run this activation command first.

### Step 1.5 — Confirm isolation

```bash
which python      # should show: .../whisper/.venv/bin/python
which pip         # should show: .../whisper/.venv/bin/pip
python --version  # confirms the venv's Python
```

---

## 3. Phase 2 — Install Dependencies

### Step 2.1 — Install ffmpeg (system-level, one-time)

ffmpeg is required for audio extraction from video files. Install via Homebrew (system-level, not in venv — this is correct):

```bash
brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
```

### Step 2.2 — Upgrade pip inside the venv

```bash
pip install --upgrade pip
```

### Step 2.3 — Install mlx-whisper

```bash
pip install mlx-whisper
```

This installs:
- `mlx-whisper` — the Apple Silicon optimized Whisper implementation
- `mlx` — Apple's ML framework (automatically pulled as dependency)
- `numpy`, `huggingface_hub`, `tqdm` — supporting libraries

### Step 2.4 — Install additional utilities

```bash
pip install ffmpeg-python tqdm rich
```

| Package | Purpose |
|---------|---------|
| `ffmpeg-python` | Python bindings to ffmpeg for video handling |
| `tqdm` | Progress bars during batch processing |
| `rich` | Beautiful terminal output (tables, colors) |

### Step 2.5 — Save dependencies to requirements.txt

```bash
pip freeze > requirements.txt
```

> This lets anyone recreate your exact environment later with `pip install -r requirements.txt`.

---

## 4. Phase 3 — Verify Installation

### Step 3.1 — Quick smoke test

Run this in your terminal (with venv active):

```bash
python -c "import mlx_whisper; print('mlx-whisper installed successfully')"
```

### Step 3.2 — Test transcription on a short audio clip

Download a short test audio (or use any video you have):

```bash
python -c "
import mlx_whisper
result = mlx_whisper.transcribe(
    'https://github.com/openai/whisper/raw/main/tests/jfk.flac',
    path_or_hf_repo='mlx-community/whisper-tiny'
)
print(result['text'])
"
```

Expected output (JFK speech clip):
```
And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

> The first run downloads the model weights from Hugging Face (~150 MB for tiny). Subsequent runs are instant (cached).

### Step 3.3 — Confirm GPU is being used

```bash
python -c "
import mlx.core as mx
print('MLX default device:', mx.default_device())
"
```

Expected output: `MLX default device: Device(gpu, 0)`

> If you see `cpu`, mlx may not have Metal support — try reinstalling with `pip install --upgrade mlx`.

---

## 5. Phase 4 — Core Transcription Script

Create the folder structure:

```bash
mkdir -p src videos output
```

### [src/utils.py](src/utils.py)

```python
import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv',
                        '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

def get_output_path(input_path: str, output_dir: str, ext: str = '.txt') -> str:
    stem = Path(input_path).stem
    return os.path.join(output_dir, stem + ext)

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def is_supported(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS
```

### [src/transcribe.py](src/transcribe.py)

```python
import argparse
import os
import mlx_whisper
from utils import get_output_path, format_timestamp, is_supported

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

def transcribe_file(
    input_path: str,
    output_dir: str = "output",
    model: str = DEFAULT_MODEL,
    task: str = "transcribe",
    language: str = None,
    output_format: str = "txt"
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if not is_supported(input_path):
        raise ValueError(f"Unsupported file type: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Model     : {model}")
    print(f"Task      : {task}")
    print(f"Language  : {language or 'auto-detect'}")
    print(f"Input     : {input_path}")

    result = mlx_whisper.transcribe(
        input_path,
        path_or_hf_repo=model,
        task=task,
        language=language,
        word_timestamps=True,
        verbose=False
    )

    detected_lang = result.get("language", "unknown")
    print(f"Detected  : {detected_lang}")

    if output_format == "txt":
        _save_txt(result, input_path, output_dir)
    elif output_format == "srt":
        _save_srt(result, input_path, output_dir)
    elif output_format == "json":
        _save_json(result, input_path, output_dir)
    else:
        _save_txt(result, input_path, output_dir)

    print(f"Saved to  : {output_dir}/")


def _save_txt(result, input_path, output_dir):
    out_path = get_output_path(input_path, output_dir, ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"[{start} --> {end}] {segment['text'].strip()}\n")
    print(f"Transcript: {out_path}")


def _save_srt(result, input_path, output_dir):
    out_path = get_output_path(input_path, output_dir, ".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            start = _srt_timestamp(segment["start"])
            end = _srt_timestamp(segment["end"])
            f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
    print(f"Subtitles : {out_path}")


def _save_json(result, input_path, output_dir):
    import json
    out_path = get_output_path(input_path, output_dir, ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"JSON      : {out_path}")


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video or audio file using mlx-whisper")
    parser.add_argument("input", help="Path to video or audio file")
    parser.add_argument("--output-dir", default="output", help="Directory to save transcript (default: output/)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model repo")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                        help="transcribe = same language | translate = to English")
    parser.add_argument("--language", default=None,
                        help="Language code e.g. 'en', 'ur', 'ar', 'fr' (optional, auto-detected if omitted)")
    parser.add_argument("--format", dest="output_format", choices=["txt", "srt", "json"], default="txt",
                        help="Output format (default: txt)")
    args = parser.parse_args()

    transcribe_file(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        task=args.task,
        language=args.language,
        output_format=args.output_format
    )
```

### Usage — Single File

```bash
# Activate venv first
source .venv/bin/activate

# Basic transcription (auto-detects language)
python src/transcribe.py videos/my_lecture.mp4

# Transcription with known language
python src/transcribe.py videos/urdu_talk.mp4 --language ur

# Translate non-English speech → English text
python src/transcribe.py videos/arabic_video.mp4 --task translate --language ar

# Generate subtitle file (.srt)
python src/transcribe.py videos/my_lecture.mp4 --format srt

# Use a different model
python src/transcribe.py videos/my_lecture.mp4 --model mlx-community/whisper-large-v3
```

---

## 6. Phase 5 — Batch Processing (Multiple Videos)

### [src/batch.py](src/batch.py)

```python
import os
import argparse
import time
from pathlib import Path
from transcribe import transcribe_file, DEFAULT_MODEL
from utils import is_supported


def batch_transcribe(
    videos_dir: str = "videos",
    output_dir: str = "output",
    model: str = DEFAULT_MODEL,
    task: str = "transcribe",
    language: str = None,
    output_format: str = "txt"
):
    video_files = [
        f for f in sorted(Path(videos_dir).iterdir())
        if f.is_file() and is_supported(str(f))
    ]

    if not video_files:
        print(f"No supported video/audio files found in '{videos_dir}/'")
        return

    total = len(video_files)
    print(f"\nFound {total} file(s) to process\n{'─' * 50}")

    success, failed = 0, []

    for i, file in enumerate(video_files, 1):
        print(f"\n[{i}/{total}] {file.name}")
        start = time.time()
        try:
            transcribe_file(
                input_path=str(file),
                output_dir=output_dir,
                model=model,
                task=task,
                language=language,
                output_format=output_format
            )
            elapsed = time.time() - start
            print(f"Done in {elapsed:.1f}s")
            success += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(file.name)

    print(f"\n{'─' * 50}")
    print(f"Completed : {success}/{total}")
    if failed:
        print(f"Failed    : {', '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch transcribe all videos in a folder")
    parser.add_argument("--videos-dir", default="videos", help="Input folder (default: videos/)")
    parser.add_argument("--output-dir", default="output", help="Output folder (default: output/)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--language", default=None)
    parser.add_argument("--format", dest="output_format", choices=["txt", "srt", "json"], default="txt")
    args = parser.parse_args()

    batch_transcribe(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        model=args.model,
        task=args.task,
        language=args.language,
        output_format=args.output_format
    )
```

### Usage — Batch

```bash
# Drop all videos into the videos/ folder, then:
python src/batch.py

# Batch translate all videos to English
python src/batch.py --task translate

# Batch with subtitle output
python src/batch.py --format srt

# Batch with specific language
python src/batch.py --language ur --format txt
```

---

## 7. Phase 6 — Output Formats

Three output formats are supported:

### Plain Text (.txt)

```
[00:00:00.00 --> 00:00:03.24] And so my fellow Americans,
[00:00:03.24 --> 00:00:07.10] ask not what your country can do for you,
[00:00:07.10 --> 00:00:10.45] ask what you can do for your country.
```

### SubRip Subtitles (.srt)

Standard subtitle format, importable into video editors (DaVinci Resolve, Premiere, Final Cut Pro) and video players (VLC):

```
1
00:00:00,000 --> 00:00:03,240
And so my fellow Americans,

2
00:00:03,240 --> 00:00:07,100
ask not what your country can do for you,
```

### JSON (.json)

Full structured output including word-level timestamps, language detection confidence, and all segment metadata. Useful for downstream processing:

```json
{
  "text": "And so my fellow Americans...",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.24,
      "text": " And so my fellow Americans,",
      "words": [...]
    }
  ]
}
```

---

## 8. Phase 7 — CLI Tool (Optional Enhancement)

Create a `.gitignore` to keep the repo clean:

### [.gitignore](.gitignore)

```
.venv/
__pycache__/
*.pyc
*.pyo
output/
videos/
.DS_Store
*.egg-info/
dist/
build/
```

---

## 9. Virtual Environment Cheatsheet

| Action | Command |
|--------|---------|
| Create venv | `python3 -m venv .venv` |
| Activate (Mac/Linux) | `source .venv/bin/activate` |
| Deactivate | `deactivate` |
| Install a package | `pip install <package>` |
| Save dependencies | `pip freeze > requirements.txt` |
| Restore from requirements | `pip install -r requirements.txt` |
| List installed packages | `pip list` |
| Check active Python | `which python` |
| Delete venv (start fresh) | `rm -rf .venv` |

### Recreating the Environment on Another Machine

```bash
git clone <your-repo>
cd whisper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

---

## 10. Troubleshooting

### Problem: `command not found: python`
```bash
# Use python3 explicitly
python3 -m venv .venv
```

### Problem: `mlx not found` or Metal not available
```bash
pip install --upgrade mlx mlx-whisper
```

### Problem: `ffmpeg not found`
```bash
brew install ffmpeg
# ffmpeg is a system tool, installed via brew, not pip
```

### Problem: Model download is slow / fails
```bash
# Models are downloaded from Hugging Face to ~/.cache/huggingface/
# Check cache:
ls ~/.cache/huggingface/hub/

# If download fails midway, delete the partial cache and retry:
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

### Problem: Forgot to activate venv
```bash
# You'll see "No module named mlx_whisper" if venv is not active
# Always activate first:
source .venv/bin/activate
```

### Problem: `(venv)` not showing in prompt
This is cosmetic — some shell configs hide it. Check if venv is active:
```bash
which python  # should show a path inside .venv/
```

### Problem: Output file is empty
- Check that `ffmpeg` is installed and accessible: `ffmpeg -version`
- Verify the video has an audio track: `ffprobe yourfile.mp4`

---

## 11. Model Reference

All models are fetched automatically from Hugging Face on first use and cached locally.

| Model HF Repo | Speed | Quality | Use Case |
|---------------|-------|---------|----------|
| `mlx-community/whisper-tiny` | Fastest | Basic | Testing only |
| `mlx-community/whisper-base` | Very fast | Fair | Short clips, clear audio |
| `mlx-community/whisper-small` | Fast | Good | General use |
| `mlx-community/whisper-medium` | Moderate | Very good | Lectures, podcasts |
| `mlx-community/whisper-large-v3-turbo` | Fast | Excellent | **Recommended default** |
| `mlx-community/whisper-large-v3` | Slower | Best | Max accuracy, difficult audio |

> Cache location: `~/.cache/huggingface/hub/`  
> Model sizes on disk: tiny ~150 MB · base ~290 MB · small ~970 MB · medium ~3 GB · large-v3-turbo ~3 GB · large-v3 ~6 GB

---

## Execution Summary

Once setup is complete, the full workflow is:

```bash
# 1. Go to project
cd /Users/muhammadmk/Downloads/Syntaxtechs/whisper

# 2. Activate environment
source .venv/bin/activate

# 3a. Single video
python src/transcribe.py videos/my_video.mp4 --format srt

# 3b. All videos in batch
python src/batch.py --format txt

# 4. Find transcripts
ls output/

# 5. Done. Deactivate when finished.
deactivate
```

---

*See [ANALYSIS.md](ANALYSIS.md) for full system analysis, Docker comparison, and cloud cost breakdown.*
