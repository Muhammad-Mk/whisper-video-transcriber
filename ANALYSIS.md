# Video Speech-to-Text — System Analysis & Options

> Prepared for: MacBook Pro M4 Pro · 48 GB RAM · macOS  
> Date: June 2026

---

## Table of Contents

1. [What Is Whisper](#1-what-is-whisper)
2. [System Specifications](#2-system-specifications)
3. [Approach Options](#3-approach-options)
4. [Local via Docker — Deep Dive](#4-local-via-docker--deep-dive)
5. [Local Native (mlx-whisper) — Deep Dive](#5-local-native-mlx-whisper--deep-dive)
6. [Cloud Providers — Deep Dive](#6-cloud-providers--deep-dive)
7. [Cloud Cost Calculator — 10 Videos × 30 Min](#7-cloud-cost-calculator--10-videos--30-min)
8. [Final Recommendation](#8-final-recommendation)

---

## 1. What Is Whisper

OpenAI Whisper is a general-purpose automatic speech recognition (ASR) system. It is:

- Trained on 680,000 hours of multilingual audio scraped from the internet
- A single model that handles: transcription, translation, language detection, and voice activity detection
- Open-source and MIT licensed (both code and model weights — free to use commercially)
- Based on a Transformer sequence-to-sequence architecture

### Supported Tasks

| Task | Description |
|------|-------------|
| `transcribe` | Speech → text in the **same language** as the audio |
| `translate` | Non-English speech → **English text** |
| `detect_language` | Identify what language is being spoken |

> **Important limitation:** Whisper translates **into English only**. It cannot translate English into another language, nor translate between two non-English languages.

### Available Model Sizes

| Model | Parameters | English-only variant | VRAM (GPU) | RAM (CPU) | Relative Speed |
|-------|-----------|---------------------|-----------|----------|----------------|
| tiny | 39M | Yes (tiny.en) | ~1 GB | ~400 MB | Fastest |
| base | 74M | Yes (base.en) | ~1 GB | ~500 MB | Fast |
| small | 244M | Yes (small.en) | ~2 GB | ~800 MB | Moderate |
| medium | 769M | Yes (medium.en) | ~5 GB | ~2 GB | Slower |
| large-v3 | 1550M | No | ~10 GB | ~4 GB | Slowest |
| turbo | ~800M | No | ~6 GB | ~3 GB | Fast (no translation) |

> **Note:** The `turbo` model does NOT support translation. Use `medium` or `large-v3` for translation tasks.

---

## 2. System Specifications

Analyzed on: MacBook Pro (2024)

```
Chip         : Apple M4 Pro
Memory       : 48 GB Unified Memory
CPU Cores    : 14 (performance + efficiency)
GPU/NPU      : Apple M4 Pro GPU + Neural Engine (built into chip)
Disk (free)  : ~215 GB available
Docker       : Docker Desktop v29.5.2 (running)
Docker RAM   : 7.75 GB allocated to VM
OS           : macOS (Darwin 25.5.0)
```

### Why Unified Memory Matters

Apple Silicon does not have separate CPU RAM and GPU VRAM. All 48 GB is shared between the CPU, GPU, and Neural Engine. This means:

- You can load large models without worrying about VRAM limits
- The large-v3 model (~4 GB) fits easily, leaving 44 GB for the system
- Tools like `mlx-whisper` can use the GPU/Neural Engine directly from this shared pool

---

## 3. Approach Options

Three viable approaches were evaluated:

```
Option A: Local — Docker (CPU only)
Option B: Local — Native mlx-whisper (GPU + Neural Engine) ← Recommended
Option C: Cloud (pay-per-use API)
```

### Quick Comparison

| Factor | Docker (Local) | Native mlx-whisper | Cloud API |
|--------|---------------|-------------------|-----------|
| Cost | Free | Free | $0.75–$7.20 per 300 min |
| Speed | Moderate (CPU only) | Very fast (GPU+NPU) | Fast (depends on API) |
| Privacy | Full (local) | Full (local) | Data leaves your machine |
| GPU access | No (VM limitation) | Yes (full M4 Pro GPU) | N/A |
| Setup effort | Low | Low | Very low |
| Works offline | Yes | Yes | No |
| Model quality | Any Whisper model | Any Whisper model | Depends on provider |
| Scalability | Manual | Manual | Auto-scales |

---

## 4. Local via Docker — Deep Dive

### How It Works

Docker Desktop on Mac runs a lightweight Linux virtual machine (using Apple Virtualization Framework). Whisper runs inside that VM. This means:

- **No GPU access** — the Apple M4 Pro GPU cannot be passed through to Docker containers on macOS
- **CPU-only inference** — all computation happens on CPU cores allocated to Docker
- Transcription is still accurate (same model weights), just slower

### Resource Usage Estimates

| Model | RAM Usage | CPU Usage | Time for 30-min video |
|-------|----------|-----------|----------------------|
| small/int8 | ~600 MB | 60–80% | ~4–6 min |
| medium/int8 | ~1.5 GB | 80–100% | ~8–12 min |
| large-v3/int8 | ~3 GB | 100% | ~20–30 min |

> Docker is currently allocated 7.75 GB RAM — sufficient for all models above.

### Docker RAM Configuration

If you want to increase Docker's RAM allocation (for headroom):
1. Open Docker Desktop → Settings → Resources
2. Increase memory slider (recommended: 12–16 GB)
3. Apply & Restart

### Sample Docker Run (faster-whisper)

```bash
# Pull and run
docker run --rm \
  -v /path/to/your/videos:/videos \
  -v /path/to/output:/output \
  python:3.11-slim bash -c "
    pip install faster-whisper ffmpeg-python -q &&
    python -c \"
from faster_whisper import WhisperModel
model = WhisperModel('medium', device='cpu', compute_type='int8')
segments, info = model.transcribe('/videos/myvideo.mp4', task='transcribe')
with open('/output/transcript.txt', 'w') as f:
    for s in segments:
        f.write(f'[{s.start:.1f}s → {s.end:.1f}s] {s.text}\n')
    \"
  "
```

### Why Docker Is Not Ideal on This Machine

Given the M4 Pro's capabilities, running inside Docker means:
- You artificially limit yourself to CPU-only processing
- A 30-min video takes 20–30 min instead of ~45–90 seconds
- You get ~20–40x slower performance for no benefit

---

## 5. Local Native (mlx-whisper) — Deep Dive

### What Is MLX?

MLX is Apple's open-source machine learning framework, purpose-built for Apple Silicon. It is designed to run efficiently on the unified memory architecture of M-series chips and directly uses:

- **Apple GPU** (via Metal)
- **Neural Engine** (the dedicated AI accelerator chip inside M4 Pro)
- **CPU** as fallback

### What Is mlx-whisper?

`mlx-whisper` is an implementation of OpenAI's Whisper model using the MLX framework. It uses the same model weights but runs them through MLX instead of PyTorch, enabling full GPU/Neural Engine utilization on Apple Silicon.

### Performance on M4 Pro

| Model | Processing time for 30-min video | Method |
|-------|----------------------------------|--------|
| medium | ~2–3 min | PyTorch CPU |
| medium | **~30–45 sec** | mlx-whisper (GPU/NPU) |
| large-v3 | ~8–15 min | PyTorch CPU |
| large-v3-turbo | **~45–90 sec** | mlx-whisper (GPU/NPU) |

> mlx-whisper is approximately **10–20x faster** than CPU-only approaches on M4 Pro.

### Resource Usage (mlx-whisper)

| Model | RAM used (unified) | GPU Activity | Neural Engine |
|-------|-------------------|-------------|---------------|
| small | ~600 MB | Moderate | Active |
| medium | ~1.5 GB | High | Active |
| large-v3-turbo | ~3 GB | High | Active |
| large-v3 | ~4 GB | Very High | Active |

> Out of 48 GB unified memory, even large-v3 uses less than 10%. System remains fully responsive during transcription.

### Accepted Video Formats

mlx-whisper uses `ffmpeg` internally to extract audio. Supported formats include:

- `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.flv`
- `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`
- Any container format that `ffmpeg` can decode

---

## 6. Cloud Providers — Deep Dive

### OpenAI Whisper API

- The same Whisper model, hosted by OpenAI
- Easiest integration if already using OpenAI APIs
- **Does not support translation** to languages other than English via API
- Max file size: 25 MB (requires chunking for longer videos)
- Rate limits apply

### AssemblyAI

- High accuracy, supports 99+ languages
- Offers speaker diarization (who said what), auto-chapters, sentiment analysis
- Universal-3 Pro model has best accuracy
- Simple REST API

### Deepgram

- Known for very fast response times (near real-time)
- Nova-3 model is competitive with Whisper large-v3 in accuracy
- Good for streaming/real-time applications
- $200 free credit for new users (no credit card required)

### AWS Transcribe

- Deep AWS ecosystem integration
- Speaker diarization included
- Custom vocabulary support
- 60 minutes/month free for 12 months

### Google Cloud Speech-to-Text

- Strong on noisy audio
- Supports 125+ languages
- Medical/phone call model variants
- Tight integration with other Google Cloud services

---

## 7. Cloud Cost Calculator — 10 Videos × 30 Min

```
Total audio: 10 videos × 30 minutes = 300 minutes = 5 hours
```

| Provider | Model/Tier | Price per minute | 300 min total | Free tier |
|----------|-----------|-----------------|--------------|-----------|
| AssemblyAI | Universal-2 | $0.0025 | **$0.75** | 100 min free |
| AssemblyAI | Universal-3 Pro | $0.0035 | **$1.05** | 100 min free |
| OpenAI Whisper API | whisper-1 | $0.006 | **$1.80** | None |
| Deepgram | Nova-3 (pay-as-you-go) | $0.0077 | **$2.31** | $200 credit |
| AWS Transcribe | Standard (Tier 1) | $0.024 | **$7.20** | 60 min/mo × 12 mo |
| Google Speech-to-Text | Standard | ~$0.024 | **~$7.20** | 60 min/mo |

### Add-on Costs (if needed)

| Feature | Provider | Additional cost |
|---------|---------|----------------|
| Speaker diarization | AssemblyAI | +$0.002/min |
| Speaker diarization | AWS | +$0.009/min |
| Auto chapters/summary | AssemblyAI | +$0.0025/min |
| Real-time streaming | Deepgram | +$0.0010/min |

### When Cloud Makes Sense

Cloud is worth considering when:
- You need **speaker diarization** (who said what at what time)
- You're processing **hundreds of hours** and want horizontal scaling
- You need **real-time/live transcription** (streaming WebSocket)
- You want **downstream NLP features** (summaries, sentiment, chapters) built in
- Your videos contain **highly sensitive domain vocabulary** (medical, legal) that benefits from specialized models

---

## 8. Final Recommendation

### For Your Setup (M4 Pro, 48 GB, local work)

**Use native mlx-whisper in a Python virtual environment.**

| Criterion | mlx-whisper wins because... |
|-----------|---------------------------|
| Speed | 10–20x faster than Docker/CPU due to M4 Pro GPU + Neural Engine |
| Cost | Completely free |
| Privacy | Videos never leave your machine |
| Quality | Access to large-v3 — highest Whisper accuracy |
| Simplicity | Single Python package, no containers |

### Model to Use

- **`mlx-community/whisper-large-v3-turbo`** — best balance of speed and quality
- Fall back to `mlx-community/whisper-large-v3` for maximum accuracy on difficult audio

### When to Reconsider

Switch to cloud only if:
- You need speaker labels (who said what)
- Volume grows to 50+ hours/month
- You need real-time transcription

---

*See [ROADMAP.md](ROADMAP.md) for the full step-by-step implementation guide.*
