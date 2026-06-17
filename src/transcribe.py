import argparse
import json
import os
import sys
import time

from faster_whisper import WhisperModel

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_output_path, format_timestamp, srt_timestamp, is_supported, print_separator

DEFAULT_MODEL = "large-v3-turbo"

# Auto-detect device: use CUDA if available, fall back to CPU
def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def transcribe_file(
    input_path: str,
    output_dir: str = "output",
    model: str = DEFAULT_MODEL,
    task: str = "transcribe",
    language: str = None,
    output_format: str = "txt",
    verbose: bool = True,
) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if not is_supported(input_path):
        raise ValueError(f"Unsupported file type: {os.path.splitext(input_path)[1]}")

    os.makedirs(output_dir, exist_ok=True)

    device, compute_type = _get_device()

    if verbose:
        print_separator()
        print(f"  File    : {os.path.basename(input_path)}")
        print(f"  Model   : {model}")
        print(f"  Device  : {device} ({compute_type})")
        print(f"  Task    : {task}")
        print(f"  Language: {language or 'auto-detect'}")
        print(f"  Format  : {output_format}")
        print_separator()

    t0 = time.time()

    fw_model = WhisperModel(model, device=device, compute_type=compute_type)
    segments_gen, info = fw_model.transcribe(
        input_path,
        task=task,
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
    )

    # Collect generator into a list and build a result dict compatible with
    # the rest of the codebase (quality.py, validate.py expect this shape)
    segments_list = []
    full_text_parts = []

    for seg in segments_gen:
        words = []
        if seg.words:
            words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in seg.words
            ]

        segments_list.append({
            "id": len(segments_list),
            "seek": 0,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "tokens": [],
            "temperature": seg.temperature if seg.temperature else 0.0,
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
            "no_speech_prob": seg.no_speech_prob,
            "words": words,
        })
        full_text_parts.append(seg.text)

    result = {
        "text": " ".join(full_text_parts),
        "segments": segments_list,
        "language": info.language,
    }

    elapsed = time.time() - t0

    if verbose:
        print(f"  Detected language : {info.language} (prob {info.language_probability:.2f})")
        print(f"  Transcription time: {elapsed:.1f}s")

    out_path = _save_output(result, input_path, output_dir, output_format)

    if verbose:
        print(f"  Saved to : {out_path}")
        print_separator()

    return result


def _save_output(result: dict, input_path: str, output_dir: str, output_format: str) -> str:
    if output_format == "srt":
        return _save_srt(result, input_path, output_dir)
    elif output_format == "json":
        return _save_json(result, input_path, output_dir)
    else:
        return _save_txt(result, input_path, output_dir)


def _save_txt(result: dict, input_path: str, output_dir: str) -> str:
    out_path = get_output_path(input_path, output_dir, ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"[{start} --> {end}] {segment['text'].strip()}\n")
    return out_path


def _save_srt(result: dict, input_path: str, output_dir: str) -> str:
    out_path = get_output_path(input_path, output_dir, ".srt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            start = srt_timestamp(segment["start"])
            end = srt_timestamp(segment["end"])
            f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
    return out_path


def _save_json(result: dict, input_path: str, output_dir: str) -> str:
    out_path = get_output_path(input_path, output_dir, ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe or translate a video/audio file using faster-whisper"
    )
    parser.add_argument("input", help="Path to video or audio file")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model size: tiny, base, small, medium, large-v3, large-v3-turbo")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--language", default=None)
    parser.add_argument("--format", dest="output_format",
                        choices=["txt", "srt", "json"], default="txt")
    args = parser.parse_args()

    transcribe_file(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        task=args.task,
        language=args.language,
        output_format=args.output_format,
    )
