import argparse
import json
import os
import sys
import time

import mlx_whisper

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_output_path, format_timestamp, srt_timestamp, is_supported, print_separator

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


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
        raise ValueError(f"Unsupported file type: {Path(input_path).suffix}")

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print_separator()
        print(f"  File    : {os.path.basename(input_path)}")
        print(f"  Model   : {model}")
        print(f"  Task    : {task}")
        print(f"  Language: {language or 'auto-detect'}")
        print(f"  Format  : {output_format}")
        print_separator()

    t0 = time.time()
    result = mlx_whisper.transcribe(
        input_path,
        path_or_hf_repo=model,
        task=task,
        language=language,
        word_timestamps=True,
        verbose=False,
    )
    elapsed = time.time() - t0

    if verbose:
        detected = result.get("language", "unknown")
        print(f"  Detected language : {detected}")
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
        description="Transcribe or translate a video/audio file using mlx-whisper (Apple Silicon GPU)"
    )
    parser.add_argument(
        "input",
        help="Path to video or audio file (mp4, mov, mkv, mp3, wav, etc.)"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to save transcript (default: output/)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Hugging Face model repo (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe",
        help="transcribe = same language  |  translate = speech → English (default: transcribe)"
    )
    parser.add_argument(
        "--language", default=None,
        help="Language code e.g. en, ur, ar, fr — omit for auto-detection"
    )
    parser.add_argument(
        "--format", dest="output_format",
        choices=["txt", "srt", "json"], default="txt",
        help="Output format (default: txt)"
    )
    args = parser.parse_args()

    transcribe_file(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        task=args.task,
        language=args.language,
        output_format=args.output_format,
    )
