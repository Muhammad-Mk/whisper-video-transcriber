import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from transcribe import transcribe_file, DEFAULT_MODEL
from utils import is_supported, print_separator


def batch_transcribe(
    videos_dir: str = "videos",
    output_dir: str = "output",
    model: str = DEFAULT_MODEL,
    task: str = "transcribe",
    language: str = None,
    output_format: str = "txt",
    progress_callback=None,
) -> dict:
    """
    Transcribe all supported files in videos_dir.
    progress_callback(current, total, filename) is called before each file if provided.
    Returns {"success": [...], "failed": [...], "total_time": float}
    """
    video_files = sorted(
        f for f in Path(videos_dir).iterdir()
        if f.is_file() and is_supported(str(f))
    )

    if not video_files:
        print(f"No supported files found in '{videos_dir}/'")
        return {"success": [], "failed": [], "total_time": 0.0}

    total = len(video_files)
    print_separator()
    print(f"  Batch job : {total} file(s) found in '{videos_dir}/'")
    print(f"  Model     : {model}")
    print(f"  Task      : {task}")
    print(f"  Format    : {output_format}")
    print_separator()

    success, failed = [], []
    job_start = time.time()

    for i, file in enumerate(video_files, 1):
        if progress_callback:
            progress_callback(i, total, file.name)

        print(f"\n[{i}/{total}] {file.name}")
        try:
            transcribe_file(
                input_path=str(file),
                output_dir=output_dir,
                model=model,
                task=task,
                language=language,
                output_format=output_format,
                verbose=True,
            )
            success.append(file.name)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append({"file": file.name, "error": str(e)})

    total_time = time.time() - job_start
    print_separator()
    print(f"  Done      : {len(success)}/{total} succeeded in {total_time:.1f}s")
    if failed:
        print(f"  Failed    : {', '.join(f['file'] for f in failed)}")
    print_separator()

    return {"success": success, "failed": failed, "total_time": total_time}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch transcribe all video/audio files in a folder using mlx-whisper"
    )
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
        output_format=args.output_format,
    )
