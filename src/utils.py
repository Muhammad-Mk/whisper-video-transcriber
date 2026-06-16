import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv',
    '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'
}


def get_output_path(input_path: str, output_dir: str, ext: str = '.txt') -> str:
    stem = Path(input_path).stem
    return os.path.join(output_dir, stem + ext)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def is_supported(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def print_separator(char='─', width=55):
    print(char * width)
