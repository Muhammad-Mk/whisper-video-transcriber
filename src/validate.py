import random
import re
from dataclasses import dataclass, field
from typing import List

import jiwer


@dataclass
class WERReport:
    wer: float            # 0.0 – 1.0
    cer: float
    accuracy: float       # 1 - WER, clamped to [0, 1]
    grade: str
    word_count_ref: int
    word_count_hyp: int
    substitutions: int
    deletions: int
    insertions: int
    hits: int


@dataclass
class SpotCheckSegment:
    index: int
    start: float
    end: float
    text: str
    timestamp_str: str


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> WERReport:
    ref_norm = _normalise(reference)
    hyp_norm = _normalise(hypothesis)

    out = jiwer.process_words(ref_norm, hyp_norm)
    wer = min(out.wer, 1.0)
    accuracy = max(0.0, 1.0 - wer)
    cer = min(jiwer.cer(ref_norm, hyp_norm), 1.0)

    if accuracy >= 0.95:
        grade = "Excellent"
    elif accuracy >= 0.85:
        grade = "Good"
    elif accuracy >= 0.70:
        grade = "Fair"
    else:
        grade = "Poor"

    return WERReport(
        wer=wer,
        cer=cer,
        accuracy=accuracy,
        grade=grade,
        word_count_ref=len(ref_norm.split()),
        word_count_hyp=len(hyp_norm.split()),
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
    )


def pick_spot_check(result: dict, n: int = 8, seed: int = None) -> List[SpotCheckSegment]:
    segments = result.get("segments", [])
    if not segments:
        return []

    # Sample spread across the full duration rather than random cluster
    step = max(1, len(segments) // n)
    indices = list(range(0, len(segments), step))[:n]
    if seed is not None:
        random.seed(seed)
        indices = sorted(random.sample(range(len(segments)), min(n, len(segments))))

    out = []
    for i in indices:
        seg = segments[i]
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        m, s = divmod(int(start), 60)
        h, m = divmod(m, 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        out.append(SpotCheckSegment(
            index=i,
            start=start,
            end=end,
            text=seg.get("text", "").strip(),
            timestamp_str=ts,
        ))
    return out


def accuracy_color(accuracy: float) -> str:
    if accuracy >= 0.95:
        return "#a6e3a1"
    elif accuracy >= 0.85:
        return "#89dceb"
    elif accuracy >= 0.70:
        return "#f9e2af"
    else:
        return "#f38ba8"
