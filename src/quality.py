from dataclasses import dataclass, field
from typing import List


@dataclass
class SegmentIssue:
    start: float
    end: float
    text: str
    reason: str


@dataclass
class QualityReport:
    score: float           # 0–100
    grade: str             # Excellent / Good / Fair / Poor
    total_segments: int
    flagged_segments: List[SegmentIssue]
    low_conf_words: List[dict]
    avg_logprob: float
    avg_no_speech_prob: float
    hallucination_risk: bool


def analyze(result: dict, low_word_prob_threshold: float = 0.6) -> QualityReport:
    segments = result.get("segments", [])
    if not segments:
        return QualityReport(
            score=0, grade="N/A", total_segments=0,
            flagged_segments=[], low_conf_words=[],
            avg_logprob=0, avg_no_speech_prob=0,
            hallucination_risk=False,
        )

    logprobs, no_speech_probs = [], []
    flagged: List[SegmentIssue] = []
    low_conf_words: List[dict] = []

    for seg in segments:
        logprob = seg.get("avg_logprob", -1.0)
        no_speech = seg.get("no_speech_prob", 0.0)
        compression = seg.get("compression_ratio", 1.0)
        temperature = seg.get("temperature", 0.0)
        text = seg.get("text", "").strip()

        logprobs.append(logprob)
        no_speech_probs.append(no_speech)

        reasons = []
        if logprob < -0.6:
            reasons.append(f"low confidence (logprob {logprob:.2f})")
        if no_speech > 0.5:
            reasons.append(f"possibly no speech (no_speech_prob {no_speech:.2f})")
        if compression > 2.4:
            reasons.append(f"repetitive text (compression_ratio {compression:.2f})")
        if temperature > 0:
            reasons.append(f"model used sampling fallback (temp {temperature:.1f})")

        if reasons:
            flagged.append(SegmentIssue(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=text,
                reason=", ".join(reasons),
            ))

        # Collect low-confidence words
        for w in seg.get("words", []):
            prob = w.get("probability", 1.0)
            if prob < low_word_prob_threshold:
                low_conf_words.append({
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0),
                    "probability": prob,
                })

    avg_logprob = sum(logprobs) / len(logprobs)
    avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
    hallucination_risk = avg_no_speech > 0.3 or len(flagged) / len(segments) > 0.3

    # Score: map avg_logprob [-1, 0] → [0, 100], penalise flagged ratio
    raw = max(0.0, min(1.0, (avg_logprob + 1.0) / 0.7))
    flagged_ratio = len(flagged) / len(segments)
    score = round((raw * (1 - 0.5 * flagged_ratio)) * 100, 1)

    if score >= 80:
        grade = "Excellent"
    elif score >= 60:
        grade = "Good"
    elif score >= 40:
        grade = "Fair"
    else:
        grade = "Poor"

    return QualityReport(
        score=score,
        grade=grade,
        total_segments=len(segments),
        flagged_segments=flagged,
        low_conf_words=low_conf_words,
        avg_logprob=avg_logprob,
        avg_no_speech_prob=avg_no_speech,
        hallucination_risk=hallucination_risk,
    )


def grade_color(grade: str) -> str:
    return {"Excellent": "#a6e3a1", "Good": "#89dceb", "Fair": "#f9e2af", "Poor": "#f38ba8"}.get(grade, "#cdd6f4")


def format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
