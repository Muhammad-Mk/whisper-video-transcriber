import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from transcribe import transcribe_file, DEFAULT_MODEL
from utils import SUPPORTED_EXTENSIONS
from quality import analyze as analyze_quality, grade_color, format_ts
from validate import compute_wer, pick_spot_check, accuracy_color

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Whisper Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #888; margin-bottom: 1.5rem; }
    .stat-box   { background: #1e1e2e; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; }
    .stat-num   { font-size: 1.8rem; font-weight: 700; color: #a6e3a1; }
    .stat-label { font-size: 0.8rem; color: #888; }
    .result-header { font-size: 1rem; font-weight: 600; margin-top: 1rem; }
    div[data-testid="stExpander"] { border: 1px solid #313244; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Settings ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    model_options = {
        "large-v3-turbo  (Recommended — fast & accurate)": "large-v3-turbo",
        "large-v3        (Best accuracy, slower)":         "large-v3",
        "medium          (Balanced)":                      "medium",
        "small           (Faster, lighter)":               "small",
        "tiny            (Fastest, for testing)":          "tiny",
    }
    selected_model_label = st.selectbox("Model", list(model_options.keys()), index=0)
    model = model_options[selected_model_label]

    st.divider()

    task = st.radio(
        "Task",
        ["transcribe", "translate"],
        captions=[
            "Speech → text in the same language",
            "Non-English speech → English text",
        ],
    )

    st.divider()

    language_options = {
        "Auto-detect": None,
        "English (en)": "en",
        "Urdu (ur)": "ur",
        "Arabic (ar)": "ar",
        "French (fr)": "fr",
        "Spanish (es)": "es",
        "German (de)": "de",
        "Chinese (zh)": "zh",
        "Hindi (hi)": "hi",
        "Japanese (ja)": "ja",
        "Portuguese (pt)": "pt",
    }
    selected_lang_label = st.selectbox("Language", list(language_options.keys()), index=0)
    language = language_options[selected_lang_label]

    st.divider()

    output_format = st.radio(
        "Output Format",
        ["txt", "srt", "json"],
        captions=[
            "Timestamped plain text",
            "Subtitle file (.srt) for video editors",
            "Full JSON with word-level timestamps",
        ],
    )

    st.divider()
    st.caption("🖥️ faster-whisper · Auto-detects CUDA or CPU")
    st.caption("Models cached at `~/.cache/huggingface/hub/`")

# ── Main — Header ─────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🎙️ Whisper Transcriber</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Local speech-to-text powered by mlx-whisper on Apple Silicon GPU · No data leaves your machine</p>', unsafe_allow_html=True)

# ── Main — File Upload ────────────────────────────────────────────────────────

ACCEPTED_TYPES = ["mp4", "mov", "mkv", "avi", "webm", "flv", "mp3", "wav", "m4a", "aac", "ogg", "flac"]

uploaded_files = st.file_uploader(
    "Drop video or audio files here",
    type=ACCEPTED_TYPES,
    accept_multiple_files=True,
    help=f"Supported: {', '.join('.' + e for e in ACCEPTED_TYPES)} · Max 4 GB per file",
)

# ── Main — Run Button ─────────────────────────────────────────────────────────

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button(
        "▶  Transcribe",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )
with col_info:
    if uploaded_files:
        total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        size_str = f"{total_mb/1024:.2f} GB" if total_mb >= 1024 else f"{total_mb:.1f} MB"
        st.info(f"**{len(uploaded_files)} file(s)** · {size_str} total · Model: `{model.split('/')[-1]}` · Task: `{task}` · Format: `{output_format}`")
    else:
        st.info("Upload one or more files above, then click **Transcribe**. Files up to 4 GB supported.")

# ── Main — Processing ─────────────────────────────────────────────────────────

if run and uploaded_files:
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    overall_progress = st.progress(0, text="Starting…")
    results = []

    for idx, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        overall_progress.progress(
            idx / len(uploaded_files),
            text=f"Processing {idx + 1}/{len(uploaded_files)}: **{filename}**"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix,
            prefix="whisper_",
        ) as tmp:
            # Write in 64 MB chunks — safe for files well over 1 GB
            CHUNK = 64 * 1024 * 1024
            while True:
                chunk = uploaded_file.read(CHUNK)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name

        status_placeholder = st.empty()
        file_progress = st.progress(0.0, text="Starting…")

        # Get audio duration for progress estimation (ffprobe ships with ffmpeg)
        audio_duration = None
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", tmp_path],
                capture_output=True, text=True, timeout=15,
            )
            audio_duration = float(json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            pass

        # Approximate real-time multiplier per model on CPU (Xeon / server benchmarks)
        speed_map = {
            "tiny": 30, "base": 20, "small": 12,
            "medium": 7, "large-v3-turbo": 5, "large-v3": 3,
        }
        speed_x = speed_map.get(model, 5)
        expected_secs = (audio_duration / speed_x) if audio_duration else None

        t0 = time.time()
        try:
            # Run transcription in background thread so UI stays responsive
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    transcribe_file,
                    input_path=tmp_path,
                    output_dir=output_dir,
                    model=model,
                    task=task,
                    language=language,
                    output_format=output_format,
                    verbose=False,
                )
                while not future.done():
                    elapsed_so_far = time.time() - t0
                    if expected_secs:
                        pct = min(elapsed_so_far / expected_secs, 0.95)
                        remaining = max(expected_secs - elapsed_so_far, 0)
                        rem_str = f"{remaining:.0f}s remaining" if remaining > 1 else "finishing…"
                        file_progress.progress(
                            pct,
                            text=f"Transcribing **{filename}** — {pct*100:.0f}% · {rem_str}"
                        )
                    else:
                        file_progress.progress(
                            0.5,
                            text=f"Transcribing **{filename}** — {elapsed_so_far:.0f}s elapsed…"
                        )
                    time.sleep(0.5)

                result = future.result()

            elapsed = time.time() - t0
            file_progress.progress(1.0, text=f"Done — {elapsed:.1f}s")
            detected_lang = result.get("language", "unknown")

            # transcribe_file() saves using the temp file's stem — find and rename to original
            ext_map = {"txt": ".txt", "srt": ".srt", "json": ".json"}
            ext = ext_map[output_format]
            saved_path = os.path.join(output_dir, Path(tmp_path).stem + ext)
            final_filename = Path(filename).stem + ext
            final_path = os.path.join(output_dir, final_filename)
            if saved_path != final_path:
                os.replace(saved_path, final_path)
            with open(final_path, "r", encoding="utf-8") as f:
                out_content = f.read()
            out_filename = final_filename

            quality = analyze_quality(result)
            status_placeholder.success(f"✅ **{filename}** — done in {elapsed:.1f}s · language: `{detected_lang}` · quality: **{quality.grade}** ({quality.score:.0f}/100)")
            results.append({
                "filename": filename,
                "out_filename": out_filename,
                "out_content": out_content,
                "result": result,
                "quality": quality,
                "elapsed": elapsed,
                "detected_lang": detected_lang,
                "error": None,
            })

        except Exception as e:
            elapsed = time.time() - t0
            file_progress.empty()
            status_placeholder.error(f"❌ **{filename}** — {e}")
            results.append({
                "filename": filename,
                "out_filename": None,
                "out_content": None,
                "result": None,
                "elapsed": elapsed,
                "detected_lang": None,
                "error": str(e),
            })
        finally:
            os.unlink(tmp_path)

    overall_progress.progress(1.0, text="All files processed.")

    # ── Summary stats ─────────────────────────────────────────────────────────

    st.divider()
    success_count = sum(1 for r in results if r["error"] is None)
    total_time = sum(r["elapsed"] for r in results)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{success_count}/{len(results)}</div><div class="stat-label">Files succeeded</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{total_time:.1f}s</div><div class="stat-label">Total time</div></div>', unsafe_allow_html=True)
    with c3:
        avg = total_time / len(results) if results else 0
        st.markdown(f'<div class="stat-box"><div class="stat-num">{avg:.1f}s</div><div class="stat-label">Avg per file</div></div>', unsafe_allow_html=True)

    # ── Per-file results ───────────────────────────────────────────────────────

    st.divider()
    st.markdown("### Results")

    for r in results:
        icon = "✅" if r["error"] is None else "❌"
        label = f"{icon} {r['filename']}"
        if r["error"] is None:
            q = r["quality"]
            label += f"  ·  `{r['detected_lang']}`  ·  {r['elapsed']:.1f}s  ·  {q.grade} ({q.score:.0f}/100)"

        with st.expander(label, expanded=(r["error"] is None)):
            if r["error"]:
                st.error(r["error"])
                continue

            tab_quality, tab_validate, tab_transcript = st.tabs([
                "📊 Quality (Confidence)",
                "🎯 Validate Accuracy (WER)",
                "📄 Transcript",
            ])

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — Confidence-based quality
            # ══════════════════════════════════════════════════════════════════
            with tab_quality:
                q = r["quality"]
                st.caption("Confidence score — how certain the model was about its own output. Not the same as accuracy.")

                qc1, qc2, qc3, qc4 = st.columns(4)
                color = grade_color(q.grade)
                with qc1:
                    st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{color}">{q.grade}</div><div class="stat-label">Confidence grade</div></div>', unsafe_allow_html=True)
                with qc2:
                    st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{color}">{q.score:.0f}<span style="font-size:1rem">/100</span></div><div class="stat-label">Confidence score</div></div>', unsafe_allow_html=True)
                with qc3:
                    flag_pct = len(q.flagged_segments) / q.total_segments * 100 if q.total_segments else 0
                    flag_color = "#a6e3a1" if flag_pct < 10 else "#f9e2af" if flag_pct < 30 else "#f38ba8"
                    st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{flag_color}">{len(q.flagged_segments)}<span style="font-size:1rem">/{q.total_segments}</span></div><div class="stat-label">Flagged segments</div></div>', unsafe_allow_html=True)
                with qc4:
                    risk_color = "#f38ba8" if q.hallucination_risk else "#a6e3a1"
                    risk_label = "Yes ⚠️" if q.hallucination_risk else "No ✅"
                    st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{risk_color}">{risk_label}</div><div class="stat-label">Hallucination risk</div></div>', unsafe_allow_html=True)

                st.caption(f"Avg log-probability: `{q.avg_logprob:.3f}`  ·  Avg no-speech prob: `{q.avg_no_speech_prob:.3f}`  ·  Low-confidence words: `{len(q.low_conf_words)}`")

                if q.flagged_segments:
                    st.markdown("**⚠️ Segments to review:**")
                    for fs in q.flagged_segments:
                        ts = f"`{format_ts(fs.start)} → {format_ts(fs.end)}`"
                        st.markdown(f"- {ts} &nbsp; _{fs.reason}_  \n  > {fs.text}")
                else:
                    st.success("No segments flagged — transcript looks clean.")

                if q.low_conf_words:
                    with st.expander(f"🔍 {len(q.low_conf_words)} low-confidence words (< 60%)"):
                        rows = [f"| `{format_ts(w['start'])}` | **{w['word']}** | {w['probability']*100:.0f}% |"
                                for w in q.low_conf_words[:50]]
                        st.markdown("| Time | Word | Confidence |\n|------|------|------------|")
                        st.markdown("\n".join(rows))
                        if len(q.low_conf_words) > 50:
                            st.caption(f"Showing first 50 of {len(q.low_conf_words)}.")

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — True accuracy validation
            # ══════════════════════════════════════════════════════════════════
            with tab_validate:
                st.caption("True accuracy = how close the transcript is to what was actually said. Two methods:")

                method = st.radio(
                    "Validation method",
                    ["WER — paste a reference transcript", "Spot-check — sample random segments"],
                    key=f"val_method_{r['filename']}",
                    horizontal=True,
                )

                # ── Method A: WER ──────────────────────────────────────────────
                if method.startswith("WER"):
                    st.markdown(
                        "Paste a known-correct transcript (even a partial excerpt). "
                        "The tool calculates **Word Error Rate (WER)** — the industry standard for ASR accuracy."
                    )
                    ref_text = st.text_area(
                        "Reference transcript",
                        placeholder="Paste the correct text here — even a few sentences is enough for a meaningful score…",
                        height=160,
                        key=f"ref_{r['filename']}",
                    )

                    # Extract plain hypothesis text from transcript
                    hyp_lines = r["out_content"].split("\n")
                    hyp_text = " ".join(
                        line.split("] ", 1)[-1] if "] " in line else line
                        for line in hyp_lines if line.strip()
                    )

                    if st.button("Calculate accuracy", key=f"wer_btn_{r['filename']}", type="primary"):
                        if not ref_text.strip():
                            st.warning("Please paste a reference transcript first.")
                        else:
                            wer = compute_wer(ref_text, hyp_text)
                            acc_color = accuracy_color(wer.accuracy)

                            w1, w2, w3, w4 = st.columns(4)
                            with w1:
                                st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{acc_color}">{wer.accuracy*100:.1f}%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
                            with w2:
                                st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{acc_color}">{wer.grade}</div><div class="stat-label">Grade</div></div>', unsafe_allow_html=True)
                            with w3:
                                st.markdown(f'<div class="stat-box"><div class="stat-num">{wer.wer*100:.1f}%</div><div class="stat-label">Word Error Rate</div></div>', unsafe_allow_html=True)
                            with w4:
                                st.markdown(f'<div class="stat-box"><div class="stat-num">{wer.cer*100:.1f}%</div><div class="stat-label">Char Error Rate</div></div>', unsafe_allow_html=True)

                            st.divider()
                            ec1, ec2, ec3, ec4 = st.columns(4)
                            with ec1:
                                st.metric("Words in reference", wer.word_count_ref)
                            with ec2:
                                st.metric("Correct words (hits)", wer.hits)
                            with ec3:
                                st.metric("Wrong words (subs)", wer.substitutions)
                            with ec4:
                                st.metric("Missing / extra words", f"{wer.deletions} / {wer.insertions}")

                            st.info(
                                f"**How to read this:** WER of {wer.wer*100:.1f}% means roughly "
                                f"{wer.wer*100:.0f} out of every 100 words had an error. "
                                f"Broadcast-quality ASR targets < 5% WER. "
                                f"Good for non-native or technical speech is < 15%."
                            )

                # ── Method B: Spot-check ───────────────────────────────────────
                else:
                    st.markdown(
                        "Randomly samples segments across the full video. "
                        "Jump to each timestamp in your video player and verify manually."
                    )
                    n_samples = st.slider("Number of segments to sample", 5, 20, 8, key=f"spot_n_{r['filename']}")

                    if st.button("Generate spot-check", key=f"spot_btn_{r['filename']}", type="primary"):
                        samples = pick_spot_check(r["result"], n=n_samples)
                        if not samples:
                            st.warning("No segments found in result.")
                        else:
                            st.markdown(f"**{len(samples)} segments sampled** — open your video and jump to each timestamp:")
                            st.markdown("")
                            for i, seg in enumerate(samples, 1):
                                st.markdown(
                                    f"**{i}.** `{seg.timestamp_str}` &nbsp;→&nbsp; {seg.text}"
                                )
                            st.divider()
                            correct = st.number_input(
                                "How many of the above were correct?",
                                min_value=0, max_value=len(samples), value=len(samples),
                                key=f"spot_correct_{r['filename']}",
                            )
                            spot_acc = correct / len(samples) * 100
                            acc_color = accuracy_color(correct / len(samples))
                            st.markdown(
                                f'<div class="stat-box" style="max-width:220px">'
                                f'<div class="stat-num" style="color:{acc_color}">{spot_acc:.0f}%</div>'
                                f'<div class="stat-label">Spot-check accuracy ({correct}/{len(samples)} correct)</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — Transcript preview + download
            # ══════════════════════════════════════════════════════════════════
            with tab_transcript:
                preview_lines = r["out_content"].split("\n")[:30]
                st.code("\n".join(preview_lines), language="")
                if len(r["out_content"].split("\n")) > 30:
                    st.caption("Showing first 30 lines — download for full transcript.")

                mime_map = {"txt": "text/plain", "srt": "text/plain", "json": "application/json"}
                st.download_button(
                    label=f"⬇️  Download {r['out_filename']}",
                    data=r["out_content"],
                    file_name=r["out_filename"],
                    mime=mime_map.get(output_format, "text/plain"),
                    key=f"dl_{r['filename']}",
                )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption("faster-whisper · Streamlit 1.58.0 · Auto-detects CUDA GPU or CPU · All processing is local")
