"""Voice transcription via faster-whisper."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE

_whisper_model: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    """Lazy-load the Whisper model on first voice input."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, compute_type="int8")
    return _whisper_model


def transcribe_audio(audio_tuple) -> str:
    """Transcribe audio from Gradio's Audio component (sample_rate, np_array)."""
    if audio_tuple is None:
        return ""
    sample_rate, audio_data = audio_tuple
    if audio_data is None or len(audio_data) == 0:
        return ""
    # Normalise to float32 mono
    audio = audio_data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.max() > 1.0:
        audio = audio / 32768.0
    # Write to a temp WAV for faster-whisper
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = tmp.name
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
