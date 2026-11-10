from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import AppConfig
from src.stt import FasterWhisperTranscriber
from src.translation import SSBDTranslator


@dataclass
class TranslationUpdate:
    audio: np.ndarray | None
    transcript: str
    draft: str
    visible_translation: str
    status: str


class TranslationEngine:
    """Coordinates audio preprocessing, ASR, SSBD, and display masking."""

    def __init__(
        self,
        config: AppConfig,
        transcriber: FasterWhisperTranscriber,
        translator: SSBDTranslator,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.translator = translator

    @staticmethod
    def to_mono_float(audio_chunk: Any) -> tuple[int | None, np.ndarray | None]:
        if not isinstance(audio_chunk, tuple) or len(audio_chunk) != 2:
            return None, None
        sr, audio = audio_chunk
        if audio is None:
            return None, None
        audio = np.asarray(audio)
        if audio.size == 0:
            return None, None
        if audio.ndim == 2:
            audio = audio.mean(axis=0 if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0] else 1)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            max_val = max(abs(info.min), info.max)
            audio = audio.astype(np.float32) / float(max_val)
        else:
            audio = audio.astype(np.float32)
        sr = int(sr)
        if sr != 16000:
            new_len = max(1, int(round(audio.size * 16000 / sr)))
            audio = np.interp(
                np.linspace(0, audio.size - 1, new_len, dtype=np.float32),
                np.arange(audio.size, dtype=np.float32),
                audio,
            ).astype(np.float32)
            sr = 16000
        return sr, audio

    def update(
        self,
        audio_chunk: Any,
        audio_state: np.ndarray | None,
        transcript_state: str,
        draft_state: str,
        source_language: str,
        target_language: str,
    ) -> TranslationUpdate:
        _, chunk = self.to_mono_float(audio_chunk)
        if chunk is None:
            return TranslationUpdate(
                audio_state,
                transcript_state,
                draft_state,
                self.translator.mask_for_display(draft_state, self.config.mask_k),
                "Waiting for microphone audio...",
            )
        audio = chunk if audio_state is None else np.concatenate((audio_state, chunk))
        seconds = len(audio) / 16000.0
        visible = self.translator.mask_for_display(draft_state, self.config.mask_k)
        if seconds < self.config.asr_min_audio_seconds:
            return TranslationUpdate(audio, transcript_state, draft_state, visible, f"Listening... {seconds:.1f}s")

        asr_language = self.config.languages.get(
            source_language,
            self.config.asr_language,
        )
        transcript = self.transcriber.transcribe(audio, asr_language)
        if not transcript:
            return TranslationUpdate(audio, transcript_state, draft_state, visible, f"Listening... {seconds:.1f}s")
        if transcript == transcript_state:
            return TranslationUpdate(audio, transcript_state, draft_state, visible, f"ASR unchanged | {seconds:.1f}s")

        draft = self.translator.translate(
            transcript,
            draft_state,
            source_language,
            target_language,
        )
        return TranslationUpdate(
            audio,
            transcript,
            draft,
            self.translator.mask_for_display(draft, self.config.mask_k),
            f"Live | {seconds:.1f}s audio | SSBD beta={self.config.ssbd_beta:.2f} | Mask-k={self.config.mask_k}",
        )

    def translate_text(self, source_text: str, source_language: str, target_language: str) -> tuple[str, str]:
        source_text = (source_text or "").strip()
        if not source_text:
            return "", ""
        full = self.translator.initial_translation(source_text, source_language, target_language)
        return full, self.translator.mask_for_display(full, self.config.mask_k)
