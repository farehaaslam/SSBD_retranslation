from __future__ import annotations

from typing import Any

from faster_whisper import WhisperModel


class FasterWhisperTranscriber:
    """Owns the Faster-Whisper model and transcribes normalized audio."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str | None = None,
        compute_type: str = "int8",
    ) -> None:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: Any, language: str | None = None) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ).strip()
