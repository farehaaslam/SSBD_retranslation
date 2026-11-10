from __future__ import annotations

import os
from dataclasses import dataclass, field


LANGUAGES: dict[str, str] = {
    "English": "en",
    "Japanese": "ja",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Mandarin)": "zh",
    "Korean": "ko",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru",
    "Bengali": "bn",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Turkish": "tr",
    "Dutch": "nl",
    "Thai": "th",
    "Vietnamese": "vi",
}


@dataclass(frozen=True)
class AppConfig:
    model_name: str = field(
        default_factory=lambda: os.getenv("SSBD_MODEL", "Qwen/Qwen3.5-0.8B")
    )
    whisper_model_size: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "medium")
    )
    source_language: str = field(
        default_factory=lambda: os.getenv("SOURCE_LANGUAGE", "English")
    )
    asr_language: str | None = field(
        default_factory=lambda: os.getenv("ASR_LANGUAGE", "en") or None
    )
    target_language: str = field(
        default_factory=lambda: os.getenv("TARGET_LANGUAGE", "Japanese")
    )
    ssbd_beta: float = field(
        default_factory=lambda: float(os.getenv("SSBD_BETA", "0.2"))
    )
    mask_k: int = field(
        default_factory=lambda: int(os.getenv("MASK_K", "3"))
    )
    max_new_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "128"))
    )
    asr_update_seconds: float = field(
        default_factory=lambda: float(os.getenv("ASR_UPDATE_SECONDS", "2.0"))
    )
    asr_min_audio_seconds: float = field(
        default_factory=lambda: float(os.getenv("ASR_MIN_AUDIO_SECONDS", "1.2"))
    )
    languages: dict[str, str] = field(
        default_factory=lambda: dict(LANGUAGES)
    )

    @classmethod
    def from_environment(cls) -> AppConfig:
        return cls()