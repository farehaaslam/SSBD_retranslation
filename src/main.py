from __future__ import annotations

import torch

from src.config import AppConfig
from src.core import TranslationEngine
from src.stt import FasterWhisperTranscriber
from src.translation import QwenTranslationModel, SSBDTranslator
from src.ui import GradioTranslationApp


def create_app() -> GradioTranslationApp:
    config = AppConfig.from_environment()
    asr_device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = FasterWhisperTranscriber(
        model_size=config.whisper_model_size,
        device=asr_device,
        compute_type="int8",
    )
    model = QwenTranslationModel(
        model_name=config.model_name,
        max_new_tokens=config.max_new_tokens,
    )
    translator = SSBDTranslator(model, beta=config.ssbd_beta)
    engine = TranslationEngine(config, transcriber, translator)
    return GradioTranslationApp(config, engine)


def main() -> None:
    create_app().launch()


if __name__ == "__main__":
    main()
