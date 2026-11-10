from __future__ import annotations

import gradio as gr

from src.config import AppConfig
from src.core import TranslationEngine


class GradioTranslationApp:
    """Builds the Gradio UI and adapts events to the translation engine."""

    def __init__(self, config: AppConfig, engine: TranslationEngine) -> None:
        self.config = config
        self.engine = engine
        self.demo = self._build()

    def _start_session(self, source_language: str, target_language: str):
        if source_language == target_language:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "Please select two different languages.",
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            f"Live translation: {source_language} -> {target_language}",
        )

    def _live_audio_update(
        self,
        audio_chunk,
        audio_state,
        transcript_state: str,
        draft_state: str,
        source_language: str,
        target_language: str,
    ):
        result = self.engine.update(
            audio_chunk,
            audio_state,
            transcript_state,
            draft_state,
            source_language,
            target_language,
        )
        return (
            result.audio,
            result.transcript,
            result.draft,
            result.transcript,
            result.visible_translation,
            result.status,
        )

    @staticmethod
    def _reset_live():
        return None, "", "", "", "", "Ready. Start speaking."

    def _translate_text(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
    ):
        return self.engine.translate_text(
            source_text,
            source_language,
            target_language,
        )

    def _build(self) -> gr.Blocks:
        with gr.Blocks(title="Live SSBD Translation") as demo:
            with gr.Column(visible=True) as language_screen:
                gr.Markdown(
                    """
# Live Speech Translation

### Select your languages to get started

Choose the language you will speak and the language you want
to translate into.
"""
                )
                source_language_dropdown = gr.Dropdown(
                    choices=list(self.config.languages),
                    value=self.config.source_language,
                    label="Input / spoken language",
                )
                target_language_dropdown = gr.Dropdown(
                    choices=list(self.config.languages),
                    value=self.config.target_language,
                    label="Output / translation language",
                )
                start_button = gr.Button("Start translating", variant="primary")
                language_status = gr.Markdown()

            with gr.Column(visible=False) as translation_screen:
                gr.Markdown(
                    f"""
# Live SSBD Translation

Speak into the microphone. ASR updates trigger SSBD translation.

- **SSBD beta:** `{self.config.ssbd_beta}`
- **Display-only Mask-k:** `{self.config.mask_k}` tokens
"""
                )
                with gr.Tab("Live microphone"):
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        streaming=True,
                        label="Microphone (start recording to stream)",
                    )
                    transcript_box = gr.Textbox(label="Live ASR transcript", lines=3)
                    visible_translation = gr.Textbox(
                        label=f"Visible translation (Mask-k={self.config.mask_k})",
                        lines=4,
                    )
                    status_box = gr.Textbox(label="Status")
                    audio_state = gr.State(None)
                    transcript_state = gr.State("")
                    draft_state = gr.State("")
                    mic.stream(
                        fn=self._live_audio_update,
                        inputs=[
                            mic,
                            audio_state,
                            transcript_state,
                            draft_state,
                            source_language_dropdown,
                            target_language_dropdown,
                        ],
                        outputs=[
                            audio_state,
                            transcript_state,
                            draft_state,
                            transcript_box,
                            visible_translation,
                            status_box,
                        ],
                        stream_every=self.config.asr_update_seconds,
                        time_limit=3600,
                    )
                    reset_button = gr.Button("Reset session")
                    reset_button.click(
                        fn=self._reset_live,
                        inputs=[],
                        outputs=[
                            audio_state,
                            transcript_state,
                            draft_state,
                            transcript_box,
                            visible_translation,
                            status_box,
                        ],
                    )

                with gr.Tab("Text translation"):
                    text_in = gr.Textbox(label="Source text", lines=4)
                    text_button = gr.Button("Translate")
                    full_out = gr.Textbox(label="Full translation", lines=4)
                    masked_out = gr.Textbox(
                        label=f"Display-only Mask-k (k={self.config.mask_k})",
                        lines=4,
                    )
                    text_button.click(
                        fn=self._translate_text,
                        inputs=[
                            text_in,
                            source_language_dropdown,
                            target_language_dropdown,
                        ],
                        outputs=[full_out, masked_out],
                    )
                gr.Markdown(
                    """
**Note:** Faster-Whisper re-transcribes accumulated microphone audio
to provide live updates. This is a prototype streaming-ASR approximation,
not a native streaming ASR decoder. Mask-k is applied only to displayed text.
"""
                )

            start_button.click(
                fn=self._start_session,
                inputs=[source_language_dropdown, target_language_dropdown],
                outputs=[language_screen, translation_screen, language_status],
            )
        return demo

    def launch(self) -> None:
        self.demo.queue().launch()
