
"""
Live microphone translation with Faster-Whisper + SSBD + display-only mask-k.

- Select input and output languages on the first screen.
- Gradio streams microphone audio while recording.
- Faster-Whisper re-transcribes accumulated audio.
- SSBD uses the previous translation as a draft.
- Mask-k hides final k tokens in the UI only.
"""

import os

import gradio as gr
import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = os.getenv("SSBD_MODEL", "Qwen/Qwen3.5-0.8B")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")

SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "English")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en") or None
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "Japanese")

BETA = float(os.getenv("SSBD_BETA", "0.2"))
MASK_K = int(os.getenv("MASK_K", "3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

ASR_UPDATE_SECONDS = float(os.getenv("ASR_UPDATE_SECONDS", "2.0"))
ASR_MIN_AUDIO_SECONDS = float(os.getenv("ASR_MIN_AUDIO_SECONDS", "1.2"))

ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE = "int8" if ASR_DEVICE == "cuda" else "int8"


# ============================================================
# Load models
# ============================================================

print(f"Loading Faster-Whisper {WHISPER_MODEL_SIZE} on {ASR_DEVICE}...")

whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=ASR_DEVICE,
    compute_type=ASR_COMPUTE,
)

print("Faster-Whisper loaded.")

print(f"Loading translation model: {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("SSBD translation model loaded.")


# ============================================================
# Translation prompt
# ============================================================

def build_prompt(source_text: str) -> torch.Tensor:
    messages = [
        {
            "role": "system",
            "content": (
                f"Translate {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}. "
                f"Return only the {TARGET_LANGUAGE} translation."
            ),
        },
        {
            "role": "user",
            "content": source_text,
        },
    ]

    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )

    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]

    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids)

    return ids.to(model.device)


# ============================================================
# Initial translation
# ============================================================

@torch.inference_mode()
def initial_translation(source_text: str) -> str:

    prompt_ids = build_prompt(source_text)

    output_ids = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output_ids[:, prompt_ids.shape[1]:]

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
    ).strip()


# ============================================================
# SSBD implementation
# ============================================================

@torch.inference_mode()
def prepare_source_cache(source_text: str):

    prompt_ids = build_prompt(source_text)

    out = model(
        input_ids=prompt_ids,
        use_cache=True,
    )

    return out.past_key_values, out.logits[:, -1, :]


@torch.inference_mode()
def verify_draft(
    draft_ids: torch.Tensor,
    past_key_values,
    first_logits,
    beta=BETA,
):

    draft_ids = draft_ids.to(model.device)

    n = draft_ids.shape[1]

    if n == 0:
        return 0, draft_ids

    draft_out = model(
        input_ids=draft_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    logits_by_position = [first_logits]

    logits_by_position.extend(
        draft_out.logits[:, i - 1, :]
        for i in range(1, n)
    )

    accepted = []

    for i in range(n):

        p = torch.softmax(
            logits_by_position[i],
            dim=-1,
        )

        draft_token = draft_ids[0, i]

        biased_p = p * (1.0 - beta)

        biased_p[0, draft_token] += beta

        if (
            torch.argmax(biased_p, dim=-1).item()
            != draft_token.item()
        ):
            break

        accepted.append(draft_token.item())

    accepted_ids = torch.tensor(
        accepted,
        dtype=torch.long,
        device=model.device,
    )

    return len(accepted), accepted_ids


@torch.inference_mode()
def generate_suffix(
    source_text: str,
    accepted_ids: torch.Tensor,
):

    prompt_ids = build_prompt(source_text)

    if accepted_ids.numel():

        accepted_ids = accepted_ids.unsqueeze(0)

        input_ids = torch.cat(
            (prompt_ids, accepted_ids),
            dim=1,
        )

    else:
        input_ids = prompt_ids

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    return output_ids[:, input_ids.shape[1]:][0]


@torch.inference_mode()
def ssbd_step(
    current_source: str,
    previous_translation: str,
) -> str:

    current_source = (current_source or "").strip()

    if not current_source:
        return ""

    if not previous_translation:
        return initial_translation(current_source)

    source_cache, first_logits = prepare_source_cache(
        current_source
    )

    draft_ids = tokenizer(
        previous_translation,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)

    _, accepted_ids = verify_draft(
        draft_ids,
        source_cache,
        first_logits,
        beta=BETA,
    )

    suffix_ids = generate_suffix(
        current_source,
        accepted_ids,
    )

    accepted_text = tokenizer.decode(
        accepted_ids,
        skip_special_tokens=True,
    )

    suffix_text = tokenizer.decode(
        suffix_ids,
        skip_special_tokens=True,
    )

    return (accepted_text + suffix_text).strip()


# ============================================================
# Display-only Mask-k
# ============================================================

def mask_k_for_display(
    full_translation: str,
    k: int = MASK_K,
) -> str:

    if not full_translation or k <= 0:
        return full_translation or ""

    ids = tokenizer(
        full_translation,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]

    if ids.numel() <= k:
        return ""

    stable_ids = ids[:-k]

    return tokenizer.decode(
        stable_ids,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# Audio processing
# ============================================================

def _to_mono_float(audio_chunk):

    if audio_chunk is None:
        return None, None

    if not isinstance(audio_chunk, tuple):
        return None, None

    if len(audio_chunk) != 2:
        return None, None

    sr, audio = audio_chunk

    if audio is None:
        return None, None

    audio = np.asarray(audio)

    if audio.size == 0:
        return None, None

    if audio.ndim == 2:

        if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):

        max_val = max(
            abs(np.iinfo(audio.dtype).min),
            np.iinfo(audio.dtype).max,
        )

        audio = audio.astype(np.float32) / float(max_val)

    else:
        audio = audio.astype(np.float32)

    sr = int(sr)

    # Resample to 16 kHz
    if sr != 16000 and audio.size:

        new_len = max(
            1,
            int(round(audio.size * 16000 / sr)),
        )

        old_x = np.arange(
            audio.size,
            dtype=np.float32,
        )

        new_x = np.linspace(
            0,
            audio.size - 1,
            new_len,
            dtype=np.float32,
        )

        audio = np.interp(
            new_x,
            old_x,
            audio,
        ).astype(np.float32)

        sr = 16000

    return sr, audio


def _transcribe_buffer(audio_16k: np.ndarray) -> str:

    segments, _ = whisper_model.transcribe(
        audio_16k,
        language=ASR_LANGUAGE,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    return " ".join(
        s.text.strip()
        for s in segments
        if s.text and s.text.strip()
    ).strip()


# ============================================================
# Live streaming callback
# ============================================================

def live_audio_update(
    audio_chunk,
    audio_state,
    transcript_state,
    draft_state,
):

    sr, chunk = _to_mono_float(audio_chunk)

    if chunk is None:

        return (
            audio_state,
            transcript_state,
            draft_state,
            transcript_state or "",
            mask_k_for_display(draft_state or ""),
            "Waiting for microphone audio...",
        )

    if audio_state is None:
        audio_state = chunk
    else:
        audio_state = np.concatenate(
            (audio_state, chunk)
        )

    seconds = len(audio_state) / 16000.0

    if seconds < ASR_MIN_AUDIO_SECONDS:

        return (
            audio_state,
            transcript_state,
            draft_state,
            transcript_state or "",
            mask_k_for_display(draft_state or ""),
            f"Listening... {seconds:.1f}s",
        )

    # Re-transcribe accumulated audio
    new_transcript = _transcribe_buffer(audio_state)

    if not new_transcript:

        return (
            audio_state,
            transcript_state,
            draft_state,
            "",
            mask_k_for_display(draft_state or ""),
            f"Listening... {seconds:.1f}s",
        )

    # Skip if transcript hasn't changed
    if new_transcript == transcript_state:

        return (
            audio_state,
            transcript_state,
            draft_state,
            new_transcript,
            mask_k_for_display(draft_state or ""),
            f"ASR unchanged | {seconds:.1f}s",
        )

    # Incremental SSBD translation
    new_draft = ssbd_step(
        new_transcript,
        draft_state or "",
    )

    visible = mask_k_for_display(
        new_draft,
        MASK_K,
    )

    return (
        audio_state,
        new_transcript,
        new_draft,
        new_transcript,
        visible,
        (
            f"Live | {seconds:.1f}s audio | "
            f"SSBD beta={BETA:.2f} | Mask-k={MASK_K}"
        ),
    )


def reset_live():

    return (
        None,
        "",
        "",
        "",
        "",
        "Ready. Start speaking.",
    )


def translate_text(source_text):

    source_text = (source_text or "").strip()

    if not source_text:
        return "", ""

    full = initial_translation(source_text)

    return (
        full,
        mask_k_for_display(full),
    )


# ============================================================
# Language selection
# ============================================================

LANGUAGES = {
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


def start_session(
    source_language,
    target_language,
):

    global SOURCE_LANGUAGE
    global ASR_LANGUAGE
    global TARGET_LANGUAGE

    if source_language == target_language:

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "Please select two different languages.",
        )

    SOURCE_LANGUAGE = source_language

    ASR_LANGUAGE = LANGUAGES[source_language]

    TARGET_LANGUAGE = target_language

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"Live translation: {SOURCE_LANGUAGE} → {TARGET_LANGUAGE}",
    )


# ============================================================
# Gradio interface
# ============================================================

with gr.Blocks(
    title="Live SSBD Translation"
) as demo:

    # --------------------------------------------------------
    # Screen 1: Language selection
    # --------------------------------------------------------

    with gr.Column(
        visible=True
    ) as language_screen:

        gr.Markdown(
            """
# 🌐 Live Speech Translation

### Select your languages to get started

Choose the language you will speak and the language you want
to translate into.
"""
        )

        source_language_dropdown = gr.Dropdown(
            choices=list(LANGUAGES.keys()),
            value="English",
            label="Input / spoken language",
        )

        target_language_dropdown = gr.Dropdown(
            choices=list(LANGUAGES.keys()),
            value="Japanese",
            label="Output / translation language",
        )

        start_button = gr.Button(
            "Start translating",
            variant="primary",
        )

        language_status = gr.Markdown()

    # --------------------------------------------------------
    # Screen 2: Translation interface
    # --------------------------------------------------------

    with gr.Column(
        visible=False
    ) as translation_screen:

        gr.Markdown(
            f"""
# 🎙️ Live SSBD Translation

Speak into the microphone. ASR updates trigger SSBD translation.

- **SSBD beta:** `{BETA}`
- **Display-only Mask-k:** `{MASK_K}` tokens
"""
        )

        with gr.Tab("Live microphone"):

            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Microphone (start recording to stream)",
            )

            transcript_box = gr.Textbox(
                label="Live ASR transcript",
                lines=3,
            )

            visible_translation = gr.Textbox(
                label=f"Visible translation (Mask-k={MASK_K})",
                lines=4,
            )

            status_box = gr.Textbox(
                label="Status",
            )

            audio_state = gr.State(None)
            transcript_state = gr.State("")
            draft_state = gr.State("")

            mic.stream(
                fn=live_audio_update,
                inputs=[
                    mic,
                    audio_state,
                    transcript_state,
                    draft_state,
                ],
                outputs=[
                    audio_state,
                    transcript_state,
                    draft_state,
                    transcript_box,
                    visible_translation,
                    status_box,
                ],
                stream_every=ASR_UPDATE_SECONDS,
                time_limit=3600,
            )

            reset_btn = gr.Button(
                "Reset session"
            )

            reset_btn.click(
                fn=reset_live,
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

        # ----------------------------------------------------
        # Text translation tab
        # ----------------------------------------------------

        with gr.Tab("Text translation"):

            text_in = gr.Textbox(
                label="Source text",
                lines=4,
            )

            text_btn = gr.Button(
                "Translate"
            )

            full_out = gr.Textbox(
                label="Full translation",
                lines=4,
            )

            masked_out = gr.Textbox(
                label=f"Display-only Mask-k (k={MASK_K})",
                lines=4,
            )

            text_btn.click(
                fn=translate_text,
                inputs=text_in,
                outputs=[
                    full_out,
                    masked_out,
                ],
            )

        gr.Markdown(
            """
**Note:** Faster-Whisper re-transcribes accumulated microphone audio
to provide live updates. This is a prototype streaming-ASR approximation,
not a native streaming ASR decoder. Mask-k is applied only to displayed text.
"""
        )

    # Start button switches from language selection to translator
    start_button.click(
        fn=start_session,
        inputs=[
            source_language_dropdown,
            target_language_dropdown,
        ],
        outputs=[
            language_screen,
            translation_screen,
            language_status,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch()
