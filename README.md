# Live Speech Translation with SSBD

A real-time microphone speech translation application built with:

- **Faster-Whisper** for speech recognition.
- **Qwen/Qwen3.5-0.8B** for translation.
- **Self-Speculative Biased Decoding (SSBD)** for incremental translation updates.
- **PyTorch** and **Transformers** for model inference.
- **Gradio** for the browser-based interface.

The application continuously accumulates microphone audio, periodically
re-transcribes the accumulated buffer, and updates the translation whenever
the ASR transcript changes. It is intentionally a low-latency approximation
using repeated transcription; it is not a native streaming ASR decoder.

## Features

- Source and target language selection before a translation session starts.
- Live microphone translation with Gradio streaming events.
- Accumulated-buffer transcription using Faster-Whisper.
- Incremental translation using the previous translation as an SSBD draft.
- Configurable SSBD beta parameter.
- Display-only Mask-k for hiding unstable trailing translation tokens.
- Text translation tab for one-off translation.
- GPU acceleration when CUDA is available.
- Per-session audio, transcript, and translation state through Gradio state
  components.
- Environment-variable configuration without modifying source code.

## How the pipeline works

The live pipeline follows this sequence:

```text
Microphone chunk
       |
       v
Audio normalization and resampling
       |
       v
Append to the current session's audio buffer
       |
       v
Minimum duration reached?
       |
       v
Faster-Whisper re-transcribes the complete buffer
       |
       v
Transcript changed?
       |                       |
      no                      yes
       |                       |
Keep current draft       Qwen + SSBD update
                               |
                               v
                    Full translation draft
                               |
                               v
                    Mask-k display projection
```

### Audio processing

Gradio supplies microphone data as a sample-rate/audio-array tuple. The
translation engine:

1. Validates the incoming tuple.
2. Converts stereo audio to mono.
3. Converts integer samples to normalized `float32`.
4. Resamples audio to 16 kHz.
5. Appends the result to the current session buffer.

ASR is not attempted until `ASR_MIN_AUDIO_SECONDS` has been accumulated.

### Faster-Whisper

Faster-Whisper receives the complete accumulated 16 kHz buffer on each
scheduled update. The current transcription settings are:

- `beam_size=1`
- `vad_filter=True`
- `condition_on_previous_text=False`

The selected source language is converted to its language code before the
transcription call. For example, `French` is passed to Faster-Whisper as
`fr`.

### Qwen and SSBD

The translation model is loaded once when the application starts. The first
transcript update uses ordinary deterministic generation. Later updates use
the previous full translation as a draft:

1. Build the source-language/target-language prompt.
2. Compute the source prompt KV cache and first logits.
3. Tokenize the previous translation draft.
4. Verify draft tokens using the SSBD biased probability rule.
5. Keep the accepted prefix.
6. Generate a replacement suffix from the accepted prefix.
7. Concatenate the accepted prefix and generated suffix.

The mathematical SSBD logic is implemented in
[`src/translation/ssbd.py`](C:/Users/fareh/coding/speech/src/translation/ssbd.py).

### Mask-k

Mask-k is a display policy, not a translation policy:

- The complete translation draft remains in Gradio state.
- The complete draft is passed into the next SSBD iteration.
- Only the final `k` tokenizer tokens are removed from the visible output.
- If the translation contains `k` or fewer tokens, the visible output is
  empty.
- Setting `MASK_K=0` disables masking.

This prevents unstable trailing tokens from being shown while preserving them
internally for subsequent incremental decoding.

## Project structure

```text
speech/
├── main.py                         # Backward-compatible launcher
├── scripts/
│   └── multi_language.py           # Legacy launcher
├── src/
│   ├── main.py                     # Application composition root
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py               # AppConfig and language mappings
│   │   └── llama_config.py         # Existing optional llama.cpp settings
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── prompt.py               # Translation prompt builder
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── whisper.py               # FasterWhisperTranscriber
│   │   └── whisper_stt.py           # Compatibility alias
│   ├── translation/
│   │   ├── __init__.py
│   │   ├── model.py                 # Qwen model/tokenizer wrapper
│   │   └── ssbd.py                  # SSBD and Mask-k implementation
│   ├── core/
│   │   ├── __init__.py
│   │   └── translation_engine.py    # Audio, ASR, and translation orchestration
│   └── ui/
│       ├── __init__.py
│       └── gradio_app.py            # Gradio layout and event callbacks
├── pyproject.toml
└── uv.lock
```

### Main classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `AppConfig` | `src/config/config.py` | Loads typed application settings and defaults. |
| `FasterWhisperTranscriber` | `src/stt/whisper.py` | Loads Faster-Whisper once and transcribes normalized audio. |
| `TranslationPrompt` | `src/prompts/prompt.py` | Builds the Qwen translation chat prompt. |
| `QwenTranslationModel` | `src/translation/model.py` | Owns Qwen loading, tokenization, generation, decoding, and forward passes. |
| `SSBDTranslator` | `src/translation/ssbd.py` | Performs initial translation, draft verification, suffix generation, and Mask-k projection. |
| `TranslationEngine` | `src/core/translation_engine.py` | Coordinates audio preprocessing, ASR, and incremental translation independently of Gradio. |
| `GradioTranslationApp` | `src/ui/gradio_app.py` | Builds the UI and connects events to the translation engine. |

Models are constructed in [`src/main.py`](C:/Users/fareh/coding/speech/src/main.py)
and injected into the engine and UI. They are not loaded inside Gradio
callbacks.

## Requirements

- Windows, Linux, or macOS.
- Python **3.12 or newer**.
- A microphone accessible to the browser.
- A CUDA-capable GPU is recommended for practical live performance.
- Sufficient disk space for the Faster-Whisper and Qwen model downloads.

The project declares these runtime dependencies in
[`pyproject.toml`](C:/Users/fareh/coding/speech/pyproject.toml):

- `faster-whisper`
- `torch`
- `torchvision`
- `torchaudio`
- `transformers`
- `accelerate`
- `gradio`
- `platformdirs`

## Installation

### Using uv

Install `uv`, then create or synchronize the project environment:

```powershell
uv sync
```

The project is configured to obtain the PyTorch packages from the CUDA 12.1
index. If you need a different PyTorch build, adjust the PyTorch source
configuration in `pyproject.toml` according to your hardware and supported
PyTorch wheels.

### Using an existing Python environment

Install the declared project dependencies with your preferred package manager.
For CUDA-enabled PyTorch, install the PyTorch build appropriate for the
installed NVIDIA driver before installing the remaining dependencies.

## Running the application

From the repository root:

```powershell
python -m src.main
```

When using the project environment created by uv:

```powershell
uv run python -m src.main
```

The legacy launchers are retained:

```powershell
python scripts/multi_language.py
python main.py
```

After startup, Gradio prints a local URL. Open that URL in a browser, select
the spoken and target languages, and choose **Start translating**.

## Using the interface

### Live microphone translation

1. Select the input/spoken language.
2. Select the output/translation language.
3. Click **Start translating**.
4. Open the **Live microphone** tab.
5. Start microphone recording.
6. Speak normally.
7. Monitor the ASR transcript and visible translation.
8. Use **Reset session** to clear the current audio buffer, transcript, and
   translation draft.

The interface does not translate unchanged transcripts repeatedly. The
translation callback runs only after a new transcript differs from the
previous session transcript.

### Text translation

The **Text translation** tab performs a regular initial Qwen translation. It
shows both:

- The complete translation.
- The display-only Mask-k version.

The text tab does not use a previous live draft because each text request is
an independent initial translation.

## Configuration

All settings can be provided as environment variables. The defaults below
match the existing implementation.

| Variable | Default | Description |
| --- | --- | --- |
| `SSBD_MODEL` | `Qwen/Qwen3.5-0.8B` | Hugging Face model identifier for translation. |
| `WHISPER_MODEL_SIZE` | `medium` | Faster-Whisper model size, such as `tiny`, `base`, `small`, `medium`, or `large-v3`. |
| `SOURCE_LANGUAGE` | `English` | Initial source-language dropdown value. |
| `ASR_LANGUAGE` | `en` | Fallback Faster-Whisper language code. |
| `TARGET_LANGUAGE` | `Japanese` | Initial target-language dropdown value. |
| `SSBD_BETA` | `0.2` | SSBD draft-token bias parameter. |
| `MASK_K` | `3` | Number of trailing tokens hidden from visible output. |
| `MAX_NEW_TOKENS` | `128` | Maximum number of tokens generated for a translation/suffix. |
| `ASR_UPDATE_SECONDS` | `2.0` | Gradio microphone callback interval. |
| `ASR_MIN_AUDIO_SECONDS` | `1.2` | Minimum accumulated audio duration before transcription. |

### PowerShell example

```powershell
$env:WHISPER_MODEL_SIZE = "small"
$env:SSBD_BETA = "0.2"
$env:MASK_K = "3"
$env:ASR_UPDATE_SECONDS = "1.5"
uv run python -m src.main
```

### Command Prompt example

```bat
set WHISPER_MODEL_SIZE=small
set SSBD_BETA=0.2
set MASK_K=3
set ASR_UPDATE_SECONDS=1.5
python -m src.main
```

## Supported languages

The language selector currently includes:

English, Japanese, Hindi, Spanish, French, German, Chinese (Mandarin),
Korean, Arabic, Portuguese, Italian, Russian, Bengali, Urdu, Tamil, Telugu,
Turkish, Dutch, Thai, and Vietnamese.

The mapping from display names to ASR language codes is defined in
[`src/config/config.py`](C:/Users/fareh/coding/speech/src/config/config.py).

## 📄 Research Paper

This project is based on the research paper:

**[Self-Speculative Biased Decoding for Faster Re-Translation](https://arxiv.org/abs/2509.21740)**

- **Authors:** Linxiao Zeng et al.
- **Publication:** arXiv preprint
- **arXiv ID:** 2509.21740
- **Year:** 2025
- **DOI:** [10.48550/arXiv.2509.21740](https://doi.org/10.48550/arXiv.2509.21740)

**Publication Note:** The referenced work is available as an arXiv preprint. No peer-reviewed conference or journal publication has been verified as of November 2026.

This project presents an independent implementation of the proposed methodology for academic purposes as part of a minor project. Please refer to the original paper for the theoretical foundations and methodology.
