# Speech Translation

Live microphone speech translation using Faster-Whisper, Qwen, SSBD, PyTorch,
Transformers, and Gradio.

## Run

```powershell
python -m src.main
```

The legacy launcher remains available:

```powershell
python scripts/multi_language.py
```

## Configuration

The application preserves these environment variables and defaults:

| Variable | Default |
| --- | --- |
| `SSBD_MODEL` | `Qwen/Qwen3.5-0.8B` |
| `WHISPER_MODEL_SIZE` | `medium` |
| `SOURCE_LANGUAGE` | `English` |
| `ASR_LANGUAGE` | `en` |
| `TARGET_LANGUAGE` | `Japanese` |
| `SSBD_BETA` | `0.2` |
| `MASK_K` | `3` |
| `MAX_NEW_TOKENS` | `128` |
| `ASR_UPDATE_SECONDS` | `2.0` |
| `ASR_MIN_AUDIO_SECONDS` | `1.2` |

The live pipeline intentionally re-transcribes the accumulated microphone
buffer. Mask-k only affects displayed text; the complete draft remains in the
session state for the next SSBD update.
