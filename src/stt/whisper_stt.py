from faster_whisper import WhisperModel

class FasterWhisperSTT:
    def __init__(self, model_size="base", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True
        )
        print(f"Detected language: {info.language} ({info.language_probability:.0%})")
        full_text = ""
        for segment in segments:
            #print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
            full_text += segment.text + " "
        return full_text.strip()


if __name__ == "__main__":
    stt = FasterWhisperSTT(model_size="base", device="cuda", compute_type="float16")
    text = stt.transcribe(r"input\C in 100 Seconds.mp3")
    print("\n--- Full Transcript ---")
    print(text)