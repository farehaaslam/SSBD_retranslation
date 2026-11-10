from __future__ import annotations


class TranslationPrompt:
    """Builds the chat messages used by the translation model."""

    def build(self, source_language: str, target_language: str, source_text: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    f"Translate {source_language} to {target_language}. "
                    f"Return only the {target_language} translation."
                ),
            },
            {"role": "user", "content": source_text},
        ]