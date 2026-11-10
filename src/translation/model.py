from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import TranslationPrompt


class QwenTranslationModel:
    """Loads Qwen and provides the model operations used by SSBD."""

    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.prompt_builder = TranslationPrompt()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def build_prompt(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> torch.Tensor:
        ids = self.tokenizer.apply_chat_template(
            self.prompt_builder.build(source_language, target_language, source_text),
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
        return ids.to(self.device)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    @torch.inference_mode()
    def forward_with_cache(self, input_ids: torch.Tensor) -> Any:
        return self.model(input_ids=input_ids, use_cache=True)

    @torch.inference_mode()
    def forward_draft(self, input_ids: torch.Tensor, past_key_values: Any) -> Any:
        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )
