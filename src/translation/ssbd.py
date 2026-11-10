from __future__ import annotations

import torch

from .model import QwenTranslationModel


class SSBDTranslator:
    """Performs the existing self-speculative biased decoding algorithm."""

    def __init__(self, model: QwenTranslationModel, beta: float) -> None:
        self.model = model
        self.beta = beta

    @torch.inference_mode()
    def initial_translation(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        prompt_ids = self.model.build_prompt(
            source_text, source_language, target_language
        )
        output_ids = self.model.generate(prompt_ids)
        generated = output_ids[:, prompt_ids.shape[1]:]
        return self.model.decode(generated[0]).strip()

    @torch.inference_mode()
    def prepare_source_cache(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
    ):
        prompt_ids = self.model.build_prompt(
            source_text, source_language, target_language
        )
        output = self.model.forward_with_cache(prompt_ids)
        return output.past_key_values, output.logits[:, -1, :]

    @torch.inference_mode()
    def verify_draft(
        self,
        draft_ids: torch.Tensor,
        past_key_values,
        first_logits: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        draft_ids = draft_ids.to(self.model.device)
        n = draft_ids.shape[1]
        if n == 0:
            return 0, draft_ids

        draft_out = self.model.forward_draft(draft_ids, past_key_values)
        logits_by_position = [first_logits]
        logits_by_position.extend(
            draft_out.logits[:, i - 1, :]
            for i in range(1, n)
        )
        accepted = []
        for i in range(n):
            p = torch.softmax(logits_by_position[i], dim=-1)
            draft_token = draft_ids[0, i]
            biased_p = p * (1.0 - self.beta)
            biased_p[0, draft_token] += self.beta
            if torch.argmax(biased_p, dim=-1).item() != draft_token.item():
                break
            accepted.append(draft_token.item())

        accepted_ids = torch.tensor(
            accepted,
            dtype=torch.long,
            device=self.model.device,
        )
        return len(accepted), accepted_ids

    @torch.inference_mode()
    def generate_suffix(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        accepted_ids: torch.Tensor,
    ) -> torch.Tensor:
        prompt_ids = self.model.build_prompt(
            source_text, source_language, target_language
        )
        if accepted_ids.numel():
            input_ids = torch.cat(
                (prompt_ids, accepted_ids.unsqueeze(0)),
                dim=1,
            )
        else:
            input_ids = prompt_ids
        output_ids = self.model.generate(input_ids)
        return output_ids[:, input_ids.shape[1]:][0]

    def translate(
        self,
        current_source: str,
        previous_translation: str,
        source_language: str,
        target_language: str,
    ) -> str:
        current_source = (current_source or "").strip()
        if not current_source:
            return ""
        if not previous_translation:
            return self.initial_translation(
                current_source, source_language, target_language
            )

        source_cache, first_logits = self.prepare_source_cache(
            current_source, source_language, target_language
        )
        draft_ids = self.model.encode(previous_translation)
        _, accepted_ids = self.verify_draft(
            draft_ids, source_cache, first_logits
        )
        suffix_ids = self.generate_suffix(
            current_source,
            source_language,
            target_language,
            accepted_ids,
        )
        return (
            self.model.decode(accepted_ids)
            + self.model.decode(suffix_ids)
        ).strip()

    def mask_for_display(self, full_translation: str, k: int) -> str:
        if not full_translation or k <= 0:
            return full_translation or ""
        ids = self.model.encode(full_translation)[0]
        if ids.numel() <= k:
            return ""
        return self.model.decode(ids[:-k]).strip()
