"""Causal LM wrapper for UTF-8 grouped byte sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from utf8_tokenizer import UTF8Tokenizer

from .embedding import UTF8GroupedEmbedding
from .group_utf8_bytes import group_utf8_bytes


class CausalLMWrapper(nn.Module):
    """
    Wraps a CausalLM to work with UTF-8 grouped byte sequences.

    Groups UTF-8 bytes into 4-byte groups (padded), uses custom embeddings,
    and provides autoregressive generation with KV-cache support.

    Args:
        model: A HuggingFace CausalLM model (hidden_size must be divisible by 4).
        tokenizer: UTF8Tokenizer for encoding/decoding bytes.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: UTF8Tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config

        hidden_size = model.config.hidden_size
        if hidden_size % 4 != 0:
            raise ValueError(f"hidden_size must be divisible by 4, got {hidden_size}")  # noqa: TRY003

        # Resize lm_head to output hidden_size so we can use logits directly for decoding
        # This avoids the overhead of output_hidden_states=True
        self.model.resize_token_embeddings(hidden_size)

        self.utf8_embedding = UTF8GroupedEmbedding(embedding_size=hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        grouped = group_utf8_bytes(input_ids)
        embeds = self.utf8_embedding.encode(grouped)

        # Create attention mask: valid groups have at least one non-zero byte
        mask = (grouped != 0).any(dim=-1).long() if attention_mask is not None else None

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=mask,
            **kwargs
        )
        # Use logits directly (lm_head outputs hidden_size, matching decode input)
        # Skip argmax during forward - only need logits for loss computation
        _, logits = self.utf8_embedding.decode(outputs.logits, compute_decoded=False)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1].contiguous()
            # Reuse grouped if labels is the same tensor as input_ids
            grouped_labels = grouped if labels is input_ids else group_utf8_bytes(labels)
            shifted_labels = grouped_labels[:, 1:].contiguous()

            # Mask out full padding groups [0,0,0,0], but keep valid ASCII like [0,0,0,63]
            # A group is padding only if ALL 4 bytes are 0
            valid_groups = (shifted_labels != 0).any(dim=-1)  # (batch, num_groups-1)
            valid_mask = valid_groups.unsqueeze(-1).expand_as(shifted_labels)  # (batch, num_groups-1, 4)

            loss = functional.cross_entropy(
                shifted_logits.view(-1, 256),
                shifted_labels.view(-1).long(),
                reduction="none",
            )
            loss = (loss * valid_mask.reshape(-1)).sum() / valid_mask.sum().clamp(min=1)

        return CausalLMOutput(loss=loss, logits=logits)

    @staticmethod
    def _get_seq_lens(grouped: torch.Tensor) -> torch.Tensor:
        """Count non-padding groups per sequence."""
        return (grouped != 0).any(dim=-1).sum(dim=1)

    def _prefill(self, grouped: torch.Tensor, seq_lens: torch.Tensor) -> tuple:
        """Run initial forward pass, return (past_key_values, first_group, positions, mask)."""
        device = grouped.device
        batch_size, num_groups, _ = grouped.shape

        embeds = self.utf8_embedding.encode(grouped)
        group_mask = (grouped != 0).any(dim=-1).long()
        position_ids = torch.arange(num_groups, device=device).unsqueeze(0).expand(batch_size, -1)

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=group_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        # Get prediction from last valid position per sequence
        # Use logits directly (lm_head outputs hidden_size)
        batch_indices = torch.arange(batch_size, device=device)
        last_positions = (seq_lens - 1).clamp(min=0)
        last_logits = outputs.logits[batch_indices, last_positions].unsqueeze(1)
        decoded, _ = self.utf8_embedding.decode(last_logits)
        first_group = decoded[:, 0]

        return outputs.past_key_values, first_group, seq_lens.clone(), group_mask

    def _decode_step(
        self,
        next_group: torch.Tensor,
        past_key_values: tuple,
        current_pos: torch.Tensor,
        decode_mask: torch.Tensor,
    ) -> tuple:
        """Run single decode step with KV cache, return (past_key_values, next_group)."""
        embeds = self.utf8_embedding.encode(next_group.unsqueeze(1))
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=decode_mask,
            position_ids=current_pos.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Use logits directly (lm_head outputs hidden_size)
        decoded, _ = self.utf8_embedding.decode(outputs.logits)
        next_group = decoded[:, 0]

        return outputs.past_key_values, next_group

    @staticmethod
    def _flatten_output(
        grouped: torch.Tensor, generated: list[torch.Tensor], eos_token_id: int
    ) -> list[torch.Tensor]:
        """Flatten groups, truncate at generated EOS, remove padding zeros."""
        batch_size, num_input_groups, _ = grouped.shape
        input_bytes = num_input_groups * 4

        # Stack generated: list of (batch, 4) -> (batch, num_gen, 4)
        gen_stacked = torch.stack(generated, dim=1)
        all_groups = torch.cat([grouped, gen_stacked], dim=1)
        flat = all_groups.flatten(1)

        # Per-sequence processing (variable-length outputs)
        result = []
        for seq in flat:
            # Only look for EOS in generated portion (after input_bytes)
            gen_portion = seq[input_bytes:]
            eos_pos = (gen_portion == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                # Truncate at first EOS in generated portion
                seq = seq[: input_bytes + eos_pos[0] + 1]
            result.append(seq[seq > 0])
        return result

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_groups: int = 50) -> list[torch.Tensor]:
        """Generate bytes autoregressively using greedy decoding with KV cache."""
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        device = input_ids.device

        grouped = group_utf8_bytes(input_ids)
        num_groups = grouped.shape[1]
        seq_lens = self._get_seq_lens(grouped)

        # Pre-allocate full mask buffer (prefill + max decode steps)
        full_mask = torch.ones(batch_size, num_groups + max_new_groups, device=device, dtype=torch.long)

        past_key_values, next_group, current_pos, prefill_mask = self._prefill(grouped, seq_lens)
        # Copy prefill mask into buffer
        full_mask[:, :num_groups] = prefill_mask
        mask_len = num_groups

        generated = [next_group]
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_groups - 1):
            done = done | (next_group == eos_token_id).any(dim=-1)
            if done.all():
                break

            # Use slice of pre-allocated mask (no allocation)
            mask_len += 1
            decode_mask = full_mask[:, :mask_len]
            past_key_values, next_group = self._decode_step(next_group, past_key_values, current_pos, decode_mask)
            current_pos = current_pos + 1
            generated.append(next_group)

        return self._flatten_output(grouped, generated, eos_token_id)
