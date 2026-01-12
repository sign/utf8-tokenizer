"""Causal LM wrapper for UTF-16/UTF-32 character sequences."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from utf8_tokenizer.char_embeddings import CharacterEmbedding
from utf8_tokenizer.tokenizer import EOS_TOKEN_ID

logger = logging.getLogger(__name__)


class CharacterCausalLMConfig(PretrainedConfig):
    """Configuration class for CharacterCausalLMWrapper.

    Args:
        base_model_name_or_path: Name or path of the base CausalLM model.
        num_bytes: Number of bytes per token (2 for UTF-16, 4 for UTF-32).
        **kwargs: Additional config parameters.
    """

    model_type = "character_causal_lm"

    def __init__(
            self,
            base_model_name_or_path: str | None = None,
            num_bytes: int = 2,
            load_base_config: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.num_bytes = num_bytes

        if load_base_config and base_model_name_or_path is not None:
            config = AutoConfig.from_pretrained(base_model_name_or_path)
            for key, value in config.to_dict().items():
                if not hasattr(self, key):
                    setattr(self, key, value)


class CharacterCausalLMWrapper(PreTrainedModel):
    """
    Wraps a CausalLM to work with UTF-16 or UTF-32 character sequences.

    Uses custom byte-decomposed embeddings for character tokens and provides
    autoregressive generation with KV-cache support.

    Args:
        config: CharacterCausalLMConfig with base model configuration.
        model: Optional pre-loaded HuggingFace CausalLM model. If not provided,
            will be loaded from config.base_model_name_or_path.
        tokenizer: Optional tokenizer. If not provided, will create UTF16Tokenizer
            or UTF32Tokenizer based on config.num_bytes.
    """

    config_class = CharacterCausalLMConfig
    base_model_prefix = "character_causal_lm"
    supports_gradient_checkpointing = True

    def __init__(
            self,
            config: CharacterCausalLMConfig,
            model: PreTrainedModel | None = None,
    ):
        super().__init__(config)

        if model is None:
            if config.base_model_name_or_path is None:
                raise ValueError("Either model or config.base_model_name_or_path must be provided")
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

        self.model = model

        num_bytes = config.num_bytes

        hidden_size = model.config.hidden_size
        if hidden_size % num_bytes != 0:
            raise ValueError(f"hidden_size must be divisible by {num_bytes}, got {hidden_size}")

        self.char_embedding = CharacterEmbedding(embedding_size=hidden_size, num_bytes=num_bytes)

        # We resize so that we can access logits
        self.model.resize_token_embeddings(hidden_size)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the underlying model."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the underlying model."""
        self.model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """Return the input embedding layer."""
        return self.char_embedding

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs,
    ) -> CausalLMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            embeds = self.char_embedding.encode(input_ids)
        else:
            embeds = inputs_embeds

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        _, logits = self.char_embedding.decode(outputs.logits, compute_decoded=False)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = self.compute_loss(shifted_logits, shifted_labels)

        return CausalLMOutput(loss=loss, logits=logits)

    @torch.compile()
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for byte-level predictions.

        Args:
            logits: Predicted logits of shape (batch, seq, num_bytes, 256).
            labels: Target token IDs of shape (batch, seq).

        Returns:
            Scalar loss tensor.
        """
        labels_bytes = self.char_embedding._split_to_bytes(labels)

        padding_mask = (labels == 0).unsqueeze(-1).expand_as(labels_bytes)
        labels_bytes = labels_bytes.masked_fill(padding_mask, -100)

        return functional.cross_entropy(
            logits.view(-1, 256),
            labels_bytes.view(-1).long(),
            ignore_index=-100,
        )

    def _prefill(
            self,
            input_ids: torch.Tensor | None,
            attention_mask: torch.Tensor | None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> tuple:
        """Run initial forward pass, return (past_key_values, first_token, positions)."""
        if inputs_embeds is not None:
            embeds = inputs_embeds
            batch_size, seq_len, _ = embeds.shape
            device = embeds.device
        else:
            embeds = self.char_embedding.encode(input_ids)
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        # Get prediction from last position per sequence
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1)
            batch_indices = torch.arange(batch_size, device=device)
            last_positions = (seq_lens - 1).clamp(min=0)
            last_logits = outputs.logits[batch_indices, last_positions].unsqueeze(1)
        else:
            last_logits = outputs.logits[:, -1:, :]

        decoded, _ = self.char_embedding.decode(last_logits)
        first_token = decoded[:, 0]

        current_pos = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
        if attention_mask is not None:
            current_pos = (seq_lens - 1).clamp(min=0)

        return outputs.past_key_values, first_token, current_pos

    def _decode_step(
            self,
            next_token: torch.Tensor,
            past_key_values: tuple,
            current_pos: torch.Tensor,
            decode_mask: torch.Tensor,
    ) -> tuple:
        """Run single decode step with KV cache, return (past_key_values, next_token)."""
        embeds = self.char_embedding.encode(next_token.unsqueeze(1))
        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=decode_mask,
            position_ids=current_pos.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
        )

        decoded, _ = self.char_embedding.decode(outputs.logits)
        next_token = decoded[:, 0]

        return outputs.past_key_values, next_token

    @staticmethod
    def _truncate_at_eos(
            input_ids: torch.Tensor,
            generated: list[torch.Tensor],
            eos_token_id: int = EOS_TOKEN_ID,
    ) -> list[torch.Tensor]:
        """Concatenate input and generated, truncate at EOS, remove padding."""
        batch_size, input_len = input_ids.shape

        gen_stacked = torch.stack(generated, dim=1)
        all_tokens = torch.cat([input_ids, gen_stacked], dim=1)

        result = []
        for seq in all_tokens:
            gen_portion = seq[input_len:]
            eos_pos = (gen_portion == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[: input_len + eos_pos[0] + 1]
            result.append(seq[seq != 0])
        return result

    @staticmethod
    def _truncate_generated_at_eos(
            generated: list[torch.Tensor],
            eos_token_id: int = EOS_TOKEN_ID,
    ) -> list[torch.Tensor]:
        """Truncate generated tokens at EOS, remove padding."""
        gen_stacked = torch.stack(generated, dim=1)

        result = []
        for seq in gen_stacked:
            eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[: eos_pos[0] + 1]
            result.append(seq[seq != 0])
        return result

    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            max_new_tokens: int = 50,
            **kwargs,
    ) -> list[torch.Tensor]:
        """Generate tokens autoregressively using greedy decoding with KV cache.

        Args:
            input_ids: Input token sequences (UTF-16 or UTF-32 code units).
                Either input_ids or inputs_embeds must be provided.
            attention_mask: Optional attention mask for padding.
            inputs_embeds: Pre-computed input embeddings of shape (batch, seq, hidden_size).
                If provided, input_ids is ignored for prefill but used for output concatenation.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments (ignored, for compatibility).

        Returns:
            List of generated token tensors. If inputs_embeds is used without input_ids,
            returns only the generated tokens (not concatenated with input).
        """
        if kwargs:
            logger.warning(f"Ignoring additional generate arguments: {list(kwargs.keys())}")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

        full_mask = torch.ones(batch_size, seq_len + max_new_tokens, device=device, dtype=torch.long)
        if attention_mask is not None:
            full_mask[:, :seq_len] = attention_mask
        mask_len = seq_len

        past_key_values, next_token, current_pos = self._prefill(
            input_ids, attention_mask, inputs_embeds=inputs_embeds
        )

        generated = [next_token]
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens - 1):
            done = done | (next_token == EOS_TOKEN_ID)
            if done.all():
                break

            mask_len += 1
            current_pos = current_pos + 1
            decode_mask = full_mask[:, :mask_len]
            past_key_values, next_token = self._decode_step(
                next_token, past_key_values, current_pos, decode_mask
            )
            generated.append(next_token)

        if input_ids is not None:
            return self._truncate_at_eos(input_ids, generated)

        return self._truncate_generated_at_eos(generated)

    @classmethod
    def from_base_model(
            cls,
            base_model_name_or_path: str,
            num_bytes: int = 2,
            **kwargs,
    ) -> CharacterCausalLMWrapper:
        """Create a CharacterCausalLMWrapper from a base model name/path.

        Args:
            base_model_name_or_path: Name or path of the base CausalLM model.
            num_bytes: Number of bytes per token (2 for UTF-16, 4 for UTF-32).
            **kwargs: Additional arguments passed to from_pretrained for the base model.

        Returns:
            CharacterCausalLMWrapper instance.
        """
        config = CharacterCausalLMConfig(
            base_model_name_or_path=base_model_name_or_path,
            num_bytes=num_bytes,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **kwargs)
        return cls(config=config, model=model)


AutoConfig.register("character_causal_lm", CharacterCausalLMConfig)
AutoModelForCausalLM.register(CharacterCausalLMConfig, CharacterCausalLMWrapper)
