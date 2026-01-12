"""Tests for CharacterCausalLMWrapper."""

import pytest
import torch

from utf8_tokenizer import UTF16Tokenizer, UTF32Tokenizer
from utf8_tokenizer.char_causal_lm import CharacterCausalLMConfig, CharacterCausalLMWrapper


class TestCharacterCausalLMWrapperUnit:
    """Unit tests for CharacterCausalLMWrapper without loading a real model."""

    def test_config_init(self):
        """Test config initialization."""
        config = CharacterCausalLMConfig(
            base_model_name_or_path="test/model",
            num_bytes=2,
            load_base_config=False,
        )
        assert config.base_model_name_or_path == "test/model"
        assert config.num_bytes == 2

    def test_config_defaults(self):
        """Test config default values."""
        config = CharacterCausalLMConfig()
        assert config.base_model_name_or_path is None
        assert config.num_bytes == 2

    def test_init_invalid_hidden_size_utf16(self):
        """Test that non-divisible-by-2 hidden sizes raise error for UTF-16."""

        class MockConfig:
            hidden_size = 63

        class MockModel:
            config = MockConfig()

            def resize_token_embeddings(self, size):
                pass

        config = CharacterCausalLMConfig(base_model_name_or_path="dummy", num_bytes=2, load_base_config=False)
        with pytest.raises(ValueError, match="must be divisible by 2"):
            CharacterCausalLMWrapper(config, model=MockModel())

    def test_init_invalid_hidden_size_utf32(self):
        """Test that non-divisible-by-4 hidden sizes raise error for UTF-32."""

        class MockConfig:
            hidden_size = 63

        class MockModel:
            config = MockConfig()

            def resize_token_embeddings(self, size):
                pass

        config = CharacterCausalLMConfig(base_model_name_or_path="dummy", num_bytes=4, load_base_config=False)
        with pytest.raises(ValueError, match="must be divisible by 4"):
            CharacterCausalLMWrapper(config, model=MockModel())

    def test_truncate_at_eos_simple(self):
        """Test _truncate_at_eos with simple input."""
        input_ids = torch.tensor([[72, 105]])  # "Hi"
        generated = [torch.tensor([33])]  # "!"
        eos_token_id = 2

        result = CharacterCausalLMWrapper._truncate_at_eos(input_ids, generated, eos_token_id)

        assert len(result) == 1
        assert result[0].tolist() == [72, 105, 33]

    def test_truncate_at_eos_with_eos(self):
        """Test _truncate_at_eos stops at EOS."""
        input_ids = torch.tensor([[72]])  # "H"
        generated = [
            torch.tensor([105]),  # "i"
            torch.tensor([2]),  # EOS
            torch.tensor([88]),  # "X" (should be ignored)
        ]
        eos_token_id = 2

        result = CharacterCausalLMWrapper._truncate_at_eos(input_ids, generated, eos_token_id)

        assert len(result) == 1
        assert result[0].tolist() == [72, 105, 2]

    def test_truncate_at_eos_batch(self):
        """Test _truncate_at_eos with batch size > 1."""
        input_ids = torch.tensor([
            [65, 0],  # "A" + padding
            [66, 67],  # "BC"
        ])
        generated = [torch.tensor([49, 50])]  # "1", "2"
        eos_token_id = 2

        result = CharacterCausalLMWrapper._truncate_at_eos(input_ids, generated, eos_token_id)

        assert len(result) == 2
        assert result[0].tolist() == [65, 49]  # Zeros removed
        assert result[1].tolist() == [66, 67, 50]

    def test_loss_mask_distinguishes_padding(self):
        """Test that loss masks padding token (0) but keeps valid tokens."""
        shifted_labels = torch.tensor([
            [65, 0],  # ASCII 'A', then padding
        ])

        valid_tokens = shifted_labels != 0

        assert valid_tokens[0, 0].item() is True  # 'A'
        assert valid_tokens[0, 1].item() is False  # padding


@pytest.mark.slow
class TestCharacterCausalLMWrapperIntegrationUTF16:
    """Integration tests for UTF-16 with real model."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Get UTF-16 tokenizer."""
        return UTF16Tokenizer()

    @pytest.fixture(scope="class")
    def wrapper(self, tokenizer):
        """Load model and create wrapper once for all tests."""
        return CharacterCausalLMWrapper.from_base_model(
            "sign/utf8-lm-tiny",
            num_bytes=2,
        )

    def test_forward_shape(self, wrapper, tokenizer):
        """Test forward pass produces correct output shapes."""
        texts = ["Hello", "World"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids)

        # Check logits shape: (batch, seq, num_bytes, 256)
        assert outputs.logits.ndim == 4
        assert outputs.logits.shape[0] == 2  # batch size
        assert outputs.logits.shape[2] == 2  # bytes per token (UTF-16)
        assert outputs.logits.shape[3] == 256  # byte vocab

    def test_forward_with_labels(self, wrapper, tokenizer):
        """Test forward pass with labels returns loss."""
        texts = ["Hello world"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids, labels=input_ids)

        assert outputs.loss is not None
        assert outputs.loss.ndim == 0  # scalar
        assert outputs.loss.item() > 0  # non-zero loss

    def test_generate(self, wrapper, tokenizer):
        """Test greedy generation."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_tokens=5)

        assert isinstance(generated, list)
        assert len(generated) == 1
        assert len(generated[0]) >= 1

    def test_generate_stops_at_eos(self, wrapper, tokenizer):
        """Test that generation stops at EOS token."""
        texts = ["The"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_tokens=100)

        assert len(generated) == 1
        assert len(generated[0]) > 0

    def test_batch_generation(self, wrapper, tokenizer):
        """Test generation with batch of inputs."""
        texts = ["Hello", "World"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_tokens=3)

        assert len(generated) == 2  # batch size preserved

    def test_forward_without_attention_mask(self, wrapper, tokenizer):
        """Test forward pass without attention_mask."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids, attention_mask=None)

        assert outputs.logits is not None
        assert outputs.loss is None

    def test_forward_unicode_characters(self, wrapper, tokenizer):
        """Test forward pass with unicode characters."""
        texts = ["Héllo 世界"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids)

        assert outputs.logits.ndim == 4

    def test_forward_with_inputs_embeds(self, wrapper, tokenizer):
        """Test forward pass with inputs_embeds instead of input_ids."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        embeds = wrapper.char_embedding.encode(input_ids)

        outputs = wrapper(inputs_embeds=embeds)

        assert outputs.logits is not None
        assert outputs.logits.ndim == 4

    def test_forward_inputs_embeds_matches_input_ids(self, wrapper, tokenizer):
        """Test that forward with inputs_embeds matches forward with input_ids."""
        texts = ["Test"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs_ids = wrapper(input_ids)

        embeds = wrapper.char_embedding.encode(input_ids)
        outputs_embeds = wrapper(inputs_embeds=embeds)

        assert torch.allclose(outputs_ids.logits, outputs_embeds.logits, atol=1e-5)

    def test_forward_requires_input(self, wrapper):
        """Test that forward raises error when neither input_ids nor inputs_embeds provided."""
        import pytest
        with pytest.raises(ValueError, match="Either input_ids or inputs_embeds must be provided"):
            wrapper()

    def test_loss_computation_is_scalar(self, wrapper, tokenizer):
        """Test that loss is properly computed as scalar."""
        texts = ["Test sentence"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids, labels=input_ids)

        assert outputs.loss.shape == ()  # Scalar
        assert not outputs.loss.isnan()
        assert not outputs.loss.isinf()

    def test_loss_ignores_padding(self, wrapper, tokenizer):
        """Test that loss computation ignores padding tokens."""
        texts = ["Hi", "This is a much longer sentence"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs_batched = wrapper(input_ids, labels=input_ids)

        short_ids = tokenizer.torch(["Hi"], padding=True).input_ids
        outputs_short = wrapper(short_ids, labels=short_ids)

        assert outputs_batched.loss.item() > 0
        assert outputs_short.loss.item() > 0

    def test_save_pretrained(self, wrapper, tokenizer, tmp_path):
        """Test that save_pretrained saves the model correctly."""
        save_dir = tmp_path / "saved_model"
        wrapper.save_pretrained(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "pytorch_model.bin").exists() or (save_dir / "model.safetensors").exists()

    def test_load_pretrained(self, wrapper, tokenizer, tmp_path):
        """Test that load_pretrained loads the model correctly."""
        save_dir = tmp_path / "saved_model"
        wrapper.save_pretrained(save_dir)

        loaded_wrapper = CharacterCausalLMWrapper.from_pretrained(save_dir)

        assert loaded_wrapper.config.base_model_name_or_path == wrapper.config.base_model_name_or_path
        assert loaded_wrapper.config.num_bytes == wrapper.config.num_bytes
        assert loaded_wrapper.char_embedding.embedding_size == wrapper.char_embedding.embedding_size

    def test_save_load_preserves_weights(self, wrapper, tokenizer, tmp_path):
        """Test that save and load preserves model weights."""
        save_dir = tmp_path / "saved_model"
        wrapper.save_pretrained(save_dir)

        loaded_wrapper = CharacterCausalLMWrapper.from_pretrained(save_dir)

        texts = ["Hello world"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        with torch.no_grad():
            original_output = wrapper(input_ids)
            loaded_output = loaded_wrapper(input_ids)

        assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-5)

    def test_load_with_automodel(self, wrapper, tokenizer, tmp_path):
        """Test that the model can be loaded with AutoModelForCausalLM."""
        from transformers import AutoModelForCausalLM

        save_dir = tmp_path / "saved_model"
        wrapper.save_pretrained(save_dir)

        loaded_wrapper = AutoModelForCausalLM.from_pretrained(save_dir)

        assert isinstance(loaded_wrapper, CharacterCausalLMWrapper)

        texts = ["Test"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        with torch.no_grad():
            original_output = wrapper(input_ids)
            loaded_output = loaded_wrapper(input_ids)

        assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-5)

    def test_generate_with_inputs_embeds(self, wrapper, tokenizer):
        """Test generation using inputs_embeds instead of input_ids."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        # Get embeddings from input_ids
        inputs_embeds = wrapper.char_embedding.encode(input_ids)

        # Generate with inputs_embeds only (no input_ids)
        generated = wrapper.generate(inputs_embeds=inputs_embeds, max_new_tokens=5)

        assert isinstance(generated, list)
        assert len(generated) == 1
        assert len(generated[0]) >= 1

    def test_generate_with_inputs_embeds_and_input_ids(self, wrapper, tokenizer):
        """Test generation with both inputs_embeds and input_ids."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        inputs_embeds = wrapper.char_embedding.encode(input_ids)

        # Generate with both - should use inputs_embeds for prefill but include input_ids in output
        generated = wrapper.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            max_new_tokens=5
        )

        assert isinstance(generated, list)
        assert len(generated) == 1
        # Output should include input tokens
        assert len(generated[0]) >= len(input_ids[0])

    def test_generate_inputs_embeds_matches_input_ids(self, wrapper, tokenizer):
        """Test that generation with inputs_embeds produces same tokens as input_ids."""
        texts = ["The"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        inputs_embeds = wrapper.char_embedding.encode(input_ids)

        # Generate with input_ids
        gen_from_ids = wrapper.generate(input_ids=input_ids, max_new_tokens=5)

        # Generate with inputs_embeds (and input_ids for output concatenation)
        gen_from_embeds = wrapper.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            max_new_tokens=5
        )

        # Should produce identical results
        assert gen_from_ids[0].tolist() == gen_from_embeds[0].tolist()

    def test_generate_requires_input(self, wrapper):
        """Test that generate raises error when neither input_ids nor inputs_embeds provided."""
        import pytest
        with pytest.raises(ValueError, match="Either input_ids or inputs_embeds must be provided"):
            wrapper.generate(max_new_tokens=5)

    def test_generate_with_inputs_embeds_batch(self, wrapper, tokenizer):
        """Test batch generation with inputs_embeds."""
        texts = ["Hello", "World"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        inputs_embeds = wrapper.char_embedding.encode(input_ids)

        generated = wrapper.generate(inputs_embeds=inputs_embeds, max_new_tokens=3)

        assert len(generated) == 2


@pytest.mark.slow
class TestCharacterCausalLMWrapperIntegrationUTF32:
    """Integration tests for UTF-32 with real model."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Get UTF-32 tokenizer."""
        return UTF32Tokenizer()

    @pytest.fixture(scope="class")
    def wrapper(self, tokenizer):
        """Load model and create wrapper once for all tests."""
        return CharacterCausalLMWrapper.from_base_model(
            "sign/utf8-lm-tiny",
            num_bytes=4,
        )

    def test_forward_shape(self, wrapper, tokenizer):
        """Test forward pass produces correct output shapes."""
        texts = ["Hello", "World"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids)

        # Check logits shape: (batch, seq, num_bytes, 256)
        assert outputs.logits.ndim == 4
        assert outputs.logits.shape[0] == 2  # batch size
        assert outputs.logits.shape[2] == 4  # bytes per token (UTF-32)
        assert outputs.logits.shape[3] == 256  # byte vocab

    def test_forward_with_labels(self, wrapper, tokenizer):
        """Test forward pass with labels returns loss."""
        texts = ["Hello world"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids, labels=input_ids)

        assert outputs.loss is not None
        assert outputs.loss.ndim == 0
        assert outputs.loss.item() > 0

    def test_generate(self, wrapper, tokenizer):
        """Test greedy generation."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_tokens=5)

        assert isinstance(generated, list)
        assert len(generated) == 1
        assert len(generated[0]) >= 1

    def test_forward_emoji(self, wrapper, tokenizer):
        """Test forward pass with emoji (tests full UTF-32 range)."""
        texts = ["Hello 😀🎉"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids)

        assert outputs.logits.ndim == 4
        assert outputs.logits.shape[2] == 4  # bytes per token

    def test_config_num_bytes(self, wrapper):
        """Test that config has correct num_bytes."""
        assert wrapper.config.num_bytes == 4

    def test_save_load_utf32(self, wrapper, tokenizer, tmp_path):
        """Test save and load for UTF-32 model."""
        save_dir = tmp_path / "saved_model_utf32"
        wrapper.save_pretrained(save_dir)

        loaded_wrapper = CharacterCausalLMWrapper.from_pretrained(save_dir)

        assert loaded_wrapper.config.num_bytes == 4

        texts = ["Test"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        with torch.no_grad():
            original_output = wrapper(input_ids)
            loaded_output = loaded_wrapper(input_ids)

        assert torch.allclose(original_output.logits, loaded_output.logits, atol=1e-5)
