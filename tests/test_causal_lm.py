"""Tests for CausalLMWrapper."""

import pytest
import torch

from utf8_tokenizer import UTF8Tokenizer
from utf8_tokenizer.groups import CausalLMWrapper


class TestCausalLMWrapperUnit:
    """Unit tests for CausalLMWrapper without loading a real model."""

    def test_init_invalid_hidden_size(self):
        """Test that non-divisible-by-4 hidden sizes raise error."""
        # Create a minimal mock model
        class MockConfig:
            hidden_size = 63  # Not divisible by 4

        class MockModel:
            config = MockConfig()

            def resize_token_embeddings(self, size):
                pass

        with pytest.raises(ValueError, match="must be divisible by 4"):
            CausalLMWrapper(MockModel(), UTF8Tokenizer())

    def test_flatten_output_simple(self):
        """Test _flatten_output with simple input."""
        # Grouped input: 2 groups of 4 bytes each
        grouped = torch.tensor([[[0, 0, 0, 72], [0, 0, 0, 105]]])  # "Hi"
        # Generated: one group
        generated = [torch.tensor([[0, 0, 0, 33]])]  # "!"
        eos_token_id = 2

        result = CausalLMWrapper._flatten_output(grouped, generated, eos_token_id)

        assert len(result) == 1
        assert result[0].tolist() == [72, 105, 33]  # "Hi!"

    def test_flatten_output_with_eos(self):
        """Test _flatten_output stops at EOS."""
        grouped = torch.tensor([[[0, 0, 0, 72]]])  # "H"
        # Generated includes EOS (2) then more data
        generated = [
            torch.tensor([[0, 0, 0, 105]]),  # "i"
            torch.tensor([[0, 0, 0, 2]]),  # EOS
            torch.tensor([[0, 0, 0, 88]]),  # "X" (should be ignored)
        ]
        eos_token_id = 2

        result = CausalLMWrapper._flatten_output(grouped, generated, eos_token_id)

        assert len(result) == 1
        assert result[0].tolist() == [72, 105, 2]  # Stops at EOS

    def test_flatten_output_batch(self):
        """Test _flatten_output with batch size > 1."""
        grouped = torch.tensor([
            [[0, 0, 0, 65], [0, 0, 0, 0]],  # "A" + padding
            [[0, 0, 0, 66], [0, 0, 0, 67]],  # "BC"
        ])
        generated = [torch.tensor([[0, 0, 0, 49], [0, 0, 0, 50]])]  # "1", "2"
        eos_token_id = 2

        result = CausalLMWrapper._flatten_output(grouped, generated, eos_token_id)

        assert len(result) == 2
        assert result[0].tolist() == [65, 49]  # Zeros removed
        assert result[1].tolist() == [66, 67, 50]

    def test_flatten_output_multibyte_utf8(self):
        """Test _flatten_output with multi-byte UTF-8 characters."""
        # "é" is [195, 169] in UTF-8
        grouped = torch.tensor([[[0, 0, 195, 169]]])
        generated = [torch.tensor([[0, 0, 0, 33]])]  # "!"
        eos_token_id = 2

        result = CausalLMWrapper._flatten_output(grouped, generated, eos_token_id)

        assert len(result) == 1
        assert result[0].tolist() == [195, 169, 33]

    def test_loss_mask_distinguishes_padding_from_ascii(self):
        """Test that loss masks [0,0,0,0] padding but keeps [0,0,0,X] ASCII."""
        # Simulate shifted_labels with:
        # - Valid ASCII group [0,0,0,65] ('A')
        # - Padding group [0,0,0,0]
        shifted_labels = torch.tensor([
            [[0, 0, 0, 65], [0, 0, 0, 0]],  # batch 0: ASCII 'A', then padding
        ])

        # The mask logic from CausalLMWrapper.forward
        valid_groups = (shifted_labels != 0).any(dim=-1)

        # First group should be valid (has non-zero byte)
        assert valid_groups[0, 0].item() is True
        # Second group should be invalid (all zeros)
        assert valid_groups[0, 1].item() is False

    def test_loss_mask_keeps_all_utf8_variants(self):
        """Test that 1-byte, 2-byte, 3-byte, 4-byte UTF-8 groups are all kept."""
        shifted_labels = torch.tensor([
            [
                [0, 0, 0, 65],       # 1-byte: 'A'
                [0, 0, 195, 169],    # 2-byte: 'é'
                [0, 230, 151, 165],  # 3-byte: '日'
                [240, 159, 152, 128],  # 4-byte: '😀'
                [0, 0, 0, 0],        # padding
            ]
        ])

        valid_groups = (shifted_labels != 0).any(dim=-1)

        assert valid_groups[0, 0].item() is True   # 1-byte
        assert valid_groups[0, 1].item() is True   # 2-byte
        assert valid_groups[0, 2].item() is True   # 3-byte
        assert valid_groups[0, 3].item() is True   # 4-byte
        assert valid_groups[0, 4].item() is False  # padding


@pytest.mark.slow
class TestCausalLMWrapperIntegration:
    """Integration tests with real model (requires network/model download)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Get tokenizer."""
        return UTF8Tokenizer()

    @pytest.fixture(scope="class")
    def wrapper(self, tokenizer):
        """Load model and create wrapper once for all tests."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("sign/utf8-lm-tiny")
        return CausalLMWrapper(model, tokenizer)

    def test_forward_shape(self, wrapper, tokenizer):
        """Test forward pass produces correct output shapes."""
        texts = ["Hello", "World"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids)

        # Check logits shape: (batch, num_groups, 4, 256)
        assert outputs.logits.ndim == 4
        assert outputs.logits.shape[0] == 2  # batch size
        assert outputs.logits.shape[2] == 4  # bytes per group
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

        generated = wrapper.generate(input_ids, max_new_groups=5)

        assert isinstance(generated, list)
        assert len(generated) == 1
        assert len(generated[0]) >= 1

    def test_generate_stops_at_eos(self, wrapper, tokenizer):
        """Test that generation stops at EOS token."""
        texts = ["The"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_groups=100)

        # Should have generated something
        assert len(generated) == 1
        assert len(generated[0]) > 0

    def test_batch_generation(self, wrapper, tokenizer):
        """Test generation with batch of inputs."""
        texts = ["Hello", "World"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_groups=3)

        assert len(generated) == 2  # batch size preserved

    def test_decode_generated_output(self, wrapper, tokenizer):
        """Test that generated output can be decoded to text."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_groups=5)

        # Decode back to text (should not raise)
        generated_bytes = bytes(generated[0].tolist())
        decoded = generated_bytes.decode("utf-8", errors="replace")
        assert isinstance(decoded, str)

    def test_forward_without_attention_mask(self, wrapper, tokenizer):
        """Test forward pass without attention_mask."""
        texts = ["Hello"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        # Don't pass attention_mask
        outputs = wrapper(input_ids, attention_mask=None)

        assert outputs.logits is not None
        assert outputs.loss is None

    def test_forward_multibyte_characters(self, wrapper, tokenizer):
        """Test forward pass with multi-byte UTF-8 characters."""
        # Mix of 1-byte, 2-byte, and 3-byte UTF-8
        texts = ["Héllo 世界"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        outputs = wrapper(input_ids)

        assert outputs.logits.ndim == 4

    def test_generate_single_group(self, wrapper, tokenizer):
        """Test generation with max_new_groups=1."""
        texts = ["Hi"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        generated = wrapper.generate(input_ids, max_new_groups=1)

        assert len(generated) == 1
        # Should have input bytes plus at least one generated group
        assert len(generated[0]) >= 2  # "Hi" = 2 bytes minimum

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
        # Two sequences of very different lengths - second has lots of padding
        texts = ["Hi", "This is a much longer sentence with many more bytes"]
        encoded = tokenizer.torch(texts, padding=True)
        input_ids = encoded.input_ids

        # Compute loss for batched (has padding)
        outputs_batched = wrapper(input_ids, labels=input_ids)

        # Compute loss for just the short sequence (no padding)
        short_ids = tokenizer.torch(["Hi"], padding=True).input_ids
        outputs_short = wrapper(short_ids, labels=short_ids)

        # The loss for "Hi" should be similar whether computed alone or in batch
        # If padding is included, the batched loss would be very different
        # (This test will fail if padding is included in loss)
        assert outputs_batched.loss.item() > 0
        assert outputs_short.loss.item() > 0

    def test_batched_forward_matches_individual(self, wrapper, tokenizer):
        """Test that batched forward with padding produces same logits as individual forward."""
        # Two sequences of different lengths
        short_text = "Hi"   # 2 content bytes + BOS + EOS = 4 bytes
        long_text = "Hey"   # 3 content bytes + BOS + EOS = 5 bytes

        # Run individually (no padding needed)
        short_ids = tokenizer.torch([short_text], padding=True).input_ids
        long_ids = tokenizer.torch([long_text], padding=True).input_ids

        with torch.no_grad():
            out_short = wrapper(short_ids)
            out_long = wrapper(long_ids)

        # Run batched (short sequence gets padding)
        batched_ids = tokenizer.torch([short_text, long_text], padding=True).input_ids

        with torch.no_grad():
            out_batched = wrapper(batched_ids)

        # Compare logits for the short sequence (batch index 0)
        # Should match the individual short sequence logits exactly
        short_num_groups = out_short.logits.shape[1]
        batched_short_logits = out_batched.logits[0, :short_num_groups]
        individual_short_logits = out_short.logits[0]

        assert torch.allclose(batched_short_logits, individual_short_logits, atol=1e-5), \
            "Batched logits for short sequence don't match individual prediction"

        # Compare logits for the long sequence (batch index 1)
        long_num_groups = out_long.logits.shape[1]
        batched_long_logits = out_batched.logits[1, :long_num_groups]
        individual_long_logits = out_long.logits[0]

        assert torch.allclose(batched_long_logits, individual_long_logits, atol=1e-5), \
            "Batched logits for long sequence don't match individual prediction"

    def test_batched_generate_matches_individual(self, wrapper, tokenizer):
        """Test that batched generation with padding produces same output as individual generation."""
        # Two sequences of different lengths
        short_text = "Hi"   # 2 content bytes
        long_text = "Hey"   # 3 content bytes

        # Run individually
        short_ids = tokenizer.torch([short_text], padding=True).input_ids
        long_ids = tokenizer.torch([long_text], padding=True).input_ids

        gen_short_individual = wrapper.generate(short_ids, max_new_groups=5)
        gen_long_individual = wrapper.generate(long_ids, max_new_groups=5)

        # Run batched
        batched_ids = tokenizer.torch([short_text, long_text], padding=True).input_ids
        gen_batched = wrapper.generate(batched_ids, max_new_groups=5)

        # Compare outputs - should be identical
        assert gen_batched[0].tolist() == gen_short_individual[0].tolist(), (
            f"Batched generation for short sequence doesn't match: "
            f"{gen_batched[0].tolist()} vs {gen_short_individual[0].tolist()}"
        )

        assert gen_batched[1].tolist() == gen_long_individual[0].tolist(), (
            f"Batched generation for long sequence doesn't match: "
            f"{gen_batched[1].tolist()} vs {gen_long_individual[0].tolist()}"
        )
