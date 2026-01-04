"""Tests for UTF-8 grouping utilities."""

import pytest
import torch

from utf8_tokenizer.groups import UTF8GroupedEmbedding, group_utf8_bytes


class TestGroupUTF8Bytes:
    """Tests for group_utf8_bytes function."""

    def test_example_from_docstring(self):
        """Test the example: [97, 215, 169, 230, 132, 155]."""
        byte_indices = torch.tensor([[97, 215, 169, 230, 132, 155]])
        result = group_utf8_bytes(byte_indices)

        expected = torch.tensor([[[0, 0, 0, 97], [0, 0, 215, 169], [0, 230, 132, 155]]])
        assert result.shape == (1, 3, 4)
        assert torch.all(result == expected)

    def test_ascii_only(self):
        """Test sequence with only ASCII characters."""
        byte_indices = torch.tensor([[65, 66, 67]])  # 'ABC'
        result = group_utf8_bytes(byte_indices)

        expected = torch.tensor([[[0, 0, 0, 65], [0, 0, 0, 66], [0, 0, 0, 67]]])
        assert result.shape == (1, 3, 4)
        assert torch.all(result == expected)

    def test_two_byte_characters(self):
        """Test 2-byte UTF-8 sequences."""
        byte_indices = torch.tensor([[195, 169, 195, 169]])  # 'éé'
        result = group_utf8_bytes(byte_indices)

        expected = torch.tensor([[[0, 0, 195, 169], [0, 0, 195, 169]]])
        assert result.shape == (1, 2, 4)
        assert torch.all(result == expected)

    def test_three_byte_characters(self):
        """Test 3-byte UTF-8 sequences."""
        byte_indices = torch.tensor([[226, 130, 172]])  # '€'
        result = group_utf8_bytes(byte_indices)

        expected = torch.tensor([[[0, 226, 130, 172]]])
        assert result.shape == (1, 1, 4)
        assert torch.all(result == expected)

    def test_four_byte_characters(self):
        """Test 4-byte UTF-8 sequences (emoji)."""
        byte_indices = torch.tensor([[240, 159, 152, 128]])  # '😀'
        result = group_utf8_bytes(byte_indices)

        expected = torch.tensor([[[240, 159, 152, 128]]])
        assert result.shape == (1, 1, 4)
        assert torch.all(result == expected)

    def test_batch_processing(self):
        """Test processing multiple sequences in a batch."""
        byte_indices = torch.tensor([
            [65, 66, 0, 0, 0, 0],  # 'AB' + padding
            [97, 215, 169, 230, 132, 155],  # 'a' + 2-byte + 3-byte
        ])
        result = group_utf8_bytes(byte_indices)

        assert result.shape == (2, 3, 4)
        assert torch.all(result[0, 0] == torch.tensor([0, 0, 0, 65]))
        assert torch.all(result[0, 1] == torch.tensor([0, 0, 0, 66]))
        assert torch.all(result[0, 2] == torch.tensor([0, 0, 0, 0]))
        assert torch.all(result[1, 0] == torch.tensor([0, 0, 0, 97]))
        assert torch.all(result[1, 1] == torch.tensor([0, 0, 215, 169]))
        assert torch.all(result[1, 2] == torch.tensor([0, 230, 132, 155]))

    def test_empty_sequence(self):
        """Test empty/all-padding sequence."""
        byte_indices = torch.tensor([[0, 0, 0]])
        result = group_utf8_bytes(byte_indices)
        assert result.shape == (1, 0, 4)

    def test_preserves_device(self):
        """Test that output is on the same device as input."""
        byte_indices = torch.tensor([[65, 66]])
        result = group_utf8_bytes(byte_indices)
        assert result.device == byte_indices.device


class TestUTF8GroupedEmbedding:
    """Tests for UTF8GroupedEmbedding class."""

    def test_init_valid_embedding_size(self):
        """Test initialization with valid embedding sizes."""
        for size in [64, 128, 256, 512]:
            layer = UTF8GroupedEmbedding(embedding_size=size)
            assert layer.embedding_size == size
            assert layer.byte_dim == size // 4

    def test_init_invalid_embedding_size(self):
        """Test that non-divisible-by-4 sizes raise error."""
        for size in [63, 65, 127, 255]:
            with pytest.raises(ValueError, match="must be divisible by 4"):
                UTF8GroupedEmbedding(embedding_size=size)

    def test_encode_shape(self):
        """Test encode output shape."""
        layer = UTF8GroupedEmbedding(embedding_size=256)
        grouped = torch.randint(0, 256, (10, 5, 4))  # (batch, seq, 4)
        encoded = layer.encode(grouped)
        assert encoded.shape == (10, 5, 256)

    def test_decode_shape(self):
        """Test decode output shapes."""
        layer = UTF8GroupedEmbedding(embedding_size=256)
        embeddings = torch.randn(10, 5, 256)
        decoded, logits = layer.decode(embeddings)

        assert decoded.shape == (10, 5, 4)
        assert logits.shape == (10, 5, 4, 256)

    def test_embedding_is_learnable(self):
        """Test that embedding is a learnable parameter."""
        layer = UTF8GroupedEmbedding(embedding_size=256)
        assert isinstance(layer.embedding, torch.nn.Module)
        assert layer.embedding.weight.requires_grad

    def test_embedding_shape(self):
        """Test embedding weight shape."""
        layer = UTF8GroupedEmbedding(embedding_size=128)
        assert layer.embedding.weight.shape == (256, 32)  # (256 bytes, byte_dim)

    def test_roundtrip_reconstruction(self):
        """Test that encode -> decode reconstructs original bytes."""
        layer = UTF8GroupedEmbedding(embedding_size=256)
        grouped = torch.randint(0, 256, (100, 10, 4))

        encoded = layer.encode(grouped)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == grouped)

    def test_roundtrip_specific_sequences(self):
        """Test roundtrip with specific UTF-8 byte sequences."""
        layer = UTF8GroupedEmbedding(embedding_size=256)

        # Shape: (batch=1, seq=4, bytes=4)
        test_cases = torch.tensor([[
            [0, 0, 0, 65],         # 'A' (1 byte)
            [0, 0, 195, 169],      # 'é' (2 bytes)
            [0, 226, 130, 172],    # '€' (3 bytes)
            [240, 159, 152, 128],  # emoji (4 bytes)
        ]])

        encoded = layer.encode(test_cases)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == test_cases)

    def test_gradients_flow(self):
        """Test that gradients flow through encode and decode."""
        layer = UTF8GroupedEmbedding(embedding_size=256)
        grouped = torch.randint(0, 256, (10, 5, 4))

        encoded = layer.encode(grouped)
        _, logits = layer.decode(encoded)

        loss = logits.sum()
        loss.backward()

        assert layer.embedding.weight.grad is not None
        assert layer.embedding.weight.grad.shape == (256, 64)

    def test_embeddings_normalized(self):
        """Test that embeddings are unit normalized on init."""
        layer = UTF8GroupedEmbedding(embedding_size=256)
        norms = torch.norm(layer.embedding.weight, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_embedding_sizes(self):
        """Test encode/decode with different embedding sizes."""
        for size in [64, 128, 256, 512]:
            layer = UTF8GroupedEmbedding(embedding_size=size)
            grouped = torch.randint(0, 256, (5, 3, 4))

            encoded = layer.encode(grouped)
            decoded, logits = layer.decode(encoded)

            assert encoded.shape == (5, 3, size)
            assert decoded.shape == (5, 3, 4)
            assert logits.shape == (5, 3, 4, 256)
            assert torch.all(decoded == grouped)
