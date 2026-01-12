"""Tests for CharacterEmbedding class."""

import pytest
import torch

from utf8_tokenizer import CharacterEmbedding


class TestCharacterEmbeddingInit:
    """Tests for CharacterEmbedding initialization."""

    def test_init_utf16_valid(self):
        """Test initialization for UTF-16 (2 bytes)."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        assert layer.embedding_size == 256
        assert layer.num_bytes == 2
        assert layer.byte_dim == 128

    def test_init_utf32_valid(self):
        """Test initialization for UTF-32 (4 bytes)."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=4)
        assert layer.embedding_size == 256
        assert layer.num_bytes == 4
        assert layer.byte_dim == 64

    def test_init_num_bytes_1_raises(self):
        """Test that num_bytes=1 raises ValueError."""
        with pytest.raises(ValueError, match="num_bytes=1 is not supported"):
            CharacterEmbedding(embedding_size=256, num_bytes=1)

    def test_init_invalid_num_bytes_raises(self):
        """Test that invalid num_bytes values raise ValueError."""
        for num_bytes in [0, 3, 5, 8]:
            with pytest.raises(ValueError, match="num_bytes must be 2 or 4"):
                CharacterEmbedding(embedding_size=256, num_bytes=num_bytes)

    def test_init_indivisible_embedding_size_raises(self):
        """Test that embedding_size not divisible by num_bytes raises error."""
        with pytest.raises(ValueError, match="must be divisible by 2"):
            CharacterEmbedding(embedding_size=255, num_bytes=2)
        with pytest.raises(ValueError, match="must be divisible by 4"):
            CharacterEmbedding(embedding_size=255, num_bytes=4)


class TestCharacterEmbeddingUTF16:
    """Tests for CharacterEmbedding with UTF-16 (2 bytes)."""

    @pytest.fixture
    def layer(self):
        return CharacterEmbedding(embedding_size=256, num_bytes=2)

    def test_split_to_bytes_single(self, layer):
        """Test byte splitting for UTF-16 tokens."""
        tokens = torch.tensor([0x0041, 0x00E9, 0x20AC])  # 'A', 'é', '€'
        bytes_tensor = layer._split_to_bytes(tokens)

        assert bytes_tensor.shape == (3, 2)
        assert torch.all(bytes_tensor[0] == torch.tensor([0x41, 0x00]))  # 'A'
        assert torch.all(bytes_tensor[1] == torch.tensor([0xE9, 0x00]))  # 'é'
        assert torch.all(bytes_tensor[2] == torch.tensor([0xAC, 0x20]))  # '€'

    def test_combine_from_bytes(self, layer):
        """Test byte combining for UTF-16."""
        bytes_tensor = torch.tensor([
            [0x41, 0x00],  # 'A' = 0x0041
            [0xE9, 0x00],  # 'é' = 0x00E9
            [0xAC, 0x20],  # '€' = 0x20AC
        ])
        tokens = layer._combine_from_bytes(bytes_tensor)

        assert torch.all(tokens == torch.tensor([0x0041, 0x00E9, 0x20AC]))

    def test_encode_shape_1d(self, layer):
        """Test encode output shape for 1D input."""
        tokens = torch.randint(0, 65536, (10,))
        encoded = layer.encode(tokens)
        assert encoded.shape == (10, 256)

    def test_encode_shape_2d(self, layer):
        """Test encode output shape for 2D input."""
        tokens = torch.randint(0, 65536, (10, 5))
        encoded = layer.encode(tokens)
        assert encoded.shape == (10, 5, 256)

    def test_decode_shape_2d(self, layer):
        """Test decode output shape for 2D input."""
        embeddings = torch.randn(10, 256)
        decoded, logits = layer.decode(embeddings)

        assert decoded.shape == (10,)
        assert logits.shape == (10, 2, 256)

    def test_decode_shape_3d(self, layer):
        """Test decode output shape for 3D input."""
        embeddings = torch.randn(10, 5, 256)
        decoded, logits = layer.decode(embeddings)

        assert decoded.shape == (10, 5)
        assert logits.shape == (10, 5, 2, 256)

    def test_roundtrip_reconstruction(self, layer):
        """Test that encode -> decode reconstructs original tokens."""
        tokens = torch.randint(0, 65536, (100, 10))

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)

    def test_roundtrip_specific_tokens(self, layer):
        """Test roundtrip with specific UTF-16 code points."""
        tokens = torch.tensor([[
            0x0041,  # 'A'
            0x00E9,  # 'é'
            0x20AC,  # '€'
            0x4E2D,  # '中'
        ]])

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)


class TestCharacterEmbeddingUTF32:
    """Tests for CharacterEmbedding with UTF-32 (4 bytes)."""

    @pytest.fixture
    def layer(self):
        return CharacterEmbedding(embedding_size=256, num_bytes=4)

    def test_split_to_bytes_single(self, layer):
        """Test byte splitting for UTF-32 tokens."""
        tokens = torch.tensor([0x0001F600])  # '😀'
        bytes_tensor = layer._split_to_bytes(tokens)

        assert bytes_tensor.shape == (1, 4)
        assert torch.all(bytes_tensor[0] == torch.tensor([0x00, 0xF6, 0x01, 0x00]))

    def test_combine_from_bytes(self, layer):
        """Test byte combining for UTF-32."""
        bytes_tensor = torch.tensor([[0x00, 0xF6, 0x01, 0x00]])  # '😀'
        tokens = layer._combine_from_bytes(bytes_tensor)

        assert torch.all(tokens == torch.tensor([0x0001F600]))

    def test_encode_shape_1d(self, layer):
        """Test encode output shape for 1D input."""
        tokens = torch.randint(0, 2**20, (10,))
        encoded = layer.encode(tokens)
        assert encoded.shape == (10, 256)

    def test_encode_shape_2d(self, layer):
        """Test encode output shape for 2D input."""
        tokens = torch.randint(0, 2**20, (10, 5))
        encoded = layer.encode(tokens)
        assert encoded.shape == (10, 5, 256)

    def test_decode_shape_2d(self, layer):
        """Test decode output shape for 2D input."""
        embeddings = torch.randn(10, 256)
        decoded, logits = layer.decode(embeddings)

        assert decoded.shape == (10,)
        assert logits.shape == (10, 4, 256)

    def test_decode_shape_3d(self, layer):
        """Test decode output shape for 3D input."""
        embeddings = torch.randn(10, 5, 256)
        decoded, logits = layer.decode(embeddings)

        assert decoded.shape == (10, 5)
        assert logits.shape == (10, 5, 4, 256)

    def test_roundtrip_reconstruction(self, layer):
        """Test that encode -> decode reconstructs original tokens."""
        tokens = torch.randint(0, 2**20, (100, 10))

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)

    def test_roundtrip_emoji(self, layer):
        """Test roundtrip with emoji code points."""
        tokens = torch.tensor([[
            0x0001F600,  # '😀'
            0x0001F4A9,  # '💩'
            0x0001F916,  # '🤖'
        ]])

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)


class TestCharacterEmbeddingGeneral:
    """General tests for CharacterEmbedding."""

    def test_embedding_is_learnable(self):
        """Test that embedding is a learnable parameter."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        assert isinstance(layer.embedding, torch.nn.Module)
        assert layer.embedding.weight.requires_grad

    def test_embedding_shape_utf16(self):
        """Test embedding weight shape for UTF-16."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        assert layer.embedding.weight.shape == (256, 128)  # (256 bytes, byte_dim)

    def test_embedding_shape_utf32(self):
        """Test embedding weight shape for UTF-32."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=4)
        assert layer.embedding.weight.shape == (256, 64)  # (256 bytes, byte_dim)

    def test_byte_shifts_buffer_registered(self):
        """Test that byte_shifts buffer is registered correctly."""
        layer_utf16 = CharacterEmbedding(embedding_size=256, num_bytes=2)
        layer_utf32 = CharacterEmbedding(embedding_size=256, num_bytes=4)

        assert hasattr(layer_utf16, "_byte_shifts")
        assert hasattr(layer_utf32, "_byte_shifts")
        assert torch.all(layer_utf16._byte_shifts == torch.tensor([0, 8]))
        assert torch.all(layer_utf32._byte_shifts == torch.tensor([0, 8, 16, 24]))

    def test_preserves_device(self):
        """Test that output is on the same device as input."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (10, 5))
        encoded = layer.encode(tokens)
        assert encoded.device == tokens.device

    def test_invalid_encode_input_dimension(self):
        """Test that encode raises error for invalid input dimensions."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens_3d = torch.randint(0, 65536, (10, 5, 3))

        with pytest.raises(ValueError, match="Expected 1D or 2D input"):
            layer.encode(tokens_3d)

    def test_invalid_decode_input_dimension(self):
        """Test that decode raises error for invalid input dimensions."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        embeddings_1d = torch.randn(256)
        embeddings_4d = torch.randn(2, 10, 5, 256)

        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            layer.decode(embeddings_1d)
        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            layer.decode(embeddings_4d)

    def test_gradients_flow_utf16(self):
        """Test that gradients flow through encode and decode for UTF-16."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (10, 5))

        encoded = layer.encode(tokens)
        _, logits = layer.decode(encoded)

        loss = logits.sum()
        loss.backward()

        assert layer.embedding.weight.grad is not None

    def test_gradients_flow_utf32(self):
        """Test that gradients flow through encode and decode for UTF-32."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=4)
        tokens = torch.randint(0, 2**20, (10, 5))

        encoded = layer.encode(tokens)
        _, logits = layer.decode(encoded)

        loss = logits.sum()
        loss.backward()

        assert layer.embedding.weight.grad is not None

    def test_embeddings_normalized(self):
        """Test that embeddings are unit normalized on init."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        norms = torch.norm(layer.embedding.weight, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward_equals_encode(self):
        """Test that forward() is equivalent to encode()."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (10, 5))

        encoded = layer.encode(tokens)
        forward_result = layer(tokens)

        assert torch.all(encoded == forward_result)

    def test_decode_compute_decoded_false(self):
        """Test decode with compute_decoded=False returns None."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        embeddings = torch.randn(10, 5, 256)

        decoded, logits = layer.decode(embeddings, compute_decoded=False)

        assert decoded is None
        assert logits is not None

    def test_different_embedding_sizes_utf16(self):
        """Test different embedding sizes for UTF-16."""
        for size in [64, 128, 256, 512]:
            layer = CharacterEmbedding(embedding_size=size, num_bytes=2)
            tokens = torch.randint(0, 65536, (5, 3))

            encoded = layer.encode(tokens)
            decoded, logits = layer.decode(encoded)

            assert encoded.shape == (5, 3, size)
            assert decoded.shape == (5, 3)
            assert logits.shape == (5, 3, 2, 256)
            assert torch.all(decoded == tokens)

    def test_different_embedding_sizes_utf32(self):
        """Test different embedding sizes for UTF-32."""
        for size in [64, 128, 256, 512]:
            layer = CharacterEmbedding(embedding_size=size, num_bytes=4)
            tokens = torch.randint(0, 2**20, (5, 3))

            encoded = layer.encode(tokens)
            decoded, logits = layer.decode(encoded)

            assert encoded.shape == (5, 3, size)
            assert decoded.shape == (5, 3)
            assert logits.shape == (5, 3, 4, 256)
            assert torch.all(decoded == tokens)

    def test_gradient_shape_utf16(self):
        """Test gradient shape for UTF-16."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (10, 5))

        encoded = layer.encode(tokens)
        _, logits = layer.decode(encoded)

        loss = logits.sum()
        loss.backward()

        assert layer.embedding.weight.grad is not None
        assert layer.embedding.weight.grad.shape == (256, 128)

    def test_gradient_shape_utf32(self):
        """Test gradient shape for UTF-32."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=4)
        tokens = torch.randint(0, 2**20, (10, 5))

        encoded = layer.encode(tokens)
        _, logits = layer.decode(encoded)

        loss = logits.sum()
        loss.backward()

        assert layer.embedding.weight.grad is not None
        assert layer.embedding.weight.grad.shape == (256, 64)

    def test_roundtrip_1d_utf16(self):
        """Test roundtrip for 1D input with UTF-16."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (100,))

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)

    def test_roundtrip_1d_utf32(self):
        """Test roundtrip for 1D input with UTF-32."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=4)
        tokens = torch.randint(0, 2**20, (100,))

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)

    def test_edge_case_max_values_utf16(self):
        """Test with maximum UTF-16 values (0xFFFF)."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.tensor([[0x0000, 0xFFFF, 0x7FFF, 0x8000]])

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)

    def test_edge_case_max_values_utf32(self):
        """Test with maximum valid UTF-32 values (Unicode codepoints <= 0x10FFFF)."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=4)
        tokens = torch.tensor([[0x00000000, 0x0010FFFF, 0x0001F600, 0x00000041]])

        encoded = layer.encode(tokens)
        decoded, _ = layer.decode(encoded)

        assert torch.all(decoded == tokens)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (1, 10))

        encoded = layer.encode(tokens)
        decoded, logits = layer.decode(encoded)

        assert encoded.shape == (1, 10, 256)
        assert decoded.shape == (1, 10)
        assert logits.shape == (1, 10, 2, 256)
        assert torch.all(decoded == tokens)

    def test_sequence_length_one(self):
        """Test with sequence length of 1."""
        layer = CharacterEmbedding(embedding_size=256, num_bytes=2)
        tokens = torch.randint(0, 65536, (10, 1))

        encoded = layer.encode(tokens)
        decoded, logits = layer.decode(encoded)

        assert encoded.shape == (10, 1, 256)
        assert decoded.shape == (10, 1)
        assert logits.shape == (10, 1, 2, 256)
        assert torch.all(decoded == tokens)
