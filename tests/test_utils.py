import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from utf8_tokenizer.utils import pad_bytearrays_to_tensor, pad_bytearrays_to_tensor_loop


def reference_pad_sequence(bytearrays: list[bytearray], padding_value: int = 0) -> torch.Tensor:
    """Reference implementation using torch.nn.utils.rnn.pad_sequence."""
    tensors = [torch.frombuffer(b, dtype=torch.uint8) for b in bytearrays]
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


class TestPadBytearraysToTensor:
    """Tests for pad_bytearrays_to_tensor implementations."""

    @pytest.fixture
    def sample_bytearrays(self):
        return [
            bytearray(b"hello"),
            bytearray(b"hi"),
            bytearray(b"world!"),
        ]

    @pytest.fixture
    def unicode_bytearrays(self):
        return [
            bytearray("שלום".encode()),
            bytearray(b"hello"),
            bytearray("世界".encode()),
        ]

    @pytest.fixture
    def single_bytearray(self):
        return [bytearray(b"test")]

    @pytest.fixture
    def empty_bytearrays(self):
        return [bytearray(b""), bytearray(b"a"), bytearray(b"")]

    # Test vectorized implementation matches reference
    @pytest.mark.parametrize("padding_value", [0, 255, 42])
    def test_vectorized_matches_reference(self, sample_bytearrays, padding_value):
        expected = reference_pad_sequence(sample_bytearrays, padding_value)
        result = pad_bytearrays_to_tensor(sample_bytearrays, torch.uint8, padding_value)
        assert torch.equal(result, expected)

    def test_vectorized_matches_reference_unicode(self, unicode_bytearrays):
        expected = reference_pad_sequence(unicode_bytearrays)
        result = pad_bytearrays_to_tensor(unicode_bytearrays)
        assert torch.equal(result, expected)

    def test_vectorized_matches_reference_single(self, single_bytearray):
        expected = reference_pad_sequence(single_bytearray)
        result = pad_bytearrays_to_tensor(single_bytearray)
        assert torch.equal(result, expected)

    def test_vectorized_matches_loop_with_empty(self, empty_bytearrays):
        # Note: torch.frombuffer can't handle empty buffers, so we compare implementations
        loop_result = pad_bytearrays_to_tensor_loop(empty_bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(empty_bytearrays)
        assert torch.equal(loop_result, vectorized_result)
        # Check shape is correct
        assert vectorized_result.shape == (3, 1)  # 3 items, max_len=1

    # Test loop implementation matches reference
    @pytest.mark.parametrize("padding_value", [0, 255, 42])
    def test_loop_matches_reference(self, sample_bytearrays, padding_value):
        expected = reference_pad_sequence(sample_bytearrays, padding_value)
        result = pad_bytearrays_to_tensor_loop(sample_bytearrays, torch.uint8, padding_value)
        assert torch.equal(result, expected)

    def test_loop_matches_reference_unicode(self, unicode_bytearrays):
        expected = reference_pad_sequence(unicode_bytearrays)
        result = pad_bytearrays_to_tensor_loop(unicode_bytearrays)
        assert torch.equal(result, expected)

    # Test vectorized matches loop implementation
    def test_vectorized_matches_loop(self, sample_bytearrays):
        loop_result = pad_bytearrays_to_tensor_loop(sample_bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(sample_bytearrays)
        assert torch.equal(loop_result, vectorized_result)

    def test_vectorized_matches_loop_unicode(self, unicode_bytearrays):
        loop_result = pad_bytearrays_to_tensor_loop(unicode_bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(unicode_bytearrays)
        assert torch.equal(loop_result, vectorized_result)

    # Test output properties
    def test_output_shape(self, sample_bytearrays):
        result = pad_bytearrays_to_tensor(sample_bytearrays)
        max_len = max(len(b) for b in sample_bytearrays)
        assert result.shape == (len(sample_bytearrays), max_len)

    def test_output_dtype(self, sample_bytearrays):
        result = pad_bytearrays_to_tensor(sample_bytearrays)
        assert result.dtype == torch.uint8

    def test_padding_applied_correctly(self):
        bytearrays = [bytearray(b"ab"), bytearray(b"abcd")]
        result = pad_bytearrays_to_tensor(bytearrays, torch.uint8, 0)

        # First row: "ab" + padding
        assert result[0].tolist() == [ord("a"), ord("b"), 0, 0]
        # Second row: "abcd" no padding
        assert result[1].tolist() == [ord("a"), ord("b"), ord("c"), ord("d")]

    def test_content_preserved(self, sample_bytearrays):
        result = pad_bytearrays_to_tensor(sample_bytearrays)

        for i, b in enumerate(sample_bytearrays):
            # Check that non-padded content matches original
            assert result[i, : len(b)].tolist() == list(b)

    # Edge cases
    def test_all_same_length(self):
        bytearrays = [bytearray(b"abc"), bytearray(b"def"), bytearray(b"ghi")]
        expected = reference_pad_sequence(bytearrays)
        result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(result, expected)
        assert result.shape == (3, 3)  # No padding needed

    def test_large_batch(self):
        bytearrays = [bytearray(f"text{i}".encode()) for i in range(1000)]
        expected = reference_pad_sequence(bytearrays)
        result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(result, expected)

    def test_varying_lengths(self):
        bytearrays = [
            bytearray(b"a"),
            bytearray(b"ab"),
            bytearray(b"abc"),
            bytearray(b"abcd"),
            bytearray(b"abcde"),
        ]
        expected = reference_pad_sequence(bytearrays)
        result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(result, expected)

    def test_all_empty_bytearrays(self):
        """Test with all empty bytearrays - edge case that produces (n, 0) tensor."""
        bytearrays = [bytearray(b""), bytearray(b""), bytearray(b"")]
        loop_result = pad_bytearrays_to_tensor_loop(bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(loop_result, vectorized_result)
        assert vectorized_result.shape == (3, 0)  # All empty

    def test_single_empty_bytearray(self):
        """Test with a single empty bytearray."""
        bytearrays = [bytearray(b"")]
        loop_result = pad_bytearrays_to_tensor_loop(bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(loop_result, vectorized_result)
        assert vectorized_result.shape == (1, 0)

    def test_bytes_input(self):
        """Test that bytes (not bytearray) work correctly - matches type annotation."""
        # The function signature says list[bytes], so test with bytes
        byte_list = [b"hello", b"hi", b"world!"]
        result = pad_bytearrays_to_tensor(byte_list)
        # Compare with bytearray version
        bytearray_list = [bytearray(b) for b in byte_list]
        expected = pad_bytearrays_to_tensor(bytearray_list)
        assert torch.equal(result, expected)

    def test_bytes_input_with_unicode(self):
        """Test bytes input with unicode content."""
        byte_list = ["שלום".encode(), b"hello", "世界".encode()]
        result = pad_bytearrays_to_tensor(byte_list)
        bytearray_list = [bytearray(b) for b in byte_list]
        expected = pad_bytearrays_to_tensor(bytearray_list)
        assert torch.equal(result, expected)

    def test_mixed_empty_positions(self):
        """Test empty bytearrays at different positions."""
        bytearrays = [bytearray(b""), bytearray(b"abc"), bytearray(b""), bytearray(b"de")]
        loop_result = pad_bytearrays_to_tensor_loop(bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(loop_result, vectorized_result)
        assert vectorized_result.shape == (4, 3)

    def test_very_long_sequences(self):
        """Test with very long byte sequences."""
        bytearrays = [
            bytearray(b"a" * 10000),
            bytearray(b"b" * 5000),
            bytearray(b"c" * 15000),
        ]
        loop_result = pad_bytearrays_to_tensor_loop(bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(loop_result, vectorized_result)
        assert vectorized_result.shape == (3, 15000)

    def test_null_bytes(self):
        """Test bytearrays containing null bytes."""
        bytearrays = [bytearray(b"\x00\x00\x00"), bytearray(b"a\x00b"), bytearray(b"\x00")]
        loop_result = pad_bytearrays_to_tensor_loop(bytearrays)
        vectorized_result = pad_bytearrays_to_tensor(bytearrays)
        assert torch.equal(loop_result, vectorized_result)
        # Verify null bytes are preserved
        assert vectorized_result[0, 0].item() == 0
        assert vectorized_result[1, 1].item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
