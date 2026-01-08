import tempfile

import pytest
import torch

from utf8_tokenizer.tokenizer import UTF8Tokenizer


@pytest.fixture
def tokenizer():
    return UTF8Tokenizer()


def test_basic_tokenization(tokenizer):
    """Test basic tokenization without special tokens."""
    text = "hello"
    result = tokenizer._original_call(text, add_special_tokens=False)

    # Expected: [104, 101, 108, 108, 111] (UTF-8 bytes for "hello")
    expected_ids = [104, 101, 108, 108, 111]
    assert result.input_ids == expected_ids
    assert result.attention_mask == [1] * len(expected_ids)


def test_unicode_string(tokenizer):
    """Test tokenization of a unicode string."""
    text = "héllo עמית"
    result = tokenizer._original_call(text, add_special_tokens=False)

    decoded = tokenizer.decode(result.input_ids)
    assert decoded == text


def test_tokenization_with_special_tokens(tokenizer):
    """Test tokenization with BOS and EOS tokens."""
    text = "hello"
    result = tokenizer._original_call(text, add_special_tokens=True)

    # Expected: [2, 104, 101, 108, 108, 111, 3] (BOS + "hello" + EOS)
    expected_ids = [2, 104, 101, 108, 108, 111, 3]
    assert result.input_ids == expected_ids
    assert result.attention_mask == [1] * len(expected_ids)


def test_torch_method_basic(tokenizer):
    """Test torch method with basic functionality."""
    texts = ["hello", "world"]
    result = tokenizer.torch(texts, add_special_tokens=False)

    # Check shapes and types
    assert isinstance(result.input_ids, list)
    assert isinstance(result.attention_mask, list)
    assert len(result.input_ids) == 2
    assert len(result.attention_mask) == 2

    # Check individual tensors
    for i, text in enumerate(texts):
        expected_ids = [ord(c) for c in text]
        assert result.input_ids[i].tolist() == expected_ids
        assert result.input_ids[i].dtype in [torch.uint8, torch.int64]
        assert result.attention_mask[i].dtype == torch.bool
        assert result.attention_mask[i].tolist() == [1] * len(expected_ids)


def test_torch_method_with_special_tokens(tokenizer):
    """Test torch method with special tokens."""
    texts = ["hello", "world"]
    result = tokenizer.torch(texts, add_special_tokens=True)

    # Check that BOS and EOS are added
    for i, text in enumerate(texts):
        expected_ids = [2] + [ord(c) for c in text] + [3]  # BOS + text + EOS
        assert result.input_ids[i].tolist() == expected_ids
        assert result.attention_mask[i].tolist() == [1] * len(expected_ids)


def test_torch_method_with_padding(tokenizer):
    """Test torch method with padding enabled."""
    texts = ["hi", "hello world"]  # Different lengths
    result = tokenizer.torch(texts, add_special_tokens=False, padding=True)

    # Check that tensors are padded to same length
    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.attention_mask, torch.Tensor)

    max_len = max(len(text) for text in texts)
    assert result.input_ids.shape == (2, max_len)
    assert result.attention_mask.shape == (2, max_len)

    # Check padding values
    assert result.input_ids[0, -1] == 0  # Pad token
    assert result.attention_mask[0, -1] == 0  # Padded attention


def test_torch_method_with_truncation(tokenizer):
    """Test torch method with truncation."""
    texts = ["hello world this is a long text"]
    max_length = 5
    result = tokenizer.torch(texts, add_special_tokens=False, truncation=True, max_length=max_length)

    # Check truncation
    assert len(result.input_ids[0]) == max_length
    assert len(result.attention_mask[0]) == max_length

    # Verify truncated content matches first 5 characters
    expected_ids = [ord(c) for c in texts[0][:max_length]]
    assert result.input_ids[0].tolist() == expected_ids


def test_torch_method_special_tokens_with_padding(tokenizer):
    """Test torch method with both special tokens and padding."""
    texts = ["hi", "hello"]
    result = tokenizer.torch(texts, add_special_tokens=True, padding=True)

    # Check dimensions
    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.attention_mask, torch.Tensor)

    # Both sequences should have BOS + text + EOS, padded to same length
    max_len = max(len(text) + 2 for text in texts)  # +2 for BOS/EOS
    assert result.input_ids.shape == (2, max_len)
    assert result.attention_mask.shape == (2, max_len)

    # Check first sequence: BOS + "hi" + EOS + padding
    expected_first = [2, ord("h"), ord("i"), 3, 0, 0, 0]
    assert result.input_ids[0].tolist() == expected_first
    assert result.attention_mask[0].tolist() == [1, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_comparison_tokenizer_vs_torch_method(add_special_tokens, tokenizer):
    """Test that tokenizer._original_call() and torch() methods produce compatible results."""
    texts = ["test string"]

    result1 = tokenizer._original_call(texts, add_special_tokens=add_special_tokens, return_tensors="pt")
    result2 = tokenizer.torch(texts, add_special_tokens=add_special_tokens, padding=False)

    assert torch.equal(result1.input_ids[0], result2.input_ids[0])
    assert torch.equal(result1.attention_mask[0], result2.attention_mask[0])


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_comparison_tokenizer_vs_torch_method_max_length(add_special_tokens, tokenizer):
    texts = ["test string"]

    params = dict(add_special_tokens=add_special_tokens, max_length=1, truncation=True)
    result1 = tokenizer._original_call(texts, return_tensors="pt", **params)
    result2 = tokenizer.torch(texts, **params)

    assert torch.equal(result1.input_ids[0], result2.input_ids[0])
    assert torch.equal(result1.attention_mask[0], result2.attention_mask[0])


def test_comparison_tokenizer_vs_torch_method_multiple_strings(tokenizer):
    texts = ["test string", "shorter"]

    result1 = tokenizer._original_call(texts, padding=True, return_tensors="pt")
    result2 = tokenizer.torch(texts, padding=True)

    assert torch.equal(result1.input_ids[0], result2.input_ids[0])
    assert torch.equal(result1.attention_mask[0], result2.attention_mask[0])


def test_empty_string(tokenizer):
    """Test tokenization of empty string."""
    result = tokenizer._original_call("", add_special_tokens=False)
    assert result.input_ids == []
    assert result.attention_mask == []

    result_with_special = tokenizer._original_call("", add_special_tokens=True)
    assert result_with_special.input_ids == [2, 3]  # BOS + EOS
    assert result_with_special.attention_mask == [1, 1]


def test_special_characters(tokenizer):
    """Test tokenization with special characters and unicode."""
    text = "hello\nworld\t!"
    result = tokenizer._original_call(text, add_special_tokens=False)

    expected_ids = [ord(c) for c in text]
    assert result.input_ids == expected_ids
    assert result.attention_mask == [1] * len(expected_ids)


def test_dtype_consistency(tokenizer):
    """Test that dtypes are consistent across different methods."""
    texts = ["hello", "world"]
    result = tokenizer.torch(texts, padding=True)

    assert result.input_ids.dtype in [torch.uint8, torch.int64]
    assert result.attention_mask.dtype == torch.bool


def test_batch_consistency(tokenizer):
    """Test that batch processing produces consistent results."""
    texts = ["hello", "world", "test"]

    # Process as batch
    batch_result = tokenizer.torch(texts, add_special_tokens=True, padding=True)

    # Process individually
    individual_results = []
    for text in texts:
        result = tokenizer.torch([text], add_special_tokens=True, padding=False)
        individual_results.append(result.input_ids[0])

    # Check that unpadded portions match
    for i, individual_ids in enumerate(individual_results):
        batch_ids = batch_result.input_ids[i][: len(individual_ids)]
        assert batch_ids.tolist() == individual_ids.tolist()


def test_vocab_properties(tokenizer):
    """Test tokenizer vocabulary properties."""
    assert tokenizer.vocab_size == 256  # 2^8
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3

    vocab = tokenizer.get_vocab()
    assert len(vocab) == 256
    assert vocab[chr(0)] == 0
    assert vocab[chr(2)] == 2
    assert vocab[chr(3)] == 3


def test_tokenizer_save_and_load():
    """Test saving and loading the tokenizer."""
    original_tokenizer = UTF8Tokenizer()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the tokenizer
        original_tokenizer.save_pretrained(temp_dir)

        # Load the tokenizer
        UTF8Tokenizer.from_pretrained(temp_dir)


class TestTorchMethodEdgeCases:
    """Additional edge case tests for the .torch() method."""

    @pytest.fixture
    def tokenizer(self):
        return UTF8Tokenizer()

    def test_single_element(self, tokenizer):
        """Test torch method with a single element."""
        result = tokenizer.torch(["hello"], padding=True)
        assert isinstance(result.input_ids, torch.Tensor)
        assert result.input_ids.shape == (1, 7)  # BOS + hello + EOS

    def test_all_empty_strings_no_special_tokens(self, tokenizer):
        """Test with all empty strings and no special tokens - extreme edge case."""
        result = tokenizer.torch(["", ""], padding=True, add_special_tokens=False)
        assert isinstance(result.input_ids, torch.Tensor)
        assert result.input_ids.shape == (2, 0)  # Empty tensor
        assert result.attention_mask.shape == (2, 0)

    def test_all_empty_strings_with_special_tokens(self, tokenizer):
        """Test with all empty strings but with special tokens."""
        result = tokenizer.torch(["", ""], padding=True, add_special_tokens=True)
        assert result.input_ids.shape == (2, 2)  # BOS + EOS only
        assert result.input_ids[0].tolist() == [2, 3]
        assert result.input_ids[1].tolist() == [2, 3]

    def test_device_parameter_cpu(self, tokenizer):
        """Test torch method with device parameter (CPU)."""
        result = tokenizer.torch(["hello", "hi"], padding=True, device="cpu")
        assert result.input_ids.device == torch.device("cpu")
        assert result.attention_mask.device == torch.device("cpu")

    def test_device_parameter_no_padding(self, tokenizer):
        """Test torch method with device parameter without padding."""
        result = tokenizer.torch(["hello", "hi"], padding=False, device="cpu")
        assert isinstance(result.input_ids, list)
        assert result.input_ids[0].device == torch.device("cpu")
        assert result.attention_mask[0].device == torch.device("cpu")

    def test_unicode_strings_with_padding(self, tokenizer):
        """Test unicode strings with padding."""
        texts = ["שלום", "hello", "世界"]  # Hebrew, English, Chinese
        result = tokenizer.torch(texts, padding=True, add_special_tokens=True)
        assert isinstance(result.input_ids, torch.Tensor)
        # Hebrew שלום = 8 bytes, Chinese 世界 = 6 bytes, so Hebrew + BOS/EOS = 10
        max_len = max(len(t.encode("utf-8")) for t in texts) + 2
        assert result.input_ids.shape == (3, max_len)

    def test_truncation_with_padding(self, tokenizer):
        """Test truncation combined with padding."""
        texts = ["hello world this is long", "hi"]
        result = tokenizer.torch(texts, padding=True, truncation=True, max_length=10, add_special_tokens=True)
        assert result.input_ids.shape == (2, 10)  # Both truncated/padded to max_length
        # First should be truncated, second should be padded
        assert result.attention_mask[0].sum().item() == 10  # All active (truncated)
        assert result.attention_mask[1].sum().item() == 4  # BOS + hi + EOS = 4

    def test_truncation_without_special_tokens(self, tokenizer):
        """Test truncation without special tokens."""
        texts = ["hello world"]
        result = tokenizer.torch(texts, truncation=True, max_length=5, add_special_tokens=False)
        assert len(result.input_ids[0]) == 5
        assert result.input_ids[0].tolist() == [ord(c) for c in "hello"]

    def test_string_with_null_bytes(self, tokenizer):
        """Test string containing null bytes (which is pad token)."""
        text = "a\x00b"
        result = tokenizer.torch([text], padding=False, add_special_tokens=False)
        # Null byte should be preserved
        assert result.input_ids[0].tolist() == [ord("a"), 0, ord("b")]

    def test_string_with_null_bytes_padding(self, tokenizer):
        """Test strings with null bytes and padding - null bytes ARE padding by design."""
        texts = ["a\x00b", "xy"]
        result = tokenizer.torch(texts, padding=True, add_special_tokens=False)
        # First string: a, \x00, b (length 3)
        # Second string: x, y, pad (length 3)
        assert result.input_ids.shape == (2, 3)
        # Null bytes (0x00) are the pad token, so they're marked as not attended
        assert result.attention_mask[0].tolist() == [True, False, True]  # a, pad, b
        assert result.attention_mask[1].tolist() == [True, True, False]  # xy + pad

    def test_very_long_strings(self, tokenizer):
        """Test with very long strings."""
        long_text = "a" * 10000
        result = tokenizer.torch([long_text], padding=False, add_special_tokens=True)
        assert len(result.input_ids[0]) == 10002  # BOS + 10000 + EOS

    def test_mixed_lengths_attention_mask(self, tokenizer):
        """Test that attention mask correctly identifies padded positions."""
        texts = ["abc", "a", "abcde"]
        result = tokenizer.torch(texts, padding=True, add_special_tokens=False)
        # Lengths: 3, 1, 5 -> padded to 5
        assert result.attention_mask[0].tolist() == [True, True, True, False, False]
        assert result.attention_mask[1].tolist() == [True, False, False, False, False]
        assert result.attention_mask[2].tolist() == [True, True, True, True, True]

    def test_return_tensors_pt_via_call(self, tokenizer):
        """Test that __call__ with return_tensors='pt' uses torch method."""
        texts = ["hello", "world"]
        result = tokenizer(texts, padding=True, return_tensors="pt")
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)

    def test_return_tensors_other_via_call(self, tokenizer):
        """Test that __call__ with other return_tensors falls back to original."""
        texts = ["hello"]
        result = tokenizer(texts, return_tensors=None, add_special_tokens=False)
        # Without return_tensors, should use original call
        assert result.input_ids == [[104, 101, 108, 108, 111]]

    def test_truncation_warning_no_max_length(self, tokenizer):
        """Test that truncation without max_length issues warning."""
        import warnings
        texts = ["hello world"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = tokenizer.torch(texts, truncation=True, max_length=None, add_special_tokens=False)
            assert len(w) == 1
            assert "no maximum length is provided" in str(w[0].message)
            # Should return untouched since truncation was disabled
            assert len(result.input_ids[0]) == 11


class TestUTF16Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        from utf8_tokenizer import UTF16Tokenizer
        return UTF16Tokenizer()

    def test_basic_tokenization(self, tokenizer):
        text = "hello"
        result = tokenizer.torch([text], add_special_tokens=False)
        # UTF-16: each ASCII char is one 16-bit code unit
        assert result.input_ids[0].tolist() == [104, 101, 108, 108, 111]

    def test_unicode_roundtrip(self, tokenizer):
        text = "你好世界"
        result = tokenizer.torch([text], add_special_tokens=False, padding=True)
        decoded = tokenizer.decode(result.input_ids[0].tolist())
        assert decoded == text

    def test_emoji_surrogate_pairs(self, tokenizer):
        text = "😀"
        result = tokenizer.torch([text], add_special_tokens=False)
        # Emoji requires surrogate pair in UTF-16 (2 code units)
        assert len(result.input_ids[0]) == 2


class TestUTF32Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        from utf8_tokenizer import UTF32Tokenizer
        return UTF32Tokenizer()

    def test_basic_tokenization(self, tokenizer):
        text = "hello"
        result = tokenizer.torch([text], add_special_tokens=False)
        # UTF-32: each char is one 32-bit code unit
        assert result.input_ids[0].tolist() == [104, 101, 108, 108, 111]

    def test_unicode_roundtrip(self, tokenizer):
        text = "你好世界"
        result = tokenizer.torch([text], add_special_tokens=False, padding=True)
        decoded = tokenizer.decode(result.input_ids[0].tolist())
        assert decoded == text

    def test_emoji_single_codepoint(self, tokenizer):
        text = "😀"
        result = tokenizer.torch([text], add_special_tokens=False)
        # Emoji is single code point in UTF-32 (1 code unit)
        assert len(result.input_ids[0]) == 1
        assert result.input_ids[0].tolist() == [128512]  # U+1F600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
