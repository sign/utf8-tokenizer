"""Tests for UTF8ValidationLogitsProcessor."""

import pytest
import torch

from utf8_tokenizer.logits_processor import UTF8ValidationLogitsProcessor


@pytest.fixture
def processor():
    """Create a UTF8ValidationLogitsProcessor instance."""
    return UTF8ValidationLogitsProcessor()


class TestUTF8State:
    """Test the internal UTF-8 state detection."""

    def test_empty_sequence(self, processor):
        """Test that empty sequence is considered complete."""
        state = processor._get_utf8_state([])
        assert state["complete"] is True

    def test_ascii_complete(self, processor):
        """Test that ASCII characters are complete."""
        # Single ASCII character
        state = processor._get_utf8_state([0x41])  # 'A'
        assert state["complete"] is True

        # Multiple ASCII characters
        state = processor._get_utf8_state([0x48, 0x65, 0x6C, 0x6C, 0x6F])  # "Hello"
        assert state["complete"] is True

    def test_two_byte_incomplete(self, processor):
        """Test incomplete 2-byte sequence."""
        state = processor._get_utf8_state([0xC2])  # Start of 2-byte sequence
        assert state["complete"] is False
        assert state["first_byte"] == 0xC2
        assert state["position"] == 1

    def test_two_byte_complete(self, processor):
        """Test complete 2-byte sequence."""
        state = processor._get_utf8_state([0xC2, 0xA9])  # © symbol
        assert state["complete"] is True

    def test_three_byte_incomplete_position_1(self, processor):
        """Test incomplete 3-byte sequence at position 1."""
        state = processor._get_utf8_state([0xE2])  # Start of 3-byte sequence
        assert state["complete"] is False
        assert state["first_byte"] == 0xE2
        assert state["position"] == 1

    def test_three_byte_incomplete_position_2(self, processor):
        """Test incomplete 3-byte sequence at position 2."""
        state = processor._get_utf8_state([0xE2, 0x82])  # Partial 3-byte sequence
        assert state["complete"] is False
        assert state["first_byte"] == 0xE2
        assert state["position"] == 2

    def test_three_byte_complete(self, processor):
        """Test complete 3-byte sequence."""
        state = processor._get_utf8_state([0xE2, 0x82, 0xAC])  # € symbol
        assert state["complete"] is True

    def test_four_byte_incomplete_position_1(self, processor):
        """Test incomplete 4-byte sequence at position 1."""
        state = processor._get_utf8_state([0xF0])  # Start of 4-byte sequence
        assert state["complete"] is False
        assert state["first_byte"] == 0xF0
        assert state["position"] == 1

    def test_four_byte_incomplete_position_2(self, processor):
        """Test incomplete 4-byte sequence at position 2."""
        state = processor._get_utf8_state([0xF0, 0x9F])
        assert state["complete"] is False
        assert state["first_byte"] == 0xF0
        assert state["position"] == 2

    def test_four_byte_incomplete_position_3(self, processor):
        """Test incomplete 4-byte sequence at position 3."""
        state = processor._get_utf8_state([0xF0, 0x9F, 0x98])
        assert state["complete"] is False
        assert state["first_byte"] == 0xF0
        assert state["position"] == 3

    def test_four_byte_complete(self, processor):
        """Test complete 4-byte sequence."""
        state = processor._get_utf8_state([0xF0, 0x9F, 0x98, 0x80])  # 😀 emoji
        assert state["complete"] is True


class TestValidStartBytes:
    """Test valid UTF-8 start bytes."""

    def test_valid_start_bytes(self, processor):
        """Test that all valid start bytes are included."""
        valid = processor._valid_start_bytes()

        # ASCII: 0x00-0x7F
        for i in range(0x00, 0x80):
            assert i in valid

        # 2-byte: 0xC2-0xDF
        for i in range(0xC2, 0xE0):
            assert i in valid

        # 3-byte: 0xE0-0xEF
        for i in range(0xE0, 0xF0):
            assert i in valid

        # 4-byte: 0xF0-0xF4
        for i in range(0xF0, 0xF5):
            assert i in valid

    def test_invalid_start_bytes_excluded(self, processor):
        """Test that invalid start bytes are not included."""
        valid = processor._valid_start_bytes()

        # Continuation bytes: 0x80-0xBF
        for i in range(0x80, 0xC0):
            assert i not in valid

        # Invalid: 0xC0-0xC1
        assert 0xC0 not in valid
        assert 0xC1 not in valid

        # Invalid: 0xF5-0xFF
        for i in range(0xF5, 0x100):
            assert i not in valid


class TestTwoByteSequences:
    """Test 2-byte UTF-8 sequences (C2-DF)."""

    def test_two_byte_continuation(self, processor):
        """Test that 2-byte sequences require 80-BF as second byte."""
        # Test with C2
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xC2]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

        # Test with DF
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xDF]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected


class TestThreeByteSequences:
    """Test 3-byte UTF-8 sequences (E0-EF)."""

    def test_e0_second_byte(self, processor):
        """Test E0 requires A0-BF as second byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xE0]))
        expected = set(range(0xA0, 0xC0))
        assert allowed == expected

    def test_e0_third_byte(self, processor):
        """Test E0 requires 80-BF as third byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xE0, 0xA0]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

    def test_ed_second_byte(self, processor):
        """Test ED requires 80-9F as second byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xED]))
        expected = set(range(0x80, 0xA0))
        assert allowed == expected

    def test_ed_third_byte(self, processor):
        """Test ED requires 80-BF as third byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xED, 0x80]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

    def test_e1_ec_second_byte(self, processor):
        """Test E1-EC require 80-BF as second byte."""
        for first_byte in [0xE1, 0xE5, 0xEC]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected

    def test_e1_ec_third_byte(self, processor):
        """Test E1-EC require 80-BF as third byte."""
        for first_byte in [0xE1, 0xE5, 0xEC]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte, 0x80]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected

    def test_ee_ef_second_byte(self, processor):
        """Test EE-EF require 80-BF as second byte."""
        for first_byte in [0xEE, 0xEF]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected

    def test_ee_ef_third_byte(self, processor):
        """Test EE-EF require 80-BF as third byte."""
        for first_byte in [0xEE, 0xEF]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte, 0x80]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected


class TestFourByteSequences:
    """Test 4-byte UTF-8 sequences (F0-F4)."""

    def test_f0_second_byte(self, processor):
        """Test F0 requires 90-BF as second byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xF0]))
        expected = set(range(0x90, 0xC0))
        assert allowed == expected

    def test_f0_third_byte(self, processor):
        """Test F0 requires 80-BF as third byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xF0, 0x90]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

    def test_f0_fourth_byte(self, processor):
        """Test F0 requires 80-BF as fourth byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xF0, 0x90, 0x80]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

    def test_f4_second_byte(self, processor):
        """Test F4 requires 80-8F as second byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xF4]))
        expected = set(range(0x80, 0x90))
        assert allowed == expected

    def test_f4_third_byte(self, processor):
        """Test F4 requires 80-BF as third byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xF4, 0x80]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

    def test_f4_fourth_byte(self, processor):
        """Test F4 requires 80-BF as fourth byte."""
        allowed = processor._get_allowed_next_bytes(torch.tensor([0xF4, 0x80, 0x80]))
        expected = set(range(0x80, 0xC0))
        assert allowed == expected

    def test_f1_f3_second_byte(self, processor):
        """Test F1-F3 require 80-BF as second byte."""
        for first_byte in [0xF1, 0xF2, 0xF3]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected

    def test_f1_f3_third_byte(self, processor):
        """Test F1-F3 require 80-BF as third byte."""
        for first_byte in [0xF1, 0xF2, 0xF3]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte, 0x80]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected

    def test_f1_f3_fourth_byte(self, processor):
        """Test F1-F3 require 80-BF as fourth byte."""
        for first_byte in [0xF1, 0xF2, 0xF3]:
            allowed = processor._get_allowed_next_bytes(torch.tensor([first_byte, 0x80, 0x80]))
            expected = set(range(0x80, 0xC0))
            assert allowed == expected


class TestLogitsProcessing:
    """Test the main logits processing functionality."""

    def test_call_masks_invalid_bytes(self, processor):
        """Test that __call__ properly masks invalid bytes."""
        # Start with a 2-byte sequence starter (C2)
        input_ids = torch.tensor([[0xC2]])
        scores = torch.zeros((1, 256))

        processed_scores = processor(input_ids, scores)

        # Only bytes 80-BF should be allowed (not -inf)
        for i in range(256):
            if 0x80 <= i < 0xC0:
                assert processed_scores[0, i] != float("-inf")
            else:
                assert processed_scores[0, i] == float("-inf")

    def test_call_batch_processing(self, processor):
        """Test batch processing with different sequences."""
        # Batch of 3: one ASCII, one 2-byte start, one 3-byte start
        input_ids = torch.tensor([[0x41], [0xC2], [0xE2]])
        scores = torch.zeros((3, 256))

        processed_scores = processor(input_ids, scores)

        # First sequence (ASCII complete) - should allow all start bytes
        valid_start = processor._valid_start_bytes()
        for i in range(256):
            if i in valid_start:
                assert processed_scores[0, i] != float("-inf")
            else:
                assert processed_scores[0, i] == float("-inf")

        # Second sequence (2-byte incomplete) - should allow 80-BF
        for i in range(256):
            if 0x80 <= i < 0xC0:
                assert processed_scores[1, i] != float("-inf")
            else:
                assert processed_scores[1, i] == float("-inf")

        # Third sequence (3-byte incomplete) - should allow 80-BF
        for i in range(256):
            if 0x80 <= i < 0xC0:
                assert processed_scores[2, i] != float("-inf")
            else:
                assert processed_scores[2, i] == float("-inf")

    def test_empty_sequence_allows_all_start_bytes(self, processor):
        """Test that empty sequence allows all valid start bytes."""
        input_ids = torch.tensor([[]])
        scores = torch.zeros((1, 256))

        processed_scores = processor(input_ids, scores)

        valid_start = processor._valid_start_bytes()
        for i in range(256):
            if i in valid_start:
                assert processed_scores[0, i] != float("-inf")
            else:
                assert processed_scores[0, i] == float("-inf")


class TestRealWorldSequences:
    """Test with real-world UTF-8 sequences."""

    def test_copyright_symbol(self, processor):
        """Test © symbol (U+00A9): C2 A9."""
        # After C2, should allow 80-BF
        input_ids = torch.tensor([[0xC2]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)

        # A9 should be allowed
        assert processed[0, 0xA9] != float("-inf")
        # 7F should not be allowed (not in 80-BF range)
        assert processed[0, 0x7F] == float("-inf")

    def test_euro_symbol(self, processor):
        """Test € symbol (U+20AC): E2 82 AC."""
        # After E2, should allow 80-BF
        input_ids = torch.tensor([[0xE2]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)
        assert processed[0, 0x82] != float("-inf")

        # After E2 82, should allow 80-BF
        input_ids = torch.tensor([[0xE2, 0x82]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)
        assert processed[0, 0xAC] != float("-inf")

    def test_emoji(self, processor):
        """Test 😀 emoji (U+1F600): F0 9F 98 80."""
        # After F0, should allow 90-BF
        input_ids = torch.tensor([[0xF0]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)
        assert processed[0, 0x9F] != float("-inf")
        assert processed[0, 0x8F] == float("-inf")  # Below 90

        # After F0 9F, should allow 80-BF
        input_ids = torch.tensor([[0xF0, 0x9F]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)
        assert processed[0, 0x98] != float("-inf")

        # After F0 9F 98, should allow 80-BF
        input_ids = torch.tensor([[0xF0, 0x9F, 0x98]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)
        assert processed[0, 0x80] != float("-inf")

    def test_mixed_sequence(self, processor):
        """Test a mixed sequence: Hello© (48 65 6C 6C 6F C2 A9)."""
        # After complete "Hello©", should allow all start bytes
        input_ids = torch.tensor([[0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xC2, 0xA9]])
        scores = torch.zeros((1, 256))
        processed = processor(input_ids, scores)

        valid_start = processor._valid_start_bytes()
        for i in range(256):
            if i in valid_start:
                assert processed[0, i] != float("-inf")
            else:
                assert processed[0, i] == float("-inf")


class TestWithActualModel:
    """Test UTF8ValidationLogitsProcessor with an actual language model."""

    @pytest.mark.parametrize(
        ("prefix_bytes", "expected_range_start", "expected_range_end"),
        [
            # 2-byte sequences
            ([0xC2], 0x80, 0xBF),  # C2 -> 80-BF
            ([0xDF], 0x80, 0xBF),  # DF -> 80-BF
            # 3-byte sequences
            ([0xE0], 0xA0, 0xBF),  # E0 -> A0-BF (special)
            ([0xE1], 0x80, 0xBF),  # E1 -> 80-BF
            ([0xED], 0x80, 0x9F),  # ED -> 80-9F (special)
            ([0xEE], 0x80, 0xBF),  # EE -> 80-BF
            # 4-byte sequences
            ([0xF0], 0x90, 0xBF),  # F0 -> 90-BF (special)
            ([0xF1], 0x80, 0xBF),  # F1 -> 80-BF
            ([0xF4], 0x80, 0x8F),  # F4 -> 80-8F (special)
            # Continuation bytes
            ([0xE2, 0x82], 0x80, 0xBF),  # E2 82 -> 80-BF
            ([0xF0, 0x9F], 0x80, 0xBF),  # F0 9F -> 80-BF
            ([0xF0, 0x9F, 0x98], 0x80, 0xBF),  # F0 9F 98 -> 80-BF
        ],
    )
    def test_model_generation_with_processor(self, prefix_bytes, expected_range_start, expected_range_end):
        """
        Test that the processor correctly constrains generation with an actual language model.

        This test loads a small language model, resizes it to 256 tokens, and verifies that
        after generating with specific UTF-8 prefixes, the next byte falls within the expected range.
        """
        from transformers import AutoModelForCausalLM

        # Load the model
        model = AutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm", torch_dtype="auto")

        # Resize model to 256 tokens for byte-level generation
        model.resize_token_embeddings(256)
        model.eval()

        # Create processor
        processor = UTF8ValidationLogitsProcessor()

        # Create input with the prefix bytes
        input_ids = torch.tensor([prefix_bytes])

        # Generate a single token with the processor
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,  # Use greedy decoding
                logits_processor=[processor],
                pad_token_id=0,
            )

        # Get the generated byte (last token in the output)
        generated_byte = outputs[0, -1].item()

        # Verify the generated byte is in the expected range
        assert expected_range_start <= generated_byte <= expected_range_end, (
            f"Generated byte {generated_byte:02X} not in expected range "
            f"{expected_range_start:02X}-{expected_range_end:02X}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
