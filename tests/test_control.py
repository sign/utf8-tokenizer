import pytest

from utf8_tokenizer.control import ControlTokens, visualize_control_tokens

message = f"""
{ControlTokens.StartOfText}{ControlTokens.StartOfHeading}system
{ControlTokens.ShiftOut}You are a helpful assistant{ControlTokens.ShiftIn}{ControlTokens.EndOfTransmissionBlock}
{ControlTokens.StartOfHeading}user
{ControlTokens.ShiftOut}How much is 1+2?{ControlTokens.ShiftIn}{ControlTokens.EndOfTransmissionBlock}
{ControlTokens.StartOfHeading}assistant
First I'll think about it.
{ControlTokens.Enquiry} The user wants me to calculate, I should call the calculator
{ControlTokens.Substitute}{{``type'': ``calculator'', ``expression'': ``1+2''}}
{ControlTokens.Escape}3{ControlTokens.Acknowledge}
1 + 2 = 3{ControlTokens.EndOfTransmissionBlock}{ControlTokens.EndOfText}
{ControlTokens.Null}{ControlTokens.Null}{ControlTokens.Null}{ControlTokens.Null}
""".strip()

visualized_message = """
␂␁system
␎You are a helpful assistant␏␗
␁user
␎How much is 1+2?␏␗
␁assistant
First I'll think about it.
␅ The user wants me to calculate, I should call the calculator
␚{``type'': ``calculator'', ``expression'': ``1+2''}
␛3␆
1 + 2 = 3␗␃
␀␀␀␀
""".strip()


def test_control_visualization():
    result = visualize_control_tokens(message)
    print("result", result)

    assert result == visualized_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_control_tokens_pattern_matches_non_whitespace_controls():
    """CONTROL_TOKENS_PATTERN is public API: words-segmentation builds its regexes from it (older versions from its
    misspelled former name, kept as an alias)."""
    import re

    from utf8_tokenizer.control import CONTROL_TOKENS_PATTERN, CONTROl_TOKENS_PATTERN

    assert CONTROl_TOKENS_PATTERN == CONTROL_TOKENS_PATTERN
    pattern = re.compile(f"[{CONTROL_TOKENS_PATTERN}]")
    assert [c for c in map(chr, range(128)) if pattern.fullmatch(c)] == \
        [*map(chr, range(0x01, 0x09)), *map(chr, range(0x0E, 0x20)), "\x7f"]
