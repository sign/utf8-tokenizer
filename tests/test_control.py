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
