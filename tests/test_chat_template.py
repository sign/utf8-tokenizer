import pytest

from utf8_tokenizer.control import visualize_control_tokens
from utf8_tokenizer.tokenizer import UTF8Tokenizer


def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 22.  # A real function should probably actually get the temperature!


messages = [
    {"role": "system",
     "content": "You are a bot that responds to weather queries. "
                "You should reply with the unit used in the queried location."},
    {"role": "user", "content": "Hey, what's the temperature in Paris right now?"},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function":
            {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}}]},
    {"role": "tool", "content": "22"}
]


def test_chat_template():
    tokenizer = UTF8Tokenizer()

    text = tokenizer.apply_chat_template(
        messages,
        tools=[get_current_temperature],
        add_generation_prompt=True,
        tokenize=False
    )

    expected = """␑␎get_current_temperature
location: The location to get the temperature for, in the format "City, Country"
unit: The unit to return the temperature in.
␏␁system
␎You are a bot that responds to weather queries. You should reply with the unit used in the queried location.␏␗
␁user
␎Hey, what's the temperature in Paris right now?␏␗
␁assistant
␚function get_current_temperature({"location": "Paris, France", "unit": "celsius"})␛␗
␁tool
␎22␏␗
␁assistant
"""

    assert expected == visualize_control_tokens(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
