import re


class ControlTokens:
    Null = "\x00"
    StartOfHeading = "\x01"
    StartOfText = "\x02"
    EndOfText = "\x03"
    EndOfTransmission = "\x04"
    Enquiry = "\x05"
    Acknowledge = "\x06"
    Alert = "\x07"
    Backspace = "\x08"
    HorizontalTab = "\x09"  # Whitespace
    LineFeed = "\x0A"  # Whitespace
    VerticalTab = "\x0B"  # Whitespace
    FormFeed = "\x0C"  # Whitespace
    CarriageReturn = "\x0D"  # Whitespace
    ShiftOut = "\x0E"
    ShiftIn = "\x0F"
    DataLinkEscape = "\x10"
    DeviceControl1 = "\x11"
    DeviceControl2 = "\x12"
    DeviceControl3 = "\x13"
    DeviceControl4 = "\x14"
    NegativeAcknowledge = "\x15"
    SynchronousIdle = "\x16"
    EndOfTransmissionBlock = "\x17"
    Cancel = "\x18"
    EndOfMedium = "\x19"
    Substitute = "\x1A"
    Escape = "\x1B"
    FileSeparator = "\x1C"
    GroupSeparator = "\x1D"
    RecordSeparator = "\x1E"
    UnitSeparator = "\x1F"
    Space = "\x20"  # Whitespace
    Delete = "\x7F"


CONTROl_TOKENS_PATTERN = "\x01-\x08\x0E-\x1F\x7F"

CONTROL_WHITESPACES = {
    ControlTokens.HorizontalTab,
    ControlTokens.LineFeed,
    ControlTokens.VerticalTab,
    ControlTokens.FormFeed,
    ControlTokens.CarriageReturn,
    ControlTokens.Space,
}


def visualize_control_tokens(text: str, include_whitespace=False) -> str:
    # Special visual handling for control characters using Control Pictures Unicode block
    # Based on https://unicode.org/charts/nameslist/n_2400.html
    def control_char_to_symbol(match):
        char = match.group(0)
        code = ord(char)
        if not include_whitespace and char in CONTROL_WHITESPACES:
            return char

        if code <= 0x1F:  # Control characters 0x00-0x1F map to 0x2400-0x241F
            return chr(0x2400 + code)
        elif code == 0x7F:  # DELETE character maps to 0x2421
            return chr(0x2421)
        return char

    return re.sub(r'[\x00-\x1F\x7F]', control_char_to_symbol, text)
