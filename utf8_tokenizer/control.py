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
    LineFeed = "\x0a"  # Whitespace
    VerticalTab = "\x0b"  # Whitespace
    FormFeed = "\x0c"  # Whitespace
    CarriageReturn = "\x0d"  # Whitespace
    ShiftOut = "\x0e"
    ShiftIn = "\x0f"
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
    Substitute = "\x1a"
    Escape = "\x1b"
    FileSeparator = "\x1c"
    GroupSeparator = "\x1d"
    RecordSeparator = "\x1e"
    UnitSeparator = "\x1f"
    Space = "\x20"  # Whitespace
    Delete = "\x7f"


CONTROL_WHITESPACES = {
    ControlTokens.HorizontalTab,
    ControlTokens.LineFeed,
    ControlTokens.VerticalTab,
    ControlTokens.FormFeed,
    ControlTokens.CarriageReturn,
    ControlTokens.Space,
}


# Control Pictures Unicode block: 0x00-0x1F map to 0x2400-0x241F, DELETE maps to 0x2421
# Based on https://unicode.org/charts/nameslist/n_2400.html
# Space is excluded on purpose: it is in CONTROL_WHITESPACES but was never visualized
CONTROL_PICTURES = {code: 0x2400 + code for code in range(0x20)} | {0x7F: 0x2421}

CONTROL_PICTURES_NO_WHITESPACE = {
    code: picture for code, picture in CONTROL_PICTURES.items() if chr(code) not in CONTROL_WHITESPACES
}


def visualize_control_tokens(text: str, include_whitespace=False) -> str:
    return text.translate(CONTROL_PICTURES if include_whitespace else CONTROL_PICTURES_NO_WHITESPACE)
