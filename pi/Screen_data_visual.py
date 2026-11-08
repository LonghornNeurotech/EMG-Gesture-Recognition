"""Show gesture predictions on a 128x32 or 128x64 SSD1306 OLED.

Example:
    from smbus import SMBus
    from Screen_data_visual import GestureDisplay

    bus = SMBus(1)
    display = GestureDisplay(bus)
    display.show_prediction("idle", None)  # Confidence is not available yet.
    # Later, in the model loop:
    # display.show_prediction(predicted_gesture, model_confidence)
    display.turn_off()
    bus.close()

``confidence`` is expected as a fraction from 0.0 to 1.0.
"""

import math


GESTURES = (
    "Forearm raise",
    "Forearm Lower",
    "Forearm right",
    "Supination",
    "idle",
    "pronation",
)

FONT = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
}


def _draw_text(frame, height, x, y, text):
    for char in text.upper():
        glyph = FONT[char]
        for dy, row in enumerate(glyph):
            for dx, pixel in enumerate(row):
                px, py = x + dx, y + dy
                if pixel == "1" and 0 <= px < 128 and 0 <= py < height:
                    frame[(py // 8) * 128 + px] |= 1 << (py % 8)
        x += 6


class GestureDisplay:
    """Control an SSD1306 display using an already-open smbus-compatible bus."""

    def __init__(self, bus, height=64, address=0x3c, rotate=False):
        if height not in (32, 64):
            raise ValueError("height must be 32 or 64")
        if not 0x03 <= address <= 0x77:
            raise ValueError("address must be a valid 7-bit I2C address")

        self.bus = bus
        self.height = height
        self.address = address
        self._command(
            0xae, 0xd5, 0x80, 0xa8, height - 1, 0xd3, 0, 0x40, 0x8d, 0x14,
            0x20, 0, 0xa0 if rotate else 0xa1, 0xc0 if rotate else 0xc8,
            0xda, 0x12 if height == 64 else 0x02, 0x81, 0x7f, 0xd9, 0xf1,
            0xdb, 0x40, 0xa4, 0xa6,
        )
        self.clear()
        self._command(0xaf)

    def _command(self, *values):
        for value in values:
            self.bus.write_byte_data(self.address, 0x00, value)

    def _show(self, frame):
        self._command(0x21, 0, 127, 0x22, 0, self.height // 8 - 1)
        for start in range(0, len(frame), 32):
            self.bus.write_i2c_block_data(
                self.address, 0x40, list(frame[start:start + 32])
            )

    def clear(self):
        """Clear the display without turning it off."""
        self._show(bytearray(128 * self.height // 8))

    def show_prediction(self, gesture, confidence=None):
        """Display a supported gesture and its optional 0.0-to-1.0 confidence."""
        if not isinstance(gesture, str):
            raise TypeError("gesture must be a string")
        normalized_gesture = " ".join(gesture.split()).casefold()
        gestures_by_name = {name.casefold(): name for name in GESTURES}
        if normalized_gesture not in gestures_by_name:
            raise ValueError(f"gesture must be one of: {', '.join(GESTURES)}")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise TypeError("confidence must be a number from 0.0 to 1.0 or None")
            if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
                raise ValueError("confidence must be finite and between 0.0 and 1.0")

        frame = bytearray(128 * self.height // 8)
        gesture_y, confidence_label_y, confidence_value_y = (
            (14, 34, 48) if self.height == 64 else (8, 16, 24)
        )
        _draw_text(frame, self.height, 0, 0, "GESTURE")
        _draw_text(
            frame, self.height, 0, gesture_y, gestures_by_name[normalized_gesture]
        )
        _draw_text(frame, self.height, 0, confidence_label_y, "CONFIDENCE")
        confidence_text = "NO MODEL" if confidence is None else f"{confidence:.0%}"
        _draw_text(frame, self.height, 0, confidence_value_y, confidence_text)
        self._show(frame)

    def turn_off(self):
        """Turn off the OLED; the caller retains ownership of the I2C bus."""
        self._command(0xae)