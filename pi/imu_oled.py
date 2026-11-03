#!/usr/bin/env python3
"""Live MPU-6050 -> SSD1306 demo. No files or recordings are created.

Requires only the already-installed python3-smbus. Bus 1, IMU 0x68,
OLED 0x3c. Default display 128x64; use --height 32 for 128x32.
Acceleration in m/s^2 INCLUDING gravity; rotation in degrees/second.
No bias correction. Same +/-2g and +/-250 degrees/s as the live terminal test.
This is a ~5 Hz visual demo, not a dataset acquisition program.
"""

import argparse
import struct
import time

# Five columns per 5x7 glyph, least-significant bit is the upper pixel.
# Embedding the small font avoids image/font packages and downloads.
FONT = {
    ' ': [0, 0, 0, 0, 0],
    '0': [0x3e, 0x51, 0x49, 0x45, 0x3e],
    '1': [0, 0x42, 0x7f, 0x40, 0],
    '2': [0x42, 0x61, 0x51, 0x49, 0x46],
    '3': [0x21, 0x41, 0x45, 0x4b, 0x31],
    '4': [0x18, 0x14, 0x12, 0x7f, 0x10],
    '5': [0x27, 0x45, 0x45, 0x45, 0x39],
    '6': [0x3c, 0x4a, 0x49, 0x49, 0x30],
    '7': [1, 0x71, 9, 5, 3],
    '8': [0x36, 0x49, 0x49, 0x49, 0x36],
    '9': [6, 0x49, 0x49, 0x29, 0x1e],
    '+': [8, 8, 0x3e, 8, 8],
    '-': [8, 8, 8, 8, 8],
    '.': [0, 0x60, 0x60, 0, 0],
    '/': [0x20, 0x10, 8, 4, 2],
    'a': [0x20, 0x54, 0x54, 0x54, 0x78],
    'd': [0x38, 0x44, 0x44, 0x48, 0x7f],
    'e': [0x38, 0x54, 0x54, 0x54, 0x18],
    'g': [0x0c, 0x52, 0x52, 0x52, 0x3e],
    'm': [0x7c, 4, 0x18, 4, 0x78],
    's': [0x48, 0x54, 0x54, 0x54, 0x20],
    'x': [0x44, 0x28, 0x10, 0x28, 0x44],
    'y': [0x0c, 0x50, 0x50, 0x50, 0x3c],
    'z': [0x44, 0x64, 0x54, 0x4c, 0x44],
}


def draw_text(frame, height, x, y, text):
    for char in text:
        for dx, column in enumerate(FONT[char]):
            for dy in range(7):
                px, py = x + dx, y + dy
                if 0 <= px < 128 and 0 <= py < height and column & (1 << dy):
                    frame[(py // 8) * 128 + px] |= 1 << (py % 8)
        x += 6


def make_frame(height, acceleration, gyro):
    frame = bytearray(128 * height // 8)
    if height == 64:
        draw_text(frame, height, 0, 0, 'a m/s2')
        draw_text(frame, height, 64, 0, 'g deg/s')
        positions = (16, 32, 48)
    else:
        positions = (0, 10, 20)
    for axis, y, a, g in zip('xyz', positions, acceleration, gyro):
        draw_text(frame, height, 0, y, f'a{axis}{a:+7.2f}')
        draw_text(frame, height, 64, y, f'g{axis}{g:+7.2f}')
    return frame


def command(bus, *values):
    for value in values:
        bus.write_byte_data(0x3c, 0x00, value)


def show(bus, frame, height):
    command(bus, 0x21, 0, 127, 0x22, 0, height // 8 - 1)
    for start in range(0, len(frame), 32):
        bus.write_i2c_block_data(0x3c, 0x40, list(frame[start:start + 32]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height', type=int, choices=(32, 64), default=64)
    parser.add_argument('--rotate', action='store_true', help='Rotate display 180 degrees')
    parser.add_argument('--seconds', type=float, default=0, help='Stop after N seconds; 0 runs until Ctrl+C')
    args = parser.parse_args()
    if not 0 <= args.seconds < float('inf'):
        parser.error('--seconds must be finite and nonnegative')

    from smbus import SMBus
    bus = SMBus(1)
    display_ready = False
    try:
        identity = bus.read_byte_data(0x68, 0x75)
        if identity != 0x68:
            raise RuntimeError(f'IMU identity is 0x{identity:02x}, expected 0x68')
        bus.write_byte_data(0x68, 0x6b, 1)
        time.sleep(0.1)
        for register, value in ((0x6c, 0), (0x1b, 0), (0x1c, 0),
                                (0x1a, 3), (0x19, 9)):
            bus.write_byte_data(0x68, register, value)
        time.sleep(0.2)

        command(bus, 0xae, 0xd5, 0x80, 0xa8, args.height - 1,
                0xd3, 0, 0x40, 0x8d, 0x14, 0x20, 0,
                0xa0 if args.rotate else 0xa1,
                0xc0 if args.rotate else 0xc8,
                0xda, 0x12 if args.height == 64 else 0x02,
                0x81, 0x7f, 0xd9, 0xf1, 0xdb, 0x40, 0xa4, 0xa6)
        show(bus, bytearray(128 * args.height // 8), args.height)
        command(bus, 0xaf)
        display_ready = True
        print('Live OLED demo: accel m/s^2, gyro deg/s. No files are saved.')
        print('Ctrl+C to stop and turn the OLED off.', flush=True)
        start = time.monotonic()
        next_print = start
        while args.seconds == 0 or time.monotonic() - start < args.seconds:
            tick = time.monotonic()
            raw = struct.unpack('>7h', bytes(bus.read_i2c_block_data(0x68, 0x3b, 14)))
            acceleration = [v * 9.80665 / 16384 for v in raw[:3]]
            gyro = [v / 131 for v in raw[4:]]
            show(bus, make_frame(args.height, acceleration, gyro), args.height)
            if tick >= next_print:
                print('a: ' + ' '.join(f'{v:+7.2f}' for v in acceleration)
                      + ' | g: ' + ' '.join(f'{v:+7.2f}' for v in gyro), flush=True)
                next_print = tick + 1
            time.sleep(max(0, 0.2 - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        try:
            if display_ready:
                command(bus, 0xae)
        finally:
            bus.close()


if __name__ == '__main__':
    main()
