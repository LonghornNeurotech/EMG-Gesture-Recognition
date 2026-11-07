#!/usr/bin/env python3
"""Collect labeled MPU-6050 gesture trials as CSV files.

Acceleration: m/s^2, including gravity
Gyroscope: degrees/second
Timestamp: monotonic nanoseconds from the Raspberry Pi
"""

import argparse
import csv
import os
import struct
import time
from pathlib import Path

from smbus import SMBus


GESTURES = (
    "pronation",
    "supination",
    "forearm_raise",
    "forearm_lower",
    "forearm_inward",
    "forearm_outward",
    "idle",
)

FIELDS = (
    "participant_id",
    "session_id",
    "gesture",
    "trial_id",
    "timestamp_ns",
    "sequence_number",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
)


def initialize_imu(bus, sample_rate):
    identity = bus.read_byte_data(0x68, 0x75)

    if identity != 0x68:
        raise RuntimeError(
            f"IMU identity is 0x{identity:02x}, expected 0x68"
        )

    # Wake MPU-6050 and use the X-axis gyroscope clock.
    bus.write_byte_data(0x68, 0x6B, 1)
    bus.write_byte_data(0x68, 0x6C, 0)

    # Gyroscope: +/-250 degrees/second.
    bus.write_byte_data(0x68, 0x1B, 0)

    # Accelerometer: +/-2g.
    bus.write_byte_data(0x68, 0x1C, 0)

    # Digital low-pass filter configuration.
    bus.write_byte_data(0x68, 0x1A, 3)

    # With the filter enabled, the internal rate is 1000 Hz.
    divider = round(1000 / sample_rate) - 1
    bus.write_byte_data(0x68, 0x19, divider)

    time.sleep(0.2)

    return 1000 / (divider + 1)


def read_imu(bus):
    block = bus.read_i2c_block_data(0x68, 0x3B, 14)
    raw = struct.unpack(">7h", bytes(block))

    # raw[3] is temperature and is not used.
    acceleration = [value * 9.80665 / 16384 for value in raw[:3]]
    gyro = [value / 131 for value in raw[4:]]

    return acceleration, gyro


def collect_trial(
    bus,
    destination,
    participant,
    session,
    gesture,
    trial_number,
    duration,
    sample_rate,
):
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. "
            "Delete it only if you intentionally want to recollect that trial."
        )

    temporary = Path(str(destination) + ".part")
    temporary.unlink(missing_ok=True)

    number_of_samples = round(duration * sample_rate)
    period_ns = round(1_000_000_000 / sample_rate)
    timestamps = []

    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        start_ns = time.monotonic_ns()

        for sequence_number in range(number_of_samples):
            target_ns = start_ns + sequence_number * period_ns
            remaining_ns = target_ns - time.monotonic_ns()

            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)

            timestamp_ns = time.monotonic_ns()
            acceleration, gyro = read_imu(bus)
            timestamps.append(timestamp_ns)

            writer.writerow(
                {
                    "participant_id": participant,
                    "session_id": session,
                    "gesture": gesture,
                    "trial_id": f"T{trial_number:03d}",
                    "timestamp_ns": timestamp_ns,
                    "sequence_number": sequence_number,
                    "ax": f"{acceleration[0]:.6f}",
                    "ay": f"{acceleration[1]:.6f}",
                    "az": f"{acceleration[2]:.6f}",
                    "gx": f"{gyro[0]:.6f}",
                    "gy": f"{gyro[1]:.6f}",
                    "gz": f"{gyro[2]:.6f}",
                }
            )

        file.flush()
        os.fsync(file.fileno())

    temporary.replace(destination)

    elapsed = (timestamps[-1] - timestamps[0]) / 1_000_000_000
    measured_rate = (len(timestamps) - 1) / elapsed

    print(
        f"Saved {destination}\n"
        f"Samples: {number_of_samples}; measured rate: {measured_rate:.2f} Hz"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--gesture", required=True, choices=GESTURES)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--start-trial", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, choices=(50, 100), default=100)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/emg-imu-data").expanduser(),
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")

    if args.duration <= 0:
        parser.error("--duration must be positive")

    output_directory = (
        args.output_root.expanduser()
        / args.participant
        / args.session
        / args.gesture
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    bus = SMBus(1)

    try:
        actual_rate = initialize_imu(bus, args.sample_rate)

        print(f"MPU-6050 initialized at {actual_rate:.2f} Hz.")
        print("Acceleration: m/s^2 including gravity.")
        print("Gyroscope: degrees/second.")

        for offset in range(args.trials):
            trial_number = args.start_trial + offset
            destination = output_directory / f"trial_{trial_number:03d}.csv"

            input(
                f"\nPrepare for {args.gesture}, trial {trial_number}. "
                "Press Enter when ready..."
            )

            for count in (3, 2, 1):
                print(count, flush=True)
                time.sleep(1)

            print("GO", flush=True)

            collect_trial(
                bus=bus,
                destination=destination,
                participant=args.participant,
                session=args.session,
                gesture=args.gesture,
                trial_number=trial_number,
                duration=args.duration,
                sample_rate=actual_rate,
            )

            print("Return to the neutral starting position.")

    finally:
        bus.close()


if __name__ == "__main__":
    main()