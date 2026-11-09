#!/usr/bin/env python3
"""Collect labeled MPU-6050 gesture trials as CSV files.

Acceleration: m/s^2, including gravity
Gyroscope: degrees/second
Timestamp: monotonic nanoseconds from the Raspberry Pi

Existing trials are never overwritten unless --recollect is supplied.
When --recollect is used, the previous trial is archived before the completed
replacement is installed.
"""

import argparse
import csv
import json
import math
import os
import re
import struct
import time
from datetime import datetime, timezone
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

# Initial warning thresholds. These produce warnings only; they do not reject
# or delete trials. Adjust them after reviewing trial_quality.jsonl.
IDLE_ACCEL_RMS_MAX = 0.50       # m/s^2 relative to initial posture
IDLE_GYRO_RMS_MAX = 5.0         # degrees/second
ACTIVE_ACCEL_RMS_MIN = 0.25     # m/s^2 relative to initial posture
ACTIVE_GYRO_RMS_MIN = 5.0       # degrees/second


def safe_identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "Use only letters, numbers, underscores, and hyphens."
        )

    return value


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
    acceleration = [
        value * 9.80665 / 16384
        for value in raw[:3]
    ]
    gyroscope = [
        value / 131
        for value in raw[4:]
    ]

    return acceleration, gyroscope


def calculate_motion_metrics(samples):
    """Return acceleration and gyroscope RMS motion measurements."""

    if not samples:
        raise ValueError("Cannot calculate motion metrics without samples")

    # The first 100 ms at 100 Hz is treated as the neutral baseline.
    baseline_count = min(10, len(samples))

    acceleration_baseline = [
        sum(sample[axis] for sample in samples[:baseline_count])
        / baseline_count
        for axis in range(3)
    ]

    acceleration_squared = []
    gyroscope_squared = []

    for sample in samples:
        ax, ay, az, gx, gy, gz = sample

        acceleration_squared.append(
            (ax - acceleration_baseline[0]) ** 2
            + (ay - acceleration_baseline[1]) ** 2
            + (az - acceleration_baseline[2]) ** 2
        )

        gyroscope_squared.append(
            gx ** 2 + gy ** 2 + gz ** 2
        )

    acceleration_rms = math.sqrt(
        sum(acceleration_squared) / len(acceleration_squared)
    )
    gyroscope_rms = math.sqrt(
        sum(gyroscope_squared) / len(gyroscope_squared)
    )

    return acceleration_rms, gyroscope_rms


def evaluate_trial_quality(gesture, acceleration_rms, gyroscope_rms):
    """Return a warning string or None."""

    reasons = []

    if gesture == "idle":
        if acceleration_rms > IDLE_ACCEL_RMS_MAX:
            reasons.append(
                f"acceleration RMS {acceleration_rms:.3f} m/s^2 "
                f"> idle limit {IDLE_ACCEL_RMS_MAX:.3f}"
            )

        if gyroscope_rms > IDLE_GYRO_RMS_MAX:
            reasons.append(
                f"gyroscope RMS {gyroscope_rms:.3f} deg/s "
                f"> idle limit {IDLE_GYRO_RMS_MAX:.3f}"
            )

        if reasons:
            return "Too much motion during idle: " + "; ".join(reasons)

    else:
        acceleration_is_low = (
            acceleration_rms < ACTIVE_ACCEL_RMS_MIN
        )
        gyroscope_is_low = (
            gyroscope_rms < ACTIVE_GYRO_RMS_MIN
        )

        # Both must be low to avoid rejecting a gesture that is clearly
        # observable through only one sensor type.
        if acceleration_is_low and gyroscope_is_low:
            return (
                "Too little motion during active gesture: "
                f"acceleration RMS {acceleration_rms:.3f} m/s^2 "
                f"< {ACTIVE_ACCEL_RMS_MIN:.3f}; "
                f"gyroscope RMS {gyroscope_rms:.3f} deg/s "
                f"< {ACTIVE_GYRO_RMS_MIN:.3f}"
            )

    return None


def archive_existing_trial(destination):
    """Move an existing trial outside the synchronized dataset directory."""

    output_root = destination.parents[3]
    archive_root = output_root.with_name(
        output_root.name + "-replaced"
    )

    relative = destination.relative_to(output_root)
    archive_directory = archive_root / relative.parent
    archive_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    archive_path = archive_directory / (
        f"{destination.name}.{timestamp}.replaced"
    )

    destination.replace(archive_path)
    return archive_path


def append_quality_log(
    destination,
    participant,
    session,
    gesture,
    trial_number,
    acceleration_rms,
    gyroscope_rms,
    warning,
    archived_path,
):
    """Append trial metrics without affecting CSV file-count checks."""

    session_directory = destination.parents[1]
    log_path = session_directory / "trial_quality.jsonl"

    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "participant_id": participant,
        "session_id": session,
        "gesture": gesture,
        "trial_id": f"T{trial_number:03d}",
        "csv_path": str(destination),
        "acceleration_rms_m_s2": acceleration_rms,
        "gyroscope_rms_deg_s": gyroscope_rms,
        "warning": warning,
        "archived_previous_trial": (
            str(archived_path)
            if archived_path is not None
            else None
        ),
        "thresholds": {
            "idle_acceleration_rms_max": IDLE_ACCEL_RMS_MAX,
            "idle_gyroscope_rms_max": IDLE_GYRO_RMS_MAX,
            "active_acceleration_rms_min": ACTIVE_ACCEL_RMS_MIN,
            "active_gyroscope_rms_min": ACTIVE_GYRO_RMS_MIN,
        },
    }

    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")


def collect_trial(
    bus,
    destination,
    participant,
    session,
    gesture,
    trial_number,
    duration,
    sample_rate,
    recollect=False,
):
    existing_trial = destination.exists()

    if existing_trial and not recollect:
        raise FileExistsError(
            f"{destination} already exists. "
            "Use --recollect to safely replace and archive it."
        )

    temporary = Path(str(destination) + ".part")
    temporary.unlink(missing_ok=True)

    number_of_samples = round(duration * sample_rate)
    period_ns = round(1_000_000_000 / sample_rate)

    timestamps = []
    motion_samples = []

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
            acceleration, gyroscope = read_imu(bus)

            timestamps.append(timestamp_ns)
            motion_samples.append(
                (
                    acceleration[0],
                    acceleration[1],
                    acceleration[2],
                    gyroscope[0],
                    gyroscope[1],
                    gyroscope[2],
                )
            )

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
                    "gx": f"{gyroscope[0]:.6f}",
                    "gy": f"{gyroscope[1]:.6f}",
                    "gz": f"{gyroscope[2]:.6f}",
                }
            )

        file.flush()
        os.fsync(file.fileno())

    # Keep the original in place unless the replacement completed successfully.
    archived_path = None

    if existing_trial:
        archived_path = archive_existing_trial(destination)
        print(f"Archived previous trial to: {archived_path}")

    temporary.replace(destination)

    acceleration_rms, gyroscope_rms = calculate_motion_metrics(
        motion_samples
    )
    warning = evaluate_trial_quality(
        gesture,
        acceleration_rms,
        gyroscope_rms,
    )

    append_quality_log(
        destination=destination,
        participant=participant,
        session=session,
        gesture=gesture,
        trial_number=trial_number,
        acceleration_rms=acceleration_rms,
        gyroscope_rms=gyroscope_rms,
        warning=warning,
        archived_path=archived_path,
    )

    elapsed = (
        timestamps[-1] - timestamps[0]
    ) / 1_000_000_000
    measured_rate = (
        (len(timestamps) - 1) / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"Saved {destination}\n"
        f"Samples: {number_of_samples}; "
        f"measured rate: {measured_rate:.2f} Hz\n"
        f"Motion QC: acceleration RMS={acceleration_rms:.3f} m/s^2; "
        f"gyroscope RMS={gyroscope_rms:.3f} deg/s"
    )

    if warning:
        print("\n" + "!" * 70)
        print(f"QUALITY WARNING: {warning}")
        print("Review this trial and consider recollection.")
        print("!" * 70)
    else:
        print("Motion QC passed.")

    return {
        "acceleration_rms": acceleration_rms,
        "gyroscope_rms": gyroscope_rms,
        "warning": warning,
        "archived_path": archived_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Record MPU-6050 gesture trials to CSV."
    )
    parser.add_argument(
        "--participant",
        required=True,
        type=safe_identifier,
    )
    parser.add_argument(
        "--session",
        required=True,
        type=safe_identifier,
    )
    parser.add_argument(
        "--gesture",
        required=True,
        choices=GESTURES,
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--start-trial", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=(50, 100),
        default=100,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/emg-imu-data").expanduser(),
    )
    parser.add_argument(
        "--recollect",
        action="store_true",
        help=(
            "Safely archive and replace existing trial files. "
            "Without this flag, existing trials cause an error."
        ),
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")

    if args.start_trial < 1:
        parser.error("--start-trial must be at least 1")

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
            destination = (
                output_directory / f"trial_{trial_number:03d}.csv"
            )

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
                recollect=args.recollect,
            )

            print("STOP - return to the neutral starting position.")

    finally:
        bus.close()


if __name__ == "__main__":
    main()