#!/usr/bin/env python3
"""Collect a complete seven-class IMU session for one participant."""

import argparse
import time
from pathlib import Path

from smbus import SMBus

from collect_imu_trials import collect_trial, initialize_imu


GESTURE_ORDER = (
    "idle",
    "pronation",
    "supination",
    "forearm_raise",
    "forearm_lower",
    "forearm_inward",
    "forearm_outward",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials for each class.",
    )
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
        "--resume",
        action="store_true",
        help="Skip files that already exist instead of stopping.",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")

    if args.duration <= 0:
        parser.error("--duration must be positive")

    print("\nComplete IMU Collection Session")
    print(f"Participant: {args.participant}")
    print(f"Session: {args.session}")
    print(f"Trials per class: {args.trials}")
    print(f"Trial duration: {args.duration} seconds")
    print(f"Target sample rate: {args.sample_rate} Hz")
    print("\nGesture order:")

    for number, gesture in enumerate(GESTURE_ORDER, start=1):
        print(f"  {number}. {gesture}")

    input(
        "\nConfirm the IMU placement and orientation are documented. "
        "Press Enter to initialize the sensor..."
    )

    bus = SMBus(1)

    try:
        actual_rate = initialize_imu(bus, args.sample_rate)

        print(f"\nMPU-6050 initialized at {actual_rate:.2f} Hz.")
        print("Acceleration: m/s^2 including gravity.")
        print("Gyroscope: degrees/second.")

        for gesture_number, gesture in enumerate(GESTURE_ORDER, start=1):
            print("\n" + "=" * 50)
            print(
                f"GESTURE {gesture_number}/{len(GESTURE_ORDER)}: "
                f"{gesture}"
            )
            print("=" * 50)

            input(
                "Explain or demonstrate this gesture to the participant. "
                "Press Enter when ready..."
            )

            output_directory = (
                args.output_root.expanduser()
                / args.participant
                / args.session
                / gesture
            )
            output_directory.mkdir(parents=True, exist_ok=True)

            for trial_number in range(1, args.trials + 1):
                destination = (
                    output_directory / f"trial_{trial_number:03d}.csv"
                )

                if destination.exists():
                    if args.resume:
                        print(f"Skipping existing file: {destination}")
                        continue

                    raise FileExistsError(
                        f"{destination} already exists. "
                        "Use --resume to skip completed trials."
                    )

                input(
                    f"\nPrepare for {gesture}, trial "
                    f"{trial_number}/{args.trials}. Press Enter..."
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
                    gesture=gesture,
                    trial_number=trial_number,
                    duration=args.duration,
                    sample_rate=actual_rate,
                )

                print("STOP — return to the neutral starting position.")

            print(f"\nCompleted all {gesture} trials.")

        print("\n" + "=" * 50)
        print("SESSION COMPLETE")
        print("=" * 50)
        print(
            f"Data saved under:\n"
            f"{args.output_root.expanduser() / args.participant / args.session}"
        )

    except KeyboardInterrupt:
        print("\nCollection stopped. Completed CSV files were preserved.")
        print("Run the same command with --resume to continue.")

    finally:
        bus.close()


if __name__ == "__main__":
    main()