#!/usr/bin/env python3
"""Plot one participant's IMU session and generate a trial-level PCA plot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


GESTURE_ORDER = (
    "idle",
    "pronation",
    "supination",
    "forearm_raise",
    "forearm_lower",
    "forearm_inward",
    "forearm_outward",
)

ACCELERATION = ("ax", "ay", "az")
GYROSCOPE = ("gx", "gy", "gz")
CHANNELS = ACCELERATION + GYROSCOPE

REQUIRED_COLUMNS = {
    "participant_id",
    "session_id",
    "gesture",
    "trial_id",
    "timestamp_ns",
    "sequence_number",
    *CHANNELS,
}


def load_session(session_directory):
    available = {
        path.name: path
        for path in session_directory.iterdir()
        if path.is_dir()
    }

    gestures = [g for g in GESTURE_ORDER if g in available]
    gestures += sorted(set(available) - set(gestures))

    if not gestures:
        raise RuntimeError(f"No gesture folders found in {session_directory}")

    records = []

    for gesture in gestures:
        csv_files = sorted(available[gesture].glob("*.csv"))

        if not csv_files:
            raise RuntimeError(f"No CSV files found for {gesture}")

        for csv_path in csv_files:
            frame = pd.read_csv(csv_path)
            missing = REQUIRED_COLUMNS - set(frame.columns)

            if missing:
                raise ValueError(
                    f"{csv_path} is missing columns: {sorted(missing)}"
                )

            sensor_data = (
                frame.loc[:, CHANNELS]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=float)
            )

            timestamps = pd.to_numeric(
                frame["timestamp_ns"], errors="raise"
            ).to_numpy(dtype=np.int64)

            if len(sensor_data) < 2:
                raise ValueError(f"{csv_path} has too few samples")

            sample_intervals = np.diff(timestamps) / 1_000_000_000
            measured_rate = 1.0 / np.median(sample_intervals)

            records.append(
                {
                    "gesture": gesture,
                    "path": csv_path,
                    "data": sensor_data,
                    "time": (timestamps - timestamps[0]) / 1_000_000_000,
                    "sample_rate": measured_rate,
                }
            )

    sample_counts = {len(record["data"]) for record in records}

    if len(sample_counts) != 1:
        raise ValueError(
            f"Trials have unequal sample counts: {sorted(sample_counts)}. "
            "Validate incomplete trials before plotting."
        )

    return gestures, records


def plot_average_timeseries(gestures, records, output_path):
    figure, axes = plt.subplots(
        len(gestures),
        2,
        figsize=(14, 2.5 * len(gestures)),
        sharex=True,
    )

    acceleration_colors = ("tab:red", "tab:green", "tab:blue")
    gyro_colors = ("tab:orange", "tab:purple", "tab:cyan")

    for row, gesture in enumerate(gestures):
        gesture_records = [
            record for record in records
            if record["gesture"] == gesture
        ]

        stacked_data = np.stack(
            [record["data"] for record in gesture_records]
        )
        stacked_time = np.stack(
            [record["time"] for record in gesture_records]
        )

        time_axis = stacked_time.mean(axis=0)
        mean = stacked_data.mean(axis=0)
        standard_deviation = stacked_data.std(axis=0)

        acceleration_axis = axes[row, 0]
        gyro_axis = axes[row, 1]

        for index, (channel, color) in enumerate(
            zip(ACCELERATION, acceleration_colors)
        ):
            acceleration_axis.plot(
                time_axis,
                mean[:, index],
                label=channel,
                color=color,
            )
            acceleration_axis.fill_between(
                time_axis,
                mean[:, index] - standard_deviation[:, index],
                mean[:, index] + standard_deviation[:, index],
                color=color,
                alpha=0.12,
            )

        for index, (channel, color) in enumerate(
            zip(GYROSCOPE, gyro_colors),
            start=3,
        ):
            gyro_axis.plot(
                time_axis,
                mean[:, index],
                label=channel,
                color=color,
            )
            gyro_axis.fill_between(
                time_axis,
                mean[:, index] - standard_deviation[:, index],
                mean[:, index] + standard_deviation[:, index],
                color=color,
                alpha=0.12,
            )

        acceleration_axis.set_ylabel(gesture)
        acceleration_axis.grid(alpha=0.25)
        gyro_axis.grid(alpha=0.25)

    axes[0, 0].set_title("Acceleration: mean ± one standard deviation")
    axes[0, 1].set_title("Gyroscope: mean ± one standard deviation")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    figure.text(
        0.01,
        0.5,
        "Acceleration (m/s²) / Gyroscope (degrees/s)",
        rotation=90,
        va="center",
    )

    figure.tight_layout(rect=(0.03, 0, 1, 1))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pca(gestures, records, output_path):
    features = np.stack(
        [record["data"].reshape(-1) for record in records]
    )
    labels = [record["gesture"] for record in records]

    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    projection = pca.fit_transform(standardized)

    figure, axis = plt.subplots(figsize=(9, 7))
    color_map = plt.get_cmap("tab10")

    for index, gesture in enumerate(gestures):
        locations = [
            position
            for position, label in enumerate(labels)
            if label == gesture
        ]

        axis.scatter(
            projection[locations, 0],
            projection[locations, 1],
            label=gesture,
            color=color_map(index),
            s=55,
            alpha=0.8,
        )

    axis.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )
    axis.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )
    axis.set_title("Trial-level PCA of IMU recordings")
    axis.grid(alpha=0.25)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="P001")
    parser.add_argument("--session", default="S01")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    session_directory = (
        repository
        / "data"
        / "raw"
        / args.participant
        / args.session
    )

    if not session_directory.exists():
        raise FileNotFoundError(
            f"Session directory does not exist: {session_directory}"
        )

    output_directory = (
        repository
        / "docs"
        / "figures"
        / args.participant
        / args.session
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    gestures, records = load_session(session_directory)

    rates = [record["sample_rate"] for record in records]

    print(f"Loaded {len(records)} trials.")
    print(f"Gestures: {', '.join(gestures)}")
    print(
        f"Measured sample rate: "
        f"median={np.median(rates):.2f} Hz, "
        f"minimum={np.min(rates):.2f} Hz, "
        f"maximum={np.max(rates):.2f} Hz"
    )

    timeseries_path = output_directory / "average_timeseries.png"
    pca_path = output_directory / "trial_pca.png"

    plot_average_timeseries(gestures, records, timeseries_path)
    plot_pca(gestures, records, pca_path)

    print(f"Saved: {timeseries_path}")
    print(f"Saved: {pca_path}")


if __name__ == "__main__":
    main()