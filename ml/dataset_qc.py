#!/usr/bin/env python3
"""Cross-participant QC for the finalized IMU gesture dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)


DEFAULT_GESTURES = (
    "idle",
    "pronation",
    "supination",
    "forearm_raise",
    "forearm_lower",
    "forearm_inward",
    "forearm_outward",
)

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

REQUIRED_COLUMNS = {
    "participant_id",
    "session_id",
    "gesture",
    "trial_id",
    "timestamp_ns",
    "sequence_number",
    *CHANNELS,
}


def load_dataset(data_root, participants, session, gestures):
    records = []
    issues = []

    for participant in participants:
        session_directory = data_root / participant / session

        if not session_directory.exists():
            raise FileNotFoundError(
                f"Missing session directory: {session_directory}"
            )

        for gesture in gestures:
            gesture_directory = session_directory / gesture
            files = sorted(gesture_directory.glob("*.csv"))

            if len(files) != 10:
                issues.append(
                    f"{participant}/{gesture}: expected 10 trials, "
                    f"found {len(files)}"
                )

            for path in files:
                frame = pd.read_csv(path)
                missing = REQUIRED_COLUMNS - set(frame.columns)

                if missing:
                    raise ValueError(
                        f"{path} is missing columns: {sorted(missing)}"
                    )

                sensor_data = (
                    frame.loc[:, CHANNELS]
                    .apply(pd.to_numeric, errors="raise")
                    .to_numpy(dtype=float)
                )

                timestamps = pd.to_numeric(
                    frame["timestamp_ns"], errors="raise"
                ).to_numpy(dtype=np.int64)

                sequence = pd.to_numeric(
                    frame["sequence_number"], errors="raise"
                ).to_numpy(dtype=np.int64)

                if not np.isfinite(sensor_data).all():
                    raise ValueError(f"{path} contains NaN or infinity")

                intervals = np.diff(timestamps) / 1_000_000_000

                if np.any(intervals <= 0):
                    raise ValueError(
                        f"{path} has non-increasing timestamps"
                    )

                median_interval = np.median(intervals)
                measured_rate = 1.0 / median_interval
                jitter_ms = (
                    np.percentile(
                        np.abs(intervals - median_interval),
                        95,
                    )
                    * 1000
                )

                sequence_gaps = int(np.sum(np.diff(sequence) != 1))

                acceleration = sensor_data[:, :3]
                gyroscope = sensor_data[:, 3:]

                acceleration_saturation = (
                    100
                    * np.mean(
                        np.abs(acceleration) >= 0.98 * 2 * 9.80665
                    )
                )
                gyro_saturation = (
                    100
                    * np.mean(np.abs(gyroscope) >= 0.98 * 250)
                )

                recorded_gestures = set(frame["gesture"].astype(str))

                if recorded_gestures != {gesture}:
                    issues.append(
                        f"{path}: folder label {gesture}, "
                        f"CSV labels {recorded_gestures}"
                    )

                records.append(
                    {
                        "participant": participant,
                        "gesture": gesture,
                        "path": str(path),
                        "data": sensor_data,
                        "samples": len(frame),
                        "rate_hz": measured_rate,
                        "jitter_ms": jitter_ms,
                        "sequence_gaps": sequence_gaps,
                        "accel_saturation_percent": (
                            acceleration_saturation
                        ),
                        "gyro_saturation_percent": gyro_saturation,
                    }
                )

    return records, issues


def prepare_features(records):
    sample_counts = sorted(
        {record["samples"] for record in records}
    )

    if sample_counts != [200]:
        raise ValueError(
            f"Expected exactly 200 samples per trial; "
            f"found {sample_counts}"
        )

    trials = np.stack([record["data"] for record in records])

    channel_mean = trials.reshape(-1, 6).mean(axis=0)
    channel_std = trials.reshape(-1, 6).std(axis=0)
    channel_std[channel_std == 0] = 1

    normalized = (trials - channel_mean) / channel_std
    flattened = normalized.reshape(len(normalized), -1)

    participants = np.array(
        [record["participant"] for record in records]
    )
    gestures = np.array(
        [record["gesture"] for record in records]
    )

    return trials, flattened, participants, gestures


def shared_pca_plot(
    features,
    participants,
    gesture_labels,
    gestures,
    output_path,
):
    projection = PCA(n_components=2).fit_transform(features)

    participant_names = sorted(set(participants))
    gesture_colors = {
        gesture: plt.get_cmap("tab10")(index)
        for index, gesture in enumerate(gestures)
    }
    participant_markers = {
        participant: marker
        for participant, marker in zip(
            participant_names,
            ("o", "s", "^", "D", "P", "X", "v"),
        )
    }

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    for gesture in gestures:
        selected = gesture_labels == gesture
        axes[0].scatter(
            projection[selected, 0],
            projection[selected, 1],
            label=gesture,
            color=gesture_colors[gesture],
            alpha=0.75,
        )

    for participant in participant_names:
        selected = participants == participant
        axes[1].scatter(
            projection[selected, 0],
            projection[selected, 1],
            label=participant,
            marker=participant_markers[participant],
            alpha=0.75,
        )

    axes[0].set_title("Shared PCA — colored by gesture")
    axes[1].set_title("Same PCA — colored by participant")

    for axis in axes:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def calculate_separation(features, labels, gestures):
    metric_projection = PCA(
        n_components=0.95,
        svd_solver="full",
    ).fit_transform(features)

    centroids = {}
    spreads = {}

    for gesture in gestures:
        values = metric_projection[labels == gesture]
        centroid = values.mean(axis=0)

        centroids[gesture] = centroid
        spreads[gesture] = np.sqrt(
            np.mean(np.sum((values - centroid) ** 2, axis=1))
        )

    matrix = np.zeros((len(gestures), len(gestures)))

    for row, first in enumerate(gestures):
        for column, second in enumerate(gestures):
            if first == second:
                matrix[row, column] = np.nan
                continue

            centroid_distance = np.linalg.norm(
                centroids[first] - centroids[second]
            )
            combined_spread = np.sqrt(
                (spreads[first] ** 2 + spreads[second] ** 2)
                / 2
            )

            matrix[row, column] = (
                centroid_distance / combined_spread
                if combined_spread > 0
                else np.inf
            )

    return metric_projection, matrix


def plot_matrix(matrix, labels, title, output_path, percent=False):
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="viridis")

    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(title)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]

            if np.isfinite(value):
                text = (
                    f"{value * 100:.0f}%"
                    if percent
                    else f"{value:.2f}"
                )
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value < np.nanmax(matrix) / 2 else "black",
                    fontsize=8,
                )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def normalize_from_training(training, testing):
    mean = training.reshape(-1, 6).mean(axis=0)
    standard_deviation = training.reshape(-1, 6).std(axis=0)
    standard_deviation[standard_deviation == 0] = 1

    training = (training - mean) / standard_deviation
    testing = (testing - mean) / standard_deviation

    return (
        training.reshape(len(training), -1),
        testing.reshape(len(testing), -1),
    )


def leave_one_participant_out(
    trials,
    participants,
    labels,
    gestures,
):
    all_true = []
    all_predicted = []
    fold_results = {}

    for test_participant in sorted(set(participants)):
        train_mask = participants != test_participant
        test_mask = participants == test_participant

        train_features, test_features = normalize_from_training(
            trials[train_mask],
            trials[test_mask],
        )

        components = min(20, len(train_features) - 1)

        pca = PCA(n_components=components)
        train_projection = pca.fit_transform(train_features)
        test_projection = pca.transform(test_features)

        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        centroids = np.stack(
            [
                train_projection[train_labels == gesture].mean(axis=0)
                for gesture in gestures
            ]
        )

        distances = np.linalg.norm(
            test_projection[:, None, :] - centroids[None, :, :],
            axis=2,
        )

        predictions = np.array(gestures)[np.argmin(distances, axis=1)]

        fold_results[test_participant] = accuracy_score(
            test_labels,
            predictions,
        )

        all_true.extend(test_labels)
        all_predicted.extend(predictions)

    matrix = confusion_matrix(
        all_true,
        all_predicted,
        labels=gestures,
        normalize="true",
    )

    return fold_results, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--participants",
        nargs="+",
        default=("P002", "P003", "P004", "P005"),
    )
    parser.add_argument("--session", default="S01")
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=DEFAULT_GESTURES,
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    data_root = repository / "data" / "raw"
    output_directory = repository / "docs" / "qc"
    output_directory.mkdir(parents=True, exist_ok=True)

    records, issues = load_dataset(
        data_root,
        args.participants,
        args.session,
        args.gestures,
    )

    summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key != "data"
            }
            for record in records
        ]
    )
    summary.to_csv(output_directory / "trial_qc.csv", index=False)

    trials, features, participants, labels = prepare_features(records)

    metric_projection, separation = calculate_separation(
        features,
        labels,
        args.gestures,
    )

    gesture_silhouette = silhouette_score(
        metric_projection,
        labels,
    )
    participant_silhouette = silhouette_score(
        metric_projection,
        participants,
    )
    davies_bouldin = davies_bouldin_score(
        metric_projection,
        labels,
    )

    shared_pca_plot(
        features,
        participants,
        labels,
        args.gestures,
        output_directory / "shared_pca.png",
    )

    plot_matrix(
        separation,
        args.gestures,
        "Pairwise gesture separation ratio — higher is better",
        output_directory / "gesture_separation.png",
    )

    fold_results, confusion = leave_one_participant_out(
        trials,
        participants,
        labels,
        args.gestures,
    )

    plot_matrix(
        confusion,
        args.gestures,
        "Leave-one-participant-out nearest-centroid confusion",
        output_directory / "loso_confusion.png",
        percent=True,
    )

    print(f"Loaded trials: {len(records)}")
    print(
        f"Sampling rate: median={summary['rate_hz'].median():.3f} Hz, "
        f"minimum={summary['rate_hz'].min():.3f} Hz, "
        f"maximum={summary['rate_hz'].max():.3f} Hz"
    )
    print(
        f"95th-percentile timing jitter: "
        f"median={summary['jitter_ms'].median():.4f} ms, "
        f"maximum={summary['jitter_ms'].max():.4f} ms"
    )
    print(f"Total sequence gaps: {summary['sequence_gaps'].sum()}")
    print(
        f"Maximum accelerometer saturation: "
        f"{summary['accel_saturation_percent'].max():.4f}%"
    )
    print(
        f"Maximum gyroscope saturation: "
        f"{summary['gyro_saturation_percent'].max():.4f}%"
    )
    print(f"Gesture silhouette score: {gesture_silhouette:.3f}")
    print(f"Participant silhouette score: {participant_silhouette:.3f}")
    print(f"Davies-Bouldin score: {davies_bouldin:.3f}")

    print("\nNearest-centroid accuracy by held-out participant:")
    for participant, accuracy in fold_results.items():
        print(f"  {participant}: {accuracy:.1%}")

    print("\nRecorded issues:")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  None")

    print(f"\nQC outputs saved to: {output_directory}")


if __name__ == "__main__":
    main()