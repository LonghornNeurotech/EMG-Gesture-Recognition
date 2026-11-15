#!/usr/bin/env python3
"""Detailed numerical QC report for the seven-class IMU dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset_qc import (
    DEFAULT_GESTURES,
    calculate_separation,
    load_dataset,
    normalize_from_training,
    prepare_features,
)


PARTICIPANTS = ("P002", "P003", "P004", "P005", "P006")
SESSION = "S01"
GESTURES = DEFAULT_GESTURES


def nearest_centroid(projection, labels, gestures):
    centroids = np.stack(
        [
            projection[labels == gesture].mean(axis=0)
            for gesture in gestures
        ]
    )

    distances = np.linalg.norm(
        projection[:, None, :] - centroids[None, :, :],
        axis=2,
    )

    nearest_indices = np.argmin(distances, axis=1)
    predictions = np.array(gestures)[nearest_indices]

    label_indices = {
        gesture: index
        for index, gesture in enumerate(gestures)
    }
    true_indices = np.array(
        [label_indices[label] for label in labels]
    )

    true_distances = distances[
        np.arange(len(distances)),
        true_indices,
    ]

    sorted_distances = np.sort(distances, axis=1)
    margins = sorted_distances[:, 1] - sorted_distances[:, 0]

    return predictions, distances, true_distances, margins


def own_class_outlier_scores(distances, labels, gestures):
    scores = np.zeros(len(labels))

    for gesture in gestures:
        selected = labels == gesture
        values = distances[selected]

        mean = values.mean()
        standard_deviation = values.std()

        if standard_deviation == 0:
            standard_deviation = 1

        scores[selected] = (
            values - mean
        ) / standard_deviation

    return scores


def leave_one_participant_out(
    trials,
    participants,
    labels,
    gestures,
):
    results = []

    for held_out in sorted(set(participants)):
        train_mask = participants != held_out
        test_mask = participants == held_out
        test_indices = np.flatnonzero(test_mask)

        train_features, test_features = normalize_from_training(
            trials[train_mask],
            trials[test_mask],
        )

        component_count = min(20, len(train_features) - 1)
        pca = PCA(n_components=component_count)

        train_projection = pca.fit_transform(train_features)
        test_projection = pca.transform(test_features)

        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        centroids = np.stack(
            [
                train_projection[
                    train_labels == gesture
                ].mean(axis=0)
                for gesture in gestures
            ]
        )

        distances = np.linalg.norm(
            test_projection[:, None, :]
            - centroids[None, :, :],
            axis=2,
        )

        nearest_indices = np.argmin(distances, axis=1)
        predictions = np.array(gestures)[nearest_indices]

        label_indices = {
            gesture: index
            for index, gesture in enumerate(gestures)
        }

        for local_index, global_index in enumerate(test_indices):
            true_label = test_labels[local_index]
            true_index = label_indices[true_label]
            sorted_distances = np.sort(distances[local_index])

            results.append(
                {
                    "trial_index": global_index,
                    "held_out_participant": held_out,
                    "loso_prediction": predictions[local_index],
                    "loso_correct": (
                        predictions[local_index] == true_label
                    ),
                    "loso_nearest_distance": (
                        sorted_distances[0]
                    ),
                    "loso_true_class_distance": (
                        distances[local_index, true_index]
                    ),
                    "loso_margin": (
                        sorted_distances[1]
                        - sorted_distances[0]
                    ),
                }
            )

    return pd.DataFrame(results).sort_values("trial_index")


def relative_path(path, repository):
    try:
        return str(Path(path).relative_to(repository))
    except ValueError:
        return str(path)


def dataframe_as_percent_table(matrix, labels):
    return pd.DataFrame(
        matrix * 100,
        index=labels,
        columns=labels,
    ).round(1)


def format_trials(frame, columns, limit=50):
    if frame.empty:
        return "  None"

    selected = frame.loc[:, columns].head(limit)
    text = selected.to_string(index=False)

    if len(frame) > limit:
        text += f"\n  ... plus {len(frame) - limit} more"

    return text


def main():
    repository = Path(__file__).resolve().parents[1]
    data_root = repository / "data" / "raw"
    output_directory = (
    repository
    / "docs"
    / "qc_detailed_7class_P002-P006"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    records, issues = load_dataset(
        data_root,
        PARTICIPANTS,
        SESSION,
        GESTURES,
    )

    trials, features, participants, labels = prepare_features(
        records
    )

    base = pd.DataFrame(
        {
            "trial_index": np.arange(len(records)),
            "participant": participants,
            "gesture": labels,
            "file": [
                relative_path(record["path"], repository)
                for record in records
            ],
            "rate_hz": [
                record["rate_hz"] for record in records
            ],
            "jitter_ms": [
                record["jitter_ms"] for record in records
            ],
        }
    )

    # Shared two-dimensional PCA used only for visualization.
    pca_2d = PCA(n_components=2)
    coordinates_2d = pca_2d.fit_transform(features)

    (
        pca2_predictions,
        _,
        pca2_true_distances,
        pca2_margins,
    ) = nearest_centroid(
        coordinates_2d,
        labels,
        GESTURES,
    )

    base["PC1"] = coordinates_2d[:, 0]
    base["PC2"] = coordinates_2d[:, 1]
    base["pca2_nearest_class"] = pca2_predictions
    base["pca2_correct"] = pca2_predictions == labels
    base["pca2_true_class_distance"] = pca2_true_distances
    base["pca2_margin"] = pca2_margins

    # Higher-dimensional PCA retaining 95% of dataset variance.
    metric_projection, separation = calculate_separation(
        features,
        labels,
        GESTURES,
    )

    (
        global_predictions,
        _,
        global_true_distances,
        global_margins,
    ) = nearest_centroid(
        metric_projection,
        labels,
        GESTURES,
    )

    base["global_nearest_class"] = global_predictions
    base["global_correct"] = global_predictions == labels
    base["global_true_class_distance"] = global_true_distances
    base["global_margin"] = global_margins
    base["own_class_outlier_z"] = own_class_outlier_scores(
        global_true_distances,
        labels,
        GESTURES,
    )

    # True cross-participant diagnostic.
    loso = leave_one_participant_out(
        trials,
        participants,
        labels,
        GESTURES,
    )

    details = base.merge(
        loso,
        on="trial_index",
        how="left",
        validate="one_to_one",
    )

    details.to_csv(
        output_directory / "trial_diagnostics.csv",
        index=False,
    )

    separation_frame = pd.DataFrame(
        separation,
        index=GESTURES,
        columns=GESTURES,
    )
    separation_frame.to_csv(
        output_directory / "pairwise_separation.csv"
    )

    loso_matrix = confusion_matrix(
        details["gesture"],
        details["loso_prediction"],
        labels=GESTURES,
        normalize="true",
    )
    loso_percent = dataframe_as_percent_table(
        loso_matrix,
        GESTURES,
    )
    loso_percent.to_csv(
        output_directory / "loso_confusion_percent.csv"
    )

    participant_accuracy = (
        details.groupby("participant")["loso_correct"]
        .mean()
        .mul(100)
        .round(1)
    )
    participant_accuracy.to_csv(
        output_directory / "participant_accuracy.csv",
        header=["accuracy_percent"],
    )

    per_class_recall = (
        details.groupby(["participant", "gesture"])["loso_correct"]
        .mean()
        .mul(100)
        .round(1)
        .unstack()
        .reindex(columns=GESTURES)
    )
    per_class_recall.to_csv(
        output_directory / "participant_class_recall.csv"
    )

    confusion_pairs = (
        details.loc[
            ~details["loso_correct"],
            ["gesture", "loso_prediction"],
        ]
        .value_counts()
        .rename("count")
        .reset_index()
    )
    confusion_pairs.to_csv(
        output_directory / "loso_confusion_pairs.csv",
        index=False,
    )

    idle = details[details["gesture"] == "idle"]

    idle_pca2_anomalies = idle[
        idle["pca2_nearest_class"] != "idle"
    ]
    idle_global_anomalies = idle[
        idle["global_nearest_class"] != "idle"
    ]
    idle_loso_anomalies = idle[
        idle["loso_prediction"] != "idle"
    ]

    strong_anomalies = details[
        (~details["global_correct"])
        & (~details["loso_correct"])
    ]

    top_outliers = details.sort_values(
        "own_class_outlier_z",
        ascending=False,
    ).head(20)

    report = []

    report.append("SEVEN-CLASS IMU DATASET QC")
    report.append("=" * 70)
    report.append(f"Trials: {len(details)}")
    report.append(
        "PC1 explained variance: "
        f"{pca_2d.explained_variance_ratio_[0] * 100:.2f}%"
    )
    report.append(
        "PC2 explained variance: "
        f"{pca_2d.explained_variance_ratio_[1] * 100:.2f}%"
    )
    report.append(
        "PC1 + PC2 explained variance: "
        f"{pca_2d.explained_variance_ratio_.sum() * 100:.2f}%"
    )
    report.append(
        "Global nearest-centroid agreement: "
        f"{accuracy_score(labels, global_predictions):.1%}"
    )
    report.append(
        "LOSO nearest-centroid accuracy: "
        f"{accuracy_score(labels, details['loso_prediction']):.1%}"
    )

    report.append("\nHELD-OUT PARTICIPANT ACCURACY")
    report.append(participant_accuracy.to_string())

    report.append("\nHELD-OUT PARTICIPANT/CLASS RECALL (%)")
    report.append(per_class_recall.to_string())

    report.append("\nOVERALL LOSO CONFUSION MATRIX (%)")
    report.append(loso_percent.to_string())

    report.append("\nLOSO ERROR PAIRS")
    if confusion_pairs.empty:
        report.append("  None")
    else:
        report.append(confusion_pairs.to_string(index=False))

    report.append("\nPAIRWISE SEPARATION — HIGHER IS BETTER")
    report.append(separation_frame.round(2).to_string())

    columns = [
        "participant",
        "gesture",
        "file",
        "pca2_nearest_class",
        "global_nearest_class",
        "loso_prediction",
        "own_class_outlier_z",
    ]

    report.append("\nIDLE TRIALS NEAREST TO ANOTHER CLASS IN 2D PCA")
    report.append(format_trials(idle_pca2_anomalies, columns))

    report.append("\nIDLE TRIALS NEAREST TO ANOTHER CLASS GLOBALLY")
    report.append(format_trials(idle_global_anomalies, columns))

    report.append("\nIDLE TRIALS MISCLASSIFIED WHEN PARTICIPANT IS HELD OUT")
    report.append(format_trials(idle_loso_anomalies, columns))

    report.append("\nTRIALS WRONG GLOBALLY AND IN HELD-OUT TESTING")
    report.append(format_trials(strong_anomalies, columns))

    report.append("\nTOP 20 OWN-CLASS OUTLIERS")
    report.append(format_trials(top_outliers, columns, limit=20))

    report.append("\nRECORDED ACQUISITION ISSUES")
    if issues:
        report.extend(f"  - {issue}" for issue in issues)
    else:
        report.append("  None")

    report_path = output_directory / "report.txt"
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Detailed report saved to: {report_path}")
    print(
        f"2D PCA idle anomalies: {len(idle_pca2_anomalies)}"
    )
    print(
        f"Higher-dimensional idle anomalies: "
        f"{len(idle_global_anomalies)}"
    )
    print(
        f"Held-out idle misclassifications: "
        f"{len(idle_loso_anomalies)}"
    )


if __name__ == "__main__":
    main()