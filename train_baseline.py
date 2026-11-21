"""Train and evaluate a first faceoff-outcome video classifier."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Dataset labels and the corresponding folders used for curated clips.
LABELS = ("CLEAN", "DISMISSAL", "RESET")
REFERENCE_DIRS = {
    "CLEAN": "clean_examples",
    "DISMISSAL": "dismissal_examples",
    "RESET": "reset_examples",
}
REQUIRED_COLUMNS = ("game_id", "clip_filename", "linesman_id", "label")
TEMPORAL_BINS = 4
SPATIAL_GRID = 2
DIRECTION_BINS = 8


# Keep each annotation together with the resolved clip path used by training.
@dataclass(frozen=True)
class ClipRecord:
    game_id: str
    clip_filename: str
    linesman_id: str
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    """Configure paths and video sampling settings from command-line options."""
    project_root = Path(__file__).resolve().parent
    default_annotations = project_root.parent / "OneDrive" / "annotations.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=default_annotations)
    parser.add_argument("--clips-root", type=Path, default=project_root / "raw_clips")
    parser.add_argument(
        "--reference-root", type=Path, default=project_root / "reference_library"
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "artifacts")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    return parser.parse_args()


def _clip_name(filename: str) -> str:
    """Allow annotation filenames with or without their media extension."""
    path = Path(filename)
    return filename if path.suffix else f"{filename}.mp4"


def load_records(
    annotations_path: Path, clips_root: Path, reference_root: Path
) -> list[ClipRecord]:
    """Validate annotations and prefer trimmed reference clips over raw clips."""
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {annotations_path}")

    records: list[ClipRecord] = []
    seen: set[tuple[str, str]] = set()
    with annotations_path.open(newline="", encoding="utf-8-sig") as annotations_file:
        reader = csv.DictReader(annotations_file)
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Annotation CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            game_id = (row.get("game_id") or "").strip()
            filename = (row.get("clip_filename") or "").strip()
            label = (row.get("label") or "").strip().upper()
            linesman_id = (row.get("linesman_id") or "").strip()
            if not game_id or not filename:
                raise ValueError(
                    f"Annotation row {row_number} has an empty game_id or clip_filename."
                )
            if label not in LABELS:
                raise ValueError(
                    f"Annotation row {row_number} has unsupported label {label!r}; "
                    f"expected one of {', '.join(LABELS)}."
                )

            key = (game_id, filename)
            if key in seen:
                raise ValueError(
                    f"Duplicate game_id/clip_filename at annotation row {row_number}: "
                    f"{game_id}/{filename}"
                )
            seen.add(key)

            # Use the curated version whenever available; raw footage is fallback.
            clip_name = _clip_name(filename)
            raw_path = clips_root / game_id / clip_name
            reference_paths = [
                reference_root / folder / clip_name
                for folder in REFERENCE_DIRS.values()
                if (reference_root / folder / clip_name).is_file()
            ]
            label_reference_path = (
                reference_root / REFERENCE_DIRS[label] / clip_name
            )
            if label_reference_path.is_file():
                clip_path = label_reference_path
            elif len(reference_paths) == 1:
                clip_path = reference_paths[0]
            elif reference_paths:
                raise ValueError(
                    f"Ambiguous reference clip for annotation row {row_number}: "
                    + ", ".join(str(path) for path in reference_paths)
                )
            elif raw_path.is_file():
                clip_path = raw_path
            else:
                raise FileNotFoundError(
                    f"No clip found for annotation row {row_number}: "
                    f"tried all label folders under {reference_root} and {raw_path}"
                )

            records.append(
                ClipRecord(
                    game_id=game_id,
                    clip_filename=filename,
                    linesman_id=linesman_id,
                    label=label,
                    path=clip_path,
                )
            )

    if not records:
        raise ValueError(f"No annotated clips found in {annotations_path}")
    missing_labels = set(LABELS) - {record.label for record in records}
    if missing_labels:
        raise ValueError(
            "The dataset needs examples for every configured class; missing: "
            + ", ".join(sorted(missing_labels))
        )
    return records


def sample_video_frames(
    video_path: Path, frame_count: int, width: int, height: int
) -> list[np.ndarray]:
    """Sample evenly spaced, resized grayscale frames from one video."""
    if frame_count < 2:
        raise ValueError("--frames must be at least 2.")
    if width < 1 or height < 1:
        raise ValueError("--width and --height must be positive.")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            raise ValueError(
                f"OpenCV could not determine a usable frame count for {video_path}"
            )

        indexes = np.linspace(
            0, total_frames - 1, num=min(frame_count, total_frames), dtype=np.int64
        )
        wanted = set(indexes.tolist())
        frames: list[np.ndarray] = []
        frame_index = 0
        while frame_index <= int(indexes[-1]):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                frames.append(gray.astype(np.float32) / 255.0)
            frame_index += 1
    finally:
        capture.release()

    if len(frames) < 2:
        raise ValueError(f"Could not sample at least two frames from {video_path}")
    return frames


def extract_flow_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Summarize motion direction and magnitude across time and image regions."""
    if len(frames) < 2:
        raise ValueError("At least two frames are needed to calculate optical flow.")

    features: list[float] = []
    flow_count = len(frames) - 1
    # Divide motion into temporal chunks so rough event timing is retained.
    for temporal_bin in range(TEMPORAL_BINS):
        start = math.floor(temporal_bin * flow_count / TEMPORAL_BINS)
        end = math.floor((temporal_bin + 1) * flow_count / TEMPORAL_BINS)
        if end <= start:
            features.extend([0.0] * (SPATIAL_GRID * SPATIAL_GRID * (DIRECTION_BINS + 1)))
            features.extend([0.0] * 4)
            continue

        direction_histograms = np.zeros(
            (SPATIAL_GRID, SPATIAL_GRID, DIRECTION_BINS), dtype=np.float64
        )
        magnitude_totals = np.zeros(
            (SPATIAL_GRID, SPATIAL_GRID), dtype=np.float64
        )
        all_horizontal: list[np.ndarray] = []
        all_vertical: list[np.ndarray] = []
        all_magnitudes: list[np.ndarray] = []

        for flow_index in range(start, end):
            # Dense optical flow estimates each pixel's movement between frames.
            flow = cv2.calcOpticalFlowFarneback(
                frames[flow_index],
                frames[flow_index + 1],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            horizontal = flow[:, :, 0]
            vertical = flow[:, :, 1]
            magnitude, angle = cv2.cartToPolar(horizontal, vertical, angleInDegrees=False)
            direction_indexes = np.floor(
                (angle % (2 * np.pi)) * DIRECTION_BINS / (2 * np.pi)
            ).astype(np.int32)
            direction_indexes = np.clip(direction_indexes, 0, DIRECTION_BINS - 1)

            height, width = magnitude.shape
            for row in range(SPATIAL_GRID):
                y0, y1 = row * height // SPATIAL_GRID, (row + 1) * height // SPATIAL_GRID
                for column in range(SPATIAL_GRID):
                    x0, x1 = (
                        column * width // SPATIAL_GRID,
                        (column + 1) * width // SPATIAL_GRID,
                    )
                    cell_bins = np.bincount(
                        direction_indexes[y0:y1, x0:x1].ravel(),
                        weights=magnitude[y0:y1, x0:x1].ravel(),
                        minlength=DIRECTION_BINS,
                    )
                    direction_histograms[row, column] += cell_bins
                    magnitude_totals[row, column] += float(
                        magnitude[y0:y1, x0:x1].mean()
                    )

            all_horizontal.append(horizontal.ravel())
            all_vertical.append(vertical.ravel())
            all_magnitudes.append(magnitude.ravel())

        for row in range(SPATIAL_GRID):
            for column in range(SPATIAL_GRID):
                histogram = direction_histograms[row, column]
                total = histogram.sum()
                # Normalize direction counts so feature size is less tied to motion volume.
                if total > 0:
                    histogram = histogram / total
                features.extend(histogram.tolist())
                features.append(
                    float(np.log1p(magnitude_totals[row, column] / (end - start)))
                )

        horizontal_values = np.concatenate(all_horizontal)
        vertical_values = np.concatenate(all_vertical)
        magnitude_values = np.concatenate(all_magnitudes)
        features.extend(
            [
                float(horizontal_values.mean()),
                float(vertical_values.mean()),
                float(magnitude_values.mean()),
                float(magnitude_values.std()),
            ]
        )

    feature_array = np.asarray(features, dtype=np.float32)
    if not np.isfinite(feature_array).all():
        raise ValueError("Optical-flow feature extraction produced non-finite values.")
    return feature_array


def features_for_records(
    records: Sequence[ClipRecord], frame_count: int, width: int, height: int
) -> tuple[list[ClipRecord], np.ndarray, list[tuple[ClipRecord, str]]]:
    """Extract features, keeping unreadable clips in a separate error report."""
    readable_records: list[ClipRecord] = []
    all_features = []
    unreadable_records: list[tuple[ClipRecord, str]] = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] Extracting {record.game_id}/{record.clip_filename}")
        try:
            frames = sample_video_frames(record.path, frame_count, width, height)
            features = extract_flow_features(frames)
        except ValueError as error:
            message = str(error)
            unreadable_records.append((record, message))
            print(f"WARNING: Excluding unreadable clip: {message}")
            continue
        readable_records.append(record)
        all_features.append(features)
    if not all_features:
        raise ValueError("No annotated clips could be decoded; no model was trained.")
    return readable_records, np.stack(all_features), unreadable_records


def make_classifier() -> SVC:
    """Build a scaled SVM that compensates for the dataset's class imbalance."""
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
    )


def evaluate(records: Sequence[ClipRecord], features: np.ndarray) -> np.ndarray:
    """Measure generalization by holding out each complete game in turn."""
    groups = np.asarray([record.game_id for record in records])
    labels = np.asarray([record.label for record in records])
    unique_games = np.unique(groups)
    if len(unique_games) < 2:
        raise ValueError("At least two distinct games are required for grouped evaluation.")

    predictions = np.empty(labels.shape, dtype=object)
    splitter = GroupKFold(n_splits=len(unique_games))
    for train_indexes, test_indexes in splitter.split(features, labels, groups):
        # Ensure every class is available for training without leaking held-out games.
        train_labels = set(labels[train_indexes])
        missing = set(LABELS) - train_labels
        if missing:
            test_game = groups[test_indexes][0]
            raise ValueError(
                f"Cannot evaluate held-out game {test_game}: its training split "
                f"would be missing classes {', '.join(sorted(missing))}."
            )
        classifier = make_classifier()
        classifier.fit(features[train_indexes], labels[train_indexes])
        predictions[test_indexes] = classifier.predict(features[test_indexes])

    print("\nLeave-one-game-out evaluation (all clips from each game held out together):")
    print(
        classification_report(
            labels, predictions, labels=LABELS, zero_division=0, digits=3
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(labels, predictions, labels=LABELS))
    return predictions


def write_predictions(
    path: Path, records: Sequence[ClipRecord], predictions: Sequence[str]
) -> None:
    """Save each cross-validation prediction beside its annotation metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "game_id",
                "clip_filename",
                "linesman_id",
                "actual_label",
                "predicted_label",
            ]
        )
        for record, prediction in zip(records, predictions):
            writer.writerow(
                [
                    record.game_id,
                    record.clip_filename,
                    record.linesman_id,
                    record.label,
                    prediction,
                ]
            )


def write_unreadable_clips(
    path: Path, unreadable_records: Sequence[tuple[ClipRecord, str]]
) -> None:
    """Save videos that could not be decoded instead of silently dropping them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["game_id", "clip_filename", "linesman_id", "actual_label", "error"]
        )
        for record, error in unreadable_records:
            writer.writerow(
                [
                    record.game_id,
                    record.clip_filename,
                    record.linesman_id,
                    record.label,
                    error,
                ]
            )


def main() -> None:
    """Run data loading, feature extraction, evaluation, training, and exports."""
    args = parse_args()
    records = load_records(args.annotations, args.clips_root, args.reference_root)
    readable_records, features, unreadable_records = features_for_records(
        records, args.frames, args.width, args.height
    )
    games = sorted({record.game_id for record in readable_records})
    print(
        f"Using {len(readable_records)} readable clips from {len(games)} games; "
        f"excluded {len(unreadable_records)} unreadable clip(s); labels: "
        + ", ".join(
            f"{label}={sum(record.label == label for record in readable_records)}"
            for label in LABELS
        )
    )
    predictions = evaluate(readable_records, features)

    # Train the deployable model on all readable clips after held-out evaluation.
    final_classifier = make_classifier()
    final_classifier.fit(
        features, np.asarray([record.label for record in readable_records])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "faceoff_baseline.joblib"
    joblib.dump(
        {
            "classifier": final_classifier,
            "labels": LABELS,
            "feature_config": {
                "frames": args.frames,
                "width": args.width,
                "height": args.height,
                "feature_type": "dense_optical_flow_histograms",
            },
        },
        model_path,
    )
    predictions_path = args.output_dir / "grouped_cv_predictions.csv"
    write_predictions(predictions_path, readable_records, predictions)
    unreadable_path = args.output_dir / "unreadable_clips.csv"
    write_unreadable_clips(unreadable_path, unreadable_records)
    print(f"Saved trained model: {model_path}")
    print(f"Saved held-out predictions: {predictions_path}")
    if unreadable_records:
        print(f"Saved unreadable clip report: {unreadable_path}")


if __name__ == "__main__":
    main()
