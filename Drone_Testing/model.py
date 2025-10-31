"""Replay pre-recorded EEG samples against a trained model and drive AirSim."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import joblib  # type: ignore[import]

from airsim_controller import AirSimDroneController, AirSimUnavailableError
from drone_config import DroneConfig


def get_project_root() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "Model_Training").exists() and (parent / "Dataset").exists():
            return parent
    # Fallback: assume two levels up to keep previous behaviour.
    return resolved.parents[1]


PROJECT_ROOT = get_project_root()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model_Training" / "LogtisticRegression_RandomForest" / "models" / "eeg_intent_model.joblib"
DEFAULT_META_PATH = DEFAULT_MODEL_PATH.with_suffix(".meta.json")
DEFAULT_DATA_PATH = PROJECT_ROOT / "Dataset" / "Modified" / "directions" / "Move.xlsx"


class NullDroneController:
    """Fallback controller used when AirSim is unavailable or dry-run is requested."""

    def __init__(self, config: DroneConfig) -> None:
        self.config = config

    def __enter__(self) -> "NullDroneController":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:  # pragma: no cover - trivial
        return None

    def takeoff_and_hover(self) -> None:
        print("[DRY-RUN] takeoff and hover at", self.config.hover_height, "meters")

    def move_direction(self, direction: str) -> None:
        print(f"[DRY-RUN] move {direction}")

    def land(self) -> None:
        print("[DRY-RUN] land")

    def shutdown(self) -> None:
        pass


@dataclass
class IntentModel:
    model_path: Path
    expected_features: Optional[int] = None

    def __post_init__(self) -> None:
        self._model = joblib.load(self.model_path)

    def predict_label(self, features: Sequence[float]) -> int:
        if self.expected_features is not None and len(features) != self.expected_features:
            raise ValueError(
                f"Model expects {self.expected_features} features but received {len(features)}."
            )

        prediction = self._model.predict([features])[0]
        return int(prediction)

    def predict_probabilities(self, features: Sequence[float]) -> Optional[Sequence[float]]:
        predict_proba = getattr(self._model, "predict_proba", None)
        if predict_proba is None:
            return None
        probs = predict_proba([features])[0]
        return [float(p) for p in probs]


def load_metadata(meta_path: Optional[Path]) -> Tuple[Optional[int], Optional[List[str]]]:
    if meta_path is None or not meta_path.exists():
        return None, None

    metadata = json.loads(meta_path.read_text())
    feature_columns = metadata.get("feature_columns")
    n_features = metadata.get("n_features")
    return n_features, feature_columns


def iter_sample_rows(data_path: Path) -> Iterator[Tuple[int, List[float], int]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "Reading Excel data requires pandas. Install it with 'pip install pandas'."
            ) from exc

        df = pd.read_excel(data_path)
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            raw_values = [value for value in row.tolist() if pd.notna(value)]
            if len(raw_values) < 3:
                continue

            try:
                numeric = [float(value) for value in raw_values[2:]]
            except ValueError as exc:
                raise ValueError(f"Row {idx} contains non-numeric data: {raw_values}") from exc

            if len(numeric) < 1:
                continue

            *feature_values, target_value = numeric
            target_label = int(round(target_value))
            yield idx, feature_values, target_label
        return

    pattern = re.compile(r"[\s,]+")
    with data_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            tokens = [token for token in pattern.split(stripped) if token]
            if len(tokens) < 3:
                continue

            payload = tokens[2:]

            try:
                values = [float(token) for token in payload]
            except ValueError:
                if idx == 1:
                    # Treat the first line as an optional header.
                    continue
                raise

            if len(values) < 1:
                raise ValueError(
                    f"Row {idx} does not contain enough values to parse features and target after removing IDs."
                )

            *feature_values, target_value = values
            target_label = int(round(target_value))
            yield idx, feature_values, target_label


def choose_controller(config: DroneConfig, dry_run: bool) -> Tuple[object, bool]:
    if dry_run:
        return NullDroneController(config), True

    try:
        return AirSimDroneController(config), False
    except AirSimUnavailableError as exc:
        print(f"Warning: {exc}. Falling back to dry-run mode.", file=sys.stderr)
        return NullDroneController(config), True


def replay_model(
    model_path: Path,
    data_path: Path,
    meta_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> None:
    n_features, feature_columns = load_metadata(meta_path)
    config = DroneConfig()
    intent_model = IntentModel(model_path=model_path, expected_features=n_features)
    controller, is_dry_run = choose_controller(config, dry_run=dry_run)

    if feature_columns:
        print(f"Using feature columns: {', '.join(feature_columns)}")

    with controller as drone:
        try:
            drone.takeoff_and_hover()

            for row_idx, features, target in iter_sample_rows(data_path):
                prompt = input(
                    f"Row {row_idx}: press Enter to send command or 'q' to quit > "
                ).strip()
                if prompt.lower().startswith("q"):
                    print("Stopping replay.")
                    break

                prediction = intent_model.predict_label(features)
                predicted_direction = DroneConfig.direction_from_target(prediction)
                target_direction = DroneConfig.direction_from_target(target)
                probs = intent_model.predict_probabilities(features)

                print(f"Predicted: {prediction} -> {predicted_direction}")
                print(f"Ground truth: {target} -> {target_direction}")
                if probs is not None:
                    print("Probabilities:", ", ".join(f"{p:.3f}" for p in probs))

                if prediction == target:
                    print("Comparison: match")
                else:
                    print("Comparison: mismatch")

                drone.move_direction(predicted_direction)

            if not is_dry_run:
                drone.land()
        finally:
            drone.shutdown()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model joblib file (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=DEFAULT_META_PATH,
        help=f"Path to the metadata JSON file (default: {DEFAULT_META_PATH}).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the replay dataset (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send commands to AirSim; only log predicted movements.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    replay_model(
        model_path=args.model_path,
        data_path=args.data_path,
        meta_path=args.meta_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

