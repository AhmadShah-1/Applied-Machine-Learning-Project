"""Replay pre-recorded EEG samples against a trained model and drive AirSim."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from mne.filter import filter_data

from airsim_controller import AirSimDroneController, AirSimUnavailableError
from drone_config import DroneConfig


def get_project_root() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "Model_Training").exists() and (parent / "Dataset").exists():
            return parent
    return resolved.parents[1]


PROJECT_ROOT = get_project_root()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model_Training" / "RandomForest" / "BCI_Competetion" / "models" / "csp_rf_model_v2.joblib"
# Default to the first subject's trials if available, otherwise fallback
DEFAULT_DATA_PATH = PROJECT_ROOT / "Dataset" / "BCI Competition 2a" / "Trials" / "A01T_trials.csv"


class NullDroneController:
    """Fallback controller used when AirSim is unavailable or dry-run is requested."""

    def __init__(self, config: DroneConfig) -> None:
        self.config = config

    def __enter__(self) -> "NullDroneController":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:
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
    # The new pipeline model expects (n_epochs, n_channels, n_times)
    # We expect 22 channels and 1000 timepoints
    expected_channels: int = 22
    expected_times: int = 1000

    # Filter settings (same from training), we need to apply the filters on the data again
    sfreq: float = 250.0
    l_freq: float = 8.0
    h_freq: float = 30.0

    def __post_init__(self) -> None:
        print(f"Loading model from {self.model_path}...")
        self._model = joblib.load(self.model_path)

    def predict_label(self, trial_data: np.ndarray) -> int:
        if trial_data.ndim == 2:
            trial_data = trial_data[np.newaxis, :, :]
        
        if trial_data.shape[1] != self.expected_channels:
             raise ValueError(f"Expected {self.expected_channels} channels, got {trial_data.shape[1]}")

        # Handle time dimension mismatch by padding or trimming
        current_times = trial_data.shape[2]
        if current_times != self.expected_times:
            if current_times < self.expected_times:
                # Pad with zeros
                pad_width = ((0, 0), (0, 0), (0, self.expected_times - current_times))
                trial_data = np.pad(trial_data, pad_width, mode='constant')
                warnings.warn(f"Input data padded from {current_times} to {self.expected_times} samples.")
            else:
                # Trim
                trial_data = trial_data[:, :, :self.expected_times]
                warnings.warn(f"Input data trimmed from {current_times} to {self.expected_times} samples.")

        # APPLY BANDPASS FILTER (8-30Hz)
        trial_data = filter_data(
            trial_data, 
            sfreq=self.sfreq, 
            l_freq=self.l_freq, 
            h_freq=self.h_freq, 
            verbose=False
        )

        prediction = self._model.predict(trial_data)[0]
        return int(prediction)

    def predict_probabilities(self, trial_data: np.ndarray) -> Optional[Sequence[float]]:
        if trial_data.ndim == 2:
            trial_data = trial_data[np.newaxis, :, :]
            
        current_times = trial_data.shape[2]
        if current_times < self.expected_times:
             pad_width = ((0, 0), (0, 0), (0, self.expected_times - current_times))
             trial_data = np.pad(trial_data, pad_width, mode='constant')
        elif current_times > self.expected_times:
             trial_data = trial_data[:, :, :self.expected_times]

        # Apply same filter for probabilities
        trial_data = filter_data(
            trial_data, 
            sfreq=self.sfreq, 
            l_freq=self.l_freq, 
            h_freq=self.h_freq, 
            verbose=False
        )

        predict_proba = getattr(self._model, "predict_proba", None)
        if predict_proba is None:
            return None
        probs = predict_proba(trial_data)[0]
        return [float(p) for p in probs]


def iter_trials_randomly(data_dir: Path) -> Iterator[Tuple[str, np.ndarray, int]]:
    # Map input keys to filenames
    key_to_file = {
        'a': "left.csv",
        'd': "right.csv",
        's': "backward.csv",
        'w': "forward.csv"
    }

    # load all dataframes so we don;t have to keep reading from disk every time
    dfs = {}
    print("Pre-loading datasets...")
    for key, filename in key_to_file.items():
        path = data_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            dfs[key] = [group for _, group in df.groupby("Trial_ID")]
            print(f"Loaded {len(dfs[key])} trials from {filename}")
        else:
            print(f"Warning: {filename} not found in {data_dir}")

    while True:
        prompt = input("\nPress (w/a/s/d) to pick a trial direction, or 'q' to quit > ").lower().strip()
        
        if prompt == 'q':
            break
            
        if prompt not in dfs:
            print(f"Invalid input. Use w, a, s, d. (Available: {list(dfs.keys())})")
            continue
            
        # Pick a random trial from the selected direction
        trials = dfs[prompt]
        if not trials:
            print("No trials available for this direction.")
            continue
            
        import random
        trial_df = random.choice(trials)
        
        # Process trial
        eeg_cols = [c for c in trial_df.columns if c.startswith("EEG")]
        features_T = trial_df[eeg_cols].to_numpy()
        features = features_T.T
        label = int(trial_df["Label"].iloc[0])
        
        source_name = key_to_file[prompt]
        yield source_name, features, label


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
    *,
    dry_run: bool = False,
) -> None:
    config = DroneConfig()
    intent_model = IntentModel(model_path=model_path)
    controller, is_dry_run = choose_controller(config, dry_run=dry_run)

    if data_path.is_file():
        print("Single file provided, running sequential replay...")
        data_dir = data_path.parent
    else:
        data_dir = data_path

    with controller as drone:
        try:
            drone.takeoff_and_hover()

            for source_name, features, target in iter_trials_randomly(data_dir):
                try:
                    prediction = intent_model.predict_label(features)
                    probs = intent_model.predict_probabilities(features)
                    
                    predicted_direction = DroneConfig.direction_from_target(prediction)
                    try:
                        target_direction = DroneConfig.direction_from_target(target)
                    except ValueError:
                        target_direction = f"Unknown({target})"

                    print(f"\nSource: {source_name}")
                    print(f"Predicted: {prediction} ({predicted_direction})")
                    print(f"Ground Truth: {target} ({target_direction})")
                    
                    if probs is not None:
                        print("Probabilities:", ", ".join(f"{p:.3f}" for p in probs))

                    if prediction == target:
                        print("Result: MATCH")
                    else:
                        print("Result: MISMATCH")

                    drone.move_direction(predicted_direction)
                    
                except ValueError as e:
                    print(f"Error processing trial: {e}")
                    continue

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
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "Drone_Testing" / "MovementCSV", 
        help=f"Path to the directory containing direction CSVs (default: Drone_Testing/MovementCSV).",
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
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
