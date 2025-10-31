'''
This file is used to split the dataset into a training set and a test set.
'''

from pathlib import Path
import sys
import pandas as pd 

def split_csv_sequential(original_csv_path: Path, modified_dir_path: Path, first_ratio: float = 0.8) -> None:
    modified_dir_path.mkdir(parents=True, exist_ok=True)

    first_output_path = modified_dir_path / "eeg_neuroprosthetic_dataset_first_80.csv"
    last_output_path = modified_dir_path / "eeg_neuroprosthetic_dataset_last_20.csv"

    dataset = pd.read_csv(original_csv_path)
    total_rows = len(dataset)
    split_index = int(total_rows * first_ratio)

    # Ensure both splits have at least one row when possible
    split_index = max(1, min(split_index, total_rows - 1)) if total_rows > 1 else total_rows
    dataset.iloc[:split_index].to_csv(first_output_path, index=False)
    dataset.iloc[split_index:].to_csv(last_output_path, index=False)

    

if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    source_csv = project_root / "Dataset" / "Original" / "eeg_neuroprosthetic_dataset.csv"
    target_dir = project_root / "Dataset" / "Modified"

    split_csv_sequential(source_csv, target_dir, first_ratio=0.8)

    print("Created:")
    print(f" - {target_dir / 'eeg_neuroprosthetic_dataset_first_80.csv'}")
    print(f" - {target_dir / 'eeg_neuroprosthetic_dataset_last_20.csv'}")


