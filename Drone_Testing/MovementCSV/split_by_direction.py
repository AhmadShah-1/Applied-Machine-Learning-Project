import pandas as pd
import os

INPUT_CSV = r"c:\Users\ahmad\OneDrive\General\Obsidian2\Classes\Fall 2025\EE595_AppliedMachineLearning\Project\Applied-Machine-Learning-Project\Dataset\BCI Competition 2a\Trials\A01T_trials.csv"
OUTPUT_DIR = r"c:\Users\ahmad\OneDrive\General\Obsidian2\Classes\Fall 2025\EE595_AppliedMachineLearning\Project\Applied-Machine-Learning-Project\Drone_Testing\MovementCSV"

LABEL_MAP = {
    1: "left.csv",
    2: "right.csv",
    3: "backward.csv",
    4: "forward.csv"
}

def split_csv_by_label():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at {INPUT_CSV}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if "Label" not in df.columns:
        print("Error: 'Label' column not found in CSV.")
        return

    # Group by label and save separate files
    for label, filename in LABEL_MAP.items():
        subset = df[df["Label"] == label]
        
        if subset.empty:
            print(f"Warning: No data found for Label {label} ({filename})")
            continue

        output_path = os.path.join(OUTPUT_DIR, filename)
        subset.to_csv(output_path, index=False)
        print(f"Saved {len(subset)} rows to {output_path}")

    print("Done splitting CSVs.")

if __name__ == "__main__":
    split_csv_by_label()

