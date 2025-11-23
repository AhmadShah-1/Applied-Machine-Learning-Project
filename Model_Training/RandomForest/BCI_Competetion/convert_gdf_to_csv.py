import mne
import pandas as pd
import numpy as np
import os
import glob

# Mapping of string event codes to labels
label_map = {
    '769': 1,  # left hand
    '770': 2,  # right hand
    '771': 3,  # feet
    '772': 4,  # tongue
}

def extract_trials_from_gdf(gdf_path, output_folder, subject_id):
    print(f"\nProcessing {gdf_path}")

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    annotations = raw.annotations
    print("Annotation descriptions:", annotations.description[:20])

    sfreq = raw.info['sfreq']

    tmax = 4  # 4-second MI window

    trials = []
    trial_id = 0

    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        desc_str = str(desc)

        if desc_str in label_map:
            label = label_map[desc_str]

            # Convert onset time (sec) to sample index
            start_sample = int(onset * sfreq)
            end_sample   = start_sample + int(tmax * sfreq)

            # Extract data
            data = raw.get_data(start=start_sample, stop=end_sample)
            df = pd.DataFrame(data.T, columns=raw.ch_names)

            df.insert(0, "Time", np.arange(len(df)) / sfreq)
            df["Trial_ID"] = trial_id
            df["Label"] = label
            df["Subject"] = subject_id

            trials.append(df)
            trial_id += 1

    if not trials:
        print("⚠️ No labeled MI trials found in:", gdf_path)
        return

    # Save combined
    out_df = pd.concat(trials, ignore_index=True)
    out_path = os.path.join(output_folder, f"{subject_id}_trials.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

def convert_all(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_path in glob.glob(os.path.join(source_folder, "*.gdf")):
        subject_id = os.path.basename(file_path).replace(".gdf", "")
        extract_trials_from_gdf(file_path, target_folder, subject_id)


if __name__ == "__main__":
    base = os.getcwd()

    source = os.path.join(base, "Dataset", "BCI Competition 2a", "BCICIV_2a_gdf")
    target = os.path.join(base, "Dataset", "BCI Competition 2a", "Trials")

    convert_all(source, target)
