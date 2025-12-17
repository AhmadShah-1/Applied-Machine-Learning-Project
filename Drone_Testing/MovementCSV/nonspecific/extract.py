# This file extracts the last 4000 data points from a CSV file and saves them to a new CSV file

import pandas as pd

input_csv = "C:/Users/ahmad/OneDrive/General/Obsidian2/Classes/Fall 2025/EE595_AppliedMachineLearning/Project/Applied-Machine-Learning-Project/Dataset/BCI Competition 2a/Trials/A01T_trials.csv"
output_csv = "A01T_trials_last4000.csv"

df = pd.read_csv(input_csv)

df_last50 = df.tail(4000)

df_last50.to_csv(output_csv, index=False)

print(f"Saved last 50 rows to {output_csv}")
