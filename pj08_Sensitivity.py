import pandas as pd
import numpy as np
import os

# ========== File Paths ==========
BASE_DIR = "/data01/rong_dataset/Result/pj08/"
INPUT_FILES = {
    "1860-1969": os.path.join(BASE_DIR, "WWP_1860-1969_Z_c_i_t.csv"),
    "1970-2010": os.path.join(BASE_DIR, "WWP_1970-2010_Z_c_i_t.csv"),
    "2011-2024": os.path.join(BASE_DIR, "WWP_2011-2024_Z_c_i_t.csv"),
}
OUTPUT_TEMPLATE = os.path.join(BASE_DIR, "pj08_Sensitivity_{seg}.csv")

# ========== Thresholds ==========
thresholds = np.round(np.arange(0.5, 3.6, 0.1), 1)

# ========== Process Each Segment ==========
for seg, input_path in INPUT_FILES.items():
    if not os.path.exists(input_path):
        print(f"File not found, skip: {input_path}")
        continue

    print(f"Processing {seg} ...")
    df = pd.read_csv(input_path)

    # Extract Z-score year columns (from 3rd column onward)
    zscore_cols = df.columns[2:]

    results = []
    for threshold in thresholds:
        # Boolean DataFrame where True means Z >= threshold
        z_bool = df[zscore_cols] >= threshold

        # Convert boolean to int (True -> 1, False -> 0), ignoring NaNs
        z_int = z_bool.astype(float).where(~df[zscore_cols].isna())

        # Total count across all cells
        total_count = int(np.nansum(z_int.values))

        # Per row counts
        per_row_counts = np.nansum(z_int.values, axis=1)

        # Summary statistics
        avg_per_row = float(np.nanmean(per_row_counts))
        median_per_row = float(np.nanmedian(per_row_counts))
        max_per_row = int(np.nanmax(per_row_counts))

        results.append([
            threshold,
            total_count,
            avg_per_row,
            median_per_row,
            max_per_row
        ])

    # Save Results
    result_df = pd.DataFrame(results, columns=[
        "Z_Threshold",
        "Total_Count",
        "Average_Count_Per_Country_IPCR",
        "Median_Count_Per_Country_IPCR",
        "Max_Count_Per_Country_IPCR"
    ])

    output_path = OUTPUT_TEMPLATE.format(seg=seg)
    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
