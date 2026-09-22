# -*- coding: utf-8 -*-
import os
import gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # 用于正态性检验

# ========================= Config =========================
BASE_DIR = "/data01/rong_dataset/Result/PhD/chapter3"

# Input files (only two periods for patent)
PATENT_FILES = {
    "1970-2010": os.path.join(BASE_DIR, "TS_patent_econ_cid_merged_Z_combined_1970_2010.csv"),
    "2011-2024": os.path.join(BASE_DIR, "TS_patent_econ_cid_merged_Z_combined_2011_2024.csv"),
}

# Output CSV for patent sensitivity
OUTPUT_PATENT = os.path.join(BASE_DIR, "Sensitivity_patent_periods.csv")

# Figure output for sensitivity (two lines)
OUTPUT_FIG_PNG = os.path.join(BASE_DIR, "Sensitivity_2lines.png")
OUTPUT_FIG_PDF = os.path.join(BASE_DIR, "Sensitivity_2lines.pdf")

# Figure output for distribution plots
DIST_PATHS = {
    "1970-2010": {
        "png": os.path.join(BASE_DIR, "Patent_1970-2010_Z_distribution.png"),
        "pdf": os.path.join(BASE_DIR, "Patent_1970-2010_Z_distribution.pdf"),
    },
    "2011-2024": {
        "png": os.path.join(BASE_DIR, "Patent_2011-2024_Z_distribution.png"),
        "pdf": os.path.join(BASE_DIR, "Patent_2011-2024_Z_distribution.pdf"),
    }
}

# Threshold sequence
thresholds = np.round(np.arange(0.5, 3.6, 0.1), 1)

# Fixed period order (only the two periods of interest)
PERIOD_ORDER = ["1970-2010", "2011-2024"]


# ========================= Helpers =========================

def load_z_table(path: str) -> pd.DataFrame:
    """
    Load Z-table for a single period: first column is 'Series', remaining are year columns (Z_{e,i,t}).
    Returns DataFrame; if file missing or empty, return empty DataFrame with only 'Series' column.
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found, skip: {path}")
        return pd.DataFrame(columns=["Series"])
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return pd.DataFrame(columns=["Series"])

    if df.shape[1] == 0:
        return pd.DataFrame(columns=["Series"])
    # Rename first column to 'Series' for consistency
    if df.columns[0] != "Series":
        df = df.rename(columns={df.columns[0]: "Series"})
    return df


def metric_for_threshold(df_period: pd.DataFrame, threshold: float) -> Tuple[int, float, float, int]:
    """
    For a single period Z-table and a given threshold, compute four metrics:
    Total_Count, Average_Count_Per_T, Median_Count_Per_T, Max_Count_Per_T
    - T denotes a time series (row)
    - NaN values are ignored.
    """
    if df_period.shape[1] <= 1:  # Only 'Series' column or empty
        return 0, 0.0, 0.0, 0

    z_cols = df_period.columns[1:]
    z_bool = df_period[z_cols] >= threshold
    z_mask = ~df_period[z_cols].isna()
    z_int = z_bool.astype(float).where(z_mask)  # True->1, False->0, NaN->NaN

    total_count = int(np.nansum(z_int.values))
    per_row_counts = np.nansum(z_int.values, axis=1)
    if per_row_counts.size == 0:
        return 0, 0.0, 0.0, 0

    avg_per_row = float(np.nanmean(per_row_counts))
    median_per_row = float(np.nanmedian(per_row_counts))
    max_per_row = int(np.nanmax(per_row_counts)) if np.isfinite(np.nanmax(per_row_counts)) else 0

    return total_count, avg_per_row, median_per_row, max_per_row


def analyze_type_periods(file_map: Dict[str, str], out_path: str) -> pd.DataFrame:
    """
    Perform sensitivity analysis for a given type (patent/paper) across periods.
    Output structure:
      Z_Threshold,
      Total_Count_<period1>, Total_Count_<period2>,
      Average_Count_Per_T_<period1>, Average_Count_Per_T_<period2>,
      Median_Count_Per_T_<period1>, Median_Count_Per_T_<period2>,
      Max_Count_Per_T_<period1>, Max_Count_Per_T_<period2>
    Also returns the DataFrame for later plotting.
    """
    dfs = {seg: load_z_table(path) for seg, path in file_map.items()}

    results = []
    for thr in thresholds:
        row = {"Z_Threshold": thr}
        totals, avgs, medians, maxs = [], [], [], []
        for seg in PERIOD_ORDER:
            dfp = dfs.get(seg, pd.DataFrame(columns=["Series"]))
            total_count, avg_t, median_t, max_t = metric_for_threshold(dfp, thr)
            totals.append(total_count)
            avgs.append(avg_t)
            medians.append(median_t)
            maxs.append(max_t)

        for i, seg in enumerate(PERIOD_ORDER):
            row[f"Total_Count_{seg}"] = totals[i]
        for i, seg in enumerate(PERIOD_ORDER):
            row[f"Average_Count_Per_T_{seg}"] = avgs[i]
        for i, seg in enumerate(PERIOD_ORDER):
            row[f"Median_Count_Per_T_{seg}"] = medians[i]
        for i, seg in enumerate(PERIOD_ORDER):
            row[f"Max_Count_Per_T_{seg}"] = maxs[i]

        results.append(row)

    # Arrange columns
    columns = ["Z_Threshold"]
    for prefix in ["Total_Count", "Average_Count_Per_T", "Median_Count_Per_T", "Max_Count_Per_T"]:
        for seg in PERIOD_ORDER:
            columns.append(f"{prefix}_{seg}")

    result_df = pd.DataFrame(results)[columns]
    result_df.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")
    return result_df


def plot_two_lines_totals(df_patent: pd.DataFrame,
                          save_png: str,
                          save_pdf: str) -> None:
    """
    Plot two lines for patent periods 1970-2010 and 2011-2024 using Total_Count.
    X-axis: Z_Threshold.
    """
    x = df_patent["Z_Threshold"].values

    # Extract Total_Count for the two periods
    patent_series = {seg: df_patent[f"Total_Count_{seg}"].values for seg in PERIOD_ORDER}

    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    # Line styles: solid with distinct markers
    markers = ['o', 's']
    colors = ['#1f77b4', '#ff7f0e']  # blue and orange

    lines = []
    for i, seg in enumerate(PERIOD_ORDER):
        ln, = ax.plot(
            x, patent_series[seg],
            linestyle='-',
            marker=markers[i],
            color=colors[i],
            linewidth=2.0,
            markersize=5.0,
            label=f"Patent {seg}",
        )
        lines.append(ln)

    # Vertical dashed line at Z=2 (common threshold)
    ax.axvline(x=2.0, linestyle='--', linewidth=1.2, color='gray')
    ax.text(2.0, ax.get_ylim()[1] * 0.98, "Z = 2", ha='right', va='top',
            rotation=90, fontsize=9, color='gray')

    # Labels and title
    ax.set_xlabel("Z threshold")
    ax.set_ylabel("Total outlier count")
    ax.set_xlim(x.min(), x.max())
    xticks = np.round(np.arange(x.min(), x.max() + 1e-9, 0.5), 1)
    ax.set_xticks(xticks)

    ax.grid(True, which='both', axis='both', alpha=0.25)

    # Legend
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    plt.title("Sensitivity to Z threshold")
    plt.tight_layout()

    # Save
    plt.savefig(save_png, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"[OK] Figure saved:\n - {save_png}\n - {save_pdf}")
    plt.close()


# ========================= Distribution and normality test =========================

def plot_z_distribution(df: pd.DataFrame, period_label: str, save_png: str, save_pdf: str) -> None:
    """
    Plot histogram of Z values from the Z-table for a given period.
    Mark the 5th and 95th percentiles.
    """
    if df.shape[1] <= 1 or df.empty:
        print(f"[WARN] No data for {period_label}, skip distribution plot.")
        return

    # Extract all Z values (excluding Series column), flatten, remove NaN
    z_cols = df.columns[1:]
    z_values = df[z_cols].values.flatten()
    z_values = z_values[~np.isnan(z_values)]

    if len(z_values) == 0:
        print(f"[WARN] No valid Z values for {period_label}, skip plot.")
        return

    # Compute 5th and 95th percentiles
    p5 = np.percentile(z_values, 5)
    p95 = np.percentile(z_values, 95)

    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    # Histogram
    n, bins, patches = ax.hist(z_values, bins=50, alpha=0.7, color='steelblue',
                                edgecolor='black', linewidth=0.5)

    # Vertical lines at percentiles (legend labels simplified to "5%" and "95%")
    ax.axvline(x=p5, linestyle='--', linewidth=1.5, color='green', label='5%')
    ax.axvline(x=p95, linestyle='--', linewidth=1.5, color='red', label='95%')

    # Add text labels near the top (show the actual percentile values)
    ylim = ax.get_ylim()
    ax.text(p5, ylim[1] * 0.9, f'{p5:.2f}', ha='center', va='bottom', fontsize=10, color='green')
    ax.text(p95, ylim[1] * 0.9, f'{p95:.2f}', ha='center', va='bottom', fontsize=10, color='red')

    # Labels and title
    ax.set_xlabel("Z value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Z values - Patent {period_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(save_png, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"[OK] Distribution plot saved for {period_label}:\n - {save_png}\n - {save_pdf}")
    plt.close()


def normality_test(z_values: np.ndarray, period_label: str) -> None:
    """
    Perform D'Agostino's K^2 normality test (normaltest) and print results.
    H0: data comes from a normal distribution.
    If p-value > 0.05, cannot reject H0 (data may be normal).
    """
    if len(z_values) < 20:
        print(f"[WARN] {period_label}: Too few samples ({len(z_values)}) for reliable normality test.")
        return

    # Remove NaNs (already done in calling code)
    statistic, p_value = stats.normaltest(z_values)
    print(f"\n--- Normality test for Patent {period_label} ---")
    print(f"  Number of valid Z values: {len(z_values)}")
    print(f"  D'Agostino's K2 statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4e}")
    if p_value > 0.05:
        print("  -> Data may follow a normal distribution (fail to reject H0 at alpha=0.05).")
    else:
        print("  -> Data does NOT follow a normal distribution (reject H0 at alpha=0.05).")
    print()


# ========================= Run =========================
if __name__ == "__main__":
    # ---- 1. Generate sensitivity table for patent (only two periods) ----
    df_patent = analyze_type_periods(PATENT_FILES, OUTPUT_PATENT)

    # ---- 2. Plot the two lines (Total_Count) ----
    plot_two_lines_totals(
        df_patent=df_patent,
        save_png=OUTPUT_FIG_PNG,
        save_pdf=OUTPUT_FIG_PDF
    )

    # ---- 3. Plot distribution of Z values for each period and perform normality test ----
    # Reload data for each period (or reuse from inside analyze_type_periods if needed)
    df_1970_2010 = load_z_table(PATENT_FILES["1970-2010"])
    df_2011_2024 = load_z_table(PATENT_FILES["2011-2024"])

    # Process 1970-2010
    plot_z_distribution(df_1970_2010, "1970-2010",
                        DIST_PATHS["1970-2010"]["png"],
                        DIST_PATHS["1970-2010"]["pdf"])
    # Extract Z values for test
    z_vals_1970 = df_1970_2010[df_1970_2010.columns[1:]].values.flatten()
    z_vals_1970 = z_vals_1970[~np.isnan(z_vals_1970)]
    normality_test(z_vals_1970, "1970-2010")

    # Process 2011-2024
    plot_z_distribution(df_2011_2024, "2011-2024",
                        DIST_PATHS["2011-2024"]["png"],
                        DIST_PATHS["2011-2024"]["pdf"])
    z_vals_2011 = df_2011_2024[df_2011_2024.columns[1:]].values.flatten()
    z_vals_2011 = z_vals_2011[~np.isnan(z_vals_2011)]
    normality_test(z_vals_2011, "2011-2024")

    # Clean up memory
    del df_patent, df_1970_2010, df_2011_2024
    gc.collect()