#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined RF + SHAP figure for pj08
==================================

This script builds a combined figure with two rows (periods):

Row 1: Y1_1970_2010
Row 2: Y2_2011_2024

Each row has three subplots:
  [left]  SHAP mean(|value|) bar plot (horizontal, blue, x-axis right-to-left)
  [mid]   Shared feature-name axis (wrapped labels, ticks on both left and right),
          labels are blue if the feature appears in BOTH periods, black otherwise
  [right] SHAP beeswarm (official SHAP style)

Inputs per period (in RESULT_DIR):
  - SHAP_<period>_mean_abs.csv
      columns: rank, feature, series_code, mean_abs_shap
  - SHAP_<period>_values.csv.gz
      columns: IPCR, Country, y_true, expected_value, shap::<series_code>...
  - RF_<period>_design.csv
      columns: IPCR, Country, Country_WDI, Pioneer_WDI, Pioneer Time,
               Z>=2 Time, y, <feature1>, <feature2>, ...

Outputs:
  - COMBINED_RF_SHAP.pdf
  - COMBINED_RF_SHAP.png

Notes:
- Uses old shap API (no "ax=" argument). We switch current axes via plt.sca().
- TOP_K controls how many top features are plotted (default 15).
- The vertical ordering of bar, labels and beeswarm is forced to be consistent
  by reading the y-ticks from the beeswarm subplot.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ===================== CONFIG =====================

RESULT_DIR = "/data01/rong_dataset/Result/pj08/Regression_result_RF_SHAP"
PERIOD_TAGS = [
    "Y1_1970_2010",
    "Y2_2011_2024",
]

# Fewer features to reduce overlap and effectively increase vertical spacing
TOP_K = 20

# Global font and resolution settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# ===================== HELPERS =====================

def pjoin(*a):
    return os.path.join(*a)

def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            return None
    else:
        print(f"[WARN] File not found: {path}")
        return None

def normalize_relaxed(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

def wrap_two_lines_labels(labels):
    """
    Wrap feature labels into at most two lines to control width.
    """
    import textwrap
    # A relatively large width to avoid over-wrapping
    wrap_width = 45

    wrapped = []
    for s in labels:
        if not isinstance(s, str):
            s = str(s)
        parts = textwrap.wrap(s, width=wrap_width)
        if len(parts) <= 1:
            wrapped.append(s)
        elif len(parts) == 2:
            wrapped.append(parts[0] + "\n" + parts[1])
        else:
            second = " ".join(parts[1:])
            second_short = textwrap.shorten(second, width=wrap_width, placeholder="...")
            wrapped.append(parts[0] + "\n" + second_short)
    return wrapped

def detect_common_features(df1, df2):
    """
    Determine which features are common between two periods.
    Preference is given to matching by series_code (if present and non-empty),
    otherwise falls back to a relaxed normalization of the feature name.
    Returns two boolean lists (mask1, mask2) aligned with df1 and df2 rows.
    """
    sc1 = []
    nm1 = []
    for _, row in df1.iterrows():
        sc = str(row.get("series_code", "")).strip()
        nm = normalize_relaxed(row.get("feature", ""))
        sc1.append(sc)
        nm1.append(nm)

    sc2 = []
    nm2 = []
    for _, row in df2.iterrows():
        sc = str(row.get("series_code", "")).strip()
        nm = normalize_relaxed(row.get("feature", ""))
        sc2.append(sc)
        nm2.append(nm)

    set_sc1 = {s for s in sc1 if s}
    set_sc2 = {s for s in sc2 if s}
    common_sc = set_sc1 & set_sc2

    set_nm1 = {n for n in nm1 if n}
    set_nm2 = {n for n in nm2 if n}
    common_nm = set_nm1 & set_nm2

    mask1 = []
    for s, n in zip(sc1, nm1):
        mask1.append((s in common_sc) or (n in common_nm))

    mask2 = []
    for s, n in zip(sc2, nm2):
        mask2.append((s in common_sc) or (n in common_nm))

    return mask1, mask2

def load_mean_abs(period):
    """
    Load mean(|SHAP|) CSV for a given period and keep TOP_K features.
    """
    path = pjoin(RESULT_DIR, f"SHAP_{period}_mean_abs.csv")
    df = safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["feature", "series_code", "mean_abs_shap"])
    d = df.copy()
    if "mean_abs_shap" not in d.columns:
        raise ValueError(f"{path} missing column 'mean_abs_shap'. Got: {list(d.columns)}")
    d["mean_abs_shap"] = pd.to_numeric(d["mean_abs_shap"], errors="coerce")
    d = d.sort_values("mean_abs_shap", ascending=False)
    d = d.head(TOP_K).reset_index(drop=True)
    if "series_code" not in d.columns:
        d["series_code"] = ""
    return d[["feature", "series_code", "mean_abs_shap"]].copy()

def load_shap_and_X(period, selected_mean_abs_df):
    """
    For a given period and a DataFrame of selected mean_abs_shap rows
    (feature, series_code, mean_abs_shap), load:

    - shap_vals: numpy array (n_samples, K)
    - X_vals:    numpy array (n_samples, K)  (design matrix for the same features)
    - labels:    list of feature names (length K)
    """
    shap_path = pjoin(RESULT_DIR, f"SHAP_{period}_values.csv.gz")
    design_path = pjoin(RESULT_DIR, f"RF_{period}_design.csv")

    shap_df = safe_read_csv(shap_path, compression="gzip")
    design_df = safe_read_csv(design_path)

    if shap_df is None or shap_df.empty or design_df is None or design_df.empty:
        print(f"[WARN] Missing SHAP or design data for {period}.")
        return None, None, []

    labels = selected_mean_abs_df["feature"].astype(str).tolist()
    series_codes = selected_mean_abs_df["series_code"].astype(str).tolist()

    # Build SHAP value matrix: columns shap::<series_code>
    shap_cols = []
    for sc in series_codes:
        col = f"shap::{sc}"
        if col not in shap_df.columns:
            print(f"[WARN] Column {col} not found in {shap_path}. Filling zeros.")
        shap_cols.append(col)

    shap_matrix = np.zeros((len(shap_df), len(shap_cols)), dtype=float)
    for j, col in enumerate(shap_cols):
        if col in shap_df.columns:
            shap_matrix[:, j] = (
                pd.to_numeric(shap_df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy()
            )
        else:
            shap_matrix[:, j] = 0.0

    # Build X matrix: columns by feature label
    missing_features = [lbl for lbl in labels if lbl not in design_df.columns]
    if missing_features:
        print(f"[WARN] Some feature labels not found in design CSV for {period}: {missing_features}")
    X_matrix = np.zeros_like(shap_matrix)
    for j, lbl in enumerate(labels):
        if lbl in design_df.columns:
            X_matrix[:, j] = (
                pd.to_numeric(design_df[lbl], errors="coerce")
                .fillna(0.0)
                .to_numpy()
            )
        else:
            X_matrix[:, j] = 0.0

    return shap_matrix, X_matrix, labels

# ===================== MAIN PLOT FUNCTION =====================

def plot_combined_rf_shap():
    if not HAS_SHAP:
        print("[ERROR] shap is not installed. Please install it before plotting.")
        return

    # Load mean_abs for both periods
    df_mean_1 = load_mean_abs(PERIOD_TAGS[0])
    df_mean_2 = load_mean_abs(PERIOD_TAGS[1])

    if df_mean_1.empty and df_mean_2.empty:
        print("[ERROR] No mean_abs_shap data found for both periods.")
        return

    # Detect common features (for label coloring)
    common_mask_1, common_mask_2 = detect_common_features(df_mean_1, df_mean_2)

    # Load SHAP values and X for both periods
    shap_1, X_1, labels_1 = load_shap_and_X(PERIOD_TAGS[0], df_mean_1)
    shap_2, X_2, labels_2 = load_shap_and_X(PERIOD_TAGS[1], df_mean_2)

    if shap_1 is None or shap_2 is None:
        print("[ERROR] Missing SHAP/X matrices. Abort combined plot.")
        return

    vals_1 = df_mean_1["mean_abs_shap"].to_numpy(dtype=float)
    vals_2 = df_mean_2["mean_abs_shap"].to_numpy(dtype=float)

    wrapped_1 = wrap_two_lines_labels(labels_1)
    wrapped_2 = wrap_two_lines_labels(labels_2)
    
        
    #1970-2010
    order1 = np.argsort(vals_1)  
    vals_1 = vals_1[order1]
    labels_1 = [labels_1[i] for i in order1]
    wrapped_1 = [wrapped_1[i] for i in order1]
    common_mask_1 = [common_mask_1[i] for i in order1]
    shap_1 = shap_1[:, order1]
    X_1 = X_1[:, order1]

    #2011-2024
    order2 = np.argsort(vals_2)
    vals_2 = vals_2[order2]
    labels_2 = [labels_2[i] for i in order2]
    wrapped_2 = [wrapped_2[i] for i in order2]
    common_mask_2 = [common_mask_2[i] for i in order2]
    shap_2 = shap_2[:, order2]
    X_2 = X_2[:, order2]


    # Larger figure and adjusted grid to give more room
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.4, 2.0, 3.1],
        height_ratios=[1.0, 1.0]
    )

    # ===================== ROW 1: Y1_1970_2010 =====================

    # Create axes
    ax_bar_1 = fig.add_subplot(gs[0, 0])
    ax_lab_1 = fig.add_subplot(gs[0, 1])
    ax_bee_1 = fig.add_subplot(gs[0, 2])

    # Bar plot (we will set y-limits/ticks after beeswarm is drawn)
    K1 = len(labels_1)
    y1_pos = np.arange(K1)
    ax_bar_1.barh(y1_pos, vals_1, color="tab:blue", alpha=0.85)
    ax_bar_1.invert_xaxis()
    ax_bar_1.set_yticks(y1_pos)
    ax_bar_1.set_yticklabels([])
    ax_bar_1.set_xlabel("mean(|SHAP value|)")
    ax_bar_1.set_title("1970-2010")
    ax_bar_1.tick_params(axis="y", left=False, right=False)

    # Beeswarm on its own axis (SHAP official style)
    plt.sca(ax_bee_1)
    shap.summary_plot(
        shap_1,
        X_1,
        feature_names=labels_1,
        show=False,
        max_display=TOP_K,
        color_bar=True,
        plot_type="dot",
        sort=False
    )

    ax_bee_1.yaxis.tick_left()
    ax_bee_1.yaxis.set_label_position("left")
    ax_bee_1.tick_params(axis="y", right=False)

    ax_bee_1.set_xlabel(
        "SHAP value",
        fontsize=plt.rcParams["axes.labelsize"]
    )
    ax_bee_1.tick_params(axis="x", labelsize=plt.rcParams["axes.labelsize"])


    ytick_pos_1 = ax_bee_1.get_yticks()
    ylim1 = ax_bee_1.get_ylim()

    ax_bee_1.set_yticks(ytick_pos_1)
    tick_texts_1_bee = ax_bee_1.set_yticklabels(wrapped_1)

    for txt, is_common in zip(tick_texts_1_bee, common_mask_1):
        txt.set_color("blue" if is_common else "black")

    for txt in tick_texts_1_bee:
        txt.set_fontsize(plt.rcParams["ytick.labelsize"])

    ax_bee_1.set_ylim(ylim1)
    ax_bee_1.set_title("1970-2010")




    # Center label axis uses the same y positions as beeswarm
    ax_lab_1.set_xlim(0.0, 1.0)
    ax_lab_1.set_ylim(ylim1)
    ax_lab_1.set_xticks([])
    ax_lab_1.set_xticklabels([])
    ax_lab_1.set_yticks(ytick_pos_1)
    ax_lab_1.set_yticklabels([])
    # Color labels (blue if common to both periods, black otherwise)
    #for txt, is_common in zip(tick_texts_1, common_mask_1):
    #    txt.set_color("blue" if is_common else "black")
    # Ticks and labels on both left and right
    ax_lab_1.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True
    )
    ax_lab_1.spines["top"].set_visible(False)
    ax_lab_1.spines["bottom"].set_visible(False)

    # Make the bar share the same vertical positions
    ax_bar_1.set_ylim(ylim1)
    ax_bar_1.set_yticks(ytick_pos_1)
    ax_bar_1.set_yticklabels([])

    # ===================== ROW 2: Y2_2011_2024 =====================

    ax_bar_2 = fig.add_subplot(gs[1, 0])
    ax_lab_2 = fig.add_subplot(gs[1, 1])
    ax_bee_2 = fig.add_subplot(gs[1, 2])

    K2 = len(labels_2)
    y2_pos = np.arange(K2)
    ax_bar_2.barh(y2_pos, vals_2, color="tab:blue", alpha=0.85)
    ax_bar_2.invert_xaxis()
    ax_bar_2.set_yticks(y2_pos)
    ax_bar_2.set_yticklabels([])
    ax_bar_2.set_xlabel("mean(|SHAP value|)")
    ax_bar_2.set_title("2011-2024")
    ax_bar_2.tick_params(axis="y", left=False, right=False)

    plt.sca(ax_bee_2)
    shap.summary_plot(
        shap_2,
        X_2,
        feature_names=labels_2,
        show=False,
        max_display=TOP_K,
        color_bar=True,
        plot_type="dot",
        sort=False
    )

    ax_bee_2.yaxis.tick_left()
    ax_bee_2.yaxis.set_label_position("left")
    ax_bee_2.tick_params(axis="y", right=False)

    ax_bee_2.set_xlabel(
        "SHAP value",
        fontsize=plt.rcParams["axes.labelsize"]
    )
    ax_bee_2.tick_params(axis="x", labelsize=plt.rcParams["axes.labelsize"])



    ytick_pos_2 = ax_bee_2.get_yticks()
    ylim2 = ax_bee_2.get_ylim()

    ax_bee_2.set_yticks(ytick_pos_2)
    tick_texts_2_bee = ax_bee_2.set_yticklabels(wrapped_2)

    for txt, is_common in zip(tick_texts_2_bee, common_mask_2):
        txt.set_color("blue" if is_common else "black")

    for txt in tick_texts_2_bee:
        txt.set_fontsize(plt.rcParams["ytick.labelsize"])

    ax_bee_2.set_ylim(ylim2)
    ax_bee_2.set_title("2011-2024")


    ax_lab_2.set_xlim(0.0, 1.0)
    ax_lab_2.set_ylim(ylim2)
    ax_lab_2.set_xticks([])
    ax_lab_2.set_xticklabels([])
    ax_lab_2.set_yticks(ytick_pos_2)
    ax_lab_2.set_yticklabels([])

    #for txt, is_common in zip(tick_texts_2, common_mask_2):
    #    txt.set_color("blue" if is_common else "black")
    ax_lab_2.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True
    )
    ax_lab_2.spines["top"].set_visible(False)
    ax_lab_2.spines["bottom"].set_visible(False)

    ax_bar_2.set_ylim(ylim2)
    ax_bar_2.set_yticks(ytick_pos_2)
    ax_bar_2.set_yticklabels([])

    import matplotlib as mpl
    for ax in fig.axes:

        if isinstance(ax, mpl.axes.Axes) and ax.get_ylabel() == "Feature value":
            ax.tick_params(labelsize=plt.rcParams["axes.labelsize"])  # High / Low
            ax.set_ylabel(ax.get_ylabel(), fontsize=plt.rcParams["axes.labelsize"])  # "Feature value"

    # Global layout (do not call fig.tight_layout to avoid colorbar conflicts)
    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.96,
        bottom=0.04,
        wspace=0.05,
        hspace=0.18
    )
    

    out_pdf = pjoin(RESULT_DIR, "COMBINED_RF_SHAP.pdf")
    out_png = pjoin(RESULT_DIR, "COMBINED_RF_SHAP.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=400)
    fig.savefig(out_png, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"[OK] Saved combined figure to:\n  {out_pdf}\n  {out_png}")

# ===================== MAIN =====================

def main():
    if not os.path.exists(RESULT_DIR):
        print(f"[ERROR] RESULT_DIR not found: {RESULT_DIR}")
        return
    plot_combined_rf_shap()

if __name__ == "__main__":
    main()
