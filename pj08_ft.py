#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================== Paths ==================
BASE_DIR = "/data01/rong_dataset/Result/pj08"

# segmented inputs (merged horizontally on Country, IPCR)
Z_SEG_FILES = [
    os.path.join(BASE_DIR, "WWP_1860-1969_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_1970-2010_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_2011-2024_Z_c_i_t.csv"),
]

OUT_FIRST_CSV = os.path.join(BASE_DIR, "WWP_First_g2.csv")
FIG_DIR = os.path.join(BASE_DIR, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)

# legacy single-figure paths (不再单独输出，但保留变量)
OUT_COMBINED_PDF = os.path.join(FIG_DIR, "Ft_combined.pdf")
OUT_REGIONAL_PDF = os.path.join(FIG_DIR, "Ft_CN_US_JP.pdf")

# New: combined two-panels in one PDF
OUT_DUAL_PDF = os.path.join(FIG_DIR, "Ft_combined_dual.pdf")

# ================== Periods (disjoint; per-period first-year) ==================
PERIODS = [
    ("1860-1969", 1860, 1969),
    ("1970-2010", 1970, 2010),
    ("2011-2024", 2011, 2024),
]

# ================== Styles & names for regional figure ==================
country_styles = {
    "US": {"color": "blue", "linestyle": "--"},
    "CN": {"color": "red", "linestyle": "-"},
    "JP": {"color": "green", "linestyle": (0, (1, 1))},
}
country_names = {"US": "United States", "CN": "China", "JP": "Japan"}

# ================== Helpers ==================
def parse_year_columns(df: pd.DataFrame) -> list:
    """Detect 4-digit year columns (string or float-like 'YYYY.0'), return as original labels."""
    year_cols = []
    for col in df.columns:
        s = str(col).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.isdigit() and len(s) == 4:
            year_cols.append(col)
    return year_cols

def normalize_year_label(col) -> str:
    """Canonical 4-digit year string for a column label."""
    s = str(col).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def to_sorted_year_array(year_cols: list) -> np.ndarray:
    """Normalize to 4-digit strings then sort and return int array."""
    years = sorted([int(normalize_year_label(c)) for c in year_cols])
    return np.array(years, dtype=int)

def find_first_ge_threshold_in_period(row_vals: pd.Series,
                                      years_sorted: np.ndarray,
                                      y0: int, y1: int,
                                      threshold: float = 2.0):
    """
    Given a row's Z values aligned to years_sorted, return the first year in [y0, y1]
    where Z >= threshold. If none, return None.
    """
    z = pd.to_numeric(row_vals, errors="coerce").to_numpy(dtype=float)
    within = (years_sorted >= y0) & (years_sorted <= y1)
    if not within.any():
        return None
    z_seg = z[within]
    yrs_seg = years_sorted[within]
    for idx, v in enumerate(z_seg):
        if not math.isnan(v) and v >= threshold:
            return int(yrs_seg[idx])
    return None

def linear_fit(x: np.ndarray, y: np.ndarray):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)

def _add_shaded_band(ax, x_years, y_vals,
                     start, end,
                     label, label_color="black",
                     alpha=0.12,
                     row=0,
                     top_padding_ratio=0.16,
                     row_gap_ratio=0.04,
                     x_offset=0.0):
    """
    Draw a vertical shaded band and place its label near the top.
    """
    if len(x_years) == 0:
        return
    x_min = int(np.nanmin(x_years))
    x_max = int(np.nanmax(x_years))
    band_start = max(start, x_min) if start is not None else x_min
    band_end = min(end, x_max) if end is not None else x_max
    if band_end <= band_start:
        return

    ax.axvspan(band_start, band_end, color="gray", alpha=alpha, zorder=0)

    y_min = float(np.nanmin(y_vals)) if len(y_vals) else 0.0
    y_max = float(np.nanmax(y_vals)) if len(y_vals) else 1.0
    y_range = max(1e-9, y_max - y_min)

    y_top_label_base = y_max + top_padding_ratio * y_range
    y_text = y_top_label_base - (row * row_gap_ratio * y_range)
    x_text = (band_start + band_end) / 2.0 + x_offset

    ax.text(x_text, y_text, label,
            color=label_color, fontsize=9,
            ha="center", va="top")

def _add_all_bands(ax, years_all, y_all):
    """Add the standard shaded bands to an axis."""
    # Shaded bands & labels
    alpha_deep = 0.18
    alpha_light = 0.10
    RED_ROW = 0
    BLACK_ROW = 4  # BLACK lower than RED by 4 rows

    # 2nd IR (向右偏移，避免与左上角图例重叠)
    _add_shaded_band(ax, years_all, y_all,
                     start=1860, end=1914,
                     label="1860-1914\n2nd IR",
                     label_color="red", alpha=alpha_light,
                     row=RED_ROW,
                     x_offset=15)

    # WW I
    _add_shaded_band(ax, years_all, y_all,
                     start=1914, end=1918,
                     label="1914-1918 \nWW I",
                     label_color="black", alpha=alpha_deep,
                     row=BLACK_ROW)

    # WW II
    _add_shaded_band(ax, years_all, y_all,
                     start=1931, end=1945,
                     label="1931-1945 \nWW II",
                     label_color="black", alpha=alpha_light,
                     row=BLACK_ROW)

    # 3rd IR
    _add_shaded_band(ax, years_all, y_all,
                     start=1970, end=2010,
                     label="1970-2010\n3rd IR",
                     label_color="red", alpha=alpha_deep,
                     row=RED_ROW)

    # 4th IR
    _add_shaded_band(ax, years_all, y_all,
                     start=2011, end=2024,
                     label="2011-present\n4th IR",
                     label_color="red", alpha=alpha_light,
                     row=RED_ROW)

def draw_single_panel(ax, years_all: np.ndarray, y_all: np.ndarray):
    """Draw the global single-line panel on a given Axes."""
    ax.plot(years_all, y_all, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency")
    ax.set_xlim([years_all.min(), years_all.max()])

    # y-range
    y_min = float(np.nanmin(y_all)) if len(y_all) else 0.0
    y_max = float(np.nanmax(y_all)) if len(y_all) else 1.0
    y_range = max(1e-9, y_max - y_min)

    _add_all_bands(ax, years_all, y_all)

    top_extra = 0.18
    ax.set_ylim(y_min - 0.02 * y_range, y_max + top_extra * y_range)

def draw_multi_country_panel(ax, years_all: np.ndarray, series_by_country: dict):
    """Draw the CN/US/JP multi-line panel on a given Axes."""
    ymax_candidate = []
    for ccode, yvals in series_by_country.items():
        style = country_styles.get(ccode, {"linestyle": "-", "color": "black"})
        label = country_names.get(ccode, ccode)
        ax.plot(years_all, yvals, linewidth=2,
                linestyle=style.get("linestyle", "-"),
                color=style.get("color", "black"),
                label=label)
        ymax_candidate.append(np.nanmax(yvals) if len(yvals) else 0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency")
    ax.set_xlim([years_all.min(), years_all.max()])

    # y-range for padding and bands
    if ymax_candidate:
        y_min = 0.0
        y_max = float(np.nanmax(ymax_candidate))
        y_range = max(1e-9, y_max - y_min)
    else:
        y_min, y_max, y_range = 0.0, 1.0, 1.0

    # Envelope for band labels
    if series_by_country:
        stacked = np.vstack([series_by_country[k] for k in series_by_country])
        y_env = np.nanmax(stacked, axis=0)
    else:
        y_env = np.zeros_like(years_all)

    _add_all_bands(ax, years_all, y_env)

    top_extra = 0.18
    ax.set_ylim(y_min - 0.02 * y_range, y_max + top_extra * y_range)

    ax.legend(loc="upper left", frameon=False)

def save_combined_two_panels(years_all: np.ndarray,
                             y_global: np.ndarray,
                             series_by_country: dict):
    """
    Create one PDF with two stacked panels:
    (a) global single-line; (b) multi-country CN/US/JP.
    """
    # 尽量保持单图的宽高比例，这里高度翻倍以容纳两幅图
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 11.6), sharex=False
    )

    # 上图
    draw_single_panel(ax_top, years_all, y_global)
    ax_top.set_title("(a) Temporal Distribution of PSO", pad=10)

    # 下图
    draw_multi_country_panel(ax_bottom, years_all, series_by_country)
    ax_bottom.set_title("(b) Temporal Distribution of PSO of US, JP, and CN", pad=10)

    plt.tight_layout()
    plt.savefig(OUT_DUAL_PDF, bbox_inches="tight")
    plt.close()

def compute_frequency_from_z(z_only: pd.DataFrame,
                             years_sorted: np.ndarray,
                             periods=PERIODS,
                             threshold: float = 2.0):
    """
    Given a Z-only table (columns: Country, IPCR, <years...>),
    compute yearly frequency of first-year >= threshold across periods,
    and return (years_axis, freq_cols_dict, freq_all).
    """
    canon_years = [str(y) for y in years_sorted]
    # Ensure all year columns exist
    for y in canon_years:
        if y not in z_only.columns:
            z_only[y] = np.nan
    z_vals = z_only[canon_years]

    # Collect Ft per period across all rows
    ft_years_by_period = {lbl: [] for (lbl, _, _) in periods}
    for i in range(len(z_only)):
        row = z_vals.iloc[i]
        for lbl, y0, y1 in periods:
            ft = find_first_ge_threshold_in_period(row, years_sorted, y0, y1, threshold=threshold)
            if ft is not None:
                ft_years_by_period[lbl].append(ft)

    # Build yearly frequency per period, then total
    min_year = int(years_sorted.min())
    max_year = int(years_sorted.max())
    years_axis = np.arange(min_year, max_year + 1, dtype=int)

    freq_cols = {}
    for lbl, y0, y1 in periods:
        counts = pd.Series(ft_years_by_period[lbl]).value_counts().sort_index() if len(ft_years_by_period[lbl]) else pd.Series(dtype=int)
        freq = pd.Series(0, index=years_axis, dtype=int)
        if not counts.empty:
            idx = counts.index[(counts.index >= y0) & (counts.index <= y1)]
            freq.loc[idx] = counts.loc[idx].astype(int)
        freq_cols[f"Frequency_{lbl}"] = freq.values

    freq_all = np.zeros_like(years_axis, dtype=int)
    for col in freq_cols.values():
        freq_all += col

    return years_axis, freq_cols, freq_all

# ================== Main ==================
def main():
    # ---- Load and horizontally combine segmented Z tables ----
    base_df = None
    for path in Z_SEG_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Z segment not found: {path}")
        seg = pd.read_csv(path, low_memory=False)
        year_cols = parse_year_columns(seg)
        seg = seg[["Country", "IPCR"] + year_cols].copy()
        # normalize year cols to numeric dtype downstream
        for c in year_cols:
            seg[c] = pd.to_numeric(seg[c], errors="coerce")
        if base_df is None:
            base_df = seg
        else:
            # only bring new year columns to avoid duplicates
            new_years = [c for c in year_cols if c not in base_df.columns]
            base_df = pd.merge(base_df,
                               seg[["Country", "IPCR"] + new_years],
                               on=["Country", "IPCR"],
                               how="outer")

    if base_df is None:
        raise RuntimeError("No Z data loaded.")

    # Full set of year columns & sorted year array
    all_year_cols = parse_year_columns(base_df)
    years_sorted = to_sorted_year_array(all_year_cols)

    # ---- Global: compute frequency over ALL rows ----
    canon_years = [str(y) for y in years_sorted]
    z_only_global = base_df.reindex(columns=["Country", "IPCR"] + canon_years).copy()
    years_axis, freq_cols_global, freq_all_global = compute_frequency_from_z(
        z_only=z_only_global, years_sorted=years_sorted, periods=PERIODS, threshold=2.0
    )

    # ---- Fit linear trend on total frequency (global) ----
    x = years_axis.astype(float)
    y = freq_all_global.astype(float)
    slope, intercept = linear_fit(x, y)
    yhat = slope * x + intercept
    residuals = y - yhat

    # ---- Output CSV (global) ----
    out_df = pd.DataFrame({"Year": years_axis})
    for k, v in freq_cols_global.items():
        out_df[k] = v
    out_df["Frequency_All"] = freq_all_global
    out_df["Predicted_All"] = yhat
    out_df["Residual_All"] = residuals
    out_df.to_csv(OUT_FIRST_CSV, index=False)

    # ---- Regional series: CN / US / JP ----
    series_by_country = {}
    for ccode in ["US", "CN", "JP"]:
        subset = z_only_global[z_only_global["Country"] == ccode].copy()
        _, _, freq_all_country = compute_frequency_from_z(
            z_only=subset, years_sorted=years_sorted, periods=PERIODS, threshold=2.0
        )
        series_by_country[ccode] = freq_all_country

    # ---- Combined PDF with two stacked panels ----
    save_combined_two_panels(
        years_all=years_axis,
        y_global=freq_all_global,
        series_by_country=series_by_country
    )

    print(f"Saved frequency CSV with per-period and total: {OUT_FIRST_CSV}")
    print(f"Saved combined two-panel figure: {OUT_DUAL_PDF}")
    print(f"Slope (All): {slope:.6f}, Intercept (All): {intercept:.6f}")

if __name__ == "__main__":
    main()
