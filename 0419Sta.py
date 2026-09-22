#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized year gap statistics and histograms for three periods.
Uses multiprocessing to parallelize IPCR group computations.
Suitable for large datasets on high?memory, multi?core servers.

Outputs:
    - year_gap_stats.csv: max, min, mean, median, std for each period.
    - Fig/<period>_year_gap_hist.pdf: histogram of year gaps for each period.
"""

import os
import re
import math
from typing import Dict, List, Optional, Tuple, Iterable
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

# -------------------- Configuration --------------------
BASE_DIR = "/data01/rong_dataset/Result/pj08/"

Z_FILES = {
    "1870-1969": os.path.join(BASE_DIR, "WWP_1860-1969_Z_c_i_t.csv"),
    "1970-2010": os.path.join(BASE_DIR, "WWP_1970-2010_Z_c_i_t.csv"),
    "2011-2024": os.path.join(BASE_DIR, "WWP_2011-2024_Z_c_i_t.csv"),
}

OUT_STATS = os.path.join(BASE_DIR, "year_gap_stats.csv")
FIG_DIR = os.path.join(BASE_DIR, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)

# Transition rules
TRANSITION: Dict[str, int] = {"SU": 1991, "CS": 1993, "YU": 2003}
PREDS = set(TRANSITION.keys())

PERIODS = ["1870-1969", "1970-2010", "2011-2024"]

# Multiprocessing settings
N_WORKERS = min(cpu_count(), 64)  # leave some headroom
CHUNK_SIZE = 200                  # IPCR groups per chunk

# -------------------- Robust CSV loader with memory optimisation --------------------
def read_csv_safely(path: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unexpected CSV read error.")

def load_z_period(path: str) -> pd.DataFrame:
    """Load a period Z file with optimal dtypes to save memory."""
    # Read only first few rows to detect year columns dynamically
    sample = pd.read_csv(path, nrows=5)
    year_candidates = [c for c in sample.columns if re.match(r"^\d{4}$", str(c))]
    if not year_candidates:
        raise ValueError(f"No year columns found in {path}")
    # Define dtypes: Country and IPCR as string, years as float32 (saves memory)
    dtype_dict = {col: "float32" for col in year_candidates}
    dtype_dict["Country"] = "string"
    dtype_dict["IPCR"] = "string"
    df = read_csv_safely(path, dtype=dtype_dict, low_memory=False)
    # Ensure year columns are float32 (they already are, but force conversion)
    for col in year_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    return df, year_candidates

# -------------------- Transition de-duplication --------------------
def extract_predecessor(country_str: str) -> Optional[str]:
    if not isinstance(country_str, str):
        return None
    if country_str in PREDS:
        return country_str
    if "/" in country_str:
        parts = country_str.split("/")
        cand = [p for p in parts if p in PREDS]
        if len(cand) == 1:
            return cand[0]
    return None

def apply_transition_deduplication(df: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
    """
    In?place modification (or copy) that sets pre?transition Z values to NaN
    for non?primary rows in each (pred, IPCR) cluster.
    Returns a modified DataFrame (copy to avoid side effects).
    """
    if "Country" not in df.columns or "IPCR" not in df.columns:
        return df.copy()

    df_adj = df.copy()
    pred_series = df_adj["Country"].apply(extract_predecessor)
    trans_mask = pred_series.notna()
    if not trans_mask.any():
        return df_adj

    trans_table = pd.DataFrame({
        "row_idx": df_adj.index,
        "Country": df_adj["Country"],
        "IPCR": df_adj["IPCR"],
        "pred": pred_series
    })
    trans_table = trans_table[trans_mask].copy()

    year_ints = np.array([int(y) for y in year_cols], dtype=int)
    pre_cols_by_pred: Dict[str, List[str]] = {}
    for p, tyear in TRANSITION.items():
        pre_cols_by_pred[p] = [str(y) for y in year_ints if y < tyear]

    # Determine primary row per (pred, IPCR)
    primary_idx_by_cluster: Dict[Tuple[str, str], int] = {}
    cluster_rows: Dict[Tuple[str, str], List[int]] = {}
    for (pred, ipcr), g in trans_table.groupby(["pred", "IPCR"], dropna=False):
        idxs = g["row_idx"].tolist()
        countries = df_adj.loc[idxs, "Country"].tolist()
        # primary = predecessor country if exists, else first in sorted order
        pri = None
        for i, c in zip(idxs, countries):
            if c == pred:
                pri = i
                break
        if pri is None:
            pairs = sorted(zip(countries, idxs), key=lambda x: str(x[0]))
            pri = pairs[0][1]
        primary_idx_by_cluster[(pred, ipcr)] = pri
        cluster_rows[(pred, ipcr)] = idxs

    # Apply NaN to non?primary rows for pre?transition years
    for (pred, ipcr), idxs in cluster_rows.items():
        pre_cols = pre_cols_by_pred.get(pred, [])
        if not pre_cols:
            continue
        primary_idx = primary_idx_by_cluster[(pred, ipcr)]
        non_primary = [i for i in idxs if i != primary_idx]
        if non_primary:
            df_adj.loc[non_primary, pre_cols] = np.nan

    return df_adj

# -------------------- Per?IPCR gap computation (for one group) --------------------
def first_year_ge2(series: pd.Series, year_cols: List[str]) -> Optional[int]:
    """Return first year (as int) where Z >= 2, or None."""
    s = series[year_cols].astype(float)
    mask = s >= 2.0
    if not mask.any():
        return None
    return int(s.index[mask][0])

def pioneer_and_gaps_for_ipcr(group_df: pd.DataFrame, year_cols: List[str]) -> List[int]:
    """
    Process a single IPCR group.
    Returns list of gaps (including 0 for pioneer).
    """
    # First compute pioneer
    first_year_map: Dict[str, Optional[int]] = {}
    for cty, sub in group_df.groupby("Country", sort=False):
        fy = first_year_ge2(sub.iloc[0], year_cols)
        first_year_map[cty] = fy

    valid = {k: v for k, v in first_year_map.items() if v is not None}
    if not valid:
        return []

    earliest = min(valid.values())
    candidates = [cty for cty, yr in valid.items() if yr == earliest]
    if len(candidates) == 1:
        pioneer_cty = candidates[0]
    else:
        # Tie: compare Z at earliest year
        ycol = str(earliest)
        tmp = group_df.set_index("Country")[ycol].astype(float)
        z_vals = {cty: tmp.get(cty, np.nan) for cty in candidates}
        max_z = np.nanmax(list(z_vals.values()))
        top = [cty for cty, z in z_vals.items() if z == max_z]
        pioneer_cty = sorted(top)[0] if top else sorted(candidates)[0]

    pioneer_year = earliest

    # Compute gaps for all countries
    gaps = []
    for cty, sub in group_df.groupby("Country", sort=False):
        fy = first_year_ge2(sub.iloc[0], year_cols)
        if fy is None:
            continue
        gaps.append(fy - pioneer_year)
    return gaps

def process_ipcr_chunk(chunk: Iterable[Tuple[str, pd.DataFrame]], year_cols: List[str]) -> List[int]:
    """Process a list of (ipcr, group_df) pairs and return aggregated gaps."""
    all_gaps = []
    for ipcr, group in chunk:
        gaps = pioneer_and_gaps_for_ipcr(group, year_cols)
        all_gaps.extend(gaps)
    return all_gaps

# -------------------- Period processing with parallelisation --------------------
def compute_gaps_parallel(df: pd.DataFrame, year_cols: List[str]) -> List[int]:
    """Split IPCR groups into chunks and process in parallel."""
    # Group by IPCR (IPCR is string)
    grouped = list(df.groupby("IPCR", sort=False))
    if not grouped:
        return []

    # Prepare chunks
    chunks = [grouped[i:i + CHUNK_SIZE] for i in range(0, len(grouped), CHUNK_SIZE)]

    with Pool(N_WORKERS) as pool:
        results = pool.starmap(process_ipcr_chunk, [(chunk, year_cols) for chunk in chunks])

    # Flatten results
    all_gaps = [gap for res in results for gap in res]
    return all_gaps

# -------------------- Main --------------------
def main():
    all_stats = []
    period_data = []  # 存储 (period, gaps_nonzero, mean, median, std)

    for period in PERIODS:
        z_path = Z_FILES[period]
        if not os.path.exists(z_path):
            raise FileNotFoundError(f"Missing Z file for period {period}: {z_path}")

        print(f"Processing {period} ...")
        zdf, year_cols = load_z_period(z_path)
        print(f"  Loaded {len(zdf)} rows, {len(year_cols)} year columns.")

        zdf_adj = apply_transition_deduplication(zdf, year_cols)
        del zdf  # free original

        gaps = compute_gaps_parallel(zdf_adj, year_cols)
        print(f"  Computed {len(gaps)} year gaps.")

        if not gaps:
            print(f"  No valid gaps found for {period}. Skipping statistics and histogram.")
            stats = {
                "Period": period,
                "max": np.nan,
                "min": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "zero_ratio": np.nan
            }
            all_stats.append(stats)
            continue

        # ---- 分离零和非零差距 ----
        gaps_nonzero = [g for g in gaps if g != 0]
        n_total = len(gaps)
        n_zero = n_total - len(gaps_nonzero)
        zero_ratio = n_zero / n_total if n_total > 0 else np.nan

        if gaps_nonzero:
            stats = {
                "Period": period,
                "max": int(max(gaps_nonzero)),
                "min": int(min(gaps_nonzero)),
                "mean": float(np.mean(gaps_nonzero)),
                "median": float(np.median(gaps_nonzero)),
                "std": float(np.std(gaps_nonzero, ddof=1)),
                "zero_ratio": zero_ratio
            }
            mean_val = stats["mean"]
            median_val = stats["median"]
            std_val = stats["std"]
            period_data.append((period, gaps_nonzero, mean_val, median_val, std_val))
        else:
            stats = {
                "Period": period,
                "max": np.nan,
                "min": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "zero_ratio": zero_ratio
            }
            print(f"  No non-zero year gaps for {period}, histogram skipped.")

        all_stats.append(stats)
        print(f"  Stats (excluding zero): min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, zero_ratio={stats['zero_ratio']:.2%}")

        # ---- 单张直方图（增加高度） ----
        if gaps_nonzero:
            plt.figure(figsize=(7, 6))  # 修改：高度从 4.5 提升至 6
            bins = np.arange(min(gaps_nonzero), max(gaps_nonzero) + 2) - 0.5
            plt.hist(gaps_nonzero, bins=bins, edgecolor="black")
            plt.xlabel(r"$\mathrm{RPSG}_{e,i}$")
            plt.ylabel("Frequency")
            title = (f"{period}\n"
                     f"mean = {stats['mean']:.2f}, median = {stats['median']:.2f}, "
                     f"std = {stats['std']:.2f}")
            plt.title(title)
            plt.tight_layout()
            fig_path = os.path.join(FIG_DIR, f"{period}_year_gap_hist.pdf")
            plt.savefig(fig_path)
            plt.close()
            print(f"  Histogram saved to {fig_path}")

    # 保存统计表
    stats_df = pd.DataFrame(all_stats)
    for col in ["mean", "median", "std", "zero_ratio"]:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(3)
    stats_df.to_csv(OUT_STATS, index=False)
    print(f"Statistics saved to {OUT_STATS}")

    # ---- 组合直方图（增加高度并调整子图间距） ----
    if len(period_data) == 3:
        fig, axes = plt.subplots(3, 1, figsize=(7, 9))  # 修改：高度从 6 提升至 9
        plt.subplots_adjust(hspace=0.4)  # 适当增大间距，避免拥挤
        for idx, (period, gaps_nonzero, mean_val, median_val, std_val) in enumerate(period_data):
            ax = axes[idx]
            if gaps_nonzero:
                bins = np.arange(min(gaps_nonzero), max(gaps_nonzero) + 2) - 0.5
                ax.hist(gaps_nonzero, bins=bins, edgecolor='black')
                ax.set_ylabel('Frequency')
                title = (f"{period}\n"
                         f"mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}")
                ax.set_title(title)
            else:
                ax.text(0.5, 0.5, 'No non-zero gaps', transform=ax.transAxes, ha='center')
                ax.set_title(period)
            if idx == 2:
                ax.set_xlabel(r"$\mathrm{RPSG}_{e,i}$")
            else:
                ax.set_xlabel('')
        plt.tight_layout()
        combined_path = os.path.join(FIG_DIR, "combined_year_gap_hist.pdf")
        plt.savefig(combined_path)
        plt.close()
        print(f"Combined histogram saved to {combined_path}")

if __name__ == "__main__":
    main()