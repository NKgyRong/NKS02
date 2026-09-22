#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count pioneer IPC groups (4-digit IPCR) for US, CN, JP across three periods.
Output a table (CSV and console).
"""

import os
import re
from typing import Optional, Dict, List, Tuple
from collections import Counter

import pandas as pd
import numpy as np

# ================= Configuration =================
BASE_DIR = "/data01/rong_dataset/Result/pj08/"

Z_SEG_FILES = [
    os.path.join(BASE_DIR, "WWP_1860-1969_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_1970-2010_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_2011-2024_Z_c_i_t.csv"),
]

# Period definitions
PERIODS: List[Tuple[str, int, int]] = [
    ("1870-1959", 1870, 1959),
    ("1960-2010", 1960, 2010),
    ("2011-2024", 2011, 2024),
]

# Target countries (codes)
TARGET_COUNTRIES = {"US", "CN", "JP"}

# Excluded countries (same as original)
COUNTRY_EXCLUDE = {"AT", "IE"}

# Output file
OUT_CSV = os.path.join(BASE_DIR, "pioneer_count_by_country.csv")

# ============== Helper functions ==============
YEAR_PATTERN = re.compile(r"^\d{4}$")

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

def select_year_cols(columns) -> List[str]:
    years = [c for c in columns if YEAR_PATTERN.match(str(c))]
    return sorted(years, key=lambda x: int(x))

def first_year_ge2_in_period(z_series: pd.Series, period_years: List[str]) -> Optional[int]:
    """Return the first year (as int) where Z >= 2, or None if none."""
    s = pd.to_numeric(z_series, errors="coerce")
    s = s.reindex(period_years)
    mask = s >= 2.0
    if mask is None or not mask.any():
        return None
    yrs = [int(y) for y in s.index[mask]]
    return min(yrs) if yrs else None

def pioneer_with_tie_break_period(group_df: pd.DataFrame, period_years: List[str]):
    """
    Given a DataFrame for one IPC group (with Country and Z values for period_years),
    return the pioneer country (according to the tie?break rules) and the earliest year with Z>=2.
    If no pioneer, return (None, None, first_year_map).
    """
    first_year_map: Dict[str, Optional[int]] = {}
    for cty, sub in group_df.groupby("Country", sort=False):
        yr = first_year_ge2_in_period(sub.iloc[0][period_years], period_years)
        first_year_map[cty] = yr

    valid = {k: v for k, v in first_year_map.items() if v is not None}
    if not valid:
        return (None, None, first_year_map)

    earliest = min(valid.values())
    candidates = [cty for cty, yr in valid.items() if yr == earliest]
    if len(candidates) == 1:
        return (candidates[0], earliest, first_year_map)

    ycol = str(earliest)
    tmp = group_df.set_index("Country")[ycol].astype(float)
    z_at_earliest = {cty: tmp.get(cty, np.nan) for cty in candidates}
    max_z = np.nanmax(list(z_at_earliest.values()))
    top_candidates = [cty for cty, z in z_at_earliest.items() if z == max_z]

    if len(top_candidates) == 1:
        return (top_candidates[0], earliest, first_year_map)

    chosen = sorted(top_candidates)[0]
    return (chosen, earliest, first_year_map)

# ============== Transition deduplication (reused from original logic) ==============
TRANSITION: Dict[str, int] = {"SU": 1991, "CS": 1993, "YU": 2003}
PREDS = set(TRANSITION.keys())

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

def build_transition_clusters(zdf: pd.DataFrame, year_cols: List[str]):
    pred_series = zdf["Country"].apply(extract_predecessor)
    trans_mask = pred_series.notna()

    trans_table = pd.DataFrame({
        "row_idx": zdf.index,
        "Country": zdf["Country"],
        "IPCR": zdf["IPCR"],
        "pred": pred_series
    })
    trans_table = trans_table[trans_mask].copy()

    year_ints = np.array([int(y) for y in year_cols], dtype=int)
    pre_cols_by_pred: Dict[str, List[str]] = {}
    for p, tyear in TRANSITION.items():
        pre_cols_by_pred[p] = [str(y) for y in year_ints if y < tyear]

    primary_idx_by_cluster: Dict[Tuple[str, str], int] = {}
    for (pred, ipcr), g in trans_table.groupby(["pred", "IPCR"], dropna=False):
        idxs = g["row_idx"].tolist()
        countries = zdf.loc[idxs, "Country"].tolist()
        pri = None
        for i, c in zip(idxs, countries):
            if c == pred:
                pri = i
                break
        if pri is None:
            pairs = sorted(zip(countries, idxs), key=lambda x: str(x[0]))
            pri = pairs[0][1]
        primary_idx_by_cluster[(pred, ipcr)] = pri

    cluster_rows: Dict[Tuple[str, str], List[int]] = {}
    for (pred, ipcr), g in trans_table.groupby(["pred", "IPCR"], dropna=False):
        cluster_rows[(pred, ipcr)] = g["row_idx"].tolist()

    return primary_idx_by_cluster, cluster_rows, pre_cols_by_pred, pred_series

def apply_transition_deduplication(zdf: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
    if "Country" not in zdf.columns or "IPCR" not in zdf.columns:
        return zdf.copy()

    primary_idx_by_cluster, cluster_rows, pre_cols_by_pred, _ = build_transition_clusters(zdf, year_cols)
    zdf_adj = zdf.copy()

    for (pred, ipcr), idxs in cluster_rows.items():
        pre_cols = pre_cols_by_pred.get(pred, [])
        if not pre_cols:
            continue
        primary_idx = primary_idx_by_cluster[(pred, ipcr)]
        non_primary = [i for i in idxs if i != primary_idx]
        if not non_primary:
            continue
        zdf_adj.loc[non_primary, pre_cols] = np.nan

    return zdf_adj

# ============== Load and combine Z data ==============
def load_and_combine_z_segments(seg_files: List[str]) -> pd.DataFrame:
    base_df = None
    for path in seg_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Z segment file not found: {path}")
        seg = read_csv_safely(path)
        seg.columns = [str(c).strip() for c in seg.columns]
        year_cols_seg = select_year_cols(seg.columns)
        seg = seg[["Country", "IPCR"] + year_cols_seg].copy()
        if base_df is None:
            base_df = seg
        else:
            new_years = [c for c in year_cols_seg if c not in base_df.columns]
            base_df = pd.merge(
                base_df,
                seg[["Country", "IPCR"] + new_years],
                on=["Country", "IPCR"],
                how="outer"
            )
    if base_df is None:
        raise ValueError("No Z data loaded from segments.")
    return base_df

# ============== Main ==============
def main():
    # 1. Load and combine Z
    zdf = load_and_combine_z_segments(Z_SEG_FILES)
    zdf.columns = [str(c).strip() for c in zdf.columns]
    if not {"Country", "IPCR"}.issubset(set(zdf.columns)):
        raise ValueError("Missing required columns in Z data.")

    year_cols = [c for c in zdf.columns if YEAR_PATTERN.match(str(c))]
    if not year_cols:
        raise ValueError("No 4-digit year columns found in Z files.")
    zdf[year_cols] = zdf[year_cols].apply(pd.to_numeric, errors="coerce")

    # 2. Transition deduplication
    zdf_adj = apply_transition_deduplication(zdf, year_cols)

    # 3. Exclude specified countries
    zdf_adj = zdf_adj[~zdf_adj["Country"].isin(COUNTRY_EXCLUDE)].copy()

    # 4. Get all unique IPC groups (IPCR)
    all_ipcr = zdf_adj["IPCR"].unique()
    print(f"Total unique IPCR groups: {len(all_ipcr)}")

    # 5. Count by period
    results = []  # list of dicts: (period, country, count)

    for period_label, y0, y1 in PERIODS:
        period_years = [str(y) for y in range(y0, y1 + 1) if str(y) in year_cols]
        if not period_years:
            print(f"Warning: No years for period {period_label}, skipping.")
            continue

        pioneer_counter = Counter()

        for ipcr in all_ipcr:
            mask = zdf_adj["IPCR"].astype(str) == ipcr
            if not mask.any():
                continue
            group_df = zdf_adj.loc[mask, ["Country", "IPCR"] + period_years].copy()
            if group_df.empty:
                continue

            pioneer_cty, _, _ = pioneer_with_tie_break_period(group_df, period_years)
            if pioneer_cty is not None and pioneer_cty in TARGET_COUNTRIES:
                pioneer_counter[pioneer_cty] += 1

        for cty in TARGET_COUNTRIES:
            results.append({
                "Period": period_label,
                "Country": cty,
                "Pioneer_IPCR_Count": pioneer_counter.get(cty, 0)
            })

    # 6. Output table
    df_out = pd.DataFrame(results)
    pivot = df_out.pivot(index="Period", columns="Country", values="Pioneer_IPCR_Count")
    pivot = pivot[list(TARGET_COUNTRIES)]  # ensure column order

    print("\n=== Pioneer IPCR group counts by country ===")
    print(pivot)

    pivot.to_csv(OUT_CSV)
    print(f"\nResult saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()