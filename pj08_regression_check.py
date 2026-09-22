#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise correlation + multicollinearity grouping for WDI X variables.

Inputs
------
- X list (40 items): /data01/Rong_backup/World_Development_Indicators/Regression_x.xlsx
    Must contain a column named 'Series Code' (case-insensitive allowed).
- Wide data:        /data01/Rong_backup/World_Development_Indicators/WDI_data.csv
    Columns: ['Country Name','Country Code','Series Name','Series Code', <year columns>]
    Year columns may be '1960 [YR1960]' or '1960'. Missing values may be '..'.

Outputs
-------
- Pairwise correlations (long form):
    /data01/rong_dataset/Result/pj08/x_check.csv
    Columns: ['A','B','pearson_r','n_common']
- Update Regression_x.xlsx with a new column 'group' containing set of group ids:
    e.g., {'1','2'} or {} if no high-correlation membership.

Grouping logic
--------------
- Build a graph with an undirected edge between A and B if:
    |r| >= EDGE_R_MIN and n_common >= MIN_COMMON_N
- Find all cliques of size >= 3 -> assign increasing group ids (1,2,...)
- Remaining edges (pairs) not fully covered by any clique -> each becomes its own group
- A variable can belong to multiple groups; write as a Python-set-like string:
    {'1','3'} ; empty set: {}

Notes
-----
- Uses chunked CSV reading to reduce memory.
- Correlations are computed across all (country, year) observations after aligning
  by the same (Country Code, Year). Pairwise NaNs are dropped per correlation.
"""

import os
import re
import math
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd

# For grouping (clique detection)
try:
    import networkx as nx
except ImportError:
    raise SystemExit("Please install networkx: pip install networkx")

# ========================== Config ==========================
XLSX_PATH = "/data01/Rong_backup/World_Development_Indicators/Regression_x.xlsx"
WDI_CSV   = "/data01/Rong_backup/World_Development_Indicators/WDI_data.csv"
OUT_CORR  = "/data01/rong_dataset/Result/pj08/x_check.csv"

# Correlation + grouping thresholds
EDGE_R_MIN     = 0.80   # absolute Pearson r threshold for a 'high correlation' edge
MIN_COMMON_N   = 30     # minimum pairwise non-missing sample size to accept the edge

# CSV read options
CSV_CHUNKSIZE  = 200_000
READ_DTYPE_STR = True   # read as string then coerce to float for year cols

# ===========================================================

def detect_year_cols(cols: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Detect year columns and normalize their names to plain 4-digit years.
    Supports: '1960', or '1960 [YR1960]'.
    Returns:
      - year_cols: original column names that are years
      - norm_map : mapping from original column name -> normalized 'YYYY' string
    """
    pat_plain   = re.compile(r"^\s*(\d{4})\s*$")
    pat_bracket = re.compile(r"^\s*(\d{4})\s*\[YR\1\]\s*$", re.IGNORECASE)
    year_cols = []
    norm_map  = {}
    for c in cols:
        s = str(c).strip()
        m1 = pat_plain.match(s)
        m2 = pat_bracket.match(s)
        if m1:
            yyyy = m1.group(1)
            year_cols.append(c)
            norm_map[c] = yyyy
        elif m2:
            yyyy = m2.group(1)
            year_cols.append(c)
            norm_map[c] = yyyy
    return year_cols, norm_map

def read_x_list(xlsx_path: str) -> pd.DataFrame:
    """
    Read Regression_x.xlsx and return a DataFrame with at least 'Series Code'.
    Also returns the original df to update 'group' later.
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    # find Series Code column case-insensitively
    colmap = {c.lower().strip(): c for c in df.columns}
    if "series code" not in colmap:
        raise ValueError("`Regression_x.xlsx` must contain a 'Series Code' column.")
    return df, colmap["series code"]

def load_wdi_for_series(series_codes: Set[str]) -> pd.DataFrame:
    """
    Load only rows for the given set of series codes from the wide WDI csv.
    Return a long-form DataFrame with index=(Country Code, year) and columns:
    ['Series Code','value'] (plus Series Name kept for reference).
    """
    # read header to detect year columns
    head = pd.read_csv(
        WDI_CSV,
        nrows=1,
        dtype=str if READ_DTYPE_STR else None,
        keep_default_na=True,
        low_memory=False
    )
    year_cols, norm_map = detect_year_cols(list(head.columns))
    base_cols = ["Country Code", "Series Code", "Series Name"]
    missing = [c for c in base_cols if c not in head.columns]
    if missing:
        raise ValueError(f"WDI_data.csv missing required columns: {missing}")

    long_frames = []
    # Chunked reading
    reader = pd.read_csv(
        WDI_CSV,
        dtype=str if READ_DTYPE_STR else None,
        keep_default_na=True,
        low_memory=False,
        chunksize=CSV_CHUNKSIZE
    )
    for chunk in reader:
        # Filter Series Code in our set
        sub = chunk[chunk["Series Code"].astype(str).isin(series_codes)]
        if sub.empty:
            continue

        # Select only needed columns
        sub = sub[base_cols + year_cols].copy()

        # Melt to long
        long = sub.melt(
            id_vars=base_cols,
            value_vars=year_cols,
            var_name="year_raw",
            value_name="value_raw"
        )
        # Normalize year names and coerce values
        long["year"] = long["year_raw"].map(norm_map).astype(int)
        # Missing values: treat ".." and blanks as NaN
        val = long["value_raw"].replace({"..": np.nan, "": np.nan, " ": np.nan})
        long["value"] = pd.to_numeric(val, errors="coerce")
        long = long[["Country Code", "Series Code", "Series Name", "year", "value"]]
        long_frames.append(long)

    if not long_frames:
        raise ValueError("No matching rows for provided Series Codes in WDI_data.csv.")

    long_all = pd.concat(long_frames, ignore_index=True)

    # Build a pivot with MultiIndex (country, year) as rows and Series Code as columns
    # We'll keep a separate lookup for Series Name (not strictly needed for correlation)
    return long_all

def build_wide_matrix(long_df: pd.DataFrame, series_codes: List[str]) -> pd.DataFrame:
    """
    Pivot long_df to a wide matrix:
        index = (Country Code, year)
        columns = Series Code
        values = value
    Only keep columns for our series_codes (order preserved).
    """
    mat = long_df.pivot_table(
        index=["Country Code", "year"],
        columns="Series Code",
        values="value",
        aggfunc="mean"   # if duplicates exist, mean across duplicates
    )
    # Ensure columns order and subset
    # Some Series Codes may be entirely missing; keep them to report as empty
    mat = mat.reindex(columns=series_codes)
    return mat

def pairwise_corr_with_counts(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations and the number of common non-NaN samples.
    Returns a long-form DataFrame with columns: ['A','B','pearson_r','n_common']
    Only unique pairs A < B.
    """
    codes = list(mat.columns)
    results = []
    for i in range(len(codes)):
        a = codes[i]
        col_a = mat[a]
        for j in range(i+1, len(codes)):
            b = codes[j]
            col_b = mat[b]
            # pairwise complete cases
            mask = col_a.notna() & col_b.notna()
            n = int(mask.sum())
            if n == 0:
                r = np.nan
            else:
                r = float(pd.Series(col_a[mask]).corr(pd.Series(col_b[mask])))
            results.append((a, b, r, n))
    out = pd.DataFrame(results, columns=["A", "B", "pearson_r", "n_common"])
    return out

def build_groups_from_corr(corr_df: pd.DataFrame,
                           r_min: float,
                           n_min: int) -> Dict[str, Set[int]]:
    """
    Build groups from correlation pairs using:
      - Cliques (size >= 3) first
      - Remaining high-corr pairs (size == 2) next
    Returns:
      groups_map: dict SeriesCode -> set({group_ids})
    """
    G = nx.Graph()
    # add nodes from all series codes mentioned
    all_nodes = set(pd.concat([corr_df["A"], corr_df["B"]]).unique().tolist())
    G.add_nodes_from(all_nodes)

    # add edges by threshold
    edges = []
    for _, row in corr_df.iterrows():
        r, n = row["pearson_r"], row["n_common"]
        if pd.notna(r) and abs(r) >= r_min and int(n) >= n_min:
            edges.append((row["A"], row["B"]))
    G.add_edges_from(edges)

    # Initialize membership map
    membership: Dict[str, Set[int]] = {node: set() for node in G.nodes}

    # 1) Find cliques size >= 3
    group_id = 0
    covered_pairs = set()
    for clique in nx.find_cliques(G):
        if len(clique) >= 3:
            group_id += 1
            for v in clique:
                membership[v].add(group_id)
            # mark all internal pairs as covered
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    covered_pairs.add(tuple(sorted((clique[i], clique[j]))))

    # 2) For remaining edges not covered by any clique, create pair groups
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        if key not in covered_pairs:
            group_id += 1
            membership[u].add(group_id)
            membership[v].add(group_id)

    return membership

def format_group_set(s: Set[int]) -> str:
    """Format a Python-set-like string of integers: {'1','2'} or {} if empty."""
    if not s:
        return "{}"
    return "{" + ",".join(f"'{i}'" for i in sorted(s)) + "}"

def main():
    # 1) Read X list
    x_df, series_code_col = read_x_list(XLSX_PATH)
    # Clean and get the ordered list of Series Codes
    series_codes = (
        x_df[series_code_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    series_codes = [s for s in series_codes if s != "nan"]

    if len(series_codes) == 0:
        raise ValueError("No 'Series Code' found in Regression_x.xlsx.")

    series_set = set(series_codes)

    # 2) Load WDI data (only selected Series Code), melt and build wide matrix
    long_df = load_wdi_for_series(series_set)
    mat = build_wide_matrix(long_df, series_codes)

    # 3) Pairwise correlation (Pearson) with counts
    corr_df = pairwise_corr_with_counts(mat)
    # Save correlations (long form)
    os.makedirs(os.path.dirname(OUT_CORR), exist_ok=True)
    corr_df.to_csv(OUT_CORR, index=False)

    # 4) Build groups based on high correlations
    membership = build_groups_from_corr(
        corr_df, r_min=EDGE_R_MIN, n_min=MIN_COMMON_N
    )

    # Ensure every variable from the Excel is included in membership (maybe isolated)
    for code in series_codes:
        membership.setdefault(code, set())

    # 5) Write 'group' back to Regression_x.xlsx (update or append a column)
    # Build a mapping Series Code -> formatted group set
    group_str_map = {k: format_group_set(v) for k, v in membership.items()}

    # Create/replace 'group' column aligned to the same sheet
    x_out = x_df.copy()
    # If multiple columns named similarly, ensure we use the detected one
    x_out["group"] = x_out[series_code_col].astype(str).map(group_str_map).fillna("{}")

    # Write back to the same file (overwrite), preserving the single sheet.
    # If multiple sheets exist and you need a specific one, adjust sheet_name.
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="w") as writer:
        x_out.to_excel(writer, index=False)

    # 6) Final echo
    print(f"Done. Saved correlation pairs to: {OUT_CORR}")
    print(f"Updated 'group' column in: {XLSX_PATH}")
    print(f"Parameters: EDGE_R_MIN={EDGE_R_MIN}, MIN_COMMON_N={MIN_COMMON_N}")
    # Optional: quick summary
    n_edges = ((corr_df["pearson_r"].abs() >= EDGE_R_MIN) &
               (corr_df["n_common"] >= MIN_COMMON_N)).sum()
    n_in_groups = sum(1 for s in membership.values() if len(s) > 0)
    print(f"High-corr edges: {n_edges}, Variables in any group: {n_in_groups}/{len(series_codes)}")

if __name__ == "__main__":
    main()
