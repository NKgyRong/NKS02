#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build per-segment WWP_{seg}_rho_c_i_mid.csv (ASCII-only).

Steps (per segment):
1) Read amount CSV (global) and rho CSV for the segment (robust to encodings).
   Strip all non-ASCII characters from column names and string cells.
2) From WWP_Amount_ipcr_new_by_T_c_i.csv, select T_c_i (Country-IPCR pairs)
   whose Amount is in the top 25% (>= 75th percentile), computed globally (not per segment).
3) From WWP_{seg}_rho_c_i.csv, keep rows whose T_c_i is in the above set and P < 0.05.
4) Output (ASCII): Country, IPCR, rho, P, Amount, Name, IPC Group Abbreviation
"""

import os
import pandas as pd
import numpy as np

# ======== Paths / Segments ========
BASE_DIR = "/data01/rong_dataset/Result/pj08/"
AMOUNT_PATH = os.path.join(BASE_DIR, "WWP_Amount_ipcr_new_by_T_c_i.csv")

SEGMENTS = ["1860-1969", "1970-2010", "2011-2024"]
RHO_PATH_TMPL = os.path.join(BASE_DIR, "WWP_{seg}_rho_c_i.csv")
OUT_PATH_TMPL = os.path.join(BASE_DIR, "WWP_{seg}_rho_c_i_mid.csv")

# ======== Expected column names (explicit) ========
# Amount file columns:
AMT_COUNTRY_COL = "Country"
AMT_IPCR_COL = "IPCR_sub4"                  # sub4 code
AMT_AMOUNT_COL = "Amount"
AMT_NAME_COL = "Name"
AMT_IPC_GROUP_ABBR_COL = "IPC Group Abbreviation"

# Rho file columns:
RHO_COUNTRY_COL = "Country"
RHO_IPCR_COL = "IPCR"                       # should match sub4 code
RHO_RHO_COL = "rho"
RHO_P_COL = "P"

# ======== ASCII Cleaners ========
def to_ascii_text(x):
    """Convert any value to ASCII-only string (drop non-ASCII)."""
    if pd.isna(x):
        return x
    s = str(x)
    return s.encode("ascii", "ignore").decode("ascii")

def df_make_ascii(df: pd.DataFrame) -> pd.DataFrame:
    """Strip non-ASCII from column names and string-like cells."""
    df.columns = [to_ascii_text(c) for c in df.columns]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(to_ascii_text)
    return df

# ======== Robust CSV loader ========
def read_csv_ascii(path: str, **kwargs) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", **kwargs)
    return df_make_ascii(df)

def coerce_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")

# ======== Load amount once (global threshold) ========
if not os.path.exists(AMOUNT_PATH):
    raise FileNotFoundError(f"Amount file not found: {AMOUNT_PATH}")

amount_df = read_csv_ascii(AMOUNT_PATH)

# Validate amount columns
missing_amt = [c for c in [AMT_COUNTRY_COL, AMT_IPCR_COL, AMT_AMOUNT_COL] if c not in amount_df.columns]
if missing_amt:
    raise KeyError(f"Amount file missing columns: {missing_amt}. Columns present: {list(amount_df.columns)}")

# Keep needed columns
keep_amt_cols = [AMT_COUNTRY_COL, AMT_IPCR_COL, AMT_AMOUNT_COL]
if AMT_NAME_COL in amount_df.columns:
    keep_amt_cols.append(AMT_NAME_COL)
if AMT_IPC_GROUP_ABBR_COL in amount_df.columns:
    keep_amt_cols.append(AMT_IPC_GROUP_ABBR_COL)
amount_df = amount_df[keep_amt_cols].copy()

# Coerce numerics
amount_df[AMT_AMOUNT_COL] = coerce_numeric(amount_df[AMT_AMOUNT_COL])

# Drop NAs
amount_df = amount_df.dropna(subset=[AMT_COUNTRY_COL, AMT_IPCR_COL, AMT_AMOUNT_COL])

# Compute global 75th percentile threshold
threshold_75 = amount_df[AMT_AMOUNT_COL].quantile(0.75)
eligible_amt = amount_df[amount_df[AMT_AMOUNT_COL] >= threshold_75].copy()

print(f"[GLOBAL] 75th percentile Amount threshold: {threshold_75}")
print(f"[GLOBAL] Eligible T_c_i (>= 75th percentile): {len(eligible_amt)}")

# ======== Process each segment ========
for seg in SEGMENTS:
    rho_path = RHO_PATH_TMPL.format(seg=seg)
    out_path = OUT_PATH_TMPL.format(seg=seg)

    if not os.path.exists(rho_path):
        print(f"[{seg}] Rho file not found, skip: {rho_path}")
        continue

    # Load rho for this segment
    rho_df = read_csv_ascii(rho_path)

    # Validate rho columns
    missing_rho = [c for c in [RHO_COUNTRY_COL, RHO_IPCR_COL, RHO_RHO_COL, RHO_P_COL] if c not in rho_df.columns]
    if missing_rho:
        raise KeyError(f"[{seg}] Rho file missing columns: {missing_rho}. Columns present: {list(rho_df.columns)}")

    # Coerce numerics
    rho_df[RHO_RHO_COL] = coerce_numeric(rho_df[RHO_RHO_COL])
    rho_df[RHO_P_COL] = coerce_numeric(rho_df[RHO_P_COL])

    # Drop NAs
    rho_df = rho_df.dropna(subset=[RHO_COUNTRY_COL, RHO_IPCR_COL, RHO_RHO_COL, RHO_P_COL])

    # Merge eligible T_c_i with segment rho
    merged = rho_df.merge(
        eligible_amt,
        left_on=[RHO_COUNTRY_COL, RHO_IPCR_COL],
        right_on=[AMT_COUNTRY_COL, AMT_IPCR_COL],
        how="inner",
        suffixes=("_rho", "_amt"),
    )

    # Build final columns
    if AMT_NAME_COL not in merged.columns:
        merged[AMT_NAME_COL] = np.nan
    if AMT_IPC_GROUP_ABBR_COL not in merged.columns:
        merged[AMT_IPC_GROUP_ABBR_COL] = np.nan

    result = merged[[RHO_COUNTRY_COL,
                     RHO_IPCR_COL,
                     RHO_RHO_COL,
                     RHO_P_COL,
                     AMT_AMOUNT_COL,
                     AMT_NAME_COL,
                     AMT_IPC_GROUP_ABBR_COL]].copy()

    # Standardize names
    result = result.rename(columns={
        RHO_COUNTRY_COL: "Country",
        RHO_IPCR_COL: "IPCR",
        RHO_RHO_COL: "rho",
        RHO_P_COL: "P",
        AMT_AMOUNT_COL: "Amount",
        AMT_NAME_COL: "Name",
        AMT_IPC_GROUP_ABBR_COL: "IPC Group Abbreviation",
    })

    # Filter significance
    result = result[result["P"] < 0.05].reset_index(drop=True)

    # ASCII-only output
    result = df_make_ascii(result)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False, encoding="ascii")

    print(f"[{seg}] Significant (P < 0.05): {len(result)}")
    print(f"[{seg}] Saved -> {out_path}")
