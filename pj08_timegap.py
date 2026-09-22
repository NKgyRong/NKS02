#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pioneer and period-wise year-gap analysis with robust CSV decoding and
transition-aware de-duplication for split-country combined series.

ASCII-only version.

Periods (UPDATED: only three):
  P1: 1870-1969
  P2: 1970-2010
  P3: 2011-2024

All outputs report results for these three periods in one file by adding
period-prefixed column groups, except Regression_y which is split into
three files.

Transition rule (unchanged)
---------------------------
For predecessor countries SU/CS/YU, before their transition year, combined
series (e.g., RU/SU, UA/SU, CS/CZ, YU/RS, etc.) must not be double-counted.
We select one primary row per (pred, IPCR) cluster to own all pre-transition
contributions; for all non-primary rows in the same cluster, pre-transition
values are set to NaN so they do not contribute to first-year >=2 detection.
From transition year onward, all rows are counted normally.

Inputs
------
- /data01/rong_dataset/Result/pj08/WWP_1860-1969_Z_c_i_t.csv
- /data01/rong_dataset/Result/pj08/WWP_1970-2010_Z_c_i_t.csv
- /data01/rong_dataset/Result/pj08/WWP_2011-2024_Z_c_i_t.csv
    Columns: Country, IPCR, <year columns as 4-digit strings>
- /data01/rong_dataset/Result/pj08/WWP_Amount_ipcr_new_by_Country.csv
    Columns: Country, Name
- /data01/rong_dataset/Result/pj08/WWP_ipcr_subclass_name.csv
    Columns: IPCR_sub3, IPC Subclass Abbreviation
- /data01/rong_dataset/Result/pj08/WWP_ipcr_group_name.csv
    Columns: IPCR_sub4, IPC Group Abbreviation

Outputs (period-wise in same file unless noted)
------------------------------------------------
- /data01/rong_dataset/Result/pj08/WWP_Country_pionner.csv
    Columns:
      Country, Name,
      1870-1969_Pioneer_no, 1870-1969_Avg_year_gap, 1870-1969_Mid_year_gap,
      1970-2010_Pioneer_no, 1970-2010_Avg_year_gap, 1970-2010_Mid_year_gap,
      2011-2024_Pioneer_no, 2011-2024_Avg_year_gap, 2011-2024_Mid_year_gap

- /data01/rong_dataset/Result/pj08/WWP_IPC_subclass_pionner.csv
    Columns:
      IPCsubclass, IPC Subclass Abbreviation,
      1870-1969_Pioneer, 1870-1969_Pioneer_no/IPCgroup_no,
      1970-2010_Pioneer, 1970-2010_Pioneer_no/IPCgroup_no,
      2011-2024_Pioneer, 2011-2024_Pioneer_no/IPCgroup_no

- /data01/rong_dataset/Result/pj08/WWP_first_year_gap.csv
    Columns:
      IPCRgroup, IPC Group Abbreviation,
      1870-1969_Pioneer, 1870-1969_Second country, 1870-1969_first_year_gap,
      1970-2010_Pioneer, 1970-2010_Second country, 1970-2010_first_year_gap,
      2011-2024_Pioneer, 2011-2024_Second country, 2011-2024_first_year_gap

- /data01/rong_dataset/Result/pj08/ww_year_gap.csv
    Long table with columns: period, ww_year_gap, count

- /data01/rong_dataset/Result/pj08/Fig/{period}_ww_year_gap.pdf
    One histogram per period (no GMM)

- Regression y (split into three files)
    - /data01/rong_dataset/Result/pj08/1870-1969_Regression_y.csv
    - /data01/rong_dataset/Result/pj08/1970-2010_Regression_y.csv
    - /data01/rong_dataset/Result/pj08/2011-2024_Regression_y.csv
"""

import os
import re
import math
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Config --------------------
BASE_DIR = "/data01/rong_dataset/Result/pj08/"

# Segmented Z inputs to be combined horizontally
Z_SEG_FILES = [
    os.path.join(BASE_DIR, "WWP_1860-1969_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_1970-2010_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_2011-2024_Z_c_i_t.csv"),
]

COUNTRY_NAME_PATH = os.path.join(BASE_DIR, "WWP_Amount_ipcr_new_by_Country.csv")
SUBCLASS_NAME_PATH = os.path.join(BASE_DIR, "WWP_ipcr_subclass_name.csv")
GROUP_NAME_PATH = os.path.join(BASE_DIR, "WWP_ipcr_group_name.csv")

OUT_COUNTRY_PIONEER = os.path.join(BASE_DIR, "WWP_Country_pionner.csv")       # keep requested spelling
OUT_SUBCLASS_PIONEER = os.path.join(BASE_DIR, "WWP_IPC_subclass_pionner.csv")  # keep requested spelling
OUT_FIRST_YEAR_GAP = os.path.join(BASE_DIR, "WWP_first_year_gap.csv")
OUT_WW_GAP_TABLE = os.path.join(BASE_DIR, "ww_year_gap.csv")

FIG_DIR = os.path.join(BASE_DIR, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)

# Regression_y per period (UPDATED: 3 files)
OUT_REG_Y_TPL = os.path.join(BASE_DIR, "{period}_Regression_y.csv")

# Period definitions (UPDATED: only three)
PERIODS: List[Tuple[str, int, int]] = [
    ("1870-1969", 1870, 1969),
    ("1970-2010", 1970, 2010),
    ("2011-2024", 2011, 2024),
]

# -------------------- Transition rules --------------------
TRANSITION: Dict[str, int] = {"SU": 1991, "CS": 1993, "YU": 2003}
PREDS = set(TRANSITION.keys())

# -------------------- Robust CSV loader --------------------
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

# -------------------- Helpers --------------------
YEAR_PATTERN = re.compile(r"^\d{4}$")

def select_year_cols(columns) -> List[str]:
    years = [c for c in columns if YEAR_PATTERN.match(str(c))]
    return sorted(years, key=lambda x: int(x))

def first_year_ge2_in_period(z_series: pd.Series, period_years: List[str]) -> Optional[int]:
    s = pd.to_numeric(z_series, errors="coerce")
    s = s.reindex(period_years)
    mask = s >= 2.0
    if mask is None or not mask.any():
        return None
    yrs = [int(y) for y in s.index[mask]]
    return min(yrs) if yrs else None

def pioneer_with_tie_break_period(group_df: pd.DataFrame, period_years: List[str]) -> Tuple[Optional[str], Optional[int], Dict[str, Optional[int]]]:
    """
    Determine the pioneer within the given period_years among rows in group_df (same IPCR).
    Tie-break rule:
      1) earliest first-year (within the period) wins;
      2) if tie, max Z at that earliest year;
      3) if still tie, lexicographically smallest country code.
    Returns: (pioneer_country, pioneer_year, first_year_map)
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

def median_half_year(first_year_list: List[int]) -> Optional[int]:
    arr = sorted(int(x) for x in first_year_list)
    if not arr:
        return None
    k_index = math.ceil(len(arr) / 2) - 1
    return arr[k_index]

def ipcr_to_sub3(ipcr_sub4: str) -> Optional[str]:
    if not isinstance(ipcr_sub4, str) or len(ipcr_sub4) < 4:
        return None
    return ipcr_sub4[:4]

# -------------------- Transition helpers --------------------
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

# -------------------- Load and combine segmented Z --------------------
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

# -------------------- Load data --------------------
zdf = load_and_combine_z_segments(Z_SEG_FILES)
zdf.columns = [str(c).strip() for c in zdf.columns]

required_cols = {"Country", "IPCR"}
missing = required_cols - set(zdf.columns)
if missing:
    raise ValueError(f"Missing required columns in combined Z file: {missing}")

year_cols = select_year_cols(zdf.columns)
if not year_cols:
    raise ValueError("No year columns found (expect 4-digit year columns).")
zdf[year_cols] = zdf[year_cols].apply(pd.to_numeric, errors="coerce")

# Apply transition de-duplication BEFORE any period logic
zdf_adj = apply_transition_deduplication(zdf, year_cols)

# meta
COUNTRY_NAME_PATH = os.path.join(BASE_DIR, "WWP_Amount_ipcr_new_by_Country.csv")
SUBCLASS_NAME_PATH = os.path.join(BASE_DIR, "WWP_ipcr_subclass_name.csv")
GROUP_NAME_PATH = os.path.join(BASE_DIR, "WWP_ipcr_group_name.csv")

cnames = read_csv_safely(COUNTRY_NAME_PATH)[["Country", "Name"]].dropna()
country_to_name = dict(zip(cnames["Country"].astype(str), cnames["Name"].astype(str)))

subclass_names = read_csv_safely(SUBCLASS_NAME_PATH).rename(columns={"IPCR_sub3": "IPCsubclass"})
sub3_to_abbrev = dict(
    zip(subclass_names["IPCsubclass"].astype(str), subclass_names["IPC Subclass Abbreviation"].astype(str))
)

group_names = read_csv_safely(GROUP_NAME_PATH).rename(columns={"IPCR_sub4": "IPCRgroup"})
sub4_to_abbrev = dict(
    zip(group_names["IPCRgroup"].astype(str), group_names["IPC Group Abbreviation"].astype(str))
)

# -------------------- Period-wise computation --------------------
period_ipcr_pioneer: Dict[str, Dict[str, Tuple[Optional[str], Optional[int]]]] = {}
period_ipcr_first_years: Dict[str, Dict[str, Dict[str, Optional[int]]]] = {}
period_ipcr_first_gap: Dict[str, Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]]] = {}
period_ipcr_ww_gap: Dict[str, Dict[str, Optional[int]]] = {}
period_country_pioneer_counts: Dict[str, Counter] = {}
period_country_gap_values: Dict[str, defaultdict] = {}

for period_label, y0, y1 in PERIODS:
    period_years = [str(y) for y in range(y0, y1 + 1) if str(y) in year_cols]
    period_ipcr_pioneer[period_label] = {}
    period_ipcr_first_years[period_label] = {}
    period_ipcr_first_gap[period_label] = {}
    period_ipcr_ww_gap[period_label] = {}
    period_country_pioneer_counts[period_label] = Counter()
    period_country_gap_values[period_label] = defaultdict(list)

    for ipcr, g in zdf_adj.groupby("IPCR", sort=False):
        if not period_years:
            period_ipcr_pioneer[period_label][ipcr] = (None, None)
            period_ipcr_first_years[period_label][ipcr] = {c: None for c in g["Country"].unique()}
            period_ipcr_first_gap[period_label][ipcr] = (None, None, None)
            continue

        use_cols = ["Country", "IPCR"] + period_years
        pioneer_cty, pioneer_year, firstmap = pioneer_with_tie_break_period(
            g[use_cols], period_years
        )
        period_ipcr_first_years[period_label][ipcr] = firstmap
        period_ipcr_pioneer[period_label][ipcr] = (pioneer_cty, pioneer_year)

        valid_years = {c: y for c, y in firstmap.items() if y is not None}
        if pioneer_cty is None or pioneer_year is None or not valid_years:
            period_ipcr_first_gap[period_label][ipcr] = (None, None, None)
            continue

        period_country_pioneer_counts[period_label][pioneer_cty] += 1

        other_years = [y for c, y in valid_years.items() if c != pioneer_cty]

        period_country_gap_values[period_label][pioneer_cty].append(0)

        for y in other_years:
            gap = int(y) - int(pioneer_year)
            if gap >= 0:  
                period_country_gap_values[period_label][pioneer_cty].append(gap)


        if len(valid_years) >= 2:
            half_year = median_half_year(list(valid_years.values()))
            if half_year is not None:
                period_ipcr_ww_gap[period_label][ipcr] = int(half_year) - int(pioneer_year)

        sorted_years = sorted(valid_years.values())
        later_years = [y for y in sorted_years if y > pioneer_year]

        second_cty: Optional[str] = None
        first_gap: Optional[int] = None
        if later_years:
            second_year = int(later_years[0])
            candidates = [c for c, y in valid_years.items() if int(y) == second_year]
            if len(candidates) == 1:
                second_cty = candidates[0]
            else:
                ycol = str(second_year)
                z_at_second = g.set_index("Country")[ycol].astype(float)
                max_z = np.nanmax([z_at_second.get(c, np.nan) for c in candidates])
                top_cands = [c for c in candidates if z_at_second.get(c, np.nan) == max_z]
                second_cty = sorted(top_cands)[0] if top_cands else sorted(candidates)[0]
            first_gap = int(second_year) - int(pioneer_year)

        period_ipcr_first_gap[period_label][ipcr] = (pioneer_cty, first_gap, second_cty)

    # Regression_y for this period
    reg_rows = []
    for ipcr, firstmap in period_ipcr_first_years[period_label].items():
        pioneer_cty, pioneer_year = period_ipcr_pioneer[period_label].get(ipcr, (None, None))
        for cty, first_y in firstmap.items():
            if first_y is None:
                reg_rows.append({
                    "IPCR": ipcr,
                    "Country": cty,
                    "Gap": np.nan,
                    "Pioneer Time": (int(pioneer_year) if pioneer_year is not None else np.nan),
                    "Z>=2 Time": np.nan
                })
                continue
            if (pioneer_cty is not None) and (pioneer_year is not None):
                gap_val = int(first_y) - int(pioneer_year)
                if cty == pioneer_cty and int(first_y) == int(pioneer_year):
                    gap_val = 0
                reg_rows.append({
                    "IPCR": ipcr,
                    "Country": cty,
                    "Gap": int(gap_val),
                    "Pioneer Time": int(pioneer_year),
                    "Z>=2 Time": int(first_y)
                })
            else:
                reg_rows.append({
                    "IPCR": ipcr,
                    "Country": cty,
                    "Gap": np.nan,
                    "Pioneer Time": np.nan,
                    "Z>=2 Time": int(first_y)
                })
    reg_df = pd.DataFrame(reg_rows).sort_values(["IPCR", "Country"], kind="mergesort")
    reg_df.to_csv(OUT_REG_Y_TPL.format(period=period_label), index=False)

# -------------------- Output 1: WWP_Country_pionner.csv --------------------
all_countries = sorted(set(zdf["Country"].astype(str)))
rows_country = []
for cty in all_countries:
    name = country_to_name.get(cty, "")
    row = {"Country": cty, "Name": name}
    for period_label, _, _ in PERIODS:
        gaps = period_country_gap_values[period_label].get(cty, [])
        pcount = int(period_country_pioneer_counts[period_label].get(cty, 0))
        avg_gap = float(np.mean(gaps)) if gaps else np.nan
        mid_gap = float(np.median(gaps)) if gaps else np.nan
        row[f"{period_label}_Pioneer_no"] = pcount
        row[f"{period_label}_Avg_year_gap"] = round(avg_gap, 3) if not np.isnan(avg_gap) else np.nan
        row[f"{period_label}_Mid_year_gap"] = round(mid_gap, 3) if not np.isnan(mid_gap) else np.nan
    rows_country.append(row)

country_df_out = pd.DataFrame(rows_country)
pioneer_sum = country_df_out[[f"{p}_Pioneer_no" for p,_,_ in PERIODS]].sum(axis=1)
country_df_out = (country_df_out
                  .assign(_sum=pioneer_sum)
                  .sort_values(["_sum","Country"], ascending=[False, True])
                  .drop(columns=["_sum"]))
country_df_out.to_csv(OUT_COUNTRY_PIONEER, index=False)

# -------------------- Output 2: WWP_IPC_subclass_pionner.csv --------------------
subclass_rows: Dict[str, Dict[str, object]] = {}
all_sub3 = sorted(set(zdf["IPCR"].astype(str).str[:4]))
for sub3 in all_sub3:
    subclass_rows[sub3] = {
        "IPCsubclass": sub3,
        "IPC Subclass Abbreviation": sub3_to_abbrev.get(sub3, "")
    }
    for period_label, _, _ in PERIODS:
        pioneers: List[str] = []
        group_no = len(set(zdf.loc[zdf["IPCR"].astype(str).str.startswith(sub3), "IPCR"]))
        ipcr_pmap = period_ipcr_pioneer[period_label]
        for ipcr_code, (pioneer_cty, _py) in ipcr_pmap.items():
            if isinstance(ipcr_code, str) and ipcr_code.startswith(sub3) and pioneer_cty is not None:
                pioneers.append(pioneer_cty)
        if group_no == 0:
            subclass_rows[sub3][f"{period_label}_Pioneer"] = ""
            subclass_rows[sub3][f"{period_label}_Pioneer_no/IPCgroup_no"] = "0/0"
            continue
        cnt = Counter(pioneers)
        if len(cnt) == 0:
            subclass_rows[sub3][f"{period_label}_Pioneer"] = ""
            subclass_rows[sub3][f"{period_label}_Pioneer_no/IPCgroup_no"] = f"0/{group_no}"
            continue
        max_freq = max(cnt.values())
        top_codes = sorted([k for k, v in cnt.items() if v == max_freq])
        top_code = top_codes[0]
        top_name = country_to_name.get(top_code, top_code)
        subclass_rows[sub3][f"{period_label}_Pioneer"] = top_name
        subclass_rows[sub3][f"{period_label}_Pioneer_no/IPCgroup_no"] = f"{max_freq}/{group_no}"

subclass_df_out = pd.DataFrame.from_dict(subclass_rows, orient="index").reset_index(drop=True)
subclass_df_out.sort_values(["IPCsubclass"], inplace=True)
subclass_df_out.to_csv(OUT_SUBCLASS_PIONEER, index=False)

# -------------------- Output 3: WWP_first_year_gap.csv --------------------
rows_firstgap = []
all_ipcr = sorted(set(zdf["IPCR"].astype(str)))
for ipcr in all_ipcr:
    row = {
        "IPCRgroup": ipcr,
        "IPC Group Abbreviation": sub4_to_abbrev.get(ipcr, "")
    }
    for period_label, _, _ in PERIODS:
        pioneer_cty_py = period_ipcr_pioneer.get(period_label, {}).get(ipcr, (None, None))
        pioneer_cty_py = pioneer_cty_py if pioneer_cty_py is not None else (None, None)
        pioneer_cty, pioneer_year = pioneer_cty_py

        pioneer_cty_fg, first_gap, second_cty = period_ipcr_first_gap.get(period_label, {}).get(
            ipcr, (None, None, None)
        )

        row[f"{period_label}_Pioneer"] = pioneer_cty_fg if pioneer_cty_fg is not None else ""
        row[f"{period_label}_Pioneer_year"] = (int(pioneer_year) if pioneer_year is not None else np.nan)
        row[f"{period_label}_Second country"] = second_cty if second_cty is not None else ""
        row[f"{period_label}_first_year_gap"] = (int(first_gap) if first_gap is not None else np.nan)

    rows_firstgap.append(row)

firstgap_df = pd.DataFrame(rows_firstgap)
sort_key = f"{PERIODS[0][0]}_first_year_gap"  # "1870-1969_first_year_gap"
firstgap_df.sort_values([sort_key, "IPCRgroup"], ascending=[True, True], inplace=True, na_position="last")
firstgap_df.to_csv(OUT_FIRST_YEAR_GAP, index=False)

# -------------------- Output 4: ww_year_gap per-period long table + histograms --------------------
ww_rows = []
for period_label, _, _ in PERIODS:
    gaps = [int(g) for g in period_ipcr_ww_gap[period_label].values() if g is not None]
    if gaps:
        freq = Counter(gaps)
        for k, v in sorted(freq.items(), key=lambda kv: kv[0]):
            ww_rows.append({"period": period_label, "ww_year_gap": int(k), "count": int(v)})

        min_g, max_g = min(gaps), max(gaps)
        bins = np.arange(min_g, max_g + 2) - 0.5  # unit width
        plt.figure(figsize=(7, 4.5))
        plt.hist(gaps, bins=bins, edgecolor="black")
        plt.xlabel("ww_year_gap (half-of-countries year - pioneer year)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of ww_year_gap: {period_label}")
        plt.tight_layout()
        fig_out = os.path.join(FIG_DIR, f"{period_label}_ww_year_gap.pdf")
        plt.savefig(fig_out)
        plt.close()

ww_table = pd.DataFrame(ww_rows, columns=["period", "ww_year_gap", "count"])
ww_table.to_csv(OUT_WW_GAP_TABLE, index=False)

# -------------------- Prints --------------------
print(f"[OK] Pioneer per country (period-wise) -> {OUT_COUNTRY_PIONEER}")
print(f"[OK] Pioneer per IPC subclass (period-wise) -> {OUT_SUBCLASS_PIONEER}")
print(f"[OK] First-year gap per IPCR (period-wise) -> {OUT_FIRST_YEAR_GAP}")
print(f"[OK] ww_year_gap long table -> {OUT_WW_GAP_TABLE}")
for period_label, _, _ in PERIODS:
    print(f"[OK] Regression-ready y -> {OUT_REG_Y_TPL.format(period=period_label)}")
    fig_path = os.path.join(FIG_DIR, f"{period_label}_ww_year_gap.pdf")
    if os.path.exists(fig_path):
        print(f"[OK] ww_year_gap histogram -> {fig_path}")
