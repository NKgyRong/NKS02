#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze pioneer distribution for 2011-2024 period.
Outputs a CSV with all IPC subclasses, their top5 pioneer countries,
and evenness metrics, for manual selection.
"""

import os
import re
from typing import Optional, Dict, List, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ================= Config =================
BASE_DIR = "/data01/rong_dataset/Result/pj08/"

Z_SEG_FILES = [
    os.path.join(BASE_DIR, "WWP_1860-1969_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_1970-2010_Z_c_i_t.csv"),
    os.path.join(BASE_DIR, "WWP_2011-2024_Z_c_i_t.csv"),
]

SUB3_AMOUNT_PATH = os.path.join(BASE_DIR, "WWP_Amount_ipcr_new_by_sub3.csv")
SUBCLASS_NAME_PATH = os.path.join(BASE_DIR, "WWP_ipcr_subclass_name.csv")
COUNTRY_NAME_PATH = os.path.join(BASE_DIR, "WWP_Amount_ipcr_new_by_Country.csv")

OUT_CANDIDATE_CSV = os.path.join(BASE_DIR, "Pioneer_2011-2024_candidates.csv")

# Period definitions
PERIODS = [("2011-2024", 2011, 2024)]

PERIOD_TO_SUB3_AMOUNT_COL = {"2011-2024": "N_2011_2024"}

# Overrides for IPC subclass names (optional, for readability)
SUBCLASS_ABBREV_OVERRIDES: Dict[str, str] = {
    "C07C": "Acyclic/carbocyclic compounds",
    "H01J": "Electric discharge tubes or discharge lamps",
    "G03B": "Photography/projection apparatus",
    "B29C": "Plastics shaping & joining",
    "C10B": "Destructive distillation (carbonaceous)",
    "G01M": "Testing of machines/structures",
    "H04L": "Digital information transmission",
    "B62D": "Motor vehicles; trailers",
    "H01B": "Cables; conductors; insulators",
    "F24S": "Solar heat collectors; systems",
    "B60M": "Power supply along rails (EV)",
    "B63B": "Ships; equipment for shipping",
    "B64D": "Equipment for aircraft or helicopters",
    "G05D": "Systems for controlling or regulating non-electric variables",
    "H03K": "Pulse technique",
    "G06F": "Digital data processing",
    "H10K": "Organic solid-state devices",
    "H01M": "Batteries & electrochemical energy",
    "F15D": "Fluid dynamics, not otherwise provided for",
    "B64U": "Unmanned Aerial Vehicles, UAV"
}

# Countries to exclude from analysis
COUNTRY_EXCLUDE = {"AT", "IE"}

# Transition rules for country codes (SU, CS, YU)
TRANSITION: Dict[str, int] = {"SU": 1991, "CS": 1993, "YU": 2003}
PREDS = set(TRANSITION.keys())


# ============== Robust CSV loader ==============
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


# ============== Helpers ==============
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

def pioneer_with_tie_break_period(group_df: pd.DataFrame, period_years: List[str]):
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

def ipcr_to_sub3(ipcr_sub4: str) -> Optional[str]:
    if not isinstance(ipcr_sub4, str) or len(ipcr_sub4) < 4:
        return None
    return ipcr_sub4[:4]


# ============== Transition helpers ==============
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


# ============== Load & combine Z ==============
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


# ================= Stats for all subclasses in a period =================
def compute_all_sub3_stats(zdf_adj: pd.DataFrame,
                           period_label: str,
                           period_years: List[str],
                           sub3_amount_df: pd.DataFrame,
                           sub3_to_abbrev: Dict[str, str]) -> pd.DataFrame:
    """
    For every IPC subclass that appears in the data, compute pioneer statistics.
    Returns a DataFrame with columns:
      IPCsubclass, IPCgroup_count, Top1..Top5_country, Top1..Top5_count,
      Period_total_amount, Coverage, StdDev, Score
    """
    # First, collect all distinct IPCR groups and their subclasses
    all_ipcr = zdf_adj["IPCR"].dropna().unique()
    sub3_to_ipcr = defaultdict(list)
    ipcr_to_sub3_cache = {}
    for ipcr in all_ipcr:
        sub = ipcr_to_sub3(str(ipcr))
        if sub is not None:
            sub3_to_ipcr[sub].append(ipcr)
            ipcr_to_sub3_cache[ipcr] = sub

    # For each sub3, we will compute pioneer counts from the data
    # This is more efficient than calling compute_sub3_period_stats for each sub3 separately
    # because we can reuse the IPCR-level pioneer mapping.

    # Build a pioneer map: IPCR -> pioneering country (for this period)
    pioneer_map = {}
    print(f"Computing pioneers for {period_label}...")
    grouped = zdf_adj.groupby("IPCR")
    n_groups = len(grouped)
    processed = 0
    for ipcr, group in grouped:
        pioneer_cty, _, _ = pioneer_with_tie_break_period(group, period_years)
        if pioneer_cty is not None:
            pioneer_map[ipcr] = pioneer_cty
        processed += 1
        if processed % 1000 == 0:
            print(f"  Processed {processed}/{n_groups} IPCR groups...", end="\r")
    print(f"\nDone. Found pioneers for {len(pioneer_map)} IPCR groups.")

    # Aggregate per subclass
    sub3_group_count = Counter()
    sub3_pioneer_counter = defaultdict(Counter)

    for ipcr, cty in pioneer_map.items():
        sub = ipcr_to_sub3_cache.get(ipcr)
        if sub:
            sub3_group_count[sub] += 1
            sub3_pioneer_counter[sub][cty] += 1

    # Now build rows for all subclasses that appear in sub3_group_count
    amount_col = PERIOD_TO_SUB3_AMOUNT_COL[period_label]
    rows = []
    for sub3 in sub3_group_count.keys():
        grp_cnt = sub3_group_count[sub3]
        counter = sub3_pioneer_counter[sub3]
        # Top5
        top_items = counter.most_common()
        if top_items:
            freq_to_ctys = defaultdict(list)
            for cty, cnt in top_items:
                freq_to_ctys[cnt].append(cty)
            stabilized = []
            for cnt in sorted(freq_to_ctys.keys(), reverse=True):
                for cty in sorted(freq_to_ctys[cnt]):
                    stabilized.append((cty, cnt))
            top_items = stabilized
        top_pairs = top_items[:5]
        while len(top_pairs) < 5:
            top_pairs.append(("", 0))

        # Get total patents for this subclass in this period
        if (sub3 in sub3_amount_df.index) and (amount_col in sub3_amount_df.columns):
            period_amount = float(sub3_amount_df.loc[sub3, amount_col])
        else:
            period_amount = np.nan

        counts = [top_pairs[i][1] for i in range(5)]
        total_counts = sum(counts)
        coverage = total_counts / grp_cnt if grp_cnt > 0 else 0.0
        std_counts = np.std(counts) if len(counts) > 1 else 0.0
        score = coverage / (std_counts + 0.1)

        row = {
            "IPCsubclass": sub3,
            "IPCgroup_count": grp_cnt,
            "Top1_country": top_pairs[0][0],
            "Top1_count": top_pairs[0][1],
            "Top2_country": top_pairs[1][0],
            "Top2_count": top_pairs[1][1],
            "Top3_country": top_pairs[2][0],
            "Top3_count": top_pairs[2][1],
            "Top4_country": top_pairs[3][0],
            "Top4_count": top_pairs[3][1],
            "Top5_country": top_pairs[4][0],
            "Top5_count": top_pairs[4][1],
            "Period_total_amount": period_amount,
            "Coverage": coverage,
            "StdDev": std_counts,
            "Score": score,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by score descending (most evenly distributed first)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


# ================= Main =================
def main():
    print("Loading Z data...")
    zdf = load_and_combine_z_segments(Z_SEG_FILES)
    zdf.columns = [str(c).strip() for c in zdf.columns]
    if not {"Country", "IPCR"}.issubset(set(zdf.columns)):
        raise ValueError("Missing required columns in Z data.")

    year_cols = [c for c in zdf.columns if YEAR_PATTERN.match(str(c))]
    if not year_cols:
        raise ValueError("No 4-digit year columns found in Z files.")
    zdf[year_cols] = zdf[year_cols].apply(pd.to_numeric, errors="coerce")

    print("Applying transition deduplication...")
    zdf_adj = apply_transition_deduplication(zdf, year_cols)

    # Exclude countries
    zdf_adj = zdf_adj[~zdf_adj["Country"].isin(COUNTRY_EXCLUDE)].copy()
    print(f"Z data shape after cleaning: {zdf_adj.shape}")

    # Load sub3 totals and abbreviations (for total patent counts)
    sub3_amount = read_csv_safely(SUB3_AMOUNT_PATH)
    sub3_amount.columns = [str(c).strip() for c in sub3_amount.columns]
    sub3_to_abbrev = dict(
        zip(sub3_amount["IPCR_sub3"].astype(str),
            sub3_amount["IPC Subclass Abbreviation"].astype(str))
    )
    # Apply manual overrides (optional, for readability)
    for k, v in SUBCLASS_ABBREV_OVERRIDES.items():
        sub3_to_abbrev[k] = v

    if os.path.exists(SUBCLASS_NAME_PATH):
        subclass_names = read_csv_safely(SUBCLASS_NAME_PATH).rename(columns={"IPCR_sub3": "IPCsubclass"})
        for k, v in zip(subclass_names["IPCsubclass"].astype(str),
                        subclass_names["IPC Subclass Abbreviation"].astype(str)):
            sub3_to_abbrev.setdefault(k, v)
    sub3_amount = sub3_amount.set_index("IPCR_sub3")

    # Period: 2011-2024
    period_label = "2011-2024"
    y0, y1 = 2011, 2024
    period_years = [str(y) for y in range(y0, y1 + 1) if str(y) in year_cols]
    if not period_years:
        raise ValueError("No year columns for 2011-2024 period")

    # Compute stats for all subclasses
    df_results = compute_all_sub3_stats(zdf_adj, period_label, period_years,
                                        sub3_amount, sub3_to_abbrev)

    # Save to CSV
    df_results.to_csv(OUT_CANDIDATE_CSV, index=False)
    print(f"\n[OK] Candidate list saved to: {OUT_CANDIDATE_CSV}")
    print(f"Total subclasses analyzed: {len(df_results)}")
    print("Top 10 most evenly distributed subclasses:")
    print(df_results[["IPCsubclass", "IPCgroup_count", "Coverage", "StdDev", "Score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()