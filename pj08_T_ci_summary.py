#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pj08 pairwise-series builder (ASCII-only)

WHAT THIS SCRIPT DOES
---------------------
Reads raw patent rows from CSV files and produces pj08 aggregates with
political-transition-aware series mapping and head filtering.

Key outputs under PJ08_OUTPUT_DIR:
- WWP_Amount_ipcr_new_by_Country.csv
- WWP_Amount_ipcr_new_by_sub3.csv
- WWP_Amount_ipcr_new_by_sub4.csv
- WWP_Amount_ipcr_new_by_T_c_i.csv          (series x IPCR_sub4 totals, after filters)
- WWP_Amount_ipcr.csv                        (series x IPCR x Year wide table, after filters)
- WWP_Amount_ipcr_new_summary.csv            (summary stats; +4 phase lines for Patent)
- WWP_Amount_ipcr_raw_summary.csv            (raw, unfiltered summary; +4 phase lines for Patent)

NEW in this version
-------------------
- Append 4 phase rows to RAW and NEW summaries (Patent totals only):
    <=1859, 1860-1969, 1970-2010, 2011-2024.
- Append 4 phase columns to each detailed output:
    WWP_Amount_ipcr_new_by_Country.csv
    WWP_Amount_ipcr_new_by_sub3.csv
    WWP_Amount_ipcr_new_by_sub4.csv
    WWP_Amount_ipcr_new_by_T_c_i.csv
  Column names: N_le_1859, N_1860_1969, N_1970_2010, N_2011_2024

FILTERS (RAW vs NEW)
--------------------
RAW summary:
  - No country exclusion and no year ceiling.
  - Parses IPC from column ipcr_kind when it contains "/" and at least two parts.

NEW stage (used for pj08 outputs):
  1) Row-level filters:
     - Exclude countries in EXCLUDE_COUNTRIES.
     - Keep rows with patent_year as an int and patent_year <= NEW_YEAR_CUTOFF (2024).
  2) Series expansion by political transitions (mapping code/year -> series id).
  3) Series head filter (TOP 75% by total rows) -> selected_series.
  4) T_c_i head filter (TOP 50% pairs by totals) applies only to:
     - WWP_Amount_ipcr_new_by_T_c_i.csv
     - WWP_Amount_ipcr.csv
"""

import os
import re
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Tuple, List, Set

import pandas as pd
from tqdm import tqdm

# ========================== Configuration ==========================
PJ08_OUTPUT_DIR = "/data01/rong_dataset/Result/pj08/"
RAW_INPUT_DIR = "/data01/Rong_backup/WWP_Merged/WWP_Merged_Data/"

NUM_WORKERS = min(70, os.cpu_count() or 1)
CSV_CHUNKSIZE = 250_000

# Metadata paths for abbreviations
IPCR_GROUP_META_CSV = "/data01/rong_dataset/Result/pj08/WWP_ipcr_group_name.csv"       # IPCRgroup, IPCRname, IPC Group Abbreviation
IPCR_SUB3_META_CSV  = "/data01/rong_dataset/Result/pj08/WWP_ipcr_subclass_name.csv"    # IPCR_sub3, IPCRname, IPC Subclass Abbreviation

COUNTRY_NAME_CSV = "/data01/rong_dataset/Result/pj08/pj08_old/WWP_Amount_ipcr_by_Country_1.csv"

EXCLUDE_COUNTRIES = {"AP", "EA", "EP", "GC", "OA", "ZA", "WO"}
NEW_YEAR_CUTOFF = 2024
TOP_RATIO = 0.75
TCI_TOP_RATIO = 0.5  # top 50% for T_c_i pairs

# Phase definitions
PHASES = {
    "N_le_1859":        (-10**9, 1859),
    "N_1860_1969":      (1860, 1969),
    "N_1970_2010":      (1970, 2010),
    "N_2011_2024":      (2011, 2024),
}
PHASE_ORDER = ["N_le_1859", "N_1860_1969", "N_1970_2010", "N_2011_2024"]

# ====================== Pairwise series definitions ======================
TRANSITION = {"SU": 1991, "CS": 1993, "YU": 2003}
SU_SUCCESSORS = ["RU", "UA", "EE", "LV", "LT", "MD", "GE"]
CS_SUCCESSORS = ["CZ", "SK"]
YU_SUCCESSORS = ["RS", "ME", "HR", "SI", "MK", "BA"]

DEFAULT_NAMES = {
    "DE": "Germany", "DD": "Germany",
    "SU": "Soviet Union", "RU": "Russia", "UA": "Ukraine", "EE": "Estonia",
    "LV": "Latvia", "LT": "Lithuania", "MD": "Moldova", "GE": "Georgia",
    "CS": "Czechoslovakia", "CZ": "Czech Republic", "SK": "Slovakia",
    "YU": "Yugoslavia", "RS": "Serbia", "ME": "Montenegro",
    "HR": "Croatia", "SI": "Slovenia", "MK": "North Macedonia", "BA": "Bosnia and Herzegovina",
}
PREDECESSOR_UNTIL = {"SU": 1991, "CS": 1993, "YU": 2003}
SUCCESSOR_START = {
    "RU": 1991, "UA": 1991, "EE": 1991, "LV": 1991, "LT": 1991, "MD": 1991, "GE": 1991,
    "CZ": 1993, "SK": 1993,
    "RS": 2003, "ME": 2003, "HR": 2003, "SI": 2003, "MK": 2003, "BA": 2003,
}

# ============================ Helpers ==============================
def read_with_fallback(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "ISO-8859-1", "latin1", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols)
        except Exception:
            continue
    return pd.read_csv(path, usecols=usecols)

def load_country_names() -> Dict[str, str]:
    name_map = dict(DEFAULT_NAMES)
    if os.path.exists(COUNTRY_NAME_CSV):
        try:
            m = pd.read_csv(COUNTRY_NAME_CSV, usecols=["Country", "Name"])
            m = m.dropna(subset=["Country"]).drop_duplicates("Country")
            for c, nm in zip(m["Country"].astype(str), m["Name"].astype(str)):
                if nm and str(nm).strip():
                    name_map[c] = nm
        except Exception:
            pass
    return name_map

def normalize_country_for_new(country: str) -> Optional[str]:
    if pd.isna(country):
        return None
    code = str(country).strip()
    if not code or code in EXCLUDE_COUNTRIES:
        return None
    return code

def get_series_ids_for_record(code: str, year: int) -> Set[str]:
    if code in {"DE", "DD"}:
        return {"DE"}
    if code == "SU":
        if year is not None and year <= TRANSITION["SU"]:
            return {f"{s}/SU" for s in SU_SUCCESSORS}
        return set()
    if code in SU_SUCCESSORS:
        return {f"{code}/SU"}
    if code == "CS":
        if year is not None and year <= TRANSITION["CS"]:
            return {f"CS/{s}" for s in CS_SUCCESSORS}
        return set()
    if code in CS_SUCCESSORS:
        return {f"CS/{code}"}
    if code == "YU":
        if year is not None and year <= TRANSITION["YU"]:
            return {f"YU/{s}" for s in YU_SUCCESSORS}
        return set()
    if code in YU_SUCCESSORS:
        return {f"YU/{code}"}
    return {code}

def series_display_name(sid: str, name_map: Dict[str, str]) -> str:
    if "/" in sid:
        a, b = sid.split("/", 1)
        return f"{name_map.get(a, a)}/{name_map.get(b, b)}"
    return name_map.get(sid, sid)

def country_display_name(code: str, name_map: Dict[str, str]) -> str:
    if code in {"DE", "DD"}:
        return "Germany"
    if code in PREDECESSOR_UNTIL:
        return f"{name_map.get(code, code)} (Until {PREDECESSOR_UNTIL[code]})"
    if code in SUCCESSOR_START:
        return f"{name_map.get(code, code)} (Started in {SUCCESSOR_START[code]})"
    return name_map.get(code, code)

def normalize_ipcr_raw(code: str) -> str:
    if pd.isna(code):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()

def ipcr_sub3(code: str) -> str:
    s = normalize_ipcr_raw(code)
    return s[:4] if len(s) >= 4 else ""

def ipcr_sub4(code: str) -> str:
    s = normalize_ipcr_raw(code)
    return s[:8] if len(s) >= 8 else ""

def extract_class_codes(cell: str) -> set:
    if pd.isna(cell):
        return set()
    codes = set()
    s = str(cell)
    for raw_code in s.split(";"):
        raw_code = raw_code.strip()
        if not raw_code or "/" not in raw_code:
            continue
        try:
            parts = raw_code.split()
            if len(parts) < 2:
                continue
            subclass = parts[0]
            group = parts[1].split("/")[0]
            group = group.zfill(4)
            norm_code = f"{subclass}{group}"
            codes.add(norm_code)
        except Exception:
            continue
    return codes

def year_to_phase_key(year: int) -> Optional[str]:
    if year is None or pd.isna(year):
        return None
    try:
        y = int(year)
    except Exception:
        return None
    for k in PHASE_ORDER:
        lo, hi = PHASES[k]
        if lo <= y <= hi:
            return k
    return None

# ================== Worker: read a CSV in chunks ==================
def _process_one_raw_csv(fp: str):
    """
    Return:
      total_rows,
      raw_country (Counter[code]),
      raw_year (Counter[year]),
      raw_sub3 (Counter[sub3]),
      raw_sub4 (Counter[sub4]),
      raw_country_sub4 (Counter[(code, sub4)]),
      new_row_counts_by_year (Counter[(code, year)]),
      new_sub3_by_year (Counter[(code, year, sub3)]),
      new_sub4_by_year (Counter[(code, year, sub4)])
    """
    try:
        header_cols = pd.read_csv(fp, nrows=0).columns.tolist()
        if not header_cols:
            return (0, Counter(), Counter(), Counter(), Counter(), Counter(),
                    Counter(), Counter(), Counter())

        needed = ["Country", "ipcr_kind", "patent_year"]
        present = [c for c in needed if c in header_cols]
        count_col = present[0] if present else header_cols[0]
        usecols = present if present else [count_col]

        dtype = {}
        if "Country" in usecols:
            dtype["Country"] = "string"
        if "ipcr_kind" in usecols:
            dtype["ipcr_kind"] = "string"
        if "patent_year" in usecols:
            dtype["patent_year"] = "Int64"

        raw_country = Counter()
        raw_year = Counter()
        raw_sub3 = Counter()
        raw_sub4 = Counter()
        raw_country_sub4 = Counter()

        new_row_counts_by_year = Counter()
        new_sub3_by_year = Counter()
        new_sub4_by_year = Counter()

        total_rows = 0

        for chunk in pd.read_csv(
            fp,
            usecols=usecols,
            dtype=dtype if dtype else None,
            chunksize=CSV_CHUNKSIZE,
            engine="c",
            low_memory=False,
        ):
            total_rows += len(chunk)

            # RAW stats
            if "Country" in chunk.columns:
                raw_country.update(chunk["Country"].dropna().astype(str).value_counts().to_dict())
            if "patent_year" in chunk.columns:
                yrs = pd.to_numeric(chunk["patent_year"], errors="coerce").dropna().astype(int)
                raw_year.update(yrs.value_counts().to_dict())

            cols = chunk.columns
            has_country = "Country" in cols
            has_year = "patent_year" in cols
            has_ipcr = "ipcr_kind" in cols

            if not (has_country or has_year or has_ipcr):
                continue

            for row in chunk.itertuples(index=False, name=None):
                row_dict = dict(zip(cols, row))

                # RAW IPCR sets per row
                codes = extract_class_codes(row_dict["ipcr_kind"]) if has_ipcr else set()
                if codes:
                    sub4_set = {ipcr_sub4(c) for c in codes if ipcr_sub4(c)}
                    sub3_set = {ipcr_sub3(c) for c in codes if ipcr_sub3(c)}
                    raw_sub4.update(sub4_set)
                    raw_sub3.update(sub3_set)
                else:
                    sub3_set = set()
                    sub4_set = set()

                # RAW (country, sub4)
                if has_country and sub4_set:
                    c_raw0 = row_dict["Country"]
                    c_raw = str(c_raw0) if not pd.isna(c_raw0) else None
                    if c_raw:
                        for s4 in sub4_set:
                            raw_country_sub4[(c_raw, s4)] += 1

                # NEW filter
                code = normalize_country_for_new(row_dict["Country"]) if has_country else None
                if code is None:
                    continue

                y_val = None
                if has_year:
                    try:
                        y_val = int(row_dict["patent_year"])
                    except Exception:
                        y_val = None
                if (y_val is None) or (y_val > NEW_YEAR_CUTOFF):
                    continue

                new_row_counts_by_year[(code, y_val)] += 1
                for s3 in sub3_set:
                    new_sub3_by_year[(code, y_val, s3)] += 1
                for s4 in sub4_set:
                    new_sub4_by_year[(code, y_val, s4)] += 1

        return (total_rows, raw_country, raw_year, raw_sub3, raw_sub4, raw_country_sub4,
                new_row_counts_by_year, new_sub3_by_year, new_sub4_by_year)

    except Exception:
        return (0, Counter(), Counter(), Counter(), Counter(), Counter(),
                Counter(), Counter(), Counter())

# ======================= Build outputs (pj08) =======================
def _build_summary_table(rows: List[Tuple[str, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Type", "Amount", "Range [min,max]", "Median", "Mean"])

def _series_stats(s: pd.Series) -> Tuple[int, int, float, float]:
    if s is None or s.empty:
        return 0, 0, 0.0, 0.0
    return int(s.min()), int(s.max()), float(s.median()), float(s.mean())

def _phase_totals_from_year_counter(counter_year: Counter) -> Dict[str, int]:
    out = {k: 0 for k in PHASE_ORDER}
    for y, cnt in counter_year.items():
        k = year_to_phase_key(y)
        if k:
            out[k] += int(cnt)
    return out

def _phase_totals_from_df(df: pd.DataFrame, year_col: str, cnt_col: str) -> Dict[str, int]:
    out = {k: 0 for k in PHASE_ORDER}
    if df is None or df.empty:
        return out
    for k in PHASE_ORDER:
        lo, hi = PHASES[k]
        s = df.loc[(df[year_col] >= lo) & (df[year_col] <= hi), cnt_col]
        out[k] = int(s.sum()) if not s.empty else 0
    return out

def process_pj08_from_raw(raw_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_country_names()
    # placeholders for T_c_i stats
    tci_min, tci_max = 0, 0
    tci_median, tci_mean = 0.0, 0.0

    csv_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
    total_files = len(csv_files)

    # RAW accumulators
    total_rows_raw = 0
    raw_country = Counter()
    raw_year = Counter()
    raw_sub3 = Counter()
    raw_sub4 = Counter()
    raw_country_sub4 = Counter()

    # NEW basis (code, year) keyed accumulators
    new_row_counts_by_year = Counter()         # (code, year) -> count
    new_sub3_by_year = Counter()               # (code, year, sub3) -> count
    new_sub4_by_year = Counter()               # (code, year, sub4) -> count

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(_process_one_raw_csv, fp) for fp in csv_files]
        for fut in tqdm(as_completed(futures), total=total_files, desc="Processing raw CSVs"):
            (nrows, c_raw, y_raw, s3_raw, s4_raw, cs4_raw,
             rows_by_year, s3_by_year, s4_by_year) = fut.result()

            total_rows_raw += nrows
            raw_country.update(c_raw)
            raw_year.update(y_raw)
            raw_sub3.update(s3_raw)
            raw_sub4.update(s4_raw)
            raw_country_sub4.update(cs4_raw)

            new_row_counts_by_year.update(rows_by_year)
            new_sub3_by_year.update(s3_by_year)
            new_sub4_by_year.update(s4_by_year)

    # ----------------------- RAW SUMMARY -----------------------
    country_series_raw = pd.Series(raw_country, dtype="int64") if raw_country else pd.Series(dtype="int64")
    sub3_series_raw = pd.Series(raw_sub3, dtype="int64") if raw_sub3 else pd.Series(dtype="int64")
    sub4_series_raw = pd.Series(raw_sub4, dtype="int64") if raw_sub4 else pd.Series(dtype="int64")
    year_series_raw = pd.Series(raw_year, dtype="int64") if raw_year else pd.Series(dtype="int64")
    tc_i_series_raw = pd.Series(raw_country_sub4, dtype="int64") if raw_country_sub4 else pd.Series(dtype="int64")

    patent_amt_raw = int(total_rows_raw)
    country_amt_raw = int(len(country_series_raw))
    sub3_amt_raw = int(len(sub3_series_raw))
    sub4_amt_raw = int(len(sub4_series_raw))
    year_amt_raw = int(len(year_series_raw))
    tc_i_amt_raw = int(len(tc_i_series_raw))

    cmin, cmax, cmed, cmean = _series_stats(country_series_raw)
    s3min, s3max, s3med, s3mean = _series_stats(sub3_series_raw)
    s4min, s4max, s4med, s4mean = _series_stats(sub4_series_raw)
    ycnt_min_raw, ycnt_max_raw, ycnt_med_raw, ycnt_mean_raw = _series_stats(year_series_raw)
    tmin_raw, tmax_raw, tmed_raw, tmean_raw = _series_stats(tc_i_series_raw)

    if not year_series_raw.empty:
        span_min_raw = int(min(year_series_raw.index))
        span_max_raw = int(max(year_series_raw.index))
        year_span_label_raw = f"{year_amt_raw} ({span_min_raw}-{span_max_raw})"
    else:
        year_span_label_raw = "0 (-)"

    raw_rows = [
        ["Patent", str(patent_amt_raw), "-", "-", "-"],
        ["Country/Region", str(country_amt_raw), f"[{cmin}, {cmax}]", f"{cmed}", f"{cmean}"],
        ["IPC Subclass (sub3)", str(sub3_amt_raw), f"[{s3min}, {s3max}]", f"{s3med}", f"{s3mean}"],
        ["IPC Group (sub4)", str(sub4_amt_raw), f"[{s4min}, {s4max}]", f"{s4med}", f"{s4mean}"],
        ["Year Span", year_span_label_raw, f"[{ycnt_min_raw}, {ycnt_max_raw}]", f"{ycnt_med_raw}", f"{ycnt_mean_raw}"],
        ["T_c_i", str(tc_i_amt_raw), f"[{tmin_raw}, {tmax_raw}]", f"{tmed_raw}", f"{tmean_raw}"],
    ]

    raw_phase_totals = _phase_totals_from_year_counter(raw_year)
    raw_rows.extend([
        [f"Patent ({label.replace('N_', '').replace('_', ' ').replace('le', '<=')})", str(raw_phase_totals[label]), "-", "-", "-"]
        for label in PHASE_ORDER
    ])

    pd.DataFrame(raw_rows, columns=["Type", "Amount", "Range [min,max]", "Median", "Mean"])\
      .to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_raw_summary.csv"), index=False)

    # --------------------- NEW: series selection ---------------------
    series_total_rows = Counter()                 # sid -> total (selected later)
    series_year_counts_by_series = Counter()      # (sid, year) -> count
    sel_series_sub3 = Counter()
    sel_series_sub4 = Counter()
    series_tci_counts = Counter()

    for (code, y), cnt in new_row_counts_by_year.items():
        for sid in get_series_ids_for_record(code, y):
            series_total_rows[sid] += cnt
            series_year_counts_by_series[(sid, y)] += cnt

    for (code, y, s3), cnt in new_sub3_by_year.items():
        for sid in get_series_ids_for_record(code, y):
            sel_series_sub3[s3] += cnt

    for (code, y, s4), cnt in new_sub4_by_year.items():
        for sid in get_series_ids_for_record(code, y):
            sel_series_sub4[s4] += cnt
            series_tci_counts[(sid, s4)] += cnt

    if not series_total_rows:
        for fn in [
            "WWP_Amount_ipcr_new_by_Country.csv",
            "WWP_Amount_ipcr_new_by_sub3.csv",
            "WWP_Amount_ipcr_new_by_sub4.csv",
            "WWP_Amount_ipcr_new_by_T_c_i.csv",
            "WWP_Amount_ipcr.csv",
            "WWP_Amount_ipcr_new_summary.csv",
        ]:
            pd.DataFrame().to_csv(os.path.join(output_dir, fn), index=False)
        return

    sorted_series = [sid for sid, _ in series_total_rows.most_common()]
    k = max(1, math.ceil(TOP_RATIO * len(sorted_series)))
    selected_series: Set[str] = set(sorted_series[:k])

    # per-year totals for selected series (for Year Span stats)
    sel_series_year_counts = Counter()  # year -> total count (selected series only)
    for (sid, y), cnt in series_year_counts_by_series.items():
        if sid in selected_series:
            sel_series_year_counts[y] += cnt

    # -------- country-year (selected_series only) --------
    country_year_selected = Counter()  # (country2, year) -> cnt
    for (code, y), cnt in new_row_counts_by_year.items():
        sids = get_series_ids_for_record(code, y)
        if not sids.intersection(selected_series):
            continue
        code2 = "DE" if code == "DD" else code
        country_year_selected[(code2, y)] += cnt

    # -------- sub3-year (selected_series only) --------
    sub3_year_selected = Counter()
    for (code, y, s3), cnt in new_sub3_by_year.items():
        for sid in get_series_ids_for_record(code, y):
            if sid in selected_series:
                sub3_year_selected[(s3, y)] += cnt

    # -------- sub4-year (selected_series only) --------
    rows_sub4_all = []
    for (code, y, s4), cnt in new_sub4_by_year.items():
        sids = get_series_ids_for_record(code, y)
        for sid in sids:
            if sid in selected_series:
                rows_sub4_all.append({"sids": sid, "sub4": s4, "year": int(y), "cnt": int(cnt)})
    df_sub4_all = pd.DataFrame(rows_sub4_all) if rows_sub4_all else pd.DataFrame(columns=["sids", "sub4", "year", "cnt"])

    # -------- T_c_i top-50% filtering --------
    if not df_sub4_all.empty:
        pair_totals_all = (
            df_sub4_all.groupby(["sids", "sub4"], as_index=False)["cnt"].sum()
            .sort_values("cnt", ascending=False)
        )
        num_pairs = len(pair_totals_all)
        if num_pairs > 0:
            k_pairs = max(1, math.ceil(TCI_TOP_RATIO * num_pairs))
            top_pairs_df = pair_totals_all.iloc[:k_pairs, :]
            selected_pairs = set(map(tuple, top_pairs_df[["sids", "sub4"]].to_records(index=False)))
            df_sub4_all["__pair"] = list(zip(df_sub4_all["sids"], df_sub4_all["sub4"]))
            df_sub4_top = df_sub4_all[df_sub4_all["__pair"].isin(selected_pairs)].drop(columns=["__pair"])
        else:
            df_sub4_top = df_sub4_all.copy()
    else:
        df_sub4_top = df_sub4_all.copy()

    # ---------------- by_Country (selected_series only) ----------------
    single_country_counts = Counter()
    for (code2, y), cnt in country_year_selected.items():
        single_country_counts[code2] += cnt

    by_country_df = (
        pd.Series(single_country_counts, dtype="int64")
        .sort_values(ascending=False)
        .reset_index()
    )
    by_country_df.columns = ["Country", "Amount"]
    name_map = load_country_names()
    by_country_df["Name"] = by_country_df["Country"].map(lambda c: country_display_name(c, name_map))

    if country_year_selected:
        cy_rows = [{"Country": c, "Year": y, "Count": int(cnt)} for (c, y), cnt in country_year_selected.items()]
        cy_df = pd.DataFrame(cy_rows)
    else:
        cy_df = pd.DataFrame(columns=["Country", "Year", "Count"])

    for ph in PHASE_ORDER:
        lo, hi = PHASES[ph]
        if not cy_df.empty:
            ph_counts = (
                cy_df.loc[(cy_df["Year"] >= lo) & (cy_df["Year"] <= hi)]
                    .groupby("Country", as_index=False)["Count"].sum()
            )
            by_country_df = by_country_df.merge(ph_counts.rename(columns={"Count": ph}),
                                                on="Country", how="left")
        else:
            by_country_df[ph] = 0
        by_country_df[ph] = by_country_df[ph].fillna(0).astype(int)

    by_country_df = by_country_df[["Country", "Name", "Amount"] + PHASE_ORDER]
    by_country_df.to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_new_by_Country.csv"), index=False)

    # ---------------- by_sub3 (selected_series only) ----------------
    sel_series_sub3_filtered = Counter()
    for (s3, y), cnt in sub3_year_selected.items():
        sel_series_sub3_filtered[s3] += cnt

    by_sub3_df = (
        pd.Series(sel_series_sub3_filtered, dtype="int64")
        .sort_values(ascending=False)
        .reset_index()
    )
    by_sub3_df.columns = ["IPCR_sub3", "Amount"]

    if sub3_year_selected:
        s3_rows = [{"IPCR_sub3": s3, "Year": y, "Count": int(cnt)} for (s3, y), cnt in sub3_year_selected.items()]
        s3_df = pd.DataFrame(s3_rows)
    else:
        s3_df = pd.DataFrame(columns=["IPCR_sub3", "Year", "Count"])

    for ph in PHASE_ORDER:
        lo, hi = PHASES[ph]
        if not s3_df.empty:
            ph_counts = (
                s3_df.loc[(s3_df["Year"] >= lo) & (s3_df["Year"] <= hi)]
                     .groupby("IPCR_sub3", as_index=False)["Count"].sum()
            )
            by_sub3_df = by_sub3_df.merge(
                ph_counts.rename(columns={"Count": ph}),
                on="IPCR_sub3", how="left"
            )
        else:
            by_sub3_df[ph] = 0
        by_sub3_df[ph] = by_sub3_df[ph].fillna(0).astype(int)

    # merge subclass abbreviation
    if os.path.exists(IPCR_SUB3_META_CSV):
        try:
            meta3 = read_with_fallback(IPCR_SUB3_META_CSV, usecols=["IPCR_sub3", "IPCRname", "IPC Subclass Abbreviation"])
            meta3 = meta3.drop_duplicates(subset=["IPCR_sub3"])
            by_sub3_df = by_sub3_df.merge(
                meta3[["IPCR_sub3", "IPC Subclass Abbreviation"]],
                on="IPCR_sub3",
                how="left"
            )
        except Exception:
            by_sub3_df["IPC Subclass Abbreviation"] = ""
    else:
        by_sub3_df["IPC Subclass Abbreviation"] = ""

    by_sub3_df.to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_new_by_sub3.csv"), index=False)

    # ---------------- by_sub4 (selected_series only) ----------------
    if not df_sub4_all.empty:
        sub4_totals = df_sub4_all.groupby("sub4", as_index=False)["cnt"].sum()\
                                 .rename(columns={"sub4": "IPCR_sub4", "cnt": "Amount"})
        by_sub4_df = sub4_totals.sort_values("Amount", ascending=False).reset_index(drop=True)
    else:
        by_sub4_df = pd.DataFrame(columns=["IPCR_sub4", "Amount"])

    for ph in PHASE_ORDER:
        lo, hi = PHASES[ph]
        if not df_sub4_all.empty:
            ph_counts = (
                df_sub4_all.loc[(df_sub4_all["year"] >= lo) & (df_sub4_all["year"] <= hi)]
                    .groupby("sub4", as_index=False)["cnt"].sum()
                    .rename(columns={"sub4": "IPCR_sub4", "cnt": ph})
            )
            by_sub4_df = by_sub4_df.merge(ph_counts, on="IPCR_sub4", how="left")
        else:
            by_sub4_df[ph] = 0
        by_sub4_df[ph] = by_sub4_df[ph].fillna(0).astype(int)

    # merge group abbreviation (IPCR_sub4 -> IPC Group Abbreviation)
    if os.path.exists(IPCR_GROUP_META_CSV):
        try:
            meta4 = read_with_fallback(IPCR_GROUP_META_CSV, usecols=["IPCRgroup", "IPCRname", "IPC Group Abbreviation"])
            meta4 = meta4.drop_duplicates(subset=["IPCRgroup"])
            by_sub4_df = by_sub4_df.merge(
                meta4[["IPCRgroup", "IPC Group Abbreviation"]].rename(columns={"IPCRgroup": "IPCR_sub4"}),
                on="IPCR_sub4",
                how="left"
            )
        except Exception:
            by_sub4_df["IPC Group Abbreviation"] = ""
    else:
        by_sub4_df["IPC Group Abbreviation"] = ""

    by_sub4_df.to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_new_by_sub4.csv"), index=False)

    # ---------------- T_c_i table (selected_series + top-50% pairs) ----------------
    if not df_sub4_top.empty:
        tci_df = (
            df_sub4_top.groupby(["sids", "sub4"], as_index=False)["cnt"].sum()
            .rename(columns={"sids": "Country", "sub4": "IPCR_sub4", "cnt": "Amount"})
        )
        tci_df["Name"] = tci_df["Country"].map(lambda s: series_display_name(s, name_map))

        for ph in PHASE_ORDER:
            lo, hi = PHASES[ph]
            if not df_sub4_top.empty:
                ph_counts = (
                    df_sub4_top.loc[(df_sub4_top["year"] >= lo) & (df_sub4_top["year"] <= hi)]
                        .groupby(["sids", "sub4"], as_index=False)["cnt"].sum()
                        .rename(columns={"sids": "Country", "sub4": "IPCR_sub4", "cnt": ph})
                )
                tci_df = tci_df.merge(ph_counts, on=["Country", "IPCR_sub4"], how="left")
            else:
                tci_df[ph] = 0
            tci_df[ph] = tci_df[ph].fillna(0).astype(int)

        # merge group abbreviation into T_c_i as well
        if os.path.exists(IPCR_GROUP_META_CSV):
            try:
                meta4_tci = read_with_fallback(IPCR_GROUP_META_CSV, usecols=["IPCRgroup", "IPCRname", "IPC Group Abbreviation"])
                meta4_tci = meta4_tci.drop_duplicates(subset=["IPCRgroup"])
                tci_df = tci_df.merge(
                    meta4_tci[["IPCRgroup", "IPC Group Abbreviation"]].rename(columns={"IPCRgroup": "IPCR_sub4"}),
                    on="IPCR_sub4",
                    how="left"
                )
            except Exception:
                tci_df["IPC Group Abbreviation"] = ""
        else:
            tci_df["IPC Group Abbreviation"] = ""

        tci_df = tci_df[["Country", "Name", "IPCR_sub4", "IPC Group Abbreviation", "Amount"] + PHASE_ORDER]

        # T_c_i stats for NEW summary (Amount over selected pairs)
        tci_min = tci_max = 0
        tci_median = tci_mean = 0.0
        if not tci_df.empty and "Amount" in tci_df.columns:
            _s = pd.to_numeric(tci_df["Amount"], errors="coerce").dropna().astype(int)
            if not _s.empty:
                tci_min = int(_s.min())
                tci_max = int(_s.max())
                tci_median = float(_s.median())
                tci_mean = float(_s.mean())

        tci_df.to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_new_by_T_c_i.csv"), index=False)
    else:
        cols = ["Country", "Name", "IPCR_sub4", "IPC Group Abbreviation", "Amount"] + PHASE_ORDER
        pd.DataFrame(columns=cols).to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_new_by_T_c_i.csv"), index=False)

    # ---------------- WWP_Amount_ipcr.csv (selected_series + top-50% pairs) ----------------
    if not df_sub4_top.empty:
        ipcr_year_df = (
            df_sub4_top.groupby(["sids", "sub4", "year"], as_index=False)["cnt"].sum()
            .rename(columns={"sids": "Country", "sub4": "IPCR", "cnt": "Count", "year": "Year"})
        )
        if not ipcr_year_df.empty:
            pivot_df = ipcr_year_df.pivot_table(
                index=["Country", "IPCR"],
                columns="Year",
                values="Count",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()
            pivot_df.columns.name = None
            fixed_cols = ["Country", "IPCR"]
            year_cols = sorted([c for c in pivot_df.columns if c not in fixed_cols])
            pivot_df = pivot_df[fixed_cols + year_cols]
            pivot_df = pivot_df.sort_values(by=["Country", "IPCR"], kind="mergesort")
            pivot_df.to_csv(os.path.join(output_dir, "WWP_Amount_ipcr.csv"), index=False)
            pivot_rows_count_for_summary = int(len(pivot_df))
        else:
            pd.DataFrame(columns=["Country", "IPCR"]).to_csv(os.path.join(output_dir, "WWP_Amount_ipcr.csv"), index=False)
            pivot_rows_count_for_summary = 0
    else:
        pd.DataFrame(columns=["Country", "IPCR"]).to_csv(os.path.join(output_dir, "WWP_Amount_ipcr.csv"), index=False)
        pivot_rows_count_for_summary = 0

    # ---------------- NEW Summary (series-level + phase lines) ----------------
    # series-level totals (selected series only)
    series_totals_selected = Counter()
    for (code, y), cnt in new_row_counts_by_year.items():
        for sid in get_series_ids_for_record(code, y):
            if sid in selected_series:
                series_totals_selected[sid] += cnt

    patent_amt_new = int(sum(series_totals_selected.values()))
    country_amt_new = int(len(selected_series))

    # sub3 totals (selected series only)
    sel_sub3_counts = Counter()
    for (s3, y), cnt in sub3_year_selected.items():
        sel_sub3_counts[s3] += cnt
    sub3_amt_new = int(len(sel_sub3_counts))

    # sub4 totals (selected series only)
    if not df_sub4_all.empty:
        _sub4_totals_series = df_sub4_all.groupby("sub4")["cnt"].sum()
        sub4_amt_new = int(_sub4_totals_series.shape[0])
    else:
        _sub4_totals_series = pd.Series(dtype="int64")
        sub4_amt_new = 0

    # year stats (selected series only)
    year_amt_new = int(len(sel_series_year_counts))

    # distributions
    series_totals_series = pd.Series(series_totals_selected) if series_totals_selected else pd.Series(dtype="int64")
    cmin_f, cmax_f, cmed_f, cmean_f = _series_stats(series_totals_series)

    sub3_series_filtered = pd.Series(sel_sub3_counts, dtype="int64") if sel_sub3_counts else pd.Series(dtype="int64")
    s3min_f, s3max_f, s3med_f, s3mean_f = _series_stats(sub3_series_filtered)

    sub4_series_filtered = pd.to_numeric(_sub4_totals_series, errors="coerce").dropna().astype(int) if not _sub4_totals_series.empty else pd.Series(dtype="int64")
    s4min_f, s4max_f, s4med_f, s4mean_f = _series_stats(sub4_series_filtered)

    year_series_filtered = pd.Series(sel_series_year_counts, dtype="int64") if sel_series_year_counts else pd.Series(dtype="int64")
    ycnt_min_new, ycnt_max_new, ycnt_med_new, ycnt_mean_new = _series_stats(year_series_filtered)

    # row count of WWP_Amount_ipcr.csv
    tci_amount_for_summary = pivot_rows_count_for_summary

    # year span label
    if not year_series_filtered.empty:
        span_min_new = int(min(year_series_filtered.index))
        year_span_label_new = f"{year_amt_new} ({span_min_new}-{NEW_YEAR_CUTOFF})"
    else:
        year_span_label_new = "0 (-)"

    # --- Phase totals aligned with Patent (Series-level) ---
    # Build per-year totals on the SAME base as `patent_amt_new` (selected series counts)
    if sel_series_year_counts:
        _per_year_series = pd.DataFrame(
            [{"Year": y, "Total": int(cnt)} for y, cnt in sel_series_year_counts.items()]
        )
    else:
        _per_year_series = pd.DataFrame(columns=["Year", "Total"])

    phase_totals = {k: 0 for k in PHASE_ORDER}
    phase_stats  = {k: ("-", "-", "-") for k in PHASE_ORDER}  # (range_str, median, mean)

    for label in PHASE_ORDER:
        lo, hi = PHASES[label]
        if not _per_year_series.empty:
            s = _per_year_series.loc[
                (_per_year_series["Year"] >= lo) & (_per_year_series["Year"] <= hi),
                "Total",
            ]
            total = int(s.sum()) if not s.empty else 0
            rng, med, mean = "-", "-", "-"
            if not s.empty:
                rng  = f"[{int(s.min())}, {int(s.max())}]"
                med  = f"{float(s.median())}"
                mean = f"{float(s.mean())}"
            phase_totals[label] = total
            phase_stats[label]  = (rng, med, mean)

    # Compose NEW summary rows
    new_rows = [
        ["Patent (Series-level)", str(patent_amt_new), "-", "-", "-"],
    ]
    for label in PHASE_ORDER:
        rng, med, mean = phase_stats[label]
        new_rows.append([
            f"Patent ({label.replace('N_', '').replace('_', ' ').replace('le', '<=')})",
            str(phase_totals[label]),
            rng, med, mean,
        ])

    # keep the rest unchanged
    new_rows.extend([
        ["Country/Region (Series)", str(country_amt_new), f"[{cmin_f}, {cmax_f}]", f"{cmed_f}", f"{cmean_f}"],
        ["IPC Subclass (sub3)", str(sub3_amt_new), f"[{s3min_f}, {s3max_f}]", f"{s3med_f}", f"{s3mean_f}"],
        ["IPC Group (sub4)", str(sub4_amt_new), f"[{s4min_f}, {s4max_f}]", f"{s4med_f}", f"{s4mean_f}"],
        ["Year Span", year_span_label_new, f"[{ycnt_min_new}, {ycnt_max_new}]", f"{ycnt_med_new}", f"{ycnt_mean_new}"],
        [
            "T_c_i",
            str(tci_amount_for_summary),
            f"[{tci_min}, {tci_max}]" if tci_amount_for_summary > 0 else "-",
            f"{tci_median}" if tci_amount_for_summary > 0 else "-",
            f"{tci_mean}" if tci_amount_for_summary > 0 else "-",
        ],
    ])

    _build_summary_table(new_rows).to_csv(os.path.join(output_dir, "WWP_Amount_ipcr_new_summary.csv"), index=False)


# ================================ Main =================================
if __name__ == "__main__":
    print(f"Starting pj08 (pairwise series) from RAW: {RAW_INPUT_DIR}, workers={NUM_WORKERS}, chunksize={CSV_CHUNKSIZE}")
    process_pj08_from_raw(RAW_INPUT_DIR, PJ08_OUTPUT_DIR)
