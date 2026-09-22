#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pioneer frequency analysis + 3x5 combined PDF figure (ASCII only).

Changes in this version
-----------------------
- IPC labels are rendered on a transparent overlay axes with high zorder (always on top).
- Country name overrides: TW -> Taiwan(China), HK -> Hong Kong(China),
  CS/SK -> Slovak, CS/CZ -> Czech; SU-era overrides kept.
- Reduced left-side blank space by shifting/widening content blocks inside each panel.

Outputs:
- pj08_Pioneer_fig.csv
- Fig/Pioneer_combined.pdf
"""

import os
import re
from typing import Optional, Dict, List, Tuple
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

OUT_CSV = os.path.join(BASE_DIR, "pj08_Pioneer_fig.csv")
FIG_DIR = os.path.join(BASE_DIR, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_FIG = os.path.join(FIG_DIR, "Pioneer_combined.pdf")

# Computational periods
PERIODS: List[Tuple[str, int, int]] = [
    ("1870-1959", 1870, 1959),
    ("1960-2010", 1960, 2010),
    ("2011-2024", 2011, 2024),
]

# Column headers for the figure
PERIOD_TITLES = {
    "1870-1959": "1860-1969 - 2nd IR",
    "1960-2010": "1970-2010 - 3rd IR",
    "2011-2024": "2011-2024 - 4th IR",
}

# Totals column per period
PERIOD_TO_SUB3_AMOUNT_COL = {
    "1870-1959": "N_1860_1969",
    "1960-2010": "N_1970_2010",
    "2011-2024": "N_2011_2024",
}

# Exactly 15 targets (5 per period)
TARGETS: List[Tuple[str, str]] = [
    ("C07C", "1870-1959"),
    ("H04L", "1870-1959"),
    ("H10N", "1870-1959"),
    ("B29C", "1870-1959"),
    ("C23C", "1870-1959"),
    
    ("C07C", "1960-2010"),
    ("H04L", "1960-2010"),
    ("H10N", "1960-2010"),
    ("B29C", "1960-2010"),
    ("C23C", "1960-2010"),
    
    ("C07C", "2011-2024"),
    ("H04L", "2011-2024"),
    ("H10N", "2011-2024"),
    ("B29C", "2011-2024"),
    ("C23C", "2011-2024"),


]

PRODUCT_LINES = {
    "C07C": "Cornerstone of Aspirin, fuel, plastics industries, etc.", 
    "H01J": "fundamental technology of plasma displays, etc.",
    "G03B": "Cameras, Projectors, etc.",
    "B29C": "plastic bottles, pipes, medical consumables, etc.",
    "C10B": "Coke, coal-derived tar, coal gas, etc.",
    "G01M": "Car testing, Balancing machines, etc.",
    "H04L": "Internet, modems, satellite communications, etc.",
    "B62D": "Cars, Truck, etc.",
    "H01B": "Cables, Wires",
    "F24S": "Solar panels",
    "B60M": "Rail power lines",
    "B63B": "shipbuilding and naval shipbuilding industry",
    "B64D": "Aircraft equipment",
    "G05D": "Cruise control, thermostat, autopilot, etc.",
    "H03K": "Pulse circuits",
    "G06F": "Computers, CPUs, RAM, etc.",
    "H10K": "smartphone components, organic semiconductors, etc.",
    "H01M": "Batteries, Fuel cells, etc.",
    "F15D": "Jet propulsion systems, wind tunnel equipment, etc.",
    "B64U": "Newly added classification, Jan 2023",
    "C23C": "Applying liquids or other fluent materials to surfaces in general",
    "F16N": "Lubrication of specific apparatus or in particular processes",
    "B25D": "e.g. Portable percussive tools with fluid-pressure drive",
    "D06B": "e.g. Solvent-treatment of textile materials",
    "H10N": "e.g. Thermoelectric devices comprising a junction of dissimilar materials",
}


ALL_LISTED_SUB3 = [
    "C07C","H01J","G03B","B29C","C10B","G01M","H04L","B62D","H01B","F24S",
    "B60M","B63B","B64D","G05D","H03K","G06F","H10K","H01M", "F15D","B64U", 
    "C23C","F16N","B25D","D06B","H10N",
]

# Manual, shorter subclass names (overrides)
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
    "B64U": "Unmanned Aerial Vehicles, UAV",
    "C23C": "Coating metallic material",
    "F16N": "Lubricating",
    "B25D": "Percussive tools",
    "D06B": "Treating textile materials by liquids, gases, or vapours",
    "H10N": "Electric solid-state devices",
}

COUNTRY_EXCLUDE = {"AT", "IE"}

# Bar colors by period
PERIOD_BAR_COLOR = {
    "1870-1959": "#91c9f7",  # light blue
    "1960-2010": "#a6e3a1",  # light green
    "2011-2024": "#f7c99a",  # light orange
}

# Transition rules
TRANSITION: Dict[str, int] = {"SU": 1991, "CS": 1993, "YU": 2003}
PREDS = set(TRANSITION.keys())

# Position constants for IPC label overlay (tweak these to adjust distance)
IPC_LABEL_X = 0.01
IPC_LABEL_Y = 0.83

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

# ============== Country label mapping ==============
def build_country_name_map() -> Dict[str, str]:
    if not os.path.exists(COUNTRY_NAME_PATH):
        return {}
    df = read_csv_safely(COUNTRY_NAME_PATH)[["Country", "Name"]].dropna()
    m = dict(zip(df["Country"].astype(str), df["Name"].astype(str)))
    return m

def label_for_country(code: str, period_label: str, name_map: Dict[str, str]) -> str:
    code = str(code)

    # Specific code overrides first
    if code == "TW":
        return "Taiwan(China)"
    if code == "HK":
        return "Hong Kong(China)"
    if code == "CS/SK":
        return "Slovak"
    if code == "CS/CZ":
        return "Czech"

    # SU overrides by period
    if "SU" in code:
        if period_label == "1870-1959":
            return "Soviet Union"
        elif period_label == "1960-2010":
            return "Soviet Union/Russia"
        elif period_label == "2011-2024":
            return "Russia"

    # Fallback to provided name map or the code itself
    return name_map.get(code, code)

# ================= Drawing helpers =================
def draw_panel(ax, rec, period_label, name_map, bar_color):
    ax.set_axis_off()

    sub3 = rec.get("IPCsubclass", "") or ""
    # apply manual override if present
    abbrev = SUBCLASS_ABBREV_OVERRIDES.get(sub3, rec.get("IPC Subclass Abbreviation", "") or "")
    amt = rec.get("Period_total_amount", np.nan)
    grp = int(rec.get("IPCgroup_count", 0) or 0)

    # Left metrics block: shifted left and slightly widened
    left = ax.inset_axes([0.005, 0.24, 0.32, 0.56])
    left.set_axis_off()
    amt_txt = "NA" if not (isinstance(amt, (int, float)) and np.isfinite(amt)) else f"{int(amt):,}"
    left.text(0.00, 0.78, "Patent Count",   ha="left", va="center", fontsize=8, transform=left.transAxes)
    left.text(0.00, 0.58, amt_txt,          ha="left", va="center", fontsize=9, weight="bold", transform=left.transAxes)
    left.text(0.00, 0.30, "IPC group No.",  ha="left", va="center", fontsize=8, transform=left.transAxes)
    left.text(0.00, 0.10, f"{grp}",         ha="left", va="center", fontsize=9, weight="bold", transform=left.transAxes)

    # Right bar block: starts earlier and extends to the right edge to remove blank space
    right = ax.inset_axes([0.31, 0.10, 0.685, 0.80])
    right.spines["top"].set_visible(False)
    right.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.set_yticks([])

    countries_codes, counts = [], []
    for k in range(1, 6):
        c = rec.get(f"Top{k}_country", "")
        n = int(rec.get(f"Top{k}_count", 0) or 0)
        if c == "" and n == 0:
            continue
        countries_codes.append(c)
        counts.append(n)

    if not countries_codes:
        right.set_xticks([])
        right.set_title("No pioneers", fontsize=8)
        right.set_xlim(-0.5, 4.5)
    else:
        countries = [label_for_country(code, period_label, name_map) for code in countries_codes]
        step = 1.8
        x = np.arange(len(countries)) * step
        bar_width = 0.60
        right.bar(x, counts, width=bar_width, edgecolor="black", color=bar_color)

        right.set_xticks(x, countries, rotation=32, ha="right", fontsize=8)
        for tick in right.get_xticklabels():
            tick.set_fontstyle("italic")

        denom = float(grp) if grp > 0 else np.nan
        y_max = max(counts) if counts else 1
        for xi, ci in zip(x, counts):
            pct = (ci / denom) if np.isfinite(denom) else np.nan
            pct_txt = f"({int(round(pct*100))}%)" if np.isfinite(pct) else "(NA)"
            right.text(xi, ci + max(0.05, 0.01 * y_max), f"{ci}\n{pct_txt}",
                       ha="center", va="bottom", fontsize=8)

        right.set_ylim(0, y_max * 1.45)

    # Calculate sample standard deviation of the top-5 counts
    if len(counts) >= 2:
        std_val = np.std(counts, ddof=1)
        std_str = f"{std_val:.2f}"
    else:
        std_str = "NA"

    # Display IPC code + subclass name with standard deviation (SD)
    product_line = PRODUCT_LINES.get(sub3, "")
    ax.text(
        IPC_LABEL_X, IPC_LABEL_Y,
        f"{sub3}\n{abbrev} (SD={std_str})",
        ha="left", va="bottom", fontsize=9,
        transform=ax.transAxes, clip_on=False, zorder=20
    )

    # Product line (smaller font)
    ax.text(
        IPC_LABEL_X, IPC_LABEL_Y - 0.01,
        f"({product_line})",
        ha="left", va="top", fontsize=7,
        transform=ax.transAxes, clip_on=False, zorder=20
    )
# ================= Main =================
def main():
    # Load and combine Z
    zdfs = []
    for path in Z_SEG_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Z segment file not found: {path}")
        seg = read_csv_safely(path)
        seg.columns = [str(c).strip() for c in seg.columns]
        zdfs.append(seg)

    base_df = None
    for seg in zdfs:
        year_cols_seg = [c for c in seg.columns if YEAR_PATTERN.match(str(c))]
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

    zdf = base_df
    zdf.columns = [str(c).strip() for c in zdf.columns]
    if not {"Country", "IPCR"}.issubset(set(zdf.columns)):
        raise ValueError("Missing required columns in Z data.")

    year_cols = [c for c in zdf.columns if YEAR_PATTERN.match(str(c))]
    if not year_cols:
        raise ValueError("No 4-digit year columns found in Z files.")
    zdf[year_cols] = zdf[year_cols].apply(pd.to_numeric, errors="coerce")

    # Transition de-dup
    zdf_adj = apply_transition_deduplication(zdf, year_cols)

    # Exclude countries
    zdf_adj = zdf_adj[~zdf_adj["Country"].isin(COUNTRY_EXCLUDE)].copy()

    # Load sub3 totals and abbreviations
    sub3_amount = read_csv_safely(SUB3_AMOUNT_PATH)
    sub3_amount.columns = [str(c).strip() for c in sub3_amount.columns]
    sub3_to_abbrev = dict(
        zip(sub3_amount["IPCR_sub3"].astype(str),
            sub3_amount["IPC Subclass Abbreviation"].astype(str))
    )
    # Apply manual overrides
    for k in ALL_LISTED_SUB3:
        if k in SUBCLASS_ABBREV_OVERRIDES:
            sub3_to_abbrev[k] = SUBCLASS_ABBREV_OVERRIDES[k]

    if os.path.exists(SUBCLASS_NAME_PATH):
        subclass_names = read_csv_safely(SUBCLASS_NAME_PATH).rename(columns={"IPCR_sub3": "IPCsubclass"})
        for k, v in zip(subclass_names["IPCsubclass"].astype(str),
                        subclass_names["IPC Subclass Abbreviation"].astype(str)):
            sub3_to_abbrev.setdefault(k, v)
    sub3_amount = sub3_amount.set_index("IPCR_sub3")

    # Country code -> full name map
    country_name_map = build_country_name_map()

    # Period year lists
    period_to_years: Dict[str, List[str]] = {}
    for label, y0, y1 in PERIODS:
        yrs = [str(y) for y in range(y0, y1 + 1) if str(y) in year_cols]
        period_to_years[label] = yrs

    # Build rows for CSV
    out_rows = []
    for sub3, period_label in TARGETS:
        gmask = zdf_adj["IPCR"].astype(str).str.startswith(sub3)
        z_sub = zdf_adj[gmask].copy()
        ipcr_groups = sorted(set(z_sub["IPCR"].astype(str)))
        group_count = len(ipcr_groups)

        p_years = period_to_years.get(period_label, [])
        pioneer_counter: Counter = Counter()

        if group_count > 0 and p_years:
            for ipcr_code in ipcr_groups:
                cols = ["Country", "IPCR"] + p_years
                g_ipcr = z_sub.loc[z_sub["IPCR"].astype(str) == ipcr_code, cols]
                if g_ipcr.empty:
                    continue
                pioneer_cty, pioneer_year, _ = pioneer_with_tie_break_period(g_ipcr, p_years)
                if pioneer_cty is not None and pioneer_year is not None:
                    pioneer_counter[pioneer_cty] += 1

        # stabilize order by count desc, country asc
        top_items = pioneer_counter.most_common()
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

        amount_col = PERIOD_TO_SUB3_AMOUNT_COL.get(period_label)
        if (sub3 in sub3_amount.index) and (amount_col in sub3_amount.columns):
            period_amount = float(sub3_amount.loc[sub3, amount_col])
        else:
            period_amount = float("nan")

        abbrev = sub3_to_abbrev.get(sub3, "")

        row = {
            "IPCsubclass": sub3,
            "Period": period_label,
            "IPCgroup_count": int(group_count),
            "Top1_country": top_pairs[0][0],
            "Top1_count": int(top_pairs[0][1]),
            "Top2_country": top_pairs[1][0],
            "Top2_count": int(top_pairs[1][1]),
            "Top3_country": top_pairs[2][0],
            "Top3_count": int(top_pairs[2][1]),
            "Top4_country": top_pairs[3][0],
            "Top4_count": int(top_pairs[3][1]),
            "Top5_country": top_pairs[4][0],
            "Top5_count": int(top_pairs[4][1]),
            "Period_total_amount": period_amount,
            "IPC Subclass Abbreviation": abbrev,
        }
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows, columns=[
        "IPCsubclass", "Period", "IPCgroup_count",
        "Top1_country", "Top1_count",
        "Top2_country", "Top2_count",
        "Top3_country", "Top3_count",
        "Top4_country", "Top4_count",
        "Top5_country", "Top5_count",
        "Period_total_amount",
        "IPC Subclass Abbreviation",
    ])
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Pioneer figure table -> {OUT_CSV}")

    # Prepare rows for plotting in TARGETS order
    # Build lookup dict (Period, IPCsubclass) -> row dict
    row_dict = {}
    for _, r in out_df.iterrows():
        row_dict[(r["Period"], r["IPCsubclass"])] = r.to_dict()

    targets_by_period = {p[0]: [sub for sub, per in TARGETS if per == p[0]] for p in PERIODS}
    period_to_rows = {}
    for per in targets_by_period:
        filt = []
        for sub3 in targets_by_period[per]:
            key = (per, sub3)
            if key in row_dict:
                filt.append(row_dict[key])
            else:
                # Fill with empty record but keep IPCsubclass
                filt.append({
                    "IPCsubclass": sub3,
                    "Period": per,
                    "IPCgroup_count": 0,
                    "Top1_country": "", "Top1_count": 0,
                    "Top2_country": "", "Top2_count": 0,
                    "Top3_country": "", "Top3_count": 0,
                    "Top4_country": "", "Top4_count": 0,
                    "Top5_country": "", "Top5_count": 0,
                    "Period_total_amount": np.nan,
                    "IPC Subclass Abbreviation": "",
                })
        period_to_rows[per] = filt

    # Build figure (tight margins)
    fig = plt.figure(figsize=(10.8, 13.2))
    gs = fig.add_gridspec(nrows=5, ncols=3, wspace=0.16, hspace=0.16)

    # Column headers via figure text
    col_x = [1.0/6.0, 3.0/6.0, 5.0/6.0]
    fig.text(col_x[0], 0.992, PERIOD_TITLES["1870-1959"], ha="center", va="top", fontsize=12, weight="bold")
    fig.text(col_x[1], 0.992, PERIOD_TITLES["1960-2010"], ha="center", va="top", fontsize=12, weight="bold")
    fig.text(col_x[2], 0.992, PERIOD_TITLES["2011-2024"], ha="center", va="top", fontsize=12, weight="bold")

    # Dashed separators between columns
    for xfrac in (1.0/3.0, 2.0/3.0):
        fig.lines.append(plt.Line2D([xfrac, xfrac], [0.02, 0.985], transform=fig.transFigure,
                                    linestyle="--", linewidth=0.9, color="black"))

    periods_order = ["1870-1959", "1960-2010", "2011-2024"]
    for c, per in enumerate(periods_order):
        rows = period_to_rows.get(per, [])
        color = PERIOD_BAR_COLOR.get(per, "#cccccc")
        for r in range(5):
            ax = fig.add_subplot(gs[r, c])
            rec = rows[r] if r < len(rows) else {
                "IPCsubclass": "", "IPC Subclass Abbreviation": "",
                "Period_total_amount": np.nan, "IPCgroup_count": 0,
                "Top1_country": "", "Top1_count": 0,
                "Top2_country": "", "Top2_count": 0,
                "Top3_country": "", "Top3_count": 0,
                "Top4_country": "", "Top4_count": 0,
                "Top5_country": "", "Top5_count": 0,
            }
            draw_panel(ax, rec, period_label=per, name_map=country_name_map, bar_color=color)

    # Compress outer whitespace further
    fig.subplots_adjust(left=0.035, right=0.995, top=0.965, bottom=0.04, wspace=0.16, hspace=0.16)

    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Combined PDF figure -> {OUT_FIG}")

if __name__ == "__main__":
    main()
