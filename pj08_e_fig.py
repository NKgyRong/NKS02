#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

BASE = "/data01/rong_dataset/Result/pj08"
IN_CSV = os.path.join(BASE, "WWP_Amount_ipcr_new_by_Country.csv")
FIRST_CSV = os.path.join(BASE, "WWP_first_year_gap.csv")
FIG_DIR = os.path.join(BASE, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIG_DIR, "Top5_rank_flow.pdf")

PERIODS = [
    ("1860-1969", 1860, 1970, "N_1860_1969"),
    ("1970-2010", 1970, 2011, "N_1970_2010"),
    ("2011-2024", 2011, 2025, "N_2011_2024"),
]
BOUNDARY_YEAR = {"1860-1969": 1970, "1970-2010": 2011}

FIG_W, FIG_H = 14, 7.5
DOT_SIZE = 36
LINE_W = 2.0


GAP_1970 = 4.0  
GAP_2011 = 10.0  

def xmap(year: float) -> float:
    x = year
    if year >= 1970:
        x += GAP_1970
    if year >= 2011:
        x += GAP_2011
    return x


XU = 1.0

BG_COLORS = [
    (0.90, 0.90, 0.90, 0.30),
    (0.70, 0.70, 0.70, 0.30),
    (0.90, 0.90, 0.90, 0.30),
]

BOX_EDGE_OFFSET = {"1970-2010": 6.0}

# enlarged spacing and box height
RANK_SPACING = 4.0
rank_to_y = {r: (6 - r) * RANK_SPACING for r in range(1, 6)}
RANK_BOX_H = 3.0

# separate vertical stacks for entrants and drops
ENTRANT_UNDER_START_Y = -1.5
ENTRANT_UNDER_STEP    = 1.0

DROP_UNDER_START_Y    = -3.0
DROP_UNDER_STEP       = 1.2

# connection offsets
ENTRANT_CONN_OFFSET = -0.5  # entrants attach 0.5 below box center
DROP_CONN_OFFSET    = +0.5  # drops leave 0.5 above box center

# at most 3 drops shown below axis (for tightening bottom limit)
MAX_DROPS_BELOW = 3

# small pioneer text box drawing params (smaller)
SMALL_LABEL_FONT = 6.5
SMALL_BOX_FACE = (0.6, 0.6, 0.6, 0.35)
SMALL_BOX_PAD = 0.00
SMALL_BOX_HALF_H = 0.32
SMALL_LINE_W = 1.0

# lane/collision params for small pioneer boxes
LANES_PER_YEAR = 3       # at most 3 vertical lanes for nearby years
LANE_STEP      = 0.85    # vertical spacing between lanes (y units)
LANE_COLUMN_DX = 0.22    # small x shift for overflow columns at same year cluster
LANE_CONFLICT_DX = 16.0   # labels within +/- 6 "years" cannot share the same lane


FIXED_COLORS = {
    "CN": "#d62728",
    "US": "#1f77b4",
    "DE": "#ff7f0e",
    "JP": "#2ca02c",
    "KR": "#9467bd",
    "GB": "#8c564b",
    "FR": "#17becf",
    "RU": "#e377c2",
    "IT": "#7f7f7f",
    "CA": "#bcbd22",
    "TW": "#aec7e8",
}

# experiment list (period key must match CSV column prefix)
EXPERIMENT_ITEMS = [
# US, 1870-1969
    ("H01L0041", "1870-1969"),
    ("H01R0033", "1870-1969"),
    ("B29C0044", "1870-1969"),
    ("A61B0001", "1870-1969"),
    ("F41H0007", "1870-1969"),
    ("C07C0327", "1870-1969"),
    ("F16D0029", "1870-1969"),
    ("G01S0019", "1870-1969"),
    ("H01B0012", "1870-1969"),
    ("G06V0040", "1870-1969"),
    ("F03D0015", "1870-1969"),
    ("G04B0013", "1870-1969"),
    ("H04N0013", "1870-1969"),
# DE, 1870-1969
    ("A62C0011", "1870-1969"),
    ("A01D0013", "1870-1969"),
    ("B62L0001", "1870-1969"),
    ("H05B0037", "1870-1969"),
    ("F28D0020", "1870-1969"),
    ("B23B0001", "1870-1969"),
    ("B62D0006", "1870-1969"),
    ("F02F0011", "1870-1969"),
    ("C12N0007", "1870-1969"),
    ("G01M0009", "1870-1969"),
    ("A61M0060", "1870-1969"),
    ("C10G0049", "1870-1969"),
# GB, 1870-1969
    ("B23F0011", "1870-1969"),
    ("C10B0015", "1870-1969"),
    ("D05B0067", "1870-1969"),
    ("H04K0001", "1870-1969"),
    ("B64C0019", "1870-1969"),
    ("G02B0021", "1870-1969"),
    ("H04L0023", "1870-1969"),
    ("E01F0003", "1870-1969"),
    ("H04N0005", "1870-1969"),
    ("G05B0019", "1870-1969"),
    ("H05H0009", "1870-1969"),
# FR, 1870-1969
    ("B64B0001", "1870-1969"),
    ("H01H0045", "1870-1969"),
    ("H01T0013", "1870-1969"),
    ("B60K0007", "1870-1969"),
    ("A62B0015", "1870-1969"),
    ("H02J0004", "1870-1969"),
    ("A62B0013", "1870-1969"),
    ("G06F0040", "1870-1969"),
    ("H04W0084", "1870-1969"),
    ("H03K0009", "1870-1969"),
# CA, 1870-1969
    ("G11C0011", "1870-1969"),
    ("G01K0007", "1870-1969"),
    ("H01B0013", "1870-1969"),
    ("A62B0035", "1870-1969"),
    ("F16L0021", "1870-1969"),
#    ("C08L0023", "1870-1969"),
#    ("G01V0003", "1870-1969"),
    ("G01R0027", "1870-1969"),
    ("B01D0017", "1870-1969"),
# JP, 1970-2010
#    ("C01D0013", "1970-2010"),
    ("B61F0011", "1970-2010"),
    ("A41H0011", "1970-2010"),
    ("F01B0027", "1970-2010"),
    ("G06F0008", "1970-2010"),
    ("B60W0060", "1970-2010"),
    ("F01N0009", "1970-2010"),
    ("H04H0009", "1970-2010"),
    ("B82Y0040", "1970-2010"),
# US, 1970-2010 
    ("H01J0041", "1970-2010"),
    ("B64G0006", "1970-2010"),
    ("F41B0006", "1970-2010"),
    ("H03F0011", "1970-2010"),
    ("G06F0111", "1970-2010"),
    ("C12S0001", "1970-2010"),
    ("B82Y0025", "1970-2010"),
    ("B64G0004", "1970-2010"),
#    ("C08F0273", "1970-2010"),
# DE, 1970-2010 
    ("H01B0011", "1970-2010"),
    ("C07G0011", "1970-2010"),
    ("C25D0007", "1970-2010"),
    ("B62D0119", "1970-2010"),
    ("B64D0023", "1970-2010"),
    ("G03B0030", "1970-2010"),
# CN, 1970-2010 
#    ("A01P0023", "1970-2010"),
#    ("B66B0020", "1970-2010"),
    ("C09K0105", "1970-2010"),
    ("C22C0111", "1970-2010"),
    ("B66C0025", "1970-2010"),
    ("G08G0007", "1970-2010"),
    ("C21C0003", "1970-2010"),
# KR, 1970-2010 
    ("B82Y0015", "1970-2010"),
    ("A62B0031", "1970-2010"),
    ("G16H0015", "1970-2010"),
    ("H10K0085", "1970-2010"),
    
# CN, 2011-2024
    ("B62J0045", "2011-2024"),
    ("B60L0053", "2011-2024"),
    ("H10K0071", "2011-2024"),
# US, 2011-2024
    ("H03F0005", "2011-2024"),
    ("B82Y0020", "2011-2024"),
    ("G06F0117", "2011-2024"),
# JP, 2011-2024
    ("B63G0001", "2011-2024"),
    ("B82Y0015", "2011-2024"),
    ("H04L0047", "2011-2024"),
# KR, 2011-2024
    ("B82Y0025", "2011-2024"),
    ("B63B0003", "2011-2024"),
    ("B64U0020", "2011-2024"),
# TW(CN), 2011-2024
    ("H01L0047", "2011-2024"),
    ("B33Y0010", "2011-2024"),
    ("H03H0011", "2011-2024"),

]

# manual abbreviation override: use these first if present
ABBR_OVERRIDE = {
# US, 1870-1969
    "H01L0041": "Semiconductor devices",
    "H01R0033": "Coupling Devices",
    "B29C0044": "Plastic shaping",
    "C07C0327": "Thiocarboxylic",
    "F16D0029": "Clutches Systems",
    "G01S0019": "Satellite positioning",
    "A61B0001": "Endoscope instruments",
    "H01B0012": "Hyper Conductors",
    "G06V0040": "Recognition Biometric",
    "F03D0015": "Mechanical Transmission",
    "H04N0013": "Stereoscopic video systems",
# DE, 1870-1969
    "A62C0011": "Portable Extinguishers",
    "H05B0037": "Electric heating",
    "F28D0020": "Heat Storage Plants",
    "B23B0001": "Turning Working",
    "B62D0006": "Automatically Steering",
    "B62L0001": "Brakes & Arrangements",
    "F02F0011": "Engines Sealings",
    "G01M0009": "Aerodynamic Testing",
    "A61M0060": "Blood Pumps",
    "C10G0049": "Treat Hydrocarbon Oils",
# GB, 1870-1969
    "B23F0011": "Worm Wheels",
    "B64C0019": "Aircraft Control",
    "H04L0023": "Telegraphic Systems",   
    "H04N0005": "Television systems",
    "G06E0001": "Digital Data Processing",   
    "G05B0019": "Programme-control",
# FR, 1870-1969
    "A62B0015": "Gas Mask",
    "H02J0004": "Circuit Arrangements",
    "A62B0013": "Gasproof Shelters",
    "H03K0009": "Demodulating Pulses", 
    "G06F0040": "NLP", 
# CA, 1870-1969
    "A62B0035": "Safety Belts", 
    "H01B0013": "Conductors/cables Manufacture", 
    "G01K0007": "Electric/magnetic Sensing", 
    "G11C0011": "Electric Digital Storage",
    "G01R0027": "Resistance meas.", 

# JP, 1970-2010
    "B61F0011": "Rail Vehicles",
    "F01B0027": "Engines", 
    "G06F0008": "Software engineering",
    "B60W0060": "Vehicle drive control",
    "F01N0009": "Gas Treating Apparatus",
    "H04H0009": "Broadcast COMM.",
    "B82Y0040": "Nanostructures",
# US, 1970-2010 
    "F41B0006": "EM Launchers",
    "H01J0041": "Discharge Tubes",
    "G06F0111": "CAD",
    "B64G0004": "Space Tools",
    "C08F0273": "Macromolecular Compounds",
# DE, 1970-2010    
    "C25D0007": "Electroplating layer",
    "H01B0011": "Communication Cables",
    "G03B0030": "Integrated Lens",
# CN, 1970-2010   
    "G08G0007": "Traffic Control Systems",
    "C21C0003": "Wrought-iron",
    "B66C0025": "Cranes",
    "C22C0111": "Metallic Fibres",
    "C09K0105": "Erosion Prevention",
# KR, 1970-2010 
    "A62B0031": "Diving Gear",
    "B82Y0015": "IT Nanostructures",
    "G16H0015": "Health ICT",
    "H10K0085": "Organic Semi",
    
# CN, 2011-2024
    "B62J0045": "Electronic Control Arrangements",
    "B60L0053": "EV Charging",
    "H10K0071": "Organic Semiconductors",
# US, 2011-2024
    "H03F0005": "General Semiconductor",
    "B82Y0020": "Nanooptics",
    "G06F0117": "ASIC Details",
# JP, 2011-2024
    "B63G0001": "Naval Guns/Missiles",
    "B82Y0015": "Nano IT",
    "H04L0047": "Digital Transmission",
# KR, 2011-2024
    "B82Y0025": "Nanomagnetism",
    "B63B0003": "Ship Hull Structure",
    "B64U0020": "UAV Structure",
# TW(CN), 2011-2024
    "H01L0047": "Optoelectronic Semiconductors",
    "B33Y0010": "3D printing materials",
    "H03H0011": "Networks Using Active Elements"
}

# map CSV period to chart period
DATA2CHART_PERIOD = {
    "1870-1969": "1860-1969",
    "1970-2010": "1970-2010",
    "2011-2024": "2011-2024",
}

def robust_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")

def rank_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.copy()
    vals = tmp[col].fillna(0)
    tmp["_sortkey"] = list(zip(-vals, tmp["Name"].astype(str)))
    tmp = tmp.sort_values("_sortkey", kind="mergesort").reset_index(drop=True)
    tmp["Rank"] = np.arange(1, len(tmp) + 1)
    tmp.drop(columns=["_sortkey"], inplace=True)
    return tmp

def top5(df_ranked: pd.DataFrame) -> pd.DataFrame:
    return df_ranked[df_ranked["Rank"] <= 5].copy()

def allocate_colors(all_codes: list) -> dict:
    palette = plt.get_cmap("tab20").colors
    cmap, used = {}, 0
    for c in all_codes:
        if c in FIXED_COLORS:
            cmap[c] = FIXED_COLORS[c]
        else:
            cmap[c] = palette[used % len(palette)]
            used += 1
    return cmap

def draw_period_backgrounds(ax):
    for i, (_, x0, x1, _) in enumerate(PERIODS):
        ax.axvspan(xmap(x0), xmap(x1 - 1), facecolor=BG_COLORS[i], edgecolor="none", zorder=0)

def draw_open_box(ax, x0, x1, y_center, h, edgecolor, lw=2.0, open_right=False):
    y0 = y_center - h/2.0
    if not open_right:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, fill=False, edgecolor=edgecolor, lw=lw, zorder=3))
    else:
        ax.plot([x0, x0], [y0, y0 + h], color=edgecolor, lw=lw, zorder=3)
        ax.plot([x0, x1], [y0 + h, y0 + h], color=edgecolor, lw=lw, zorder=3)
        ax.plot([x0, x1], [y0, y0], color=edgecolor, lw=lw, zorder=3)

def hline(ax, x0, x1, y, color):
    ax.plot([x0, x1], [y, y], "-", color=color, lw=LINE_W, solid_capstyle="round", zorder=4)

def vline(ax, x, y0, y1, color, lw=LINE_W, z=4):
    ax.plot([x, x], [y0, y1], "-", color=color, lw=lw, solid_capstyle="round", zorder=z)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    # main data
    df = robust_read_csv(IN_CSV)
    required = ["Country", "Name", "N_1860_1969", "N_1970_2010", "N_2011_2024"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    for c in ["N_1860_1969", "N_1970_2010", "N_2011_2024"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    # pioneer data
    df_first = robust_read_csv(FIRST_CSV)
    if "IPCRgroup" not in df_first.columns or "IPC Group Abbreviation" not in df_first.columns:
        raise ValueError("WWP_first_year_gap.csv missing required columns.")

    # build period dicts
    P = {}
    for (label, x0, x1, col) in PERIODS:
        ranked = rank_series(df[["Country", "Name", col]].rename(columns={col: "Value"}), "Value")
        P[label] = {
            "x0": x0, "x1": x1, "col": col,
            "ranked": ranked,
            "top5": top5(ranked),
            "full_rank": dict(zip(ranked["Country"], ranked["Rank"])),
        }

    color_map = allocate_colors(df["Country"].unique().tolist())

    # -------- compute tight bottom limit --------
    def count_entrants_and_drops(prev_label, curr_label):
        prev_top = P[prev_label]["top5"].set_index("Country")
        curr_top = P[curr_label]["top5"].set_index("Country")
        prev_set, curr_set = set(prev_top.index), set(curr_top.index)
        e = len(curr_set - prev_set)
        d = len(prev_set - curr_set)
        d = min(d, MAX_DROPS_BELOW)
        return e, d

    e1, d1 = count_entrants_and_drops("1860-1969", "1970-2010")
    e2, d2 = count_entrants_and_drops("1970-2010", "2011-2024")

    entrant_candidates = []
    if e1 > 0:
        entrant_candidates.append(ENTRANT_UNDER_START_Y - (e1 - 1) * ENTRANT_UNDER_STEP)
    if e2 > 0:
        entrant_candidates.append(ENTRANT_UNDER_START_Y - (e2 - 1) * ENTRANT_UNDER_STEP)

    drop_candidates = []
    if d1 > 0:
        drop_candidates.append(DROP_UNDER_START_Y - (d1 - 1) * DROP_UNDER_STEP)
    if d2 > 0:
        drop_candidates.append(DROP_UNDER_START_Y - (d2 - 1) * DROP_UNDER_STEP)

    bottom_candidates = []
    if entrant_candidates:
        bottom_candidates.append(min(entrant_candidates))
    if drop_candidates:
        bottom_candidates.append(min(drop_candidates))

    PAD_BOTTOM = 0.6
    if bottom_candidates:
        tight_bottom_y = min(bottom_candidates) - PAD_BOTTOM
    else:
        tight_bottom_y = -1.2

    # -------- plotting --------
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    draw_period_backgrounds(ax)

    # bottom spine at the bottom edge of Rank-5 box (erase the gap)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    axis_y = rank_to_y[5] - RANK_BOX_H/2 - 1  
    ax.spines["bottom"].set_position(("data", axis_y))

    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")

    ax.set_xlim(xmap(1860) - 2.0, xmap(2024))
    top_y = rank_to_y[1] + RANK_BOX_H/2 + 0.8
    ax.set_ylim(tight_bottom_y, top_y)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=1)

    ticks = [1860, 1900, 1950, 1970, 1990, 2010, 2011, 2024]
    ax.set_xticks([xmap(t) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])

    # first period entry dots and labels
    p1_label, p1_x0, p1_x1, _ = PERIODS[0]
    p1 = P[p1_label]
    p1_top = p1["top5"].sort_values("Rank")

    dot_x = xmap(p1_x0) - 1.0 * XU
    for _, row in p1_top.iterrows():
        code = row["Country"]
        name = df.loc[df["Country"] == code, "Name"].iloc[0]
        rk = int(row["Rank"])
        y = rank_to_y[rk]
        color = color_map.get(code, "#333333")
        ax.scatter([dot_x], [y], s=DOT_SIZE, color=color, zorder=5)
        ax.text(dot_x - 1.0 * XU, y, name, ha="right", va="center", fontsize=9, color=color)
        hline(ax, dot_x, xmap(p1_x0), y, color)
        draw_open_box(ax, xmap(p1_x0), xmap(p1_x1 - 1), y, RANK_BOX_H, edgecolor=color, lw=2.0, open_right=False)

    def transitions(prev_label, curr_label):
        prev = P[prev_label]
        curr = P[curr_label]
        prev_top = prev["top5"].set_index("Country")
        curr_top = curr["top5"].set_index("Country")
        prev_set, curr_set = set(prev_top.index), set(curr_top.index)

        boundary = BOUNDARY_YEAR[prev_label]
        xb = xmap(boundary)
        prev_end_year = prev["x1"] - 1
        prev_end_x = xmap(prev_end_year) + BOX_EDGE_OFFSET.get(prev_label, 0.0) 


        # current boxes
        is_last = (curr_label == "2011-2024")
        offset_curr = BOX_EDGE_OFFSET.get(curr_label, 0.0)
        for code, row in curr_top.sort_values("Rank").iterrows():
            rk = int(row["Rank"])
            y = rank_to_y[rk]
            color = color_map.get(code, "#333333")
            right_edge = curr["x1"] - 1
            draw_open_box(ax,
                          xmap(curr["x0"]),
                          xmap(right_edge) + offset_curr,
                          y, RANK_BOX_H,
                          edgecolor=color, lw=2.0, open_right=is_last)


        # persisting top5
        for code in (prev_set & curr_set):
            prev_rk = int(prev_top.loc[code, "Rank"])
            curr_rk = int(curr_top.loc[code, "Rank"])
            y0 = rank_to_y[prev_rk]
            y1 = rank_to_y[curr_rk]
            color = color_map.get(code, "#333333")
            elbow_x = xb - 2.5 * XU
            hline(ax, prev_end_x, elbow_x, y0, color)
            vline(ax, elbow_x, y0, y1, color)
            hline(ax, elbow_x, xb, y1, color)

        # entrants
        entrants = []
        for code in (curr_set - prev_set):
            curr_rk = int(curr_top.loc[code, "Rank"])
            prev_rk = prev["full_rank"].get(code, None)
            prev_order = prev_rk if prev_rk is not None else 10**9
            entrants.append((prev_order, code, curr_rk, prev_rk))
        entrants.sort(key=lambda z: z[0])

        for i, (_, code, curr_rk, prev_rk_true) in enumerate(entrants):
            y_under = ENTRANT_UNDER_START_Y - i * ENTRANT_UNDER_STEP
            color = color_map.get(code, "#333333")
            name = df.loc[df["Country"] == code, "Name"].iloc[0]
            xl_text = xb - 11.0 * XU
            xl_dot = xb - 10.0 * XU
            xl_mid = xb - 4.0 * XU
            label = f"{name} (#{int(prev_rk_true)})" if prev_rk_true is not None else f"{name} (prev: n/a)"
            ax.text(xl_text, y_under, label, ha="right", va="center", fontsize=9, color=color)
            ax.scatter([xl_dot], [y_under], s=DOT_SIZE, color=color, zorder=5)
            hline(ax, xl_dot, xl_mid, y_under, color)
            target_y = rank_to_y[curr_rk] + ENTRANT_CONN_OFFSET
            vline(ax, xl_mid, y_under, target_y, color)
            hline(ax, xl_mid, xb, target_y, color)

        # drops
        drops = []
        for code in (prev_set - curr_set):
            curr_rk = curr["full_rank"].get(code, None)
            curr_order = curr_rk if curr_rk is not None else 10**9
            drops.append((curr_order, code, curr_rk))
        drops.sort(key=lambda z: z[0])

        for i, (_, code, curr_rk_true) in enumerate(drops[:MAX_DROPS_BELOW]):  # at most 3 shown
            y0 = rank_to_y[int(prev_top.loc[code, "Rank"])] + DROP_CONN_OFFSET
            y_under = DROP_UNDER_START_Y - i * DROP_UNDER_STEP
            color = color_map.get(code, "#333333")
            name = df.loc[df["Country"] == code, "Name"].iloc[0]
            x_mid = xb - 1.0 * XU
            hline(ax, prev_end_x, x_mid, y0, color)
            vline(ax, x_mid, y0, y_under, color)
            x_dot = xb + 5.0 * XU
            x_text = xb + 6.0 * XU
            hline(ax, x_mid, x_dot, y_under, color)
            ax.scatter([x_dot], [y_under], s=DOT_SIZE, color=color, zorder=5)
            lab = f"{name} #{int(curr_rk_true)}" if curr_rk_true is not None else f"{name} #-"
            ax.text(x_text, y_under, lab, ha="left", va="center", fontsize=9, color=color)

    # draw transitions and boxes
    transitions("1860-1969", "1970-2010")
    transitions("1970-2010", "2011-2024")

    # -------- pioneer labels (small gray text boxes with left vertical line) --------
    # build items by (chart_period, pioneer_country), then place with lane collision detection
    period_top5_index = {pl: P[pl]["top5"].set_index("Country") for pl, _, _, _ in PERIODS}
    first_rows = df_first.set_index("IPCRgroup")

    # collect items first
    items_by_key = {}  # (chart_period, pioneer_country) -> list of dicts
    for ipcr, data_period in EXPERIMENT_ITEMS:
        if ipcr not in first_rows.index:
            continue
        row = first_rows.loc[ipcr]

        # abbreviation override
        abbr_override = ABBR_OVERRIDE.get(ipcr, None)
        if abbr_override is not None:
            abbr = abbr_override
        else:
            abbr = str(row["IPC Group Abbreviation"]) if pd.notna(row["IPC Group Abbreviation"]) else ""

        pioneer_col = f"{data_period}_Pioneer"
        year_col    = f"{data_period}_Pioneer_year"
        if pioneer_col not in row.index or year_col not in row.index:
            continue

        pioneer = row[pioneer_col]
        year = row[year_col]
        if pd.isna(pioneer) or pd.isna(year):
            continue

        pioneer = str(pioneer).strip()
        try:
            year = int(float(year))
        except Exception:
            continue

        chart_period = DATA2CHART_PERIOD.get(data_period)
        if chart_period is None or chart_period not in P:
            continue

        # only if the pioneer country has a visible top-5 box in this chart period
        if pioneer not in period_top5_index[chart_period].index:
            continue

        items_by_key.setdefault((chart_period, pioneer), []).append(
            {"ipcr": ipcr, "abbr": abbr, "year": year}
        )

    # place per (period, country)
    for (chart_period, pioneer), items in items_by_key.items():
        # big box geom
        rk = int(period_top5_index[chart_period].loc[pioneer, "Rank"])
        y_center = rank_to_y[rk]
        x0 = xmap(P[chart_period]["x0"])
        x1 = xmap(P[chart_period]["x1"] - 1)

        # registry of placed boxes for collision checks: list of {"x", "lane"}
        placed = []

        # stable placement: sort by year asc
        items.sort(key=lambda d: d["year"])

        for d in items:
            ipcr = d["ipcr"]
            abbr = d["abbr"]
            year = d["year"]

            # base x and clamp inside big box
            x_base = clamp(xmap(year), x0, x1)

            # choose a lane 0..LANES_PER_YEAR-1 that has no conflict within +/- LANE_CONFLICT_DX;
            # if none, shift a small column to the right and retry
            col_shift = 0
            max_col_trials = 10
            chosen_lane = None
            x_left_final = None

            while col_shift < max_col_trials and chosen_lane is None:
                x_try = clamp(x_base + col_shift * LANE_COLUMN_DX, x0, x1)

                for lane in range(LANES_PER_YEAR):
                    conflict = False
                    for it in placed:
                        # same lane cannot be too close in x
                        if it["lane"] == lane and abs(x_try - it["x"]) <= LANE_CONFLICT_DX:
                            conflict = True
                            break
                    if not conflict:
                        chosen_lane = lane
                        x_left_final = x_try
                        break

                if chosen_lane is None:
                    col_shift += 1

            if chosen_lane is None:
                # fallback
                chosen_lane = 0
                x_left_final = x_base

            # y by lane (symmetric around center)
            y_offset = ((LANES_PER_YEAR - 1) / 2.0 - chosen_lane) * LANE_STEP
            y_label = y_center + y_offset

            # keep inside the big box with a small margin
            y_top_margin = 0.28
            y_bottom_margin = 0.28
            y_label = clamp(
                y_label,
                y_center - RANK_BOX_H/2 + y_bottom_margin,
                y_center + RANK_BOX_H/2 - y_top_margin
            )

            # remember
            placed.append({"x": x_left_final, "lane": chosen_lane})

            # 2-line text: (year) IPCRgroup \n Abbreviation
            text_str = f"({year}) {ipcr}\n{abbr}" if abbr else f"({year}) {ipcr}"

            # draw text box
            ax.text(
                x_left_final, y_label, text_str,
                ha="left", va="center", fontsize=SMALL_LABEL_FONT, color="black", zorder=6,
                bbox=dict(facecolor=SMALL_BOX_FACE, edgecolor="none", boxstyle=f"round,pad={SMALL_BOX_PAD}")
            )

            # left vertical line (country color)
            col_color = color_map.get(pioneer, "#333333")
            vline(ax, x_left_final, y_label - SMALL_BOX_HALF_H, y_label + SMALL_BOX_HALF_H,
                  color=col_color, lw=SMALL_LINE_W, z=7)



    # period titles
    for i, (label, x0, x1, _) in enumerate(PERIODS):
        xc = 0.5 * (xmap(x0) + xmap(x1 - 1))
        if i == 0:
            xc -= 0.6
        plt.text(xc, top_y - 0.5, label, ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ymin, ymax = ax.get_ylim()
    bt = mtransforms.blended_transform_factory(ax.transAxes, ax.transData) 

    note_text = "The IPC groups marked are abbreviations.\n Only a few representative Pioneering IPC groups are shown for each economy."
    ax.text(
        0.01, 0.01, note_text,              
        transform=ax.transAxes,            
        ha="left", va="bottom",           
        fontsize=SMALL_LABEL_FONT,         
        bbox=dict(facecolor="none", edgecolor="none",
                  boxstyle=f"round,pad={SMALL_BOX_PAD}"),
        zorder=10,
    )

    plt.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure to:", OUT_PDF)

if __name__ == "__main__":
    main()
