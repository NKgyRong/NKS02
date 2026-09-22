#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ===================== Config =====================
IPCR_META_PATH = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr_new_by_sub4.csv"   # expects: IPCR_sub4, IPC Group Abbreviation, Amount
Z_PATH = "/data01/rong_dataset/Result/pj08/WWP_Z_c_i_t.csv"                            # expects: Country, IPCR, <year cols>
COUNTRY_NAME_PATH = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr_new_by_Country.csv"  # expects: Country, Name
AMOUNT_TCI_PATH = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr_new_by_T_c_i.csv"  # expects: Country, Name, IPCR_sub4, Amount, N_1860_1969, N_1970_2010, N_2011_2024

OUTPUT_DIR = "/data01/rong_dataset/Result/pj08/Fig/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many IPCR to draw per phase
TOP_IPCR_PER_PHASE = 1

# Threshold for highlighting
THRESHOLD = 2

# Panel sizing (single V1 figures)
FIG_WIDTH = 12
FIG_HEIGHT_PER_ROW_V1 = 0.7
HSPACE_V1 = 0.0
Y_SCALE_EXPAND = 1.6
THR_COLOR = "#888"
THR_LW = 0.7

# Fonts (smaller overall)
FONT_TINY = 6
FONT_SMALL = 7
FONT_BASE = 7
FONT_TITLE = 9
FONT_COLHDR = 10

# Exclude regional org codes & unwanted pseudo-countries
EXCLUDE_CODES = {"AP", "EA", "EP", "GC", "OA", "WO", "UA/SU"}

# Three phases (inclusive ranges)
PHASES = [
    ("N_1860_1969", 1860, 1969, "Phase_1860_1969"),
    ("N_1970_2010", 1970, 2010, "Phase_1970_2010"),
    ("N_2011_2024", 2011, 2024, "Phase_2011_2024"),
]

# Custom IPC classification names (override meta abbreviations if provided)
CUSTOM_IPCR_NAMES = {
    "H01J0029": "Cathode-ray/e-beam tube details",
    "H04L0012": "Data switching networks",

    "B60P0001": "Load-transport vehicles",
    "H04N0005": "Television system details",
    "H01L0031": "Radiation-sensitive semiconductors",
    "H04Q0003": "Selecting arrangements",
    "G06F0017": "Specific-function computing",
    "G06N0003": "Biological-model computing"

}

# ===================== Helpers =====================
def read_csv_with_fallback(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "latin1", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path} with tried encodings.")

def detect_year_columns_generic(df: pd.DataFrame):
    """
    Detect columns representing yearly values (e.g., Z scores).
    Accept columns that:
      - are exactly a 4-digit year, or
      - contain a 4-digit year substring (e.g., 'Z_1999', 'Y2001', 'zscore_2010').
    Exclude 2025.
    """
    year_pat = re.compile(r"(19|20)\d{2}")
    candidates = {}
    for col in df.columns:
        m = year_pat.search(str(col))
        if not m:
            continue
        year = int(m.group(0))
        if year == 2025:
            continue
        non_null = df[col].notna().sum()
        is_exact = bool(re.fullmatch(r"\d{4}", str(col)))
        candidates.setdefault(year, []).append((col, non_null, is_exact))
    chosen = []
    for year, items in candidates.items():
        items_sorted = sorted(items, key=lambda x: (x[1], x[2]), reverse=True)
        col_best = items_sorted[0][0]
        chosen.append((year, col_best))
    chosen.sort(key=lambda x: x[0])
    year_ints = [y for (y, _) in chosen]
    year_cols = [c for (_, c) in chosen]
    return year_cols, year_ints

def first_crossing_year(series: pd.Series, years: list, thr: float):
    """Return (year, value) of the first year where value >= thr; None if never crosses."""
    for col, y in zip(series.index, years):
        val = series[col]
        if pd.notna(val) and val >= thr:
            return y, float(val)
    return None

def safe_filename(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def sparse_year_ticks(x_years):
    """Choose sparse year ticks (5y/10y) to avoid overlap; include ends."""
    if len(x_years) == 0:
        return x_years
    y0, y1 = int(x_years[0]), int(x_years[-1])
    span = y1 - y0
    interval = 5 if span <= 80 else 10
    ticks = []
    start = y0 - (y0 % interval) if (y0 % interval) == 0 else y0 + (interval - (y0 % interval))
    for y in range(start, y1 + 1, interval):
        if y >= y0:
            ticks.append(y)
    if y0 not in ticks:
        ticks = [y0] + ticks
    if y1 not in ticks:
        ticks = ticks + [y1]
    return ticks

# ---------- Country name mapping ----------
country_map_df = read_csv_with_fallback(COUNTRY_NAME_PATH)
if not {"Country", "Name"}.issubset(country_map_df.columns):
    raise ValueError("Country name file must have columns: Country, Name")
code_to_name = dict(zip(country_map_df["Country"].astype(str), country_map_df["Name"].astype(str)))

# Forced overrides
SPECIAL_NAME_OVERRIDES = {
    "SU/RU": "Soviet Union/Russia",
    "RU/SU": "Russia",  # per request: display RU/SU as Russia
    "SU": "Soviet Union",
    "RU": "Russia",
    "TW": "Taiwan (China)",
    "Taiwan": "Taiwan (China)",
}

def wrap_name(s: str, width: int = 16):
    if not s:
        return s
    lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(lines) if lines else s

def country_label(code: str) -> str:
    c = str(code)
    if c in SPECIAL_NAME_OVERRIDES:
        return SPECIAL_NAME_OVERRIDES[c]
    name = code_to_name.get(c, c)
    return SPECIAL_NAME_OVERRIDES.get(name, name)

# ---------- Figure-level text helpers ----------
def _points_to_pixels(fig, pt):
    return pt * fig.dpi / 72.0

def figtext_at_data(fig, ax, x_data, y_data, text, dx_pt=0, dy_pt=0, **kwargs):
    x_pix, y_pix = ax.transData.transform((x_data, y_data))
    x_pix += _points_to_pixels(fig, dx_pt)
    y_pix += _points_to_pixels(fig, dy_pt)
    x_fig, y_fig = fig.transFigure.inverted().transform((x_pix, y_pix))
    return fig.text(x_fig, y_fig, text, transform=fig.transFigure, **kwargs)

def figtext_at_axesfrac(fig, ax, x_axes, y_axes, text, dx_pt=0, dy_pt=0, **kwargs):
    x_pix, y_pix = ax.transAxes.transform((x_axes, y_axes))
    x_pix += _points_to_pixels(fig, dx_pt)
    y_pix += _points_to_pixels(fig, dy_pt)
    x_fig, y_fig = fig.transFigure.inverted().transform((x_pix, y_pix))
    return fig.text(x_fig, y_fig, text, transform=fig.transFigure, **kwargs)

# ===================== IPCR meta & Abbreviation =====================
ipcr_meta = read_csv_with_fallback(IPCR_META_PATH)
for req in ["IPCR_sub4", "Amount", "IPC Group Abbreviation"]:
    if req not in ipcr_meta.columns:
        raise ValueError(f"Missing column in meta: {req}")

# build base map, then allow custom overrides at use time
ipcr_to_abbrev = dict(zip(ipcr_meta["IPCR_sub4"].astype(str), ipcr_meta["IPC Group Abbreviation"].astype(str)))

# ===================== Z timeseries =====================
z_df = read_csv_with_fallback(Z_PATH)
for req in ["Country", "IPCR"]:
    if req not in z_df.columns:
        raise ValueError(f"Missing column in Z file: {req}")

year_cols_all, year_ints_all = detect_year_columns_generic(z_df)
if not year_cols_all:
    raise RuntimeError("No year-like columns detected in Z file.")

keep_cols = ["Country", "IPCR"] + year_cols_all
z_df = z_df[keep_cols].copy()
z_df[year_cols_all] = z_df[year_cols_all].apply(pd.to_numeric, errors="coerce")
z_df["IPCR"] = z_df["IPCR"].astype(str)
z_df["Country"] = z_df["Country"].astype(str)

# ===================== (Country, IPCR) -> Total Patent_Count =====================
tci_df = read_csv_with_fallback(AMOUNT_TCI_PATH)
for req in ["Country", "IPCR_sub4", "Amount"]:
    if req not in tci_df.columns:
        raise ValueError(f"Missing column in T_c_i amount file: {req}")
tci_df["Country"] = tci_df["Country"].astype(str)
tci_df["IPCR_sub4"] = tci_df["IPCR_sub4"].astype(str)
tci_df["Amount"] = pd.to_numeric(tci_df["Amount"], errors="coerce").fillna(0).astype(int)

# Map for overall totals (reference)
patent_count_map = {}
for _, r in tci_df[["Country", "IPCR_sub4", "Amount"]].dropna(subset=["Country", "IPCR_sub4"]).iterrows():
    key = (r["Country"], r["IPCR_sub4"])
    patent_count_map[key] = int(r["Amount"])

# ===================== Core drawing (single figures) =====================
def build_v1_figure(ipcr, abbrev, year_ints, year_cols, sub_df,
                    *, outer_titles=True, outer_frame=True, title_suffix=""):
    # collect first crossings
    crossing_info = []
    for _, row in sub_df.iterrows():
        info = first_crossing_year(row[year_cols], year_ints, THRESHOLD)
        if info is not None:
            crossing_info.append((row["Country"], info))
    if not crossing_info:
        return None, []

    # tie-break: when first-crossing years are equal, use phase Patent_Count (desc), then country code
    patcnt_map_local = dict(zip(sub_df["Country"], sub_df["Patent_Count"]))

    def sort_key(item):
        c, (y, z) = item
        pc = patcnt_map_local.get(c, 0)
        return (y, -pc, c)

    crossing_info_sorted = sorted(crossing_info, key=sort_key)
    sorted_countries = [c for c, _ in crossing_info_sorted]
    crossing_dict = dict(crossing_info_sorted)
    earliest_cross_year = min(y for (_, (y, _)) in crossing_info_sorted)
    pioneer_country = next(c for (c, (y, _)) in crossing_info_sorted if y == earliest_cross_year)

    # keep only in the sorted order
    sub_df = sub_df[sub_df["Country"].isin(sorted_countries)].copy()
    country_series_map = {row["Country"]: row[year_cols] for _, row in sub_df.iterrows()}

    nrows = len(sorted_countries)
    fig, axes = plt.subplots(nrows=nrows, ncols=1,
                             figsize=(FIG_WIDTH, max(FIG_HEIGHT_PER_ROW_V1 * nrows, 3)),
                             sharex=True)
    if nrows == 1:
        axes = [axes]
    plt.subplots_adjust(top=0.985, hspace=HSPACE_V1)

    x_years = np.array(year_ints, dtype=int)

    for idx, (ax, country) in enumerate(zip(axes, sorted_countries)):
        series = country_series_map[country].values.astype(float)

        # main line & threshold (no "Z=2" text label)
        ax.plot(x_years, series, linewidth=1.2, zorder=1)
        ax.axhline(y=THRESHOLD, linestyle="--", linewidth=THR_LW, color=THR_COLOR, zorder=0)

        # pioneer vline, highlight RPSG segment, and crossing point
        ax.axvline(x=earliest_cross_year, linestyle="--", linewidth=1.0, color="red", alpha=0.9, zorder=50)
        cy, cz = crossing_dict[country]
        if cy > earliest_cross_year:
            mask = (x_years >= earliest_cross_year) & (x_years <= cy)
            if mask.any():
                ax.plot(x_years[mask], series[mask], linewidth=1.6, color="red", zorder=60)
        ax.scatter([cy], [cz], s=18, color="red", zorder=70)

        # year and RPSG label at the crossing point
        if cy == earliest_cross_year:
            figtext_at_data(fig, ax, cy, cz, f"{cy}",
                            dx_pt=6, dy_pt=3, ha="left", va="bottom",
                            fontsize=FONT_SMALL, color="red", zorder=1000)
        else:
            rpsg = max(0, cy - earliest_cross_year)
            figtext_at_data(fig, ax, cy, cz, f"{cy} (RPSG={rpsg})",
                            dx_pt=0, dy_pt=3, ha="center", va="bottom",
                            fontsize=FONT_SMALL, color="red", zorder=1000)

        # cosmetics
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.set_visible(False)

        # LEFT-SIDE 3-LINE LABELS: right-aligned and tight to plot (x=0.0 with ha='right')
        disp = wrap_name(country_label(country), width=16)
        patcnt = patcnt_map_local.get(country, np.nan)
        left_name = disp
        left_count = f"{int(patcnt):,}" if pd.notna(patcnt) else "-"
        left_pioneer = "(Pioneer)" if country == pioneer_country else ""
        ax.text(-0.03, 0.78, left_name, transform=ax.transAxes,
                ha="right", va="center", fontsize=FONT_BASE, color="#111", linespacing=1.2)
        ax.text(-0.03, 0.52, left_count, transform=ax.transAxes,
                ha="right", va="center", fontsize=FONT_BASE, color="#333")
        if left_pioneer:
            ax.text(-0.03, 0.28, left_pioneer, transform=ax.transAxes,
                    ha="right", va="center", fontsize=FONT_TINY, color="red")

        # bottom x-axis ticks only on last row
        if idx < nrows - 1:
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        else:
            ticks = sparse_year_ticks(x_years)
            ax.set_xticks(ticks)
            ax.set_xlabel("Year", fontsize=FONT_BASE)
            ax.tick_params(axis="x", pad=4, labelsize=FONT_BASE)

        # y-lims
        try:
            ymin = float(np.nanmin(series))
            ymax = float(np.nanmax(series))
            if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                ymin, ymax = -3, 3
            ymin = min(ymin, THRESHOLD - 1.5)
            ymax = max(ymax, THRESHOLD + 1.5)
            center = 0.5 * (ymin + ymax)
            half = 0.5 * (ymax - ymin) * Y_SCALE_EXPAND
            ax.set_ylim(center - half, center + half)
        except Exception:
            pass
        ax.grid(False)

    # outer elements (frame and/or titles)
    fig.canvas.draw()
    bbox_top = axes[0].get_position(); bbox_bottom = axes[-1].get_position()
    x0 = min(bbox_top.x0, bbox_bottom.x0); x1 = max(bbox_top.x1, bbox_bottom.x1)
    y0 = min(bbox_bottom.y0, bbox_top.y0); y1 = max(bbox_top.y1, bbox_bottom.y1)
    pad_x, pad_y = 0.01, 0.01
    fx0 = max(0.0, x0 - pad_x); fx1 = min(1.0, x1 + pad_x)
    fy0 = max(0.0, y0 - pad_y); fy1 = min(1.0, y1 + pad_y)

    inv = fig.transFigure.inverted()
    left_anchor_fig_x = inv.transform(axes[0].transAxes.transform((-0.02, 0.5)))[0]

    if outer_frame:
        rect = patches.Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0,
                                 fill=False, linewidth=1.0, edgecolor="black",
                                 transform=fig.transFigure, zorder=10)
        fig.add_artist(rect)

    if outer_titles:
        title_text = f"{ipcr}-{abbrev}"  # without [Phase_xxx]
        fig.text((fx0 + fx1) / 2, fy1 + 0.018, title_text, ha="center", va="bottom", fontsize=FONT_TITLE, zorder=900)
        fig.text(left_anchor_fig_x, fy1 + 0.018, "Z_c_i_t", ha="right", va="bottom", fontsize=FONT_BASE, color="#000", zorder=900)

    return fig, axes

def save_fig(fig, base):
    pdf_path = os.path.join(OUTPUT_DIR, base + ".pdf")
    png_path = os.path.join(OUTPUT_DIR, base + ".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}

def make_v1_single(ipcr, abbrev, year_ints, year_cols, sub_df, *,
    keep_frame=True, title_suffix=""):
    fig, _ = build_v1_figure(
        ipcr, abbrev, year_ints, year_cols, sub_df,
        outer_titles=True, outer_frame=keep_frame, title_suffix=title_suffix
    )
    if fig is None:
        return None
    safe_ipcr = safe_filename(ipcr); safe_abbrev = safe_filename(abbrev)
    base = f"{safe_ipcr}_{safe_abbrev}_Zge{THRESHOLD}_V1_stacked"
    paths = save_fig(fig, base)
    paths.update({"ipcr": ipcr, "title": f"{ipcr}-{abbrev}"})
    print(f"Saved V1: {paths['png']}")
    return paths

# ===================== Phase Top-30 selection =====================
def select_top_ipcr_for_phase(tci_df: pd.DataFrame, phase_col: str, topk: int) -> list:
    if phase_col not in tci_df.columns:
        return []
    tmp = tci_df[["IPCR_sub4", phase_col]].copy()
    tmp[phase_col] = pd.to_numeric(tmp[phase_col], errors="coerce").fillna(0).astype(int)
    top = (tmp.groupby("IPCR_sub4", as_index=False)[phase_col].sum()
              .sort_values(phase_col, ascending=False)
              .head(topk))
    return top["IPCR_sub4"].astype(str).tolist()

# ===================== Prepare per-IPCR subset =====================
def prepare_sub_for_ipcr_phase(ipcr: str, phase_lo: int, phase_hi: int, *, phase_count_map=None):
    # abbreviation or custom classification name
    abbrev_default = ipcr_to_abbrev.get(ipcr, "")
    abbrev = CUSTOM_IPCR_NAMES.get(ipcr, abbrev_default)

    # phase windowed year columns
    mask = [(y >= phase_lo and y <= phase_hi) for y in year_ints_all]
    yints = [y for y, m in zip(year_ints_all, mask) if m]
    ycols = [c for c, m in zip(year_cols_all, mask) if m]
    if len(ycols) == 0:
        return None

    # subset rows for this IPCR
    sub = z_df[z_df["IPCR"] == ipcr].copy()
    sub = sub[~sub["Country"].isin(EXCLUDE_CODES)]

    # special rule: for G06N0003 remove CA and RU/SU entirely
    if ipcr == "G06N0003":
        sub = sub[(sub["Country"] != "CA") & (sub["Country"] != "RU/SU")]

    if sub.empty:
        return None

    # fill phase-specific patent counts for left-side labels
    if phase_count_map is None:
        sub["Patent_Count"] = np.nan
    else:
        sub["Patent_Count"] = [
            int(phase_count_map.get((row["Country"], ipcr), 0)) for _, row in sub.iterrows()
        ]

    # determine selection size: default up to 10; for G06N0003 in 2011-2024 select top 9
    limit_n = 10
    if ipcr == "G06N0003" and phase_lo == 2011 and phase_hi == 2024:
        limit_n = 9

    # select countries
    if (sub["Patent_Count"] > 0).any():
        sub = (sub.sort_values("Patent_Count", ascending=False, na_position="last")
                 .drop_duplicates(subset=["Country"])
                 .head(limit_n)
                 .copy())
    else:
        zmax = sub[[c for c in ycols if c in sub.columns]].max(axis=1)
        sub = (sub.assign(_ZMAX=zmax)
                 .sort_values("_ZMAX", ascending=False)
                 .drop(columns=["_ZMAX"])
                 .drop_duplicates(subset=["Country"])
                 .head(limit_n)
                 .copy())

    if sub.empty:
        return None

    return abbrev, yints, ycols, sub

# =============== 3x3 composite figure ===============
def compute_crossing_and_order(sub_df, year_cols, year_ints):
    """Return sorted countries, pioneer year, and crossing dict using tie-break by Patent_Count."""
    crossing_info = []
    for _, row in sub_df.iterrows():
        info = first_crossing_year(row[year_cols], year_ints, THRESHOLD)
        if info is not None:
            crossing_info.append((row["Country"], info))

    if not crossing_info:
        return [], None, {}

    patcnt_map_local = dict(zip(sub_df["Country"], sub_df["Patent_Count"]))

    def sort_key(item):
        c, (y, z) = item
        pc = patcnt_map_local.get(c, 0)
        return (y, -pc, c)

    crossing_info_sorted = sorted(crossing_info, key=sort_key)
    sorted_countries = [c for c, _ in crossing_info_sorted]
    earliest_cross_year = min(y for (_, (y, _)) in crossing_info_sorted)
    crossing_dict = dict(crossing_info_sorted)
    return sorted_countries, earliest_cross_year, crossing_dict

def draw_compact_panel(fig, cell_spec, ipcr, abbrev, year_ints, year_cols, sub_df,
                       *, show_outer_frame=True, show_title=True, title_text="",
                       fontsize=FONT_BASE, xlim=None, ylim=None, tight_hspace=0.02):
    """
    Draw a compact stacked panel inside a given GridSpec cell.
    Uses an inner GridSpec with nrows equal to number of countries.
    xlim and ylim are enforced to unify units across panels (unless overridden).
    """
    sorted_countries, pioneer_year, crossing_dict = compute_crossing_and_order(sub_df, year_cols, year_ints)
    if not sorted_countries:
        ax_empty = fig.add_subplot(cell_spec)
        ax_empty.axis("off")
        return

    sub_df = sub_df.set_index("Country").loc[sorted_countries].reset_index()
    patcnt_map_local = dict(zip(sub_df["Country"], sub_df["Patent_Count"]))
    x_years = np.array(year_ints, dtype=int)

    nrows = len(sorted_countries)
    inner_gs = GridSpecFromSubplotSpec(nrows, 1, subplot_spec=cell_spec, hspace=tight_hspace, wspace=0.0)

    for r, country in enumerate(sorted_countries):
        ax = fig.add_subplot(inner_gs[r, 0])

        series = sub_df[sub_df["Country"] == country][year_cols].values[0].astype(float)

        # main line & threshold (no "Z=2" text label)
        ax.plot(x_years, series, linewidth=1.0, zorder=1)
        ax.axhline(y=THRESHOLD, linestyle="--", linewidth=THR_LW, color=THR_COLOR, zorder=0)

        # pioneer vline, highlight RPSG segment and crossing point
        ax.axvline(x=pioneer_year, linestyle="--", linewidth=0.9, color="red", alpha=0.9, zorder=50)
        cy, cz = crossing_dict[country]
        if cy > pioneer_year:
            mask = (x_years >= pioneer_year) & (x_years <= cy)
            if mask.any():
                ax.plot(x_years[mask], series[mask], linewidth=1.4, color="red", zorder=60)
        ax.scatter([cy], [cz], s=14, color="red", zorder=70)

        # year and RPSG label
        if cy == pioneer_year:
            ax.annotate(f"{cy}", xy=(cy, cz), xytext=(5, 3), textcoords="offset points",
                        ha="left", va="bottom", fontsize=FONT_TINY, color="red", clip_on=False)
        else:
            rpsg = max(0, cy - pioneer_year)
            ax.annotate(f"{cy} (RPSG={rpsg})", xy=(cy, cz), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FONT_TINY, color="red", clip_on=False)

        # LEFT-SIDE 3-LINE LABELS: right-aligned and tight to plot (x=0.0 with ha='right')
        disp = wrap_name(country_label(country), width=16)
        patcnt = patcnt_map_local.get(country, np.nan)
        left_name = disp
        left_count = f"{int(patcnt):,}" if pd.notna(patcnt) else "-"
        left_pioneer = "(Pioneer)" if cy == pioneer_year else ""
        ax.text(-0.03, 0.78, left_name, transform=ax.transAxes,
                ha="right", va="center", fontsize=FONT_TINY, color="#111", linespacing=1.2)
        ax.text(-0.03, 0.52, left_count, transform=ax.transAxes,
                ha="right", va="center", fontsize=FONT_TINY, color="#333")
        if left_pioneer:
            ax.text(-0.03, 0.28, left_pioneer, transform=ax.transAxes,
                    ha="right", va="center", fontsize=FONT_TINY, color="red")

        # cosmetics
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.set_visible(False)

        # enforce unified x/y units across panels
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        # x axis only on bottom-most row
        if r < nrows - 1:
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        else:
            xl = ax.get_xlim()
            ticks = sparse_year_ticks(np.array(range(int(xl[0]), int(xl[1]) + 1)))
            ax.set_xticks(ticks)
            ax.set_xlabel("Year", fontsize=FONT_TINY)
            ax.tick_params(axis="x", pad=2, labelsize=FONT_TINY)

        ax.grid(False)

    # outer frame and title
    bbox = fig.add_subplot(cell_spec)
    bbox.axis("off")
    for spine in bbox.spines.values():
        spine.set_visible(False)
    pos = bbox.get_position()
    rect = patches.Rectangle((pos.x0, pos.y0), pos.width, pos.height,
                             fill=False, linewidth=0.9, edgecolor="black",
                             transform=fig.transFigure, zorder=5)
    fig.add_artist(rect)

    if show_title:
        cx = pos.x0 + pos.width/2.0
        cy = pos.y1 + 0.006
        fig.text(cx, cy, title_text, ha="center", va="bottom", fontsize=FONT_SMALL)

def make_3x3_composite():
    """
    Build a 3x3 composite PDF figure with column headers as periods and (a)-(i) labels.
    """
    phase_map = {
        "1860-1969": ("N_1860_1969", 1860, 1969, "Phase_1860_1969"),
        "1970-2010": ("N_1970_2010", 1970, 2010, "Phase_1970_2010"),
        "2011-2024": ("N_2011_2024", 2011, 2024, "Phase_2011_2024"),
        "1860_1969": ("N_1860_1969", 1860, 1969, "Phase_1860_1969"),
    }

    panels = [
        ("(a)", "H01J0029", "1860-1969"),
        ("(b)", "H04L0012", "1970-2010"),
        ("(c)", "G06V0010", "2011-2024"),
        ("(d)", "B60P0001", "1860-1969"),
        ("(e)", "H04N0005", "1970-2010"),
        ("(f)", "H01L0031", "2011-2024"),
        ("(g)", "H04Q0003", "1860-1969"),
        ("(h)", "G06F0017", "1970-2010"),
        ("(i)", "G06N0003", "2011-2024"),
    ]

    # per-panel xlim overrides
    xlim_overrides = {
        ("H01J0029", "1860-1969"): (1930, 1969),
        ("B60P0001", "1860-1969"): (1940, 1969),
        ("H04Q0003", "1860-1969"): (1910, 1969),
    }

    # Build phase-specific count maps once
    phase_count_maps = {}
    for phase_col, lo, hi, tag in PHASES:
        tci_phase = tci_df[["Country", "IPCR_sub4", phase_col]].copy()
        tci_phase[phase_col] = pd.to_numeric(tci_phase[phase_col], errors="coerce").fillna(0).astype(int)
        phase_count_maps[phase_col] = {
            (r["Country"], r["IPCR_sub4"]): int(r[phase_col]) for _, r in tci_phase.iterrows()
        }

    # Prepare data for all panels first, to compute global y-limits
    prepared = []
    global_y_min = np.inf
    global_y_max = -np.inf

    for label, ipcr, period_key in panels:
        if period_key not in phase_map:
            prepared.append((label, ipcr, period_key, None))
            continue
        phase_col, lo, hi, tag = phase_map[period_key]
        phase_count_map = phase_count_maps[phase_col]
        prep = prepare_sub_for_ipcr_phase(ipcr, lo, hi, phase_count_map=phase_count_map)
        if prep is None:
            prepared.append((label, ipcr, period_key, None))
            continue
        abbrev, yints, ycols, sub = prep
        vals = sub[ycols].to_numpy().astype(float)
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if np.isfinite(vmin): global_y_min = min(global_y_min, vmin)
        if np.isfinite(vmax): global_y_max = max(global_y_max, vmax)
        prepared.append((label, ipcr, period_key, (abbrev, yints, ycols, sub, (lo, hi), tag)))

    if not np.isfinite(global_y_min) or not np.isfinite(global_y_max) or global_y_min == global_y_max:
        global_y_min, global_y_max = -3.0, 3.0

    global_y_min = min(global_y_min, THRESHOLD - 1.8)
    global_y_max = max(global_y_max, THRESHOLD + 1.8)

    fig = plt.figure(figsize=(13, 13))
    outer_gs = GridSpec(3, 3, figure=fig, wspace=0.28, hspace=0.18)

    # Column headers (period labels)
    col_titles = ["1860-1969", "1970-2010", "2011-2024"]
    for col in range(3):
        dummy_ax = fig.add_subplot(outer_gs[0, col])
        pos = dummy_ax.get_position()
        dummy_ax.remove()
        cx = (pos.x0 + pos.x1) / 2.0
        fig.text(cx, pos.y1 + 0.03, col_titles[col], ha="center", va="bottom", fontsize=FONT_COLHDR, fontweight="bold")

    # Draw each panel with unified y limits and per-panel x overrides if specified
    for idx, (label, ipcr, period_key, pack) in enumerate(prepared):
        row = idx // 3
        col = idx % 3
        cell = outer_gs[row, col]

        if pack is None:
            ax_empty = fig.add_subplot(cell)
            ax_empty.axis("off")
            continue

        abbrev, yints, ycols, sub, (lo, hi), tag = pack

        # panel title without [Phase_xxx]
        title_text = f"{label} {ipcr}-{abbrev}"

        # xlim: panel-specific override > default column range
        xlim = xlim_overrides.get((ipcr, period_key), (lo, hi))

        draw_compact_panel(
            fig, cell, ipcr, abbrev, yints, ycols, sub,
            show_outer_frame=True, show_title=True, title_text=title_text,
            fontsize=FONT_SMALL,
            xlim=xlim,
            ylim=(global_y_min, global_y_max),
            tight_hspace=0.012
        )

    base = f"Composite_3x3_Zge{THRESHOLD}"
    pdf_path = os.path.join(OUTPUT_DIR, base + ".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Composite] Saved: {pdf_path}")

# ===================== Main =====================
def main():
    print("Starting phase-wise IPCR plotting ...")

    for phase_col, lo, hi, phase_tag in PHASES:
        if phase_col not in tci_df.columns:
            print(f"[Phase] Column {phase_col} not found in {AMOUNT_TCI_PATH}; skip.")
            continue

        # Build phase-specific count map
        tci_phase = tci_df[["Country", "IPCR_sub4", phase_col]].copy()
        tci_phase[phase_col] = pd.to_numeric(tci_phase[phase_col], errors="coerce").fillna(0).astype(int)
        phase_count_map = {
            (r["Country"], r["IPCR_sub4"]): int(r[phase_col])
            for _, r in tci_phase.iterrows()
        }

        # pick Top N IPCR by phase totals
        top_ipcr_list = select_top_ipcr_for_phase(tci_df, phase_col, TOP_IPCR_PER_PHASE)
        if not top_ipcr_list:
            print(f"[Phase] No IPCR found for {phase_col}; skip.")
            continue

        for ipcr in top_ipcr_list:
            prep = prepare_sub_for_ipcr_phase(ipcr, lo, hi, phase_count_map=phase_count_map)
            if prep is None:
                print(f"[Warn] Skip IPCR={ipcr} for {phase_col}: no data in this phase.")
                continue
            abbrev, year_ints, year_cols, sub = prep

            make_v1_single(ipcr, abbrev, year_ints, year_cols, sub,
                           keep_frame=True, title_suffix=phase_tag)

    # Build the 3x3 composite figure
    make_3x3_composite()

    print("Done.")

if __name__ == "__main__":
    main()
