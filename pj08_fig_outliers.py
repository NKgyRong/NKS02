#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ==================== Paths ====================
DATA_PATH = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr.csv"
TCI_INFO_PATH = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr_new_by_T_c_i.csv"
METRICS_DIR = "/data01/rong_dataset/Result/pj08/"
OUTPUT_DIR = "/data01/rong_dataset/Result/pj08/Fig/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== Phase shading ====================
LIGHT_GRAY = (0.90, 0.90, 0.90, 0.30)
DARK_GRAY  = (0.70, 0.70, 0.70, 0.30)
PHASES_BG = [
    (-1_000_000_000, 1859, LIGHT_GRAY),
    (1860, 1969, DARK_GRAY),
    (1970, 2010, LIGHT_GRAY),
    (2011, 2024, DARK_GRAY),
]
SEG_TO_IR_NAME = {
    "1860-1969": "2nd IR",
    "1970-2010": "3rd IR",
    "2011-2024": "4th IR",
}

def add_phase_shading(ax, xmin, xmax):
    for a, b, color in PHASES_BG:
        left = max(a, xmin)
        right = min(b, xmax)
        if left < right:
            ax.axvspan(left, right, facecolor=color, edgecolor="none", zorder=0)

# ==================== Utils ====================
def read_with_fallback(path):
    for enc in ["utf-8", "utf-8-sig", "ISO-8859-1", "latin1", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def select_year_cols(columns, upper=2024):
    years = []
    for c in columns:
        try:
            y = int(float(c))
            if y <= upper:
                years.append(c)
        except Exception:
            continue
    return years

def fmt_comma(x, _pos):
    try:
        return f"{int(x):,}"
    except Exception:
        return f"{x}"

def parse_phase_label_to_segment(phase_label: str):
    if phase_label in ("N_<=1859", "N_le_1859", "N_1859"):
        return None, None, None
    m = re.match(r"N_(\d{4})_(\d{4})$", phase_label)
    if not m:
        return None, None, None
    a, b = int(m.group(1)), int(m.group(2))
    return f"{a}-{b}", a, b

def load_segment_frames(seg_key: str):
    path_z_comb  = os.path.join(METRICS_DIR, f"WWP_{seg_key}_Z_c_i_t.csv")
    path_z_delta = os.path.join(METRICS_DIR, f"WWP_{seg_key}_Z_delta_c_i_t.csv")
    path_z_nhat  = os.path.join(METRICS_DIR, f"WWP_{seg_key}_Z_nhat_c_i_t.csv")
    path_rho     = os.path.join(METRICS_DIR, f"WWP_{seg_key}_rho_c_i.csv")

    z_comb_df  = read_with_fallback(path_z_comb)
    z_delta_df = read_with_fallback(path_z_delta)
    z_nhat_df  = read_with_fallback(path_z_nhat)
    rho_df     = read_with_fallback(path_rho)

    for df, nm in [(z_comb_df, "Z_c_i_t"), (z_delta_df, "Z_delta_c_i_t"), (z_nhat_df, "Z_nhat_c_i_t")]:
        if not set(["Country", "IPCR"]).issubset(df.columns):
            raise ValueError(f"{nm} for {seg_key} missing columns Country/IPCR")
    if not set(["Country", "IPCR", "rho", "P"]).issubset(rho_df.columns):
        raise ValueError(f"rho for {seg_key} missing columns Country/IPCR/rho/P")

    return z_comb_df, z_delta_df, z_nhat_df, rho_df

def build_rho_map(rho_df):
    rho_map = {}
    for _, r in rho_df.iterrows():
        key = (str(r["Country"]), str(r["IPCR"]))
        try:
            rv = float(r["rho"])
        except Exception:
            rv = np.nan
        try:
            pv = float(r["P"])
        except Exception:
            pv = np.nan
        rho_map[key] = (rv, pv)
    return rho_map

def rho_with_stars(rho_map, country, ipcr):
    rho_val, p_val = rho_map.get((country, ipcr), (np.nan, np.nan))
    if np.isnan(rho_val):
        return "rho=NA"
    mark = ""
    if not np.isnan(p_val):
        if p_val < 0.01:
            mark = "**"
        elif p_val < 0.05:
            mark = "*"
    return f"rho={rho_val:.3f}{mark}"

def get_series(df, country, ipcr, cols):
    m = df[(df["Country"] == country) & (df["IPCR"] == ipcr)]
    if m.empty:
        return None
    row = m.iloc[0]
    return row[cols].astype(float).to_numpy()

def normalize_ipcr(ipcr: str) -> str:
    s = ipcr.strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z]\d{2}[A-Z])(\d+)$", s)
    if m:
        head, tail = m.group(1), m.group(2)
        tail = tail.zfill(4)
        return head + tail
    return s

# ==================== Load base tables ====================
data_df = read_with_fallback(DATA_PATH)
tci_info_df = read_with_fallback(TCI_INFO_PATH)

ipcr_name_col = (
    "IPC Group Abbreviation"
    if "IPC Group Abbreviation" in tci_info_df.columns
    else ("IPCRname" if "IPCRname" in tci_info_df.columns else None)
)
needed_cols_tci_base = {"Country", "Name", "IPCR_sub4", "Amount"}
missing_base = needed_cols_tci_base - set(tci_info_df.columns)
if missing_base:
    raise ValueError(f"Missing columns in {TCI_INFO_PATH}: {missing_base}")
if ipcr_name_col is None:
    raise ValueError("Missing IPCR name column in TCI info: need 'IPC Group Abbreviation' or 'IPCRname'.")

phase_cols = [c for c in tci_info_df.columns if c.startswith("N_")]
if not phase_cols:
    raise ValueError("No phase columns (N_*) found in TCI info CSV.")

tci_grouped = (
    tci_info_df
    .groupby(["Country", "IPCR_sub4"], as_index=False)
    .agg({"Name": "first", ipcr_name_col: "first", "Amount": "sum", **{c: "sum" for c in phase_cols}})
)

info_map_total = {
    (row["Country"], row["IPCR_sub4"]): {
        "CountryName": row["Name"],
        "IPCRname": row[ipcr_name_col],
        "Amount_total": int(row["Amount"]) if pd.notna(row["Amount"]) else 0
    }
    for _, row in tci_grouped.iterrows()
}

ipcr_name_map = (
    tci_grouped.groupby("IPCR_sub4", as_index=False)[ipcr_name_col].first()
    .set_index("IPCR_sub4")[ipcr_name_col].to_dict()
)

# ==================== Year bounds from base data ====================
year_cols_all = select_year_cols(data_df.columns[2:])
if not year_cols_all:
    raise ValueError("No year columns detected in WWP_Amount_ipcr.csv")
years_int_all = np.array([int(float(y)) for y in year_cols_all], dtype=int)
YEAR_MIN, YEAR_MAX = int(years_int_all.min()), int(years_int_all.max())

# ==================== PSO label adjustment rules ====================
def pso_label_adjust(country: str, ipcr: str, seg_key: str):
    """
    Returns (dx, dy, ha, va) in data coordinates for the first Z>=2 label.
    Defaults to place above and centered (dx=0, dy=+0.3).
    Custom overrides for specified cases to avoid overlap.
    """
    # defaults
    dx, dy, ha, va = 0.0, 0.3, "center", "bottom"

    key = (country.upper(), ipcr.upper(), seg_key)
    overrides = {
        ("FR", "C08K0005", "1860-1969"): (-0.8, 0.3, "right", "bottom"),  # move left
        ("US", "C08K0005", "1860-1969"): (-0.8, 0.3, "right", "bottom"),  # NEW: move left
        ("US", "H01L0021", "1970-2010"): (-0.8, 0.3, "right", "bottom"),  # move left
        ("US", "G06N0003", "2011-2024"): (0.0, -0.6, "center", "top"),    # move below
    }

    return overrides.get(key, (dx, dy, ha, va))

# ==================== Draw one cell onto given axes ====================
def draw_tci_on_axes(ax_left, country, ipcr, seg_key, seg_start, seg_end,
                     z_comb_df, z_delta_df, z_nhat_df, rho_map,
                     phase_label_for_count, panel_label=None):
    seg_year_cols = select_year_cols(z_comb_df.columns[2:], upper=seg_end)
    seg_year_cols = [c for c in seg_year_cols if seg_start <= int(float(c)) <= seg_end]
    if not seg_year_cols:
        ax_left.text(0.5, 0.5, "MISSING (No years)", ha="center", va="center", fontsize=12, alpha=0.6)
        ax_left.axis("off")
        return False
    seg_years_int = np.array([int(float(c)) for c in seg_year_cols], dtype=int)

    zc = get_series(z_comb_df, country, ipcr, seg_year_cols)
    zd = get_series(z_delta_df, country, ipcr, seg_year_cols)
    zn = get_series(z_nhat_df, country, ipcr, seg_year_cols)
    if any(s is None for s in [zc, zd, zn]):
        ax_left.text(0.5, 0.5, "MISSING (No Z-series)", ha="center", va="center", fontsize=12, alpha=0.6)
        ax_left.axis("off")
        return False

    amt_all_year_cols = select_year_cols(data_df.columns[2:], upper=seg_end)
    amt_year_cols = [c for c in amt_all_year_cols if seg_start <= int(float(c)) <= seg_end]
    amt = get_series(data_df, country, ipcr, amt_year_cols)
    amt_years_int = np.array([int(float(c)) for c in amt_year_cols], dtype=int)

    amt_aligned = np.zeros_like(seg_years_int, dtype=float)
    if amt is not None and amt_years_int.size > 0:
        idx_map = {y: i for i, y in enumerate(amt_years_int)}
        for j, y in enumerate(seg_years_int):
            amt_aligned[j] = float(amt[idx_map[y]]) if y in idx_map else 0.0

    ax_right = ax_left.twinx()

    ax_right.bar(seg_years_int, amt_aligned, alpha=0.4, zorder=2)
    ax_right.set_ylabel("Amount (n_c_i_t)")
    ax_right.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_comma))

    ax_left.plot(seg_years_int, zc, linewidth=1.6, label="Z_c_i_t (combined)", zorder=3)
    ax_left.plot(seg_years_int, zd, linewidth=1.2, label="Z_delta_c_i_t", zorder=3)
    ax_left.plot(seg_years_int, zn, linewidth=1.2, label="Z_nhat_c_i_t", zorder=3)
    ax_left.set_ylabel("Z-score")

    ax_left.set_xlim(seg_years_int[0], seg_years_int[-1])
    xmin, xmax = ax_left.get_xlim()
    add_phase_shading(ax_left, xmin, xmax)

    ax_left.set_zorder(2); ax_left.patch.set_alpha(0)
    ax_right.set_zorder(1); ax_right.patch.set_alpha(0)
    ax_left.grid(False); ax_right.grid(False)

    # === Annotations for Z thresholds ===
    zc_arr = np.asarray(zc, dtype=float)
    mask_ge2 = (~np.isnan(zc_arr)) & (zc_arr >= 2.0)
    mask_le_neg2 = (~np.isnan(zc_arr)) & (zc_arr <= -2.0)

    # Indices
    idxs_ge2 = np.where(mask_ge2)[0]
    idxs_le_neg2 = np.where(mask_le_neg2)[0]

    # Plot all Z<=-2 points (scatter + small label)
    if idxs_le_neg2.size > 0:
        ax_left.scatter(seg_years_int[idxs_le_neg2], zc_arr[idxs_le_neg2], s=18, zorder=4)
        for k in idxs_le_neg2:
            yr, zval = seg_years_int[k], zc_arr[k]
            ax_left.text(yr, zval + 0.1, f"{zval:.1f}\n({yr})",
                         fontsize=7, ha="center", va="bottom", zorder=5)

    # Plot Z>=2 points except the first (scatter + small label, default style)
    if idxs_ge2.size > 1:
        for k in idxs_ge2[1:]:
            yr, zval = seg_years_int[k], zc_arr[k]
            ax_left.scatter([yr], [zval], s=18, zorder=4)
            ax_left.text(yr, zval + 0.1, f"{zval:.1f}\n({yr})",
                         fontsize=7, ha="center", va="bottom", zorder=5)

    # Highlight the first Z>=2 (PSO) if exists: red dot + red label "(PSO, YEAR)"
    if idxs_ge2.size >= 1:
        k0 = idxs_ge2[0]
        yr0, z0 = seg_years_int[k0], zc_arr[k0]
        # red point
        ax_left.scatter([yr0], [z0], s=40, zorder=6, color="red")
        # label placement with custom adjustments per rules
        dx, dy, ha, va = pso_label_adjust(country, ipcr, seg_key)
        ax_left.text(yr0 + dx, z0 + dy, f"{z0:.1f}\n(PSO, {yr0})",
                     fontsize=8, color="red", ha=ha, va=va, zorder=7, fontweight="bold")

    # Legend
    ax_left.legend(loc="upper left", fontsize=7, frameon=True)

    # Title with (a-i)
    info = info_map_total.get((country, ipcr), {})
    country_name = info.get("CountryName", country)
    rho_str = rho_with_stars(rho_map, country, ipcr)
    count_val = np.nan
    if phase_label_for_count in tci_grouped.columns:
        row = tci_grouped[(tci_grouped["Country"] == country) & (tci_grouped["IPCR_sub4"] == ipcr)]
        if not row.empty:
            count_val = pd.to_numeric(row.iloc[0][phase_label_for_count], errors="coerce")
    count_str = "0" if pd.isna(count_val) else f"{int(count_val):,}"

    prefix = f"({panel_label}) " if panel_label else ""
    title_line = f"{prefix}{country_name} - {ipcr} (Count={count_str}, {rho_str}) [{seg_key}]"
    ax_left.set_title(title_line, fontsize=10)
    ax_left.set_xlabel("Year")

    return True

# ==================== Main ====================
def main():
    seg_frames_cache = {}

    def ensure_seg(seg_key):
        if seg_key in seg_frames_cache:
            return seg_frames_cache[seg_key]
        m = re.match(r"(\d{4})-(\d{4})", seg_key)
        if not m:
            raise ValueError(f"Bad seg_key: {seg_key}")
        a, b = int(m.group(1)), int(m.group(2))
        zc, zd, zn, rh = load_segment_frames(seg_key)
        rmap = build_rho_map(rh)
        seg_frames_cache[seg_key] = (zc, zd, zn, rmap, a, b)
        return seg_frames_cache[seg_key]

    # Requested 9 plots
    requested = [
        ("US", "C08K005",  "1860-1969"),  # a
        ("JP", "H01L0021", "1970-2010"),  # b
        ("CN", "G06N0003", "2011-2024"),  # c
        ("GB", "C08K005",  "1860-1969"),  # d
        ("US", "H01L0021", "1970-2010"),  # e
        ("US", "G06N0003", "2011-2024"),  # f
        ("FR", "C08K005",  "1860-1969"),  # g
        ("CN", "H01L0021", "1970-2010"),  # h
        ("JP", "G06N0003", "2011-2024"),  # i
    ]
    labels = list("abcdefghi")

    # Draw 3x3 vector panel
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = np.asarray(axes)

    for idx, (cc, ip_raw, seg_key) in enumerate(requested):
        row, col = divmod(idx, 3)
        ax_left = axes[row, col]
        ip = normalize_ipcr(ip_raw)
        zc, zd, zn, rmap, a, b = ensure_seg(seg_key)

        phase_label_for_count = f"N_{a}_{b}"
        if phase_label_for_count not in tci_grouped.columns:
            cand = [c for c in tci_grouped.columns if c.startswith("N_") and f"{a}" in c and f"{b}" in c]
            if cand:
                phase_label_for_count = cand[0]

        draw_tci_on_axes(
            ax_left, cc, ip, seg_key, a, b,
            zc, zd, zn, rmap,
            phase_label_for_count=phase_label_for_count,
            panel_label=labels[idx]
        )

    # Reserve top space for column headers
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    # Column headers (two lines per column)
    col_seg_keys = ["1860-1969", "1970-2010", "2011-2024"]
    col_ipcrs    = ["C08K0005", "H01L0021", "G06N0003"]
    header_y1 = 0.98
    header_y2 = 0.9605

    for j in range(3):
        top_ax = axes[0, j]
        bb = top_ax.get_position()
        cx = (bb.x0 + bb.x1) / 2.0

        seg = col_seg_keys[j]
        ir_name = SEG_TO_IR_NAME.get(seg, "")
        ipcr = col_ipcrs[j]
        ipcr_abbrev = ipcr_name_map.get(ipcr, ipcr)

        line1 = f"{seg} - {ir_name}"
        line2 = f"{ipcr} - {ipcr_abbrev}"

        fig.text(cx, header_y1, line1, ha="center", va="bottom", fontsize=14, fontweight="bold")
        fig.text(cx, header_y2, line2, ha="center", va="bottom", fontsize=12)

    # Vertical dashed separators between columns
    bb00 = axes[0, 0].get_position()
    bb01 = axes[0, 1].get_position()
    bb02 = axes[0, 2].get_position()
    bb20 = axes[2, 0].get_position()
    bb21 = axes[2, 1].get_position()
    bb22 = axes[2, 2].get_position()

    x_sep_01 = (bb00.x1 + bb01.x0) / 2.0  # between col 0 and 1
    x_sep_12 = (bb01.x1 + bb02.x0) / 2.0  # between col 1 and 2

    y_bottom = min(bb20.y0, bb21.y0, bb22.y0)
    y_top    = max(bb00.y1, bb01.y1, bb02.y1)

    for x_sep in (x_sep_01, x_sep_12):
        line = Line2D([x_sep, x_sep], [y_bottom, y_top],
                      transform=fig.transFigure,
                      linestyle="--", linewidth=1.2, color="black",
                      alpha=0.6, zorder=20, solid_capstyle="butt")
        fig.add_artist(line)

    out_panel_pdf = os.path.join(OUTPUT_DIR, "3x3_PSO.pdf")
    plt.savefig(out_panel_pdf, format="pdf")
    plt.close(fig)
    print(f"Saved panel (vector): {out_panel_pdf}")

if __name__ == "__main__":
    main()
