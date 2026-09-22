#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Paths ----------------
BASE = "/data01/rong_dataset/Result/pj08"
AMT_CSV = os.path.join(BASE, "WWP_Amount_ipcr_new_by_Country.csv")
PIO_CSV = os.path.join(BASE, "WWP_Country_pionner.csv")
FIG_DIR = os.path.join(BASE, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIG_DIR, "Avg_gap_top10_union.pdf")


# ---------------- Robust CSV Reader ----------------
def read_csv_robust(path):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


amt = read_csv_robust(AMT_CSV)
pio = read_csv_robust(PIO_CSV)


# ---------------- Expected Columns ----------------
amt_expected = [
    "Country", "Name", "Amount",
    "N_le_1859", "N_1860_1969", "N_1970_2010", "N_2011_2024"
]
pio_expected = [
    "Country", "Name",
    "1870-1969_Pioneer_no", "1870-1969_Avg_year_gap", "1870-1969_Mid_year_gap",
    "1970-2010_Pioneer_no", "1970-2010_Avg_year_gap", "1970-2010_Mid_year_gap",
    "2011-2024_Pioneer_no", "2011-2024_Avg_year_gap", "2011-2024_Mid_year_gap"
]

missing_amt = [c for c in amt_expected if c not in amt.columns]
missing_pio = [c for c in pio_expected if c not in pio.columns]
if missing_amt:
    raise ValueError(f"Missing columns in Amount CSV: {missing_amt}")
if missing_pio:
    raise ValueError(f"Missing columns in Pioneer CSV: {missing_pio}")


# ---------------- Helpers: period rankings ----------------
def period_rank_series(amt_df, count_col):
    s = pd.to_numeric(amt_df[count_col], errors="coerce").fillna(0)
    tmp = (
        amt_df[["Country"]]
        .assign(cnt=s.values)
        .sort_values("cnt", ascending=False)
    )
    tmp["rank"] = tmp["cnt"].rank(method="min", ascending=False).astype(int)
    return tmp.set_index("Country")["rank"]

# Build per-period rank maps (descending by period counts)
rank_1860 = period_rank_series(amt, "N_1860_1969").to_dict()
rank_1970 = period_rank_series(amt, "N_1970_2010").to_dict()
rank_2011 = period_rank_series(amt, "N_2011_2024").to_dict()


# ---------------- Top10 by period (union of countries) ----------------
period_specs = [
    ("1860_1969", "N_1860_1969", "1870-1969_Avg_year_gap"),
    ("1970_2010", "N_1970_2010", "1970-2010_Avg_year_gap"),
    ("2011_2024", "N_2011_2024", "2011-2024_Avg_year_gap"),
]

top_sets = []
for _, amt_col, _ in period_specs:
    tmp = amt.copy()
    tmp[amt_col] = pd.to_numeric(tmp[amt_col], errors="coerce")
    top10 = tmp.nlargest(10, amt_col)
    top_sets.append(set(top10["Country"].astype(str)))

union_countries = set().union(*top_sets)


# ---------------- Merge for Avg_year_gap and Name ----------------
pio_sub = pio[pio["Country"].astype(str).isin(union_countries)].copy()

# Map Country -> Name (full name) from amt
name_map = (
    amt[["Country", "Name"]]
    .dropna(subset=["Country"])
    .drop_duplicates(subset=["Country"], keep="first")
    .set_index("Country")["Name"]
    .to_dict()
)

period_frames = {}
for pretty, _, gap_col in period_specs:
    df = pio_sub[["Country", gap_col]].copy()
    df["Name"] = df["Country"].map(name_map).fillna(df["Country"].astype(str))
    df.rename(columns={gap_col: "Avg_year_gap"}, inplace=True)
    df["Avg_year_gap"] = pd.to_numeric(df["Avg_year_gap"], errors="coerce")
    df = df.dropna(subset=["Avg_year_gap"])
    df = df.sort_values("Avg_year_gap", ascending=True)
    period_frames[pretty] = df[["Country", "Name", "Avg_year_gap"]].reset_index(drop=True)


# ---------------- Plot ----------------
max_rows = max((len(df) for df in period_frames.values()), default=10)
fig_h = max(4.0, 0.40 * max_rows)
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(16, fig_h),
    sharex=False, sharey=False
)

titles = ["1860_1969", "1970_2010", "2011_2024"]

for ax, key, title_txt in zip(axes, ["1860_1969", "1970_2010", "2011_2024"], titles):
    df = period_frames.get(key, pd.DataFrame(columns=["Country", "Name", "Avg_year_gap"]))

    # Remove all borders/spines
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.grid(False)

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_title(title_txt, pad=8, fontsize=12, fontweight="bold")
        ax.xaxis.set_visible(False)
        ax.set_yticks([])
        continue

    y_pos = np.arange(len(df))
    ax.invert_yaxis()

    # Draw bars
    bars = ax.barh(y_pos, df["Avg_year_gap"].values)

    xmax = float(np.nanmax(df["Avg_year_gap"].values)) if len(df) else 1.0
    pad = 0.01 * xmax

    # Make room on the left for two-line labels (Name and rank info)
    left_pad = max(0.30 * xmax, 0.5)
    ax.set_xlim(left=-left_pad, right=xmax * 1.06 if xmax > 0 else 1.0)

    # Choose period rank dicts for current/previous period
    if key == "1860_1969":
        curr_rank_map = rank_1860
        prev_rank_map = None
    elif key == "1970_2010":
        curr_rank_map = rank_1970
        prev_rank_map = rank_1860
    else:
        curr_rank_map = rank_2011
        prev_rank_map = rank_1970

    # Draw value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df["Avg_year_gap"].values)):
        if pd.notna(val):
            ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2.0,
                    f"{val:.2f}", va="center", ha="left", fontsize=9, clip_on=False)

    # Hide default ticks; we will draw custom two-line labels at left
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])

    # Draw two-line labels: Name and rank line
    for i, row in df.iterrows():
        country = str(row["Country"])
        name = str(row["Name"])

        # Line 1: full name
        ax.text(
            -left_pad * 0.98, y_pos[i] - 0.05, name,
            ha="left", va="center", fontsize=9
        )

        # Line 2: period rank "# n" and optional delta vs previous period
        curr_r = curr_rank_map.get(country, None)
        if curr_r is not None:
            rank_line = f"# {int(curr_r)}"
        else:
            rank_line = "# NA"

        color = "black"
        if prev_rank_map is not None:
            prev_r = prev_rank_map.get(country, None)
            if (curr_r is not None) and (prev_r is not None):
                delta = int(prev_r) - int(curr_r)  # positive = improved (up)
                if delta > 0:
                    rank_line = f"{rank_line} (^ {delta})"
                    color = "green"
                elif delta < 0:
                    rank_line = f"{rank_line} (v {abs(delta)})"
                    color = "red"
                else:
                    rank_line = f"{rank_line} (= 0)"
                    color = "black"

        ax.text(
            -left_pad * 0.98, y_pos[i] + 0.35, rank_line,
            ha="left", va="center", fontsize=9, color=color
        )

    ax.set_title(title_txt, pad=8, fontsize=12, fontweight="bold")
    ax.xaxis.set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved figure to: {OUT_PDF}")
