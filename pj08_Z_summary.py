#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= Paths =========
BASE_DIR = "/data01/rong_dataset/Result/pj08/"
FIG_DIR = os.path.join(BASE_DIR, "Fig")
os.makedirs(FIG_DIR, exist_ok=True)

# Per-segment inputs
SEGMENTS = ["1860-1969", "1970-2010", "2011-2024"]
Z_PATHS = {seg: os.path.join(BASE_DIR, f"WWP_{seg}_Z_c_i_t.csv") for seg in SEGMENTS}
RHO_PATHS = {seg: os.path.join(BASE_DIR, f"WWP_{seg}_rho_c_i.csv") for seg in SEGMENTS}

# Per-segment outputs
OUT_SUMMARY_TMPL = os.path.join(BASE_DIR, "WWP_{seg}_Z_summary.csv")

# One combined figure (PNG + PDF share base)
OUT_FIG_BASE = os.path.join(FIG_DIR, "WWP_Z_time_lines_combined")

# ========= Thresholds / Config =========
POS_THR = 2.0      # Z >= 2
NEG_THR = -2.0     # Z <= -2 (kept for stats, not used in the plot)
YEAR_START = 1860  # x-axis starts from 1860 as requested
COUNTRY_CODES = ["JP", "US", "CN"]

# Shaded band appearance
alpha_light = 0.10
alpha_deep  = 0.18
RED_ROW = 2
BLACK_ROW = RED_ROW + 4  # BLACK_ROW is 4 rows lower than RED_ROW

# ========= Helpers =========
def normalize_ci_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).lower(): c for c in df.columns}
    cand_country = None
    for key in ["country", "country code", "c", "code"]:
        if key in cols:
            cand_country = cols[key]; break
    if cand_country is None and "Country" in df.columns:
        cand_country = "Country"

    cand_ipcr = None
    for key in ["ipcr", "i", "ipcr code", "ipc"]:
        if key in cols:
            cand_ipcr = cols[key]; break
    if cand_ipcr is None and "IPCR" in df.columns:
        cand_ipcr = "IPCR"

    rename_map = {}
    if cand_country and cand_country != "Country":
        rename_map[cand_country] = "Country"
    if cand_ipcr and cand_ipcr != "IPCR":
        rename_map[cand_ipcr] = "IPCR"

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_year_columns(df: pd.DataFrame):
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c)) is not None]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    year_cols = sorted(year_cols, key=lambda x: int(x))
    return df, year_cols


def safe_mean(x):
    x = pd.Series(x).dropna()
    return float(x.mean()) if len(x) else float("nan")


def normal_two_sided_p_from_z(z):
    # two-sided p-value using normal approximation
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def welch_t_fallback(a, b):
    a = pd.Series(a).dropna().to_numpy()
    b = pd.Series(b).dropna().to_numpy()
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean() if n1 else float("nan"), b.mean() if n2 else float("nan")
    s1, s2 = (a.std(ddof=1) if n1 > 1 else float("nan")), (b.std(ddof=1) if n2 > 1 else float("nan"))
    if n1 < 2 or n2 < 2 or np.isnan(s1) or np.isnan(s2):
        return 0.0, float("inf"), 1.0, "Welch t-test (normal approx; insufficient data)"
    se = math.sqrt((s1**2)/n1 + (s2**2)/n2)
    if se == 0.0:
        t_stat = float("inf") if (m1 - m2) != 0 else 0.0
        df = float("inf")
        p = 0.0 if t_stat != 0.0 else 1.0
        return t_stat, df, p, "Welch t-test (normal approx; zero SE)"
    t_stat = (m1 - m2) / se
    num = ((s1**2)/n1 + (s2**2)/n2) ** 2
    den = ((s1**2)/n1) ** 2 / (n1 - 1) + ((s2**2)/n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float("inf")
    p = normal_two_sided_p_from_z(t_stat)
    return t_stat, df, p, "Welch t-test (normal approx)"


def welch_t_with_scipy(a, b):
    try:
        from scipy import stats  # type: ignore
        res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        a = pd.Series(a).dropna().to_numpy()
        b = pd.Series(b).dropna().to_numpy()
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return welch_t_fallback(a, b)
        s1, s2 = a.std(ddof=1), b.std(ddof=1)
        num = ((s1**2)/n1 + (s2**2)/n2) ** 2
        den = ((s1**2)/n1) ** 2 / (n1 - 1) + ((s2**2)/n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else float("inf")
        return float(res.statistic), float(df), float(res.pvalue), "Welch t-test (scipy)"
    except Exception:
        return welch_t_fallback(a, b)


def cohens_d(m1, s1, n1, m2, s2, n2):
    if n1 < 2 or n2 < 2 or np.isnan(s1) or np.isnan(s2):
        return float("nan")
    sp2 = ((n1 - 1)*(s1**2) + (n2 - 1)*(s2**2)) / (n1 + n2 - 2)
    sp = math.sqrt(sp2) if sp2 > 0 else float("nan")
    if sp == 0 or math.isnan(sp):
        return float("nan")
    return (m1 - m2) / sp


def hedges_g(d, n1, n2):
    if math.isnan(d):
        return float("nan")
    J = 1.0 - 3.0 / (4.0*(n1 + n2) - 9.0)
    return J * d


def _add_shaded_band(ax, x_years, y_unit, start, end, label,
                     label_color, alpha, row, x_offset=0.0):

    """
    Draw a vertical shaded band from 'start' to 'end' (inclusive-exclusive style)
    and put the label at the horizontal midpoint at a y-position controlled by 'row'.

    y position scheme:
      - Let ylim = (0, y_top). We set y_top higher than max count.
      - Define a row unit y_unit (computed from max_y).
      - Place text at y = y_top - row * y_unit (row counts down from top).
    """
    # Shade
    ax.axvspan(start, end, color=label_color, alpha=alpha, linewidth=0)

    # Compute y position for text
    y_min, y_top = ax.get_ylim()
    y_pos = y_top - row * y_unit

    # Center x
    x_mid = (start + end) / 2.0 + x_offset


    # Add label (centered, two-line allowed via \n)
    ax.text(
        x_mid, y_pos, label,
        ha="center", va="center",
        color=label_color, fontsize=10, fontweight="bold"
    )


def plot_time_lines_counts_with_bands(z_df, year_cols, country_codes, pos_thr, start_year, out_path_base):
    # years from start_year
    years_all = [int(y) for y in year_cols if int(y) >= start_year]
    if not years_all:
        raise ValueError("No year >= start_year found in combined Z table.")
    years_all = sorted(years_all)
    x = np.array(years_all, dtype=int)
    
    country_names = {"US": "United States", "CN": "China", "JP": "Japan"}


    # styles
    country_styles = {
        "US": {"color": "blue", "linestyle": "--"},
        "CN": {"color": "red", "linestyle": "-"},
        "JP": {"color": "green", "linestyle": (0, (1, 1))},
    }


    fig, ax = plt.subplots(figsize=(12, 6))

    # plot JP/US/CN lines
    max_y = 0
    for code in country_codes:
        sub = z_df[z_df["Country"] == code]
        if sub.empty:
            continue

        first_years = []
        for _, row in sub.iterrows():
            ge2_years = [int(y) for y in year_cols if row[y] >= pos_thr]
            if ge2_years:
                first_years.append(min(ge2_years))

        pos_counts_by_year = pd.Series(first_years).value_counts().to_dict()
        pos_y = np.array([pos_counts_by_year.get(y, 0) for y in years_all], dtype=int)

        style = country_styles.get(code, {"color": None, "linestyle": "-"})
        ax.plot(x, pos_y, linestyle=style["linestyle"],
                label=f"{country_names.get(code, code)} PSO Count",
                color=style["color"])

        if pos_y.size > 0:
            max_y = max(max_y, int(np.nanmax(pos_y)))

    # ensure headroom for labels
    if max_y <= 0:
        max_y = 1
    y_unit = max(1.0, max_y * 0.06)  # row height
    y_top = max_y + y_unit * 10      # leave room for up to ~10 rows of labels
    ax.set_ylim(0, y_top)

    # x-axis range
    ax.set_xlim([years_all[0], years_all[-1]])

    # ===== Shaded bands & labels (rows: BLACK_ROW lower than RED_ROW by 4) =====
    _add_shaded_band(
        ax, years_all, y_unit,
        start=1860, end=1914,
        label="1860-1914\n2nd IR",
        label_color="black", alpha=alpha_light,
        row=RED_ROW
    )

    _add_shaded_band(
        ax, years_all, y_unit,
        start=1914, end=1918,
        label="1914-1918 \nWW I",
        label_color="black", alpha=alpha_deep,
        row=BLACK_ROW
    )

    _add_shaded_band(
        ax, years_all, y_unit,
        start=1931, end=1945,
        label="1931-1945 \nWW II",
        label_color="black", alpha=alpha_light,
        row=BLACK_ROW
    )

    _add_shaded_band(
        ax, years_all, y_unit,
        start=1970, end=2010,
        label="1970-2010\n3rd IR",
        label_color="black", alpha=alpha_deep,
        row=RED_ROW,
        x_offset=5  
    )

    _add_shaded_band(
        ax, years_all, y_unit,
        start=2011, end=2024,
        label="2011-2024\n4th IR",
        label_color="black", alpha=alpha_light,
        row=RED_ROW
    )


    # labels and layout
    ax.set_xlabel("Year")
    ax.set_ylabel("Count (Z>=2)")
    ax.legend(loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = out_path_base + ".png"
    out_pdf = out_path_base + ".pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved combined time lines plot to: {out_png} and {out_pdf}")


def build_segment_summary(z_df_seg, rho_df_seg, out_summary_path):
    # Normalize id columns and detect year columns
    z_df_seg = normalize_ci_columns(z_df_seg)
    rho_df_seg = normalize_ci_columns(rho_df_seg)

    for col in ["Country", "IPCR"]:
        if col not in z_df_seg.columns:
            raise ValueError(f"Column '{col}' not found in segment Z file")
        if col not in rho_df_seg.columns:
            raise ValueError(f"Column '{col}' not found in segment rho file")

    # rho / P names (case-insensitive)
    rho_cols_lower = {str(c).lower(): c for c in rho_df_seg.columns}
    rho_name = rho_cols_lower.get("rho", "rho" if "rho" in rho_df_seg.columns else None)
    p_name = rho_cols_lower.get("p", "P" if "P" in rho_df_seg.columns else None)
    if rho_name is None or p_name is None:
        for c in rho_df_seg.columns:
            cl = str(c).lower()
            if rho_name is None and cl in ["r", "corr", "correlation", "pearson_r"]:
                rho_name = c
            if p_name is None and cl in ["pvalue", "p_value", "p-val", "pval"]:
                p_name = c
    if rho_name is None or p_name is None:
        raise ValueError("rho_df must contain rho and P columns (e.g., 'rho' and 'P').")

    # year columns
    z_df_seg, year_cols_seg = get_year_columns(z_df_seg)
    if not year_cols_seg:
        raise ValueError("No year columns detected in segment Z file (expect 4-digit years).")

    # Per-series counts
    ge2_counts = (z_df_seg[year_cols_seg] >= POS_THR).sum(axis=1, min_count=1).fillna(0).astype(int)
    le_neg2_counts = (z_df_seg[year_cols_seg] <= NEG_THR).sum(axis=1, min_count=1).fillna(0).astype(int)

    z_df_seg = pd.concat([z_df_seg, pd.DataFrame({
        "_ge2_count": ge2_counts,
        "_le_neg2_count": le_neg2_counts,
    })], axis=1)
    z_df_seg["_has_ge2"] = z_df_seg["_ge2_count"] > 0
    z_df_seg["_has_le_neg2"] = z_df_seg["_le_neg2_count"] > 0

    # Merge rho / P
    rho_mini = rho_df_seg[["Country", "IPCR", rho_name, p_name]].copy()
    rho_mini = rho_mini.rename(columns={rho_name: "_rho", p_name: "_P"})
    merged = pd.merge(
        z_df_seg[["Country", "IPCR", "_ge2_count", "_le_neg2_count", "_has_ge2", "_has_le_neg2"]],
        rho_mini,
        on=["Country", "IPCR"],
        how="left"
    )

    # Summary stats
    total_series = int(len(z_df_seg))
    with_ge2 = int((z_df_seg["_has_ge2"]).sum())
    prop_with_ge2 = with_ge2 / total_series if total_series else float("nan")
    avg_ge2_points_overall = safe_mean(z_df_seg["_ge2_count"])

    with_leneg2 = int((z_df_seg["_has_le_neg2"]).sum())
    prop_with_leneg2 = with_leneg2 / total_series if total_series else float("nan")
    avg_leneg2_points_overall = safe_mean(z_df_seg["_le_neg2_count"])

    # Welch t-test on rho
    grp_has_ge2 = merged.loc[z_df_seg["_has_ge2"], "_rho"].dropna().to_numpy()
    grp_no_ge2 = merged.loc[~z_df_seg["_has_ge2"], "_rho"].dropna().to_numpy()

    def _mean_std_n(arr):
        n = int(arr.size)
        m = float(np.mean(arr)) if n else float("nan")
        s = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        return m, s, n

    m1, s1, n1 = _mean_std_n(grp_has_ge2)
    m2, s2, n2 = _mean_std_n(grp_no_ge2)
    t_stat, df_t, p_val, method_label = welch_t_with_scipy(grp_has_ge2, grp_no_ge2)
    d = cohens_d(m1, s1, n1, m2, s2, n2)
    g = hedges_g(d, n1, n2)

    # Build CSV
    rows = []
    rows.append(("Total time series $\\mathcal{T}_{c,i}$", total_series, "-"))
    rows.append(("$\\mathcal{T}_{c,i}$ with any $Z_{c,i,t} \\geq 2$", with_ge2, f"{prop_with_ge2:.0%}"))
    rows.append(("Average number of $Z_{c,i,t} \\geq 2$ points per $\\mathcal{T}_{c,i}$", round(avg_ge2_points_overall, 2), "-"))
    rows.append(("$\\mathcal{T}_{c,i}$ with any $Z_{c,i,t} \\leq -2$", with_leneg2, f"{prop_with_leneg2:.0%}"))
    rows.append(("Average number of $Z_{c,i,t} \\leq -2$ points per $\\mathcal{T}_{c,i}$", round(avg_leneg2_points_overall, 2), "-"))
    rows.append(("Welch t-test on $\\rho$: method", method_label, "-"))
    rows.append(("Welch t-test on $\\rho$: group1 (has any $Z\\geq2$) n", n1, "-"))
    rows.append(("Welch t-test on $\\rho$: group1 mean", round(m1, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: group1 std", round(s1, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: group2 (no $Z\\geq2$) n", n2, "-"))
    rows.append(("Welch t-test on $\\rho$: group2 mean", round(m2, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: group2 std", round(s2, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: mean diff (g1-g2)", round(m1 - m2, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: t statistic", round(t_stat, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: df", round(df_t, 6), "-"))
    rows.append(("Welch t-test on $\\rho$: p-value (two-sided)", "{:.3e}".format(p_val), "-"))
    rows.append(("Effect size: Cohen's d", round(d, 6), "-"))
    rows.append(("Effect size: Hedges' g", round(g, 6), "-"))

    summary_df = pd.DataFrame(rows, columns=["Metric", "Value", "Proportion"])
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
    summary_df.to_csv(out_summary_path, index=False)
    print(f"Saved summary to: {out_summary_path}")

    return z_df_seg, year_cols_seg  # return for possible combination later


# ========= Main run =========
if __name__ == "__main__":
    # Per-segment summaries, and keep pieces for the combined figure
    combined_df = None
    combined_year_cols = []

    for seg in SEGMENTS:
        z_path = Z_PATHS[seg]
        rho_path = RHO_PATHS[seg]
        if not os.path.exists(z_path):
            raise FileNotFoundError(f"Missing Z file for segment {seg}: {z_path}")
        if not os.path.exists(rho_path):
            raise FileNotFoundError(f"Missing rho file for segment {seg}: {rho_path}")

        z_df_seg = pd.read_csv(z_path)
        rho_df_seg = pd.read_csv(rho_path)
        out_summary = OUT_SUMMARY_TMPL.format(seg=seg)

        # Build and save per-segment summary
        z_df_seg_norm, year_cols_seg = build_segment_summary(z_df_seg, rho_df_seg, out_summary)

        # Accumulate for combined plot:
        if combined_df is None:
            combined_df = z_df_seg_norm.copy()
        else:
            new_years = [c for c in year_cols_seg if c not in combined_df.columns]
            combined_df = pd.merge(
                combined_df,
                z_df_seg_norm[["Country", "IPCR"] + new_years],
                on=["Country", "IPCR"],
                how="outer"
            )

        for y in year_cols_seg:
            if y not in combined_year_cols:
                combined_year_cols.append(y)

    # Plot once with all segments combined
    if combined_df is None or not combined_year_cols:
        raise ValueError("No data aggregated for the combined figure.")
    combined_year_cols = sorted(combined_year_cols, key=lambda x: int(x))
    combined_df = normalize_ci_columns(combined_df)  # ensure Country/IPCR standardized

    plot_time_lines_counts_with_bands(
        z_df=combined_df,
        year_cols=combined_year_cols,
        country_codes=COUNTRY_CODES,
        pos_thr=POS_THR,
        start_year=YEAR_START,
        out_path_base=OUT_FIG_BASE
    )

    print(f"Saved figure PNG/PDF to base: {OUT_FIG_BASE}")
