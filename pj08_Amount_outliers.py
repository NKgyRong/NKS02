#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp

# ========= I/O =========
INPUT_PATH = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr.csv"
OUTPUT_DIR = "/data01/rong_dataset/Result/pj08/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filenames are per-segment, e.g.:
# WWP_1860-1969_Z_c_i_t.csv
# WWP_1860-1969_Z_delta_c_i_t.csv
# WWP_1860-1969_Z_nhat_c_i_t.csv
# WWP_1860-1969_rho_c_i.csv

# ========= Parallel + Params =========
NUM_WORKERS = min(60, mp.cpu_count())
CHUNKSIZE = 256
W1 = 1.0
W2 = 1.0
ALPHA = 0.05  # significance level for correlation

# ========= Segments (inclusive year ranges) =========
SEGMENTS = [
    (1860, 1969),
    (1970, 2010),
    (2011, 2024),
]


def select_year_cols(columns):
    year_cols = []
    for col in columns:
        try:
            yr = int(float(col))
            if yr <= 2024:
                year_cols.append(col)
        except Exception:
            continue
    return year_cols


def corr_pvalue_fisher(r, n):
    """
    Two-sided p-value for Pearson correlation using Fisher z transform.
    n: number of paired observations (common years).
    """
    if n is None or n < 4 or np.isnan(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n - 3.0)
    # two-sided p under N(0,1)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return p


def compute_one_row_segment(args):
    """
    Compute Z_delta, Z_nhat (formerly Z_y), Z_comb, rho and p for a single row within ONE segment.

    args:
        idx: row index
        r_full: full-length row values (float array of length n_years_full)
        years_full: list of int years for the full matrix columns
        seg_start: int, inclusive
        seg_end: int, inclusive

    returns:
        (
          idx,
          seg_key,             # e.g. "1860-1969"
          z_delta_row_seg,     # shape = len(seg_years)
          z_nhat_row_seg,      # shape = len(seg_years)
          z_comb_row_seg,      # shape = len(seg_years)
          rho_value,           # float
          p_value              # float
        )
    """
    (idx, r_full, years_full, seg_start, seg_end) = args
    seg_key = f"{seg_start}-{seg_end}"

    # Map full years to indices for quick access
    year_to_pos = {y: i for i, y in enumerate(years_full)}

    # Build segment mask and year list
    seg_years = [y for y in years_full if seg_start <= y <= seg_end]
    m = len(seg_years)
    if m == 0:
        # No years in this segment (should not happen); return NaNs
        nan_vec = np.full(0, np.nan)
        return idx, seg_key, nan_vec, nan_vec, nan_vec, np.nan, np.nan

    # Extract raw values for segment (note: we still need r_{t-1} for delta at the first year)
    r_seg = np.array([r_full[year_to_pos[y]] for y in seg_years], dtype=float)

    # ---------- Z_nhat (formerly Z_y): level component within segment ----------
    # Start index for Z_nhat within the segment: first non-zero within the segment
    r_for_mask_seg = np.nan_to_num(r_seg, nan=0.0)
    nz_seg = np.nonzero(r_for_mask_seg != 0.0)[0]
    z_nhat_row = np.full(m, np.nan, dtype=float)
    has_zn = False
    if nz_seg.size > 0:
        s = int(nz_seg[0])
        # y = log(1 + x) within segment from s onward
        y_seg = np.log1p(r_seg[s:])
        mu_y = np.nanmean(y_seg)
        sigma_y = np.nanstd(y_seg, ddof=0)
        if not (np.isnan(sigma_y) or sigma_y == 0.0):
            z_nhat_row[s:] = (y_seg - mu_y) / sigma_y
            has_zn = True

    # ---------- Z_delta within segment with cross-boundary diff at segment start ----------
    # diff at year t is v[t] - v[t-1]; for the first year IN THE SEGMENT, use the previous calendar year from the FULL series.
    z_delta_row = np.full(m, np.nan, dtype=float)
    has_zd = False

    # Build plain (non-standardized) diffs for the segment years
    diffs = np.full(m, np.nan, dtype=float)
    for j, y in enumerate(seg_years):
        pos_t = year_to_pos[y]
        y_prev = y - 1
        v_t = r_full[pos_t]

        if y_prev in year_to_pos:
            pos_prev = year_to_pos[y_prev]
            v_prev = r_full[pos_prev]
            if np.isnan(v_prev):
                v_prev = 0.0   # <-- change: treat missing previous as 0
        else:
            v_prev = 0.0       # <-- change: no previous year ¡ú baseline 0

        if np.isnan(v_t):
            diffs[j] = np.nan
        else:
            diffs[j] = v_t - v_prev

    # standardize diffs within segment using segment-only stats
    # (ignore NaN; require non-zero std)
    if np.any(~np.isnan(diffs)):
        mu_d = np.nanmean(diffs)
        sigma_d = np.nanstd(diffs, ddof=0)
        if not (np.isnan(sigma_d) or sigma_d == 0.0):
            z_delta_row = (diffs - mu_d) / sigma_d
            has_zd = True

    # ---------- Combine with correlation adjustment and significance gating ----------
    z_comb_row = np.full(m, np.nan, dtype=float)
    rho = np.nan
    pval = np.nan

    if has_zn and has_zd:
        common_mask = (~np.isnan(z_delta_row)) & (~np.isnan(z_nhat_row))
        common_idx = np.nonzero(common_mask)[0]

        if common_idx.size >= 2:
            a = z_delta_row[common_idx]
            b = z_nhat_row[common_idx]
            if np.std(a) == 0.0 or np.std(b) == 0.0:
                rho = 0.0
                pval = 1.0
            else:
                rho = float(np.corrcoef(a, b)[0, 1])
                rho = float(np.clip(rho, -0.99, 0.99))
                pval = corr_pvalue_fisher(rho, common_idx.size)
        else:
            rho = 0.0
            pval = np.nan

        rho_used = rho if (not np.isnan(pval) and pval < ALPHA) else 0.0
        denom = math.sqrt(W1 * W1 + W2 * W2 + 2.0 * rho_used * W1 * W2)

        if common_idx.size > 0 and denom > 0.0:
            z_comb_row[common_idx] = (W1 * z_delta_row[common_idx] + W2 * z_nhat_row[common_idx]) / denom

        # fill positions that have only one component
        only_n = (~np.isnan(z_nhat_row)) & (np.isnan(z_delta_row))
        if np.any(only_n):
            z_comb_row[only_n] = z_nhat_row[only_n]

        only_d = (~np.isnan(z_delta_row)) & (np.isnan(z_nhat_row))
        if np.any(only_d):
            z_comb_row[only_d] = z_delta_row[only_d]

    elif has_zn and not has_zd:
        z_comb_row = z_nhat_row.copy()
        rho = np.nan
        pval = np.nan

    elif has_zd and not has_zn:
        z_comb_row = z_delta_row.copy()
        rho = np.nan
        pval = np.nan

    return idx, seg_key, z_delta_row, z_nhat_row, z_comb_row, rho, pval


def main():
    # ===== Read input and identify columns =====
    df = pd.read_csv(INPUT_PATH)
    id_cols = ["Country", "IPCR"]

    # Collect year columns from the third column onward
    all_year_cols = df.columns[2:]
    year_cols_all = select_year_cols(all_year_cols)

    # Convert to numeric values
    values_all = df[year_cols_all].apply(pd.to_numeric, errors="coerce")
    vals_full = values_all.to_numpy(dtype=float, copy=False)
    n_rows, n_years_full = vals_full.shape

    # Convert year headers to int for internal handling
    years_full = [int(float(c)) for c in year_cols_all]

    # ===== Per-segment computation =====
    # We will compute and write separate CSVs per segment, each only with columns of that segment.
    for (seg_start, seg_end) in SEGMENTS:
        seg_key = f"{seg_start}-{seg_end}"
        seg_years = [y for y in years_full if seg_start <= y <= seg_end]
        if len(seg_years) == 0:
            # Nothing to do for this segment
            continue

        # Pre-allocate outputs for this segment
        m = len(seg_years)
        Z_DELTA_seg = np.full((n_rows, m), np.nan, dtype=float)
        Z_NHAT_seg = np.full((n_rows, m), np.nan, dtype=float)
        Z_COMB_seg = np.full((n_rows, m), np.nan, dtype=float)
        RHO_seg = np.full(n_rows, np.nan, dtype=float)
        PVAL_seg = np.full(n_rows, np.nan, dtype=float)

        # Prepare iterator
        row_iter = (
            (i, vals_full[i, :], years_full, seg_start, seg_end)
            for i in range(n_rows)
        )

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            for (
                idx,
                _seg_key,
                zd_row,
                zn_row,
                zc_row,
                rho,
                pval,
            ) in tqdm(
                ex.map(compute_one_row_segment, row_iter, chunksize=CHUNKSIZE),
                total=n_rows,
                desc=f"Computing Z-delta/Z-nhat/Z-comb for {seg_key}",
            ):
                Z_DELTA_seg[idx, :] = zd_row
                Z_NHAT_seg[idx, :] = zn_row
                Z_COMB_seg[idx, :] = zc_row
                RHO_seg[idx] = rho
                PVAL_seg[idx] = pval

        # Build id df and per-segment year columns (as strings)
        id_df = df[id_cols].reset_index(drop=True)
        seg_year_cols_str = [str(y) for y in seg_years]

        # ----- Write per-segment outputs -----
        out_z_comb = os.path.join(OUTPUT_DIR, f"WWP_{seg_key}_Z_c_i_t.csv")
        out_z_delta = os.path.join(OUTPUT_DIR, f"WWP_{seg_key}_Z_delta_c_i_t.csv")
        out_z_nhat = os.path.join(OUTPUT_DIR, f"WWP_{seg_key}_Z_nhat_c_i_t.csv")
        out_rho = os.path.join(OUTPUT_DIR, f"WWP_{seg_key}_rho_c_i.csv")

        zcomb_df = pd.concat([id_df, pd.DataFrame(Z_COMB_seg, columns=seg_year_cols_str)], axis=1)
        zcomb_df.to_csv(out_z_comb, index=False)

        zdelta_df = pd.concat([id_df, pd.DataFrame(Z_DELTA_seg, columns=seg_year_cols_str)], axis=1)
        zdelta_df.to_csv(out_z_delta, index=False)

        znhat_df = pd.concat([id_df, pd.DataFrame(Z_NHAT_seg, columns=seg_year_cols_str)], axis=1)
        znhat_df.to_csv(out_z_nhat, index=False)

        rho_df = pd.DataFrame(
            {
                "Country": df["Country"].values,
                "IPCR": df["IPCR"].values,
                "rho": RHO_seg,
                "P": PVAL_seg,
            }
        )
        rho_df.to_csv(out_rho, index=False)


if __name__ == "__main__":
    main()
