#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RF-only regression (two periods) + SHAP explainability
=====================================================
- Keep X/Y construction exactly the same as previous pipeline.
- Remove LASSO & OLS completely. Use ONLY RandomForestRegressor.
- Feed ALL features that were given to LASSO into RF.
- Group-aware CV by IPCR (GroupKFold) to report OOS R2 / RMSE.
- Export RF built-in MDI, permutation importance (CV folds), and SHAP outputs.

SHAP exports per period:
  1) SHAP_{period}_mean_abs.csv           # global mean(|SHAP|) ranking
  2) SHAP_{period}_summary_bar.png        # bar summary
  3) SHAP_{period}_summary_beeswarm.png   # beeswarm summary
  4) SHAP_{period}_dependence_TOP10_*.png # dependence plots for top-10 features
  5) SHAP_{period}_values.csv.gz          # (large) per-sample SHAP matrix with metadata

Notes:
- RF is trained on raw X (no standardization).
- Matplotlib backend is set to 'Agg' for headless environments.
- If 'shap' is not installed, script prints a clear hint and skips SHAP parts.

Reference (for methodology / plots):
- DataCamp: "Explainable AI - Understanding and Trusting ML Models" (LIME & SHAP overview)
"""

import os
import re
import gc
import math
import shutil
import warnings
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ---------- Try importing shap (soft dependency) ----------
try:
    import shap
    _HAS_SHAP = True
except Exception as _e:
    _HAS_SHAP = False


# ===================== Paths & Config =====================
X_XLSX = "/data01/Rong_backup/World_Development_Indicators/Regression_x.xlsx"
WDI_CSV = "/data01/Rong_backup/World_Development_Indicators/WDI_data.csv"

# y split into two periods (files must exist)
Y_CSV_PERIODS = {
    "Y1_1970_2010": "/data01/rong_dataset/Result/pj08/1970-2010_Regression_y.csv",
    "Y2_2011_2024": "/data01/rong_dataset/Result/pj08/2011-2024_Regression_y.csv",
}

RESULT_DIR = "/data01/rong_dataset/Result/pj08/Regression_result_RF_SHAP"
TEMP_DIR = "/data01/rong_dataset/temp"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# consolidated report (human-readable)
RF_TXT_PATH = os.path.join(RESULT_DIR, "RF_importance_SHAP_all.txt")

# WDI reading
CSV_CHUNKSIZE = 200000

# Numerics
EPS = 1e-12

# CV config
N_FOLDS_MAX = 5
RANDOM_STATE = 42

# Random Forest config (you can tune if needed)
RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1
RF_N_JOBS = -1

RF_PERM_REPEATS = 10


# -----------------------------------------------------------------------------------
# Indicator categories (kept exactly as previous version for X construction)
# -----------------------------------------------------------------------------------
ABS_CODES_27 = {
    "SM.POP.NETM", "PA.NUS.ATLS", "PA.NUS.FCRF", "AG.YLD.CREL.KG",
    "EN.GHG.ALL.PC.CE.AR5", "FI.RES.TOTL.CD", "EG.USE.ELEC.KH.PC",
    "SP.POP.TOTL", "EN.POP.DNST", "NY.GDP.MKTP.CD", "NY.GDP.PCAP.CD",
    "SH.DTH.MORT",
}
PCT_CODES_27 = {
    "SP.ADO.TFRT", "SP.POP.BRTH.MF", "SP.DYN.CDRT.IN", "AG.LND.AGRI.ZS",
    "NY.ADJ.AEDU.GN.ZS", "MS.MIL.XPND.GD.ZS", "BX.KLT.DINV.WD.GD.ZS",
    "SP.URB.TOTL.IN.ZS", "NE.TRD.GNFS.ZS", "NE.CON.TOTL.ZS",
    "SP.DYN.IMRT.IN", "TM.VAL.MRCH.HI.ZS", "TX.VAL.MRCH.HI.ZS",
}
GROWTH_CODES_27 = {
    "NY.GDP.DEFL.KD.ZG", "FM.AST.CGOV.ZG.M3",
}


# ===================== Year column detection =====================
YEAR_PLAIN = re.compile(r"^\s*(\d{4})\s*$")
YEAR_BRACKET = re.compile(r"^\s*(\d{4})\s*\[YR\1\]\s*$", re.IGNORECASE)


def detect_year_cols(columns: List[str]) -> Tuple[List[str], Dict[str, str], List[int]]:
    year_cols = []
    norm_map = {}
    for c in columns:
        s = str(c).strip()
        m1 = YEAR_PLAIN.match(s)
        m2 = YEAR_BRACKET.match(s)
        if m1:
            y = m1.group(1)
            year_cols.append(c)
            norm_map[c] = y
        elif m2:
            y = m2.group(1)
            year_cols.append(c)
            norm_map[c] = y
    years = sorted({int(v) for v in norm_map.values()})
    return year_cols, norm_map, years


def safe_numeric(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip()
    if s in ("", ".."):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# ===================== Load feature meta =====================
def load_x_meta(xlsx_path: str) -> Tuple[List[str], Dict[str, str]]:
    xmeta = pd.read_excel(xlsx_path, engine="openpyxl")
    cols = {c.lower().strip(): c for c in xmeta.columns}
    if "series code" not in cols:
        raise ValueError("Regression_x.xlsx must contain column 'Series Code'.")
    sc_col = cols["series code"]

    sname_col = None
    for cand in ["Series Name", "Indicator Name", "series name", "indicator name"]:
        if cand in xmeta.columns:
            sname_col = cand
            break

    xmeta = xmeta[[sc_col] + ([sname_col] if sname_col else [])].copy()
    xmeta[sc_col] = xmeta[sc_col].astype(str).str.strip()
    series_codes = xmeta[sc_col].astype(str).tolist()

    scode_to_sname: Dict[str, str] = {}
    if sname_col:
        for sc, nm in zip(xmeta[sc_col], xmeta[sname_col]):
            nm_str = str(nm).strip() if isinstance(nm, str) else ""
            if nm_str:
                scode_to_sname[sc] = nm_str
    return series_codes, scode_to_sname


# ===================== Read WDI and build time series =====================
def build_wdi_ts_map(series_codes_all: List[str]) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, str], List[int]]:
    head = pd.read_csv(WDI_CSV, nrows=1, dtype=str, keep_default_na=False, low_memory=False)
    year_cols_orig, norm_map, all_years = detect_year_cols(head.columns)
    if not year_cols_orig:
        raise ValueError("No year columns detected in WDI_data.csv")

    has_series_name = "Series Name" in head.columns
    year_to_idx = {y: i for i, y in enumerate(all_years)}
    ylen = len(all_years)

    ts_map: Dict[Tuple[str, str], np.ndarray] = {}
    scode_to_sname: Dict[str, str] = {}

    use_cols = ["Country Code", "Series Code"] + (["Series Name"] if has_series_name else []) + [c for c in year_cols_orig if c in head.columns]

    reader = pd.read_csv(
        WDI_CSV,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        usecols=[c for c in use_cols if c in head.columns],
        chunksize=CSV_CHUNKSIZE
    )

    for chunk in tqdm(reader, desc=" - Scanning WDI", unit="chunk", dynamic_ncols=True):
        sub = chunk[chunk["Series Code"].astype(str).isin(series_codes_all)]
        if sub.empty:
            continue
        sub = sub[use_cols].copy()

        for _, row in sub.iterrows():
            ccode = str(row["Country Code"]).strip()
            scode = str(row["Series Code"]).strip()

            if has_series_name:
                nm = str(row.get("Series Name", "")).strip()
                if nm and scode not in scode_to_sname:
                    scode_to_sname[scode] = nm

            key = (ccode, scode)
            arr = ts_map.get(key)
            if arr is None:
                arr = np.full(ylen, np.nan, dtype=float)
                ts_map[key] = arr

            for oc in year_cols_orig:
                if oc not in row:
                    continue
                y = int(norm_map[oc])
                v = safe_numeric(row.get(oc, np.nan))
                arr[year_to_idx[y]] = v

    return ts_map, scode_to_sname, all_years


# ===================== Country code helpers =====================
def build_wdi_iso3_set(ts_map: Dict[Tuple[str, str], np.ndarray]) -> Set[str]:
    return set(cc for (cc, sc) in ts_map.keys())


PREDS = {"SU", "CS", "YU"}
EXTRA_MAP = {
    "UK": "GBR", "EL": "GRC", "XK": "XKX", "MO": "MAC", "HK": "HKG", "TW": "TWN",
    "PS": "PSE", "VA": "VAT", "SS": "SSD", "TL": "TLS", "CW": "CUW", "SX": "SXM",
    "BQ": "BES", "CG": "COG", "CD": "COD", "CI": "CIV", "KP": "PRK", "LA": "LAO",
    "VN": "VNM", "MM": "MMR", "BO": "BOL", "IR": "IRN", "SY": "SYR", "MK": "MKD",
}

def iso2_to_iso3_try(c2: str, wdi_iso3_set: Set[str]) -> str:
    c2u = c2.upper()
    if c2u in EXTRA_MAP:
        return EXTRA_MAP[c2u]
    common = {
        "US": "USA", "CN": "CHN", "RU": "RUS", "UA": "UKR", "GB": "GBR",
        "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP", "PT": "PRT",
        "NL": "NLD", "BE": "BEL", "LU": "LUX", "IE": "IRL", "DK": "DNK",
        "SE": "SWE", "NO": "NOR", "FI": "FIN", "IS": "ISL", "JP": "JPN",
        "KR": "KOR", "IN": "IND", "SG": "SGP", "MY": "MYS", "TH": "THA",
        "ID": "IDN", "PH": "PHL", "AU": "AUS", "NZ": "NZL", "CA": "CAN",
        "MX": "MEX", "BR": "BRA", "AR": "ARG", "CL": "CHL", "PE": "PER",
        "CO": "COL", "ZA": "ZAF", "EG": "EGY", "MA": "MAR", "DZ": "DZA",
        "TN": "TUN", "TR": "TUR", "GR": "GRC", "PL": "POL", "CZ": "CZE",
        "SK": "SVK", "HU": "HUN", "RO": "ROU", "BG": "BGR", "EE": "EST",
        "LV": "LVA", "LT": "LTU", "MD": "MDA", "GEO": "GEO", "BY": "BLR",
        "AM": "ARM", "AZ": "AZE", "KZ": "KAZ", "KG": "KGZ", "TJ": "TJK",
        "TM": "TKM", "UZ": "UZB", "RS": "SRB", "ME": "MNE", "HR": "HRV",
        "SI": "SVN", "BA": "BIH", "AL": "ALB", "NG": "NGA", "KE": "KEN",
        "TZ": "TZA", "UG": "UGA", "GH": "GHA", "SN": "SEN", "CM": "CMR",
        "ET": "ETH", "SD": "SDN",
    }
    c3 = common.get(c2u, "")
    if c3 and c3 in wdi_iso3_set:
        return c3
    return ""


def pick_successor_token(raw: str) -> str:
    toks = [t.strip() for t in str(raw).split("/") if str(t).strip()]
    if not toks:
        return ""
    for t in toks:
        if t.upper() not in PREDS:
            return t
    return toks[0]


def canon_country_to_wdi(code_raw: str, wdi_iso3_set: Set[str]) -> str:
    if not isinstance(code_raw, str):
        return ""
    c = code_raw.strip()
    if c == "":
        return ""
    if "/" in c:
        c = pick_successor_token(c)
    cu = c.upper()
    if len(cu) == 3 and cu in wdi_iso3_set:
        return cu
    if len(cu) == 2:
        c3 = iso2_to_iso3_try(cu, wdi_iso3_set)
        if c3 and c3 in wdi_iso3_set:
            return c3
    return ""


# ===================== x-construction (same as before) =====================
def avg_ratio_and_raw_country(
    pioneer_iso3: str,
    country_iso3: str,
    scode: str,
    t_start: int,
    t_end: int,
    year_to_idx: Dict[int, int],
    ts_map: Dict[Tuple[str, str], np.ndarray],
    min_year: int,
    max_year: int
) -> Tuple[float, float]:
    # identical to previous version
    if np.isnan(t_start) or np.isnan(t_end):
        return np.nan, np.nan
    t1 = int(t_start); t2 = int(t_end)
    if t2 <= t1:
        return np.nan, np.nan
    if t1 < min_year:
        t1 = min_year
    if t2 > max_year:
        t2 = max_year
    if t2 <= t1:
        return np.nan, np.nan

    kp = (pioneer_iso3, scode)
    kc = (country_iso3, scode)
    if kp not in ts_map or kc not in ts_map:
        return np.nan, np.nan

    ap = ts_map[kp]
    ac = ts_map[kc]
    i1 = year_to_idx.get(t1, None)
    i2 = year_to_idx.get(t2, None)
    if i1 is None or i2 is None:
        return np.nan, np.nan

    rpsg = float(t2 - t1)
    sum_diff = 0.0
    n_valid_pairs = 0
    raw_vals_c = []

    # category rules
    if (scode in PCT_CODES_27) or (scode in GROWTH_CODES_27):
        # diff = vp - vc
        for yidx in range(i1, i2 + 1):
            vp = ap[yidx]; vc = ac[yidx]
            if np.isnan(vp) or np.isnan(vc):
                continue
            sum_diff += (vp - vc)
            n_valid_pairs += 1
            raw_vals_c.append(vc)
    else:
        # log1p level diff
        for yidx in range(i1, i2 + 1):
            vp = ap[yidx]; vc = ac[yidx]
            if np.isnan(vp) or np.isnan(vc):
                continue
            lp = math.log(max(vp, 0.0) + 1.0)
            lc = math.log(max(vc, 0.0) + 1.0)
            sum_diff += (lp - lc)
            n_valid_pairs += 1
            raw_vals_c.append(vc)

    if n_valid_pairs == 0:
        return np.nan, np.nan

    x_val = sum_diff / (rpsg + 1.0)
    mean_raw_country = float(np.mean(raw_vals_c)) if len(raw_vals_c) > 0 else np.nan
    return x_val, mean_raw_country


def build_y_block(
    y_csv_path: str,
    all_years: List[int],
    ts_map: Dict[Tuple[str, str], np.ndarray],
    z2_fill_year: Optional[int] = None,
    period_clip_max: Optional[int] = None
) -> pd.DataFrame:
    # identical to previous for Y building
    min_year, max_year = min(all_years), max(all_years)
    wdi_iso3_set = build_wdi_iso3_set(ts_map)

    ydf_raw = pd.read_csv(y_csv_path, dtype=str)
    req_y = {"IPCR", "Country", "Pioneer Time", "Z>=2 Time", "Gap"}
    missing = req_y - set(ydf_raw.columns)
    if missing:
        raise ValueError(f"{y_csv_path} missing columns: {missing}")

    ydf_raw["Pioneer Time"] = pd.to_numeric(ydf_raw["Pioneer Time"], errors="coerce")
    z2 = ydf_raw["Z>=2 Time"].astype(str).str.strip()
    if z2_fill_year is not None:
        z2 = z2.mask(z2.isin(["", "nan", "None"]), str(z2_fill_year))
    ydf_raw["Z>=2 Time"] = pd.to_numeric(z2, errors="coerce")

    ydf = ydf_raw.copy()
    upper_clip = period_clip_max if period_clip_max is not None else max_year
    ydf["Pioneer Time"] = pd.to_numeric(ydf["Pioneer Time"], errors="coerce").clip(lower=min_year, upper=upper_clip)
    ydf["Z>=2 Time"] = pd.to_numeric(ydf["Z>=2 Time"], errors="coerce").clip(lower=min_year, upper=upper_clip)

    ydf["Country_WDI"] = ydf["Country"].apply(lambda c: canon_country_to_wdi(str(c), wdi_iso3_set))

    gap_num = pd.to_numeric(ydf["Gap"], errors="coerce")
    is_pioneer_row = (gap_num == 0)

    pioneer_rows = ydf.loc[is_pioneer_row & (ydf["Country_WDI"] != ""), ["IPCR", "Country_WDI"]].copy()
    pioneer_map = pioneer_rows.drop_duplicates("IPCR").set_index("IPCR")["Country_WDI"].to_dict()

    missing_ipcr = set(ydf["IPCR"].unique()) - set(pioneer_map.keys())
    if missing_ipcr:
        tmp = ydf[["IPCR", "Country_WDI", "Pioneer Time"]].dropna(subset=["Pioneer Time"]).copy()
        if not tmp.empty:
            idx_min = tmp.groupby("IPCR")["Pioneer Time"].idxmin()
            fallback = tmp.loc[idx_min].set_index("IPCR")["Country_WDI"].to_dict()
        else:
            fallback = {}
        for k in missing_ipcr:
            pioneer_map[k] = fallback.get(k, "")

    ydf["Pioneer_WDI"] = ydf["IPCR"].map(pioneer_map).fillna("")
    ydf["Gap_Raw"] = (ydf["Z>=2 Time"] - ydf["Pioneer Time"]).astype(float)

    ydf = ydf[(ydf["Gap_Raw"] > 0) & (ydf["Country_WDI"] != "") & (ydf["Pioneer_WDI"] != "")]
    ydf["y"] = ydf["Gap_Raw"].astype(float)
    return ydf


# ===================== RF helpers =====================
def _append_lines(path: str, lines: List[str]):
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + ("\n" if not ln.endswith("\n") else ""))


def _rf_cv_stats(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
    gkf = GroupKFold(n_splits=n_splits)
    r2_list, rmse_list = [], []
    perm_vals = []
    for tr, va in gkf.split(X, y, groups=groups):
        Xtr = X[tr]; ytr = y[tr]; Xva = X[va]; yva = y[va]
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=RF_N_JOBS,
            bootstrap=True,
            oob_score=False
        )
        rf.fit(Xtr, ytr)
        yhat = rf.predict(Xva)
        resid = yva - yhat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((yva - np.mean(yva)) ** 2)) + EPS
        r2 = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        r2_list.append(r2); rmse_list.append(rmse)

        try:
            pi = permutation_importance(
                rf, Xva, yva,
                n_repeats=RF_PERM_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=RF_N_JOBS
            )
            perm_vals.append(pi.importances_mean)
        except Exception:
            perm_vals.append(np.full(X.shape[1], np.nan))

    perm_vals = np.vstack(perm_vals) if perm_vals else np.empty((0, X.shape[1]))
    perm_mean = np.nanmean(perm_vals, axis=0) if perm_vals.size else np.array([])
    perm_std  = np.nanstd(perm_vals, axis=0)  if perm_vals.size else np.array([])
    return r2_list, rmse_list, perm_mean, perm_std


# ===== SHAP parallel helper (no sampling, full data) =====
def _compute_shap_parallel_full(explainer, X, *, n_jobs=-1, chunk_size=5000, check_additivity=False):
    import math
    from joblib import Parallel, delayed

    N = X.shape[0]
    if N == 0:
        return np.empty((0, X.shape[1]), dtype=float)

    n_chunks = math.ceil(N / chunk_size)
    chunks = [X[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    def _work(Xc):
        return explainer.shap_values(Xc, check_additivity=check_additivity)

    shap_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_work)(c) for c in chunks
    )
    shap_values = np.vstack(shap_list)
    return shap_values
# ===== END helper =====


# ===================== RF modeling per period + SHAP =====================
def run_rf_for_period(
    period_tag: str,
    y_csv_path: str,
    series_codes_all: List[str],
    ts_map: Dict[Tuple[str, str], np.ndarray],
    all_years: List[int],
    scode_to_sname_seed: Dict[str, str]
):
    print(f"\n[Period] {period_tag}: reading y and building design matrix ...")

    # policy on z2 fill / clip
    if period_tag.startswith("Y1"):
        z2_fill = 2010
        clip_max = 2010
    elif period_tag.startswith("Y2"):
        z2_fill = 2024
        clip_max = 2024
    else:
        z2_fill = None
        clip_max = max(all_years)

    ydf = build_y_block(
        y_csv_path=y_csv_path,
        all_years=all_years,
        ts_map=ts_map,
        z2_fill_year=z2_fill,
        period_clip_max=clip_max
    )
    if ydf.empty:
        print(f" - No rows after filtering for {period_tag}. Skipping.")
        return

    min_year, max_year = min(all_years), max(all_years)
    year_to_idx = {y: i for i, y in enumerate(all_years)}

    scode_to_sname = dict(scode_to_sname_seed)
    for sc in series_codes_all:
        if sc not in scode_to_sname:
            scode_to_sname[sc] = sc

    base_cols = ["IPCR", "Country", "Country_WDI", "Pioneer_WDI", "Pioneer Time", "Z>=2 Time", "Gap_Raw", "y"]
    model_base = ydf[base_cols].copy()

    N = len(model_base)
    F = len(series_codes_all)
    X_full = np.full((N, F), np.nan, dtype=float)

    print(" - Computing X rows (ALL features into RF) ...")
    for i in tqdm(range(N), desc=" rows", unit="row", dynamic_ncols=True):
        row = model_base.iloc[i]
        ccode = str(row["Country_WDI"]).strip()
        pcode = str(row["Pioneer_WDI"]).strip()
        t_p = int(row["Pioneer Time"])
        t_c = int(row["Z>=2 Time"])
        for j, scode in enumerate(series_codes_all):
            avg_ratio, _ = avg_ratio_and_raw_country(
                pcode, ccode, scode, t_p, t_c, year_to_idx, ts_map, min_year, max_year
            )
            X_full[i, j] = avg_ratio

    y_vec = model_base["y"].astype(float).to_numpy()
    mask = (~np.isnan(y_vec)) & (~np.isnan(X_full).any(axis=1))
    X = X_full[mask, :]
    y = y_vec[mask]
    meta = model_base.loc[mask].reset_index(drop=True)
    ipcr_groups = meta["IPCR"].astype(str).to_numpy()

    if X.shape[0] < 10:
        print(f" - Too few rows after complete-case for {period_tag}: n={X.shape[0]}. Skip.")
        return

    # feature names for interpretability (prefer Series Name)
    feat_labels = [scode_to_sname.get(sc, sc) for sc in series_codes_all]

    # ----------- Export design (may be large) -----------
    design_df = meta[["IPCR", "Country", "Country_WDI", "Pioneer_WDI", "Pioneer Time", "Z>=2 Time"]].copy()
    design_df["y"] = y
    for j, disp in enumerate(feat_labels):
        design_df[disp] = X[:, j]
    design_csv = os.path.join(RESULT_DIR, f"RF_{period_tag}_design.csv")
    design_df.to_csv(design_csv, index=False)

    # ----------- CV eval (GroupKFold) -----------
    uniq_groups = np.unique(ipcr_groups)
    n_splits = int(min(N_FOLDS_MAX, len(uniq_groups)))
    if n_splits < 2:
        n_splits = 2

    rf_cv_r2, rf_cv_rmse, perm_mean, perm_std = _rf_cv_stats(X, y, ipcr_groups, n_splits)
    rf_cv_rows = [{"fold_id": i+1, "R2": float(r2i), "RMSE": float(rmsei)} for i, (r2i, rmsei) in enumerate(zip(rf_cv_r2, rf_cv_rmse))]
    rf_cv_df = pd.DataFrame(rf_cv_rows, columns=["fold_id", "R2", "RMSE"])
    rf_cv_csv = os.path.join(RESULT_DIR, f"RF_{period_tag}_cv_folds.csv")
    rf_cv_df.to_csv(rf_cv_csv, index=False)

    # ----------- Fit full RF (for importance + SHAP) -----------
    print(" - Fitting full RandomForestRegressor ...")
    rf_full = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        bootstrap=True,
        oob_score=True
    )
    rf_full.fit(X, y)
    oob_r2 = getattr(rf_full, "oob_score_", np.nan)

    # MDI importances
    fi = rf_full.feature_importances_
    order_mdi = np.argsort(-fi)
    mdi_rows = [{"rank": int(r+1), "feature": feat_labels[j], "series_code": series_codes_all[j], "mdi": float(fi[j])}
                for r, j in enumerate(order_mdi)]
    rf_mdi_df = pd.DataFrame(mdi_rows, columns=["rank", "feature", "series_code", "mdi"])
    rf_mdi_csv = os.path.join(RESULT_DIR, f"RF_{period_tag}_importance_mdi.csv")
    rf_mdi_df.to_csv(rf_mdi_csv, index=False)

    # Permutation importance (from CV result: perm_mean/std already computed on val folds)
    rf_perm_csv = os.path.join(RESULT_DIR, f"RF_{period_tag}_importance_perm.csv")
    if perm_mean.size:
        order_perm = np.argsort(-perm_mean)
        perm_rows = [{"rank": int(r+1), "feature": feat_labels[j], "series_code": series_codes_all[j],
                      "perm_mean": float(perm_mean[j]), "perm_std": float(perm_std[j])}
                     for r, j in enumerate(order_perm)]
        pd.DataFrame(perm_rows, columns=["rank", "feature", "series_code", "perm_mean", "perm_std"]).to_csv(rf_perm_csv, index=False)
    else:
        pd.DataFrame(columns=["rank", "feature", "series_code", "perm_mean", "perm_std"]).to_csv(rf_perm_csv, index=False)

    # ----------- SHAP explainability -----------
    shap_summary_bar_png = os.path.join(RESULT_DIR, f"SHAP_{period_tag}_summary_bar.png")
    shap_summary_bee_png = os.path.join(RESULT_DIR, f"SHAP_{period_tag}_summary_beeswarm.png")
    shap_mean_abs_csv = os.path.join(RESULT_DIR, f"SHAP_{period_tag}_mean_abs.csv")
    shap_values_csv_gz = os.path.join(RESULT_DIR, f"SHAP_{period_tag}_values.csv.gz")
    shap_depend_dir = os.path.join(RESULT_DIR, f"SHAP_{period_tag}_dependence_TOP10")
    os.makedirs(shap_depend_dir, exist_ok=True)

    if not _HAS_SHAP:
        print(" [WARN] shap is not installed. Run: pip install shap")
        _append_lines(RF_TXT_PATH, [
            "",
            "=" * 80,
            f"RANDOM FOREST + SHAP - {period_tag}",
            "=" * 80,
            f"n={len(y)}, p={X.shape[1]}, OOB_R2={oob_r2:.6g}",
            f"CV R2(mean)={np.mean(rf_cv_r2):.6g}, CV RMSE(mean)={np.mean(rf_cv_rmse):.6g}",
            f"MDI CSV: {rf_mdi_csv}",
            f"Perm CSV: {rf_perm_csv}",
            "SHAP: skipped (package not installed)."
        ])
    else:
        print(" - Computing SHAP values with TreeExplainer (this may take a while) ...")
        # For RF regression, TreeExplainer is appropriate.
        explainer = shap.TreeExplainer(rf_full, feature_names=feat_labels)
        shap_values = _compute_shap_parallel_full(
            explainer, X, n_jobs=-1, chunk_size=5000, check_additivity=False
        )

        # Mean absolute SHAP (global importance)
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_abs_rows = [{"rank": int(r+1), "feature": feat_labels[j], "series_code": series_codes_all[j],
                          "mean_abs_shap": float(mean_abs[j])}
                         for r, j in enumerate(np.argsort(-mean_abs))]
        pd.DataFrame(mean_abs_rows, columns=["rank", "feature", "series_code", "mean_abs_shap"]).to_csv(shap_mean_abs_csv, index=False)

        # SHAP summary bar
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=feat_labels, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(shap_summary_bar_png, dpi=200)
            plt.close()
        except Exception as e:
            print(f" [WARN] Failed to save SHAP bar summary: {e}")

        # SHAP summary beeswarm
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=feat_labels, show=False)
            plt.tight_layout()
            plt.savefig(shap_summary_bee_png, dpi=200)
            plt.close()
        except Exception as e:
            print(f" [WARN] Failed to save SHAP beeswarm summary: {e}")

        # SHAP dependence plots for top-10 features by mean|SHAP|
        try:
            topk = 10 if X.shape[1] >= 10 else X.shape[1]
            top_idx = np.argsort(-mean_abs)[:topk]
            for rank, j in enumerate(top_idx, 1):
                fname = re.sub(r"[^\w\-\.\(\) ]+", "_", feat_labels[j])  # safe filename
                out_png = os.path.join(shap_depend_dir, f"{rank:02d}_{fname}.png")
                plt.figure(figsize=(7, 5))
                shap.dependence_plot(
                    ind=j,
                    shap_values=shap_values,
                    features=X,
                    feature_names=feat_labels,
                    interaction_index=None,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(out_png, dpi=200)
                plt.close()
        except Exception as e:
            print(f" [WARN] Failed to save SHAP dependence plots: {e}")

        # (optional, large) Export per-sample SHAP matrix with metadata
        try:
            shap_df = pd.DataFrame(shap_values, columns=[f"shap::{sc}" for sc in series_codes_all])
            meta_cols = meta[["IPCR", "Country"]].copy()
            meta_cols["y_true"] = y.astype(float)
            base_val = explainer.expected_value
            if np.isscalar(explainer.expected_value):
                base_val = float(base_val)
            else:
                base_val = float(np.mean(explainer.expected_value))
            meta_cols["expected_value"] = float(base_val)
            out_df = pd.concat([meta_cols, shap_df], axis=1)
            out_df.to_csv(shap_values_csv_gz, index=False, compression="gzip")
        except Exception as e:
            print(f" [WARN] Failed to export per-sample SHAP values: {e}")

        _append_lines(RF_TXT_PATH, [
            "",
            "=" * 80,
            f"RANDOM FOREST + SHAP - {period_tag}",
            "=" * 80,
            f"Samples (n): {len(y)} | Features (p): {X.shape[1]}",
            f"OOB R2 (full fit): {oob_r2:.6g}",
            f"CV R2(mean)={np.mean(rf_cv_r2):.6g}, CV RMSE(mean)={np.mean(rf_cv_rmse):.6g}",
            f"MDI CSV: {rf_mdi_csv}",
            f"Perm CSV: {rf_perm_csv}",
            f"SHAP mean|SHAP| CSV: {shap_mean_abs_csv}",
            f"SHAP bar: {shap_summary_bar_png}",
            f"SHAP bees: {shap_summary_bee_png}",
            f"SHAP dependence dir: {shap_depend_dir}",
            f"SHAP values (per-sample): {shap_values_csv_gz}",
        ])

    print(f" [EXPORT] {period_tag}:")
    print(f" - design: {design_csv}")
    print(f" - RF CV folds: {rf_cv_csv}")
    print(f" - RF MDI: {rf_mdi_csv}")
    print(f" - RF Perm. import.: {rf_perm_csv}")
    if _HAS_SHAP:
        print(f" - SHAP mean|SHAP|: {shap_mean_abs_csv}")
        print(f" - SHAP summary (bar): {shap_summary_bar_png}")
        print(f" - SHAP summary (bee): {shap_summary_bee_png}")
        print(f" - SHAP dependence dir: {shap_depend_dir}")
        print(f" - SHAP values matrix: {shap_values_csv_gz}")
    else:
        print(" - SHAP outputs skipped (shap not installed).")


# ===================== MAIN =====================
def main():
    print("[1/4] Loading ALL feature codes from Excel ...")
    series_codes_all, scode_to_sname_seed = load_x_meta(X_XLSX)
    print(f" - Total features from Regression_x.xlsx: {len(series_codes_all)}")

    print("[2/4] Building WDI time series for selected indicators ...")
    ts_map, scode_to_sname_wdi, all_years = build_wdi_ts_map(series_codes_all)

    for sc, nm in scode_to_sname_wdi.items():
        if sc not in scode_to_sname_seed or not scode_to_sname_seed[sc].strip():
            scode_to_sname_seed[sc] = nm

    print(f" - Time series built: {len(ts_map)} keys, years {min(all_years)}..{max(all_years)}")

    # init consolidated txt
    with open(RF_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST IMPORTANCE + SHAP (Consolidated)\n")
        f.write("=" * 80 + "\n")
        if not _HAS_SHAP:
            f.write("NOTE: shap is not installed. Install with pip install shap to enable SHAP outputs.\n")

    print("[3/4] Running RF per period ...")
    for tag, ypath in Y_CSV_PERIODS.items():
        if not os.path.exists(ypath):
            print(f" - WARNING: {ypath} not found; skip {tag}")
            continue
        run_rf_for_period(
            period_tag=tag,
            y_csv_path=ypath,
            series_codes_all=series_codes_all,
            ts_map=ts_map,
            all_years=all_years,
            scode_to_sname_seed=scode_to_sname_seed
        )
        gc.collect()

    print("\n[4/4] Cleaning temp ...")
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception as e:
        warnings.warn(f"Failed to clean TEMP_DIR: {e}")

    print("\nDone.")
    print(f"Results directory -> {RESULT_DIR}")
    print(f"Consolidated RF+SHAP report : {RF_TXT_PATH}")
    if not _HAS_SHAP:
        print("Install SHAP if you want SHAP-based explainability: pip install shap")


if __name__ == "__main__":
    main()
