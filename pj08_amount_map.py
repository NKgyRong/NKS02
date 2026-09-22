#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
World patent choropleth with a collapsible 3-level sidebar (built-in map) + per-view timeline.
Updated to:
- Read WWP_Amount_ipcr.csv (series = country codes or pairwise like RU/SU).
- Country display names from WWP_Amount_ipcr_new_by_T_c_i.csv.
- Transition-aware rendering: before transition years, successors are hidden
  and their territories are colored as the predecessor (counted once).
- Add Hong Kong (China) micro point overlay like Singapore.

Outputs:
  /data01/rong_dataset/Result/pj08/world_patent_choropleth_sidebar.html
"""

import os
import re
import json
import html
import numpy as np
import pandas as pd
from string import Template

# -------------------------- Paths --------------------------
INPUT_CSV = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr.csv"
SERIES_NAME_CSV = "/data01/rong_dataset/Result/pj08/WWP_Amount_ipcr_new_by_T_c_i.csv"
IPCR_DESC_XLSX = "/data01/Rong_backup/Worldwide_Patent/IPCR.xlsx"
OUT_HTML = "/data01/rong_dataset/Result/pj08/world_patent_choropleth_sidebar.html"

# -------------------- Transition rules --------------------
TRANSITION = {"SU": 1991, "CS": 1993, "YU": 2003}
SU_SUCCESSORS = ["RU", "UA", "EE", "LV", "LT", "MD", "GE"]
CS_SUCCESSORS = ["CZ", "SK"]
YU_SUCCESSORS = ["RS", "ME", "HR", "SI", "MK", "BA"]

PREDS = set(TRANSITION.keys())
CLUSTER_SUCCESSORS = {
    "SU": SU_SUCCESSORS,
    "CS": CS_SUCCESSORS,
    "YU": YU_SUCCESSORS,
}

# predecessor -> a representative ISO3 (used to color territories as predecessor)
# (Plotly choropleth uses modern ISO3 boundaries; we color each successor territory
# by mapping its value to the predecessor ISO3 bounds as a proxy. Here we use a
# representative code for legend/colorbar consistency.)
PRED_ISO3 = {"SU": "RUS", "CS": "CZE", "YU": "SRB"}

# -------------------- Country code mapping --------------------
# Map WIPO-like two-letter to ISO3 for Plotly
WIPO_TO_ISO3 = {
    "AF":"AFG","AL":"ALB","DZ":"DZA","AR":"ARG","AM":"ARM","AU":"AUS","AT":"AUT","AZ":"AZE",
    "BH":"BHR","BD":"BGD","BY":"BLR","BE":"BEL","BZ":"BLZ","BJ":"BEN","BO":"BOL","BA":"BIH",
    "BW":"BWA","BR":"BRA","BN":"BRN","BG":"BGR","BF":"BFA","BI":"BDI",
    "KH":"KHM","CM":"CMR","CA":"CAN","CV":"CPV","CF":"CAF","TD":"TCD","CL":"CHL","CN":"CHN",
    "CO":"COL","KM":"COM","CD":"COD","CG":"COG","CR":"CRI","CI":"CIV","HR":"HRV","CU":"CUB",
    "CY":"CYP","CZ":"CZE","DK":"DNK","DJ":"DJI","DO":"DOM","EC":"ECU","EG":"EGY","SV":"SLV",
    "EE":"EST","ET":"ETH","FI":"FIN","FR":"FRA","GA":"GAB","GM":"GMB","GE":"GEO","DE":"DEU",
    "GH":"GHA","GR":"GRC","GT":"GTM","GN":"GIN","GY":"GUY","HT":"HTI","HN":"HND","HK":"HKG",
    "HU":"HUN","IS":"ISL","IN":"IND","ID":"IDN","IR":"IRN","IQ":"IRQ","IE":"IRL","IL":"ISR",
    "IT":"ITA","JM":"JAM","JP":"JPN","JO":"JOR","KZ":"KAZ","KE":"KEN","KR":"KOR","KW":"KWT",
    "KG":"KGZ","LA":"LAO","LV":"LVA","LB":"LBN","LS":"LSO","LR":"LBR","LY":"LBY","LT":"LTU",
    "LU":"LUX","MO":"MAC","MK":"MKD","MG":"MDG","MW":"MWI","MY":"MYS","ML":"MLI","MT":"MLT",
    "MR":"MRT","MU":"MUS","MX":"MEX","MD":"MDA","MN":"MNG","ME":"MNE","MA":"MAR","MZ":"MOZ",
    "MM":"MMR","NA":"NAM","NP":"NPL","NL":"NLD","NZ":"NZL","NI":"NIC","NE":"NER","NG":"NGA",
    "NO":"NOR","OM":"OMN","PK":"PAK","PA":"PAN","PG":"PNG","PY":"PRY","PE":"PER","PH":"PHL",
    "PL":"POL","PT":"PRT","PR":"PRI","QA":"QAT","RO":"ROU","RU":"RUS","RW":"RWA","SA":"SAU",
    "SN":"SEN","RS":"SRB","SG":"SGP","SK":"SVK","SI":"SVN","SO":"SOM","ZA":"ZAF","ES":"ESP",
    "LK":"LKA","SD":"SDN","SR":"SUR","SE":"SWE","CH":"CHE","SY":"SYR","TW":"TWN","TJ":"TJK",
    "TZ":"TZA","TH":"THA","TL":"TLS","TG":"TGO","TT":"TTO","TN":"TUN","TR":"TUR","TM":"TKM",
    "UG":"UGA","UA":"UKR","AE":"ARE","GB":"GBR","UK":"GBR","US":"USA","UY":"URY","UZ":"UZB",
    "VE":"VEN","VN":"VNM","YE":"YEM","ZM":"ZMB","ZW":"ZWE",
    # non-sovereign or org codes (ignored on map)
    "EP": None, "WO": None, "AP": None, "OA": None, "EA": None, "GC": None,
    # historical proxies
    "SU":"RUS","DD":"DEU","CS":"CZE","YU":"SRB",
}

# Micro-country overlay points (include Singapore and Hong Kong as requested)
MICRO_POINTS = [
    {"iso3": "SGP", "name": "Singapore",          "lat": 1.3521, "lon": 103.8198, "size": 8},
    {"iso3": "HKG", "name": "Hong Kong (China)",  "lat": 22.3193, "lon": 114.1694, "size": 8},
]

# ---------------------- Helpers: years ----------------------
def extract_year_token(colname):
    if isinstance(colname, (int, float)) and not isinstance(colname, bool):
        y = int(float(colname))
        return y if 1800 <= y <= 2025 else None
    s = str(colname).strip()
    m = re.findall(r"(\d{4})", s)
    for tok in m:
        y = int(tok)
        if 1800 <= y <= 2025:
            return y
    return None

def detect_year_columns(df):
    cols = list(df.columns)
    if len(cols) <= 2:
        return [], {}
    year_map = {}
    for c in cols[2:]:
        y = extract_year_token(c)
        if y is not None and y <= 2024:
            if y not in year_map:
                year_map[y] = c
    years = sorted(year_map.keys())
    return years, year_map

# ---------------------- IPCR descriptions ----------------------
def load_ipcr_descriptions(xlsx_path):
    desc = {}
    if not os.path.exists(xlsx_path):
        print("IPCR.xlsx not found, skip descriptions:", xlsx_path)
        return desc
    try:
        s = pd.read_excel(xlsx_path, header=None, dtype=str).iloc[:, 0].dropna().astype(str)
        for line in s:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 1:
                code, d = parts[0].strip().upper(), ""
            else:
                code, d = parts[0].strip().upper(), parts[1].strip()
            if code:
                desc[code] = d
    except Exception as e:
        print("Failed to read IPCR.xlsx, skip descriptions:", e)
    return desc

def short_desc(text, max_len=64):
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[:max_len - 3] + "..."

def esc(s):
    return html.escape(s or "")

# -------------------- Series / transition utils --------------------
def parse_series_country(s):
    """Return (succ, pred) for pair like 'RU/SU'; for single 'US' return (s, None)."""
    if not isinstance(s, str):
        return (None, None)
    s = s.strip().upper()
    if "/" in s:
        a, b = s.split("/", 1)
        return (a, b)
    return (s, None)

def is_pair_series(s):
    a, b = parse_series_country(s)
    return (a is not None) and (b in PREDS)

def cluster_key(s):
    a, b = parse_series_country(s)
    return b if b in PREDS else None

def transition_year_for_pred(pred):
    return TRANSITION.get(pred, None)

def successors_for_pred(pred):
    return CLUSTER_SUCCESSORS.get(pred, [])

def choose_primary_series(pred):
    """Deterministic choose a primary series within a predecessor cluster."""
    sucs = successors_for_pred(pred)
    if not sucs:
        return None
    # pick lexicographically smallest 'succ/pred'
    return f"{sorted(sucs)[0]}/{pred}"

def name_left_side(name):
    """For a series name like 'Russia/Soviet Union', return 'Russia'."""
    if not isinstance(name, str):
        return name
    if "/" in name:
        return name.split("/", 1)[0].strip()
    return name.strip()

# -------------------- Data preparation --------------------
def prepare_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Input not found: " + INPUT_CSV)

    # Load names from series table (Country, Name)
    series_name_map = {}
    if os.path.exists(SERIES_NAME_CSV):
        try:
            nm = pd.read_csv(SERIES_NAME_CSV, dtype=str, usecols=["Country", "Name"])
            nm = nm.dropna(subset=["Country"]).drop_duplicates("Country")
            for sid, n in zip(nm["Country"].astype(str), nm["Name"].astype(str)):
                series_name_map[sid.strip().upper()] = n.strip()
        except Exception:
            pass

    # Load IPCR descriptions
    ipcr_desc = load_ipcr_descriptions(IPCR_DESC_XLSX)

    # Read main amount table
    raw = pd.read_csv(INPUT_CSV, dtype={"Country": str, "IPCR": str})
    # Detect year columns
    years, year_col_map = detect_year_columns(raw)
    if not years:
        raise ValueError("No valid year columns <= 2024 found in input.")
    year_cols = [year_col_map[y] for y in years]

    # Normalize
    base = raw[["Country", "IPCR"] + year_cols].copy()
    base["Country"] = base["Country"].astype(str).str.strip().str.upper()
    base["IPCR"] = base["IPCR"].astype(str).str.strip().str.upper()
    base_years = base[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    base[year_cols] = base_years.values

    # Build hierarchy codes
    base["L1"] = base["IPCR"].str.slice(0, 1)
    base["L2"] = base["IPCR"].str.slice(0, 3)
    base["L3"] = base["IPCR"].str.slice(0, 4)

    # -------- Transition-aware de-duplication at series level (pre-transition) --------
    # For each predecessor cluster (SU/CS/YU) and each IPCR, keep only ONE series line
    # before the transition year; other member series are zeroed for years < transition.
    # This prevents multiple counting caused by upstream expansion.
    years_int = np.array(years, dtype=int)
    for pred, tyear in TRANSITION.items():
        # rows belonging to this cluster
        mask_cluster = base["Country"].str.endswith("/" + pred)
        if not mask_cluster.any():
            continue
        # choose primary series tag for this cluster
        primary_tag = choose_primary_series(pred)  # e.g., 'EE/SU' (lexicographically first)
        if primary_tag is None:
            continue
        pre_year_cols = [year_col_map[y] for y in years if y < tyear]
        if not pre_year_cols:
            continue

        # group by IPCR and within each, zero out non-primary pre-transition
        idx_cluster = base.index[mask_cluster]
        if len(idx_cluster) == 0:
            continue
        g = base.loc[idx_cluster].groupby("IPCR", sort=False).indices
        for ipcr, idxs in g.items():
            idxs = list(idxs)
            # identify primary row among idxs if present; otherwise pick lexicographically min
            candidates = base.loc[idxs, "Country"].tolist()
            if primary_tag in candidates:
                pri_idx = idxs[candidates.index(primary_tag)]
            else:
                # stable deterministic choice
                pairs = sorted(zip(candidates, idxs), key=lambda x: str(x[0]))
                pri_idx = pairs[0][1]
            non_primary = [i for i in idxs if i != pri_idx]
            if non_primary:
                base.loc[non_primary, pre_year_cols] = 0.0  # zero out before transition

    # -------- Aggregate per view (TOTAL/L1/L2/L3) at SERIES level --------
    # Build series-year totals for each view, then cumulative across years.
    def agg_series(view_col, view_key=None):
        if view_col == "TOTAL":
            df = base.groupby("Country", as_index=False)[year_cols].sum()
        else:
            df = base[base[view_col] == view_key].groupby("Country", as_index=False)[year_cols].sum()
        # cumulative
        df[year_cols] = df[year_cols].cumsum(axis=1)
        return df

    # Sidebar keys
    l1_keys = sorted([c for c in base["L1"].unique().tolist() if c in list("ABCDEFGH")])
    l2_all = sorted(base["L2"].unique().tolist())
    l3_all = sorted(base["L3"].unique().tolist())
    l2_by_l1 = {l1: [c for c in l2_all if c.startswith(l1)] for l1 in l1_keys}
    l3_by_l2 = {l2: [c for c in l3_all if c.startswith(l2)] for l2 in l2_all}

    # -------- Build per-view ISO3 series with transition-aware mapping --------
    # Rule:
    # - single country 'US' -> ISO3('US') all years
    # - pair 'UA/SU':
    #     year < 1991 -> mapped to predecessor ISO3 (RUS proxy), but we will display
    #                    this predecessor value across all successor territories (visual)
    #     year >= 1991 -> mapped to successor ISO3('UA')
    #
    # Implementation strategy:
    #   For each view, build a dict: iso3 -> list(float) of length len(years)
    #   When a pair series is pre-transition, add its value to the predecessor ISO3 bucket.
    #   Post-transition, add to successor ISO3 bucket.
    #
    #   For the "visual broadcast" of predecessor color across all successors before transition,
    #   we will duplicate the predecessor value to successors' micro points layer only (or
    #   simply color the map by predecessor area; Plotly uses ISO3 shapes, so we color once
    #   via predecessor ISO3; visually successors remain uncolored polygons until transition,
    #   which matches "belong to predecessor" semantic).
    #   To better match "territories belong to predecessor", we will ALSO set the predecessor
    #   value into each successor ISO3 before transition for display (not for totals),
    #   but keep the accumulation "once" by maintaining a separate display dict.
    #
    # We thus prepare:
    #   - iso3_values: the actual accumulation per ISO3 (no duplication)
    #   - iso3_display_values: a derived frame for display where, pre-transition, each successor
    #     ISO3 receives the predecessor's value (so the territory is colored).
    #
    # Name map for hover: iso3 -> human name (from series_name_map as best effort)
    def build_iso_frames(series_df):
        # series_df: columns Country + year_cols (cumulative)
        # 1) actual per-ISO3 (no duplication)
        iso_vals = {}
        for idx, row in series_df.iterrows():
            sid = str(row["Country"])
            succ, pred = parse_series_country(sid)
            vals = row[year_cols].to_numpy(dtype=float)

            if pred in PREDS:
                tyear = transition_year_for_pred(pred)
                pred_iso = PRED_ISO3.get(pred)  # proxy
                succ_iso = WIPO_TO_ISO3.get(succ)
                if pred_iso is None:
                    continue

                # split contributions: pre-transition -> pred_iso; post -> succ_iso
                for j, y in enumerate(years):
                    target_iso = pred_iso if y < tyear else succ_iso
                    if target_iso is None:
                        continue
                    iso_vals.setdefault(target_iso, np.zeros(len(years), dtype=float))
                    iso_vals[target_iso][j] += float(vals[j])
            else:
                # single
                iso = WIPO_TO_ISO3.get(succ)
                if iso is None:
                    continue
                iso_vals.setdefault(iso, np.zeros(len(years), dtype=float))
                iso_vals[iso] += row[year_cols].to_numpy(dtype=float)

        # 2) build a display dict: copy iso_vals, but for each predecessor cluster and pre-transition
        #    years, broadcast predecessor value to each successor ISO3 so their territories are colored.
        iso_display = {}
        # start with a copy
        for iso, arr in iso_vals.items():
            iso_display[iso] = np.array(arr, dtype=float)

        # For each predecessor cluster, compute predecessor frame and replicate to successors pre-transition
        for pred, tyear in TRANSITION.items():
            pred_iso = PRED_ISO3.get(pred)
            if pred_iso is None:
                continue
            pred_arr = iso_vals.get(pred_iso, None)
            if pred_arr is None:
                continue
            for succ in successors_for_pred(pred):
                succ_iso = WIPO_TO_ISO3.get(succ)
                if succ_iso is None:
                    continue
                iso_display.setdefault(succ_iso, np.zeros(len(years), dtype=float))
                for j, y in enumerate(years):
                    if y < tyear:
                        # display same predecessor value (visual territory belongs to pred)
                        iso_display[succ_iso][j] = float(pred_arr[j])
                    # y >= tyear: keep actual (already filled through iso_vals accumulation)
                    # (do nothing)

        # to python dict of lists
        iso_vals_dict = {iso: arr.tolist() for iso, arr in iso_vals.items()}
        iso_display_dict = {iso: arr.tolist() for iso, arr in iso_display.items()}
        return iso_vals_dict, iso_display_dict

    # precompute per-view iso frames
    # TOTAL
    total_series = agg_series("TOTAL")
    total_iso, total_iso_display = build_iso_frames(total_series)

    # L1/L2/L3 dicts
    l1_iso = {}
    l1_iso_display = {}
    for l1 in l1_keys:
        df = agg_series("L1", l1)
        l1_iso[l1], l1_iso_display[l1] = build_iso_frames(df)

    l2_iso = {}
    l2_iso_display = {}
    for code in l2_all:
        df = agg_series("L2", code)
        l2_iso[code], l2_iso_display[code] = build_iso_frames(df)

    l3_iso = {}
    l3_iso_display = {}
    for code in l3_all:
        df = agg_series("L3", code)
        l3_iso[code], l3_iso_display[code] = build_iso_frames(df)

    # -------- Hover name map (iso3 -> display name) --------
    # Build from series_name_map best effort:
    # - for singles like 'US', use its Name
    # - for pairs like 'RU/SU', take left side (e.g., 'Russia/Soviet Union' -> 'Russia')
    iso_name = {}
    for sid, nm in series_name_map.items():
        succ, pred = parse_series_country(sid)
        if succ is None:
            continue
        base_name = name_left_side(nm)
        iso = WIPO_TO_ISO3.get(succ)
        if iso:
            iso_name.setdefault(iso, base_name)

    # overrides
    iso_name["CHN"] = "China(mainland)"
    iso_name["TWN"] = "Taiwan (China)"
    iso_name["HKG"] = "Hong Kong (China)"

    return {
        "years": [int(y) for y in years],
        "total_iso_values": total_iso,           # actual
        "total_iso_display": total_iso_display,  # with pre-transition broadcast
        "l1_iso_values": l1_iso,
        "l1_iso_display": l1_iso_display,
        "l2_iso_values": l2_iso,
        "l2_iso_display": l2_iso_display,
        "l3_iso_values": l3_iso,
        "l3_iso_display": l3_iso_display,
        "l1_keys": l1_keys,
        "l2_by_l1": l2_by_l1,
        "l3_by_l2": l3_by_l2,
        "name_map": iso_name,
        "micro": MICRO_POINTS,
        "ipcr_desc": ipcr_desc,
    }

# ---------------------- HTML writer ----------------------
def write_html(ctx):
    years = ctx["years"]
    total_iso_values = ctx["total_iso_values"]
    total_iso_display = ctx["total_iso_display"]
    l1_iso_values = ctx["l1_iso_values"]
    l1_iso_display = ctx["l1_iso_display"]
    l2_iso_values = ctx["l2_iso_values"]
    l2_iso_display = ctx["l2_iso_display"]
    l3_iso_values = ctx["l3_iso_values"]
    l3_iso_display = ctx["l3_iso_display"]
    l1_keys = ctx["l1_keys"]
    l2_by_l1 = ctx["l2_by_l1"]
    l3_by_l2 = ctx["l3_by_l2"]
    name_map = ctx["name_map"]
    micro_points = ctx["micro"]
    ipcr_desc = ctx["ipcr_desc"]

    # ---- payload for JS (data only) ----
    payload = {
        "years": years,
        "nameMap": name_map,
        "data": {
            "TOTAL": total_iso_values,
            "TOTAL_DISPLAY": total_iso_display,
            "L1": l1_iso_values,
            "L1_DISPLAY": l1_iso_display,
            "L2": l2_iso_values,
            "L2_DISPLAY": l2_iso_display,
            "L3": l3_iso_values,
            "L3_DISPLAY": l3_iso_display,
        },
        "micro": micro_points,
    }
    payload_json = json.dumps(payload)

    # ---- helpers for labeling with descriptions (HTML) ----
    def fmt_label(code):
        full = ipcr_desc.get(code, "")
        short = short_desc(full, 72)
        if full:
            return esc(code) + " - " + esc(short), esc(full)
        else:
            return esc(code), ""

    def fmt_label_l1(code):
        full = ipcr_desc.get(code, "")
        short = short_desc(full, 72)
        if full:
            return esc(code) + " (" + esc(short) + ")", esc(full)
        else:
            return esc(code), ""

    # ---- build nested sidebar HTML ----
    groups = []

    # Total item
    groups.append(
        '<div class="menu-group">'
        '  <div class="menu-item active" data-type="TOTAL" data-key="TOTAL" title="Total patents">Total patents</div>'
        '</div>'
    )

    # L1 -> collapsible -> L2 (3-char) -> collapsible -> L3 (4-char)
    for l1 in l1_keys:
        l1_txt, l1_full = fmt_label_l1(l1)
        l2_list = l2_by_l1.get(l1, [])
        l2_blocks = []
        for l2 in l2_list:
            l2_txt, l2_full = fmt_label(l2)
            l3_list = l3_by_l2.get(l2, [])
            l3_items = ''.join(
                f'<div class="menu-subitem" data-type="L3" data-key="{esc(code)}" title="{esc(ipcr_desc.get(code, ""))}">{fmt_label(code)[0]}</div>'
                for code in l3_list
            ) or '<div class="menu-subitem disabled">No level-3 codes</div>'
            l2_block = (
                '<details class="sub">'
                f'  <summary title="{l2_full}"><span class="menu-item-l2" data-type="L2" data-key="{esc(l2)}">{l2_txt}</span></summary>'
                f'  <div class="menu-sublist-2">{l3_items}</div>'
                '</details>'
            )
            l2_blocks.append(l2_block)
        l2_block_html = ''.join(l2_blocks) or '<div class="menu-subitem disabled">No level-2 codes</div>'
        section_html = (
            '<div class="menu-group">'
            f'  <details>'
            f'    <summary title="{l1_full}"><span class="menu-item-l1" data-type="L1" data-key="{esc(l1)}">{l1_txt}</span></summary>'
            f'    <div class="menu-sublist">{l2_block_html}</div>'
            f'  </details>'
            '</div>'
        )
        groups.append(section_html)

    sidebar_html = '\n'.join(groups)

    # ---- HTML template via string.Template (avoid brace escaping issues) ----
    tmpl = Template(r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>World Patent Choropleth with 3-Level Sidebar & Timeline</title>
<script src="https://cdn.plot.ly/plotly-2.33.0.min.js"></script>
<style>
  html, body { height: 100%; margin: 0; font-family: Arial, Helvetica, sans-serif; }
  .app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; width: 100vw; }
  #sidebar {
    border-right: 1px solid #ddd; padding: 12px; overflow: auto; resize: horizontal;
    min-width: 220px; max-width: 45vw; background: #fafafa;
  }
  #sidebar h2 { margin: 0 0 8px 0; font-size: 16px; }
  .menu-group { margin-bottom: 8px; }
  .menu-item, .menu-item-l1, .menu-item-l2, .menu-subitem {
    display: block; padding: 6px 8px; margin: 4px 0; border-radius: 6px; cursor: pointer; user-select: none;
  }
  .menu-item:hover, .menu-item-l1:hover, .menu-item-l2:hover, .menu-subitem:hover { background: #eee; }
  .menu-item.active, .menu-item-l1.active, .menu-item-l2.active, .menu-subitem.active { background: #2b6cb0; color: white; }
  .menu-sublist { padding-left: 6px; }
  .menu-sublist-2 { padding-left: 16px; }
  details > summary {
    list-style: none; cursor: pointer; padding: 6px 8px; border-radius: 6px; background: #f0f0f0; user-select: none;
  }
  details > summary::before { content: ">"; display: inline-block; margin-right: 6px; }
  details[open] > summary::before { content: "v"; }
  .menu-item-l1, .menu-item-l2 { display: inline; padding: 0; background: transparent; color: #2b6cb0; }
  .menu-item-l1:hover, .menu-item-l2:hover { text-decoration: underline; }
  .menu-subitem.disabled { color: #999; cursor: default; }
  #main { padding: 8px 14px; display: flex; flex-direction: column; height: 100%; }
  #chart { flex: 1 1 auto; min-height: 560px; }
  .tip { color: #666; font-size: 12px; }
</style>
</head>
<body>
<div class="app">
  <div id="sidebar">
    <h2>Catalog</h2>
    $SIDEBAR_HTML
    <div class="tip">
    Tip: Hover to see full descriptions.<br>
    Data sources: European Patent Office (EPO) and United States Patent and Trademark Office (USPTO).<br>
    Figure author: Guoyang Rong. All rights reserved.
    </div>
  </div>
  <div id="main"><div id="chart"></div></div>
</div>

<script>
const PAYLOAD = $PAYLOAD_JSON;

let YEARS = PAYLOAD.years;
let NAME_MAP = PAYLOAD.nameMap;
let DATA_TOTAL = PAYLOAD.data.TOTAL;               // actual accumulation
let DATA_TOTAL_DISPLAY = PAYLOAD.data.TOTAL_DISPLAY; // with pre-transition broadcast
let DATA_L1 = PAYLOAD.data.L1;
let DATA_L1_DISPLAY = PAYLOAD.data.L1_DISPLAY;
let DATA_L2 = PAYLOAD.data.L2;
let DATA_L2_DISPLAY = PAYLOAD.data.L2_DISPLAY;
let DATA_L3 = PAYLOAD.data.L3;
let DATA_L3_DISPLAY = PAYLOAD.data.L3_DISPLAY;
let MICRO = PAYLOAD.micro || [];

let state = {
  datasetType: "TOTAL",  // "TOTAL" | "L1" | "L2" | "L3"
  datasetKey: "TOTAL",
  startYearIndex: YEARS.length - 1
};

function getIsoValues(dsType, dsKey, forDisplay=true) {
  if (dsType === "TOTAL") return forDisplay ? DATA_TOTAL_DISPLAY : DATA_TOTAL;
  if (dsType === "L1")    return (forDisplay ? (DATA_L1_DISPLAY[dsKey] || {}) : (DATA_L1[dsKey] || {}));
  if (dsType === "L2")    return (forDisplay ? (DATA_L2_DISPLAY[dsKey] || {}) : (DATA_L2[dsKey] || {}));
  if (dsType === "L3")    return (forDisplay ? (DATA_L3_DISPLAY[dsKey] || {}) : (DATA_L3[dsKey] || {}));
  return {};
}

function computeFrame(isoValues, yearIdx) {
  const locs = [], vals = [], texts = [];
  for (const iso in isoValues) {
    const series = isoValues[iso];
    const v = series[yearIdx] || 0;
    if (v > 0) {
      locs.push(iso);
      vals.push(v);
      const label = NAME_MAP[iso] || iso;
      const y = YEARS[yearIdx];
      texts.push(label + "<br>Year: " + y + "<br>Cumulative: " + Math.trunc(v).toLocaleString());
    }
  }
  let zmin = 0, zmax = 1;
  if (vals.length > 0) {
    zmin = Math.min(...vals);
    zmax = Math.max(...vals);
    if (zmin === zmax) { zmin = Math.max(0, zmin - 1); zmax = zmax + 1; }
  }
  const mlat = [], mlon = [], mtext = [], mval = [];
  for (const pt of MICRO) {
    const iso3 = pt.iso3;
    if (iso3 in isoValues) {
      const v = isoValues[iso3][yearIdx] || 0;
      if (v > 0) {
        mlat.push(pt.lat); mlon.push(pt.lon);
        const y = YEARS[yearIdx];
        mtext.push(pt.name + "<br>Year: " + y + "<br>Cumulative: " + Math.trunc(v).toLocaleString());
        mval.push(v);
      }
    }
  }
  return { locs, vals, texts, zmin, zmax, mlat, mlon, mtext, mval };
}

function buildFigureWithFrames(dsType, dsKey, initIdx) {
  const isoValues = getIsoValues(dsType, dsKey, true);
  const C0 = computeFrame(isoValues, initIdx);

  const choro0 = {
    type: "choropleth",
    locations: C0.locs, z: C0.vals, text: C0.texts,
    hovertemplate: "%{text}<extra></extra>",
    colorscale: "OrRd", zmin: C0.zmin, zmax: C0.zmax,
    marker: { line: { color: "gray", width: 0.3 } },
    colorbar: { title: "Cumulative patents" },
    showscale: true, locationmode: "ISO-3"
  };

  const microSize = (MICRO.length ? (MICRO[0].size || 8) : 8);
  const micro0 = {
    type: "scattergeo", mode: "markers",
    lat: C0.mlat, lon: C0.mlon, text: C0.mtext,
    hovertemplate: "%{text}<extra></extra>",
    marker: {
      size: (C0.mval.length > 0 ? microSize : 0),
      color: C0.mval, colorscale: "OrRd", cmin: C0.zmin, cmax: C0.zmax,
      line: { width: 0.5, color: "black" }, sizemode: "diameter"
    },
    showlegend: false
  };

  const frames = YEARS.map((y, idx) => {
    const C = computeFrame(isoValues, idx);
    return {
      name: String(y),
      data: [
        { type: "choropleth", locations: C.locs, z: C.vals, text: C.texts,
          hovertemplate: "%{text}<extra></extra>", colorscale: "OrRd",
          zmin: C.zmin, zmax: C.zmax, marker: { line: { color: "gray", width: 0.3 } },
          colorbar: { title: "Cumulative patents" }, showscale: true, locationmode: "ISO-3" },
        { type: "scattergeo", mode: "markers", lat: C.mlat, lon: C.mlon, text: C.mtext,
          hovertemplate: "%{text}<extra></extra>",
          marker: { size: (C.mval.length > 0 ? microSize : 0), color: C.mval,
                    colorscale: "OrRd", cmin: C.zmin, cmax: C.zmax,
                    line: { width: 0.5, color: "black" }, sizemode: "diameter" },
          showlegend: false }
      ]
    };
  });

  const steps = YEARS.map((y) => ({
    method: "animate", label: String(y),
    args: [[String(y)], {mode: "immediate", frame: {duration: 0, redraw: true}, transition: {duration: 0}}]
  }));

  const titleKey = (dsType === "TOTAL") ? "Total patents" : dsKey;

  const layout = {
    title: "World Patent Counts by Country/Region (Cumulative) - " + titleKey,
    paper_bgcolor: "white", plot_bgcolor: "white",
    margin: { l: 10, r: 10, t: 60, b: 10 },
    geo: { showframe: false, showcoastlines: true, coastlinecolor: "lightgray",
           showland: true, landcolor: "white", bgcolor: "white",
           projection: { type: "natural earth" } },
    sliders: [{
      active: initIdx, pad: {t: 35, b: 0},
      currentvalue: {visible: true, prefix: "Year: ", xanchor: "right"},
      steps: steps, x: 0.05, y: 0.02, len: 0.9
    }],
    updatemenus: [{
      type: "buttons", showactive: false, x: 0.05, y: 0.0, xanchor: "left", yanchor: "bottom", direction: "left",
      buttons: [
        { label: "Play", method: "animate", args: [null, {frame: {duration: 300, redraw: true}, transition: {duration: 0}, fromcurrent: true}] },
        { label: "Pause", method: "animate", args: [[null], {mode: "immediate", frame: {duration: 0, redraw: true}, transition: {duration: 0}}] }
      ]
    }]
  };

  return { data: [choro0, micro0], layout: layout, frames: frames };
}

function render(dsType, dsKey) {
  const initIdx = state.startYearIndex;
  const fig = buildFigureWithFrames(dsType, dsKey, initIdx);
  Plotly.newPlot("chart", fig.data, fig.layout, {responsive: true, displaylogo: false})
    .then(function() { if (fig.frames && fig.frames.length) { Plotly.addFrames("chart", fig.frames); } });
}

function setActiveMenu(el) {
  document.querySelectorAll("#sidebar .menu-item, #sidebar .menu-item-l1, #sidebar .menu-item-l2, #sidebar .menu-subitem")
    .forEach(function(x) { x.classList.remove("active"); });
  el.classList.add("active");
}

document.addEventListener("DOMContentLoaded", function() {
  render(state.datasetType, state.datasetKey);

  document.getElementById("sidebar").addEventListener("click", function(e) {
    const t = e.target;
    if (t.classList.contains("menu-item")) {
      state.datasetType = t.getAttribute("data-type");
      state.datasetKey = t.getAttribute("data-key");
      setActiveMenu(t); render(state.datasetType, state.datasetKey);
    }
    if (t.classList.contains("menu-item-l1")) {
      state.datasetType = "L1";
      state.datasetKey = t.getAttribute("data-key");
      setActiveMenu(t); render(state.datasetType, state.datasetKey);
    }
    if (t.classList.contains("menu-item-l2")) {
      state.datasetType = "L2";
      state.datasetKey = t.getAttribute("data-key");
      setActiveMenu(t); render(state.datasetType, state.datasetKey);
    }
    if (t.classList.contains("menu-subitem") && !t.classList.contains("disabled")) {
      state.datasetType = t.getAttribute("data-type");  // "L3"
      state.datasetKey = t.getAttribute("data-key");
      setActiveMenu(t); render(state.datasetType, state.datasetKey);
    }
  });
});
</script>
</body>
</html>
""")

    html_doc = tmpl.substitute(
        SIDEBAR_HTML=sidebar_html,
        PAYLOAD_JSON=payload_json,
    )

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("Saved HTML:", OUT_HTML)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    ctx = prepare_data()
    write_html(ctx)
