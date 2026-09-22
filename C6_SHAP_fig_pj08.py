#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 6 — Combined RF + SHAP figures (8 models, 4 groups × CN/EN)
===================================================================

- 使用 /chapter6/8_model/ 下的 RF + TreeSHAP 结果，
  为以下 4 组模型各画一张“combined RF+SHAP”图：
    1) 专利 ROSG: patent_ROSG_1970_2010, patent_ROSG_2011_2024
    2) 论文 ROSG: paper_ROSG_1970_2010, paper_ROSG_2011_2024
    3) 专利 RISG: patent_RISG_1970_2010, patent_RISG_2011_2024
    4) 论文 RISG: paper_RISG_1970_2010, paper_RISG_2011_2024

- 每组图：
    行 1: 1970–2010
    行 2: 2011–2024
    列 1: SHAP mean(|value|) barh（蓝色，x 轴从右到左）
    列 2: shared label 轴（仅刻度线，风格沿用 pj08）
    列 3: SHAP beeswarm（官方风格）

- 每组输出：
    - 英文版：*_en.pdf / *_en.png
    - 中文版：*_zh.pdf / *_zh.png

- 特征标签：
    - 中文版：直接使用 {model_key}_shap_mean_abs.csv 中的 feature 列
    - 英文版：根据 WDI_indicator_subgroup_table.csv 中的 “Series Code” → “指标” 映射

- 数据文件（model_dir = /chapter6/8_model/{model_key}/）：
    - {model_key}_shap_mean_abs.csv
        columns: rank, feature, series_code, mean_abs_shap
    - {model_key}_shap_values.csv.gz
        columns: meta..., shap_{SeriesCode}, ...
    - {model_key}_design_final.csv
        columns: meta..., y, X_{SeriesCode}, ...

- 图形风格：
    - 完全沿用 pj08 的 combined RF+SHAP 图格式，只替换为新数据
"""

import os
import re
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ===================== 全局配置 =====================

# 新 8 模型结果根目录
MODEL8_DIR = "/data01/rong_dataset/Result/PhD/chapter6/8_model_subPCA"

# 指标英文名表
WDI_INDICATOR_TABLE = "/data01/rong_dataset/Result/PhD/chapter6/WDI_indicator_subgroup_table.csv"

# 输出图目录
FIG_DIR = os.path.join(MODEL8_DIR, "RF_SHAP_combined_figs")
os.makedirs(FIG_DIR, exist_ok=True)

# 是否只画“专利 ROSG”这一组
PLOT_ONLY_PATENT_ROSG = True

# 每图最多显示的特征数
TOP_K = 20

# 全局字体与分辨率（在下方再补中文字体）
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# 中文字体配置（若找不到字体，则退回默认）
FONT_PATH = "/data01/rong_dataset/Fonts/NotoSansSC-VariableFont_wght.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    matplotlib.rcParams["font.family"] = prop.get_name()
else:
    prop = font_manager.FontProperties()
    matplotlib.rcParams["font.family"] = prop.get_name()

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ===================== 工具函数 =====================

def pjoin(*a):
    return os.path.join(*a)


def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            return None
    else:
        print(f"[WARN] File not found: {path}")
        return None


def wrap_two_lines_labels(labels, width=45):
    """
    Wrap feature labels into at most two lines to control width.
    """
    import textwrap
    wrapped = []
    for s in labels:
        if not isinstance(s, str):
            s = str(s)
        parts = textwrap.wrap(s, width=width)
        if len(parts) <= 1:
            wrapped.append(s)
        elif len(parts) == 2:
            wrapped.append(parts[0] + "\n" + parts[1])
        else:
            second = " ".join(parts[1:])
            second_short = textwrap.shorten(second, width=width, placeholder="...")
            wrapped.append(parts[0] + "\n" + second_short)
    return wrapped


def load_indicator_en_names():
    """
    从 WDI_indicator_subgroup_table.csv 中构建 SeriesCode → 英文名(“指标”) 映射。
    """
    if not os.path.exists(WDI_INDICATOR_TABLE):
        print(f"[WDI EN] WARNING: indicator table not found: {WDI_INDICATOR_TABLE}")
        return {}

    df = pd.read_csv(WDI_INDICATOR_TABLE, dtype=str)
    if df.empty:
        print(f"[WDI EN] WARNING: indicator table is empty.")
        return {}

    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Series Code 列
    sc_col = None
    for cand in ["series code", "series_code", "Series Code", "Series_Code"]:
        cl = cand.lower()
        if cl in cols_lower:
            sc_col = cols_lower[cl]
            break
    if sc_col is None:
        # 尝试包含 "series" 和 "code" 的列
        for c in df.columns:
            if "series" in c.lower() and "code" in c.lower():
                sc_col = c
                break
    if sc_col is None:
        print("[WDI EN] WARNING: Series Code column not found.")
        return {}

    # “指标”列（英文指标名）
    ind_col = None
    for cand in ["指标", "indicator", "Indicator", "name_en", "Name_EN"]:
        if cand in df.columns:
            ind_col = cand
            break
        cl = cand.lower()
        if cl in cols_lower:
            ind_col = cols_lower[cl]
            break
    if ind_col is None:
        print("[WDI EN] WARNING: indicator name column not found, use Series Code as fallback.")
        mapping = {}
        for _, row in df.iterrows():
            sc = str(row[sc_col]).strip()
            if sc:
                mapping[sc] = sc
        return mapping

    df_sc = df[[sc_col, ind_col]].copy()
    df_sc[sc_col] = df_sc[sc_col].astype(str).str.strip()
    df_sc[ind_col] = df_sc[ind_col].astype(str).str.strip()

    mapping = {}
    for _, row in df_sc.iterrows():
        sc = row[sc_col]
        nm = row[ind_col]
        if sc and nm:
            mapping[sc] = nm
    print(f"[WDI EN] Loaded {len(mapping)} indicator English names from {WDI_INDICATOR_TABLE}")
    return mapping

def load_prior_group_mapping():
    """
    从 WDI_indicator_subgroup_table.csv 读取 SeriesCode → 先验组代码 的映射。
    若文件或列缺失，返回空字典。
    """
    if not os.path.exists(WDI_INDICATOR_TABLE):
        print(f"[Prior Group] WARNING: table not found: {WDI_INDICATOR_TABLE}")
        return {}

    df = pd.read_csv(WDI_INDICATOR_TABLE, dtype=str)
    if df.empty:
        print("[Prior Group] WARNING: table is empty.")
        return {}

    cols_lower = {c.lower().strip(): c for c in df.columns}

    # 定位 Series Code 列
    sc_col = None
    for cand in ["series code", "series_code", "Series Code", "Series_Code"]:
        if cand in df.columns:
            sc_col = cand
            break
        cl = cand.lower()
        if cl in cols_lower:
            sc_col = cols_lower[cl]
            break
    if sc_col is None:
        for c in df.columns:
            if "series" in c.lower() and "code" in c.lower():
                sc_col = c
                break
    if sc_col is None:
        print("[Prior Group] WARNING: Series Code column not found.")
        return {}

    # 定位“先验组代码”列（支持多种可能列名）
    pg_col = None
    for cand in ["先验组代码", "prior_group_code", "prior group code"]:
        if cand in df.columns:
            pg_col = cand
            break
        cl = cand.lower()
        if cl in cols_lower:
            pg_col = cols_lower[cl]
            break
    if pg_col is None:
        print("[Prior Group] WARNING: '先验组代码' column not found.")
        return {}

    mapping = {}
    for _, row in df.iterrows():
        sc = str(row[sc_col]).strip()
        pg = str(row[pg_col]).strip()
        if sc and pg:
            mapping[sc] = pg
    print(f"[Prior Group] Loaded {len(mapping)} mappings.")
    return mapping

def load_mean_abs_for_model(model_key):
    """
    读取某个 model_key 的 SHAP mean(|value|) 结果：
        {MODEL8_DIR}/{model_key}/{model_key}_shap_mean_abs.csv
    保留 TOP_K。
    返回 DataFrame: [feature, series_code, mean_abs_shap]
    """
    path = pjoin(MODEL8_DIR, model_key, f"{model_key}_shap_mean_abs.csv")
    df = safe_read_csv(path)
    if df is None or df.empty:
        print(f"[ERROR] No mean_abs_shap data for model {model_key}")
        return pd.DataFrame(columns=["feature", "series_code", "mean_abs_shap"])

    d = df.copy()
    if "mean_abs_shap" not in d.columns:
        raise ValueError(f"{path} missing column 'mean_abs_shap'. Got: {list(d.columns)}")

    # 统一列名
    if "series_code" not in d.columns:
        # 可能是 "Series_Code" 等
        for c in d.columns:
            if "series" in c.lower() and "code" in c.lower():
                d.rename(columns={c: "series_code"}, inplace=True)
                break
        if "series_code" not in d.columns:
            d["series_code"] = ""

    d["mean_abs_shap"] = pd.to_numeric(d["mean_abs_shap"], errors="coerce")
    d = d.sort_values("mean_abs_shap", ascending=False)
    d = d.head(TOP_K).reset_index(drop=True)

    return d[["feature", "series_code", "mean_abs_shap"]].copy()


def load_shap_and_X_for_model(model_key, mean_df):
    shap_path = pjoin(MODEL8_DIR, model_key, f"{model_key}_shap_values.csv.gz")
    design_path = pjoin(MODEL8_DIR, model_key, f"{model_key}_design_final.csv")

    shap_df = safe_read_csv(shap_path, compression="gzip")
    design_df = safe_read_csv(design_path)

    if shap_df is None or shap_df.empty or design_df is None or design_df.empty:
        print(f"[WARN] Missing SHAP or design data for model {model_key}.")
        return None, None

    # ======= 新增：对齐 shap_df 和 design_df 的行数 =======
    n_shap = len(shap_df)
    n_design = len(design_df)
    if n_shap != n_design:
        print(f"[WARN] {model_key}: shap 行数 = {n_shap}, design 行数 = {n_design}，进行截取对齐。")
        n = min(n_shap, n_design)
        shap_df = shap_df.iloc[:n].reset_index(drop=True)
        design_df = design_df.iloc[:n].reset_index(drop=True)
    else:
        n = n_shap
    # ===============================================

    series_codes = mean_df["series_code"].astype(str).tolist()

    # SHAP 值列：shap_{SeriesCode}
    shap_cols = []
    for sc in series_codes:
        col = f"shap_{sc}"
        if col not in shap_df.columns:
            print(f"[WARN] Column {col} not found in {shap_path}. Filling zeros.")
        shap_cols.append(col)

    K = len(shap_cols)
    shap_matrix = np.zeros((n, K), dtype=float)
    for j, col in enumerate(shap_cols):
        if col in shap_df.columns:
            shap_matrix[:, j] = (
                pd.to_numeric(shap_df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy()
            )
        else:
            shap_matrix[:, j] = 0.0

    # 设计矩阵：X_{SeriesCode}
    X_matrix = np.zeros((n, K), dtype=float)
    for j, sc in enumerate(series_codes):
        col = f"X_{sc}"
        if col in design_df.columns:
            X_matrix[:, j] = (
                pd.to_numeric(design_df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy()
            )
        else:
            print(f"[WARN] Column {col} not found in {design_path}. Filling zeros.")
            X_matrix[:, j] = 0.0

    return shap_matrix, X_matrix


def make_common_mask(df1, df2):
    """
    根据 series_code 交集判断两个时期中哪些特征是“共同出现”的。
    返回 mask1, mask2（与 df1 / df2 行对齐的 bool 列表）。
    """
    sc1 = df1["series_code"].astype(str).tolist()
    sc2 = df2["series_code"].astype(str).tolist()
    set1 = set(sc1)
    set2 = set(sc2)
    common = set1 & set2

    mask1 = [s in common for s in sc1]
    mask2 = [s in common for s in sc2]
    return mask1, mask2


# ===================== 绘图主函数（单组模型 × 单语言） =====================

def plot_combined_for_model_pair(
    model_key_1970,
    model_key_2011,
    title_prefix,
    lang,
    sc_to_en_name
):
    """
    对一组模型（1970-2010 & 2011-2024）画一张 combined 图（2×3 布局），
    lang ∈ {"zh", "en"} 控制特征标签语言。
    标签颜色根据“先验组代码”区分，共12组，每组一种颜色。
    每行图下方自动添加该时期的颜色图例，并标注每个先验组在该行中的出现频次。
    """

    if not HAS_SHAP:
        print("[ERROR] shap is not installed. Please install it before plotting.")
        return

    # 加载先验组映射
    pg_mapping = load_prior_group_mapping()

    # 1. 读取两个时期的 mean(|SHAP|)
    df1 = load_mean_abs_for_model(model_key_1970)
    df2 = load_mean_abs_for_model(model_key_2011)

    if df1.empty and df2.empty:
        print(f"[ERROR] No mean_abs_shap data for {model_key_1970} & {model_key_2011}")
        return

    # 2. 计算“共同特征”mask
    common_mask_1, common_mask_2 = make_common_mask(df1, df2)

    # 3. 构造标签（根据语言）和先验组列表
    if lang == "zh":
        labels_1 = df1["feature"].astype(str).tolist()
        labels_2 = df2["feature"].astype(str).tolist()
    else:  # "en"
        labels_1 = [sc_to_en_name.get(sc, sc) for sc in df1["series_code"].astype(str)]
        labels_2 = [sc_to_en_name.get(sc, sc) for sc in df2["series_code"].astype(str)]

    wrapped_1 = wrap_two_lines_labels(labels_1)
    wrapped_2 = wrap_two_lines_labels(labels_2)

    vals_1 = df1["mean_abs_shap"].to_numpy(dtype=float)
    vals_2 = df2["mean_abs_shap"].to_numpy(dtype=float)

    # 获取先验组代码（按原始顺序）
    series_codes_1 = df1["series_code"].astype(str).tolist()
    prior_groups_1 = [pg_mapping.get(sc, '') for sc in series_codes_1]
    series_codes_2 = df2["series_code"].astype(str).tolist()
    prior_groups_2 = [pg_mapping.get(sc, '') for sc in series_codes_2]

    # 4. 读取 SHAP & X 矩阵（按 df1/df2 当前顺序）
    shap_1, X_1 = load_shap_and_X_for_model(model_key_1970, df1)
    shap_2, X_2 = load_shap_and_X_for_model(model_key_2011, df2)

    if shap_1 is None or shap_2 is None:
        print(f"[ERROR] Missing SHAP/X matrices for {model_key_1970} & {model_key_2011}. Abort.")
        return

    # 5. 按 mean_abs_shap 从小到大排序（保持与 pj08 一致）
    order1 = np.argsort(vals_1)  # ascending
    vals_1 = vals_1[order1]
    labels_1 = [labels_1[i] for i in order1]
    wrapped_1 = [wrapped_1[i] for i in order1]
    common_mask_1 = [common_mask_1[i] for i in order1]
    shap_1 = shap_1[:, order1]
    X_1 = X_1[:, order1]
    prior_groups_1 = [prior_groups_1[i] for i in order1]   # 同步排序

    order2 = np.argsort(vals_2)
    vals_2 = vals_2[order2]
    labels_2 = [labels_2[i] for i in order2]
    wrapped_2 = [wrapped_2[i] for i in order2]
    common_mask_2 = [common_mask_2[i] for i in order2]
    shap_2 = shap_2[:, order2]
    X_2 = X_2[:, order2]
    prior_groups_2 = [prior_groups_2[i] for i in order2]   # 同步排序

    # 6. 定义先验组颜色映射（12种颜色，按组名排序分配）
    PRIOR_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78'
    ]
    # 合并两组所有组别（包括空字符串，但后续图例只显示非空）
    all_groups = sorted(set(prior_groups_1 + prior_groups_2))
    color_map = {g: PRIOR_COLORS[i % len(PRIOR_COLORS)] for i, g in enumerate(all_groups)}
    default_color = '#333333'   # 缺失组别默认颜色
    color_1 = [color_map.get(g, default_color) for g in prior_groups_1]
    color_2 = [color_map.get(g, default_color) for g in prior_groups_2]

    # 统计每行的先验组频次（仅非空）
    from collections import Counter
    counter_1 = Counter([g for g in prior_groups_1 if g != ''])
    counter_2 = Counter([g for g in prior_groups_2 if g != ''])

    # 7. 建立图像（增加高度，为两行图例预留空间）
    fig = plt.figure(figsize=(24, 22))  # 高度从20增至22
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.4, 2.0, 3.1],
        height_ratios=[1.0, 1.0],
    )

    # ===================== Row 1: 1970-2010 =====================

    ax_bar_1 = fig.add_subplot(gs[0, 0])
    ax_lab_1 = fig.add_subplot(gs[0, 1])
    ax_bee_1 = fig.add_subplot(gs[0, 2])

    K1 = len(labels_1)
    y1_pos = np.arange(K1)

    ax_bar_1.barh(y1_pos, vals_1, color="tab:blue", alpha=0.85)
    ax_bar_1.invert_xaxis()
    ax_bar_1.set_yticks(y1_pos)
    ax_bar_1.set_yticklabels([])
    ax_bar_1.set_xlabel("mean(|SHAP value|)")
    ax_bar_1.set_title("1970-2010")
    ax_bar_1.tick_params(axis="y", left=False, right=False)

    # Beeswarm
    plt.sca(ax_bee_1)
    shap.summary_plot(
        shap_1,
        X_1,
        feature_names=labels_1,
        show=False,
        max_display=TOP_K,
        color_bar=True,
        plot_type="dot",
        sort=False,
    )
    ax_bee_1.invert_yaxis()
    ax_bee_1.yaxis.tick_left()
    ax_bee_1.yaxis.set_label_position("left")
    ax_bee_1.tick_params(axis="y", right=False)

    ax_bee_1.set_xlabel(
        "SHAP value",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    ax_bee_1.tick_params(axis="x", labelsize=plt.rcParams["axes.labelsize"])

    ytick_pos_1 = ax_bee_1.get_yticks()
    ylim1 = ax_bee_1.get_ylim()

    ax_bee_1.set_yticks(ytick_pos_1)
    tick_texts_1_bee = ax_bee_1.set_yticklabels(wrapped_1)

    # 应用颜色（按先验组）
    for txt, col in zip(tick_texts_1_bee, color_1):
        txt.set_color(col)
    # 共同特征加粗
    for txt, is_common in zip(tick_texts_1_bee, common_mask_1):
        if is_common:
            txt.set_weight("bold")
        else:
            txt.set_weight("normal")

    for txt in tick_texts_1_bee:
        txt.set_fontsize(plt.rcParams["ytick.labelsize"])

    ax_bee_1.set_ylim(ylim1)
    ax_bee_1.set_title("1970-2010")

    # 中间 label 轴
    ax_lab_1.set_xlim(0.0, 1.0)
    ax_lab_1.set_ylim(ylim1)
    ax_lab_1.set_xticks([])
    ax_lab_1.set_xticklabels([])
    ax_lab_1.set_yticks(ytick_pos_1)
    ax_lab_1.set_yticklabels([])
    ax_lab_1.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True,
    )
    ax_lab_1.spines["top"].set_visible(False)
    ax_lab_1.spines["bottom"].set_visible(False)

    ax_bar_1.set_ylim(ylim1)
    ax_bar_1.set_yticks(ytick_pos_1)
    ax_bar_1.set_yticklabels([])

    # ===================== Row 2: 2011-2024 =====================

    ax_bar_2 = fig.add_subplot(gs[1, 0])
    ax_lab_2 = fig.add_subplot(gs[1, 1])
    ax_bee_2 = fig.add_subplot(gs[1, 2])

    K2 = len(labels_2)
    y2_pos = np.arange(K2)

    ax_bar_2.barh(y2_pos, vals_2, color="tab:blue", alpha=0.85)
    ax_bar_2.invert_xaxis()
    ax_bar_2.set_yticks(y2_pos)
    ax_bar_2.set_yticklabels([])
    ax_bar_2.set_xlabel("mean(|SHAP value|)")
    ax_bar_2.set_title("2011-2024")
    ax_bar_2.tick_params(axis="y", left=False, right=False)

    plt.sca(ax_bee_2)
    shap.summary_plot(
        shap_2,
        X_2,
        feature_names=labels_2,
        show=False,
        max_display=TOP_K,
        color_bar=True,
        plot_type="dot",
        sort=False,
    )
    ax_bee_1.invert_yaxis()
    ax_bee_2.yaxis.tick_left()
    ax_bee_2.yaxis.set_label_position("left")
    ax_bee_2.tick_params(axis="y", right=False)

    ax_bee_2.set_xlabel(
        "SHAP value",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    ax_bee_2.tick_params(axis="x", labelsize=plt.rcParams["axes.labelsize"])

    ytick_pos_2 = ax_bee_2.get_yticks()
    ylim2 = ax_bee_2.get_ylim()

    ax_bee_2.set_yticks(ytick_pos_2)
    tick_texts_2_bee = ax_bee_2.set_yticklabels(wrapped_2)

    for txt, col in zip(tick_texts_2_bee, color_2):
        txt.set_color(col)
    for txt, is_common in zip(tick_texts_2_bee, common_mask_2):
        if is_common:
            txt.set_weight("bold")
        else:
            txt.set_weight("normal")

    for txt in tick_texts_2_bee:
        txt.set_fontsize(plt.rcParams["ytick.labelsize"])

    ax_bee_2.set_ylim(ylim2)
    ax_bee_2.set_title("2011-2024")

    ax_lab_2.set_xlim(0.0, 1.0)
    ax_lab_2.set_ylim(ylim2)
    ax_lab_2.set_xticks([])
    ax_lab_2.set_xticklabels([])
    ax_lab_2.set_yticks(ytick_pos_2)
    ax_lab_2.set_yticklabels([])
    ax_lab_2.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True,
    )
    ax_lab_2.spines["top"].set_visible(False)
    ax_lab_2.spines["bottom"].set_visible(False)

    ax_bar_2.set_ylim(ylim2)
    ax_bar_2.set_yticks(ytick_pos_2)
    ax_bar_2.set_yticklabels([])

    # 调整 “Feature value” 颜色条的字号（如果存在）
    import matplotlib as mpl
    for ax in fig.axes:
        if isinstance(ax, mpl.axes.Axes) and ax.get_ylabel() == "Feature value":
            ax.tick_params(labelsize=plt.rcParams["axes.labelsize"])
            ax.set_ylabel(ax.get_ylabel(), fontsize=plt.rcParams["axes.labelsize"])

    # ===================== 调整布局，为两行图例留出空间 =====================
    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.94,          # 略微下调，使整体更均衡
        bottom=0.06,       # 增大底部空间
        wspace=0.05,
        hspace=0.25,       # 增大行间距，为图例留白
    )

    # ===================== 添加两行图例 =====================
    import matplotlib.patches as mpatches

    # ---- 图例1：1970-2010 ----
    if counter_1:
        # 按组名排序，保持一致性
        sorted_groups_1 = sorted(counter_1.keys())
        handles1 = []
        labels1 = []
        for g in sorted_groups_1:
            freq = counter_1[g]
            label = f"{g} (n={freq})"
            handles1.append(mpatches.Patch(color=color_map[g]))
            labels1.append(label)
        # 每行最多显示6个
        ncol1 = min(6, len(handles1))
        # 获取第一行轴的下边界（相对于figure坐标）
        y0_ax1 = ax_bee_1.get_position().y0
        # 图例放在该行下方，y0_ax1 - 0.02（但要保证在figure内）
        y_legend1 = max(y0_ax1 - 0.02, 0.02)  # 不低于bottom边界
        fig.legend(
            handles=handles1,
            labels=labels1,
            loc='upper center',
            bbox_to_anchor=(0.5, y_legend1),
            ncol=ncol1,
            fontsize=7,
            frameon=False,
            title="1970-2010" if lang == "en" else "1970-2010",
            title_fontsize=7,
        )

    # ---- 图例2：2011-2024 ----
    if counter_2:
        sorted_groups_2 = sorted(counter_2.keys())
        handles2 = []
        labels2 = []
        for g in sorted_groups_2:
            freq = counter_2[g]
            label = f"{g} (n={freq})"
            handles2.append(mpatches.Patch(color=color_map[g]))
            labels2.append(label)
        ncol2 = min(6, len(handles2))
        y0_ax2 = ax_bee_2.get_position().y0
        y_legend2 = max(y0_ax2 - 0.02, 0.02)
        fig.legend(
            handles=handles2,
            labels=labels2,
            loc='upper center',
            bbox_to_anchor=(0.5, y_legend2),
            ncol=ncol2,
            fontsize=7,
            frameon=False,
            title="2011-2024" if lang == "en" else "2011-2024",
            title_fontsize=7,
        )

    # 输出文件名
    lang_tag = "zh" if lang == "zh" else "en"
    base_name = f"RF_SHAP_{title_prefix}_{lang_tag}"
    out_pdf = pjoin(FIG_DIR, base_name + ".pdf")
    out_png = pjoin(FIG_DIR, base_name + ".png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=400)
    fig.savefig(out_png, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"[OK] Saved combined figure ({lang_tag}) to:\n  {out_pdf}\n  {out_png}")


# ===================== MAIN =====================

def main():
    if not os.path.exists(MODEL8_DIR):
        print(f"[ERROR] MODEL8_DIR not found: {MODEL8_DIR}")
        return
    if not HAS_SHAP:
        print("[ERROR] shap is not installed. Please install it before plotting.")
        return

    sc_to_en_name = load_indicator_en_names()

    # 四组模型配置
    groups = [
        {
            "prefix": "patent_ROSG",
            "model_1970": "patent_ROSG_1970_2010",
            "model_2011": "patent_ROSG_2011_2024",
        },
        {
            "prefix": "paper_ROSG",
            "model_1970": "paper_ROSG_1970_2010",
            "model_2011": "paper_ROSG_2011_2024",
        },
        {
            "prefix": "patent_RISG",
            "model_1970": "patent_RISG_1970_2010",
            "model_2011": "patent_RISG_2011_2024",
        },
        {
            "prefix": "paper_RISG",
            "model_1970": "paper_RISG_1970_2010",
            "model_2011": "paper_RISG_2011_2024",
        },
    ]

    for grp in groups:
        prefix = grp["prefix"]

        if PLOT_ONLY_PATENT_ROSG and prefix != "patent_ROSG":
            print(f"[SKIP] {prefix} (PLOT_ONLY_PATENT_ROSG = True)")
            continue

        m1970 = grp["model_1970"]
        m2011 = grp["model_2011"]

        print(f"\n[Group] {prefix}: {m1970} & {m2011}")

        # 英文版
        plot_combined_for_model_pair(
            m1970,
            m2011,
            title_prefix=prefix,
            lang="en",
            sc_to_en_name=sc_to_en_name,
        )

        # 中文版
        plot_combined_for_model_pair(
            m1970,
            m2011,
            title_prefix=prefix,
            lang="zh",
            sc_to_en_name=sc_to_en_name,
        )

    print("\nDone. All combined figures are under:")
    print(f"  {FIG_DIR}")


if __name__ == "__main__":
    main()
