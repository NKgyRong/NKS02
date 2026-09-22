import os
import glob
import ast
import statistics
from collections import defaultdict
import multiprocessing as mp
import pandas as pd

def parse_cid(cid_str):
    """
    解析CID字段，返回该专利包含的所有IPC subclass和IPC group（主组部分）。
    输入：CID字段的字符串，格式如 "{'A45D  34/04 ...', 'B05B  11/00 ...'}"
    输出：subclass列表, group列表（group为 "subclass 主组号"，如 "A45D 34"）
    """
    subclasses = []
    groups = []
    try:
        cid_set = ast.literal_eval(cid_str)  # 转换为Python set
    except Exception:
        # 如果解析失败，返回空列表
        return subclasses, groups

    for cat in cid_set:
        # 每个类别字符串，如 'A45D  34/04        20060101AFI20201210BHUS'
        parts = cat.split()
        if len(parts) >= 2:
            subclass = parts[0]                # 例如 'A45D'
            group_part = parts[1]               # 例如 '34/04'
            # 提取主组部分（'/'之前的内容）
            if '/' in group_part:
                main_group = group_part.split('/')[0]   # 例如 '34'
            else:
                main_group = group_part                   # 如果没有'/'，则整个作为主组
            group = f"{subclass} {main_group}"            # 例如 'A45D 34'
            subclasses.append(subclass)
            groups.append(group)
    return subclasses, groups


def process_file(file_path):
    """
    处理单个CSV文件，返回该文件的统计计数器。
    返回一个元组： (year_counts, economy_counts, subclass_counts, group_counts, economy_group_counts, total_patents)
    """
    # 初始化计数器
    year_counts = defaultdict(int)
    economy_counts = defaultdict(int)
    subclass_counts = defaultdict(int)
    group_counts = defaultdict(int)
    economy_group_counts = defaultdict(int)
    total_patents = 0

    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return (year_counts, economy_counts, subclass_counts,
                group_counts, economy_group_counts, total_patents)

    # 检查必要列是否存在
    required = {'Economies', 'UID', 'Year', 'CID'}
    if not required.issubset(df.columns):
        print(f"文件 {file_path} 缺少必要列，跳过")
        return (year_counts, economy_counts, subclass_counts,
                group_counts, economy_group_counts, total_patents)

    # 逐行处理
    for row in df.itertuples(index=False):
        economy = getattr(row, 'Economies')
        uid = getattr(row, 'UID')
        year = getattr(row, 'Year')
        cid_str = getattr(row, 'CID')

        # 跳过关键信息缺失的行
        if pd.isna(economy) or pd.isna(uid) or pd.isna(year) or pd.isna(cid_str):
            continue

        try:
            year = int(year)
        except:
            continue   # 年份无法转换为整数，跳过

        # 计数（假设UID不跨文件重复，直接累加）
        total_patents += 1
        year_counts[year] += 1
        economy_counts[economy] += 1

        # 解析CID，获取subclass和group
        subclasses, groups = parse_cid(cid_str)
        for sub in subclasses:
            subclass_counts[sub] += 1
        for grp in groups:
            group_counts[grp] += 1
            # 经济体-IPC Group组合
            economy_group_counts[(economy, grp)] += 1

    # 将defaultdict转换为普通dict以便序列化
    return (dict(year_counts), dict(economy_counts),
            dict(subclass_counts), dict(group_counts),
            dict(economy_group_counts), total_patents)


def merge_counts(results):
    """
    合并多个进程返回的计数结果。
    results: 每个元素是process_file返回的元组
    """
    total_patents = sum(r[5] for r in results)

    # 合并字典
    year_counts = defaultdict(int)
    economy_counts = defaultdict(int)
    subclass_counts = defaultdict(int)
    group_counts = defaultdict(int)
    economy_group_counts = defaultdict(int)

    for yc, ec, sc, gc, egc, _ in results:
        for k, v in yc.items():
            year_counts[k] += v
        for k, v in ec.items():
            economy_counts[k] += v
        for k, v in sc.items():
            subclass_counts[k] += v
        for k, v in gc.items():
            group_counts[k] += v
        for k, v in egc.items():
            economy_group_counts[k] += v

    return (dict(year_counts), dict(economy_counts),
            dict(subclass_counts), dict(group_counts),
            dict(economy_group_counts), total_patents)


def calculate_stats(counts_dict):
    """
    给定一个字典 {key: count}，返回 (数量, 最大值, 最小值, 中位数, 平均值)
    """
    values = list(counts_dict.values())
    if not values:
        return (0, 0, 0, 0, 0)
    n = len(counts_dict)
    max_val = max(values)
    min_val = min(values)
    med = statistics.median(values)
    avg = statistics.mean(values)
    return (n, max_val, min_val, med, avg)


def calculate_year_range_stats(year_counts, start, end):
    """
    计算指定年份区间内的统计量。
    返回 (区间内专利总数, 最大, 最小, 中位数, 平均值)
    """
    filtered = [cnt for year, cnt in year_counts.items() if start <= year <= end]
    if not filtered:
        return (0, 0, 0, 0, 0)
    total = sum(filtered)
    max_val = max(filtered)
    min_val = min(filtered)
    med = statistics.median(filtered)
    avg = statistics.mean(filtered)
    return (total, max_val, min_val, med, avg)


def main():
    data_dir = '/data01/rong_dataset/PhD_dataset/WWP_data/WWP_data_V1010/'
    output_file = '/data01/rong_dataset/PhD_dataset/WWP_data/all_summary.csv'

    # 获取所有CSV文件
    file_pattern = os.path.join(data_dir, '*.csv')
    file_list = glob.glob(file_pattern)
    if not file_list:
        print(f"在 {data_dir} 中未找到CSV文件")
        return

    print(f"找到 {len(file_list)} 个CSV文件")

    # 设置进程数，不超过CPU核心数，也不超过文件数
    num_workers = min(mp.cpu_count(), len(file_list))
    print(f"使用 {num_workers} 个进程并行处理")

    # 多进程处理文件
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_file, file_list)

    # 合并结果
    year_counts, economy_counts, subclass_counts, group_counts, economy_group_counts, total_patents = merge_counts(results)

    print(f"总专利数: {total_patents}")
    print(f"年份范围: {min(year_counts.keys())} - {max(year_counts.keys())}")

    # 2. 整体年份统计
    year_vals = list(year_counts.values())
    year_span = f"{min(year_counts.keys())}-{max(year_counts.keys())}"
    year_max = max(year_vals)
    year_min = min(year_vals)
    year_median = statistics.median(year_vals)
    year_mean = statistics.mean(year_vals)

    # 3-6. 各时间段统计
    range1 = calculate_year_range_stats(year_counts, 1782, 1859)
    range2 = calculate_year_range_stats(year_counts, 1860, 1969)
    range3 = calculate_year_range_stats(year_counts, 1970, 2010)
    range4 = calculate_year_range_stats(year_counts, 2011, 2024)

    # 7. 经济体统计
    economy_stats = calculate_stats(economy_counts)

    # 8. IPC Subclass 统计
    subclass_stats = calculate_stats(subclass_counts)

    # 9. IPC Group 统计
    group_stats = calculate_stats(group_counts)

    # 10. 经济体-IPC Group 组合统计
    economy_group_stats = calculate_stats(economy_group_counts)

    # 构建输出行（一行包含所有统计量）
    output_row = {
        'total_patents': total_patents,
        'year_span': year_span,
        'year_max': year_max,
        'year_min': year_min,
        'year_median': year_median,
        'year_mean': year_mean,
        'year_1782_1859_total': range1[0],
        'year_1782_1859_max': range1[1],
        'year_1782_1859_min': range1[2],
        'year_1782_1859_median': range1[3],
        'year_1782_1859_mean': range1[4],
        'year_1860_1969_total': range2[0],
        'year_1860_1969_max': range2[1],
        'year_1860_1969_min': range2[2],
        'year_1860_1969_median': range2[3],
        'year_1860_1969_mean': range2[4],
        'year_1970_2010_total': range3[0],
        'year_1970_2010_max': range3[1],
        'year_1970_2010_min': range3[2],
        'year_1970_2010_median': range3[3],
        'year_1970_2010_mean': range3[4],
        'year_2011_2024_total': range4[0],
        'year_2011_2024_max': range4[1],
        'year_2011_2024_min': range4[2],
        'year_2011_2024_median': range4[3],
        'year_2011_2024_mean': range4[4],
        'economy_count': economy_stats[0],
        'economy_max': economy_stats[1],
        'economy_min': economy_stats[2],
        'economy_median': economy_stats[3],
        'economy_mean': economy_stats[4],
        'subclass_count': subclass_stats[0],
        'subclass_max': subclass_stats[1],
        'subclass_min': subclass_stats[2],
        'subclass_median': subclass_stats[3],
        'subclass_mean': subclass_stats[4],
        'group_count': group_stats[0],
        'group_max': group_stats[1],
        'group_min': group_stats[2],
        'group_median': group_stats[3],
        'group_mean': group_stats[4],
        'economy_group_count': economy_group_stats[0],
        'economy_group_max': economy_group_stats[1],
        'economy_group_min': economy_group_stats[2],
        'economy_group_median': economy_group_stats[3],
        'economy_group_mean': economy_group_stats[4],
    }

    # 输出为CSV（一行）
    df_out = pd.DataFrame([output_row])
    df_out.to_csv(output_file, index=False)
    print(f"汇总统计已保存至: {output_file}")


if __name__ == "__main__":
    main()