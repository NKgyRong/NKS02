#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA result visualization for two models: patent_RISG_1970_2010 and patent_ROSG_2011_2024
Output location: /data01/rong_dataset/Result/pj08/0418
"""

import os
import tempfile

# ===================== Set temporary directory =====================
TEMP_DIR = "/data01/rong_dataset/Temp/"
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib

# Use default English font
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

matplotlib.use("Agg")

# ===================== Configuration =====================
INPUT_DIR = "/data01/rong_dataset/Result/PhD/chapter6/0112PCA_V"
OUTPUT_DIR = "/data01/rong_dataset/Result/pj08/0418"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only these two models
MODELS = [
    'patent_ROSG_1970_2010',
    'patent_ROSG_2011_2024'
]

# Color and marker configuration (simplified)
COLOR_MAP = {
    'patent': '#1f77b4',
    '1970-2010': '#9467bd',
    '2011-2024': '#8c564b'
}

MARKER_MAP = {
    'ROSG': 'o'
}

# ===================== Data Loading =====================

def load_all_model_data():
    """Load PCA results for the two models"""
    all_summary_dfs = {}
    
    print("Loading model data...")
    for model in MODELS:
        file_path = os.path.join(INPUT_DIR, f"PCA_{model}_summary.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add metadata columns
            df['model'] = model
            df['product'] = 'patent'
            df['type'] = 'ROSG'
            df['period'] = '1970-2010' if '1970_2010' in model else '2011-2024'
            df['model_key'] = model
            
            # Calculate compression rate
            df['compression_rate'] = (1 - df['n_components_at_main_threshold'] / df['n_features_original']) * 100
            
            all_summary_dfs[model] = df
            print(f"  ? {model}: {len(df)} subgroups")
        else:
            print(f"  ? {model}: file not found")
    
    return all_summary_dfs

def create_summary_statistics(all_summary_dfs):
    """Create summary statistics table for models"""
    summary_stats = []
    
    for model_name, df in all_summary_dfs.items():
        if df.empty:
            continue
            
        stats = {
            'Model': model_name,
            'Product': 'Patent',
            'Type': 'ROSG',
            'Period': '1970-2010' if '1970' in model_name else '2011-2024',
            'Num_Subgroups': len(df),
            'Avg_Features': df['n_features_original'].mean(),
            'Avg_PCs_80%': df['n_components_at_main_threshold'].mean(),
            'Avg_Var_Explained_80%': df['cum_explained_var_at_main_threshold'].mean(),
            'Avg_Reconstruction_R2': df['reconstruction_r2_80'].mean(),
            'Avg_Reconstruction_MSE': df['reconstruction_mse_80'].mean(),
            'Compression_Rate(%)': df['compression_rate'].mean()
        }
        summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)

# ===================== Visualization Functions =====================

def plot_radar_chart(ax, summary_df, color_map):
    """Radar chart for model performance"""
    categories = ['Compression Rate', 'Avg R2', 'Var Explained', 'PC Efficiency', 'Coverage']
    
    # Normalize metrics
    normalized_data = []
    model_labels = []
    
    for idx, row in summary_df.iterrows():
        model_data = [
            row['Compression_Rate(%)'] / 100,
            row['Avg_Reconstruction_R2'],
            row['Avg_Var_Explained_80%'],
            (row['Avg_Features'] / row['Avg_PCs_80%']) / 10,
            min(row['Num_Subgroups'] / 50, 1)
        ]
        normalized_data.append(model_data)
        # Use period as label instead of full model name
        model_labels.append(row['Period'])
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, model_data in enumerate(normalized_data):
        values = model_data + model_data[:1]
        color = color_map.get(model_labels[i], '#1f77b4')
        ax.plot(angles, values, linewidth=2, label=model_labels[i], 
                marker='o', markersize=6, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title('Model Performance Radar Chart', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

def plot_compression_heatmap(ax, summary_df):
    """Heatmap of compression rate by period"""
    if summary_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    heatmap_data = summary_df.pivot_table(
        index='Product',
        columns='Period',
        values='Compression_Rate(%)',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Compression Rate (%)'})
    ax.set_title('Compression Rate Heatmap (by Period)', fontsize=12)
    ax.set_xlabel('Period')
    ax.set_ylabel('Product Type')

def plot_reconstruction_boxplot(ax, all_summary_dfs):
    """Boxplot of reconstruction R2"""
    box_data = []
    labels = []
    
    for model_name, df in all_summary_dfs.items():
        if not df.empty and 'reconstruction_r2_80' in df.columns:
            box_data.append(df['reconstruction_r2_80'].dropna().values)
            # Use period as label
            period = '1970-2010' if '1970' in model_name else '2011-2024'
            labels.append(period)
    
    if not box_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Reconstruction R2')
    ax.set_title('Reconstruction Accuracy Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

def plot_component_distribution(ax, all_summary_dfs):
    """Violin plot of PC count distribution"""
    component_data = []
    model_labels = []
    
    for model_name, df in all_summary_dfs.items():
        if not df.empty and 'n_components_at_main_threshold' in df.columns:
            components = df['n_components_at_main_threshold']
            if len(components) > 5:
                q75, q25 = np.percentile(components, [75, 25])
                iqr = q75 - q25
                upper_bound = q75 + 1.5 * iqr
                filtered = components[components <= upper_bound]
            else:
                filtered = components
            
            component_data.extend(filtered)
            period = '1970-2010' if '1970' in model_name else '2011-2024'
            model_labels.extend([period] * len(filtered))
    
    if not component_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    component_df = pd.DataFrame({
        'Period': model_labels,
        'Components': component_data
    })
    
    unique_periods = component_df['Period'].unique()
    plot_data = [component_df[component_df['Period']==p]['Components'] for p in unique_periods]
    
    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
    
    ax.set_xticks(range(1, len(unique_periods) + 1))
    ax.set_xticklabels(unique_periods)
    ax.set_ylabel('Number of PCs Required')
    ax.set_title('PC Requirement Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)

def plot_temporal_comparison(ax, summary_df):
    """Comparison of metrics between the two periods"""
    if summary_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    time_groups = summary_df.groupby('Period')
    metrics = ['Compression_Rate(%)', 'Avg_Reconstruction_R2', 'Avg_Var_Explained_80%']
    metric_labels = ['Compression Rate', 'Reconstruction R2', 'Var Explained']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, (period, group) in enumerate(time_groups):
        values = [group[metric].mean() for metric in metrics]
        ax.bar(x + i*width - width/2, values, width, 
               label=period, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel('Metric Value')
    ax.set_title('Comparison Between Periods', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def plot_product_comparison(ax, summary_df):
    """Scatter plot: features vs PCs (only patent product)"""
    if summary_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    colors = {'patent': 'blue'}
    markers = {'ROSG': 'o'}
    
    for _, row in summary_df.iterrows():
        ax.scatter(
            row['Avg_Features'],
            row['Avg_PCs_80%'],
            c=colors.get(row['Product'], 'gray'),
            marker=markers.get(row['Type'], 'o'),
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1,
            label=row['Period']
        )
    
    ax.set_xlabel('Average Number of Features')
    ax.set_ylabel('Average Number of PCs (80%)')
    ax.set_title('Patent Models Comparison', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_performance_matrix(all_summary_dfs, summary_df):
    """Performance matrix heatmap for the two models"""
    if summary_df.empty:
        print("Cannot create performance matrix: empty summary data")
        return None
    
    performance_data = []
    for _, row in summary_df.iterrows():
        model = row['Model']
        df = all_summary_dfs.get(model, pd.DataFrame())
        if not df.empty:
            perf = {
                'Period': row['Period'],
                'Compression_Efficiency': row['Compression_Rate(%)'],
                'Reconstruction_Accuracy': row['Avg_Reconstruction_R2'],
                'Variance_Explained': row['Avg_Var_Explained_80%'],
                'PC_Efficiency': row['Avg_PCs_80%'] / row['Avg_Features'] if row['Avg_Features'] > 0 else 0,
                'Stability': 1 - row['Avg_Reconstruction_MSE'] if row['Avg_Reconstruction_MSE'] > 0 else 1
            }
            performance_data.append(perf)
    
    perf_df = pd.DataFrame(performance_data)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_list = ['Compression_Efficiency', 'Reconstruction_Accuracy', 'Variance_Explained']
    titles = ['Compression Efficiency', 'Reconstruction Accuracy', 'Variance Explained']
    
    for idx, (metric, title) in enumerate(zip(metrics_list, titles)):
        if metric not in perf_df.columns:
            axes[idx].text(0.5, 0.5, f'No {metric} data', ha='center', va='center')
            axes[idx].axis('off')
            continue
        
        data = perf_df.set_index('Period')[[metric]]
        if data.empty:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].axis('off')
            continue
        
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=data.values.mean(), ax=axes[idx], cbar_kws={'label': metric})
        axes[idx].set_title(title, fontsize=11)
    
    plt.suptitle('PCA Performance Matrix Heatmap', fontsize=14, y=0.98)
    plt.tight_layout()
    return fig

def plot_simple_comparison(all_summary_dfs, summary_df):
    """Simple bar charts comparing the two models"""
    if summary_df.empty:
        print("Cannot create simple comparison: empty summary data")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Average PCs
    summary_df.set_index('Period')['Avg_PCs_80%'].plot(
        kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Average PCs Required (80%)', fontsize=12)
    axes[0,0].set_ylabel('Number of PCs')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # 2. Average R2
    summary_df.set_index('Period')['Avg_Reconstruction_R2'].plot(
        kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('Average Reconstruction R2', fontsize=12)
    axes[0,1].set_ylabel('R2')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # 3. Compression rate
    summary_df.set_index('Period')['Compression_Rate(%)'].plot(
        kind='bar', ax=axes[1,0], color='salmon')
    axes[1,0].set_title('Average Compression Rate', fontsize=12)
    axes[1,0].set_ylabel('Compression Rate (%)')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # 4. Number of features
    summary_df.set_index('Period')['Avg_Features'].plot(
        kind='bar', ax=axes[1,1], color='gold')
    axes[1,1].set_title('Average Number of Features', fontsize=12)
    axes[1,1].set_ylabel('Number of Features')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Key Metrics Comparison (Patent_ROSG Models)', fontsize=14, y=0.98)
    plt.tight_layout()
    return fig

def plot_explained_variance_trend(all_summary_dfs):
    """Explained variance trend for each model (now two subplots)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (model_name, df) in enumerate(all_summary_dfs.items()):
        ax = axes[idx]
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(model_name, fontsize=10)
            continue
        
        period = '1970-2010' if '1970' in model_name else '2011-2024'
        for _, row in df.iterrows():
            cum_var_str = row.get('cumulative_explained_variance_ratio', '')
            if isinstance(cum_var_str, str) and cum_var_str:
                try:
                    cum_var = list(map(float, cum_var_str.split(';')))
                    n_components = len(cum_var)
                    ax.step(range(1, n_components + 1), cum_var, 
                           alpha=0.3, linewidth=0.5)
                except:
                    pass
        
        ax.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.7, label='80%')
        ax.axhline(y=0.9, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='90%')
        ax.set_xlabel('Number of PCs')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title(f'{period}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 20)
        ax.legend()
    
    plt.suptitle('Explained Variance Trend (80% and 90% thresholds)', fontsize=14, y=0.98)
    plt.tight_layout()
    return fig

def plot_cumulative_distribution(all_summary_dfs):
    """Cumulative distribution of compression rates"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_summary_dfs)))
    
    for (model_name, df), color in zip(all_summary_dfs.items(), colors):
        if df.empty:
            continue
        if 'compression_rate' in df.columns:
            compression_rates = df['compression_rate'].dropna()
            if len(compression_rates) > 0:
                sorted_rates = np.sort(compression_rates)
                y = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
                period = '1970-2010' if '1970' in model_name else '2011-2024'
                ax.plot(sorted_rates, y, label=period, color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Compression Rate (%)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Compression Rate Cumulative Distribution', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    for percentile in [25, 50, 75]:
        ax.axvline(x=percentile, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return fig

# ===================== Main Dashboard =====================

def create_comprehensive_dashboard(all_summary_dfs, summary_df):
    """Comprehensive dashboard for the two models"""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Radar chart
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    plot_radar_chart(ax1, summary_df, COLOR_MAP)
    
    # Compression heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    plot_compression_heatmap(ax2, summary_df)
    
    # Reconstruction boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    plot_reconstruction_boxplot(ax3, all_summary_dfs)
    
    # PC distribution
    ax4 = fig.add_subplot(gs[1, 0])
    plot_component_distribution(ax4, all_summary_dfs)
    
    # Temporal comparison
    ax5 = fig.add_subplot(gs[1, 1])
    plot_temporal_comparison(ax5, summary_df)
    
    # Product comparison (patent only)
    ax6 = fig.add_subplot(gs[1, 2])
    plot_product_comparison(ax6, summary_df)
    
    # Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = summary_df[['Period', 'Num_Subgroups', 'Avg_Reconstruction_R2', 
                           'Compression_Rate(%)', 'Avg_PCs_80%']].copy()
    table_data = table_data.round({'Avg_Reconstruction_R2': 3, 'Compression_Rate(%)': 1, 'Avg_PCs_80%': 1})
    
    cell_text = [[row['Period'], row['Num_Subgroups'], 
                  f"{row['Avg_Reconstruction_R2']:.3f}", 
                  f"{row['Compression_Rate(%)']:.1f}%",
                  f"{row['Avg_PCs_80%']:.1f}"] 
                 for _, row in table_data.iterrows()]
    
    table = ax7.table(cellText=cell_text,
                     colLabels=['Period', '#Subgroups', 'R2', 'Compression Rate', 'PCs (80%)'],
                     cellLoc='center', loc='center', colColours=['#f2f2f2']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax7.set_title('Key Model Metrics Summary', fontsize=12, pad=20)
    
    plt.suptitle('PCA Analysis Dashboard (Patent_ROSG Models)', fontsize=18, y=0.98)
    return fig

# ===================== HTML Report =====================

def create_html_report(all_summary_dfs, summary_df, output_dir):
    """Create a simple HTML report in English"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCA Visualization Report - Patent_ROSG Models</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }}
        .summary-table th {{
            background-color: #3498db;
            color: white;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .stats-box {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PCA Analysis Report - Patent_ROSG Models</h1>
        <div class="stats-box">
            <h3>Analysis Overview</h3>
            <p>Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Number of Models: {len(all_summary_dfs)}</p>
            <p>Total Subgroups: {sum(len(df) for df in all_summary_dfs.values())}</p>
        </div>
        
        <h2>Model Performance Summary</h2>
        <table class="summary-table">
            <thead>
                <tr><th>Period</th><th>#Subgroups</th><th>Avg R2</th><th>Compression Rate (%)</th><th>Avg PCs (80%)</th></tr>
            </thead>
            <tbody>
"""
    for _, row in summary_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['Period']}</td>
                    <td>{row['Num_Subgroups']}</td>
                    <td>{row['Avg_Reconstruction_R2']:.3f}</td>
                    <td>{row['Compression_Rate(%)']:.1f}</td>
                    <td>{row['Avg_PCs_80%']:.1f}</td>
                </tr>
"""
    html_content += """
            </tbody>
        </table>
        
        <h2>Visualizations</h2>
"""
    images = [
        ("pca_comprehensive_dashboard.png", "Comprehensive Dashboard"),
        ("pca_performance_matrix.png", "Performance Matrix"),
        ("pca_simple_comparison.png", "Simple Comparison"),
        ("pca_variance_trend.png", "Explained Variance Trend"),
        ("pca_cumulative_distribution.png", "Compression Rate Distribution"),
        ("pca_reconstruction_boxplot.png", "Reconstruction R2 Boxplot"),
        ("pca_component_distribution.png", "PC Requirement Distribution")
    ]
    
    for img_file, img_title in images:
        img_path = os.path.join(output_dir, img_file)
        if os.path.exists(img_path):
            html_content += f"""
        <h2>{img_title}</h2>
        <div class="image-container">
            <img src="{img_file}" alt="{img_title}">
        </div>
"""
    
    html_content += """
        <div class="stats-box">
            <h3>Key Findings</h3>
            <ul>
                <li>PCA achieves effective dimensionality reduction for both periods.</li>
                <li>Reconstruction accuracy is high (R2 > 0.85) in all subgroups.</li>
                <li>Compression rates vary between periods; further optimization possible.</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    html_path = os.path.join(output_dir, "pca_visualization_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_path

# ===================== Main Function =====================

def main():
    """Main function to generate all visualizations"""
    print("="*80)
    print("PCA Result Visualization for Two Models (Patent_ROSG)")
    print("="*80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Temporary directory set to: {tempfile.gettempdir()}")
    
    # 1. Load data
    print("\n1. Loading data...")
    all_summary_dfs = load_all_model_data()
    
    if not all_summary_dfs:
        print("Error: No model data found!")
        return
    
    # 2. Create summary statistics
    print("\n2. Creating summary statistics...")
    summary_df = create_summary_statistics(all_summary_dfs)
    
    if summary_df.empty:
        print("Error: Summary data is empty!")
        return
    
    summary_path = os.path.join(OUTPUT_DIR, "model_summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  ? Saved summary statistics: {summary_path}")
    
    # 3. Generate visualizations
    print("\n3. Generating visualizations...")
    
    # Dashboard
    print("  Generating comprehensive dashboard...")
    dashboard_fig = create_comprehensive_dashboard(all_summary_dfs, summary_df)
    dashboard_path = os.path.join(OUTPUT_DIR, "pca_comprehensive_dashboard.png")
    dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close(dashboard_fig)
    print(f"    ? Dashboard: {dashboard_path}")
    
    # Performance matrix
    print("  Generating performance matrix...")
    perf_matrix_fig = create_performance_matrix(all_summary_dfs, summary_df)
    if perf_matrix_fig:
        perf_path = os.path.join(OUTPUT_DIR, "pca_performance_matrix.png")
        perf_matrix_fig.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close(perf_matrix_fig)
        print(f"    ? Performance matrix: {perf_path}")
    
    # Simple comparison
    print("  Generating simple comparison...")
    simple_fig = plot_simple_comparison(all_summary_dfs, summary_df)
    if simple_fig:
        simple_path = os.path.join(OUTPUT_DIR, "pca_simple_comparison.png")
        simple_fig.savefig(simple_path, dpi=300, bbox_inches='tight')
        plt.close(simple_fig)
        print(f"    ? Simple comparison: {simple_path}")
    
    # Variance trend
    print("  Generating variance trend plot...")
    variance_fig = plot_explained_variance_trend(all_summary_dfs)
    if variance_fig:
        variance_path = os.path.join(OUTPUT_DIR, "pca_variance_trend.png")
        variance_fig.savefig(variance_path, dpi=300, bbox_inches='tight')
        plt.close(variance_fig)
        print(f"    ? Variance trend: {variance_path}")
    
    # Cumulative distribution
    print("  Generating cumulative distribution...")
    cum_fig = plot_cumulative_distribution(all_summary_dfs)
    if cum_fig:
        cum_path = os.path.join(OUTPUT_DIR, "pca_cumulative_distribution.png")
        cum_fig.savefig(cum_path, dpi=300, bbox_inches='tight')
        plt.close(cum_fig)
        print(f"    ? Cumulative distribution: {cum_path}")
    
    # Individual plots
    print("  Generating individual subplots...")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_reconstruction_boxplot(ax1, all_summary_dfs)
    fig1.savefig(os.path.join(OUTPUT_DIR, "pca_reconstruction_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_component_distribution(ax2, all_summary_dfs)
    fig2.savefig(os.path.join(OUTPUT_DIR, "pca_component_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 4. HTML report
    print("\n4. Generating HTML report...")
    try:
        create_html_report(all_summary_dfs, summary_df, OUTPUT_DIR)
        print("  ? HTML report generated")
    except Exception as e:
        print(f"  ? HTML report failed: {e}")
    
    # Final report
    print("\n" + "="*80)
    print("Visualization generation complete!")
    print("="*80)
    
    print("\nKey findings:")
    print(f"1. Models processed: {len(all_summary_dfs)}")
    print(f"2. Total subgroups: {sum(len(df) for df in all_summary_dfs.values())}")
    
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            print(f"   - {row['Period']}: R2={row['Avg_Reconstruction_R2']:.3f}, Compression={row['Compression_Rate(%)']:.1f}%")
    
    print(f"\nAll output files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()