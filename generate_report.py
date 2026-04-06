"""Generate PDF report with all experiment results."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Data
# ============================================================
methods_all = ['none','eacp','eacp_gini','eacp_top2','eacp_gini_norm','eacp_top2_norm',
               'eacp_adaptive','eacp_online','eacp_sliding',
               'eacp_top2_norm_adaptive','eacp_adaptive_scaling',
               'tent_ecp','tent_ecp_adaptive']
labels_all = ['SplitCP','EaCP','EaCP+Gini(raw)','EaCP+Top2(raw)',
              'EaCP+Gini_norm','EaCP+Top2_norm',
              'EaCP+AdaptBeta','EaCP+Online','EaCP+Sliding',
              'EaCP+Top2n+AdaptB','EaCP+AdaptScale',
              'Tent+ECP','Tent+AdaptBeta']

# Classify methods
type_map = {
    'SplitCP': 'Baseline',
    'EaCP': 'Original',
    'EaCP+Gini(raw)': 'Mod1: Uncertainty',
    'EaCP+Top2(raw)': 'Mod1: Uncertainty',
    'EaCP+Gini_norm': 'Mod1: Uncertainty',
    'EaCP+Top2_norm': 'Mod1: Uncertainty',
    'EaCP+AdaptBeta': 'Mod2: Adaptive Beta',
    'EaCP+Online': 'Mod2: Adaptive Beta',
    'EaCP+Sliding': 'Mod2: Adaptive Beta',
    'EaCP+Top2n+AdaptB': 'Combination',
    'EaCP+AdaptScale': 'Mod3: Adaptive Scale',
    'Tent+ECP': 'Mod3: Tent TTA',
    'Tent+AdaptBeta': 'Combination',
}
colors_type = {
    'Baseline': '#888888',
    'Original': '#2196F3',
    'Mod1: Uncertainty': '#FF9800',
    'Mod2: Adaptive Beta': '#4CAF50',
    'Mod3: Adaptive Scale': '#E91E63',
    'Mod3: Tent TTA': '#9C27B0',
    'Combination': '#F44336',
}

t1 = pd.read_csv('results/summary_table1.csv')
t2 = pd.read_csv('results/summary_table2_avg.csv')
t2d = pd.read_csv('results/summary_table2_detail.csv')

corrs = ['contrast', 'brightness', 'gaussian_noise', 'motion_blur']
corr_labels = ['Contrast', 'Brightness', 'Gaussian Noise', 'Motion Blur']

# ============================================================
# PDF generation
# ============================================================
with PdfPages('results/experiment_report.pdf') as pdf:

    # ---- PAGE 1: Title ----
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.6, 'Adapting Prediction Sets to\nDistribution Shifts Without Labels',
             ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.45, 'Course Project: Extensions & Experimental Results',
             ha='center', va='center', fontsize=16, color='#555')
    fig.text(0.5, 0.30, 'Modifications:\n'
             '1. Alternative Uncertainty Functions (Gini, Top-2 Margin)\n'
             '2. Adaptive Beta Quantile Selection\n'
             '3. Tent TTA & Adaptive Scaling Factor',
             ha='center', va='center', fontsize=13, color='#333',
             linespacing=1.6)
    pdf.savefig(fig)
    plt.close()

    # ---- PAGE 2: Table 1 ----
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Table 1: Coverage and Set Size on Stationary Distribution Shifts\n(Target Coverage = 0.90)',
                 fontsize=14, fontweight='bold', pad=20)

    col_labels = ['Method', 'Type', 'V2\nCov', 'V2\nSize', 'R\nCov', 'R\nSize', 'A\nCov', 'A\nSize']
    cell_data = []
    cell_colors = []
    for _, row in t1.iterrows():
        m = row['Method']
        tp = type_map.get(m, '')
        cells = [m, tp,
                 f"{row['V2_Cov']:.3f}", f"{row['V2_Size']:.1f}",
                 f"{row['R_Cov']:.3f}", f"{row['R_Size']:.1f}",
                 f"{row['A_Cov']:.3f}", f"{row['A_Size']:.1f}"]
        cell_data.append(cells)
        # Color coverage cells
        c_row = ['white'] * 8
        for idx, col in [(2, 'V2_Cov'), (4, 'R_Cov'), (6, 'A_Cov')]:
            v = row[col]
            if v >= 0.9:
                c_row[idx] = '#C8E6C9'  # green
            elif v >= 0.85:
                c_row[idx] = '#FFF9C4'  # yellow
            else:
                c_row[idx] = '#FFCDD2'  # red
        cell_colors.append(c_row)

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    fig.text(0.5, 0.08, 'Green = Coverage >= 0.90 (target met)  |  Yellow = 0.85-0.90  |  Red = < 0.85',
             ha='center', fontsize=10, color='#555')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---- PAGE 3: Table 2 (avg) ----
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Table 2: Average Coverage and Set Size on ImageNet-C\n(Averaged over Severity 1-5, Target Coverage = 0.90)',
                 fontsize=14, fontweight='bold', pad=20)

    col_labels2 = ['Method', 'Contr\nCov', 'Contr\nSize', 'Bright\nCov', 'Bright\nSize',
                   'GaussN\nCov', 'GaussN\nSize', 'MotBlur\nCov', 'MotBlur\nSize']
    cell_data2 = []
    cell_colors2 = []
    for _, row in t2.iterrows():
        cells = [row['Method']]
        c_row = ['white'] * 9
        for i, corr in enumerate(corrs):
            cov = row[f'{corr}_Cov']
            size = row[f'{corr}_Size']
            cells.extend([f"{cov:.3f}", f"{size:.1f}"])
            idx = 1 + i * 2
            if cov >= 0.9:
                c_row[idx] = '#C8E6C9'
            elif cov >= 0.85:
                c_row[idx] = '#FFF9C4'
            else:
                c_row[idx] = '#FFCDD2'
        cell_data2.append(cells)
        cell_colors2.append(c_row)

    table2 = ax.table(cellText=cell_data2, colLabels=col_labels2,
                      cellColours=cell_colors2,
                      loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8.5)
    table2.scale(1, 1.4)
    for j in range(len(col_labels2)):
        table2[0, j].set_facecolor('#1565C0')
        table2[0, j].set_text_props(color='white', fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---- PAGE 4: Table 2 Detail (per severity) ----
    for corr, corr_label in zip(corrs, corr_labels):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title(f'Table 2 Detail: {corr_label} — Coverage per Severity Level',
                     fontsize=14, fontweight='bold', pad=20)

        col_labels3 = ['Method', 'S1', 'S2', 'S3', 'S4', 'S5', 'Avg']
        cell_data3 = []
        cell_colors3 = []
        sub = t2d[t2d['Corruption'] == corr]
        for label in labels_all:
            rows = sub[sub['Method'] == label].sort_values('Severity')
            if len(rows) == 0:
                continue
            covs = rows['Coverage'].values
            avg = covs.mean()
            cells = [label] + [f"{c:.3f}" for c in covs] + [f"{avg:.3f}"]
            c_row = ['white'] * 7
            for idx in range(1, 7):
                v = float(cells[idx])
                if v >= 0.9:
                    c_row[idx] = '#C8E6C9'
                elif v >= 0.85:
                    c_row[idx] = '#FFF9C4'
                else:
                    c_row[idx] = '#FFCDD2'
            cell_data3.append(cells)
            cell_colors3.append(c_row)

        table3 = ax.table(cellText=cell_data3, colLabels=col_labels3,
                          cellColours=cell_colors3,
                          loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 1.4)
        for j in range(len(col_labels3)):
            table3[0, j].set_facecolor('#1565C0')
            table3[0, j].set_text_props(color='white', fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    # ---- PAGE 8: Bar chart — Coverage comparison on V2 ----
    fig, ax = plt.subplots(figsize=(11, 6))
    covs = t1['V2_Cov'].values
    colors = [colors_type.get(type_map.get(l, ''), '#888') for l in labels_all]
    bars = ax.barh(range(len(labels_all)), covs, color=colors, edgecolor='white')
    ax.set_yticks(range(len(labels_all)))
    ax.set_yticklabels(labels_all, fontsize=10)
    ax.axvline(x=0.9, color='red', linestyle='--', linewidth=2, label='Target (0.90)')
    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_title('Coverage Comparison on ImageNet-V2', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0.7, 0.95)
    ax.invert_yaxis()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---- PAGE 9: Coverage vs Set Size scatter (V2) ----
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (label, method) in enumerate(zip(labels_all, methods_all)):
        matched = t1[t1['Method'] == label]
        if len(matched) == 0:
            continue
        row = matched.iloc[0]
        c = colors_type.get(type_map.get(label, ''), '#888')
        ax.scatter(row['V2_Size'], row['V2_Cov'], c=c, s=120, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(label, (row['V2_Size'], row['V2_Cov']),
                    textcoords="offset points", xytext=(8, 4), fontsize=7.5)
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, label='Target Coverage (0.90)')
    ax.set_xlabel('Set Size (lower is better)', fontsize=12)
    ax.set_ylabel('Coverage (higher is better)', fontsize=12)
    ax.set_title('Coverage vs Set Size Trade-off on ImageNet-V2', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---- PAGE 10: Coverage vs Severity curves (4 corruptions) ----
    key_methods = ['none', 'eacp', 'eacp_adaptive', 'eacp_adaptive_scaling', 'tent_ecp_adaptive',
                   'eacp_gini', 'eacp_gini_norm']
    key_labels = ['SplitCP', 'EaCP', 'EaCP+AdaptBeta', 'EaCP+AdaptScale', 'Tent+AdaptBeta',
                  'EaCP+Gini(raw)', 'EaCP+Gini_norm']
    key_colors = ['#888888', '#2196F3', '#4CAF50', '#E91E63', '#F44336', '#FF9800', '#FFC107']
    key_styles = ['-', '-', '--', '--', ':', '-', '-']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for idx, (corr, corr_label) in enumerate(zip(corrs, corr_labels)):
        ax = axes[idx // 2][idx % 2]
        sub = t2d[t2d['Corruption'] == corr]
        for method, label, color, ls in zip(key_methods, key_labels, key_colors, key_styles):
            rows = sub[sub['Method'] == label].sort_values('Severity')
            if len(rows) > 0:
                ax.plot(rows['Severity'], rows['Coverage'], marker='o', label=label,
                        color=color, linestyle=ls, linewidth=2, markersize=5)
        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(corr_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Coverage')
        ax.set_ylim(0, 1.0)
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[0][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7.5)
    fig.suptitle('Coverage vs Severity on ImageNet-C', fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---- PAGE 11: Coverage vs Severity (Set Size) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for idx, (corr, corr_label) in enumerate(zip(corrs, corr_labels)):
        ax = axes[idx // 2][idx % 2]
        sub = t2d[t2d['Corruption'] == corr]
        for method, label, color, ls in zip(key_methods, key_labels, key_colors, key_styles):
            rows = sub[sub['Method'] == label].sort_values('Severity')
            if len(rows) > 0:
                ax.plot(rows['Severity'], rows['SetSize'], marker='s', label=label,
                        color=color, linestyle=ls, linewidth=2, markersize=5)
        ax.set_title(corr_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Set Size')
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[0][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7.5)
    fig.suptitle('Set Size vs Severity on ImageNet-C', fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("Report saved to: results/experiment_report.pdf")
