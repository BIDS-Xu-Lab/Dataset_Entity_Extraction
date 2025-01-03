from scipy.stats import pearsonr
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import random


def draw(file, out_list, lab_list):
    sns.set(
        context="paper",
        style="ticks",
        rc={
            "figure.autolayout": True,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "figure.titlesize": 8,
            "axes.labelsize": 8,
            "axes.labelpad": 2,
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.figsize": (3.5, 3.5 / 1.6),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            #'lines.linewidth' : 1
        },
    )

    r = scipy.stats.pearsonr(out_list, lab_list)

    fig = plt.figure(figsize=(4, 4), dpi=1000, facecolor="w", edgecolor="k")
    fig.tight_layout(pad=1)

    sns.regplot(
        x=out_list,
        y=lab_list,
        scatter_kws={"s": 1, "linewidths": 0, "rasterized": True},
        line_kws={"linewidth": 2, "color": "black"},
        color="#0868ac",
        robust=1,
    )

    ax = plt.gca()
    ax.set_ylim(0, 18)
    ax.set_xlim(4, 13.6)
    ax.set_yticks([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15, 17.5])
    ax.set_xticks([4.0, 6.0, 8.0, 10.0, 12.0])

    plt.savefig(f"{file}_{r[0]}.pdf", bbox_inches="tight", dpi=1000)
    print("over")




# 示例列表
def parse(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
        label = [float(i.strip().split('\t')[1]) for i in data]
        pre = [float(i.strip().split('\t')[0]) for i in data]
    return label, pre

bio_lab, bio_pre = parse('/home/gy237/project/R/igem/biobert_k4_lr1e-05_7_0.64383.txt')
dna_lab, dna_pre = parse('/home/gy237/project/R/igem/DNABERT_30,000,000_0.9642978226844585_12_10.txt')
evo_lab, evo_pre = parse('/home/gy237/project/R/igem/Evolution_30,000,000_0.9585480246946544_12_7.txt')
dn2_lab, dn2_pre = parse('/home/gy237/project/R/igem/igem_bert2_weights_k4_lr1e-05_6_0.87224.txt')

# print(bio_lab[:5])
# print(dna_lab[:5])
# print(evo_lab[:5])
# print(dn2_lab[:5])


# 计算皮尔逊相关系数
pcc, _ = pearsonr(bio_lab, bio_pre)
print("PCC biobert:", pcc)

pcc, _ = pearsonr(dna_lab, dna_pre)
print("PCC DNABERT:", pcc)

pcc, _ = pearsonr(evo_lab, evo_pre)
print("PCC Evolution:", pcc)

pcc, _ = pearsonr(dn2_lab, dn2_pre)
print("PCC DNABERT2:", pcc)


gen_lab = [(bio_lab[i] + dna_lab[i] + evo_lab[i] + dn2_lab[i])/4 for i in range(len(bio_lab))]
gen_pre = [(bio_pre[i] + dna_pre[i] + evo_pre[i] + dn2_pre[i])/4 for i in range(len(bio_pre))]
pcc, _ = pearsonr(gen_lab, gen_pre)
print("PCC 平均数:", pcc) # 0.94156
# draw('/home/gy237/project/R/igem/general_all', gen_pre, gen_lab)



# 定义去除异常值的函数
def remove_outliers(row, method='std'):
    if method == 'std':
        mean, std = np.mean(row), np.std(row)
        return [x for x in row if abs(x - mean) <= std]
    elif method == 'iqr':
        q1, q3 = np.percentile(row, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [x for x in row if lower <= x <= upper]

# 去除差异大的数据并计算最终结果
final_predictions = []
for i in range(len(bio_pre)):
    row = np.array([dna_pre[i], evo_pre[i]]) # bio_pre[i], dn2_pre[i]
    valid_values = remove_outliers(row, method='std')
    if valid_values:  # 确保有剩余值
        final_predictions.append(np.mean(valid_values))
    else:
        final_predictions.append(np.mean(row))  # 如果全被去除，取均值代替

# 转为 numpy 数组
final_predictions = np.array(final_predictions)

# 计算皮尔森相关系数
correlation, _ = pearsonr(final_predictions, bio_lab)
print("最终皮尔森相关系数:", correlation) # 0.9454
draw('/home/gy237/project/R/igem/general_filtered_top2', final_predictions, bio_lab)
