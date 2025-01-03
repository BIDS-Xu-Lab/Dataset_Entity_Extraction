import matplotlib.pyplot as plt
import json

# 数据
with open('/home/gy237/project/Biomedical_datasets/total_pubmed/abstracts_filtere_1-1575.json', 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

# 提取标签和数据
labels = list(data_dict.keys())
sizes = list(data_dict.values())

# 总和
total = sum(sizes)

# 动态隐藏小于1%的标签
filtered_labels = [label if size / total >= 0.01 else '' for label, size in zip(labels, sizes)]

# 绘制饼状图
plt.figure(figsize=(10, 8))

# 自定义函数：隐藏小于1%的标签
def autopct_func(pct):
    return f'{pct:.1f}%' if pct >= 1 else ''  # 仅显示 >=1% 的标签

# 绘制饼状图
plt.pie(
    sizes,
    labels=filtered_labels,  # 隐藏小标签
    autopct=autopct_func,  # 使用自定义函数
    startangle=140,
    textprops={'fontsize': 10},
)

plt.title("Data Distribution", fontsize=16)
plt.axis('equal')  # 确保饼图为圆形
output_path = "/home/gy237/abstracts_filtere_1-1575.png"  # 设置文件保存路径和名称
plt.savefig(output_path, dpi=1000, bbox_inches='tight')