from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

# 加载预训练的BERT模型和tokenizer
model_path = "aitslab/biobert_huner_gene_v1"
tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir='/home/gy237/project/download_data')
model = BertForSequenceClassification.from_pretrained(model_path, cache_dir='/home/gy237/project/download_data')
if True:
    class RegressionModel(nn.Module):
        def __init__(self):
            super(RegressionModel, self).__init__()
            self.fc1 = nn.Linear(768, 128)  # Linear：对输入做一次线性变换，由768维到128维
            self.relu = (
                nn.ReLU()
            )  # ReLU：作为进入下一层神经元的激活函数，定义了在进入下一层前的非线性输出结果，通常为max(0,x)。
            self.fc2 = nn.Linear(128, 64)

            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            # x=x.view(-1)，即reshape这个输入的x
            # 其实就是定义前馈这个过程的函数

            x = self.fc1(x)  # 线性变换由768维到128维
            x = self.relu(x)  # 线性整流，激活下一层
            x = self.fc2(x)  # 线性变换由128维到64维
            x = self.relu(x)  # 线性整流，激活下一层
            x = self.fc3(x)  # 线性变换由64维到1维
            return x

        # 替换模型的分类器层

    model.classifier = RegressionModel()


# 将模型设置为评估模式
model.eval()

# 代表性的启动子序列
promoter_sequences = [
    "ATGCGTAGCTAGCTAGCTAGCTGACTGATGCGTAGCTAGCTAGCTAGCTGACTGATGCGTAGCTAGCTAGCTAGCTGACTG",
    "ATGCGTAGCTAGCTAGCTAGCTGACTGATGCGTAGCTAGCTAGCTAGCTGACTGATGCGTAGCTAGCTAGCTAGCTGACTG"
]

# 将序列tokenize
inputs = tokenizer(
                promoter_sequences,
                padding=True,
                truncation=True,
                max_length=110,
                return_tensors="pt",
            )

print("Input sequence length:", inputs['input_ids'].shape[1])  # 打印输入序列的长度
decoded_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
sequences = []
for i in decoded_tokens:
    seq = i
    sequences.append([i.split('#')[-1] for i in seq[1:-1]])
# print(sequences)
# exit()
# 禁用梯度计算
with torch.no_grad():
    # 模型前向传播，返回注意力权重
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # 这是一个包含所有层attention的tuple

# 选择最后一层的注意力权重
last_layer_attentions = attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)
# 遍历每一层的注意力权重
for i, attention in enumerate(attentions):
    print(f"Layer {i+1}: attention shape = {attention.shape}")

import matplotlib.pyplot as plt
import seaborn as sns

# 假设你只关注第一个头部的注意力
head_idx = 0

# 遍历每个序列
for i, sequence in enumerate(promoter_sequences):
    attention = last_layer_attentions[i][head_idx].detach().numpy()  # 获取第i个序列的注意力
    # 可视化Attention Map
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        attention, 
        xticklabels=list(sequences[i]), 
        yticklabels=list(sequences[i]), 
        cmap="hot_r"
    )
    plt.title(f"Attention Map for Promoter Sequence {i+1}")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)

    plt.savefig(f"/home/gy237/project/tubian_atten.png", dpi=1000)

    # 进一步分析哪些位置的注意力值较大，可以帮助确定对表达强度影响较大的部分