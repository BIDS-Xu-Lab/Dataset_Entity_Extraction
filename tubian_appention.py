import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# with open("/home/s211870092/igem_bert_qz_top1000_tubian100.txt", "r") as f:
#     raw_data = [i.rstrip("\n") for i in f.readlines()]
# raw_seqlist = [i.split("\t")[0] for i in raw_data]
# raw_labelist = [i.split("\t")[1][1:][:-1].split(", ") for i in raw_data]
# x = [i for i in range(len(raw_labelist[1]))]

# label_mean = []
# for i in range(len(raw_labelist)):
#     raw_labelist[i] = [float(j) for j in raw_labelist[i]]
#     label_mean += [sum(raw_labelist[i])]
# _label_men = sorted(label_mean.copy())  # 1000

# high_index = []
# low_index = []
# mean_index = []
# for i in range(len(label_mean)):
#     if label_mean[i] in _label_men[-2:]:
#         high_index += [i]
#     if label_mean[i] in _label_men[:2]:
#         low_index += [i]
#     if label_mean[i] == _label_men[500]:
#         mean_index += [i]


import torch
import torch.nn as nn
import numpy as np
import pdb
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import argparse
import numpy as np
import random
import shutil
from tqdm import tqdm
import torch
import numpy as np
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    pipeline,
)
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 环境创建一定要按照以下顺序安装安装，不要装错顺序了，后面的就是缺什么就装什么
# conda create --name dnaBERT python=3.8
# pip install transformers==4.28.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
## CUDA 11.1  pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
from torch.utils.data.distributed import DistributedSampler
import datetime
import pandas as pd
iii = 0

os.environ["CUDA_VISIBLE_DEVICES"] = f"{iii}"  #'0,1,2,3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchamark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 0
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  ## 固定随机种子

bs = 500
km = 4
# 20%d, 128, magic
ll = [
    "/home/s211870092/biobert/biobert_k4_lr1e-05_10%, Adam, 去整+1/biobert_weights_k4_lr1e-05_10.pth",
    "igem_bert_k4_lr0.005_100%-2/igem_bert_weights_k4_lr0.005_25.pth",
    "igem_bert_k4_lr1e-06-20%, Adam, 128, 去整/igem_bert_weights_k4_lr1e-06_6.pth",
    "igem_bert_k4_lr1e-06-10%, Adam, 128 去整=1/igem_bert_weights_k4_lr1e-06_10.pth",
]
ls = ["30,000,000", "6,000,000", "3,000,000"]

# for i in range(len(ls)):
if True:
    ii = iii
    tokenizer = AutoTokenizer.from_pretrained(
        f"aitslab/biobert_huner_gene_v1", trust_remote_code=True, local_files_only=True
    )  # kmer

    # zhihan1996/DNA_bert_{km}  aitslab/biobert_huner_gene_v1
    # AutoModelForSequenceClassification, AutoModel, num_labels=1,output_attentions =False,output_hidden_states =False,
    # 定义回归模型
    model = AutoModelForSequenceClassification.from_pretrained(
        f"aitslab/biobert_huner_gene_v1",
        output_attentions=True,
        trust_remote_code=True,
        local_files_only=True
    )  # kmer

    # model.requires_grad_(True)       #冻结参数,False
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
    model = model.to(device)
    model.load_state_dict(torch.load(ll[ii]))
    per = ls[ii]  # 30,000,000
    print(per)


    # 定义训练集和测试集
    def sentence2word(str_set, kmer_length):
        """

        Arguments
        ---------
        str_set:序列列表
        kmer_length:kmer的长度

        Returns
        -------
        序列对应的kmer列表的集合
        """

        word_seq = []
        for sr in str_set:
            tmp = []
            for i in range(len(sr) - kmer_length + 1):
                if "N" in sr[i : i + kmer_length]:
                    tmp.append("null")
                else:
                    tmp.append(sr[i : i + kmer_length])
            word_seq.append(" ".join(tmp))
        return word_seq

    class MyData(Dataset):
        def __init__(self, seq, labellist, tokenizers):
            seqlist = sentence2word(seq, km)  # kmer
            # 下面的max_length需要根据最大长度进行修改(序列最大长度127，所以设成128), 96
            self.encodings = tokenizers(
                seqlist,
                padding=True,
                truncation=True,
                max_length=110,
                return_tensors="pt",
            )
            self.label = labellist

        def __getitem__(self, index):
            item = {
                key: torch.tensor(val[index]) for key, val in self.encodings.items()
            }
            # item['labels'] = torch.tensor(self.label[index])

            return item, self.label[index]

        def __len__(self):
            return len(self.label)


## 使用BERT模型生成输入序列的attention map
# from transformers import BertModel, BertTokenizer
# import torch

# # 加载预训练的BERT模型和tokenizer
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name, output_attentions=True)

h_seq = []
m_seq = []
l_seq = []
for i in high_index:
    h_seq.append(raw_seqlist[i])
for i in low_index:
    m_seq.append(raw_seqlist[i])
    # print(raw_seqlist[i])
    # print(raw_labelist[i])
    # exit()
for i in mean_index:
    l_seq.append(raw_seqlist[i])


import matplotlib.pyplot as plt
import seaborn as sns

count = 0
h_seq_ = ['TGCATTTTTTTCACATCCTACCGCCTGGACTGCTCAGTATCCTCGAGAATTGGCGCGAAGGGAGCTACAGGTCCTACCATGCTAGGCTTCCGCTTGGGGTTACGGCTGTT',
          'TGCATTTTTTTCACATCAGCAGCTTGTCTTACTCACGAAGTTATCGCATGTGGTGCCCACCGCGGGACTGTATCTAGGTGGCATCTTACCACACGGGTTACGGCTGTT']
for text in h_seq:
    print(text)
    count += 1
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # outputs包含last_hidden_state, pooler_output, attentions
    # 提取所有层的attention map
    attentions = outputs.attentions

    # 获取最后一层的第一个头的注意力权重
    last_layer_attention = attentions[-1][0, 0].detach().cpu().numpy()
    print(last_layer_attention)
    np.save(f"/home/s211870092/tubian/tubian_atten_{count}", last_layer_attention)
        
    # 可视化注意力权重
    sns.heatmap(last_layer_attention, cmap="viridis")
    plt.title("Attention Map")
    plt.xlabel("Input Tokens Key")
    plt.ylabel("Input Tokens Query")
    plt.savefig(f"/home/s211870092/tubian/tubian_atten_{count}.png", dpi=1000)
