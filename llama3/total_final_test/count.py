import json
from datasets import load_dataset
import pandas as pd
import os
import ast
import csv
# YBXL/JAMA_FINAL_test
# YBXL/NEJM_Reasoning_Final_JM_PROMPT_test

data = load_dataset(f"YBXL/NEJM_summary_302", cache_dir='/home/gy237/project/download_data')
data = data["train"]
# topic = [ast.literal_eval(i) for i in data['topic']]
topic = data['topic']
assert len(data) == len(topic)

sumary = [j for i in topic for j in i]
sumary = list(set(sumary))

dic = {}
for i in sumary:
    dic[i] = 0
    for j in topic:
        if i in j:
            dic[i] += 1

with open('/home/gy237/project/llama3/total_final_test/NEJM_count.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行（可选）
    writer.writerow(['Topic', 'number'])
    # 将字典的每个键值对写入CSV文件的两列
    for key, value in dic.items():
        writer.writerow([key, str(value)])

print("字典已成功保存到 CSV 文件")
# print(dic)
