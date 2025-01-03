import os
import json
import random
# MedQA_Reasoning_train, JAMA_Reasoning_Rare_train, JAMA_Reasoning_Common_train全用

folder_path = '/home/gy237/project/llama3/new_data/final_data'
file_names = os.listdir(folder_path)
out_names = [i.split('.')[0] for i in file_names if i.endswith('json')]
print(len(out_names))

priority = ['MedQA_Reasoning_train', 'JAMA_Reasoning_Rare_train', 'JAMA_Reasoning_Common_train']

out_names = [i for i in out_names if i not in priority]

# s5list = []
# s4list = []

# for i in priority:
#     output_file = f"{folder_path}/{i}.json"
#     with open(output_file, 'r', encoding='utf-8') as file:
#         output = json.load(file)
#     s5list.extend(output)
#     s4list.extend(output)

# for i in out_names:
#     output_file = f"{folder_path}/{i}.json"
#     with open(output_file, 'r', encoding='utf-8') as file:
#         output = json.load(file)
#     for j in output:
#         if j['score']=='5':
#             s5list.append(j)
#             s4list.append(j)
#         if j['score']=='4':
#             s4list.append(j)
# print(len(s5list))
# print(len(s4list))
# with open('/home/gy237/project/llama3/new_data/scroe_5.json', 'w', encoding='utf-8') as file:
#     json.dump(s5list, file, ensure_ascii=False, indent=4)
# with open('/home/gy237/project/llama3/new_data/scroe_4.json', 'w', encoding='utf-8') as file:
#     json.dump(s4list, file, ensure_ascii=False, indent=4)




s45list = []
s45list_dic = {}

for i in priority:
    x = len(s45list)
    output_file = f"{folder_path}/{i}.json"
    with open(output_file, 'r', encoding='utf-8') as file:
        output = json.load(file)
    s45list.extend(output)
    
    y = len(s45list)
    s45list_dic[i] = y-x

_list = []
for i in out_names:
    x = len(s45list)
    output_file = f"{folder_path}/{i}.json"
    with open(output_file, 'r', encoding='utf-8') as file:
        output = json.load(file)
    for j in output:
        if j['score']=='5':
            s45list.append(j)
    if i=='medical_book_train':
        for j in output:
            if j['score']=='4':
                _list.append(j)
        # 设置随机种子
        random.seed(42)
        # 随机抽取20k的数据
        sampled_data = random.sample(_list, 20000)
        s45list.extend(sampled_data)
    
    y = len(s45list)
    s45list_dic[i] = y-x

# four_names = ['MultiCaRe_Reasoning_test_diagnosis', 'PMC_Patients_diagnosis', 'GENE_REVIEW_SY_train', 'GI_Reasoning_train', 'liveqa_train']
four_names = ['MultiCaRe_Reasoning_test_diagnosis_explanation', 'PMC_Patients_diagnosis_explanation', 'GENE_REVIEW_SY_train', 'GI_Reasoning_train', 'liveqa_train']

# medical_book_train
for i in four_names:
    x = len(s45list)
    output_file = f"{folder_path}/{i}.json"
    with open(output_file, 'r', encoding='utf-8') as file:
        output = json.load(file)
    for j in output:
        if j['score']=='4':
            s45list.append(j)
    
    y = len(s45list)
    s45list_dic[i] += y-x

print(s45list_dic)
print(len(s45list))
print(sum([i for i in s45list_dic.values()]))
# print(s45list[-1])
# exit()

# with open('/home/gy237/project/llama3/new_data/combined_data/scroe_4_5_explanation.json', 'w', encoding='utf-8') as file:
#     json.dump(s45list, file, ensure_ascii=False, indent=4)

# MultiCaRe_Reasoning_test_diagnosis, PMC_Patients_diagnosis, 这两个4, 5分一起生成explation

# Alpaca_train 5
# medical_book_train 5+4(20k)
# iCliniq_train 5
# Dolly_train 5
# medical_meadow_mediqa_train 5
# liveqa_train 5+4
# GENE_REVIEW_SY_train 5+4
# GI_Reasoning_train 5+4
# HealthCareMagic_train 5
# MedInstruct_train 5
# DDXPlus_Reasoning_train 5
# GENE_OMIM_SY_train 5