import os
import json


folder_path = '/home/gy237/project/llama3/new_data/final_data'
file_names = os.listdir(folder_path)
out_names = [i.split('.')[0] for i in file_names if i.endswith('json')]
print(len(out_names))

priority = ['MedQA_Reasoning_train', 'JAMA_Reasoning_Rare_train', 'JAMA_Reasoning_Common_train']

out_names = [i for i in out_names if i not in priority]
total = 0
s5 = 0
s4 = 0
odd = 0
for i in out_names:
    score_dic = {}

    output_file = f"/home/gy237/project/llama3/new_data/final_data/{i}.json"
    with open(output_file, 'r', encoding='utf-8') as file:
        output = json.load(file)
    
    for j in output:
        score = j['score']
        if score in score_dic.keys():
            score_dic[score] += 1
        else:
            score_dic[score] = 1
    print(i)

    for k,v in score_dic.items():
        if len(k) > 1:
            odd += 1

    s5 += score_dic['5']
    s4 += score_dic['4']
    total += len(output)
    
    score_dic = {k:v for k,v in score_dic.items() if k=='5' or k=='4'}

    if len(score) >7:
        print(score_dic[:3])
    else:
        print(score_dic)
print(f'total: {total}')
print(f'5: {s5}')
print(f'4: {s4}')
print(f'odd: {odd}')
# folder_path = '/home/gy237/project/llama3/new_data/final_filter'
# file_names = os.listdir(folder_path)
# target_names = [i.split('.')[0] for i in file_names if i.endswith('json')]
# print(len(target_names))


# for i in target_names:
#     for j in out_names:
#         if i == j:
#             target_file = f"/home/gy237/project/llama3/new_data/final_filter/{i}.json"
#             output_file = f"/home/gy237/project/llama3/new_data/final_data/{j}.json"
    
#             data = []
#             with open(target_file,'r', encoding='utf-8') as file:
#                 target = json.load(file)
#             with open(output_file, 'r', encoding='utf-8') as file:
#                 output = json.load(file)
#             if len(target) != len(output):
#                 print(i)
#                 print(len(target))
#                 print(len(output))


# MultiCaRe_Reasoning_test_diagnosis, PMC_Patients_diagnosis, 这两个4, 5分一起生成explation

# 5分，4分一部分
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