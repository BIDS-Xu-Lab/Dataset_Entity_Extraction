from datasets import load_dataset
from tqdm import tqdm, trange
from datasets import Dataset, DatasetDict
import json
from openai import OpenAI
import re


ds1 = load_dataset("YBXL/JAMA_Reasoning_Common_train", cache_dir='/home/gy237/project/download_data')
ds2 = load_dataset("YBXL/JAMA_Reasoning_Rare_train", cache_dir='/home/gy237/project/download_data')

print(ds1)
print(ds2)
# print(ds1['train'][0])
print(len(ds1['train'][0]['conversations']))


id_ = [i['id'] for i in ds1['train']] + [i['id'] for i in ds2['train']]
inpt = [i['conversations'][0]['content'] for i in ds1['train']] + [i['conversations'][0]['content'] for i in ds2['train']]
final_diagnosis = [i['conversations'][1]['content'] for i in ds1['train']] + [i['conversations'][1]['content'] for i in ds2['train']]
topic = [i['topic'] for i in ds1['train']] + [i['topic'] for i in ds2['train']]

test_dic = [{'id': id_[i], 'input': inpt[i], 'final_diagnosis': final_diagnosis[i], 'topic': topic[i]} for i in range(len(id_))]
with open(f'/home/gy237/project/llama3/new_data/JAMA_Reasoning/jama.json', 'w', encoding='utf-8') as file:
    json.dump(test_dic, file, ensure_ascii=False, indent=4)

# inpt = [i.split('INPUT:')[-1].split('What Is Your Diagnosis?')[0] for i in inpt]
# final_diagnosis = [i.split('Diagnosis:')[-1].split('Discussion:')[0] for i in final_diagnosis]

dic = []
for i in range(len(id_)):
    if 'Diagnosis: ' in final_diagnosis[i]:
        inpt[i] = inpt[i].replace('\n    OUTPUT:', '\n\n    OUTPUT:')
        inpt_ = inpt[i].split('INPUT: ')[-1].split('\n\n')[0].strip()
        assert len(inpt[i].split('INPUT:')) == 2  and len(inpt[i].split('INPUT: ')[-1].split('\n\n')) in [2,3], str(i)+id_[i]+inpt[i]
        
        final = final_diagnosis[i].split('Diagnosis: ')[-1].split('\n')[0]
        assert len(final) > 2

        pattern = r'[ABCDE]\. (.*?)(?:\n|$)'
        matches = re.findall(pattern, final)
        if len(matches) == 1:
            final = matches[0]
        
        _inpt_ = f'''Develop a 10-Step differential diagnosis, for each step, give only one differential diagnosis with only the disease name at the current step, do not provide repeated hypotheses or diagnoses that appear in previous steps. Rerank all 10 differential diagnoses using all patient information and test results. Let’s think step by step.
### INPUT:
{inpt_}
Final diagnosis: {final}
### OUTPUT:'''

        dic.append({'id': id_[i], 'input':_inpt_, 'topic': topic[i]}, )

with open(f'/home/gy237/project/llama3/new_data/JAMA_Reasoning/jama_filtered_data.json', 'w', encoding='utf-8') as file:
    json.dump(dic, file, ensure_ascii=False, indent=4)
print(len(dic))

# JAMA_Reasoning_Common4
# 