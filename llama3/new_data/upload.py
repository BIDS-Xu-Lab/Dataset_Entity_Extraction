from datasets import Dataset, DatasetDict
import json
from datasets import load_dataset
from tqdm import tqdm, trange
import json


with open(f'/home/gy237/project/llama3/new_data/combined_data/scroe_4_5_explanation.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(len(data))

upload_data = []
for i in data:
    id_ = i['id']
    conversations = [ { "role": "user", "content": f"{i['query']}"}, {"role": "assistant", "content": f"{i['answer']}"} ]
    text = i['query']
    upload_data.append({'id': id_, 'conversations': conversations, 'text': text})

print(len(upload_data))
# print(upload_data[0])
# exit()
train_dataset = Dataset.from_list(upload_data)
dataset_dict = DatasetDict({
    "train": train_dataset,
    "valid":train_dataset,
    "test": train_dataset
})
# print(dataset_dict['train'][0])
# exit()
dataset_dict.push_to_hub("YBXL/17dataset_mixed_scroe_4_5_explanation")
# 	
# [ { "role": "user", "content": "" }, { "role": "assistant", "content": "" } ]
# /home/gy237/project/llama3/new_data/NEJM_Reasoning_Final_NEW_PROMPT_test.json


# with open(f'/home/gy237/project/llama3/new_data/NEJM_Reasoning_Final_NEW_PROMPT_test.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
# print(len(data))
# train_dataset = Dataset.from_list(data)
# dataset_dict = DatasetDict({
#     "train": train_dataset,
#     "valid":train_dataset,
#     "test": train_dataset
# })
# # print(dataset_dict['train'][0])
# # exit()
# dataset_dict.push_to_hub("YBXL/NEJM_Reasoning_Final_Old_PROMPT_test_refined")

# ds = load_dataset("YBXL/NEJM_Reasoning_Final_NEW_PROMPT_test_refined", cache_dir='/home/gy237/project/llama3/new_data/DDXPlus_Reasoning_train')
# data = ds['train']

# ds = load_dataset("YBXL/NEJM_Reasoning_Final_test", cache_dir='/home/gy237/project/llama3/new_data/DDXPlus_Reasoning_train')
# prompt = ds['train']['query'][0].split('INPUT:')[0]

# # print(prompt)
# # exit()

# new_data = []
# for i in data:
#     inpt = i['query'].split('INPUT:')[1]
#     query = 'INPUT:'.join([prompt, inpt])
#     new_data.append({'id': i['id'], 'query': query, 'answer': i['answer']})
#     # print(new_data)
#     # exit()

# train_dataset = Dataset.from_list(new_data)
# dataset_dict = DatasetDict({
#     "train": train_dataset,
#     "valid":train_dataset,
#     "test": train_dataset
# })
# # print(dataset_dict['train'][0])
# # exit()
# dataset_dict.push_to_hub("YBXL/NEJM_Reasoning_Final_Old_PROMPT_test_refined")
