# {"text": "<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
import json
from datasets import load_dataset
from datasets import DatasetDict

# 加载数据
with open(f'/home/gy237/project/llama3/new_data/combined_data/scroe_5.json', 'r', encoding='utf-8') as file:
    train_data = json.load(file)
train_file = '/home/gy237/project/llama3/unsloth/filtered_data/traindata.json'

for i in train_data:
    # print(i)
    # exit()
    question = i['query']
    answer = i['answer']
    
    qa = {"text": f"<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
    with open(f'{train_file}', 'a', encoding='utf-8') as f:
        f.write(json.dumps(qa, ensure_ascii=False) + '\n')

print(f"数据已写入到 {train_file}")



# "<s>Human: "+问题+"\n</s><s>Assistant: "+答案+"\n"</s>

test = load_dataset("YBXL/NEJM_Reasoning_Subset_test", cache_dir='/home/gy237/project/download_data')
test_file = '/home/gy237/project/llama3/unsloth/filtered_data/testdata.json'
# print(test)
test_dataset = test["test"]
for i in range(len(test_dataset)):
    question = test["test"]['query'][i]
    answer = test["test"]['answer'][i]
    
    qa = {"text": f"<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
    with open(f'{test_file}', 'a', encoding='utf-8') as f:
        f.write(json.dumps(qa, ensure_ascii=False) + '\n')

print(f"数据已写入到 {test_file}")