# {"text": "<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
import json
from datasets import load_dataset
from datasets import DatasetDict

# 加载数据
train = load_dataset("YBXL/diagnosis_train")
train_file = '/home/gy237/project/llama3/unsloth/data/traindata.json'

# 获取train数据集
train_dataset = train["train"]["conversations"]
# print(train_dataset["conversations"][:1])
for i in range(len(train_dataset)):
    train_list = train_dataset[i]
    question = train_list[0]['content'].replace('\n', ' ')
    answer = train_list[1]['content'].replace('\n', ' ')
    
    qa = {"text": f"<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
    with open(f'{train_file}', 'a', encoding='utf-8') as f:
        f.write(json.dumps(qa, ensure_ascii=False) + '\n')

print(f"数据已写入到 {train_file}")



# "<s>Human: "+问题+"\n</s><s>Assistant: "+答案+"\n"</s>

test = load_dataset("YBXL/NEJM_Reasoning_Subset_test")
test_file = '/home/gy237/project/llama3/unsloth/data/testdata.json'
# print(test)
test_dataset = test["test"]
for i in range(len(test_dataset)):
    question = test["test"]['query'][i]
    answer = test["test"]['answer'][i]
    
    qa = {"text": f"<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
    with open(f'{test_file}', 'a', encoding='utf-8') as f:
        f.write(json.dumps(qa, ensure_ascii=False) + '\n')

print(f"数据已写入到 {test_file}")