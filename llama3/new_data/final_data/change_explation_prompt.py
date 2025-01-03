import os
import json
from datasets import Dataset, DatasetDict, load_dataset


name = 'NEJM_Reasoning_Final_NEW_PROMPT_test_refined'
# 加载数据
data = load_dataset(f"YBXL/{name}", cache_dir='/home/gy237/project/download_data')
input_ = data["train"]['query']
prompt = input_[0].split('INPUT:')[0]
print(prompt)


name_list = []
def upload(data, name, score):
    upload_data = []
    for i in data:
        id_ = i['id']
        inpt = i['query'].split('INPUT:')
        assert len(inpt) == 2
        query = 'INPUT:'.join([prompt, inpt[1]])
        conversations = [ { "role": "user", "content": query}, {"role": "assistant", "content": f"{i['answer']}"} ]
        upload_data.append({'id': id_, 'conversations': conversations, 'text': query})

    train_dataset = Dataset.from_list(upload_data)
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "valid":train_dataset,
        "test": train_dataset
    })
    # print(dataset_dict['train'][0])
    # exit()
    name_list.append(f"YBXL/{name}_score{score}")
    dataset_dict.push_to_hub(f"YBXL/{name}_score{score}")



folder_path = '/home/gy237/project/llama3/new_data/final_data'
file_names = os.listdir(folder_path)
out_names = [i.split('.')[0] for i in file_names if i.endswith('explanation.json')]
print(out_names)


for i in out_names:
    data_4 = []
    data_5 = []
    output_file = f"{folder_path}/{i}.json"
    with open(output_file, 'r', encoding='utf-8') as file:
        output = json.load(file)
    for j in output:
        if j['score']=='4':
            data_4.append(j)
            j.pop('score')
        elif j['score']=='5':
            data_5.append(j)
            j.pop('score')
    upload(data_4, i, '4')
    upload(data_5, i, '5')
            
print(name_list)