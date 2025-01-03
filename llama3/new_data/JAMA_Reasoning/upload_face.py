from datasets import Dataset, DatasetDict
import json
from datasets import load_dataset
from tqdm import tqdm, trange
import json


with open(f'/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate_refained.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = [json.loads(i.strip()) for i in lines]
print(len(data))

with open(f'/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate_error.json', 'r', encoding='utf-8') as file:
    error = json.load(file)
print(len(error))

error_id = [i['id'] for i in error]

prompt = '''Develop a 10-Step differential diagnosis, for each step, give only one differential diagnosis with only the disease name at the current step, do not provide repeated hypotheses or diagnoses that appear in previous steps. Rerank all 10 differential diagnoses using all patient information and test results. Let's think step by step.\n### INPUT:'''
upload = []
for i in data:
    if i['id'] not in error_id:
        assert len(i['input'].split('### INPUT:')) == 2 == len(i['input'].split('### INPUT:')[-1].split('\nFinal diagnosis:'))
        inpt = i['input'].split('### INPUT:')[-1].split('\nFinal diagnosis:')[0]
        inpt = prompt + inpt
        try:
            upload.append({'id': i['id'], 'query': inpt, 'answer': i['refined_output'], 'topic': i['topic']})
        except:
            upload.append({'id': i['id'], 'query': inpt, 'answer': i['output'], 'topic': i['topic']})
print(len(upload))


train_dataset = Dataset.from_list(upload)
t = Dataset.from_list(upload[:10])

dataset_dict = DatasetDict({
    "train": train_dataset,
    "valid":t,
    "test": t
})
print(dataset_dict['train'][0])
# exit()
dataset_dict.push_to_hub("YBXL/JAMA_Reasoning_o1refained")