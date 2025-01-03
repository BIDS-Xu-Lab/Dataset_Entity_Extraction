from datasets import load_dataset
from tqdm import tqdm, trange
from datasets import Dataset, DatasetDict
import json
from openai import OpenAI
import re


# openai API 
def generator(prompt, model):
    # client = OpenAI(api_key="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")
    client = OpenAI()
    if model == 'o1-preview':
        chat_return = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    else:
        chat_return = client.chat.completions.create(model=model,messages=[{"role": "system", "content": prompt},
                                                                        {"role":"user", "content": input_}])
    
    result = chat_return.choices[0].message.content
    return result


def test_format(output):
    """
    test o1's generation's format
    """
    predicts = output
    predicts = predicts.replace('\n\n','\n')
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, predicts)
    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

    if len(processed_list) < 10:
        return 'No'
    else:
        return 'Yes'


ds1 = load_dataset("YBXL/JAMA_Reasoning_o1refained", cache_dir='/home/gy237/project/download_data')
data = ds1['train']

with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/name_list.txt', 'r') as f:
    exist = f.readlines()
exist = [i.strip() for i in exist]
print('Exist_Name_list:', len(exist))

data = [i for i in data if i['id'] not in exist]
print('Leftover:', len(data))

prompt = '''Based on the Final diagnosis, develop a 10-Step differential diagnosis, for each step, give only one differential diagnosis with only the disease name at the current step, do not provide repeated hypotheses or diagnoses that appear in previous steps. Rerank all 10 differential diagnoses using all patient information and test results. Let's think step by step.\n### INPUT:\n'''

total = 0
for i in tqdm(data, ncols=100):
    if i['id'] not in exist:
        inpt = i['input']
        assert len(inpt.split("\n### INPUT:\n")) == 2
        inpt = prompt + inpt.split("\n### INPUT:\n")[-1]

        flag = True
        count = 0
        while flag:
            result = generator(inpt, 'o1-preview')
            total += 1
            yn = test_format(result)
            if yn == 'Yes':
                flag = False
            elif count > 2:
                flag = False
                print('Error format!', i['id'])
            else:
                count += 1

        entry = {'id': i['id'], 'input': inpt, 'output': result, 'topic': i['topic']}
        # print(entry)
        # exit()

        with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate.jsonl', 'a', encoding='utf-8') as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/name_list.txt', 'a') as f:
            f.write(i['id'] + '\n')
        
print('This batch is over!')
print('Total usage:', total)