import openai
from openai import OpenAI
import json
from datasets import load_dataset
from tqdm import tqdm, trange
import re
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool


# openai API 
def generator(prompt, input_, model):
    # return 'Y'

    # client = OpenAI()
    client = OpenAI()
    if 'o1' in model:
        chat_return = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt + '\n### INPUT:\n' + input_
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
    # return 'Yes'

    predicts = output
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, predicts)
    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

    if len(processed_list) < 10:
        return 'No'
    else:
        return 'Yes'



name = 'JAMA_FINAL_test_gender_transferred'

# generate diagnosis
model = 'gpt-4-turbo'
# model = 'gpt-3.5-turbo'
# model = 'gpt-4o'
# model = 'o1-preview'

# 加载数据
data = load_dataset(f"YBXL/{name}", cache_dir='/home/gy237/project/download_data')
# print(data)
# print(data['train']['query'][0])
# exit()

df = pd.DataFrame(data['train'])

id_list = []
prompt_list = []
question_list = []

for index, i in df.iterrows():
    id_list.append(index)
    prompt_list.append(i['query'].split('\n### INPUT:\n')[0])
    question_list.append(i['query'].split('\n### INPUT:\n')[1])
    # print(i['query'])
    # print('<<'*50)
    # print(i['before_gender_tansferred_query'])
    # exit()

# check the prompt
for i in range(len(prompt_list) - 1):
    if prompt_list[i] != prompt_list[i+1]:
        print('Error')


def process_chunk(index_split, questions_split, prompt_split):
    results = []
    for i in trange(len(index_split), ncols=100):     #数量是len(index)/10，相当于一整个子进程，trange监视进度
        flag = True
        count = 0
        while flag:
            result = generator(prompt_split[i], questions_split[i], model)
            yn = test_format(result)
            if yn == 'Yes':
                flag = False
            elif count > 4:
                flag = False
                print('Error format!')
            else:
                count += 1

        results.append((index_split[i], result))
    return results
    

# 进行分数据
num_tasks = 20

index_split = np.array_split(id_list, num_tasks)
questions_split = np.array_split(question_list, num_tasks)
prompt_split = np.array_split(prompt_list, num_tasks)

with Pool(num_tasks) as pool:
    results = pool.starmap(process_chunk, zip(index_split, questions_split, prompt_split))

for chunk in results:
    for i in chunk:
        df.loc[i[0], 'predict'] = i[1]

for i, row in df.iterrows():
    entry = {'id': row['id'], 'input': row['query'], 'true': row['answer'], "predict": row['predict'], 'topic': row['topic']}

    json_path = f'/home/gy237/project/llama3/total_final_test/change_gender_12_19/GPT_result_12_20/GPT_generate_diagnoses/{name}_{model}.jsonl'
    with open(json_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
