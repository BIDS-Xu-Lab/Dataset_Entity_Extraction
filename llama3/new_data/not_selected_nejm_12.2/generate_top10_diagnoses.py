from datasets import load_dataset
from tqdm import tqdm, trange
from datasets import Dataset, DatasetDict
import json
from openai import OpenAI
import re
import pandas as pd
import numpy as np
from multiprocessing import Pool

# openai API 
def generator(prompt):
    # client = OpenAI(api_key="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")
    # return 'Y'

    client = OpenAI()
    chat_return = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    
    result = chat_return.choices[0].message.content
    return result


def test_format(output):
    """
    test o1's generation's format
    """
    # return 'Yes'
    
    predicts = output
    predicts = predicts.replace('\n\n','\n')
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, predicts)
    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

    if len(processed_list) < 10:
        return 'No'
    else:
        return 'Yes'


def process_chunk(index_split, question_split):
    out = []
    for i in trange(len(index_split), ncols=100):
        inpt = question_split[i]
        assert len(inpt.split("\n### INPUT:\n")) == 2

        flag = True
        count = 0
        while flag:
            result = generator(inpt)
            yn = test_format(result)
            if yn == 'Yes':
                flag = False
            elif count > 2:
                flag = False
                print('Error format!', i[0])
            else:
                count += 1

        out.append({'id': index_split[i], 'input': question_split[i], 'output': result})
    return out


num_tasks = 30
data = pd.read_csv('/home/gy237/project/llama3/new_data/not_selected_nejm_12.2/not_selected_nejm_refined.tsv', sep='\t')

prompt = '''Based on the Final diagnosis, develop a 10-Step differential diagnosis, for each step, give only one differential diagnosis with only the disease name at the current step, do not provide repeated hypotheses or diagnoses that appear in previous steps. Rerank all 10 differential diagnoses using all patient information and test results. Let's think step by step.\n### INPUT:\n'''

index = data.index.tolist()
question = [prompt + data.loc[i, 'refined_case_report'] + '\nFinal diagnosis: ' + data.loc[i, 'final_diagnosis_comb'] + '\n### OUTPUT:' for i in index]

index_split = np.array_split(index, num_tasks)
question_split = np.array_split(question, num_tasks)

with Pool(num_tasks) as pool:
    results = pool.starmap(process_chunk, zip(index_split, question_split))

for chunk in results:
    for out in chunk:
        data.loc[out['id'], 'input'] = out['input']
        data.loc[out['id'], 'output'] = out['output']
        

data.to_csv("/home/gy237/project/llama3/new_data/not_selected_nejm_12.2/not_selected_nejm_generated_top10.tsv", sep='\t', index=False)
print('This batch is over!')