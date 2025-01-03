import json
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
import ast
from openai import OpenAI


def test_diagnose(inputcase,Prompt,client):
    #Prompt="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words.
    #"""
    chat_return = client.chat.completions.create(model='gpt-4o',temperature=0.0, messages=[{"role": "system", "content": "pysician"},
                                                                     {"role":"user",
                                                                      "content":f"{Prompt} \n"\
                                                                      f"Input case: {inputcase}\n",
                                                                      }])
    result=chat_return.choices[0].message.content
    return result
def get_age_gender(inputcase):
    b = '1'
    inputcase = inputcase.split('INPUT:\n')[1]
    #change for your environment
    
    client = OpenAI()
    Prompt="""Your task is to identify the age and gender of the patient in the presented case report. Please only return a list with two elements: [age, gender]. Age should be a number, if not presented in the case, return '-1' for age. Gender should be 'M' for male and 'F' for female. If not presented in the case, return 'U' for gender.
    Example: [70, 'M']
    INPUT:
    """
    flag=0
    while flag==0:
        result = test_diagnose(inputcase, Prompt, client)
        try:
            lis = eval(result)
            age = int(lis[0])
            gender = lis[1]
            if gender != 'F' and gender != 'M' and gender != 'U':
                c = b - 1
            flag = 1
        except Exception:
            print(result)
            flag=0
    return lis



path = '/home/gy237/project/llama3/total_final_test/Llama3.1_final_test'
flie_list = os.listdir(path)
flie_list = [i for i in flie_list if i.endswith('.csv')]


df = pd.read_csv('/home/gy237/project/llama3/total_final_test/JAMA_final_test/JAMA_FINAL_test_gpt-3.5-turbo_test.csv')

list_all = []
for i in df.case_report:
    list_all.append(get_age_gender(i))

age = []
gender = []
for i in list_all:
    age.append(i[0])
    gender.append(i[1])

print('Over')

for file_name in flie_list:
    print(file_name)
    file_name = path + '/' + file_name

    df = pd.read_csv(file_name)

    df['Age'] = age
    df['Gender'] = gender

    # 将更新后的数据保存回CSV文件，可以覆盖原文件或保存为新文件
    df.to_csv(file_name, index=False)
    
    # save_path = file_name.split('.csv')[0]
    # with open(f'{save_path}_result.json', 'w', encoding='utf-8') as file:
    #     json.dump(save_data, file, ensure_ascii=False, indent=4)