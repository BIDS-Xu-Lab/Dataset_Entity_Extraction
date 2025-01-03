import openai
import os
import datetime
from openai import OpenAI
import argparse
import pandas as pd
import json
import re
from tqdm import tqdm
import csv

def test_diagnose(each_diagnosis,true_diagnosis):
    """
    # Determine whether the each_diagnosis and true_diagnosis is a same diagnosis
    # each_diagnosis: generated diagnosis, str
    # true_diagnosis: true diagnosis, str
    """
    # return 'Y'    # use, when you check the diagnoses list
    client = OpenAI()
    
    prompt ="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words."""
    
    chat_return = client.chat.completions.create(model='gpt-4o',temperature=0.0, messages=[{"role": "system", "content": "pysician"},
                                                                     {"role":"user",
                                                                      "content":f"{prompt} \n"\
                                                                      f"Predicted Diagnosis: {each_diagnosis}\n"\
                                                                      f"True Diagnosis: {true_diagnosis}",
                                                                      }])

    result = chat_return.choices[0].message.content
    return result


# parse the prediction
def segement_predict(predict):

    prediction = predict.replace('\n\n','\n')
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, prediction)
    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

    if len(processed_list) < 10:
        processed_list = prediction.split('\n')

    if len(processed_list) > 10:
            processed_list = processed_list[-10:]

    return processed_list



def main(file_name):
    save_path = file_name.split('.jsonl')[0]

    tltle = ['ID', 'case_report', 'gold', 'predict', 'topic', 'top1', 'top2', 'top3', 'top4', 'top5', 'top6', 'top7', 'top8', 'top9', 'top10']
    with open(f'{save_path}_test.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(tltle)

    data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data.append(item)

    print('-'*100)
    acc_dic = {}
    error_list = []
    for i in tqdm(data, ncols=100):
        diagnosis = i['true']
        predict = i['predict']

        processed_list = segement_predict(predict)
        if len(processed_list) < 10:
            error_list.append(i)

        i['acc'] = []
        for prediction in processed_list:
            if 1 in i['acc']:
                i['acc'].append(1)
            else:
                acc = test_diagnose(prediction, diagnosis)

                while acc!='Y' and acc!='N':
                    print('GPT 4 OUTPUT WRONG RESPONCE! Trying again!')
                    print(acc)
                    acc = test_diagnose(prediction, diagnosis)

                if acc == 'N':
                    i['acc'].append(0)
                elif acc == 'Y':
                    i['acc'].append(1)
                else:
                    print('Error')
        if len(i['acc']) >10:
            i['acc'] = i['acc'][-10:]
        elif len(i['acc']) < 10:
            i['acc'] = i['acc'] + [0]*(10-len(i['acc']))


        entry = [i['id'], i['input'], i['true'], i['predict'], i['topic']] + i['acc']
        with open(f'{save_path}_test.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(entry)


    print(f'Number of daignoses less than 10: {len(error_list)}')
    with open(f'{save_path}_error.json', 'w', encoding='utf-8') as file:
        json.dump(error_list, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    file_path = '/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/Llama370BInsJAMAreasoninginstr700003_JAMAFinalAll.jsonl'

    print('file_path:', file_path)
    main(file_path)