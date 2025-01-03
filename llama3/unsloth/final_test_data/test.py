import openai
import os
import datetime
from openai import OpenAI
import argparse
import pandas as pd
import json
import re

def test_diagnose(each_diagnosis,true_diagnosis_array,prompt,client):            
    chat_return = client.chat.completions.create(model='gpt-4o',messages=[{"role": "system", "content": "pysician"},
                                                                     {"role":"user", 
                                                                      "content":f"{prompt} \n"\
                                                                      f"Predict Diagnosis: str({each_diagnosis})\n"\
                                                                      f"True Differential Diagnosis: {true_diagnosis_array}",
                                                                                                                                      }])
    result=chat_return.choices[0].message.content
    return result

def acc_top_n(predict_diagnosis,true_diagnosis):
    #change for your environment    
    #api_key=os.environ.get("OPENAI_API_KEY")
    client = OpenAI()

    Prompt="""
    As an experienced pysician, your task is to identify whether the provided predicted diagnosis is in the true differential diagnosis. 
    Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no. 
    """
    result = []
    if len(predict_diagnosis) != len(true_diagnosis):
        print("Number of predicted and true samples not match!")
        return
    #if one predict sample is correct previously, directly skip for larger N.
    skip_index=[0 for i in range(len(predict_diagnosis))]

    for N in range(10):
        for i in range(len(predict_diagnosis)):
            #if number of predict diagnosis is less than 10, skip if index exceeds
            if skip_index[i]==0:
                try:
                    each_diagnosis=predict_diagnosis[i][N]
                    if each_diagnosis == "Diagnose:" or each_diagnosis == "" or each_diagnosis == "differential diagnosis:":
                        each_diagnosis = each_diagnosis+str(i) 
                        #print(each_diagnosis)
                except Exception:
                    continue
                
                #print(each_diagnosis) 
                    
                true_diagnosis_array=true_diagnosis[i]

                #print(true_diagnosis_array)
                acc=test_diagnose(each_diagnosis,true_diagnosis_array,Prompt,client)
                while acc!='Y' and acc!='N':
                    print('GPT 4 OUTPUT WRONG RESPONCE! Trying again!')
                    print(acc)
                    acc=test_diagnose(each_diagnosis,true_diagnosis_array,Prompt,client)
                if acc=='Y':
                    skip_index[i]=1
            else:
                continue
        _result = 'TOP '+ str(N+1) + " ACC: " + str(sum(skip_index)/len(skip_index))
        result.append(_result)
        print(_result)
    return result


def read_and_process_jsonl(file_path):
    logit_0_lists = []
    truth_values = []
    with open(file_path, 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    for data in input_data:
        split_list = data['truth'].split(';')
        split_list = [item.strip() for item in split_list if item.strip()]
        truth_values.append(split_list)
        logit_0_content = data["logit_0"].split('Differential diagnosis:')[-1]
        #processed_list = [entry.strip() for entry in logit_0_content.split(',') if entry.strip()]

        if re.search(r'\d+\.', logit_0_content):
            pattern = r'(\d+)\.\s*([^,.\d]+(?:\s+(?!\d+\.)[^,.\d]+)*)'
            diagnosis_list = re.findall(pattern, logit_0_content)
            diagnosis_list = diagnosis_list[-10:]

            processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

        # elif '#' in logit_0_content:
        #     processed_list = [entry.strip() for entry in logit_0_content.split('#') if entry.strip()]
        # else:
        #     processed_list = [entry.strip() for entry in re.split(r',|\n+', data['logit_0']) if entry.strip()]
        #     if 10 < len(processed_list):
        #         processed_list = processed_list[:10]

        logit_0_lists.append(processed_list)
    return logit_0_lists, truth_values


def test(path):
    predict_list, true_list = read_and_process_jsonl(path)
    # print(predict_list[:2])
    # print(true_list[:2])
    # exit()
    result = acc_top_n(predict_list,true_list)

    return result

# test('/home/gy237/project/llama3/unsloth/final_test_data/final_test_Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt102.json')