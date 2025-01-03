import openai
import os
import datetime
from openai import OpenAI
import argparse
import pandas as pd
import json
import re
from tqdm import tqdm

def test_diagnose(each_diagnosis,true_diagnosis_array,prompt,client):
    chat_return = client.chat.completions.create(model='gpt-4o',temperature=0.0, messages=[{"role":"user",
                                                                      "content":f"{prompt}\nPredict Differential Diagnosis: str({each_diagnosis})\nTrue Diagnosis: {true_diagnosis_array}",
                                                                      }])
    result=chat_return.choices[0].message.content
    return result

def acc_top_n(predict_diagnosis,true_diagnosis):
    client = OpenAI()

    Prompt="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words.
    """

    if len(predict_diagnosis) != len(true_diagnosis):
        print("Number of predicted and true samples not match!")
        return
    #if one predict sample is correct previously, directly skip for larger N.
    skip_index=[0 for i in range(len(predict_diagnosis))]

    for N in range(10):
        for i in tqdm(range(len(predict_diagnosis))):
            #if number of predict diagnosis is less than 10, skip if index exceeds
            if skip_index[i]==0:
                try:
                    each_diagnosis=predict_diagnosis[i][N]
                    if each_diagnosis == "Diagnose:" or each_diagnosis == "" or each_diagnosis == "differential diagnosis:":
                        each_diagnosis = each_diagnosis+str(i) 
                except Exception:
                    continue
                
                    
                true_diagnosis_array=true_diagnosis[i]

                acc=test_diagnose(each_diagnosis,true_diagnosis_array,Prompt,client)
                while acc!='Y' and acc!='N':
                    print('GPT 4 OUTPUT WRONG RESPONCE! Trying again!')
                    print(acc)
                    acc=test_diagnose(each_diagnosis,true_diagnosis_array,Prompt,client)
                if acc=='Y':
                    skip_index[i]=1
            else:
                continue

        print('TOP '+str(N+1)+ " ACC: " + str(sum(skip_index)/len(skip_index)))  
    return 

def read_and_process_jsonl(file_path):
    logit_0_lists = []
    truth_values = []
    input_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            input_data.append(json.loads(line.strip()))

    for data in input_data:
        try:
            logit_0_content = data["resp"].split("Reranked")[1]
        except:
            continue
        split_list = data['truth']
        truth_values.append(split_list)
        
        pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
        diagnosis_list = re.findall(pattern, logit_0_content)
        if diagnosis_list:
            processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]
        else:
            pattern_comma_separated = r'([^,]+(?:, [^,\n]+)*)'
            diagnosis_list = re.findall(pattern_comma_separated, logit_0_content)
            if diagnosis_list:
                processed_list = [diag.strip() for diag in diagnosis_list[0].split(',')]
            else:
                processed_list = data["resp"]
        if 10 < len(processed_list):
            processed_list = processed_list[:10]
        logit_0_lists.append(processed_list)
    return logit_0_lists, truth_values

# predict_list, true_list = read_and_process_jsonl("/blue/yonghui.wu/qx68.yale/lm-evaluation-harness/local/results/YBXL__MeLLaMA-70B-chat/samples_NEJMSubset_2024-07-01T14-55-09.832141.jsonl")#("/blue/yonghui.wu/qx68.yale/lm-evaluation-harness/local/results/meta-llama__Llama-2-70b-chat-hf/samples_NEJMSubset_2024-06-26T16-52-41.422846.jsonl")
predict_list, true_list = read_and_process_jsonl("/home/gy237/project/llama3/total_final_test/magic_result.jsonl")
acc_top_n(predict_list,true_list)  