import openai
import os
import datetime
from openai import OpenAI
import argparse
import pandas as pd
import json
import re
from tqdm import tqdm

def test_diagnose(each_diagnosis,true_diagnosis):
    """
    # Determine whether the each_diagnosis and true_diagnosis is a same diagnosis
    # each_diagnosis: generated diagnosis, str
    # true_diagnosis: true diagnosis, str
    """
    # return 'Y', 'test'     # use, when you check the diagnoses list
    client = OpenAI()
    # prompt0="""
    # Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis.
    # Carefully review the information and determine the correctness of the prediction.
    # Please notice same diagnosis might be in different words.
    # Only return "Y" for yes or "N" for no, without any other words.
    # """
    # prompt1 ="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words.
    # """
    prompt2 ="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words."""
#     prompt3 = '''#Background#
# We now have two sets of data, one for real diagnoses and the other for diagnoses predicted by the AI model. We would like you to judge the accuracy of the model's predictions.
# #Role#
# You are an excellent and experienced Attending Physician who can accurately determine whether the two diagnoses are the same disease.
# #Task#
# Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis.
# #Constrains#
# 1. Carefully review the information and determine the correctness of the prediction.
# 2. Please notice same diagnosis might be in different words.
# 3. Some of the text given to you is an explanation of the diagnosis. You can use it as an aid to your judgment.
# 4. When the diagnosis predicted by the model is very close to the true diagnosis, but it is not the same diagnosis, please combine with the explanation of the diagnosis to make a comprehensive judgment
# #OutputFormat#
# Only return "Y" for yes or "N" for no, without any other words.'''
#     prompt4 = '''
# # Context
# You are an experienced and highly skilled Attending Physician tasked with evaluating the accuracy of diagnostic predictions made by an AI model. 
# # Objective
# Determine whether the AI model's predicted diagnosis matches the true diagnosis provided.
# # Instructions
# Fellow the instruction and use step-by-step Inference to make a judgement
# 1. **Thoroughly review** both the true diagnosis and the AI's predicted diagnosis.
# 2. **Consider synonymous and related terms**: Recognize that the same disease may be described using different terminology or phrasing.
# 3. **Utilize provided explanations**: Use your your medical expertise and any additional descriptive information accompanying AI's diagnosis to inform your judgment. 
# 4. **Assess closely related diagnoses**: If the predicted diagnosis is very similar to the true diagnosis, please relax the criteria a bit and response Y when the AI's diagnosis site and symptoms correspond to the true diagnosis.
# # Response Format
# Only response with Y or N
# - Y, if the AI's predicted diagnosis correctly matches the true diagnosis.
# - N, if the AI's predicted diagnosis does not match the true diagnosis.
# '''
#     prompt5 ="""Your task is to identify whether the provided predicted diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words."""
#     prompt6 ="""Your task is to identify whether the provided predicted diagnosis is correct based on the true diagnosis. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words."""
    # prompt = prompt6
    prompt = prompt2  # best
    chat_return = client.chat.completions.create(model='gpt-4o',temperature=0.0, messages=[{"role": "system", "content": "pysician"},
                                                                     {"role":"user",
                                                                      "content":f"{prompt} \n"\
                                                                      f"Predicted Diagnosis: {each_diagnosis}\n"\
                                                                      f"True Diagnosis: {true_diagnosis}",
                                                                      }])
    result=chat_return.choices[0].message.content
    return result, prompt


def acc_top_n(predict_diagnosis, true_diagnoses, acc_dic):
    less_than_10 = 0

    if len(predict_diagnosis) != len(true_diagnoses):
        print("Number of predicted and true samples not match!")
        return

    #if one predict sample is correct previously, directly skip for larger N.
    skip_index=[0 for i in range(len(predict_diagnosis))]
    
    for N in range(10):
        for i in range(len(predict_diagnosis)):
            if skip_index[i] == 0:
                #if number of predict diagnosis is less than 10, skip if index exceeds
                try:
                    each_diagnosis = predict_diagnosis[i][N]   #只看前十个
                    true_diagnosis = true_diagnoses[i]

                    acc, prompt = test_diagnose(each_diagnosis,true_diagnosis)

                    # keep the results of the evaluation obey the format
                    while acc!='Y' and acc!='N':
                        print('GPT 4 OUTPUT WRONG RESPONCE! Trying again!')
                        print(acc)
                        acc, prompt = test_diagnose(each_diagnosis,true_diagnosis)
                    if acc == 'Y':
                        skip_index[i] = 1

                except:
                    less_than_10 += 1

        print('TOP '+str(N+1)+ " ACC: " + str(sum(skip_index)/len(skip_index)))
        acc_dic[f'TOP {N+1} ACC'] = str(sum(skip_index)/len(skip_index))
    
    acc_dic['less_than_10_another'] = less_than_10
    acc_dic['prompt'] = prompt
    print(f'Number of daignoses less than 10: {acc_dic["less_than_10_another"]}')
    return acc_dic


# parse the prediction
def segement_predict(predicts, acc_dic, data):
    processed_lists = []
    error_list = []
    for pre in range(len(predicts)):
        prediction = predicts[pre].replace('\n\n','\n')
        pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
        diagnosis_list = re.findall(pattern, prediction)
        processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]


        if len(processed_list) > 10:
            processed_list = processed_list[-10:]
        
        if len(processed_list) < 10:
            error_list.append(data[pre])

        if len(processed_list) < 10:
            processed_list = prediction.split('\n')
            # print('-'*100)
            # print(id_[pre])
            

        processed_lists.append(processed_list)

    acc_dic['more_than_10'] = len([i for i in processed_lists if len(i) > 10])
    acc_dic['less_than_10'] = len([i for i in processed_lists if len(i) < 10])
    acc_dic['equal_than_10'] = len([i for i in processed_lists if len(i) == 10])
    
    return processed_lists, acc_dic, error_list



def fellowing_test(save_path, labels, predicts, data):
    acc_list = []
    # repeats = ['1','2','3','4','5']
    repeats = ['1']
    for repeat in repeats:
        print('-'*100)
        print(f'The {repeat} time')

        acc_dic = {}
        acc_dic['time'] = repeat

        processed_list, acc_dic, error_list = segement_predict(predicts, acc_dic, data)
        print(f'Number of daignoses more than 10: {acc_dic["more_than_10"]}')
        print(f'Number of daignoses less than 10: {acc_dic["less_than_10"]}')
        print(f'Number of daignoses equal than 10: {acc_dic["equal_than_10"]}')
        
        with open(f'{save_path}_error.json', 'w', encoding='utf-8') as file:
            json.dump(error_list, file, ensure_ascii=False, indent=4)


        acc_dic = acc_top_n(processed_list, labels, acc_dic)
        acc_list.append(acc_dic)

        with open(f'{save_path}_test.json', 'w', encoding='utf-8') as file:
            json.dump(acc_list, file, ensure_ascii=False, indent=4)


    acc_dic = {}
    acc_dic['time'] = 'mean'
    for i in range(10):
        acc_dic[f'TOP {i+1} ACC'] = str(sum([float(j[f'TOP {i+1} ACC']) for j in acc_list]) / len(repeats))
    
    print(acc_dic)
    acc_list.append(acc_dic)
    
    with open(f'{save_path}_test.json', 'w', encoding='utf-8') as file:
        json.dump(acc_list, file, ensure_ascii=False, indent=4)

    # with open(f'{save_path}_test.json', 'a', encoding='utf-8') as file:
    #     json.dump(acc_dic, file, ensure_ascii=False, indent=4)
    
    return


def parse_file(file_name):
    save_path = file_name.split('.jsonl')[0]
    data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            data.append(item)
    labels = [item['true'] for item in data]
    try:
        predicts = [item['parsed_predict'] for item in data]
    except:
        predicts = [item['predict'] for item in data]
    
    predicts = [i.replace(':', '. ') for i in predicts]
    fellowing_test(save_path, labels, predicts, data)




if __name__ == '__main__':
    p1 = '/home/gy237/project/llama3/total_final_test/JAMA_4o_diagnosis_generation/Llama370BInsJAMAreasoninginstr700003_NEJM-New-Prompt.jsonl'
    
    file_path = p1
    print('file_path:', file_path)
    parse_file(file_path)