import json


name = 'Llama-3.3-70B-Instruct_JAMAFinalAll'




file_name = f'/home/gy237/project/llama3/total_final_test/Llama3.1_final_test_row/{name}.jsonl'
data = []
with open(file_name, 'r', encoding='utf-8') as file:
    for line in file:
        item = json.loads(line.strip())
        data.append(item)
# print(data[0])


for i in data:
    entry = {'id': i['doc']['id'], 'input': i['doc']['query'], 'true': i['doc']['answer'], "predict": i['resps'][0][0], 'topic': i['doc']['topic']}
    # print(entry)
    # exit()

    with open(f'/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/{name}.jsonl', 'a', encoding='utf-8') as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")




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


def spl(predict, key):
    predict = predict.split(key)[1]
    prediction = predict.replace('\n\n','\n')
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, prediction)
    if len(diagnosis_list) > 10:
        diagnosis_list = diagnosis_list[:10]
    return diagnosis_list


# parse the prediction
def segement_predict(predict):
    diagnosis_list = []
    key_word = ['RANKING', 'Reranked', 're-ranking', 'ranking', 'rerank']

    for i in key_word:
        if i in predict:
            diagnosis_list = spl(predict, i)
        if len(diagnosis_list) == 10:
            break
    

    if len(diagnosis_list)<10:
        if 'The final answer is' in predict:
            predict = predict.split('The final answer is')[0]
        prediction = predict.replace('\n\n','\n')
        pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
        diagnosis_list = re.findall(pattern, prediction)


    index = 0
    for i in range(len(diagnosis_list)):
        if diagnosis_list[i][0] == '1':
            index = i

    diagnosis_list = diagnosis_list[index:]

    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list]
    processed_list = [i.split('\n')[0] if '\n' in i else i for i in processed_list]
    numbers = [number for number, diagnosis in diagnosis_list]

    return processed_list, numbers



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

        processed_list, numbers = segement_predict(predict)
        save_list = [f'{numbers[i]}. {processed_list[i]}' for i in range(len(numbers))]
        if len(numbers) < 10:
            error_list.append({'predict': i['predict'], 'predict_list': save_list})
        elif numbers[0] != '1' or numbers[-1] != '10':
            error_list.append({'predict': i['predict'], 'predict_list': save_list})

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
    file_path = f'/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/{name}.jsonl'

    print('file_path:', file_path)
    main(file_path)



import pandas as pd

# 读取 CSV 文件为 DataFrame
file_name= f'/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/{name}_test.csv'
df = pd.read_csv(file_name)

df2 = pd.read_csv('/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/Llama-3.1-8B-Instruct_JAMAFinalAll_test.csv')

df = pd.merge(df, df2[['ID', 'Age']], on='ID', how='left')
df = pd.merge(df, df2[['ID', 'Gender']], on='ID', how='left')
# 显示合并后的 DataFrame
df.to_csv(file_name, index=False)



import json
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
import ast


path = '/home/gy237/project/llama3/total_final_test/Llama3.1_final_test'
output_folder = '/home/gy237/project/llama3/total_final_test/Final_result_11-7_JAMA2NEJM'


flie_list = os.listdir(path)
flie_list = [i for i in flie_list if i.endswith('.csv')]
# print(len(flie_list))
# exit()

def com_rare(name, inpt, topic):
    common = load_dataset(f"YBXL/JAMA_Reasoning_test_{name}_test", cache_dir='/home/gy237/project/download_data')
    common = common["train"]['query']
    common = [i.split('INPUT:')[-1].strip().split('OUTPUT:')[0].strip() for i in common]
    com_index = []
    
    for index in range(len(inpt)):
        for com in common:
            if com in inpt[index]:
                com_index.append(index)
    assert len(com_index) == len(common)
    each_dic = {}
    each_dic['topic'] = topic + '_' + name
    each_dic['munber'] = len(com_index)
    for k in range(10):
        column_name = 'top' + str(k+1)
        top = df[column_name].tolist()
        top = [0 if pd.isna(x) else x for x in top]
        assert len(top) == 498

        each_top = [top[index] for index in com_index]
        each_acc = sum(each_top)/len(each_top)
        each_dic[f'Top{k+1}'] = round(each_acc, 3)
    return each_dic


def index2dic(index, tpoic):
    each_dic = {}
    each_dic['topic'] = tpoic
    each_dic['munber'] = len(index)

    for k in range(10):
        column_name = 'top' + str(k+1)
        top = df[column_name].tolist()
        top = [0 if pd.isna(x) else x for x in top]
        assert len(top) == 498

        each_top = [top[i] for i in index]
        each_acc = sum(each_top)/len(each_top)
        each_dic[f'Top{k+1}'] = round(each_acc, 3)
    return each_dic



# for file_name in flie_list:
if True:
    file_name = f'{name}_test.csv'
    print(file_name)
    file_name = path + '/' + file_name

    df = pd.read_csv(file_name)


    ma = pd.read_csv('/home/gy237/project/llama3/total_final_test/JAMA_topic_mappedtoNEJM.csv')
    dic = {}
    for index, row in ma.iterrows():
        if str(row['Mapped Topic 1']) == 'nan':
            dic[row['Topic']] = [row['Topic']]
        else:
            dic[row['Topic']] = [row['Mapped Topic 1']]
            if str(row['Mapped Topic 2']) != 'nan':
                dic[row['Topic']].append(row['Mapped Topic 2'])
                if str(row['Mapped Topic 3']) != 'nan':
                    dic[row['Topic']].append(row['Mapped Topic 3'])
    # print(dic)

    for index, row in df.iterrows():
        topic = []
        _t_ = ast.literal_eval(row['topic'])
        for i in _t_:
            topic.extend(dic[i])
        df.at[index, 'topic'] = topic  # 修改 value 列中的值



    save_data = []

    # 总计算acc
    total_dic = {}
    total_dic['topic'] = 'Total'
    for i in range(10):
        column_name = 'top' + str(i+1)
        top = df[column_name].tolist()
        top = [0 if pd.isna(x) else x for x in top]
        assert len(top) == 498

        acc = sum(top)/len(top)
        total_dic[f'Top{i+1}'] = round(acc, 3)
    save_data.append(total_dic)

    # common rare计算acc
    inpt = df['case_report'].tolist()
    each_dic = com_rare('Common', inpt, 'Total')
    save_data.append(each_dic)
    each_dic = com_rare('Rare', inpt, 'Total')
    save_data.append(each_dic)


    # 计算每个性别的模型结果
    
    gender = df['Gender'].tolist()

    m_index = []
    f_index = []
    error = []
    for i in range(len(gender)):
        if gender[i] == 'M':
            m_index.append(i)
        elif gender[i] == 'F':
            f_index.append(i)
        elif gender[i] == 'U':
            error.append(i)
        else:
            print('adqeq2e2')

    
    each_dic = index2dic(m_index, 'Male')
    save_data.append(each_dic)
    each_dic = index2dic(f_index, 'Female')
    save_data.append(each_dic)

    
    # 计算每段年纪的acc
    age = df['Age'].tolist()
    # 创建一个字典，键为区间，值为索引的列表
    bins = {f'{i}-{i+10}': [] for i in range(0, 100, 10)}

    # 遍历列表，根据值归类到不同的区间
    for idx, value in enumerate(age):
        for i in range(0, 100, 10):
            if i <= value < i + 10:
                bins[f'{i}-{i+10}'].append(idx)
                break

    bins['80+'] = bins['80-90'] + bins['90-100']
    del bins['80-90']
    del bins['90-100']
    for k, v in bins.items():
        each_dic = index2dic(v, k)
        save_data.append(each_dic)

    
    # 每个topic计算acc
    _topic = df['topic'].tolist()
    # print(_topic)
    sumary = []
    for i in _topic:
        sumary.extend(i)
    sumary = list(set(sumary))
    # print(len(_topic))
    # print(sumary)
    # print(len(sumary))

    lens = []
    _save_data = []
    for i in sumary:
        index_list = []
        for j in range(len(_topic)):
            if i in _topic[j]:
                index_list.append(j)
        lens.append(len(index_list))
        # inpt_ = [inpt[w] for w in index_list]
        # assert len(inpt_) == len(index_list)
        # each_dic = com_rare('Common', inpt_, i)
        # save_data.append(each_dic)
        # each_dic = com_rare('Rare', inpt_, i)
        # save_data.append(each_dic)


        if len(index_list) >= 20:
            each_dic = index2dic(index_list, i)
            _save_data.append(each_dic)
    
    # 前十个topic
    lens = sorted(lens)[-10:][::-1]
    for _lens in lens:
        for dic in _save_data:
            if dic['munber'] == _lens:
                save_data.append(dic)
    

    # print(save_data)
    # exit()
    df_list = []
    for data in save_data:
        df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
        df_list.append(df)
        # 保存为 CSV 文件
    df_combined = pd.concat(df_list, ignore_index=True)

    name = file_name.split('.csv')[0].split('/')[-1]
    df_combined.to_csv(f'{output_folder}/{name}_result.csv', index=False)

    # with open(f'{save_path}_result.json', 'w', encoding='utf-8') as file:
    #     json.dump(save_data, file, ensure_ascii=False, indent=4)
