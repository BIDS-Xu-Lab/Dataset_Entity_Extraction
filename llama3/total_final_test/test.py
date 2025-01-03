import json
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
import ast


# path = '/home/gy237/project/llama3/total_final_test/Llama3.1_final_test'
path = '/home/gy237/project/llama3/total_final_test/JAMA_final_test'
output_folder = '/home/gy237/project/llama3/total_final_test/Final_result_11-7_JAMA2NEJM'
# name = 'Llama370BInsJAMAreasoninginstr1000003_JAMAFinalAll_test.csv'

names = os.listdir(path)
names = [i for i in names if i.endswith('.csv')]


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



for name in names:
# if True:
    file_name = name
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
    lens = sorted(lens)#[-10:][::-1]
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
    df_combined.to_csv(f'{output_folder}/{name}_12_20.csv', index=False)

    # with open(f'{save_path}_result.json', 'w', encoding='utf-8') as file:
    #     json.dump(save_data, file, ensure_ascii=False, indent=4)