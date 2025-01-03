import pandas as pd
import ast


ma = pd.read_csv('/home/gy237/project/llama3/total_final_test/JAMA_topic_mappedtoNEJM.csv')
dic = {}
# print(ma['Mapped Topic 2'].tolist())
for index, row in ma.iterrows():
    if str(row['Mapped Topic 1']) == 'nan':
        dic[row['Topic']] = [row['Topic']]
    else:
        dic[row['Topic']] = [row['Mapped Topic 1']]
        if str(row['Mapped Topic 2']) != 'nan':
            dic[row['Topic']].append(row['Mapped Topic 2'])
            if str(row['Mapped Topic 3']) != 'nan':
                dic[row['Topic']].append(row['Mapped Topic 3'])
print(dic)


file_name= '/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/Llama370BInsJAMAreasoninginstr500003_JAMAFinalAll_test.csv'
df = pd.read_csv(file_name)

for index, row in df.iterrows():
    topic = []
    _t_ = ast.literal_eval(row['topic'])
    for i in _t_:
        topic.extend(dic[i])
    df.at[index, 'topic'] = topic  # 修改 value 列中的值

# print(df)


