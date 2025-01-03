import pandas as pd

# 读取 CSV 文件为 DataFrame
file_name= '/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/Llama370BInsJAMAreasoninginstr500003_JAMAFinalAll_test.csv'
df = pd.read_csv(file_name)

## 修改指定列名，例如将 'old_column_name' 修改为 'new_column_name'
# df.rename(columns={'id': 'ID'}, inplace=True)
# df.rename(columns={'input': 'case_report'}, inplace=True)
# df.rename(columns={'true': 'gold'}, inplace=True)
# df.rename(columns={'model_predict': 'predict'}, inplace=True)




df2 = pd.read_csv('/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/Llama-3.1-8B-Instruct_JAMAFinalAll_test.csv')


# 基于 'key_column' 合并 df2 中的 'new_column' 到 df1 中
# df = pd.merge(df, df2[['ID', 'topic']], on='ID', how='left')

df = pd.merge(df, df2[['ID', 'Age']], on='ID', how='left')
df = pd.merge(df, df2[['ID', 'Gender']], on='ID', how='left')
# 显示合并后的 DataFrame
df.to_csv(file_name, index=False)