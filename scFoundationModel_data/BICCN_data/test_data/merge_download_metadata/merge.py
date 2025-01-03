import pandas as pd


metadata_file = 'BICCN_data/test_data/merge_metadata/NeMO_manifest_metadata_823298bba.tsv'
data_file = 'BICCN_data/test_data/merge_metadata/NeMO_manifest_876f7e5ef.tsv'

metadata = pd.read_csv(metadata_file, sep='\t')
data = pd.read_csv(data_file, sep='\t')

# 合并相同的行
rows_as_lists = []
sample_id_list = []
for index, row in metadata.iterrows():
    list_ = row.tolist()
    sample_id = list_[0]
    if sample_id not in sample_id_list:
        sample_id_list.append(sample_id)
        rows_as_lists.append(row.tolist())
print(len(rows_as_lists))

column_names = metadata.columns.tolist()
merged_metadata = pd.DataFrame(rows_as_lists, columns=column_names)

# 根据 sample_id 合并两个 DataFrame
merged_df = pd.merge(merged_metadata, data, on='sample_id', how='inner')  # 'inner' 保证只保留相同的 sample_id

# 将结果保存到新文件中，例如 CSV
merged_df.to_csv('BICCN_data/test_data/merge_metadata/merged_download_datas.csv', index=False)