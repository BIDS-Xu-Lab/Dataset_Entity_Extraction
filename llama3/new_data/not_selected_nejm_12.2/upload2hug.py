from datasets import load_dataset
from datasets import Dataset, DatasetDict

# 下载数据
ds = load_dataset("YBXL/JAMA-reasoning-instruction", cache_dir='/home/gy237/project/download_data')
print(ds)
exit()

# 上传数据
upload = []
upload.append({'id': i['id'], 'query': inpt, 'answer': i['output'], 'topic': i['topic']})
train_dataset = Dataset.from_list(upload)
t = Dataset.from_list(upload[:10])

dataset_dict = DatasetDict({
    "train": train_dataset,
    "valid":t,
    "test": t
})
dataset_dict.push_to_hub("YBXL/JAMA_Reasoning_o1refained")