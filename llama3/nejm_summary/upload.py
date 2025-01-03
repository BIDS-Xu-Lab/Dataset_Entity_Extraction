from datasets import Dataset, DatasetDict
import json
from datasets import load_dataset
from tqdm import tqdm, trange
import json


with open(f'/home/gy237/project/llama3/new_data/JAMA_Reasoning/jama_filtered_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(len(data))


train_dataset = Dataset.from_list(data)
t = Dataset.from_list(data[:10])

dataset_dict = DatasetDict({
    "train": train_dataset,
    "valid":t,
    "test": t
})
print(dataset_dict['train'][0])
# exit()
dataset_dict.push_to_hub("YBXL/JAMA_Reasoning_o1refained")