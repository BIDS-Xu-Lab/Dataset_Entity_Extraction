import json

with open('/home/gy237/project/llama3/new_data/combined_data/scroe_4_5_explanation.json', 'r', encoding='utf-8') as file:
    output = json.load(file)
print(len(output))
with open('/home/gy237/project/llama3/new_data/combined_data/scroe_4_5.json', 'r', encoding='utf-8') as file:
    output = json.load(file)
print(len(output))