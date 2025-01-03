import json


name = 'Llama370BInsJAMAreasoninginstr700003_JAMAFinalAll.jsonl'


file_name = f'/home/gy237/project/llama3/total_final_test/Llama3.1_final_test_row/{name}'
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

    with open(f'/home/gy237/project/llama3/total_final_test/Llama3.1_final_test/{name}', 'a', encoding='utf-8') as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")