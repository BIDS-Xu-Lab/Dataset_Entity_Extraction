import json

# YBXL/GI_Reasoning_train   #要用的知识  gpt-4o-mini for generating differential diagnosis lists
# YBXL/GENE_OMIM_SY_train  #背景知识 gpt-4o mini to refine the output
# MultiCaRe_PMC_Patients_PMC_CaseReport.json    gpt-4o mini to judge whether it is a case report
# MultiCaRe_PMC_Patients_PMC_CaseReport.json    gpt-4o mini to generate diagnosis list

name = 'MultiCaRe_Reasoning_test'

input_file = f"/home/gy237/project/llama3/new_data/download_openai/MultiCaRe_Reasoning_test/{name}_diagnosis_7.jsonl"
target_file = f"/home/gy237/project/llama3/new_data/upload_hug/{name}.json"
output_file = f"/home/gy237/project/llama3/new_data/upload_hug/{name}_diagnosis.json"

data = []
with open(input_file,'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))
with open(target_file, 'r', encoding='utf-8') as file:
    target = json.load(file)


assert len([i for i in data if i["response"]["body"]["choices"][0]["finish_reason"] != 'stop' or i["error"]]) == 0, 'Error'

print(len(data))
print(len(target))


upload_data = []

with open(output_file, 'r', encoding='utf-8') as file:
   upload_data = json.load(file)


for i in data:
    id_ = i["custom_id"]
    flag = True
    # print(i)
    for j in target:
        # print(j)
        # exit()
        if j['id'] == id_:
            flag = False
            j['answer'] = i["response"]["body"]["choices"][0]["message"]["content"]
            upload_data.append(j)
            # yn = i["response"]["body"]["choices"][0]["message"]["content"]
            # # print(yn)
            # # exit()
            # if yn != 'Yes' and yn != 'No':
            #     print(yn)
            # if yn.startswith('Yes'):
            #     upload_data.append(j)
            # elif not yn.startswith('No'):
            #     print(yn)
    if flag:
        print("Error, id don't match")

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(upload_data, file, ensure_ascii=False, indent=4)
    
