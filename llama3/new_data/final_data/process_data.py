import os
import json

# folder_path = '/home/gy237/project/llama3/new_data/final_filter'
# file_names = os.listdir(folder_path)
# target_names = [i.split('.')[0] for i in file_names if i.endswith('json')]
# print(len(target_names))


# folder_path = '/home/gy237/project/llama3/new_data/final_filter/download_openai'
# file_names = os.listdir(folder_path)
# input_names = [i.split('.')[0] for i in file_names if i.endswith('jsonl')]
# print(len(input_names))

# folder_path = '/home/gy237/project/llama3/new_data/final_data'
# file_names = os.listdir(folder_path)
# out_names = [i.split('.')[0] for i in file_names if i.endswith('json')]
# print(len(out_names))

# target_name = []
# for i in target_names:
#     if i not in out_names and i != 'PMC_Patients':
#         target_name.append(i)
# print(len(target_name))
# exit()
# for name in target_name:
#     score_dic = {}
#     upload_data = []
#     for in_name in input_names:
#         if name in in_name:
#             input_file = f"/home/gy237/project/llama3/new_data/final_filter/download_openai/{in_name}.jsonl"
#             target_file = f"/home/gy237/project/llama3/new_data/final_filter/{name}.json"
#             output_file = f"/home/gy237/project/llama3/new_data/final_data/{name}.json"

    
#             data = []
#             with open(input_file,'r', encoding='utf-8') as file:
#                 for line in file:
#                     data.append(json.loads(line))
#             with open(target_file, 'r', encoding='utf-8') as file:
#                 target = json.load(file)


#             assert len([i for i in data if i["response"]["body"]["choices"][0]["finish_reason"] != 'stop' or i["error"]]) == 0, 'Error'
            
#             print(f'{name}: {len(target)}, {in_name}: {len(data)}')

#             for i in data:
#                 id_ = i["custom_id"]
#                 flag = True
#                 # print(i)
#                 for j in target:
#                     # print(j)
#                     # exit()
#                     if j['id'] == id_:
#                         flag = False
#                         score = i["response"]["body"]["choices"][0]["message"]["content"]
#                         # print(score)
#                         # exit()
#                         j['score'] = score
#                         upload_data.append(j)
#                         if score in score_dic.keys():
#                             score_dic[score] += 1
#                         else:
#                             score_dic[score] = 1
                        
#                 if flag:
#                     print("Error, id don't match")

#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(upload_data, file, ensure_ascii=False, indent=4)
#     print(f'{name}: {len(target)}, upload_data: {len(upload_data)}')
#     print(score_dic)

# title 去掉，检查prompt
prompt1 = "Your task is to provide at least 10 accurate and distinct patient diagnoses and the final diagnosis based on the input case report. Key points: 1) Diagnoses are confirmed by clinical or anatomic pathology tests, or sometimes by clinical criteria or expert opinion. 2) You will be informed at the end of the case description if diagnostic tests are being ordered to confirm the diagnosis. Ensure that you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities. \n Follow the guidelines for a generation: 1. Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities. 2. List one diagnosis per line. 3. Generate at least 10 differential diagnoses related to the input case report. 4. Generate the final diagnosis. Think step by step. \n \n***Output format***:Differential diagnosis: 1. \n2. \n3.\n4. \n5. \n6. \n7. \n8. \n9. \n10. \n\n final diagnosis: \n"
# YBXL/GI_Reasoning_train   #要用的知识  gpt-4o-mini for generating differential diagnosis lists
prompt2 = '''Your task is to provide at least 10 accurate and distinct patient diagnoses based on the input case report. Key points: 1) Diagnoses are confirmed by clinical or anatomic pathology tests, or sometimes by clinical criteria or expert opinion. 2) You will be informed at the end of the case description if diagnostic tests are being ordered to confirm the diagnosis. Ensure that you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities.\n Follow the guidelines for a generation: 1. Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities. 2. List one diagnosis per line. 3. Generate at least 10 differential diagnoses related to the input case report. Think step by step.\n \n***Output format***:Differential diagnosis: 1. \n2. \n3.\n4. \n5. \n6. \n7. \n8. \n9. \n10. \n\n\n'''

path = '/home/gy237/project/llama3/new_data/final_data/PMC_Patients_diagnosis.json'

with open(path, 'r', encoding='utf-8') as file:
    target = json.load(file)
for i in target:

    pr = i['query'].split('\nTitle: ')
    de = i['query'].split('\nDescription: \n')
    assert len(pr)==len(de)==2, print(i)
    i['query'] = pr[0] + '\nDescription: \n' + de[-1]

with open(path, 'w', encoding='utf-8') as file:
    json.dump(target, file, ensure_ascii=False, indent=4)

# Title: Suspected Anaphylactic Reaction Prior to Induction of Anesthesia\nDescription:
# Title: Long-Term/nAbstract:
# \nTitle: Lithi\nAbstract: