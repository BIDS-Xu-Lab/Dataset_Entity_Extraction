import os
import pandas as pd
import json
import re
from tqdm import tqdm
from openai import OpenAI

# openai API 
def generator(prompt, model):
    # client = OpenAI()
    client = OpenAI()
    if model == 'o1-preview':
        chat_return = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    else:
        chat_return = client.chat.completions.create(model=model,messages=[{"role": "system", "content": prompt},
                                                                        {"role":"user", "content": input_}])
    
    result = chat_return.choices[0].message.content
    return result


def test_format(output):
    """
    test o1's generation's format
    """
    predicts = output
    predicts = predicts.replace('\n\n','\n')
    predicts = predicts.replace('#','')
    predicts = predicts.replace(':\n',': ')
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, predicts)
    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

    if len(processed_list) < 10:
        predicts = predicts.replace('\n\n','\n')
        pattern = r'(\d+):\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
        diagnosis_list = re.findall(pattern, predicts)
        processed_list = [diagnosis.split('\n')[0].strip() for number, diagnosis in diagnosis_list if int(number) <= 10]
        # print(processed_list)
    
    if len(processed_list) < 10:
        return 'No'
    else:
        processed_list = processed_list[-10:]
        final = processed_list[0].lower()
        final = final.replace('#', '')
        return final


with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate.jsonl', 'r') as f:
    lines = f.readlines()
    data = [json.loads(i.strip()) for i in lines]
print(len(data))

with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate_refained.jsonl', 'r') as f:
    lines = f.readlines()
    exist_data = [json.loads(i.strip()) for i in lines]
exist_id = [i['id'] for i in exist_data]
print(len(exist_id))


# refine the output
# for i in tqdm(range(len(data)), ncols=100):
#     if data[i]['id'] not in exist_id:
#         final_dia = data[i]['input'].split('Final diagnosis: ')[-1].lower().split('\n')[0]
#         final_dia = final_dia.replace('#', '')
#         if final_dia.endswith('.'):
#             final_dia = final_dia[:-1]

#         yn = test_format(data[i]['output'])
#         if yn == 'No':
#             print('Error')
#         elif final_dia not in yn and yn not in final_dia:
#             inpt = data[i]['input']

#             flag = True
#             count = 0
#             while flag:
#                 result = generator(inpt, 'o1-preview')
#                 yn_ = test_format(result)
#                 if final_dia in yn_ or yn_ in final_dia:
#                     flag = False
#                 elif count > 2:
#                     flag = False
#                     print('Error format!', data[i]['id'])
#                 else:
#                     count += 1
#             data[i]['refined_output'] = result

#         with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate_refained.jsonl', 'a', encoding='utf-8') as file:
#                 file.write(json.dumps(data[i], ensure_ascii=False) + "\n")





def test_diagnose(each_diagnosis,true_diagnosis):
    """
    # Determine whether the each_diagnosis and true_diagnosis is a same diagnosis
    # each_diagnosis: generated diagnosis, str
    # true_diagnosis: true diagnosis, str
    """
    # return 'Y', 'test'     # use, when you check the diagnoses list
    client = OpenAI()
    
    prompt ="""Your task is to identify whether the provided predicted differential diagnosis is correct based on the true diagnosis. Carefully review the information and determine the correctness of the prediction. Please notice same diagnosis might be in different words. Only return "Y" for yes or "N" for no, without any other words."""
    chat_return = client.chat.completions.create(model='gpt-4o',temperature=0.0, messages=[{"role": "system", "content": "pysician"},
                                                                     {"role":"user",
                                                                      "content":f"{prompt} \n"\
                                                                      f"Predicted Diagnosis: {each_diagnosis}\n"\
                                                                      f"True Diagnosis: {true_diagnosis}",
                                                                      }])
    result=chat_return.choices[0].message.content
    return result




error = []
count = 0
for i in range(len(exist_data)):   
    final_dia = exist_data[i]['input'].split('Final diagnosis: ')[-1].lower().split('\n')[0]
    final_dia = final_dia.replace('#', '')
    if final_dia.endswith('.'):
        final_dia = final_dia[:-1]

    try:
        yn = test_format(exist_data[i]['refined_output'])
    except:
        yn = test_format(exist_data[i]['output'])

    if yn == 'No':
        error.append(exist_data[i])
        print('Error')
    elif final_dia not in yn and yn not in final_dia:
        result = test_diagnose(final_dia,yn)
        if result == 'Y':
            count+=1
        elif result == 'N':
            error.append(exist_data[i])
            print(exist_data[i]['id'])
            print(final_dia)
            print(yn)
        else:
            print(result)
        

with open('/home/gy237/project/llama3/new_data/JAMA_Reasoning/o1_generate_error.json', 'w', encoding='utf-8') as file:
    json.dump(error, file, ensure_ascii=False, indent=4)
print(len(error))

