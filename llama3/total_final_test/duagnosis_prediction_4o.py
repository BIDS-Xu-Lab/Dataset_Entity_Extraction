import openai
from openai import OpenAI
import json
from datasets import load_dataset
from tqdm import tqdm, trange
import re


# openai API 
def generator(prompt, input_, model):
    # client = OpenAI()
    client = OpenAI()
    if model == 'o1-preview':
        chat_return = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt + '\n' + input_
            }
        ]
    )
    else:
        chat_return = client.chat.completions.create(model=model,messages=[{"role": "system", "content": prompt},
                                                                        {"role":"user", "content": input_}])
    
    result = chat_return.choices[0].message.content
    return result



prompt102 = '''
#Background#
1.Here is a case report of a patient, and we want to know what disease the patient may have.
2.Diagnoses should be confirmed by clinical or anatomic pathology tests, or sometimes by clinical criteria or expert opinion.
3.You will be informed at the end of the case description if diagnostic tests are being ordered to confirm the diagnosis.
#Role#
You are an experienced attending physician, familiar with a variety of conditions, and particularly good at interpreting a patient's case report and extracting key information from it to arrive at a diagnosis.
#Task#
Follow the steps below to process the case report. Each step will build on the previous one, ensuring a thorough and accurate diagnosis.
##Step 1##
You are an experienced Resident, highly skilled in organizing patient case reports and emphasizing critical information. Please carefully organize the Case Report following the section labeled #Case report#. Ensure that all patient information is retained without any additions or omissions, and highlight key details to aid the subsequent diagnosis by other doctors.
##Step 2##
Now, you are an expert Radiologist, proficient in interpreting imaging reports. Please analyze the imaging findings in the case report organized in ##Step 1##. Describe any abnormalities or pathological features identified. If the case report lacks imaging data, skip this step without providing any output.
##Step 3##
You are now a Multidisciplinary Team (MDT) composed of specialists across various fields. Evaluate the case report from your areas of expertise, including any specialized tests, if available. Provide diagnostic insights and suggest the ten most probable diagnoses based on your analysis.
##Step 4##
Now, assume the role of an expert Pathologist, adept at examining pathology reports to extract critical information. Please analyze the pathology report included in the #Case report# and provide a detailed pathological analysis and diagnosis.
##Step 5##
You are now an experienced Attending Physician, renowned for making accurate diagnoses in complex cases. Synthesize the information from the Case Report, along with the analyses from the Resident, Radiologist, MDT, and Pathologist. Provide at least 10 accurate and distinct diagnoses, ranked by likelihood and covering a wide range of possibilities.
##Step 6##
As the MDT, re-evaluate the case report using your collective expertise. Critically assess the diagnoses provided by the Attending Physician in ##Step 5##, offering evaluation results and any recommended modifications.
##Step 7##
Now, as the Attending Physician, refine the diagnoses based on the MDT's suggestions in ##Step 6##, incorporating all previous analyses and the original case report.
##Step 8##
Repeat ##Step 6## and ##Step 7## until a final, consensus diagnosis is reached.
#Constrains#
1.Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities.
2.List one diagnosis per line.
3.Generate at least 10 differential diagnoses related to the input case report. Think step by step.
#Output format#
Please follow the format below to output. {xx} represents a placeholder. Please think and output step by step
##Step1##
{Resident's report}
##Step2##
{Radiologist's report}
##Step3##
{MDT's report}
##Step4##
{Pathologist's report}
##Step5##
{Attending Physician's diagnoses}
##Step6##
{MDT's modification suggestions}
##Step7##
{How many rounds of conversation did they have while discussing this matter?}
##Step8##
Differential diagnosis: 1. {diagnosis}\n2. {diagnosis}\n3. {diagnosis}\n4. {diagnosis}\n5. {diagnosis}\n6. {diagnosis}\n7. {diagnosis}\n8. {diagnosis}\n9. {diagnosis}\n10. {diagnosis}
#Case report#
'''
prompt103 = '''
As a team of physicians, your task is to collaboratively generate and refine a list of at least 10 accurate and distinct differential diagnoses based on the provided case report. The process will be divided into three key steps:
Step 1: Initial Differential Diagnosis (Role 1 - Primary Physician)
As the primary physician, carefully analyze the case report and generate an initial list of 10 differential diagnoses. Ensure that each diagnosis is precise, unique, and listed in order of likelihood, with the most probable diagnosis at the top.
Step 2: Review and Critique (Role 2 - Consulting Physician)
As the consulting physician, critically evaluate the initial list of differential diagnoses. Provide feedback on each diagnosis, pointing out any potential oversights, alternative possibilities, or areas where the reasoning could be improved. Based on this feedback, suggest any necessary revisions.
Step 3: Final Differential Diagnosis (Role 1 - Primary Physician)
Return to the primary physician role and incorporate the feedback received. Refine the list to finalize 10 differential diagnoses, ensuring they are accurate, distinct, and ordered by likelihood.
Output format:
Initial Differential Diagnosis:
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
Consulting Physician Feedback:
Diagnosis 1: [Feedback]
Diagnosis 2: [Feedback]
Diagnosis 3: [Feedback]
[Continue for all 10 diagnoses]
Differential diagnosis:
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
Note: Ensure that the final list reflects the most accurate and likely diagnoses, taking into consideration the critique provided.
Guideline: 1. Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities. 2. List one diagnosis per line. 3. Generate at least 10 differential diagnoses related to the input case report. Note: Ensure that you provide at least 10 diagnoses in order of likelihood, with the most likely diagnosis listed first. 4. think step by step.
'''

# name = 'NEJM_Reasoning_Final_Old_PROMPT_test_refined'
name = 'JAMA_FINAL_test'

# 加载数据
data = load_dataset(f"YBXL/{name}", cache_dir='/home/gy237/project/download_data')
# print(data)
# print(data['train']['query'][0])
# exit()

id_ = data["train"]['id']
input_ = data["train"]['query']
topic = data["train"]['topic']

prompt = [i.split('INPUT:')[0] for i in input_]
query = [i.split('INPUT:')[1] for i in input_]
answer = data["train"]['answer']


# check the prompt
for i in range(len(prompt) - 1):
    if prompt[i] != prompt[i+1]:
        print('Error')



def test_format(output):
    """
    test o1's generation's format
    """
    predicts = output
    pattern = r'(\d+)\.\s*([^\d\n]+(?:\n(?!\d+\.)[^\d\n]+)*)'
    diagnosis_list = re.findall(pattern, predicts)
    processed_list = [diagnosis.strip() for number, diagnosis in diagnosis_list if int(number) <= 10]

    if len(processed_list) < 10:
        return 'No'
    else:
        return 'Yes'


# generate diagnosis
# model_list = ['gpt-4-turbo']
model_list = ['gpt-3.5-turbo']
# model_list = ['gpt-4o']
# model_list = ['o1-preview']
for model in model_list:
    print('-'*100)
    print(model)
    json_path = f'/home/gy237/project/llama3/total_final_test/JAMA_final_test/{name}_{model}.jsonl'
    for i in tqdm(range(len(id_)), ncols=100):
        flag = True
        count = 0
        while flag:
            result = generator(prompt[i], query[i], model)
            yn = test_format(result)
            if yn == 'Yes':
                flag = False
            elif count > 4:
                flag = False
                print('Error format!', id_[i])
            else:
                count += 1

        entry = {'id': id_[i], 'input': prompt[i] + 'INPUT:' + query[i], 'true': answer[i], "predict": result, 'topic': topic[i]}
        # print(entry)
        # exit()

        with open(json_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

