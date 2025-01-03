import subprocess
import os
import json
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import io
import logging
logging.getLogger("transformers.utils.hub").setLevel(logging.CRITICAL+1)
from unsloth import FastLanguageModel
import torch
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
path = 'output_3.1_filtered_data/checkpoint-10000'
MODEL_NAME = '/home/gy237/project/llama3/unsloth/' + path

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = None,
        dtype = None,
        load_in_4bit = True,
    )
FastLanguageModel.for_inference(model)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = tuple(set(stop_token_ids))
    pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_token_ids
    pass
pass

def async_process_chatbot(message, history):
    
    eos_token = tokenizer.eos_token
    stop_on_tokens = StopOnTokens([eos_token,])
    text_streamer  = TextIteratorStreamer(tokenizer, skip_prompt = True)

    history_transformer_format = history + [[message, ""]]
    messages = []
    for item in history_transformer_format:
        if item:
            messages.append({"role": "user",      "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    pass
    # print(message)
    # exit()

    # Remove last assistant and instead use add_generation_prompt
    messages.pop(-1)
   
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda", non_blocking = True)
    
    outputs = model.generate(
        input_ids, 
        streamer = text_streamer,
        max_new_tokens = 102400, 
        stopping_criteria = StoppingCriteriaList([stop_on_tokens,]),
        temperature = 0.7,
        do_sample = True,
        use_cache = True)
    outputs = tokenizer.batch_decode(outputs)

    return outputs[0]

# result = async_process_chatbot('hello, do you know what is the whether today', [])

import openai
from openai import OpenAI
def test_diagnose(prompt, input_, mo):   
    client = OpenAI()         
    chat_return = client.chat.completions.create(model=mo,messages=[{"role": "system", "content": prompt},
                                                                     {"role":"user", "content": input_}])
    result=chat_return.choices[0].message.content
    return result

# print(test_diagnose(prompt103, i, 'gpt-4o-mini'))


# {"text": "<human>: {question}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
import json
from datasets import load_dataset
from datasets import DatasetDict
from tqdm import tqdm, trange
from collections import Counter
# 加载数据
data = load_dataset("YBXL/NEJM_Reasoning_Final_test", cache_dir='/home/gy237/project/download_data')

dataset = data["train"]
_query = data["train"]['query']
answer = data["train"]['answer']

prompt = [i.split('INPUT:')[0] for i in _query]
query = [i.split('INPUT:')[1] for i in _query]

for i in range(len(prompt) - 1):
    if prompt[i] != prompt[i+1]:
        print('Error')


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


dic_list = []
for i in trange(len(prompt), desc="Step"):
    result = async_process_chatbot(f'{prompt[i]} {query[i]}', [])
    result = result.split('Differential')[-1].split('<|eot_id|>')[0]

    # result = test_diagnose(prompt[i], query[i], 'gpt-4o-mini')
    # result = result.split('Differential')[-1]
    # print(result)
    # exit()
    dic_list.append({'truth': answer[i], "logit_0": result})
    
# path = 'Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt102'
json_path = f'/home/gy237/project/llama3/unsloth/final_test_data/{path}.json'
os.makedirs(f'/home/gy237/project/llama3/unsloth/final_test_data/{path}', exist_ok=True)
with open(json_path, 'w', encoding='utf-8') as file:
    json.dump(dic_list, file, ensure_ascii=False, indent=4)

import sys
sys.path.append('/home/gy237/project/llama3/unsloth/final_test_data/')
from test import*

result = test(json_path)

dct = {
    f'{path}' : result
}
with open(f'/home/gy237/project/llama3/unsloth/final_test_data/{path}_result.json', 'w', encoding='utf-8') as file:
    json.dump(dct, file, ensure_ascii=False, indent=4)