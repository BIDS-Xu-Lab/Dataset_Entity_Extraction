import json
import random
random.seed(42)

# YBXL/GENE_OMIM_SY_train  #背景知识 gpt-4o mini to refine the output
prompt1 =  '''You are an expert in genetics and gene-related diseases. You will be provided with a text related to specific genetic diseases. The text contains many '\n's which made it poorly structured and confusing. Your task is to turn these incoherent raw phrases into coherent, natural, complete and easy-to-understand paragraphs without changing medical information. Do not add, delete, or modify any medical information—simply refine the formatting and clarity. Be careful not to break lines and only respond with reorganized long sentences and long paragraphs.'''

# YBXL/GI_Reasoning_train   #要用的知识  gpt-4o-mini for generating differential diagnosis lists
prompt2 = '''Generate the next 9 accurate and distinct differential diagnoses based on the input case report, considering the true final diagnosis provided after “OUTPUT:“. Your goal is to list the 9 most likely alternative diagnoses in descending order of likelihood, ensuring that each possibility is unique and covers a broad spectrum of differential diagnoses. \n Follow the guidelines for a generation: 1. Each diagnosis should be precise and unique, ensuring a variety of the next 9 possibilities. 2. List one diagnosis per line and the true diagnosis we provide should be the first one. 3. Generate the next 9 differential diagnoses related to the input case report. Think step by step. \n \n***Output format***:Differential diagnosis: 1. \n2. \n3.\n4. \n5. \n6. \n7. \n8. \n9. \n10. \n'''

# MultiCaRe_PMC_Patients_PMC_CaseReport.json    gpt-4o mini to judge whether it is a case report
prompt3 = '''You are an expert medical professional with extensive experience in clinical diagnosis. You will be provided with a text, which may contain elements such as patient history, clinical findings, and diagnostic details. Your task is to critically analyze the text to determine whether it qualifies as a case report that includes detailed descriptions of the patient's history, symptoms, and the diagnostic reasoning process. Use your medical expertise and reasoning skills to make an informed decision. Respond with a simple 'Yes' if the text qualifies as a case report with these specific details, or 'No' if it does not.'''

# MultiCaRe_PMC_Patients_PMC_CaseReport        gpt-4o mini to generate diagnosis list
prompt4 = '''Your task is to provide at least 10 accurate and distinct patient diagnoses based on the input case report.   Ensure you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities.  \n Follow the guidelines for a generation: 1.   Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities.   2.   List one diagnosis per line.   3.   Generate 10 differential diagnoses related to the input case report.   Think step by step.  \n \n***Output format***:Differential diagnosis: 1.   \n2.   \n3.\n4.   \n5.   \n6.   \n7.   \n8.   \n9.   \n10.'''

prompt5 = '''You are a medical expert tasked with evaluating the quality of the following training sample, which consists of an input (patient case or question) and/or an output (diagnosis or answer).
***Task***: Rate the training pair on a scale of 1 to 5 based on the criteria provided below. A higher score indicates that the training pair is of high quality and useful for enhancing models' diagnostic reasoning ability and medical knowledge.
***Rate the training pair according to the following criteria***:
1: The training pair is entirely irrelevant to improving the model's diagnostic reasoning ability and medical knowledge. The output is not aligned with the input and fails to address the input's context or question. The input and output exhibit poor writing style and formatting, making it very difficult to understand the aim of the task.
2: The input is somewhat irrelevant to improving the model's diagnostic reasoning ability and medical knowledge. The output partially addresses the input but does so inadequately, lacking depth or accuracy.
3: The input is relevant to improving the model's diagnostic reasoning ability and medical knowledge. The output is generally aligned with the input, but may lack thoroughness or clarity in its explanation or reasoning. The training pair is moderately useful but has room for improvement.
4: The input is relevant and useful for improving the model's diagnostic reasoning ability and medical knowledge. The output effectively addresses the input, demonstrating solid reasoning and appropriate depth. The training pair is of good quality, with minor issues that do not significantly detract from its usefulness.
5: The training pair is highly relevant and valuable for improving the model's diagnostic reasoning ability and medical knowledge. The output is perfectly aligned with the input, providing a comprehensive, accurate, and well-explained response. The training pair is of exceptional quality, with clear and well-structured content that is easy to follow and understand.
Response: Only respond with a score of 1, 2, 3, 4, or 5.
INPUT:'''

name = 'DDXPlus_Reasoning_train'
# MedQA_Reasoning_train 不过滤

file_path = f'/home/gy237/project/llama3/new_data/final_filter/{name}.json'
output_file = f'/home/gy237/project/llama3/new_data/final_filter/upload_openai'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(len(data))
# print(data[0])
# exit()

data = random.sample(data, 50000)

assert len(data) == len(set([i['id'] for i in data])), "Error"

count = 0
for i in data:
    inpt = i['query'].split('INPUT:')[1] + i['answer']
    dic = {"custom_id": i['id'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": prompt5},{"role": "user", "content": f"{inpt}"}],"max_tokens": 12800}}
    # print(dic)
    # exit()
    count += 1
    if count//17000 == 0:
        with open(f'{output_file}/{name}.jsonl','a', encoding='utf-8') as file:
            file.write(json.dumps(dic) + '\n')
    else:
        with open(f'{output_file}/{name}_{count//17000}.jsonl','a', encoding='utf-8') as file:
            file.write(json.dumps(dic) + '\n')

with open(f'{output_file}/{name}.jsonl','r', encoding='utf-8') as file:
    data = file.readlines()
print(len(data))
# print(data[0])

# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

# # limit
# Per-batch limits: A single batch may include up to 50,000 requests, and a batch input file can be up to 100 MB in size. 
# Note that /v1/embeddings batches are also restricted to a maximum of 50,000 embedding inputs across all requests in the batch.
# Enqueued prompt tokens per model: Each model has a maximum number of enqueued prompt tokens allowed for batch processing.