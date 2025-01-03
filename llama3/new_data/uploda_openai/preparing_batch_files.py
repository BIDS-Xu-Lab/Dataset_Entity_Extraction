import json

# YBXL/GENE_OMIM_SY_train  #背景知识 gpt-4o mini to refine the output
prompt1 =  '''You are an expert in genetics and gene-related diseases. You will be provided with a text related to specific genetic diseases. The text contains many '\n's which made it poorly structured and confusing. Your task is to turn these incoherent raw phrases into coherent, natural, complete and easy-to-understand paragraphs without changing medical information. Do not add, delete, or modify any medical information—simply refine the formatting and clarity. Be careful not to break lines and only respond with reorganized long sentences and long paragraphs.'''

# YBXL/GI_Reasoning_train   #要用的知识  gpt-4o-mini for generating differential diagnosis lists
prompt2 = '''Generate the next 9 accurate and distinct differential diagnoses based on the input case report, considering the true final diagnosis provided after “OUTPUT:“. Your goal is to list the 9 most likely alternative diagnoses in descending order of likelihood, ensuring that each possibility is unique and covers a broad spectrum of differential diagnoses. \n Follow the guidelines for a generation: 1. Each diagnosis should be precise and unique, ensuring a variety of the next 9 possibilities. 2. List one diagnosis per line and the true diagnosis we provide should be the first one. 3. Generate the next 9 differential diagnoses related to the input case report. Think step by step. \n \n***Output format***:Differential diagnosis: 1. \n2. \n3.\n4. \n5. \n6. \n7. \n8. \n9. \n10. \n'''

# MultiCaRe_PMC_Patients_PMC_CaseReport.json    gpt-4o mini to judge whether it is a case report
prompt3 = '''You are an expert medical professional with extensive experience in clinical diagnosis. You will be provided with a text, which may contain elements such as patient history, clinical findings, and diagnostic details. Your task is to critically analyze the text to determine whether it qualifies as a case report that includes detailed descriptions of the patient's history, symptoms, and the diagnostic reasoning process. Use your medical expertise and reasoning skills to make an informed decision. Respond with a simple 'Yes' if the text qualifies as a case report with these specific details, or 'No' if it does not.'''

# MultiCaRe_PMC_Patients_PMC_CaseReport  and  MultiCaRe_Reasoning_test      gpt-4o mini to generate diagnosis list
prompt4 = '''Your task is to provide at least 10 accurate and distinct patient diagnoses based on the input case report.   Ensure you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities.  \n Follow the guidelines for a generation: 1.   Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities.   2.   List one diagnosis per line.   3.   Generate 10 differential diagnoses related to the input case report.   Think step by step.  \n \n***Output format***:Differential diagnosis: 1.   \n2.   \n3.\n4.   \n5.   \n6.   \n7.   \n8.   \n9.   \n10.'''

# PMC_Patients_PMC_CaseReport  and  MultiCaRe_Reasoning_test  gpt-4o mini to generate diagnosis and explantion list
prompt5 = '''As a meticulous and evidence-driven physician, your task is to generate exactly 10 accurate and distinct differential diagnoses based on the diagnosis information mentioned in the input case report and its title. Each diagnosis must be followed by a concise, evidence-informed explanation. Your diagnoses should span a wide range of possibilities, listed in order of likelihood, with the most likely diagnosis appearing first. Focus on clinical findings, considering their diagnostic accuracy and relevance as emphasized in evidence-based physical diagnosis practices. Ensure each diagnosis reflects key physical findings relevant to the patient's symptoms, with emphasis on findings that significantly alter the likelihood of specific conditions.
**Output format**:
Differential diagnosis:
1. [Diagnosis 1]: [Evidence-informed One-sentence explanation]
2. [Diagnosis 2]: [Evidence-informed One-sentence explanation]
3. [Diagnosis 3]: [Evidence-informed One-sentence explanation]
4. [Diagnosis 4]: [Evidence-informed One-sentence explanation]
5. [Diagnosis 5]: [Evidence-informed One-sentence explanation]
6. [Diagnosis 6]: [Evidence-informed One-sentence explanation]
7. [Diagnosis 7]: [Evidence-informed One-sentence explanation]
8. [Diagnosis 8]: [Evidence-informed One-sentence explanation]
9. [Diagnosis 9]: [Evidence-informed One-sentence explanation]
10. [Diagnosis 10]: [Evidence-informed One-sentence explanation]
**Example**:
Differential diagnosis:
1. Mitochondrial disease: Considering the patient's history of celiac disease, fatigue, and somatic symptoms, a mitochondrial disorder could be a possibility, especially with the presence of a prolonged QT interval.
Please generate exactly 10 differential diagnoses with corresponding evidence-based one-sentence explanations, adhering strictly to the output format without including any other outputs.
INPUT: '''

name = 'MultiCaRe_Reasoning_test_diagnosis'

score_path = f'/home/gy237/project/llama3/new_data/final_data/MultiCaRe_Reasoning_test_diagnosis.json'
file_path = f'/home/gy237/project/llama3/new_data/temporary_data/MultiCaRe_Reasoning_test.json'
output_file = f'/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation/{name}_explation.jsonl'


with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
with open(score_path, 'r', encoding='utf-8') as file:
    score_data = json.load(file)
print(len(data))
# print(data[0])
# exit()

assert len(data) == len(set([i['id'] for i in data])), "Error"

count = 0
number = 12500
for i in score_data:
    id_ = i['id']
    if i['score'] == '4' or i['score'] == '5':
        for j in data:
            if j['id'] == id_:
                inpt = j['query'].split('INPUT:')[1]
                dic = {"custom_id": j['id'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": prompt5},{"role": "user", "content": f"{inpt}"}],"max_tokens": 49000}}
                # print(dic)
                # exit()
                count += 1
                if count//number == 0:
                    with open(output_file,'a', encoding='utf-8') as file:
                        file.write(json.dumps(dic) + '\n')
                else:
                    with open(f'/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation/{name}_{count//number}_explation.jsonl','a', encoding='utf-8') as file:
                        file.write(json.dumps(dic) + '\n')

with open(output_file,'r', encoding='utf-8') as file:
    data = file.readlines()
print(len(data))
print(data[0])

# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

# # limit
# Per-batch limits: A single batch may include up to 50,000 requests, and a batch input file can be up to 100 MB in size. 
# Note that /v1/embeddings batches are also restricted to a maximum of 50,000 embedding inputs across all requests in the batch.
# Enqueued prompt tokens per model: Each model has a maximum number of enqueued prompt tokens allowed for batch processing.