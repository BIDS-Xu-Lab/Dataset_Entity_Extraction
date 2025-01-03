# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

from openai import OpenAI
import os
client = OpenAI()


import json
folder_path = '/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation'
file_names = os.listdir(folder_path)
file_names = [i for i in file_names if i.endswith('jsonl')]
print(len(file_names))
for name in file_names:
  with open(f'/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation/{name}', 'r', encoding='utf-8') as file:
      # 逐行读取
      for line in file:
          # 将每行解析为 JSON 对象
          data = json.loads(line)
          data['body']['max_tokens'] = 16380
          with open(f'/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation_new/{name}','a', encoding='utf-8') as file:
              file.write(json.dumps(data) + '\n')
exit()     


def uplo(name):
  name = name
  # upload your batch input file
  batch_input_file = client.files.create(
    file=open(f"/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation/{name}", "rb"),
    purpose="batch"
  )
  # create the batch, only the description can change
  batch_input_file_id = batch_input_file.id
  client.batches.create(
      input_file_id=batch_input_file_id,
      endpoint="/v1/chat/completions",
      completion_window="24h",
      metadata={
        "description": f'{name}'
      }
  )
folder_path = '/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation'
file_names = os.listdir(folder_path)
file_names = [i for i in file_names if i.endswith('jsonl')]
print(len(file_names))
# for name in file_names:
#   print(name)
#   uplo(name)

# Getting a list of All Batches
print(client.batches.list(limit=50))
exit()
# 从openai下载
a = client.batches.list(limit=30)
des_list = []
for i in a:
  i = str(i)
  status = i.split("status='")[1].split("', ")[0]
  description = i.split("'description': '")[1].split("'}")[0]
  dic = {'description':description, 'status':status}
  if "output_file_id='" in i:
    # print('awd')
    output_file_id = i.split("output_file_id='")[1].split("', ")[0]
    dic['output'] = output_file_id

  des_list.append(dic)

des_list = [i for i in des_list if i['description'] in file_names and i['status']=='completed']
print(len(des_list))
print(des_list)
exit()

import os
for i in des_list:
  des = i['description']
  out = i['output']

  os.system(f'''curl https://api.openai.com/v1/files/{out}/content \
  -H "Authorization: Bearer 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" > /home/gy237/project/llama3/new_data/download_openai/multicare_pmc_explation/{des}''')



# for name in file_names:
#   if name not in des_list:
#     print(name)
#     uplo(name)

# file-ABjfnaITuKDTyF6Qws2RBUhk
# file-MMxFQbs35xfRXs2OWOaRg1kS
# file-eAtRpk2Arf1IfxBc4dpKoXlD
# file-2V1vkIFNr8tq0prPShEGdHEV
# file-rCcuI0W2ykABXOId3InyzGTB
# file-i5X1KLHZtYl7k67prv6Ia7fW
# file-qHNVioWyohK9X0ZXbwSxNE3M

'''curl https://api.openai.com/v1/files/file-eiYSxGXBiJKd8mhasGXrkfVu \
  -H "Authorization: Bearer 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" > /home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation/error.jsonl
'''

# # Retrieving the Results
# # file-tOrHABMkkqiTCGQrclhxKe8m
# file_response = client.files.content("file-XSXK81QI2uzTnHT8SnX0sNh1")
# print(file_response)


# # Checking the Status of a Batch
# batch1 = "batch_KTZk1pOwJ83RFKTVMICDsTbV"
# batch2 = ""
# print(client.batches.retrieve(batch2))


# # Cancelling a Batch
# a = client.batches.list(limit=30)
# des_list = []
# for i in a:
#     i = str(i)
#     status = i.split("status='")[1].split("', ")[0]
#     patch_id = i.split("Batch(id='")[1].split("', ")[0]
#     if status=='in_progress':
#         client.batches.cancel(patch_id)


# # limit
# Per-batch limits: A single batch may include up to 50,000 requests
# a batch input file can be up to 100 MB in size
# Note that /v1/embeddings batches are also restricted to a maximum of 50,000 embedding inputs across all requests in the batch
# Enqueued prompt tokens per model: Each model has a maximum number of enqueued prompt tokens allowed for batch processing

{"custom_id": "MultiCaRe_Reasoning64387", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "As a meticulous and evidence-driven physician, your task is to generate exactly 10 accurate and distinct differential diagnoses based on the diagnosis information mentioned in the input case report and its title. Each diagnosis must be followed by a concise, evidence-informed explanation. Your diagnoses should span a wide range of possibilities, listed in order of likelihood, with the most likely diagnosis appearing first. Focus on clinical findings, considering their diagnostic accuracy and relevance as emphasized in evidence-based physical diagnosis practices. Ensure each diagnosis reflects key physical findings relevant to the patient's symptoms, with emphasis on findings that significantly alter the likelihood of specific conditions.\n**Output format**:\nDifferential diagnosis:\n1. [Diagnosis 1]: [Evidence-informed One-sentence explanation]\n2. [Diagnosis 2]: [Evidence-informed One-sentence explanation]\n3. [Diagnosis 3]: [Evidence-informed One-sentence explanation]\n4. [Diagnosis 4]: [Evidence-informed One-sentence explanation]\n5. [Diagnosis 5]: [Evidence-informed One-sentence explanation]\n6. [Diagnosis 6]: [Evidence-informed One-sentence explanation]\n7. [Diagnosis 7]: [Evidence-informed One-sentence explanation]\n8. [Diagnosis 8]: [Evidence-informed One-sentence explanation]\n9. [Diagnosis 9]: [Evidence-informed One-sentence explanation]\n10. [Diagnosis 10]: [Evidence-informed One-sentence explanation]\n**Example**:\nDifferential diagnosis:\n1. Mitochondrial disease: Considering the patient's history of celiac disease, fatigue, and somatic symptoms, a mitochondrial disorder could be a possibility, especially with the presence of a prolonged QT interval.\nPlease generate exactly 10 differential diagnoses with corresponding evidence-based one-sentence explanations, adhering strictly to the output format without including any other outputs.\nINPUT: "}, {"role": "user", "content": " Age: 17.0\nSex: Male\nTitle: A Pediatric Case of Cowden Syndrome with Graves' Disease\nKeywords: None\nAbstract: \nCowden syndrome (CS) is a rare dominantly inherited multisystem disorder, characterized by an extraordinary malignant potential. In 80% of cases, the human tumor suppressor gene phosphatase and tensin homolog (PTEN) is mutated. We present a case of a 17-year-old boy with genetically confirmed CS and Graves' disease (GD). At the age of 15, he presented with intention tremor, palpitations, and marked anxiety. On examination, he had macrocephaly, coarse facies, slight prognathism, facial trichilemmomas, abdominal keratoses, leg hemangioma, and a diffusely enlarged thyroid gland. He started antithyroid drug (ATD) therapy with methimazole and, after a 2-year treatment period without achieving a remission status, a total thyroidectomy was performed. Diagnosis and management of CS should be multidisciplinary. Thyroid disease is frequent, but its management has yet to be fully defined. The authors present a case report of a pediatric patient with CS and GD and discuss treatment options.\nImage Caption and Description: \nImage caption: []\nImage description: []\nCase Report: \nThe patient is a 17-year-old Caucasian boy that presented with intention tremor, palpitations, and marked anxiety at age 15. He was medicated with propranolol and hydroxyzine with partial improvement of symptoms and was referred to a hospital pediatric appointment.\n    OUTPUT:\n    "}], "max_tokens": 49000}}


{"custom_id": "GI_Reasoning1218", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "Generate the next 9 accurate and distinct differential diagnoses based on the input case report, considering the true final diagnosis provided after \u201cOUTPUT:\u201c. Your goal is to list the 9 most likely alternative diagnoses in descending order of likelihood, ensuring that each possibility is unique and covers a broad spectrum of differential diagnoses. \n Follow the guidelines for a generation: 1. Each diagnosis should be precise and unique, ensuring a variety of the next 9 possibilities. 2. List one diagnosis per line and the true diagnosis we provide should be the first one. 3. Generate the next 9 differential diagnoses related to the input case report. Think step by step. \n \n***Output format***:Differential diagnosis: 1. \n2. \n3.\n4. \n5. \n6. \n7. \n8. \n9. \n10. \n"}, {"role": "user", "content": "  a 92-year-old man presented to our emergency department due to black stool  passage for 3 days with general fatigue. he had history of bilateral renal cysts and duodenal  bulb ulcer bleeding post suture ligation 6 months ago. on examination, his conjunctiva was  pale and his abdomen was soft without tenderness. digital rectal examination revealed tarry  stool. laboratory data exhibited anemia (hemoglobin, 6.9 g/dl).  esophagogastroduodenoscopy (egd) revealed a 2.5cm bleeding ulcer at distal bulb and one  1cm out-punching area at proximal 2nd potion of duodenum with internal debris (figure a, b).  computed tomography (ct) of the abdomen (figure c) and further ugi series were  performed (figure d, e).\n    OUTPUT:\n    Duodenal-renal Fistula, right renal cyst"}], "max_tokens": 8100}}