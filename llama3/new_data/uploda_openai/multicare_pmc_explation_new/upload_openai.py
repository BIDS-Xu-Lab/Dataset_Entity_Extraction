# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

from openai import OpenAI
import os
client = OpenAI()  
folder_path = '/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation_new'
file_names = os.listdir(folder_path)
file_names = [i for i in file_names if i.endswith('jsonl')]
print(len(file_names))


# def uplo(name):
#   name = name
#   # upload your batch input file
#   batch_input_file = client.files.create(
#     file=open(f"/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation_new/{name}", "rb"),
#     purpose="batch"
#   )
#   # create the batch, only the description can change
#   batch_input_file_id = batch_input_file.id
#   client.batches.create(
#       input_file_id=batch_input_file_id,
#       endpoint="/v1/chat/completions",
#       completion_window="24h",
#       metadata={
#         "description": f'{name}'
#       }
#   )
# for name in file_names:
#   print(name)
#   uplo(name)

# Getting a list of All Batches
# print(client.batches.list(limit=20))
# exit()

# 从openai下载
a = client.batches.list(limit=10)
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

des_list = [i for i in des_list if i['description'] in file_names and i['status']=='completed' and 'output' in i.keys()]
print(len(des_list))
# print(des_list)
exit()

import os
for i in des_list:
  des = i['description']
  out = i['output']




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
