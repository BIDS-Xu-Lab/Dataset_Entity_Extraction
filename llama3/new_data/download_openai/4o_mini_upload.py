# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

from openai import OpenAI
client = OpenAI()


# name = "MultiCaRe_Reasoning_test_diagnosis_7_explation"

# # upload your batch input file
# batch_input_file = client.files.create(
#   file=open(f"/home/gy237/project/llama3/new_data/uploda_openai/multicare_pmc_explation_new/MultiCaRe_Reasoning_test_diagnosis_7_explation.jsonl", "rb"),
#   purpose="batch"
# )


# # create the batch, only the description can change
# batch_input_file_id = batch_input_file.id
# client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#       "description": f'{name}'
#     }
# )


# Getting a list of All Batches
print(client.batches.list(limit=10))
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


# Cancelling a Batch
# client.batches.cancel("batch_MMR1qtZDy46Ey3Pry4bQe3Uz")


# # limit
# Per-batch limits: A single batch may include up to 50,000 requests
# a batch input file can be up to 100 MB in size
# Note that /v1/embeddings batches are also restricted to a maximum of 50,000 embedding inputs across all requests in the batch
# Enqueued prompt tokens per model: Each model has a maximum number of enqueued prompt tokens allowed for batch processing