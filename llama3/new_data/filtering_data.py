from datasets import load_dataset
from tqdm import tqdm, trange
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel
import json
from openai import OpenAI

ds1 = load_dataset("YBXL/MultiCaRe_Reasoning_test", cache_dir='/home/gy237/project/download_data')
ds2 = load_dataset("YBXL/PMC_Patients_Reasoning_test", cache_dir='/home/gy237/project/download_data')
ds3 = load_dataset("YBXL/PMC_CaseReport_Reasoning_test", cache_dir='/home/gy237/project/download_data')

# print(ds1)
# print(ds2)
# print(ds3)
# print(ds1['train']['query'][1])
# print(ds2['train']['query'][1])
# print(ds3['train']['query'][1])

id1 = ds1['train']['id'] + ds2['train']['id'] + ds3['train']['id']
query1 = ds1['train']['query'] + ds2['train']['query'] + ds3['train']['query']
answer1 = ds1['train']['answer'] + ds2['train']['answer'] + ds3['train']['answer']
print(len(id1))

# paired_dict = dict()
# for key, value in zip(query1, answer1):
#     if key not in paired_dict.keys():
#         paired_dict[key] = value
# print(len(paired_dict.keys()))
# exit()

max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def yes_or_no(inpt):
    client = OpenAI()
    Prompt="""
    You are an expert medical professional with extensive experience in clinical diagnosis. You will be provided with a text, which may contain elements such as patient history, clinical findings, and diagnostic details. Your task is to critically analyze the text to determine whether it qualifies as a case report that includes detailed descriptions of the patient's history, symptoms, and the diagnostic reasoning process. Use your medical expertise and reasoning skills to make an informed decision. Respond with a simple 'Yes' if the text qualifies as a case report with these specific details, or 'No' if it does not.
    """           
    chat_return = client.chat.completions.create(model='gpt-4o-mini',messages=[{"role": "system", "content": "pysician"},
                                                                     {"role":"user", 
                                                                      "content":f"{Prompt} \n"\
                                                                      f"Text: {inpt}"
                                                                      }])
    result=chat_return.choices[0].message.content
    return result





min_len = 686
max_len = 0
upload_data = []
no_data = []
error_data = []
error = []
for i in trange(len(id1), desc="Step"):
    query1_list = query1[i].split('INPUT:')
    assert len(query1_list) == 2
    prompt = query1_list[0]
    inpt = query1_list[1]



    inputs = tokenizer(
            [
                alpaca_prompt.format(
                    f"{prompt}", # instruction
                    f"{inpt}", # input
                    f"", # output - leave this blank for generation!
                )
            ], return_tensors = "pt")
    input_len = len(inputs['input_ids'][0])
    # print(inputs)
    # print(input_len)
    # exit()

    if 50 < input_len < 8100 :
        # yn = yes_or_no(inpt)
        query = f"{prompt}INPUT:{inpt}"
        upload_data.append({"id": id1[i], "query": query, "answer": answer1[i]})

        # if yn.startswith('Yes'):
        #     upload_data.append({"id": id1[i], "query": query, "answer": answer1[i]})
        # elif yn.startswith('No'):
        #     no_data.append({"id": id1[i], "query": query, "answer": answer1[i]})
        # else:
        #     error_data.append({"id": id1[i], "query": query, "answer": answer1[i]})
        #     error.append(yn)
        #     print(yn)

        # print(upload_data[-1])
        # print(ds1['train'][i])
        # exit()

    if input_len > max_len:
            max_len = input_len
    if input_len < min_len:
            min_len = input_len
    

print(min_len)
print(max_len)
print(len(upload_data))
# print(len(no_data))

# train_dataset = Dataset.from_list(upload_data)
# dataset_dict = DatasetDict({
#     "train": train_dataset,
# })
# print(dataset_dict)


# with open(f'/home/gy237/project/llama3/new_data/MultiCaRe_PMC_Patients_PMC_CaseReport.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data, file, ensure_ascii=False, indent=4)

# dataset_dict.push_to_hub("YBXL/MultiCaRe_PMC_Patients_PMC_CaseReport")