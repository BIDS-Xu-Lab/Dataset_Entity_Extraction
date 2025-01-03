from datasets import load_dataset
from tqdm import tqdm, trange
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
import json

ds = load_dataset("YBXL/DDXPlus_Reasoning_train", cache_dir='/home/gy237/project/llama3/new_data/DDXPlus_Reasoning_train')
data = ds['train']['conversations']

print(ds)
# print(ds['train']['conversations'][1])
# print(len(data))
# print(len(data[0]))

human = [i[0]['value'] for i in data]

prompt = [i.split('INPUT:')[0] for i in human]
inpt = [i.split('INPUT:')[-1].split('OUTPUT:')[0] for i in human]
agent =  [i[-1]['value'] for i in data]

_prompt = '''Your task is to provide at least 10 accurate and distinct patient diagnoses based on the input case report. Key points: 1) Diagnoses are confirmed by clinical or anatomic pathology tests, or sometimes by clinical criteria or expert opinion. 2) You will be informed at the end of the case description if diagnostic tests are being ordered to confirm the diagnosis. Ensure that you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities.\n Follow the guidelines for a generation: 1. Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities. 2. List one diagnosis per line. 3. Generate at least 10 differential diagnoses related to the input case report. Think step by step.\n \n***Output format***:Differential diagnosis: 1. \n2. \n3.\n4. \n5. \n6. \n7. \n8. \n9. \n10. \n'''

for i in prompt:
    if i != prompt[1]:
        print('Erroe')

# print(prompt[0])
# print(_prompt)
# print(inpt[0])
# print(agent[0])
# print(agent[0].split('differential diagnosis:')[-1].split('final diagnosis:')[0].split('\n'))


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

min_len = 686
max_len = 0
upload_data = []
for i in trange(len(agent), desc="Step"):
    diease_list = agent[i].split('differential diagnosis:')[-1].split('final diagnosis:')[0].split('\n')
    diease_list = [i.strip() for i in diease_list if i.strip()]
    final = agent[i].split('differential diagnosis:')[-1].split('final diagnosis:')[-1]
    

    if len(diease_list) > 9:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    f"{_prompt}", # instruction
                    f"{inpt[i]}", # input
                    f"", # output - leave this blank for generation!
                )
            ], return_tensors = "pt")
        input_len = len(inputs['input_ids'][0])
        # print(inputs)
        # print(input_len)

        if 25 < input_len < 8100 :
            query = f"{_prompt}\nINPUT: {inpt[i]}\nOUTPUT:\n"
            answer = "Differential diagnosis: " + "\n".join(diease_list[:10]) + '\n\n final diagnosis:' + final
            upload_data.append({"id": f"DDXPlus_Reasoning_train_subset{i}", "query": query, "answer": answer})

            # print(upload_data[-1])
            # print(data[i])
            # print(ds['train'][i])
            # exit()

            if input_len > max_len:
                max_len = input_len
            if input_len < min_len:
                min_len = input_len
        else:
            print(input_len)

print(min_len)
print(max_len)
print(upload_data[0])

train_dataset = Dataset.from_list(upload_data)
dataset_dict = DatasetDict({
    "train": train_dataset,
})

print(dataset_dict)

with open(f'/home/gy237/project/llama3/new_data/DDXPlus_Reasoning_train_gui_processed.json', 'w', encoding='utf-8') as file:
    json.dump(upload_data, file, ensure_ascii=False, indent=4)


dataset_dict.push_to_hub("YBXL/DDXPlus_Reasoning_train_gui_processed")

# filtered : YBXL/DDXPlus_Reasoning_train
# check prompt with raw training data
# differential diagnosis >=10, label cut to 10 left
# input token >25, <4096
