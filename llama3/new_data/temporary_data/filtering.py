from datasets import load_dataset
from tqdm import tqdm, trange
from datasets import Dataset, DatasetDict
import json
from unsloth import FastLanguageModel

max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

def input_filter(_id, inpt, oupt, tokenizer, mode):
    upload_data = []
    in_len = []
    ou_len = []
    # for i in trange(len(_id), desc='Step'):
    for i in range(len(_id)):
        inputs = tokenizer([inpt[i]], return_tensors = "pt")
        in_input_len = len(inputs['input_ids'][0])
        inputs = tokenizer([oupt[i]], return_tensors = "pt")
        ou_input_len = len(inputs['input_ids'][0])
        
        if mode == 'input':
            if 50 < in_input_len < 8100 and 10 < ou_input_len < 8100:
                upload_data.append({"id": _id[i], "query": inpt[i], "answer": oupt[i]})

        elif mode == 'output':
            if 10 < in_input_len < 8100 and 50 < ou_input_len < 8100:
                upload_data.append({"id": _id[i], "query": inpt[i], "answer": oupt[i]})
        
        elif mode == 'bs4':
            if 50 < in_input_len < 8100 and 0 < ou_input_len < 8100:
                upload_data.append({"id": _id[i], "query": inpt[i], "answer": oupt[i]})

        elif mode == 'both':
            if 50 < in_input_len < 8100 and 50 < ou_input_len < 8100:
                upload_data.append({"id": _id[i], "query": inpt[i], "answer": oupt[i]})
        
        else:
            assert False, "Error"

        in_len.append(in_input_len)
        ou_len.append(ou_input_len)
    
    return upload_data

def filter_empty(_id, inpt, oupt):
    id_ = []
    inpt_ = []
    oupt_ = []
    for i in range(len(_id)):
        _in = inpt[i].split('INPUT:')[-1]
        if len(_in) > 10 and len(oupt[i]) > 5:
            id_.append(_id[i])
            inpt_.append(inpt[i])
            oupt_.append(oupt[i])
    return id_, inpt_, oupt_


# ds1 = load_dataset("YBXL/medical_book_train", cache_dir='/home/gy237/project/download_data') #背景知识 output token < 50  
# ds2 = load_dataset("YBXL/GENE_REVIEW_SY_train", cache_dir='/home/gy237/project/download_data') #背景知识 output: <50 tokens.  50 <input len <8192 remove \n\n2  
# ds3 = load_dataset("YBXL/GENE_OMIM_SY_train", cache_dir='/home/gy237/project/download_data')  #背景知识 output< 50, gpt-4o mini to refine the output
# ds4 = load_dataset("YBXL/MedQA_Reasoning_train", cache_dir='/home/gy237/project/download_data') #背景知识  filter null input and input > 8192
# ds5 = load_dataset("YBXL/GI_Reasoning_train", cache_dir='/home/gy237/project/download_data')  #要用的知识  filter null input and input > 8192, and use gpt-4o for generating differential diagnosis lists
# ds6 = load_dataset("hippocrates/MedInstruct_train", cache_dir='/home/gy237/project/download_data')  #背景知识
# ds7 = load_dataset("YBXL/JAMA_Reasoning_Rare_train", cache_dir='/home/gy237/project/download_data') # 解释您的诊断

# id1 = ds1['train']['id']
# inpt1 = [ i[0]["content"] for i in ds1['train']['conversations']]
# oupt1 = [ i[1]["content"] for i in ds1['train']['conversations']]
# id1, inpt1, oupt1 = filter_empty(id1, inpt1, oupt1)
# upload_data1 = input_filter(id1, inpt1, oupt1, tokenizer, 'output')
# print(len(id1))
# print(len(upload_data1))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/medical_book_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data1, file, ensure_ascii=False, indent=4)


# id2 = ds2['train']['id']
# inpt2 = [ i[0]["content"] for i in ds2['train']['conversations']]
# oupt2 = [ i[1]["content"] for i in ds2['train']['conversations']]
# id2, inpt2, oupt2 = filter_empty(id2, inpt2, oupt2)
# upload_data2 = input_filter(id2, inpt2, oupt2, tokenizer, 'both')
# print(len(id2))
# print(len(upload_data2))
# for i in range(len(upload_data2)):
#     if upload_data2[i]['answer'].endswith('\n\n2'):
#         upload_data2[i]['answer'] = upload_data2[i]['answer'][:-3]
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/GENE_REVIEW_SY_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data2, file, ensure_ascii=False, indent=4)


# id3 = ds3['train']['id']
# inpt3 = [ i[0]["content"] for i in ds3['train']['conversations']]
# oupt3 = [ i[1]["content"] for i in ds3['train']['conversations']]
# id3, inpt3, oupt3 = filter_empty(id3, inpt3, oupt3)
# upload_data3 = input_filter(id3, inpt3, oupt3, tokenizer, 'output')
# print(len(id3))
# print(len(upload_data3))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/GENE_OMIM_SY_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data3, file, ensure_ascii=False, indent=4)


# id4 = ds4['train']['id']
# inpt4 = [ i[0]["content"] for i in ds4['train']['conversations']]
# oupt4 = [ i[1]["content"] for i in ds4['train']['conversations']]
# id4, inpt4, oupt4 = filter_empty(id4, inpt4, oupt4)
# upload_data4 = input_filter(id4, inpt4, oupt4, tokenizer, 'bs4')
# print(len(id4))
# print(len(upload_data4))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/MedQA_Reasoning_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data4, file, ensure_ascii=False, indent=4)


# id5 = ds5['train']['id']
# inpt5 = [ i[0]["value"] for i in ds5['train']['conversations']]
# oupt5 = [ i[1]["value"] for i in ds5['train']['conversations']]
# id5, inpt5, oupt5 = filter_empty(id5, inpt5, oupt5)
# upload_data5 = input_filter(id5, inpt5, oupt5, tokenizer, 'input')
# print(len(id5))
# print(len(upload_data5))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/GI_Reasoning_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data5, file, ensure_ascii=False, indent=4)


# id6 = ds6['train']['id']
# inpt6 = [ i[0]["value"] for i in ds6['train']['conversations']]
# oupt6 = [ i[1]["value"] for i in ds6['train']['conversations']]
# id6, inpt6, oupt6 = filter_empty(id6, inpt6, oupt6)
# upload_data6 = input_filter(id6, inpt6, oupt6, tokenizer, 'both')
# print(len(id6))
# print(len(upload_data6))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/MedInstruct_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data6, file, ensure_ascii=False, indent=4)


# id7 = ds7['train']['id']
# inpt7 = [ i[0]["content"] for i in ds7['train']['conversations']]
# oupt7 = [ i[1]["content"] for i in ds7['train']['conversations']]
# id7, inpt7, oupt7 = filter_empty(id7, inpt7, oupt7)
# upload_data7 = input_filter(id7, inpt7, oupt7, tokenizer, 'both')
# print(len(id7))
# print(len(upload_data7))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/JAMA_Reasoning_Rare_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data7, file, ensure_ascii=False, indent=4)




# ds8 = load_dataset("YBXL/MultiCaRe_Reasoning_test", cache_dir='/home/gy237/project/download_data')
# # ds9 = load_dataset("YBXL/PMC_Patients_Reasoning_test", cache_dir='/home/gy237/project/download_data')
# # ds10 = load_dataset("YBXL/PMC_CaseReport_Reasoning_test", cache_dir='/home/gy237/project/download_data')

# id8 = ds8['train']['id']
# inpt8 = ds8['train']['query']
# oupt8 = ds8['train']['answer']
# upload_data8 = input_filter(id8, inpt8, oupt8, tokenizer, 'bs4')
# print(upload_data8[:2])
# print(len(id8))
# print(len(upload_data8))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/MultiCaRe_Reasoning_test.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data8, file, ensure_ascii=False, indent=4)


# ds1 = load_dataset("YBXL/JAMA_Reasoning_Common_train", cache_dir='/home/gy237/project/download_data')
# id1 = ds1['train']['id']
# inpt1 = [ i[0]["content"] for i in ds1['train']['conversations']]
# oupt1 = [ i[1]["content"] for i in ds1['train']['conversations']]
# id1, inpt1, oupt1 = filter_empty(id1, inpt1, oupt1)
# upload_data1 = input_filter(id1, inpt1, oupt1, tokenizer, 'both')
# print(len(id1))
# print(len(upload_data1))
# with open(f'/home/gy237/project/llama3/new_data/temporary_data/JAMA_Reasoning_Common_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data1, file, ensure_ascii=False, indent=4)